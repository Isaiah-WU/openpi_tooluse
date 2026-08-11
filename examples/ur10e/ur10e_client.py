"""
UR10e 客户端：从策略服务器获取动作，并在真实机器人上执行。

运行前提：
1. 已经在另一个终端 / 另一台机器上启动了 serve_policy.py，服务监听在 8000 端口
2. 已经 pip install -e . 安装了 openpi_tooluse/packages/openpi-client
3. 装好以下依赖：
   pip install ur_rtde pyrealsense2 numpy
4. 修改下面 ROBOT_IP 为你 UR10e 的真实 IP
5. 确认示教器上装了 Robotiq 的 URCap 插件（socket 端口默认 63352）
6. 使用 `rs-enumerate-devices` 核对 Base/Wrist 相机序列号
"""

import logging
import socket
import time

import numpy as np
import pyrealsense2 as rs
import rtde_control
import rtde_receive

from openpi_client import image_tools
from openpi_client import websocket_client_policy
from async_policy import AsyncPolicyProcess
from runtime_timing import ControlCycleTiming
from timing_recorder import TimingRecorder
from action_trajectory import resample_action_chunk

logging.basicConfig(level=logging.INFO)

# ========================

ROBOT_IP = "192.168.1.9"           
GRIPPER_PORT = 63352               
GRIPPER_MAX_POS = 255                

HOST_IP = "127.0.0.1"
HOST_PORT = 8000                     

BASE_CAMERA_SERIAL = "244222070262"
WRIST_CAMERA_SERIAL = "213522071124"

# ============================================================

CHECKPOINT_PROMPTS = {
    "pi05_ur10e_demo": (
        "pour water from the kettle into the cup, "
        "then move the cup away and wipe the table with the cloth"
    ),
    "pi05_ur10e_long_horizon_lora": (
        "Pick up the yellow cup at the back and place it into the blue cup in the front. "
        "Lift the nested cups and place them into the blue cup inside the basket. "
        "Pick up the blue plate at the back and place it onto the blue plate in the front. "
        "Lift the stacked plates and place them onto the blue plate inside the basket. "
        "Take the rag and wipe the table."
    ),
}

ACTIVE_CHECKPOINT = "pi05_ur10e_long_horizon_lora"


class RobotiqGripper:

    def __init__(self, robot_ip: str, port: int = GRIPPER_PORT):
        self.sock = socket.create_connection((robot_ip, port), timeout=2.0)
        # 激活夹爪（第一次连接后通常需要激活一次）
        self._send("SET ACT 1")
        self._send("SET GTO 1")

    def _send(self, cmd: str) -> str:
        self.sock.sendall((cmd + "\n").encode("utf-8"))
        return self.sock.recv(1024).decode("utf-8")

    def move(self, pos_0_to_1: float, speed: int = 150, force: int = 100):
        pos = int(np.clip(pos_0_to_1, 0.0, 1.0) * GRIPPER_MAX_POS)
        self._send(f"SET POS {pos}")
        self._send(f"SET SPE {speed}")
        self._send(f"SET FOR {force}")

    def get_position(self) -> float:
        reply = self._send("GET POS")
        pos = int(reply.strip().split()[-1])
        return pos / GRIPPER_MAX_POS


rtde_c = None
rtde_r = None
gripper = None
base_pipeline = None
wrist_pipeline = None


def _resolve_camera_serials() -> tuple[str, str]:
    """Return explicitly configured Base and Wrist camera serial numbers."""
    if BASE_CAMERA_SERIAL and WRIST_CAMERA_SERIAL:
        if BASE_CAMERA_SERIAL == WRIST_CAMERA_SERIAL:
            raise ValueError(
                "Base and Wrist camera serial numbers must be different"
            )

        return BASE_CAMERA_SERIAL, WRIST_CAMERA_SERIAL

    context = rs.context()
    detected_serials = [
        device.get_info(rs.camera_info.serial_number)
        for device in context.query_devices()
    ]

    raise RuntimeError(
        "BASE_CAMERA_SERIAL and WRIST_CAMERA_SERIAL must be configured. "
        f"Detected RealSense serial numbers: {detected_serials}"
    )


def _start_camera_pipeline(
    serial: str,
    label: str,
) -> rs.pipeline:
    camera_pipeline = rs.pipeline()
    camera_config = rs.config()
    camera_config.enable_device(serial)
    camera_config.enable_stream(
        rs.stream.color,
        640,
        480,
        rs.format.bgr8,
        30,
    )
    camera_pipeline.start(camera_config)
    logging.info(
        "%s camera pipeline started (serial=%s)",
        label,
        serial,
    )
    return camera_pipeline


def initialize_hardware():
    global rtde_c
    global rtde_r
    global gripper
    global base_pipeline
    global wrist_pipeline

    rtde_c = rtde_control.RTDEControlInterface(
        ROBOT_IP
    )
    rtde_r = rtde_receive.RTDEReceiveInterface(
        ROBOT_IP
    )
    gripper = RobotiqGripper(
        ROBOT_IP
    )

    base_serial, wrist_serial = (
        _resolve_camera_serials()
    )

    base_pipeline = _start_camera_pipeline(
        base_serial,
        "base",
    )
    wrist_pipeline = _start_camera_pipeline(
        wrist_serial,
        "wrist",
    )


def get_camera_image(
    camera_pipeline: rs.pipeline,
) -> np.ndarray:
    frames = camera_pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()

    if color_frame is None:
        raise RuntimeError(
            "RealSense color frame is unavailable"
        )

    img_bgr = np.asanyarray(color_frame.get_data())
    img_rgb = img_bgr[:, :, ::-1]   # BGR -> RGB
    return img_rgb


def stop_camera_pipelines():
    """Stop both RealSense pipelines if they were started."""
    global base_pipeline
    global wrist_pipeline

    for camera_pipeline in (
        base_pipeline,
        wrist_pipeline,
    ):
        if camera_pipeline is not None:
            camera_pipeline.stop()

    base_pipeline = None
    wrist_pipeline = None


def get_robot_state():
    """6 个关节角度 + 1 个夹爪开合值，拼成长度为 7 的一维数组"""
    joints = rtde_r.getActualQ()          # list，长度 6
    gripper_pos = gripper.get_position()  # 0~1 之间
    return np.array(list(joints) + [gripper_pos], dtype=np.float32)


def capture_observation(task_instruction):
    observation_start = time.perf_counter()

    if base_pipeline is None or wrist_pipeline is None:
        raise RuntimeError(
            "Camera pipelines must be initialized before capture"
        )

    base_img = get_camera_image(base_pipeline)
    wrist_img = get_camera_image(wrist_pipeline)
    state = get_robot_state()

    observation = {
        "observation/image": image_tools.convert_to_uint8(
            image_tools.resize_with_pad(
                base_img,
                224,
                224,
            )
        ),
        "observation/wrist_image": image_tools.convert_to_uint8(
            image_tools.resize_with_pad(
                wrist_img,
                224,
                224,
            )
        ),
        "observation/state": state,
        "prompt": task_instruction,
    }

    observation_ready = time.perf_counter()

    return (
        observation,
        observation_start,
        observation_ready,
    )


def execute_action(action):
    """action 是长度为 7 的一维数组：前 6 维是关节目标角度，第 7 维是夹爪目标位置"""
    joint_targets = action[:6].tolist()
    gripper_target = float(action[6])

    # asynchronous=True：不阻塞等这一步走完，配合外层 sleep 控制节奏
    rtde_c.moveJ(joint_targets, speed=0.5, acceleration=0.5, asynchronous=True)
    gripper.move(gripper_target)


# ======================================================================


def main_async():
    initialize_hardware()
    task_instruction = CHECKPOINT_PROMPTS[
        ACTIVE_CHECKPOINT
    ]

    num_steps = 200
    query_every_n_steps = 10

    policy_action_hz = 30.0
    control_hz = 10.0
    control_period_s = 1.0 / control_hz

    policy = AsyncPolicyProcess(
        remote_host=HOST_IP,
        remote_port=HOST_PORT,
        fake_delay_ms=None,
        action_horizon=50,
        action_dim=7,
    )

    recorder = TimingRecorder(
        "ur10e_timing",
    )

    action_chunk = None
    chunk_index = 0
    next_query_step = 0
    action_chunk_timing = None
    last_action = None

    policy.start()

    try:
        next_cycle_deadline = time.perf_counter()

        for step in range(num_steps):
            cycle = ControlCycleTiming(
                control_step=step,
                cycle_start=time.perf_counter(),
            )

            response = policy.poll()

            if response is not None:
                returned_timing = response.get(
                    "timing",
                )

                if not response["ok"]:
                    if returned_timing is not None:
                        recorder.record_request(
                            returned_timing,
                        )

                    raise RuntimeError(
                        response["error"]
                    )

                returned_timing.chunk_prepare_start = (
                    time.perf_counter()
                )

                new_chunk = np.asarray(
                    response["actions"],
                    dtype=np.float32,
                )

                if (
                    new_chunk.ndim != 2
                    or new_chunk.shape[0] == 0
                    or new_chunk.shape[1] != 7
                ):
                    raise ValueError(
                        "Policy actions must have "
                        f"shape (H, 7), got {new_chunk.shape}"
                    )

                new_chunk = resample_action_chunk(
                    new_chunk,
                    source_hz=policy_action_hz,
                    target_hz=control_hz,
                )

                returned_timing.chunk_prepare_end = (
                    time.perf_counter()
                )

                action_chunk = new_chunk
                chunk_index = 0
                action_chunk_timing = returned_timing

                returned_timing.accept_step = step
                returned_timing.chunk_accept = (
                    time.perf_counter()
                )

                recorder.record_request(
                    returned_timing,
                )

            if (
                not policy.inflight
                and step >= next_query_step
            ):
                (
                    observation,
                    observation_start,
                    observation_ready,
                ) = capture_observation(
                    task_instruction,
                )

                submitted_timing = policy.submit(
                    observation,
                    observation_step=step,
                    submit_step=step,
                    observation_start=observation_start,
                    observation_ready=observation_ready,
                )

                recorder.record_request(
                    submitted_timing,
                )

                next_query_step = (
                    step + query_every_n_steps
                )

            if (
                action_chunk is not None
                and chunk_index < len(action_chunk)
            ):
                if action_chunk_timing is None:
                    raise RuntimeError(
                        "Action chunk has no request timing"
                    )

                action = action_chunk[chunk_index]

                cycle.request_id = (
                    action_chunk_timing.request_id
                )
                cycle.action_index = chunk_index
                cycle.action_source = (
                    "new_chunk"
                    if chunk_index == 0
                    else "current_chunk"
                )

                action_start = time.perf_counter()
                cycle.execute_action_start = action_start

                if chunk_index == 0:
                    action_chunk_timing.first_action_start = (
                        action_start
                    )

                execute_action(action)

                last_action = action.copy()

                action_end = time.perf_counter()
                cycle.execute_action_end = action_end

                if chunk_index == 0:
                    action_chunk_timing.first_action_end = (
                        action_end
                    )
                    recorder.record_request(
                        action_chunk_timing,
                    )

                chunk_index += 1

            elif last_action is not None:
                cycle.request_id = (
                    action_chunk_timing.request_id
                    if action_chunk_timing is not None
                    else None
                )
                cycle.action_index = max(
                    0,
                    chunk_index - 1,
                )
                cycle.action_source = "hold_last"

                cycle.execute_action_start = (
                    time.perf_counter()
                )

                execute_action(last_action)

                cycle.execute_action_end = (
                    time.perf_counter()
                )

            else:
                cycle.action_source = (
                    "waiting_for_first_chunk"
                )

            next_cycle_deadline += (
                control_period_s
            )

            now = time.perf_counter()
            sleep_s = (
                next_cycle_deadline - now
            )

            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                next_cycle_deadline = now

            cycle.cycle_end = time.perf_counter()
            recorder.record_cycle(cycle)

    finally:
        policy.stop()
        recorder.save()
        recorder.print_summary()
        stop_camera_pipelines()


def main():
    # host/port 已经在文件顶部的 HOST_IP / HOST_PORT 里配置，这里直接引用，不用改这一行
    initialize_hardware()
    client = websocket_client_policy.WebsocketClientPolicy(host=HOST_IP, port=HOST_PORT)

    task_instruction = CHECKPOINT_PROMPTS[
        ACTIVE_CHECKPOINT
    ]

    num_steps = 200          # 想让机器人跑多少个时间步，自己定
    query_every_n_steps = 10  # 每隔多少步重新问一次服务器要新动作（不用每一步都问）

    action_chunk = None

    try:
        for step in range(num_steps):
            # 每隔 N 步，或者还没有可用的动作块时，才重新查询服务器
            if action_chunk is None or step % query_every_n_steps == 0:
                (
                    observation,
                    observation_start,
                    observation_ready,
                ) = capture_observation(task_instruction)

                result = client.infer(observation)
                action_chunk = result["actions"]   # 形状是 (action_horizon, 7)
                chunk_index = 0

            # 从这一块动作序列里，依次取出一步来执行（"开环"执行，直到用完这一块再重新问）
            action = action_chunk[chunk_index]
            execute_action(action)
            chunk_index += 1

            time.sleep(0.1)   # 按你机器人实际控制频率调整，比如 10Hz 对应 0.1 秒
    finally:
        stop_camera_pipelines()


if __name__ == "__main__":
    main()
