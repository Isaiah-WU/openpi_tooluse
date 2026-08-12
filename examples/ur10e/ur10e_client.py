# ruff: noqa: RUF002, RUF003
"""
UR10e 客户端：从策略服务器获取动作，并在真实机器人上执行。

运行前提：
1. 已经在另一个终端 / 另一台机器上启动了 serve_policy.py，服务监听在 8000 端口
2. 已经 pip install -e . 安装了 openpi_tooluse/packages/openpi-client
3. 装好以下依赖：
   pip install ur_rtde pyrealsense2 numpy
4. 修改下面 ROBOT_IP 为你 UR10e 的真实 IP
5. 确认示教器上装了 Robotiq 的 URCap 插件（socket 端口默认 63352）
6. 两个 RealSense 相机型号相同，跑之前用 `rs-enumerate-devices` 查出各自的序列号，
   填进下面的 BASE_CAMERA_SERIAL / WRIST_CAMERA_SERIAL，不填的话程序会直接报错退出
   （宁可现在报错，也不要让 base/wrist 画面被静默换错）

首次验证 RTC 时运行：
    python examples/ur10e/ur10e_client.py --dry-run --num-steps 30
该模式仍会读取真实相机、机器人状态和夹爪位置，但不会创建 RTDE 控制连接、不会调用
moveJ()，也不会向夹爪发送移动命令；它只请求策略动作并打印结果。
"""

import argparse
import logging
import socket
import time

import numpy as np
from openpi_client import action_chunk_broker
from openpi_client import image_tools
from openpi_client import websocket_client_policy
import pyrealsense2 as rs
import rtde_control
import rtde_receive

# 提前配置好日志格式：下面的相机绑定代码在 import 阶段(模块级)就会打印 INFO 日志，
# 如果留到文件末尾 `if __name__ == "__main__":` 里才 basicConfig，那时早就晚了，
# 日志会被 root logger 的默认 WARNING 阈值悄悄吞掉。
logging.basicConfig(level=logging.INFO)

# ========================

ROBOT_IP = "192.168.1.9"
GRIPPER_PORT = 63352
GRIPPER_MAX_POS = 255

HOST_IP = "127.0.0.1"
HOST_PORT = 8000

# pi0.5 每次返回 50 步动作；RTC 至少执行 10 个控制步后才开始异步生成下一块。
# 这两个 horizon 含义不同，不能再用同一个 action_horizon 混在一起。
PREDICTION_HORIZON = 50
EXECUTION_HORIZON = 10

# 两个 RealSense 相机型号完全相同，SDK 没法按名字/型号区分哪个是外部(base)相机、
# 哪个是腕部(wrist)相机 —— 必须按各自的序列号(Serial Number)绑定，否则每个 pipeline
# 的 start() 具体连到哪个物理相机是不确定的，可能导致 base/wrist 画面被换，模型会学到
# 错误的视角关系而不自知。用命令行工具 `rs-enumerate-devices` 查看两个相机各自的
# "Device Serial No"（可以先只插一个相机跑一次这个命令记下序列号，再插第二个，这样
# 能分清哪个序列号对应哪个物理相机），然后填在下面：
BASE_CAMERA_SERIAL = "244222070262"    # world camera —— 固定装在工作台上方/外部的相机
WRIST_CAMERA_SERIAL = "213522071124"   # wrist camera —— 装在机械臂末端腕部的相机

# ============================================================

# 每个 checkpoint 训练时用的语言指令(必须和 config.py 里对应 TrainConfig 的数据集
# 转换脚本里写死的 "task" 字段完全一致,否则模型收到的指令和它训练时学到的对不上,
# 会直接导致输出动作不相关/错误 —— 这曾经是导致成功率异常低的一个真实 bug,不要再
# 把 prompt 写死成任意一句话,而是从这里按 checkpoint 名字挑)。
CHECKPOINT_PROMPTS = {
    # config.py: pi05_ur10e / pi05_ur10e_lora / pi05_ur10e_lora_bs32
    # 数据集: wbjsamuel/ur10e_demo
    "pi05_ur10e_demo": (
        "pour water from the kettle into the cup, then move the cup away and wipe the table with the cloth"
    ),
    # config.py: pi05_ur10e_long_horizon_lora
    # 数据集: wbjsamuel/ur10e_long_horizon
    "pi05_ur10e_long_horizon_lora": (
        "Pick up the yellow cup at the back and place it into the blue cup in the front. "
        "Lift the nested cups and place them into the blue cup inside the basket. "
        "Pick up the blue plate at the back and place it onto the blue plate in the front. "
        "Lift the stacked plates and place them onto the blue plate inside the basket. "
        "Take the rag and wipe the table."
    ),
}

# 改这一行来选择当前 serve_policy.py 实际加载的是哪个 checkpoint/config。
ACTIVE_CHECKPOINT = "pi05_ur10e_long_horizon_lora"


class RobotiqGripper:

    def __init__(self, robot_ip: str, port: int = GRIPPER_PORT, *, activate: bool = True):
        self.sock = socket.create_connection((robot_ip, port), timeout=2.0)
        if activate:
            # 激活夹爪（第一次连接后通常需要激活一次）。dry-run 不发送这些写命令。
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

    def close(self) -> None:
        self.sock.close()

def _resolve_camera_serials() -> tuple[str, str]:
    """返回 (base_serial, wrist_serial)。两个相机型号相同,无法靠名字区分,必须显式绑定
    序列号(Intel 官方多相机文档的推荐做法:enumerate 设备 -> config.enable_device(serial))。
    没配的话直接报错退出,而不是猜一个默认绑定 —— 一旦 base/wrist 被静默换过,模型收到
    的每一帧观测都是错的,而且不会有任何报错提示,比启动时报错难排查得多。
    """
    if BASE_CAMERA_SERIAL and WRIST_CAMERA_SERIAL:
        return BASE_CAMERA_SERIAL, WRIST_CAMERA_SERIAL
    ctx = rs.context()
    serials = [d.get_info(rs.camera_info.serial_number) for d in ctx.query_devices()]
    raise RuntimeError(
        "BASE_CAMERA_SERIAL / WRIST_CAMERA_SERIAL 还没填(文件顶部)。"
        f"当前检测到的 RealSense 序列号: {serials}。"
        "先跑 `rs-enumerate-devices`,分清哪个序列号是外部相机、哪个是腕部相机"
        "(比如先拔掉一个只留一个插着跑一次命令),再把两个常量填上。"
    )


def _start_camera_pipeline(serial: str, label: str) -> rs.pipeline:
    pipeline = rs.pipeline()
    rs_config = rs.config()
    rs_config.enable_device(serial)
    rs_config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(rs_config)
    logging.info(f"{label} 相机 pipeline 已启动 (serial={serial})")
    return pipeline


def get_camera_image(pipeline: rs.pipeline) -> np.ndarray:
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    img_bgr = np.asanyarray(color_frame.get_data())
    return img_bgr[:, :, ::-1]  # BGR -> RGB


def get_robot_state(rtde_r: rtde_receive.RTDEReceiveInterface, gripper: RobotiqGripper):
    """6 个关节角度 + 1 个夹爪开合值，拼成长度为 7 的一维数组"""
    joints = rtde_r.getActualQ()          # list，长度 6
    gripper_pos = gripper.get_position()  # 0~1 之间
    return np.array([*list(joints), gripper_pos], dtype=np.float32)


def execute_action(action, rtde_c: rtde_control.RTDEControlInterface, gripper: RobotiqGripper):
    """action 是长度为 7 的一维数组：前 6 维是关节目标角度，第 7 维是夹爪目标位置"""
    joint_targets = action[:6].tolist()
    gripper_target = float(action[6])

    # asynchronous=True：不阻塞等这一步走完，配合外层 sleep 控制节奏
    rtde_c.moveJ(joint_targets, speed=0.5, acceleration=0.5, asynchronous=True)
    gripper.move(gripper_target)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the UR10e client with RTC action chunking.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Read observations and print policy actions without creating RTDE control or moving the robot/gripper.",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=None,
        help="Number of controller steps. Defaults to 30 in dry-run mode and 200 otherwise.",
    )
    parser.add_argument("--host", default=HOST_IP, help="Policy server hostname or IP address.")
    parser.add_argument("--port", type=int, default=HOST_PORT, help="Policy server WebSocket port.")
    parser.add_argument("--control-period", type=float, default=0.1, help="Seconds between controller steps.")
    args = parser.parse_args()
    if args.num_steps is not None and args.num_steps <= 0:
        parser.error("--num-steps must be positive")
    if args.control_period <= 0:
        parser.error("--control-period must be positive")
    return args


# ======================================================================


def main(args: argparse.Namespace):
    # host/port 已经在文件顶部的 HOST_IP / HOST_PORT 里配置，这里直接引用，不用改这一行
    client = websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)
    task_instruction = CHECKPOINT_PROMPTS[ACTIVE_CHECKPOINT]

    num_steps = args.num_steps if args.num_steps is not None else (30 if args.dry_run else 200)

    # RtcActionChunkBroker = PipelinedActionChunkBroker(后台线程提前预取下一块动作,
    # 隐藏网络+推理延迟) + RTC 平滑衔接:预取时把当前 chunk 还没执行完的尾部动作
    # (committed tail)一起发给模型,让新 chunk 的开头几步被"锚定"到旧轨迹上,权重从
    # 1 衰减到 0,而不是在 chunk 边界处硬切换 —— 倒水/擦桌子这类连续运动任务在衔接
    # 点的顿挫本身就是失败原因之一。
    broker = action_chunk_broker.RtcActionChunkBroker(
        policy=client,
        prediction_horizon=PREDICTION_HORIZON,
        execution_horizon=EXECUTION_HORIZON,
        # 首次推理延迟按 1 个控制步估计；之后 broker 会用最近实际延迟的最大值更新。
        initial_delay_steps=1,
    )

    rtde_c = None
    rtde_r = None
    gripper = None
    base_pipeline = None
    wrist_pipeline = None
    try:
        # dry-run 只建立只读状态连接；不创建 RTDEControlInterface，因此不可能调用 moveJ()。
        if not args.dry_run:
            rtde_c = rtde_control.RTDEControlInterface(ROBOT_IP)
        rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_IP)
        gripper = RobotiqGripper(ROBOT_IP, activate=not args.dry_run)

        base_serial, wrist_serial = _resolve_camera_serials()
        base_pipeline = _start_camera_pipeline(base_serial, "base")
        wrist_pipeline = _start_camera_pipeline(wrist_serial, "wrist")

        if args.dry_run:
            logging.warning("DRY RUN enabled: actions will only be printed; robot and gripper motion are disabled.")

        for step in range(num_steps):
            img = get_camera_image(base_pipeline)
            wrist_img = get_camera_image(wrist_pipeline)
            state = get_robot_state(rtde_r, gripper)

            observation = {
                # resize_with_pad + convert_to_uint8：跟训练时的预处理方式对齐，官方推荐这么写
                "observation/image": image_tools.convert_to_uint8(image_tools.resize_with_pad(img, 224, 224)),
                "observation/wrist_image": image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(wrist_img, 224, 224)
                ),
                "observation/state": state,   # 不需要自己归一化，服务器端会自动处理
                "prompt": task_instruction,
            }

            infer_start = time.monotonic()
            result = broker.infer(observation)
            infer_ms = (time.monotonic() - infer_start) * 1000
            action = np.asarray(result["actions"])
            if infer_ms > 20:
                logging.info("[step %d] broker.infer() blocked for %.1f ms", step, infer_ms)

            if args.dry_run:
                logging.info(
                    "[dry-run step %d] action=%s",
                    step,
                    np.array2string(action, precision=5, suppress_small=True),
                )
            else:
                assert rtde_c is not None
                execute_action(action, rtde_c, gripper)

            time.sleep(args.control_period)
    finally:
        broker.reset()
        if base_pipeline is not None:
            base_pipeline.stop()
        if wrist_pipeline is not None:
            wrist_pipeline.stop()
        if gripper is not None:
            gripper.close()
        if rtde_c is not None:
            rtde_c.disconnect()
        if rtde_r is not None:
            rtde_r.disconnect()


if __name__ == "__main__":
    main(parse_args())
