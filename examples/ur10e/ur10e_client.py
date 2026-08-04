"""
UR10e 客户端：从策略服务器获取动作，并在真实机器人上执行。

运行前提：
1. 已经在另一个终端 / 另一台机器上启动了 serve_policy.py，服务监听在 8000 端口
2. 已经 pip install -e . 安装了 openpi_tooluse/packages/openpi-client
3. 装好以下依赖：
   pip install ur_rtde pyrealsense2 numpy
4. 修改下面 ROBOT_IP 为你 UR10e 的真实 IP
5. 确认示教器上装了 Robotiq 的 URCap 插件（socket 端口默认 63352）
"""

import logging
import socket
import time

import numpy as np
import pyrealsense2 as rs
import rtde_control
import rtde_receive

from openpi_client import action_chunk_broker
from openpi_client import image_tools
from openpi_client import websocket_client_policy

# ========================

ROBOT_IP = "192.168.1.9"
GRIPPER_PORT = 63352
GRIPPER_MAX_POS = 255

HOST_IP = "127.0.0.1"
HOST_PORT = 8000

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
        "Stack the cup from the left plate into the cup on the right plate, then lift this nested pair and "
        "stack it onto the third cup standing alone on the tabletop. Transfer the complete three-cup stack "
        "into the basket. Next, stack the left plate onto the right plate, then place this stacked pair onto "
        "the plate already inside the basket, aligning their edges. Finally, take the rag from the right side "
        "of the workspace, thoroughly wipe the entire tabletop, and return the rag to its original position."
    ),
}

# 改这一行来选择当前 serve_policy.py 实际加载的是哪个 checkpoint/config。
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


rtde_c = rtde_control.RTDEControlInterface(ROBOT_IP)
rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_IP)
gripper = RobotiqGripper(ROBOT_IP)

pipeline = rs.pipeline()
rs_config = rs.config()
rs_config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(rs_config)


def get_camera_image():
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    img_bgr = np.asanyarray(color_frame.get_data())
    img_rgb = img_bgr[:, :, ::-1]   # BGR -> RGB
    return img_rgb


def get_robot_state():
    """6 个关节角度 + 1 个夹爪开合值，拼成长度为 7 的一维数组"""
    joints = rtde_r.getActualQ()          # list，长度 6
    gripper_pos = gripper.get_position()  # 0~1 之间
    return np.array(list(joints) + [gripper_pos], dtype=np.float32)


def execute_action(action):
    """action 是长度为 7 的一维数组：前 6 维是关节目标角度，第 7 维是夹爪目标位置"""
    joint_targets = action[:6].tolist()
    gripper_target = float(action[6])

    # asynchronous=True：不阻塞等这一步走完，配合外层 sleep 控制节奏
    rtde_c.moveJ(joint_targets, speed=0.5, acceleration=0.5, asynchronous=True)
    gripper.move(gripper_target)


# ======================================================================


def main():
    # host/port 已经在文件顶部的 HOST_IP / HOST_PORT 里配置，这里直接引用，不用改这一行
    client = websocket_client_policy.WebsocketClientPolicy(host=HOST_IP, port=HOST_PORT)
    task_instruction = CHECKPOINT_PROMPTS[ACTIVE_CHECKPOINT]

    num_steps = 200            # 想让机器人跑多少个时间步，自己定
    query_every_n_steps = 10   # 每隔多少步重新问一次服务器要新动作（不用每一步都问）

    # RtcActionChunkBroker = PipelinedActionChunkBroker(后台线程提前预取下一块动作,
    # 隐藏网络+推理延迟) + RTC 平滑衔接:预取时把当前 chunk 还没执行完的尾部动作
    # (committed tail)一起发给模型,让新 chunk 的开头几步被"锚定"到旧轨迹上,权重从
    # 1 衰减到 0,而不是在 chunk 边界处硬切换 —— 倒水/擦桌子这类连续运动任务在衔接
    # 点的顿挫本身就是失败原因之一。
    broker = action_chunk_broker.RtcActionChunkBroker(
        policy=client,
        action_horizon=query_every_n_steps,
        # 调参提示:
        # - replan_trigger_step: 默认 action_horizon//2。如果日志常出现
        #   "broker.infer() blocked for xxx ms",说明预取不够早,调大它(0~9)。
        # - prefix_attention_horizon: RTC 软约束衰减到 0 的边界,默认=action_horizon。
        #   调小会让模型更早恢复"完全自由重新规划",调大则让衔接更贴旧轨迹。
    )

    for step in range(num_steps):
        img = get_camera_image()
        state = get_robot_state()

        observation = {
            # resize_with_pad + convert_to_uint8：跟训练时的预处理方式对齐，官方推荐这么写
            "observation/image": image_tools.convert_to_uint8(image_tools.resize_with_pad(img, 224, 224)),
            "observation/state": state,   # 不需要自己归一化，服务器端会自动处理
            "prompt": task_instruction,
        }

        infer_start = time.monotonic()
        action = broker.infer(observation)
        infer_ms = (time.monotonic() - infer_start) * 1000
        if infer_ms > 20:
            # 缓存命中应该 <1ms；如果经常看到几十/上百 ms，说明后台预取没能在动作块
            # 用完前提前拿到结果，可以调大 replan_trigger_step（更早触发预取）或者
            # 检查服务器端 server_timing/infer_ms 看推理本身是不是太慢。
            logging.info(f"[step {step}] broker.infer() blocked for {infer_ms:.1f} ms")

        execute_action(action)
        time.sleep(0.1)   # 按你机器人实际控制频率调整，比如 10Hz 对应 0.1 秒


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()