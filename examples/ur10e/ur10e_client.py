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

import socket
import time

import numpy as np
import pyrealsense2 as rs
import rtde_control
import rtde_receive

from openpi_client import image_tools
from openpi_client import websocket_client_policy

# ============ 按你自己的实际情况修改这几个参数 ============

ROBOT_IP = "192.168.1.9"           # UR10e 控制柜的真实 IP（机械臂 + 夹爪共用这一个）
GRIPPER_PORT = 63352                 # Robotiq 通过 URCap 暴露的 socket 端口，通常就是这个
GRIPPER_MAX_POS = 255                # Robotiq 夹爪原始指令范围通常是 0~255，按实际情况调整

# 服务器（serve_policy.py）和客户端（这个脚本）都跑在同一台机器上
# （这台机器既有 GPU 又能直连机械臂），所以用 127.0.0.1
HOST_IP = "127.0.0.1"
HOST_PORT = 8000                     # serve_policy.py 默认监听的端口，一般不用改

# ============================================================


class RobotiqGripper:
    """通过 URCap 暴露的 socket（默认 63352 端口）控制 Robotiq 夹爪。
    协议是简单的文本指令，这是社区里最常用的实现方式。"""

    def __init__(self, robot_ip: str, port: int = GRIPPER_PORT):
        self.sock = socket.create_connection((robot_ip, port), timeout=2.0)
        # 激活夹爪（第一次连接后通常需要激活一次）
        self._send("SET ACT 1")
        self._send("SET GTO 1")

    def _send(self, cmd: str) -> str:
        self.sock.sendall((cmd + "\n").encode("utf-8"))
        return self.sock.recv(1024).decode("utf-8")

    def move(self, pos_0_to_1: float, speed: int = 150, force: int = 100):
        """pos_0_to_1: 0 = 全开, 1 = 全闭（跟你训练数据里 gripper 归一化的方向要对上）"""
        pos = int(np.clip(pos_0_to_1, 0.0, 1.0) * GRIPPER_MAX_POS)
        self._send(f"SET POS {pos}")
        self._send(f"SET SPE {speed}")
        self._send(f"SET FOR {force}")

    def get_position(self) -> float:
        reply = self._send("GET POS")
        pos = int(reply.strip().split()[-1])
        return pos / GRIPPER_MAX_POS


# 全局初始化一次连接，避免每一步都重新建立连接（这几行会在 import 时立刻尝试连接机器人）
rtde_c = rtde_control.RTDEControlInterface(ROBOT_IP)
rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_IP)
gripper = RobotiqGripper(ROBOT_IP)

# RealSense 相机初始化
pipeline = rs.pipeline()
rs_config = rs.config()
rs_config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(rs_config)


def get_camera_image():
    """从 RealSense 读取一帧彩色图像，转成 (H, W, 3) 的 RGB numpy array"""
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

    task_instruction = "pour water from the kettle into the cup, then move the cup away and wipe the table with the cloth"

    num_steps = 200          # 想让机器人跑多少个时间步，自己定
    query_every_n_steps = 10  # 每隔多少步重新问一次服务器要新动作（不用每一步都问）

    action_chunk = None

    for step in range(num_steps):
        # 每隔 N 步，或者还没有可用的动作块时，才重新查询服务器
        if action_chunk is None or step % query_every_n_steps == 0:
            img = get_camera_image()
            state = get_robot_state()

            observation = {
                # resize_with_pad + convert_to_uint8：跟训练时的预处理方式对齐，官方推荐这么写
                "observation/image": image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(img, 224, 224)
                ),
                "observation/state": state,   # 不需要自己归一化，服务器端会自动处理
                "prompt": task_instruction,
            }

            result = client.infer(observation)
            action_chunk = result["actions"]   # 形状是 (action_horizon, 7)
            chunk_index = 0

        # 从这一块动作序列里，依次取出一步来执行（"开环"执行，直到用完这一块再重新问）
        action = action_chunk[chunk_index]
        execute_action(action)
        chunk_index += 1

        time.sleep(0.1)   # 按你机器人实际控制频率调整，比如 10Hz 对应 0.1 秒


if __name__ == "__main__":
    main()