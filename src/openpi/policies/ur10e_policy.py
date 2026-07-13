import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_ur10e_example() -> dict:
    return {
        "observation/state": np.random.rand(7),  # 6 个关节角度 + 1 个夹爪开合状态
        "observation/image": np.random.randint(256, size=(256, 256, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(256, 256, 3), dtype=np.uint8),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    """把图像统一转成 uint8 的 (H, W, C) 格式。LeRobot 内部存的是 float32 (C, H, W)，
    训练时需要转换；推理阶段传进来的图像本身格式已经对，这一步会被跳过。"""
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class UR10eInputs(transforms.DataTransformFn):

    # 决定用的是 pi0 / pi0-FAST / pi0.5 中的哪个模型，这个字段本身不用改
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])

        # 下面字典里的 key 名（"state" / "image" / "image_mask" 等）是模型固定要求的，不要改。
        # 需要改的只是等号右边取自 data 的部分，以对应你自己数据集的字段。
        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
            # 两路腕部图像都用全零占位，而不是只占位右边这一路
                "left_wrist_0_rgb": np.zeros_like(base_image),
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                # 两路腕部图都是占位，mask 逻辑也要相应统一处理
                "left_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }

        if "actions" in data:
            inputs["actions"] = data["actions"]

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class UR10eOutputs(transforms.DataTransformFn):
    """
    把模型输出的动作，转换回 UR10e 能直接执行的格式。只在推理阶段使用。
    """

    def __call__(self, data: dict) -> dict:
        # 模型内部可能把动作维度统一 padding 过，这里只取回真正属于 UR10e 的前 7 维
        # （6 个关节 + 1 个夹爪）。这个 7 必须和 TrainConfig 里模型配置的 action_dim 一致。
        return {"actions": np.asarray(data["actions"][:, :7])}