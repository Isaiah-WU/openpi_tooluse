"""
Minimal example script for converting a dataset to LeRobot format.

We use the Libero dataset (stored in RLDS) for this example, but it can be easily
modified for any other data you have saved in a custom format.

Usage:
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/data

If you want to push your dataset to the Hugging Face Hub, you can use the following command:
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/data --push_to_hub

Note: to run the script, you need to install tensorflow_datasets:
`uv pip install tensorflow tensorflow_datasets`

You can download the raw Libero datasets from https://huggingface.co/datasets/openvla/modified_libero_rlds
The resulting dataset will get saved to the $HF_LEROBOT_HOME directory.
Running this conversion script will take approximately 30 minutes.
"""

import shutil

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import tensorflow_datasets as tfds
import tyro

REPO_NAME = "wbjsamuel/ur10e_demo"   # 改成和 repo_id 一致  # Name of the output dataset, also used for the Hugging Face Hub
RAW_DATASET_NAMES = [
    "libero_10_no_noops",
    "libero_goal_no_noops",
    "libero_object_no_noops",
    "libero_spatial_no_noops",
]  # For simplicity we will combine multiple Libero datasets into one training dataset


def main(data_dir: str, *, push_to_hub: bool = False):
    # Clean up any existing dataset in the output directory
    output_path = HF_LEROBOT_HOME / REPO_NAME
    if output_path.exists():
        shutil.rmtree(output_path)

    # Create LeRobot dataset, define features to store
    # OpenPi assumes that proprio is stored in `state` and actions in `action`
    # LeRobot assumes that dtype of image data is `image`
    dataset = LeRobotDataset.create(
        repo_id="wbjsamuel/ur10e_demo",
        robot_type="ur10e",
        fps=30,
        features={
            "image": {
                "dtype": "image",
                "shape": (480, 640, 3),          # 改成真实分辨率
                "names": ["height", "width", "channel"],
        },
        # "wrist_image" 这一整块直接删掉——你没有真实腕部相机，
        # 不需要在数据集里存一份假的全零图像，白白占用磁盘空间
        "state": {
            "dtype": "float32",
            "shape": (7,),
            "names": ["state"],
        },
        "actions": {
            "dtype": "float32",
            "shape": (7,),
            "names": ["actions"],
        },
    },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    # Loop over raw Libero datasets and write episodes to the LeRobot dataset
    # You can modify this for your own data format
    # 举例：假设你的数据是存成一堆 HDF5 文件，每个文件是一次演示
    import h5py
    import glob

    for episode_path in glob.glob(f"{data_dir}/*.hdf5"):
        with h5py.File(episode_path, "r") as f:
            num_steps = f["action"].shape[0]          # 改成单数 "action"
            for i in range(num_steps):
                dataset.add_frame({
                    "image": f["observations/rgb"][i],   # rgb，不是 image
                    "state": f["observations/qpos"][i],   # qpos，不是 state
                    "actions": f["action"][i],
                    "task": "pour water from the kettle into the cup, then move the cup away and wipe the table with the cloth",
                })
        dataset.save_episode()

    # Optionally push to the Hugging Face Hub
    if push_to_hub:
        dataset.push_to_hub(
            tags=["ur10e", "pour-water", "real-robot"],
            private=True,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    tyro.cli(main)
