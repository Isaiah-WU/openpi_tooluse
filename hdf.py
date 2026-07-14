# import h5py

# # with h5py.File("/home/wbjsamuel/projects/openpi_demo/demo/episode_0.hdf", "r") as f:
# #     def print_structure(name, obj):
# #         if isinstance(obj, h5py.Dataset):
# #             print(f"{name} -> shape: {obj.shape}")
# #         else:
# #             print(f"{name} -> group")
# #     f.visititems(print_structure)
# with h5py.File("/home/wbjsamuel/projects/openpi_demo/demo/episode_0.hdf5", "r") as f:
#     print(f["observations/qpos"][:5, 6])
#     print(f["observations/gripper"][:5, 0])

import h5py

# 请将路径替换为你实际处理的 hdf5 文件路径
file_path = '/home/wbjsamuel/projects/openpi_demo/demo/episode_0.hdf5' 
with h5py.File(file_path, 'r') as f:
    print("文件中的顶层键有:", list(f.keys()))