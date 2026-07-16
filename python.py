from examples.ur10e.ur10e_client import get_camera_image, get_robot_state, gripper

img = get_camera_image()
print(img.shape)

state = get_robot_state()
print(state)

gripper.move(0.3)