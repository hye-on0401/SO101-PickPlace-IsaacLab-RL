import torch

from .reward import Homepose


WORKSPACE_X_MIN = 0.08108
WORKSPACE_X_MAX = 0.28908
WORKSPACE_Y_MIN = -0.06786
WORKSPACE_Y_MAX = 0.22214
CUBE_MIN_HEIGHT = -0.02


def task_success(env, joint_tolerance: float):
    return Homepose(env, joint_tolerance)


def cube_out_of_bounds(env):
    scene = env.scene
    cube_pos_w = scene["cube"].data.root_pos_w
    cube_pos_local = cube_pos_w - scene.env_origins

    outside_x = ((cube_pos_local[:, 0] < WORKSPACE_X_MIN) | (cube_pos_local[:, 0] > WORKSPACE_X_MAX))
    outside_y = ((cube_pos_local[:, 1] < WORKSPACE_Y_MIN) | (cube_pos_local[:, 1] > WORKSPACE_Y_MAX))
    fallen = cube_pos_local[:, 2] < CUBE_MIN_HEIGHT

    return outside_x | outside_y | fallen



