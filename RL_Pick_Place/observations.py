"""Policy observations shared by every SO-101 curriculum stage."""

import torch

import isaaclab.envs.mdp as mdp
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_unique, subtract_frame_transforms

RED_GOAL_POSITION = (0.26685, 0.06608, 0.016)


def cube_position_in_robot_root_frame(env):
    robot = env.scene["robot"]
    cube = env.scene["cube"]
    position, _ = subtract_frame_transforms(
        robot.data.root_pos_w,
        robot.data.root_quat_w,
        cube.data.root_pos_w,
    )
    return position


def cube_orientation_in_robot_root_frame(env):
    robot = env.scene["robot"]
    cube = env.scene["cube"]
    _, orientation = subtract_frame_transforms(
        robot.data.root_pos_w,
        robot.data.root_quat_w,
        cube.data.root_pos_w,
        cube.data.root_quat_w,
    )
    # q and -q represent the same rotation. Keeping w >= 0 prevents the
    # observation from jumping between two equivalent quaternion signs.
    return quat_unique(orientation)


def goal_position_in_robot_root_frame(env):
    robot = env.scene["robot"]
    goal_pos_l = torch.tensor(
        RED_GOAL_POSITION,
        device=env.device,
        dtype=robot.data.root_pos_w.dtype,
    ).unsqueeze(0)
    goal_pos_w = goal_pos_l + env.scene.env_origins
    position, _ = subtract_frame_transforms(
        robot.data.root_pos_w,
        robot.data.root_quat_w,
        goal_pos_w,
    )
    return position


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        cube_position = ObsTerm(func=cube_position_in_robot_root_frame)
        goal_position = ObsTerm(func=goal_position_in_robot_root_frame)
        previous_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.concatenate_terms = True
            self.enable_corruption = False

    policy = PolicyCfg()


@configclass
class ObservationsWithCubeOrientationCfg:
    """Comparison observation set that adds the cube quaternion."""

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        cube_position = ObsTerm(func=cube_position_in_robot_root_frame)
        cube_orientation = ObsTerm(func=cube_orientation_in_robot_root_frame)
        goal_position = ObsTerm(func=goal_position_in_robot_root_frame)
        previous_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.concatenate_terms = True
            self.enable_corruption = False

    policy = PolicyCfg()
