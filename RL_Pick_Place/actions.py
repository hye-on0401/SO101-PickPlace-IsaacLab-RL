import isaaclab.envs.mdp as mdp

from isaaclab.utils import configclass

ARM_JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
]

ARM_ACTION_SCALE = 0.2
GRIPPER_OPEN_POSITION = 1.0
GRIPPER_CLOSED_POSITION = 0.14

@configclass
class ActionsCfg:
    arm = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=ARM_JOINT_NAMES,
        scale=ARM_ACTION_SCALE,
        # use_zero_offset=True,
        use_default_offset=True,
        preserve_order=True,
    )

    gripper = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["gripper"],
        open_command_expr={"gripper": GRIPPER_OPEN_POSITION},
        close_command_expr={"gripper": GRIPPER_CLOSED_POSITION},
    )
