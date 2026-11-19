import isaaclab.envs.mdp as mdp
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import RewardTermCfg as RewardTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as TerminationTerm
from isaaclab.utils import configclass

from . import reward, termination
from .actions import ActionsCfg
from .events import EventsCfg, PlaceEventsCfg
from .observations import ObservationsCfg, ObservationsWithCubeOrientationCfg
from .scene_cfg import SO101SceneCfg


@configclass
class RewardCfg:
    reach = RewardTerm(
        func=reward.reach_reward,
        weight=1.0,
        params={"std": 0.1},
    )

    lift = RewardTerm(
        func=reward.LiftReward,
        weight=30.0,
        params={"target_height_gain": 0.025},
    )

    cube_push = RewardTerm(
        func=reward.cube_push,
        weight=-2.0,
        params={"free_distance": 0.01},
    )

    action_rate = RewardTerm(func=mdp.action_rate_l2, weight=-1e-4)
    joint_velocity = RewardTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    cube_out_of_bounds = RewardTerm(
        func=termination.cube_out_of_bounds,
        weight=-15.0,
    )


@configclass
class TerminationCfg:
    cube_out_of_bounds = TerminationTerm(
        func=termination.cube_out_of_bounds,
    )
    time_out = TerminationTerm(
        func=mdp.time_out,
        time_out=True,
    )


@configclass
class PlaceRewardCfg:
    reach = RewardTerm(
        func=reward.reach_reward,
        weight=1.0,
        params={"std": 0.1},
    )
    lift = RewardTerm(
        func=reward.LiftReward,
        weight=10.0,
        params={
            "target_height_gain": 0.025,
            # "goal_deactivation_distance": 0.05,
        },
    )
    transport_coarse = RewardTerm(
        func=reward.TransportReward,
        weight=4.0,
        params={"std": 0.3},
    )
    transport_fine = RewardTerm(
        func=reward.TransportReward,
        weight=1.0,
        params={"std": 0.05},
    )
    pre_lowering_penalty = RewardTerm(
        func=reward.pre_lowering_penalty,
        weight=-2.0,
        params={"min_dist": 0.05},
    )
    lowering = RewardTerm(
        func=reward.LoweringReward,
        weight=2.0,
        params={"min_dist": 0.05},
    )
    up_lowering_penalty = RewardTerm(
        func=reward.up_lowering_penalty,
        weight=-2.0,
        params={"min_dist": 0.05},
    )
    premature_open = RewardTerm(
        func=reward.premature_open_penalty,
        weight=-5.0,
    )
    # place = RewardTerm(
    #     func=reward.place_reward,
    #     weight=0.0,
    # )
    release = RewardTerm(
        func=reward.release_success_reward,
        weight=15.0,
    )
    home_pose = RewardTerm(
        func=reward.HomeposeReward,
        weight=5.0,
        params={"std":0.5},
    )
    home_joint_deviation = RewardTerm(
        func=reward.home_joint_deviation_l1,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "shoulder_pan",
                    "shoulder_lift",
                    "elbow_flex",
                    "wrist_flex",
                    "wrist_roll",
                ],
            ),
        },
    )
    home_success = RewardTerm(
        func=reward.success_reward,
        weight=30.0,
        params={"joint_tolerance":0.2},
    )
    cube_push = RewardTerm(
        func=reward.cube_push,
        weight=-2.0,
        params={
            "free_distance": 0.01,
            "disable_above_height": 0.041,
        },
    )
    action_rate = RewardTerm(func=mdp.action_rate_l2, weight=-1e-4)
    joint_velocity = RewardTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    cube_out_of_bounds = RewardTerm(
        func=termination.cube_out_of_bounds,
        weight=-15.0,
    )


@configclass
class PlaceTerminationCfg(TerminationCfg):
    success = TerminationTerm(
        func=termination.task_success,
        params={"joint_tolerance":0.2},
    )


@configclass
class SO101EnvCfg(ManagerBasedRLEnvCfg):
    scene = SO101SceneCfg(num_envs=256, env_spacing=1.0)
    actions = ActionsCfg()
    observations = ObservationsCfg()
    events = EventsCfg()
    rewards = RewardCfg()
    terminations = TerminationCfg()

    def __post_init__(self):
        self.sim.dt = 0.01
        self.decimation = 2
        self.sim.render_interval = self.decimation
        self.episode_length_s = 7.0

        self.viewer.resolution = (1920, 1080)
        self.viewer.eye = (0.70, 0.70, 0.55)
        self.viewer.lookat = (0.18, 0.07, 0.08)

        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.friction_correlation_distance = 0.00625


@configclass
class SO101PoseObsEnvCfg(SO101EnvCfg):
    observations = ObservationsWithCubeOrientationCfg()


@configclass
class SO101PlaceEnvCfg(SO101EnvCfg):
    events = PlaceEventsCfg()
    rewards = PlaceRewardCfg()
    terminations = PlaceTerminationCfg()
