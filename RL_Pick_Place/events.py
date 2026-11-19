import isaaclab.envs.mdp as mdp

from isaaclab.utils import configclass
from isaaclab.managers import(EventTermCfg as EventTerm, SceneEntityCfg,)

from . import reward

@configclass
class EventsCfg:
    reset_scene = EventTerm(
        func = mdp.reset_scene_to_default,
        mode = "reset" ,
    )

    # randomize_cube_position = EventTerm(
    #     func = mdp.reset_root_state_uniform,
    #     mode = "reset",
    #     params={
    #         "pose_range" : {
    #             "x": (-0.003, 0.003),
    #             "y": (-0.003, 0.003),
    #         },
    #         "velocity_range" : {},
    #         "asset_cfg" : SceneEntityCfg("cube"),
    #         },
    #     )


@configclass
class PlaceEventsCfg(EventsCfg):
    reset_lift_history = EventTerm(
        func=reward.reset_lift_history,
        mode="reset",
    )
    reset_place_history = EventTerm(
        func=reward.reset_place_history,
        mode="reset",
    )
