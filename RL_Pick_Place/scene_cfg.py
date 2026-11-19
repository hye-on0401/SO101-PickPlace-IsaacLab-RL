from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformerCfg, OffsetCfg
from isaaclab.utils import configclass


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SO101_USD_PATH = (
    PROJECT_ROOT
    / "SO-ARM100"
    / "Simulation"
    / "SO101"
    / "SO101_git.usd"
)


SO101_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(SO101_USD_PATH),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=2,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        joint_pos={
            "shoulder_pan": 0.0,
            "shoulder_lift": -1.686,
            "elbow_flex": 1.689,
            "wrist_flex": 1.361,
            "wrist_roll": -1.449,
            "gripper": 1.0,
        },
        joint_vel={".*": 0.0},

    ),
    actuators={
        # Shoulder Pan      moves: ALL masses                   (~0.8kg total)
        # Shoulder Lift     moves: Everything except base       (~0.65kg)
        # Elbow             moves: Lower arm, wrist, gripper    (~0.38kg)
        # Wrist Pitch       moves: Wrist and gripper            (~0.24kg)
        # Wrist Roll        moves: Gripper assembly             (~0.14kg)
        # Jaw               moves: Only moving jaw              (~0.034kg)
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["shoulder_.*", "elbow_flex", "wrist_.*"],
            effort_limit_sim=1.9,
            velocity_limit_sim=1.5,
            stiffness={
                "shoulder_pan": 200.0,  # Highest - moves all mass
                "shoulder_lift": 170.0,  # Slightly less than rotation
                "elbow_flex": 120.0,  # Reduced based on less mass
                "wrist_flex": 80.0,  # Reduced for less mass
                "wrist_roll": 50.0,  # Low mass to move
            },
            damping={
                "shoulder_pan": 80.0,
                "shoulder_lift": 65.0,
                "elbow_flex": 45.0,
                "wrist_flex": 30.0,
                "wrist_roll": 20.0,
            },
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["gripper"],
            effort_limit_sim=2.5,  # Increased from 1.9 to 2.5 for stronger grip
            velocity_limit_sim=1.5,
            stiffness=60.0,  # Increased from 25.0 to 60.0 for more reliable closing
            damping=20.0,  # Increased from 10.0 to 20.0 for stability
        ),
    },
    soft_joint_pos_limit_factor=0.9,
    )

def boundary_and_marker(
    prim_path: str,
    size: tuple[float, float, float],
    position: tuple[float, float, float],
    color: tuple[float, float, float],
) -> AssetBaseCfg:
    return AssetBaseCfg(
        prim_path=prim_path,
        spawn=sim_utils.CuboidCfg(
            size=size,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=color,
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=position,
        ),
    )


@configclass
class SO101SceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(
            color=(0.25, 0.25, 0.25),
        ),
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(
            intensity=2500.0,
            color=(0.8, 0.8, 0.8),
        ),
    )

    workspace = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Workspace",
        spawn=sim_utils.CuboidCfg(
            size=(0.42, 0.50, 0.001),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 1.0, 1.0),
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.20, 0.08, 0.0005),
        ),
    )

    boundary_top = boundary_and_marker(
        prim_path="{ENV_REGEX_NS}/BoundaryTop",
        size=(0.21, 0.002, 0.003),
        position=(0.18508, 0.22214, 0.0025),
        color=(0.0, 0.0, 0.0),
    )

    boundary_bottom = boundary_and_marker(
        prim_path="{ENV_REGEX_NS}/BoundaryBottom",
        size=(0.21, 0.002, 0.003),
        position=(0.18508, -0.06786, 0.0025),
        color=(0.0, 0.0, 0.0),
    )

    boundary_right = boundary_and_marker(
        prim_path="{ENV_REGEX_NS}/BoundaryRight",
        size=(0.002, 0.29, 0.003),
        position=(0.08108, 0.07614, 0.0025),
        color=(0.0, 0.0, 0.0),
    )

    boundary_left = boundary_and_marker(
        prim_path="{ENV_REGEX_NS}/BoundaryLeft",
        size=(0.002, 0.29, 0.003),
        position=(0.28908, 0.07614, 0.0025),
        color=(0.0, 0.0, 0.0),
    )

    center_marker = boundary_and_marker(
        prim_path="{ENV_REGEX_NS}/CenterMarker",
        size=(0.025, 0.020, 0.001),
        position=(0.18508, 0.06891, 0.0015),
        color=(0.0, 0.0, 0.0),
    )
    red_goal_marker = boundary_and_marker(
        prim_path="{ENV_REGEX_NS}/RedGoalMarker",
        size=(0.027, 0.020, 0.001),
        position=(0.26685, 0.06608, 0.0015),
        color=(1.0, 0.0, 0.0),
    )

    green_goal_marker = boundary_and_marker(
        prim_path="{ENV_REGEX_NS}/GreenGoalMarker",
        size=(0.027, 0.020, 0.001),
        position=(0.26685, 0.17285, 0.0015),
        color=(0.0, 1.0, 0.0),
    )

    blue_goal_marker = boundary_and_marker(
        prim_path="{ENV_REGEX_NS}/BlueGoalMarker",
        size=(0.027, 0.020, 0.001),
        position=(0.26685, -0.023, 0.0015),
        color=(0.0, 0.0, 1.0),
    )

    robot: ArticulationCfg = SO101_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot"
    )

    ee_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/gripper_link",
                name="gripper_frame_link",
                offset=OffsetCfg(
                    pos=(-0.0079, -0.000218121, -0.0981274),
                    rot=(0.0, 0.0, 1.0, 0.0),
                ),
            ),
        ],
        debug_vis=False,
    )

    cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.03, 0.03, 0.03),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(
                mass=0.018,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.0, 0.0),
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.18508, 0.06891, 0.016),
        ),
    )
