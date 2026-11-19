import torch

import isaaclab.envs.mdp as mdp
from isaaclab.managers import ManagerTermBase, RewardTermCfg

CUBE_INITIAL_HEIGHT = 0.016
LIFT_HEIGHT_THRESHOLD = 0.025
LIFT_SUCCESS_HEIGHT = CUBE_INITIAL_HEIGHT + LIFT_HEIGHT_THRESHOLD

GRIPPER_OPEN_THRESHOLD = 0.80

PLACE_XY_THRESHOLD = 0.025
PLACE_MAX_HEIGHT = 0.020
LOWERING_STEP_HEIGHT = 0.005
TCP_MAX_HEIGHT = 0.03
PREMATURE_OPEN_MAX_HEIGHT = 0.025

RED_GOAL_CUBE_POSITION = (0.26685, 0.06608, 0.016)
FIXED_CUBE_POSITION = (0.18508, 0.06891, 0.016)

HORIZONTAL_GRASP_OFFSET_W = (-0.0013, 0.0430, -0.0011)


def grasp_target_pos_w(env):
    cube_pos_w = env.scene["cube"].data.root_pos_w
    return cube_pos_w + cube_pos_w.new_tensor(HORIZONTAL_GRASP_OFFSET_W)


def _goal_position_w(env):
    goal_pos_l = torch.tensor(RED_GOAL_CUBE_POSITION, device=env.device,).unsqueeze(0)
    return goal_pos_l + env.scene.env_origins


def reach_reward(env, std: float):
    scene = env.scene

    tcp_pos_w = scene["ee_frame"].data.target_pos_w[:, 0, :]
    grasp_pos_w = grasp_target_pos_w(env)

    distance = torch.linalg.norm(grasp_pos_w - tcp_pos_w, dim=-1)
    reward = 1.0 - torch.tanh(distance / std)
    was_lifted = update_lift_history(env)

    return reward * (~was_lifted).float() * pick_place_active(env)


# def lift_reward(env):
#     scene = env.scene
#     cube_pos_w = scene["cube"].data.root_pos_w
#     cube_pos_l = cube_pos_w - scene.env_origins
#     cube_height = cube_pos_l[:, 2]

#     lift_height = torch.clamp(cube_height - CUBE_INITIAL_HEIGHT,min=0.0,)

#     lifted = lift_height > LIFT_HEIGHT_THRESHOLD
#     reward = 5.0 * lift_height + 1.0 * lifted.float()
#     return reward


# def lift_progress_reward(
#     env,
#     max_height_gain: float,
#     goal_deactivation_distance: float | None = None,
# ):
# #     cube_pos_w = env.scene["cube"].data.root_pos_w
#     cube_pos_l = cube_pos_w - env.scene.env_origins
#     height_gain = torch.clamp(
#         cube_pos_l[:, 2] - CUBE_INITIAL_HEIGHT,
#         min=0.0,
#     )
#     reward = torch.clamp(height_gain / max_height_gain, max=1.0)
#     if goal_deactivation_distance is not None:
#         goal_xy_distance = torch.linalg.norm(
#             cube_pos_w[:, :2] - _goal_position_w(env)[:, :2],
#             dim=-1,
#         )
#         reward = reward * (goal_xy_distance >= goal_deactivation_distance).float()
#     return reward

class LiftReward(ManagerTermBase):

    def __init__(self, cfg: RewardTermCfg, env):
        super().__init__(cfg, env)

        self.target_height_gain = cfg.params["target_height_gain"]

        self.prev_lift_potential = torch.zeros(env.num_envs,device=env.device,)
        self.lift_completed = torch.zeros(env.num_envs,dtype=torch.bool,device=env.device,)

    def reset(self, env_ids):
        cube_pos_l = (self._env.scene["cube"].data.root_pos_w- self._env.scene.env_origins)
        current_height = cube_pos_l[:, 2]

        lift_potential = torch.clamp((current_height - CUBE_INITIAL_HEIGHT) / self.target_height_gain,min=0.0,max=1.0,)

        self.prev_lift_potential[env_ids] = (lift_potential[env_ids])
        self.lift_completed[env_ids] = ( current_height[env_ids] > LIFT_SUCCESS_HEIGHT)

    def __call__(self,env,target_height_gain: float,):
        cube_pos_l = (env.scene["cube"].data.root_pos_w- env.scene.env_origins)
        current_height = cube_pos_l[:, 2]

        current_lift_potential = torch.clamp((current_height - CUBE_INITIAL_HEIGHT) / target_height_gain,min=0.0,max=1.0,)

        lift_potential_change = (current_lift_potential- self.prev_lift_potential) / env.step_dt
        lift_active = ~self.lift_completed

        reward = (
            lift_potential_change
            * lift_active.float()
            * pick_place_active(env)
        )

        newly_completed = (current_height > LIFT_SUCCESS_HEIGHT)

        self.lift_completed |= newly_completed # 기존값 판정 OR 새로운 값판정 => True가 되면 lift_active False
        self.prev_lift_potential.copy_(current_lift_potential)

        return reward

def cube_push(
    env,
    free_distance: float,
    disable_above_height: float | None = None,
):
    cube_pos_l = env.scene["cube"].data.root_pos_w - env.scene.env_origins
    start_xy = torch.tensor(
        FIXED_CUBE_POSITION[:2],
        device=env.device,
        dtype=cube_pos_l.dtype,
    )
    distance = torch.linalg.norm(cube_pos_l[:, :2] - start_xy, dim=-1)
    penalty = torch.clamp(distance - free_distance, min=0.0)
    if disable_above_height is not None:
        penalty = penalty * (cube_pos_l[:, 2] < disable_above_height).float()
    return penalty * pick_place_active(env)



class TransportReward(ManagerTermBase):

    def __init__(self, cfg: RewardTermCfg, env):
        super().__init__(cfg, env)
        self.std = cfg.params["std"]

        self.prev_potential = torch.zeros(env.num_envs, device=env.device,)
        self.prev_lifted = torch.zeros(env.num_envs,dtype=torch.bool,device=env.device,)

    def reset(self, env_ids):
        scene = self._env.scene
        cube_pos_w = scene["cube"].data.root_pos_w
        goal_pos_w = _goal_position_w(self._env)

        goal_xy_distance = torch.linalg.norm(cube_pos_w[:, :2] - goal_pos_w[:, :2],dim=-1,)
        potential = 1.0 - torch.tanh(goal_xy_distance / self.std)

        self.prev_potential[env_ids] = potential[env_ids]
        self.prev_lifted[env_ids] = False

    def __call__(self, env, std: float):
        scene = env.scene

        cube_pos_w = scene["cube"].data.root_pos_w
        cube_pos_l = cube_pos_w - scene.env_origins
        goal_pos_w = _goal_position_w(env)

        goal_xy_distance = torch.linalg.norm(cube_pos_w[:, :2] - goal_pos_w[:, :2],dim=-1,)
        current_potential = 1.0 - torch.tanh(goal_xy_distance / std)

        cube_lifted = (cube_pos_l[:, 2] > LIFT_SUCCESS_HEIGHT)

        transport_active = (cube_lifted & self.prev_lifted)

        progress = (current_potential - self.prev_potential) / env.step_dt

        reward = (
            progress
            * transport_active.float()
            * pick_place_active(env)
        )

        self.prev_potential.copy_(current_potential)
        self.prev_lifted.copy_(cube_lifted)

        return reward

class LoweringReward(ManagerTermBase):

    def __init__(self, cfg:RewardTermCfg, env):
        super().__init__(cfg, env)

        self.prev_height = torch.zeros(env.num_envs,device=env.device,)
        self.ever_lifted = torch.zeros(env.num_envs,dtype=torch.bool,device=env.device,)
        self.lowering_started = torch.zeros(env.num_envs,dtype=torch.bool,device=env.device,)
        self.prev_tcp_pos_w = torch.zeros(env.num_envs, device=env.device,)

    def reset(self, env_ids):
        cube_pos_w = self._env.scene["cube"].data.root_pos_w
        cube_pos_l = cube_pos_w - self._env.scene.env_origins
        current_height = cube_pos_l[:, 2]
        tcp_pos_w = self._env.scene["ee_frame"].data.target_pos_w[:, 0, :]

        self.prev_height[env_ids] = current_height[env_ids]
        self.ever_lifted[env_ids] = False
        self.lowering_started[env_ids] = False
        self.prev_tcp_pos_w[env_ids] = tcp_pos_w[env_ids, 2]

    def __call__(self, env, min_dist: float):
        scene = env.scene

        cube_pos_w = scene["cube"].data.root_pos_w
        cube_pos_l = cube_pos_w - scene.env_origins
        current_height = cube_pos_l[:, 2]
        gripper_closed = (scene["robot"].data.joint_pos[:, 5] <= GRIPPER_OPEN_THRESHOLD)
        tcp_pos_w = scene["ee_frame"].data.target_pos_w[:, 0, :]
        current_tcp_pos_w = tcp_pos_w[:, 2]

        goal_pos_w = _goal_position_w(env)
        goal_xy_distance = torch.linalg.norm(cube_pos_w[:, :2] - goal_pos_w[:, :2], dim=-1,)

        self.ever_lifted |= current_height > LIFT_SUCCESS_HEIGHT
        near_goal = goal_xy_distance < min_dist

        reached_lowering_region = (self.ever_lifted& near_goal)
        self.lowering_started |= reached_lowering_region
        lowering_active = (self.lowering_started& near_goal)

        prev_height_error = torch.clamp(self.prev_height - PLACE_MAX_HEIGHT, min=0.0)
        current_height_error = torch.clamp(current_height - PLACE_MAX_HEIGHT, min=0.0)
        height_error = prev_height_error - current_height_error
        lowering_progress = torch.clamp(height_error, min=0.0)

        cube_moved_down = (self.prev_height - current_height)
        tcp_moved_down = (self.prev_tcp_pos_w - current_tcp_pos_w)
        moved_down_error = torch.abs(cube_moved_down - tcp_moved_down)
        moved_down_scaling= 1.0-torch.tanh(moved_down_error/0.002)

        lowering_reward = (
            1000.0
            * lowering_progress
            * lowering_active.float()
            * moved_down_scaling
            * gripper_closed.float()
            * pick_place_active(env)
        )

        self.prev_height.copy_(current_height)
        self.prev_tcp_pos_w.copy_(current_tcp_pos_w)

        return lowering_reward



def lift_history_buffer(env):
    if not hasattr(env, "lifted_history"):
        env.lifted_history = torch.zeros(env.num_envs,dtype=torch.bool,device=env.device,)
    return env.lifted_history

def place_history_buffer(env):
    if not hasattr(env,"placed_history"):
        env.placed_history = torch.zeros(env.num_envs,dtype=torch.bool,device=env.device,)
    return env.placed_history


def pick_place_active(env):
    return (~place_history_buffer(env)).float()


def update_lift_history(env):
    was_lifted = lift_history_buffer(env)
    cube_pos_l = env.scene["cube"].data.root_pos_w - env.scene.env_origins
    was_lifted |= cube_pos_l[:, 2] > LIFT_SUCCESS_HEIGHT
    return was_lifted


def reset_lift_history(env, env_ids):
    lift_history_buffer(env)[env_ids] = False

def reset_place_history(env, env_ids):
    place_history_buffer(env)[env_ids] = False


def pre_lowering_penalty(env, min_dist: float):
    scene = env.scene
    cube_pos_w = scene["cube"].data.root_pos_w
    cube_pos_l = cube_pos_w - scene.env_origins
    current_height = cube_pos_l[:, 2]

    goal_pos_w = _goal_position_w(env)
    goal_xy_distance = torch.linalg.norm(cube_pos_w[:, :2] - goal_pos_w[:, :2], dim=-1,)
    was_lifted = update_lift_history(env)

    height_shortage = torch.clamp((LIFT_SUCCESS_HEIGHT - current_height)/LIFT_HEIGHT_THRESHOLD, min=0.0, max=1.0)
    before_goal = (goal_xy_distance >= min_dist)
    penalty = (
        was_lifted.float()
        * height_shortage
        * before_goal.float()
        * pick_place_active(env)
    )

    return penalty

class up_lowering_penalty(ManagerTermBase):
    def __init__(self, cfg:RewardTermCfg, env):
        super().__init__(cfg, env)
        self.prev_height_error = torch.zeros(env.num_envs, device=env.device,)
        self.lowering_started = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device,)

    def reset(self, env_ids):
        cube_pos_l = self._env.scene["cube"].data.root_pos_w - self._env.scene.env_origins
        current_height = cube_pos_l[:, 2]
        current_height_error = torch.clamp(current_height - PLACE_MAX_HEIGHT, min=0.0)

        self.prev_height_error[env_ids] = current_height_error[env_ids]
        self.lowering_started[env_ids] = False

    def __call__(self, env, min_dist: float):
        scene = env.scene
        cube_pos_w = scene["cube"].data.root_pos_w
        cube_pos_l = cube_pos_w - scene.env_origins
        current_height = cube_pos_l[:, 2]

        goal_pos_w = _goal_position_w(env)
        goal_xy_distance = torch.linalg.norm(cube_pos_w[:, :2] - goal_pos_w[:, :2], dim=-1,)
        was_lifted = update_lift_history(env)
        near_goal = (goal_xy_distance < min_dist)

        self.lowering_started |= (was_lifted & near_goal)

        current_height_error = torch.clamp(current_height - PLACE_MAX_HEIGHT, min=0.0)
        raising_amount = torch.clamp(current_height_error - self.prev_height_error, min=0.0)
        penalty = (
            1000.0
            * raising_amount
            * self.lowering_started.float()
            * pick_place_active(env)
        )

        self.prev_height_error.copy_(current_height_error)

        return penalty


def release_reward(env):
    scene = env.scene
    robot = scene["robot"]
    cube_pos_w = scene["cube"].data.root_pos_w
    cube_pos_l = cube_pos_w - scene.env_origins
    tcp_pos_w = scene["ee_frame"].data.target_pos_w[:, 0, :]
    tcp_pos_l = tcp_pos_w - scene.env_origins
    goal_pos_w = _goal_position_w(env)
    was_lifted = update_lift_history(env)

    goal_xy_distance = torch.linalg.norm(cube_pos_w[:, :2] - goal_pos_w[:, :2],dim=-1,)
    inside_goal = goal_xy_distance <= PLACE_XY_THRESHOLD
    cube_lowered = cube_pos_l[:, 2] <= PLACE_MAX_HEIGHT
    tcp_low = tcp_pos_l[:, 2] <= TCP_MAX_HEIGHT
    gripper_open = (robot.data.joint_pos[:, 5] > GRIPPER_OPEN_THRESHOLD)
    released = was_lifted & inside_goal & cube_lowered & tcp_low & gripper_open
    return released.float()


def premature_open_penalty(env):
    scene = env.scene
    robot = scene["robot"]
    cube_pos_w = scene["cube"].data.root_pos_w
    cube_pos_l = cube_pos_w - scene.env_origins
    goal_pos_w = _goal_position_w(env)
    was_lifted = update_lift_history(env)

    goal_xy_distance = torch.linalg.norm(
        cube_pos_w[:, :2] - goal_pos_w[:, :2],
        dim=-1,
    )
    inside_goal = goal_xy_distance <= PLACE_XY_THRESHOLD
    low_enough = cube_pos_l[:, 2] <= PREMATURE_OPEN_MAX_HEIGHT
    gripper_open = (
        robot.data.joint_pos[:, 5]
        > GRIPPER_OPEN_THRESHOLD
    )

    ready_to_release = inside_goal & low_enough
    premature_open = was_lifted & gripper_open & (~ready_to_release)

    return (
        premature_open.float()
        * pick_place_active(env)
        / env.step_dt
    )


class HomeposeReward(ManagerTermBase):
    def __init__(self, cfg:RewardTermCfg, env):
        super().__init__(cfg, env)
        self.std = cfg.params["std"]
        self.prev_potential= torch.zeros(env.num_envs, device=env.device,)

    def reset(self, env_ids):
        robot = self._env.scene["robot"]
        home_error = torch.linalg.norm(robot.data.joint_pos[:,:5]-robot.data.default_joint_pos[:,:5],dim=-1,)
        potential = 1.0-torch.tanh(home_error / self.std)
        self.prev_potential[env_ids] = potential[env_ids]

    def __call__(self, env, std:float):
        robot = env.scene["robot"]
        home_error = torch.linalg.norm(robot.data.joint_pos[:,:5]-robot.data.default_joint_pos[:,:5],dim=-1,)
        current_potential = 1.0-torch.tanh(home_error / std)
        progress = (current_potential-self.prev_potential)/env.step_dt
        reward= progress*place_history_buffer(env).float()
        self.prev_potential.copy_(current_potential)

        return reward


def home_joint_deviation_l1(env, asset_cfg):
    deviation = mdp.joint_deviation_l1(env, asset_cfg)
    return deviation * place_history_buffer(env).float()


def Homepose(env, joint_tolerance: float):
    robot = env.scene["robot"]
    arm_joint_pos = robot.data.joint_pos[:,:5]
    home_joint_pos = robot.data.default_joint_pos[:,:5]
    arm_home = torch.all(torch.abs(arm_joint_pos-home_joint_pos)<joint_tolerance, dim=-1,)
    
    return place_history_buffer(env)&arm_home

def success_reward(env, joint_tolerance: float):
    return Homepose(env, joint_tolerance).float()/ env.step_dt

def release_success_reward(env):
    placed = place_history_buffer(env)
    released_now = release_reward(env).bool()
    newly_placed = released_now & (~placed)
    placed |= released_now
    return newly_placed.float() / env.step_dt
