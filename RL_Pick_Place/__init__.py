import gymnasium as gym

gym.register(
    id="Isaac-SO101-PickPlace-PoseObs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            "RL_Pick_Place.ManagerBasedRLEnvcfg:SO101PoseObsEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "RL_Pick_Place.rsl_rl_ppo:SO101PPORunnerCfg"
        ),
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-SO101-PickPlace-Full-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            "RL_Pick_Place.ManagerBasedRLEnvcfg:SO101PlaceEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "RL_Pick_Place.rsl_rl_ppo:SO101PPORunnerCfg"
        ),
    },
    disable_env_checker=True,
)
