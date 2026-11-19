# SO-101 Pick and Place with Isaac Lab
## Introduction
This repository implements a custom reinforcement learning environment for the
SO-101 robot arm using NVIDIA Isaac Lab and RSL-RL.

A PPO-based policy is trained to complete the following sequence:
**Reach → Grasp → Lift → Transport → Lower → Release → Return Home**

The current policy uses state-based observations and has successfully completed
the full pick-and-place task in simulation.

> **Note:**
> This project is currently experimental and has only been validated in
> simulation.

### Current Limitations:

- State-based observations only; RGB observations are not yet supported.
- A single red cube and fixed target position are used.
- Color classification and object detection are not yet implemented.
- Real-robot deployment has not yet been validated.

## Environment

- GPU: NVIDIA GeForce RTX 5080, 16 GB
- NVIDIA Driver: 580.178.04
- nvidia-smi CUDA : 13.0
- PyTorch: 2.7.0+cu128
- PyTorch CUDA Build: 12.8
- Isaac Sim: 5.0.0
- Isaac Lab: v2.2.1
- RSL-RL: 2.3.3
- Python: 3.11.15
- Ubuntu: 22.04.5 LTS

## Policy

- Algorithm: PPO
- Framework: RSL-RL
- Actor network: `24 → 128 → 64 → 32 → 6`
- Critic network: `24 → 128 → 64 → 32 → 1`
- Activation function: ELU
- Observation type: State-based
- Action space: 5 arm joints and 1 gripper joint

## 🎥 Demo
https://github.com/user-attachments/assets/783bc93d-64ba-467a-9a84-a4bc9f18c40e

## Reward
**Reach → Lift → Transport → Lower → Release → Return Homepose**

After a successful release, the pick-and-place rewards are disabled and only the home-pose rewards remain active.

| Reward Term | Weight | Activation / Condition | Purpose |
|---|---:|---|---|
| `reach` | `+1.0` | Before the cube has been lifted | Moves the end-effector toward the cube grasp target |
| `lift` | `+10.0` | Until the cube reaches the target lift height | Rewards positive progress in cube height |
| `transport_coarse` | `+4.0` | While the cube remains lifted | Guides the cube broadly toward the target marker |
| `transport_fine` | `+1.0` | While the cube remains lifted | Provides finer guidance near the target marker |
| `pre_lowering_penalty` | `-2.0` | When the lifted cube is lowered before reaching the target | Prevents premature lowering during transport |
| `lowering` | `+2.0` | Near the target, with the gripper closed | Rewards lowering the cube and TCP together |
| `up_lowering_penalty` | `-2.0` | After lowering has started | Penalizes raising the cube again |
| `premature_open` | `-5.0` | When the gripper opens before a valid release pose | Prevents dropping the cube too early |
| `release` | `+15.0` | On the first valid release | Rewards placing and releasing the cube inside the target |
| `home_pose` | `+5.0` | After a valid release | Rewards progress toward the default joint pose |
| `home_joint_deviation` | `-0.5` | After a valid release | Penalizes arm-joint deviation from the home pose |
| `home_success` | `+30.0` | When all arm joints return within the home tolerance | Rewards completion of the full task |
| `cube_push` | `-2.0` | While the cube is low and displaced from its initial position | Discourages pushing the cube across the table |
| `action_rate` | `-1e-4` | Throughout the episode | Encourages smoother changes in actions |
| `joint_velocity` | `-1e-4` | Throughout the episode | Discourages excessive joint velocity |
| `cube_out_of_bounds` | `-15.0` | When the cube leaves the workspace or falls | Penalizes failure and terminates the episode |


```text
Reach
  ↓ cube lifted
Lift
  ↓ cube remains above 0.041 m
Transport
  ↓ cube enters the target region
Lowering
  ↓ cube is low and gripper opens
Release
  ↓ placed history becomes true
Return Home
  ↓ all arm joints are within 0.20 rad
Success
```

## Training

Run the following command from the repository root:

```bash
conda activate isaaclab
source ~/isaacsim/setup_conda_env.sh

~/IsaacLab/isaaclab.sh -p scripts/rsl_rl/train.py \
    --task Isaac-SO101-PickPlace-Full-v0 \
    --num_envs 256 \
    --max_iterations 1500 \
    --headless \
    --video \
    --video_interval 300 \
    --video_length 250 \
    --run_name so101_pick_and_place
```

## Play

Run the following command from the repository root:

```bash
conda activate isaaclab
source ~/isaacsim/setup_conda_env.sh

~/IsaacLab/isaaclab.sh -p scripts/rsl_rl/play.py \
    --task Isaac-SO101-PickPlace-Full-v0 \
    --num_envs 8 \
    --checkpoint logs/rsl_rl/so101_pick_place_v1/YOUR_RUN_DIRECTORY/model_1499.pt
```

## Acknowledgements
- [TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100)
  - Source of the SO-101 URDF and mesh assets
  - Licensed under the Apache License 2.0
  - The original URDF was converted locally to USD for Isaac Sim

- [MuammerBay/isaac_so_arm101](https://github.com/MuammerBay/isaac_so_arm101)
  - Reference for SO-101 actuator stiffness and damping parameters
  - Licensed under the BSD 3-Clause License

- [NVIDIA Isaac Lab](https://github.com/isaac-sim/IsaacLab)
  - Base training and playback scripts
  - Licensed under the BSD 3-Clause License

## 📄 License

The original code in this repository is licensed under the
[MIT License](LICENSE).

Third-party robot assets and adapted code remain subject to their respective
licenses. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for details.


## Future Work

1. Extend the task to randomized multi-color cube sorting.
2. Add RGB or depth-camera-based object perception.
3. Sim to Real SO-101 robot.
