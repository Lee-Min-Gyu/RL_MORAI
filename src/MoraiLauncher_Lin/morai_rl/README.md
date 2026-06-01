# MORAI RL ROS Sync

MORAI Simulator를 ROS Noetic synchronous mode로 제어하기 위한 PPO 강화학습 환경입니다.
현재 코드는 ROS service/topic 기반 학습 경로만 유지합니다.

## What Stays

- ROS synchronous mode control: `/SyncModeCtrlCmd`, `/SyncModeWaitForTick`
- ROS scenario reset: `/SyncModeScenarioLoad`
- ROS vehicle status / object status / IMU yaw-rate subscription
- Gymnasium wrapper and Stable-Baselines3 PPO training
- ROACH-style BeV encoder for `192x192` BeV observations
- racing guide vector observation with lookahead `5m`, `10m`, `15m`
- action space: `[accel_brake, steering]`
- checkpoint, resume, progress bar, action statistics, crash recovery save

## Layout

- `config/`: runtime TOML loader and dataclass config
- `core/`: reset manager, step clock, simulator recovery helpers, common types
- `envs/`: MORAI RL environment, Gym wrapper, reward, termination, observation
- `io/`: ROS synchronous transport
- `maps/`: reference path, route corridor, local BeV utilities
- `policies/`: ROACH-style feature extractor
- `scripts/`: training and ROS sync diagnostics

## Environment

Inside the ROS Noetic container:

```bash
source /opt/ros/noetic/setup.bash
source /root/catkin_ws/devel/setup.bash
source /opt/rl_venv/bin/activate
export PYTHONPATH=/root/catkin_ws/src/MoraiLauncher_Lin:$PYTHONPATH
```

Before training, verify that MORAI/rosbridge advertised the required services:

```bash
rosservice list | grep -E "SyncMode|Scenario"
```

Expected services include:

```text
/SyncModeCmd
/SyncModeCtrlCmd
/SyncModeSetGear
/SyncModeWaitForTick
/SyncModeScenarioLoad
```

If `/SyncModeScenarioLoad` is missing, reconnect MORAI network settings or restart rosbridge/MORAI before running training.

## Main Config

Use:

```text
morai_rl/stage1_ros_sync_config.toml
```

Important assumptions:

- ROS sync is enabled.
- `time_step = 20`, `step_hz = 50.0`
- scenario reset uses `/SyncModeScenarioLoad`
- scenario names are loaded round-robin:
  `ROS_RL`, `ROS_RL2`, `ROS_RL3`, `ROS_RL4`
- BeV is `192x192`
- static BeV and route corridor are built from K-City selected links

## Smoke Checks

Reset loop:

```bash
python -m morai_rl.scripts.check_env_reset_loop \
  --config /root/catkin_ws/src/MoraiLauncher_Lin/morai_rl/stage1_ros_sync_config.toml \
  --repeats 1
```

Short sync I/O check:

```bash
python -m morai_rl.scripts.check_ros_sync_io \
  --config /root/catkin_ws/src/MoraiLauncher_Lin/morai_rl/stage1_ros_sync_config.toml \
  --steps 50
```

Manual progress / BeV monitor:

```bash
python -m morai_rl.scripts.monitor_manual_progress \
  --config /root/catkin_ws/src/MoraiLauncher_Lin/morai_rl/stage1_ros_sync_config.toml \
  --show-bev
```

## Training

Example continuation from the steering-only checkpoint:

```bash
python -m morai_rl.scripts.train_ppo \
  --config /root/catkin_ws/src/MoraiLauncher_Lin/morai_rl/stage1_ros_sync_config.toml \
  --timesteps 2000000 \
  --save-dir runs/ppo_morai \
  --run-name ros_sync_racing_accel_steer_ab03_s015_2m \
  --resume-from /root/catkin_ws/runs/ppo_morai/ros_sync_racing_steer2d_roach_1m/checkpoints/ppo_checkpoint_1010641_steps.zip \
  --checkpoint-freq 5000 \
  --max-restarts 3 \
  --progress-bar \
  --n-steps 1024 \
  --batch-size 128 \
  --action-log-freq 1000 \
  --set-accel-brake-mean 0.3 \
  --set-accel-brake-std 0.2 \
  --set-steering-std 0.15
```

## Notes

- `policy_kwargs ignored when resuming from a saved model` is expected when loading an existing SB3 checkpoint.
- `--set-accel-brake-mean`, `--set-accel-brake-std`, and `--set-steering-std` mutate the loaded policy before training resumes.
- ROS sync `send()` advances one MORAI tick via `/SyncModeWaitForTick`; the environment does not add an extra wall-clock sleep in sync mode.
