# MORAI RL ROS2

`morai_rl` ROS2 전용 패키지입니다. 기존 `/home/mglee/Morai/MoraiLauncher_Lin/morai_rl`에서 학습 실행에 필요한 코드만 추려 `/home/mglee/Morai/src` 아래 `ament_python` 패키지 형태로 정리했습니다.

## Included

- ROS2 topic I/O
  - `/ctrl_cmd_0`: `morai_ros2_msgs/msg/CtrlCmd`
  - `/ego_vehicle_status`: `morai_ros2_msgs/msg/EgoVehicleStatus`
  - `/multi_ego_setting`: `morai_ros2_msgs/msg/MultiEgoSetting`
  - `/object_status`: `morai_ros2_msgs/msg/ObjectStatusList`
- Gym-like RL environment
- PPO training entrypoint
- Step-loop smoke test
- Rule-based simple driver
- Reference path, route corridor, local BeV utilities

UDP sender/receiver and UDP diagnostic scripts are intentionally not included.

## Build

From `/home/mglee/Morai`:

```bash
colcon build --packages-select morai_ros2_msgs morai_rl
source install/setup.bash
```

If `morai_ros2_msgs` has already been built, rebuilding only this package is enough:

```bash
colcon build --packages-select morai_rl
source install/setup.bash
```

Python training dependencies are still needed in the active environment:

```bash
pip install numpy gymnasium stable-baselines3 torch
```

Optional BeV viewer:

```bash
pip install pygame
```

## Run

Smoke-test one environment loop:

```bash
ros2 run morai_rl check_step_loop \
  --config /home/mglee/Morai/src/morai_rl/config/stage1_rl8_ros2.toml \
  --steps 50
```

Or via launch:

```bash
ros2 launch morai_rl check_step_loop.launch.py steps:=50
```

Repeated reset check:

```bash
ros2 run morai_rl check_reset \
  --config /home/mglee/Morai/src/morai_rl/config/stage1_rl8_ros2.toml \
  --repeats 10
```

Or via launch:

```bash
ros2 launch morai_rl check_reset.launch.py repeats:=10
```

Repeated reset plus first-drive check:

```bash
ros2 run morai_rl check_reset_drive \
  --config /home/mglee/Morai/src/morai_rl/config/stage1_rl8_ros2.toml \
  --episodes 5 \
  --steps 60 \
  --throttle 0.25
```

PPO training:

```bash
ros2 run morai_rl train_ppo \
  --config /home/mglee/Morai/src/morai_rl/config/stage1_rl8_ros2.toml \
  --timesteps 50000 \
  --run-name ros2_morai_rl
```

Or via launch:

```bash
ros2 launch morai_rl train_ppo.launch.py timesteps:=50000 run_name:=ros2_morai_rl
```

## Notes

- `config/stage1_rl8_ros2.toml` still references route and BeV assets under `/home/mglee/Morai/MoraiLauncher_Lin`, because those are data assets rather than ROS2 package code.
- Current reset uses ROS2 `MultiEgoSetting` with `gear = 1` and `ctrl_mode = 16`, then switches to `gear = 4` once just before the first control step.
- Synchronous simulation control is not implemented here yet. This package only moves the communication layer to ROS2 and packages the minimal RL runtime.
