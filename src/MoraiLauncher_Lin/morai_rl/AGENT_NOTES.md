# MORAI ROS Sync RL Agent Notes

This workspace is a ROS Noetic + MORAI synchronous-mode RL setup inside Docker.
The RL package has been cleaned to use ROS sync transport only; legacy UDP/asynchronous RL files were removed.

## Latest Handoff Summary

Read this section first. It reflects the work completed in the most recent debugging session.

- The user is running the container with Docker `--pid=host`; this is required so training code can see and terminate host `Simulator.x86_64`.
- Training-time automatic recovery is implemented and has been tested with injected `RuntimeError`.
- Expected recovery path:
  `RuntimeError -> save ppo_model_crash.zip -> close env -> kill Simulator.x86_64 -> clean stale msc_ros/api.py -> roslaunch msc_ros msc_ros.launch -> wait ROS services -> resume PPO`.
- The injected recovery test confirmed that the simulator can be killed, relaunched, and training can resume from the crash checkpoint.
- For normal training, do not pass `--inject-runtime-error-after-steps`; it is only for recovery testing.
- A hard recovery-level wait on `/SyncModeInfo.can_send_tick` was tried and removed. Do not re-add it as a startup gate; it fired too early before sync/scenario readiness and broke recovery.
- The public README was updated with setup commands, `msc_ros.launch`, `network_file`, diagnostics, training, recovery, and injected test usage.
- `params.txt` contains sensitive launcher account fields; do not copy credentials into README or commits.

Important recent code changes:

- `scripts/train_ppo.py`
  - catches `RuntimeError` as recoverable during training
  - saves `ppo_model_crash.zip`
  - calls runtime recovery
  - resumes with `reset_num_timesteps=False`
  - adds `--inject-runtime-error-after-steps` for one-shot testing
- `core/simulator_process.py`
  - terminates processes by exact executable name
  - terminates stale processes by command-line substrings
  - launches relaunch command detached with stdio redirected to `DEVNULL`
- `config/runtime.py`
  - adds `relaunch_cleanup_cmdline_substrings`
- `stage1_ros_sync_config.toml`
  - recovery is enabled
  - relaunch command is `roslaunch msc_ros msc_ros.launch`
  - stale cleanup patterns are `roslaunch msc_ros msc_ros.launch` and `msc_ros/scripts/api.py`
- `io/ros_sync.py`
  - service calls are timeout-wrapped to avoid indefinite blocking
  - `_can_send_tick` is still stored from `/SyncModeInfo`, but not used as a recovery readiness gate
- `ros_drive/.../path_manager.py`
  - removed noisy `velocity_profile` debug print

Why relaunch was previously failing:

- `Simulator.x86_64` was killed correctly.
- Old `roslaunch msc_ros msc_ros.launch` / `msc_ros/scripts/api.py` could remain alive.
- New `api.py` then died quickly, likely due to stale launcher/network state or port ownership.
- Fix: recovery now kills stale `msc_ros`/`api.py` command-line matches before launching a new `msc_ros`.
- Do not broaden cleanup patterns enough to kill `rosbridge_server`; current patterns are intentionally specific.

Expected recovery logs:

```text
runtime_recovery killed_pids=[...]
runtime_recovery cleanup_pids=[...]
runtime_recovery launched_pid=...
runtime_recovery waiting 60.0s
runtime_recovery services_ready ...
runtime_recovery post_services_wait 15.0s
```

Waiting behavior during recovery:

- simulator terminate wait: up to `terminate_timeout_sec = 5.0`
- stale `msc_ros` cleanup wait: up to `terminate_timeout_sec = 5.0`
- fixed relaunch wait after `roslaunch`: `relaunch_wait_sec = 60.0`
- ROS service polling: up to `wait_services_timeout_sec = 300.0`, every `2.0` sec
- post-service settle wait: `post_services_wait_sec = 15.0`

Injected recovery test command:

```bash
python -m morai_rl.scripts.train_ppo \
  --config /root/catkin_ws/src/MoraiLauncher_Lin/morai_rl/stage1_ros_sync_config.toml \
  --timesteps 1010941 \
  --save-dir runs/ppo_morai \
  --run-name recovery_injected_test \
  --resume-from /root/catkin_ws/runs/ppo_morai/ros_sync_racing_steer2d_roach_1m/checkpoints/ppo_checkpoint_1010641_steps.zip \
  --checkpoint-freq 100 \
  --max-restarts 2 \
  --progress-bar \
  --n-steps 1024 \
  --batch-size 128 \
  --inject-runtime-error-after-steps 1010671
```

Here `1010671` is an absolute total timestep. With a `1010641` checkpoint, that means about 30 steps after resume.

Validation notes:

- Compile-only syntax check passed for recently edited Python files.
- `py_compile` may fail because existing `__pycache__` directories are owned by another user; prefer `compile(path.read_text(), path, "exec")` style checks if needed.
- Many files are owned by `nobody:nogroup`; editing may require temporarily adding file write permission.
- `git` may report dubious ownership at `/root/catkin_ws`; avoid destructive git commands.

## Current Git State

- Repo root: `/root/catkin_ws`
- Active branch: `ros-noetic`
- Latest local commit: `6aa558a Prune legacy UDP RL code`
- `ros-noetic` is currently ahead of `origin/ros-noetic` by 1 commit unless the user has pushed it.
- Push command from host/container with GitHub auth:

```bash
cd /root/catkin_ws
git push origin ros-noetic
```

## Important Paths

- Workspace: `/root/catkin_ws`
- RL package: `/root/catkin_ws/src/MoraiLauncher_Lin/morai_rl`
- Active config: `/root/catkin_ws/src/MoraiLauncher_Lin/morai_rl/stage1_ros_sync_config.toml`
- Reference path: `/root/catkin_ws/src/MoraiLauncher_Lin/output/reference_path_centerline.csv`
- Link selection: `/root/catkin_ws/src/MoraiLauncher_Lin/output/selected_links.json`
- Static BeV: `/root/catkin_ws/src/MoraiLauncher_Lin/output/kcity_selected_links_bev/static_bev.npz`
- Main steering checkpoint:
  `/root/catkin_ws/runs/ppo_morai/ros_sync_racing_steer2d_roach_1m/checkpoints/ppo_checkpoint_1010641_steps.zip`

## Runtime Setup

Inside the ROS Noetic container:

```bash
source /opt/ros/noetic/setup.bash
source /root/catkin_ws/devel/setup.bash
source /opt/rl_venv/bin/activate
export PYTHONPATH=/root/catkin_ws/src/MoraiLauncher_Lin:$PYTHONPATH
```

Before training, confirm MORAI/rosbridge advertised the sync services:

```bash
rosservice list | grep -E "SyncMode|Scenario"
```

Required services:

```text
/SyncModeCmd
/SyncModeCtrlCmd
/SyncModeSetGear
/SyncModeWaitForTick
/SyncModeScenarioLoad
```

If `/SyncModeScenarioLoad` is missing, reconnect MORAI network settings or restart rosbridge/MORAI. This is a runtime advertise/reconnect issue, not an RL-code issue.

## Current RL Design

- Transport is ROS synchronous mode only.
- `MoraiRLEnv` now rejects non-ROS-sync configs.
- `RosControlClient.send()` advances MORAI one sync tick via `/SyncModeCtrlCmd` and `/SyncModeWaitForTick`.
- `morai_env.py` skips extra `clock.sleep()` in ROS sync mode.
- Reset uses ROS sync scenario load through `/SyncModeScenarioLoad`.
- IMU yaw rate comes from `/imu` `angular_velocity.z` and is inserted into `VehicleState.wz`.
- Observation mode is `hybrid`: local BeV plus vector.
- BeV is ROACH-style `192x192`, 5 px/m equivalent:
  - `front_range_m = 30.4`
  - `rear_range_m = 8.0`
  - ego is about 40 px above the bottom.
- Vector profile is `racing_guide`; `target_speed_mps` is not part of that vector.
- Action space is 2D: `[accel_brake, steering]`.
  - `accel_brake > 0` maps to throttle.
  - `accel_brake < 0` maps to brake.
  - `steering` maps directly to MORAI normalized front steer command.
- Config currently uses `action_mode = "accel_brake_steering"`.

## Current Training Command

This command was confirmed to work after reconnecting MORAI/rosbridge:

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

Expected early logs:

```text
resuming_from=...
set_accel_brake_mean ...
set_accel_brake_std ...
set_steering_std ...
training_budget target=2000000 completed=1010641 remaining=989359
```

`policy_kwargs ignored when resuming from a saved model` is normal for SB3 checkpoint resume.

## Smoke Checks

Reset check:

```bash
python -m morai_rl.scripts.check_env_reset_loop \
  --config /root/catkin_ws/src/MoraiLauncher_Lin/morai_rl/stage1_ros_sync_config.toml \
  --repeats 1
```

Sync I/O check:

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

## Cleanup Summary

Removed from `morai_rl`:

- UDP transport modules: `io/*_udp.py`
- UDP diagnostics and send scripts
- old UDP configs: `stage1_rl5_config.toml`, `stage1_rl8_config.toml`, `example_config.toml`, `safe_config.toml`
- rule-based baseline and old example reference path

Kept:

- MORAI/ROS packages outside `morai_rl`
- `morai_msgs`
- `msc_ros` and MORAI launch-related packages
- `MORAI-ADModule` and K-City MGeo data
- current output/reference path/BeV files

Validation already performed after cleanup:

- all remaining `morai_rl` Python files passed `py_compile`
- `stage1_ros_sync_config.toml` loaded successfully
- `train_ppo --help` worked
- checkpoint `ppo_checkpoint_1010641_steps.zip` loaded with action space `Box(-1.0, 1.0, (2,), float32)` and observation `Dict(bev=(4,192,192), vector=(18,))`

## Runtime Recovery Work

- `core/simulator_process.py` can find and terminate host simulator processes by name if the container is launched with host PID visibility, e.g. Docker `--pid=host`.
- `scripts/demo_runtime_recovery.py` is a small demo loop that can intentionally kill `Simulator.x86_64` and observe reset behavior.
- Full automatic simulator relaunch during training is now implemented in `scripts/train_ppo.py`.
- On recoverable `RuntimeError`, training saves `ppo_model_crash.zip`, closes the env, kills `Simulator.x86_64`, cleans stale `msc_ros`/`api.py`, relaunches `roslaunch msc_ros msc_ros.launch`, waits for required sync services, then resumes from the crash checkpoint.
- `stage1_ros_sync_config.toml` controls recovery through the `[recovery]` section.
- Stale launcher cleanup is important. Without it, a new `api.py` can die immediately while an old `msc_ros` process is still alive.
- Injected `RuntimeError` recovery testing has passed.

## Next Planned Work

1. Observe a real MORAI unresponsive failure during long training and verify it follows the same recovery logs as the injected test.
2. If real failures reveal additional readiness issues after relaunch, add a targeted health check. Do not re-add a hard `can_send_tick` startup wait.
3. After single-simulator recovery is stable over long runs, distributed/parallel training can be investigated as a future direction.
