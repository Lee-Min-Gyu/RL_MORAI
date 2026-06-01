# MORAI ROS Sync RL Agent Notes

This workspace is a ROS Noetic + MORAI synchronous-mode RL setup inside Docker.

## Current Important Paths

- Workspace: `/root/catkin_ws`
- Package root: `/root/catkin_ws/src/MoraiLauncher_Lin/morai_rl`
- Active config: `/root/catkin_ws/src/MoraiLauncher_Lin/morai_rl/stage1_ros_sync_config.toml`
- Reference path: `/root/catkin_ws/src/MoraiLauncher_Lin/output/reference_path_centerline.csv`
- Link selection: `/root/catkin_ws/src/MoraiLauncher_Lin/output/selected_links.json`
- Global BeV: `/root/catkin_ws/src/MoraiLauncher_Lin/output/kcity_selected_links_bev/static_bev.npz`
- Main checkpoint: `/root/catkin_ws/runs/ppo_morai/ros_sync_racing_steer2d_roach_1m/checkpoints/ppo_checkpoint_1010641_steps.zip`

## Current RL Design

- Transport is ROS sync mode, not UDP.
- `RosControlClient.send()` advances MORAI one sync tick via `/SyncModeCtrlCmd` + `/SyncModeWaitForTick`.
- `morai_env.py` skips extra `clock.sleep()` for ROS sync mode.
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
- For steering-only staging, set `action_mode = "steering_only"`; the 2D policy still outputs both dims, but only steering is applied and throttle is fixed by config.

## Training Command Pattern

Resume from the 1M steering checkpoint and start accel/brake + steering training with controlled exploration:

```bash
export PYTHONPATH=/root/catkin_ws/src/MoraiLauncher_Lin:$PYTHONPATH
python -m morai_rl.scripts.train_ppo   --config /root/catkin_ws/src/MoraiLauncher_Lin/morai_rl/stage1_ros_sync_config.toml   --timesteps 2000000   --save-dir runs/ppo_morai   --run-name ros_sync_racing_accel_steer_roach_2m   --resume-from /root/catkin_ws/runs/ppo_morai/ros_sync_racing_steer2d_roach_1m/checkpoints/ppo_checkpoint_1010641_steps.zip   --checkpoint-freq 5000   --max-restarts 3   --progress-bar   --n-steps 1024   --batch-size 128   --action-log-freq 1000   --initial-step-log-count 5   --set-accel-brake-mean 0.3   --set-accel-brake-std 0.2   --set-steering-std 0.15
```

## Runtime Recovery Work

- `core/simulator_process.py` can find and terminate host simulator processes by name if the container is launched with host PID visibility, e.g. Docker `--pid=host`.
- `scripts/demo_runtime_recovery.py` is a small demo loop that can intentionally kill `Simulator.x86_64` and observe reset behavior.
- Full automatic simulator relaunch during training is still the next planned step. The user wants: on simulator unresponsive RuntimeError, kill `Simulator.x86_64`, then restart ROS launch with something like `roslaunch msc_ros msc_ros.launch`.

## Recovery Caveat After Accidental rm

On 2026-06-01, source files under `src` were accidentally deleted and restored mostly from `/root/catkin_ws/src.zip`.
Recovered successfully:

- `morai_rl`
- `MORAI-ADModule`
- `output`
- existing checkpoints under `/root/catkin_ws/runs`

Not fully recovered from that zip:

- `/root/catkin_ws/src/MSC/msc_ros` source package. Only an empty-ish directory/pycache fragments were visible after deletion. If a clean rebuild needs `msc_ros`, recover it from another backup, git remote, or host copy.

After recovery, `train_ppo.py`, `gym_wrapper.py`, `ros_sync.py`, `simulator_process.py`, `demo_runtime_recovery.py`, and `stage1_ros_sync_config.toml` were patched back to the latest ROS sync/2D-action training state.
