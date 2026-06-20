# Current MORAI RL Handoff

This file summarizes the current distributed PPO/MORAI RL state as of the latest debugging session. It is intended as the first file a new agent should read before touching the code.

## Goal

Train a distributed PPO racing agent in MORAI using this desktop as the learner/server and multiple workers. The target is not just survival, but fast 1-lap completion and eventually robust multi-lap driving.

The current map/path is K-City 2025. The reference lap length is estimated around 1750 m, and the current lap-completion threshold is 1720 m.

## Current Config Snapshot

Main config:

`/root/catkin_ws/src/MoraiLauncher_Lin/morai_rl/stage1_ros_sync_config.toml`

Important current values:

```toml
[env]
step_hz = 25.0
action_repeat = 2
max_steps = 6000
action_mode = "throttle_brake_steering"
progress_reward_scale = 8.0
step_penalty = 0.08
steering_delta_penalty_scale = 0.1
brake_penalty_scale = 0.02
boundary_proximity_penalty_scale = 0.03
boundary_proximity_margin_m = 0.3
lateral_error_penalty_scale = 0.02
heading_error_penalty_scale = 0.5
off_track_penalty = 1500.0
lap_completion_enabled = true
lap_complete_distance_m = 1720.0
lap_completed_bonus = 2000.0
timeout_penalty = 2500.0
stalled_penalty = 1500.0
```

Scenario curriculum is currently all scenarios with ROS_RL9/ROS_RL10 duplicated because they were historically hard:

```toml
scenario_load_file_names = [
  "ROS_RL1","ROS_RL2","ROS_RL3","ROS_RL4","ROS_RL5","ROS_RL6",
  "ROS_RL9","ROS_RL10","ROS_RL7","ROS_RL8","ROS_RL9","ROS_RL10",
  "ROS_RL11","ROS_RL12","ROS_RL13"
]
```

Observation is hybrid BeV + vector racing guide. Lookahead has been increased for fresh training:

```toml
lookahead_distances_m = [5.0, 10.0, 15.0, 20.0, 25.0, 30.0]
```

The current `racing_guide` vector is 36D. It uses `longitudinal_speed_mps`,
`lateral_speed_mps`, and `steer_angle_norm` instead of scalar `speed_mps` and
raw `steer_angle`, keeps previous action as `previous_steering` and
`previous_throttle_brake`, adds current left/right boundary margin and track width,
then 6 lookahead samples x `x/y/heading_error/track_width`.

Because lookahead count changed, old checkpoints with the previous observation shape should not be resumed directly.

## Reward Design State

Reward is still primarily dense progress based:

```text
reward =
  progress_reward
  + alive_bonus
  - step_penalty
  - steering_delta_penalty
  - brake_penalty
  - lateral_error_penalty
  - heading_error_penalty
  - boundary_proximity_penalty
  - off_track_penalty
  - stalled_penalty
  + lap_completed_bonus when lap completed
  - proportional timeout_penalty when max_steps reached without lap completion
```

Lap completion:

- Uses `episode_progress_m`, not absolute path `progress_m`.
- `episode_progress_m >= lap_complete_distance_m` triggers:
  - `terminated = True`
  - `reason = "lap_completed"`
  - `reward_terms["lap_completed_bonus"] = lap_completed_bonus`

Timeout penalty:

- Applied only when `reason == "max_steps"`.
- It is proportional to the remaining lap distance:

```text
actual_timeout_penalty =
  timeout_penalty * clamp(1 - episode_progress_m / lap_complete_distance_m, 0, 1)
```

- With `timeout_penalty = 2500` and `lap_complete_distance_m = 1720`:
  - timeout at 860 m gives about -1250
  - timeout near 1600 m gives a much smaller penalty

Off-track:

- Off-track episodes currently receive `off_track_penalty` only.
- They do not also receive timeout/incomplete-lap penalty. This was intentional to avoid making early off-track failures too negative.

## Major Code Changes

### BeV-Based Off-Track

File:

`/root/catkin_ws/src/MoraiLauncher_Lin/morai_rl/envs/morai_env.py`

Off-track is now BeV footprint based when BeV is available:

```python
off_track = footprint_off_track if bev_contact["available"] else projection_off_track
```

Reason:

- Route corridor projection works per link segment.
- In corners/link joins, BeV can show a continuous drivable corridor while projection says the vehicle is slightly outside one link.
- This caused false off-track near linked-road/corner areas.

Projection/corridor information is still logged and used for observation/reward terms, but it no longer terminates episodes when BeV contact metrics are available.

### Manual Monitor Off-Track Debug

File:

`/root/catkin_ws/src/MoraiLauncher_Lin/morai_rl/scripts/monitor_manual_progress.py`

Manual monitor now logs:

```text
off_track=Y/N
off_reason=footprint/projection_fallback/-
proj_out=Y/N
boundary_touch=Y/N
bev_out=<pixels>
bev_boundary=<pixels>
```

This helps distinguish:

- projection corridor says outside
- BeV footprint actually outside drivable area
- BeV boundary touch only

### Lap Completion And Timeout

Files:

- `/root/catkin_ws/src/MoraiLauncher_Lin/morai_rl/config/runtime.py`
- `/root/catkin_ws/src/MoraiLauncher_Lin/morai_rl/envs/morai_env.py`

Added config fields:

```python
lap_completion_enabled
lap_complete_distance_m
lap_completed_bonus
timeout_penalty
```

### Distributed Action Stats

File:

`/root/catkin_ws/src/MoraiLauncher_Lin/morai_rl/scripts/train_ppo_distributed_worker.py`

Worker supports:

```bash
--action-log-freq 1000
```

This prints actual rollout action statistics:

```text
raw_steer_mean/raw_steer_std
squashed_steer_mean/min/max
raw_throttle_brake_mean/raw_throttle_brake_std
squashed_throttle_brake_mean/min/max
saturation_ratio
clip_ratio
```

Interpretation:

- `raw_*_mean`: policy distribution mean for each observation, averaged over the logged window.
- `raw_*_std`: policy distribution std; for SB3 PPO this is close to the learned per-action global std.
- `squashed_*`: actual sampled action after tanh/clip sent to the env.

This is worker-side because it reflects actual states and actions encountered during rollout. Learner-side policy std logging would be cleaner for pure policy parameters, but worker logs are useful for behavior debugging.

## Current Training Observations

Earlier issue:

- ROS_RL9/ROS_RL10 had repeated off-track at a right-angle turn.
- Root cause was partly false projection-based off-track around linked route geometry.
- After switching off-track to BeV footprint, those scenarios improved substantially.

Recent issue:

- Agent often reaches `max_steps`, but only progresses around 800-900 m in 6000 steps.
- At 50 Hz, 6000 steps = 120 s.
- To cover 1720-1750 m in 120 s, average speed must be about 14.3-14.6 m/s, about 51-53 km/h.
- Current 800-900 m in 120 s is roughly 24-27 km/h, too slow.

Current direction:

- Use stronger progress/lap objective rather than resetting action means.
- Do not force-reset accel/steering std during resume.
- For fresh training with new observation shape, recommended action initialization:

```bash
--action-dist tanh_squashed \
--set-throttle-brake-mean 0.35 \
--set-throttle-brake-std 0.25 \
--set-steering-std 0.20
```

Avoid `steering_std` too high initially because it caused unstable/noisy driving. Avoid `throttle_brake_mean` too high because it can push into corners too aggressively.

## Recommended Fresh Distributed Run

Because lookahead changed, use a new run instead of resuming old checkpoints.

Learner example:

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m morai_rl.scripts.train_ppo_distributed_learner \
  --host 0.0.0.0 \
  --port 50051 \
  --workers 3 \
  --rollout-steps 1024 \
  --timesteps 3000000 \
  --save-dir /root/catkin_ws/runs/ppo_morai_distributed \
  --batch-size 512 \
  --n-epochs 4 \
  --learning-rate 1e-4 \
  --device cuda \
  --run-name <date_or_date_time> \
  --action-dist tanh_squashed \
  --set-throttle-brake-mean 0.35 \
  --set-throttle-brake-std 0.25 \
  --set-steering-std 0.20 \
  --checkpoint-freq 6144 \
  --progress-bar
```

Worker example:

```bash
CUDA_VISIBLE_DEVICES=1 python3 -m morai_rl.scripts.train_ppo_distributed_worker \
  --server-host 127.0.0.1 \
  --server-port 50051 \
  --worker-id worker_server \
  --rollout-steps 1024 \
  --device cuda \
  --action-dist tanh_squashed \
  --action-log-freq 1000 \
  --max-restarts 3
```

Remote workers must use the same code and config, especially the same lookahead dimensions.

## ROS Environment Reminder

Before running training inside the container:

```bash
cd /root/catkin_ws/src/MoraiLauncher_Lin
source /opt/ros/noetic/setup.bash
source /root/catkin_ws/devel/setup.bash
source /opt/rl_venv/bin/activate
export PYTHONPATH=/root/catkin_ws/src/MoraiLauncher_Lin:/root/catkin_ws/devel/lib/python3/dist-packages:/opt/ros/noetic/lib/python3/dist-packages:$PYTHONPATH
```

If `morai_msgs` is missing, ROS/catkin setup or `PYTHONPATH` is incomplete.

## Things To Watch Next

Do not judge by total reward alone. Watch:

- `reason=lap_completed` starts appearing.
- `reason=max_steps` with `timeout_penalty` should become less common.
- `episode_progress_m` should trend from 800-900 m toward 1720 m.
- `squashed_throttle_brake_mean` should not collapse below about 0.15 for long periods.
- `raw_throttle_brake_mean` should generally stay positive and adapt by segment.
- `steering_delta_penalty` should not explode; if steering oscillation persists, consider `steering_delta_penalty_scale = 0.15`, but be careful because too high can hurt tight corner learning.
- If speed remains too low despite lap/timeout terms, consider adding episode speed metrics to worker logs rather than changing action means immediately.

## Important Caveats

- `target_speed_mps` is not currently used as a direct speed reward.
- Fixed target-speed reward is not recommended yet because racing needs high speed on straights and lower speed in sharp turns.
- `boundary_proximity_penalty` still uses route-corridor projection distance, not BeV red-pixel overlap. Its current scale is very small, so it is mostly a weak warning.
- Checkpoints from the old 3-lookahead observation should not be used with the new 6-lookahead config.
