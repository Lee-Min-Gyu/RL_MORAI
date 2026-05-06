from __future__ import annotations

from morai_rl.core.types import ControlCommand, PathProjection
from morai_rl.maps.route_corridor import CorridorProjection


def compute_reward(
    progress_delta_m: float,
    corridor_projection: CorridorProjection | None,
    action: ControlCommand,
    previous_action: ControlCommand,
    off_track: bool,
    blocked_collision: bool,
    stalled: bool,
    progress_reward_scale: float,
    alive_bonus: float,
    step_penalty_value: float,
    steering_delta_penalty_scale: float,
    brake_penalty_scale: float,
    off_track_penalty_value: float,
    stalled_penalty_value: float,
) -> tuple[float, dict[str, float]]:
    progress_reward = float(progress_reward_scale) * progress_delta_m
    alive_bonus_value = float(alive_bonus)
    step_penalty = float(step_penalty_value)
    steering_delta_penalty = float(steering_delta_penalty_scale) * abs(
        action.steering - previous_action.steering
    )
    brake_penalty = float(brake_penalty_scale) * action.brake

    off_track_penalty = float(off_track_penalty_value) if off_track else 0.0
    stalled_penalty = float(stalled_penalty_value) if stalled else 0.0

    reward = (
        progress_reward
        + alive_bonus_value
        - step_penalty
        - steering_delta_penalty
        - brake_penalty
        - off_track_penalty
        - stalled_penalty
    )
    terms = {
        "progress_reward": progress_reward,
        "alive_bonus": alive_bonus_value,
        "step_penalty": step_penalty,
        "steering_delta_penalty": steering_delta_penalty,
        "brake_penalty": brake_penalty,
        "off_track_penalty": off_track_penalty,
        "stalled_penalty": stalled_penalty,
    }
    return reward, terms
