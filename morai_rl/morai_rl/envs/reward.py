from __future__ import annotations

from morai_rl.core.types import ControlCommand, PathProjection
from morai_rl.maps.route_corridor import CorridorProjection


def compute_reward(
    progress_delta_m: float,
    projection: PathProjection,
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
    lateral_error_penalty_scale: float,
    lateral_error_penalty_clip_m: float,
    heading_error_penalty_scale: float,
    heading_error_penalty_clip_rad: float,
    boundary_proximity_penalty_scale: float,
    boundary_proximity_margin_m: float,
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
    lateral_error_penalty = float(lateral_error_penalty_scale) * min(
        abs(projection.lateral_error_m),
        max(0.0, float(lateral_error_penalty_clip_m)),
    )
    heading_error_penalty = float(heading_error_penalty_scale) * min(
        abs(projection.heading_error_rad),
        max(0.0, float(heading_error_penalty_clip_rad)),
    )
    boundary_proximity_penalty = 0.0
    if corridor_projection is not None and corridor_projection.inside:
        boundary_margin_m = max(0.0, -float(corridor_projection.corridor_distance_m))
        proximity_m = max(0.0, float(boundary_proximity_margin_m) - boundary_margin_m)
        boundary_proximity_penalty = float(boundary_proximity_penalty_scale) * proximity_m

    off_track_penalty = float(off_track_penalty_value) if off_track else 0.0
    stalled_penalty = float(stalled_penalty_value) if stalled else 0.0

    reward = (
        progress_reward
        + alive_bonus_value
        - step_penalty
        - steering_delta_penalty
        - brake_penalty
        - lateral_error_penalty
        - heading_error_penalty
        - boundary_proximity_penalty
        - off_track_penalty
        - stalled_penalty
    )
    terms = {
        "progress_reward": progress_reward,
        "alive_bonus": alive_bonus_value,
        "step_penalty": step_penalty,
        "steering_delta_penalty": steering_delta_penalty,
        "brake_penalty": brake_penalty,
        "lateral_error_penalty": lateral_error_penalty,
        "heading_error_penalty": heading_error_penalty,
        "boundary_proximity_penalty": boundary_proximity_penalty,
        "off_track_penalty": off_track_penalty,
        "stalled_penalty": stalled_penalty,
    }
    return reward, terms
