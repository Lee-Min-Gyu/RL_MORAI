from __future__ import annotations

from morai_rl.core.types import PathProjection
from morai_rl.maps.route_corridor import CorridorProjection


def compute_reward(
    progress_delta_m: float,
    projection: PathProjection,
    corridor_projection: CorridorProjection | None,
    off_track: bool,
    blocked_collision: bool,
    stalled: bool,
    progress_reward_scale: float,
    alive_bonus: float,
    step_penalty_value: float,
    lateral_error_penalty_scale: float,
    track_width_m: float,
    vehicle_width_m: float,
    heading_error_penalty_scale: float,
    heading_error_penalty_clip_rad: float,
    boundary_proximity_penalty_scale: float,
    boundary_proximity_margin_m: float,
    footprint_boundary_margin_m: float | None,
    off_track_penalty_value: float,
    stalled_penalty_value: float,
) -> tuple[float, dict[str, float]]:
    progress_reward = float(progress_reward_scale) * progress_delta_m
    alive_bonus_value = float(alive_bonus)
    step_penalty = float(step_penalty_value)
    usable_half_width_m = max(
        0.0,
        0.5 * (max(0.0, float(track_width_m)) - max(0.0, float(vehicle_width_m))),
    )
    if usable_half_width_m > 1e-6:
        lateral_error_ratio = min(
            abs(float(projection.lateral_error_m)) / usable_half_width_m,
            1.0,
        )
    else:
        lateral_error_ratio = 0.0
    lateral_error_penalty = float(lateral_error_penalty_scale) * lateral_error_ratio
    heading_error_clip_rad = max(1e-6, float(heading_error_penalty_clip_rad))
    heading_error_ratio = min(
        abs(float(projection.heading_error_rad)) / heading_error_clip_rad,
        1.0,
    )
    heading_error_penalty = float(heading_error_penalty_scale) * heading_error_ratio
    boundary_proximity_penalty = 0.0
    if footprint_boundary_margin_m is not None and footprint_boundary_margin_m >= 0.0:
        boundary_margin_m = max(0.0, float(footprint_boundary_margin_m))
        safe_margin_m = max(1e-6, float(boundary_proximity_margin_m))
        proximity_ratio = min(
            max(0.0, safe_margin_m - boundary_margin_m) / safe_margin_m,
            1.0,
        )
        boundary_proximity_penalty = float(boundary_proximity_penalty_scale) * proximity_ratio
    elif corridor_projection is not None and corridor_projection.inside:
        boundary_margin_m = max(0.0, -float(corridor_projection.corridor_distance_m))
        safe_margin_m = max(1e-6, float(boundary_proximity_margin_m))
        proximity_ratio = min(
            max(0.0, safe_margin_m - boundary_margin_m) / safe_margin_m,
            1.0,
        )
        boundary_proximity_penalty = float(boundary_proximity_penalty_scale) * proximity_ratio

    off_track_penalty = float(off_track_penalty_value) if off_track else 0.0
    stalled_penalty = float(stalled_penalty_value) if stalled else 0.0

    reward = (
        progress_reward
        + alive_bonus_value
        - step_penalty
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
        "lateral_error_usable_half_width_m": usable_half_width_m,
        "lateral_error_ratio": lateral_error_ratio,
        "lateral_error_penalty": lateral_error_penalty,
        "heading_error_ratio": heading_error_ratio,
        "heading_error_penalty": heading_error_penalty,
        "boundary_proximity_penalty": boundary_proximity_penalty,
        "off_track_penalty": off_track_penalty,
        "stalled_penalty": stalled_penalty,
    }
    return reward, terms
