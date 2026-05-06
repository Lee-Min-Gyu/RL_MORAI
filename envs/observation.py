from __future__ import annotations

import numpy as np

from morai_rl.core.types import ControlCommand, Observation, PathProjection, VehicleState
from morai_rl.maps.local_bev import LocalBeVRenderer
from morai_rl.maps.route_corridor import CorridorProjection


VECTOR_OBSERVATION_KEYS = [
    "speed_mps",
    "target_speed_mps",
    "yaw_rate_rps",
    "progress_ratio",
    "episode_progress_m",
    "progress_delta_m",
    "corridor_distance_m",
    "heading_error_rad",
    "lookahead_heading_error_5m",
    "lookahead_heading_error_10m",
    "lateral_error_m",
    "previous_steering",
    "previous_throttle",
    "previous_brake",
]
OBSERVATION_KEYS = VECTOR_OBSERVATION_KEYS


def build_observation(
    state: VehicleState,
    projection: PathProjection,
    corridor_projection: CorridorProjection | None,
    previous_action: ControlCommand,
    target_speed_mps: float,
    episode_progress_m: float,
    progress_delta_m: float,
    observation_mode: str = "vector",
    bev_renderer: LocalBeVRenderer | None = None,
) -> Observation:
    corridor_distance_m = (
        float(corridor_projection.corridor_distance_m)
        if corridor_projection is not None
        else float(projection.distance_m)
    )
    named = {
        "speed_mps": state.speed_mps,
        "target_speed_mps": target_speed_mps,
        "yaw_rate_rps": state.wz,
        "progress_ratio": projection.progress_ratio,
        "episode_progress_m": episode_progress_m,
        "progress_delta_m": progress_delta_m,
        "corridor_distance_m": corridor_distance_m,
        "heading_error_rad": projection.heading_error_rad,
        "lookahead_heading_error_5m": projection.lookahead_heading_error_5m,
        "lookahead_heading_error_10m": projection.lookahead_heading_error_10m,
        "lateral_error_m": projection.lateral_error_m,
        "previous_steering": previous_action.steering,
        "previous_throttle": previous_action.throttle,
        "previous_brake": previous_action.brake,
    }
    vector_values = [float(named[key]) for key in VECTOR_OBSERVATION_KEYS]

    mode = observation_mode.strip().lower()
    if mode == "vector":
        values = np.asarray(vector_values, dtype=np.float32)
        return Observation(values=values, named=named, vector_values=vector_values, bev=None)

    if bev_renderer is None:
        raise ValueError(f"observation mode '{mode}' requires a local BeV renderer")

    bev = bev_renderer.render(state)
    if mode == "bev":
        return Observation(values=bev, named=named, vector_values=vector_values, bev=bev)
    if mode == "hybrid":
        return Observation(
            values={
                "vector": np.asarray(vector_values, dtype=np.float32),
                "bev": bev,
            },
            named=named,
            vector_values=vector_values,
            bev=bev,
        )
    raise ValueError(f"unsupported observation mode: {observation_mode}")
