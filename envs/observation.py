from __future__ import annotations

import math

import numpy as np

from morai_rl.core.types import (
    ControlCommand,
    Observation,
    PathProjection,
    VehicleState,
    normalize_angle_rad,
)
from morai_rl.maps.local_bev import LocalBeVRenderer
from morai_rl.maps.reference_path import ReferencePath
from morai_rl.maps.route_corridor import CorridorProjection


LEGACY_VECTOR_OBSERVATION_KEYS = [
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

PROPRIO_VECTOR_OBSERVATION_KEYS = [
    "speed_mps",
    "target_speed_mps",
    "yaw_rate_rps",
    "steer_angle",
    "previous_steering",
    "previous_throttle",
    "previous_brake",
]

GUIDE_VECTOR_OBSERVATION_KEYS = [
    *PROPRIO_VECTOR_OBSERVATION_KEYS,
    "lateral_error_m",
    "heading_error_rad",
    "corridor_distance_m",
]

VECTOR_OBSERVATION_KEYS = LEGACY_VECTOR_OBSERVATION_KEYS
OBSERVATION_KEYS = VECTOR_OBSERVATION_KEYS


def _lookahead_label(distance_m: float) -> str:
    if float(distance_m).is_integer():
        return f"{int(distance_m)}m"
    return f"{str(float(distance_m)).replace('.', 'p')}m"


def lookahead_observation_keys(lookahead_distances_m: list[float]) -> list[str]:
    keys: list[str] = []
    for distance_m in lookahead_distances_m:
        label = _lookahead_label(distance_m)
        keys.extend(
            [
                f"lookahead_{label}_x",
                f"lookahead_{label}_y",
                f"lookahead_{label}_heading_error",
            ]
        )
    return keys


def resolve_vector_observation_keys(
    vector_profile: str = "legacy",
    lookahead_distances_m: list[float] | None = None,
) -> list[str]:
    profile = vector_profile.strip().lower()
    distances = [5.0, 10.0] if lookahead_distances_m is None else list(lookahead_distances_m)
    if profile == "legacy":
        return list(LEGACY_VECTOR_OBSERVATION_KEYS)
    if profile in {"proprio", "proprioception", "proprioceptive"}:
        return list(PROPRIO_VECTOR_OBSERVATION_KEYS)
    if profile in {"guide", "guided"}:
        return list(GUIDE_VECTOR_OBSERVATION_KEYS) + lookahead_observation_keys(distances)
    raise ValueError(f"unsupported observation vector profile: {vector_profile}")


def resolve_guide_observation_keys(
    vector_profile: str = "legacy",
    lookahead_distances_m: list[float] | None = None,
) -> set[str]:
    profile = vector_profile.strip().lower()
    if profile not in {"guide", "guided"}:
        return set()
    distances = [5.0, 10.0] if lookahead_distances_m is None else list(lookahead_distances_m)
    return {
        "lateral_error_m",
        "heading_error_rad",
        "corridor_distance_m",
        *lookahead_observation_keys(distances),
    }


def _add_lookahead_observations(
    named: dict[str, float],
    state: VehicleState,
    projection: PathProjection,
    reference_path: ReferencePath | None,
    lookahead_distances_m: list[float],
) -> None:
    if reference_path is None:
        for distance_m in lookahead_distances_m:
            label = _lookahead_label(distance_m)
            named[f"lookahead_{label}_x"] = 0.0
            named[f"lookahead_{label}_y"] = 0.0
            named[f"lookahead_{label}_heading_error"] = 0.0
        return

    cos_yaw = math.cos(state.yaw_rad)
    sin_yaw = math.sin(state.yaw_rad)
    for distance_m in lookahead_distances_m:
        label = _lookahead_label(distance_m)
        target_index = reference_path.lookahead_index(projection.nearest_index, float(distance_m))
        target = reference_path.points[target_index]
        dx = target.x - state.x
        dy = target.y - state.y
        local_x = cos_yaw * dx + sin_yaw * dy
        local_y = -sin_yaw * dx + cos_yaw * dy
        named[f"lookahead_{label}_x"] = local_x
        named[f"lookahead_{label}_y"] = local_y
        named[f"lookahead_{label}_heading_error"] = normalize_angle_rad(
            state.yaw_rad - target.yaw_rad
        )


def _vector_values_with_dropout(
    named: dict[str, float],
    vector_keys: list[str],
    guide_keys: set[str],
    guide_dropout_prob: float,
) -> list[float]:
    dropout_prob = max(0.0, min(1.0, float(guide_dropout_prob)))
    vector_values: list[float] = []
    for key in vector_keys:
        value = float(named[key])
        if key in guide_keys and dropout_prob > 0.0 and np.random.random() < dropout_prob:
            value = 0.0
        vector_values.append(value)
    return vector_values


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
    vector_profile: str = "legacy",
    guide_dropout_prob: float = 0.0,
    lookahead_distances_m: list[float] | None = None,
    reference_path: ReferencePath | None = None,
) -> Observation:
    distances = [5.0, 10.0] if lookahead_distances_m is None else list(lookahead_distances_m)
    corridor_distance_m = (
        float(corridor_projection.corridor_distance_m)
        if corridor_projection is not None
        else float(projection.distance_m)
    )
    named = {
        "speed_mps": state.speed_mps,
        "target_speed_mps": target_speed_mps,
        "yaw_rate_rps": state.wz,
        "steer_angle": state.steer_angle,
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
    _add_lookahead_observations(
        named=named,
        state=state,
        projection=projection,
        reference_path=reference_path,
        lookahead_distances_m=distances,
    )
    vector_keys = resolve_vector_observation_keys(vector_profile, distances)
    guide_keys = resolve_guide_observation_keys(vector_profile, distances)
    vector_values = _vector_values_with_dropout(
        named=named,
        vector_keys=vector_keys,
        guide_keys=guide_keys,
        guide_dropout_prob=guide_dropout_prob,
    )

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
