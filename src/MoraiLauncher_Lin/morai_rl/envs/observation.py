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


FRONT_STEER_ANGLE_MAX_DEG = 31.151662826538086
LONGITUDINAL_SPEED_SCALE_MPS = 50.0
LATERAL_SPEED_SCALE_MPS = 3.0
YAW_RATE_SCALE_RPS = 1.5
BOUNDARY_MARGIN_SCALE_M = 0.8

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
    "previous_throttle_brake",
]

PROPRIO_VECTOR_OBSERVATION_KEYS = [
    "speed_mps",
    "target_speed_mps",
    "yaw_rate_rps",
    "steer_angle",
    "previous_steering",
    "previous_throttle_brake",
]

GUIDE_VECTOR_OBSERVATION_KEYS = [
    *PROPRIO_VECTOR_OBSERVATION_KEYS,
    "lateral_error_m",
    "heading_error_rad",
]

RACING_GUIDE_VECTOR_OBSERVATION_KEYS = [
    "longitudinal_speed_mps",
    "lateral_speed_mps",
    "yaw_rate_rps",
    "steer_angle_norm",
    "previous_steering",
    "previous_throttle_brake",
    "lateral_error_m",
    "heading_error_rad",
    "left_boundary_margin_m",
    "right_boundary_margin_m",
    "track_width_m",
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
                f"lookahead_{label}_track_width_m",
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
    if profile in {"racing_guide", "racing-guided", "racing"}:
        return list(RACING_GUIDE_VECTOR_OBSERVATION_KEYS) + lookahead_observation_keys(distances)
    raise ValueError(f"unsupported observation vector profile: {vector_profile}")


def resolve_guide_observation_keys(
    vector_profile: str = "legacy",
    lookahead_distances_m: list[float] | None = None,
) -> set[str]:
    profile = vector_profile.strip().lower()
    if profile not in {"guide", "guided", "racing_guide", "racing-guided", "racing"}:
        return set()
    distances = [5.0, 10.0] if lookahead_distances_m is None else list(lookahead_distances_m)
    return {
        "lateral_error_m",
        "heading_error_rad",
        "left_boundary_margin_m",
        "right_boundary_margin_m",
        "track_width_m",
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
            named[f"lookahead_{label}_track_width_m"] = 0.0
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
        named[f"lookahead_{label}_track_width_m"] = reference_path.track_width_at(target_index)


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


def _normalize_vector_named(
    named: dict[str, float],
    reference_path: ReferencePath | None,
    lookahead_distances_m: list[float],
) -> dict[str, float]:
    vector_named = dict(named)
    vector_named["longitudinal_speed_mps"] = float(
        np.clip(
            float(named["longitudinal_speed_mps"]) / LONGITUDINAL_SPEED_SCALE_MPS,
            0.0,
            1.0,
        )
    )
    vector_named["lateral_speed_mps"] = _clip_unit(
        float(named["lateral_speed_mps"]) / LATERAL_SPEED_SCALE_MPS
    )
    vector_named["yaw_rate_rps"] = _clip_unit(
        float(named["yaw_rate_rps"]) / YAW_RATE_SCALE_RPS
    )
    vector_named["heading_error_rad"] = _clip_unit(
        float(named["heading_error_rad"]) / math.pi
    )

    track_width_m = max(0.0, float(named.get("track_width_m", 0.0)))
    vehicle_width_m = max(0.0, float(named.get("vehicle_width_m", 0.0)))
    usable_half_width_m = max(0.0, 0.5 * (track_width_m - vehicle_width_m))
    if usable_half_width_m > 1e-6:
        vector_named["lateral_error_m"] = _clip_unit(
            float(named["lateral_error_m"]) / usable_half_width_m
        )
    else:
        vector_named["lateral_error_m"] = 0.0

    for key in ("left_boundary_margin_m", "right_boundary_margin_m"):
        vector_named[key] = float(
            np.clip(
                float(named.get(key, 0.0)) / BOUNDARY_MARGIN_SCALE_M,
                0.0,
                1.0,
            )
        )

    max_track_width_m = (
        reference_path.max_track_width_m()
        if reference_path is not None
        else 0.0
    )
    if max_track_width_m > 1e-6:
        vector_named["track_width_m"] = float(
            np.clip(track_width_m / max_track_width_m, 0.0, 1.0)
        )
    else:
        vector_named["track_width_m"] = 0.0

    for distance_m in lookahead_distances_m:
        label = _lookahead_label(distance_m)
        distance_scale_m = max(1e-6, float(distance_m))
        lookahead_track_width_m = max(
            0.0,
            float(named.get(f"lookahead_{label}_track_width_m", 0.0)),
        )
        lookahead_half_width_m = 0.5 * lookahead_track_width_m
        vector_named[f"lookahead_{label}_x"] = _clip_unit(
            float(named[f"lookahead_{label}_x"]) / distance_scale_m
        )
        if lookahead_half_width_m > 1e-6:
            vector_named[f"lookahead_{label}_y"] = _clip_unit(
                float(named[f"lookahead_{label}_y"]) / lookahead_half_width_m
            )
        else:
            vector_named[f"lookahead_{label}_y"] = 0.0
        vector_named[f"lookahead_{label}_heading_error"] = _clip_unit(
            float(named[f"lookahead_{label}_heading_error"]) / math.pi
        )
        if max_track_width_m > 1e-6:
            vector_named[f"lookahead_{label}_track_width_m"] = float(
                np.clip(lookahead_track_width_m / max_track_width_m, 0.0, 1.0)
            )
        else:
            vector_named[f"lookahead_{label}_track_width_m"] = 0.0
    return vector_named


def _clip_unit(value: float) -> float:
    return float(np.clip(float(value), -1.0, 1.0))


def normalize_front_steer_angle(front_steer_angle_deg: float) -> float:
    return float(
        np.clip(
            -float(front_steer_angle_deg) / FRONT_STEER_ANGLE_MAX_DEG,
            -1.0,
            1.0,
        )
    )


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
    ego_vehicle_width_m: float = 0.0,
) -> Observation:
    distances = [5.0, 10.0] if lookahead_distances_m is None else list(lookahead_distances_m)
    corridor_distance_m = (
        float(corridor_projection.corridor_distance_m)
        if corridor_projection is not None
        else float(projection.distance_m)
    )
    vehicle_width_m = (
        float(state.width_m)
        if state.width_m is not None and state.width_m > 0.0
        else float(ego_vehicle_width_m)
    )
    if reference_path is None:
        left_boundary_margin_m = 0.0
        right_boundary_margin_m = 0.0
        track_width_m = 0.0
    else:
        left_boundary_margin_m, right_boundary_margin_m, track_width_m = (
            reference_path.boundary_margins_at(
                projection.nearest_index,
                projection.lateral_error_m,
                vehicle_width_m,
            )
        )
    named = {
        "speed_mps": state.speed_mps,
        "longitudinal_speed_mps": state.vx,
        "lateral_speed_mps": state.vy,
        "target_speed_mps": target_speed_mps,
        "yaw_rate_rps": state.wz,
        "steer_angle": state.steer_angle,
        "steer_angle_norm": normalize_front_steer_angle(state.steer_angle),
        "progress_ratio": projection.progress_ratio,
        "episode_progress_m": episode_progress_m,
        "progress_delta_m": progress_delta_m,
        "corridor_distance_m": corridor_distance_m,
        "left_boundary_margin_m": left_boundary_margin_m,
        "right_boundary_margin_m": right_boundary_margin_m,
        "track_width_m": track_width_m,
        "vehicle_width_m": vehicle_width_m,
        "heading_error_rad": projection.heading_error_rad,
        "lookahead_heading_error_5m": projection.lookahead_heading_error_5m,
        "lookahead_heading_error_10m": projection.lookahead_heading_error_10m,
        "lateral_error_m": projection.lateral_error_m,
        "previous_steering": previous_action.steering,
        "previous_throttle_brake": previous_action.throttle - previous_action.brake,
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
    vector_named = _normalize_vector_named(named, reference_path, distances)
    vector_values = _vector_values_with_dropout(
        named=vector_named,
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

    bev = bev_renderer.render(state, projection=projection)
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
