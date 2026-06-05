from __future__ import annotations

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ModuleNotFoundError as exc:  # pragma: no cover - runtime guard
    gym = None
    spaces = None
    _GYM_IMPORT_ERROR = exc
else:
    _GYM_IMPORT_ERROR = None

from morai_rl.config.runtime import load_config
from morai_rl.envs.observation import resolve_vector_observation_keys


def build_observation_space(config_path: str):
    if gym is None or spaces is None:  # pragma: no cover - runtime guard
        raise ModuleNotFoundError("gymnasium is required for distributed PPO") from _GYM_IMPORT_ERROR

    config = load_config(config_path)
    mode = config.observation.mode.strip().lower()
    vector_keys = resolve_vector_observation_keys(
        config.observation.vector_profile,
        config.observation.lookahead_distances_m,
    )
    vector_space = spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(len(vector_keys),),
        dtype=np.float32,
    )
    channel_count = 4 + (1 if config.bev.include_lane_marking else 0)
    bev_space = spaces.Box(
        low=0,
        high=255,
        shape=(channel_count, int(config.bev.height_px), int(config.bev.width_px)),
        dtype=np.uint8,
    )

    if mode == "vector":
        return vector_space
    if mode == "bev":
        return bev_space
    if mode == "hybrid":
        return spaces.Dict({"vector": vector_space, "bev": bev_space})
    raise ValueError(f"unsupported observation mode: {config.observation.mode!r}")


def build_action_space():
    if spaces is None:  # pragma: no cover - runtime guard
        raise ModuleNotFoundError("gymnasium is required for distributed PPO") from _GYM_IMPORT_ERROR
    return spaces.Box(
        low=np.array([-1.0, -1.0], dtype=np.float32),
        high=np.array([1.0, 1.0], dtype=np.float32),
        dtype=np.float32,
    )

