from __future__ import annotations

from pathlib import Path

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

from morai_rl.core.types import ControlCommand
from morai_rl.envs.morai_env import MoraiRLEnv
from morai_rl.envs.observation import VECTOR_OBSERVATION_KEYS


class GymMoraiEnv(gym.Env if gym is not None else object):
    metadata = {"render_modes": []}

    def __init__(self, config_path: str | Path) -> None:
        if gym is None or spaces is None:  # pragma: no cover - runtime guard
            raise ModuleNotFoundError(
                "gymnasium is required for GymMoraiEnv. Install it with "
                "`pip install gymnasium`."
            ) from _GYM_IMPORT_ERROR

        super().__init__()
        self.config_path = str(config_path)
        self.env = MoraiRLEnv.from_toml(config_path)
        self.observation_mode = self.env.config.observation.mode.strip().lower()
        self._latest_observation = None
        self._latest_info: dict | None = None
        obs_dim = len(VECTOR_OBSERVATION_KEYS)
        if self.observation_mode == "vector":
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(obs_dim,),
                dtype=np.float32,
            )
        elif self.observation_mode == "bev":
            if self.env.local_bev_renderer is None:
                raise ValueError("bev observation mode requires a local BeV renderer")
            self.observation_space = spaces.Box(
                low=0,
                high=255,
                shape=(
                    len(self.env.local_bev_renderer.channel_names),
                    self.env.config.bev.height_px,
                    self.env.config.bev.width_px,
                ),
                dtype=np.uint8,
            )
        elif self.observation_mode == "hybrid":
            if self.env.local_bev_renderer is None:
                raise ValueError("hybrid observation mode requires a local BeV renderer")
            self.observation_space = spaces.Dict(
                {
                    "vector": spaces.Box(
                        low=-np.inf,
                        high=np.inf,
                        shape=(obs_dim,),
                        dtype=np.float32,
                    ),
                    "bev": spaces.Box(
                        low=0,
                        high=255,
                        shape=(
                            len(self.env.local_bev_renderer.channel_names),
                            self.env.config.bev.height_px,
                            self.env.config.bev.width_px,
                        ),
                        dtype=np.uint8,
                    ),
                }
            )
        else:
            raise ValueError(f"unsupported observation mode: {self.observation_mode}")
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        obs, info = self.env.reset()
        formatted = self._format_observation(obs)
        self._latest_observation = formatted
        self._latest_info = info
        return formatted, info

    def step(self, action):
        action_np = np.asarray(action, dtype=np.float32)
        if self.env.config.env.steering_only_control:
            command = ControlCommand(
                throttle=float(self.env.config.env.steering_only_fixed_throttle),
                brake=float(self.env.config.env.steering_only_fixed_brake),
                steering=float(action_np[2]),
            ).clipped()
        else:
            command = ControlCommand(
                throttle=float(action_np[0]),
                brake=float(action_np[1]),
                steering=float(action_np[2]),
            ).clipped()
        try:
            obs, reward, terminated, truncated, info = self.env.step(command)
        except TimeoutError as exc:
            obs, reward, terminated, truncated, info = self.env.timeout_transition(
                reason="state_timeout",
                error_message=str(exc),
            )
            print(
                "step timeout: terminating episode and moving to next scenario "
                f"(scenario={info.get('scenario_name')}, error={exc})"
            )
        formatted = self._format_observation(obs)
        self._latest_observation = formatted
        self._latest_info = info
        return formatted, float(reward), terminated, truncated, info

    def close(self) -> None:
        self.env.close()

    def get_latest_bev(self):
        if self._latest_observation is None:
            return None
        if self.observation_mode == "bev":
            return self._latest_observation
        if self.observation_mode == "hybrid":
            return self._latest_observation["bev"]
        return None

    def get_latest_viewer_snapshot(self) -> dict:
        return {
            "bev": self.get_latest_bev(),
            "channel_names": tuple(getattr(self.env.local_bev_renderer, "channel_names", ())),
            "info": {} if self._latest_info is None else dict(self._latest_info),
        }

    def _format_observation(self, obs):
        if self.observation_mode == "vector":
            return np.asarray(obs, dtype=np.float32)
        if self.observation_mode == "bev":
            return np.asarray(obs, dtype=np.uint8)
        return {
            "vector": np.asarray(obs["vector"], dtype=np.float32),
            "bev": np.asarray(obs["bev"], dtype=np.uint8),
        }
