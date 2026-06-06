from __future__ import annotations

import hashlib
import io
import math
from typing import Any

import numpy as np

try:
    import gymnasium as gym
    import torch as th
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
except ModuleNotFoundError as exc:  # pragma: no cover - runtime guard
    gym = None
    th = None
    PPO = None
    DummyVecEnv = None
    _SB3_IMPORT_ERROR = exc
else:
    _SB3_IMPORT_ERROR = None

from morai_rl.distributed.spaces import build_action_space, build_observation_space


class SpaceOnlyEnv(gym.Env if gym is not None else object):
    metadata = {"render_modes": []}

    def __init__(self, observation_space, action_space) -> None:
        if gym is None:  # pragma: no cover - runtime guard
            raise ModuleNotFoundError("gymnasium is required for distributed PPO") from _SB3_IMPORT_ERROR
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        return _zero_observation(self.observation_space), {}

    def step(self, action):
        del action
        return _zero_observation(self.observation_space), 0.0, False, False, {}


def build_space_only_vec_env(config_path: str, n_envs: int):
    if DummyVecEnv is None:  # pragma: no cover - runtime guard
        raise ModuleNotFoundError("stable-baselines3 is required") from _SB3_IMPORT_ERROR
    observation_space = build_observation_space(config_path)
    action_space = build_action_space()
    return DummyVecEnv(
        [
            lambda obs_space=observation_space, act_space=action_space: SpaceOnlyEnv(
                obs_space,
                act_space,
            )
            for _ in range(int(n_envs))
        ]
    )


def build_distributed_ppo(
    *,
    config_path: str,
    n_envs: int,
    n_steps: int,
    batch_size: int,
    n_epochs: int,
    learning_rate: float,
    gamma: float,
    gae_lambda: float,
    device: str,
    policy: str = "auto",
    features_extractor: str = "auto",
    std_init: float = 0.1,
    log_std_init: float | None = None,
    verbose: int = 0,
    tensorboard_log: str | None = None,
):
    if PPO is None:  # pragma: no cover - runtime guard
        raise ModuleNotFoundError("stable-baselines3 and torch are required") from _SB3_IMPORT_ERROR

    vec_env = build_space_only_vec_env(config_path, n_envs=n_envs)
    policy_name = _resolve_policy_name(policy, vec_env.observation_space)
    policy_kwargs = _build_policy_kwargs(
        features_extractor=features_extractor,
        policy_name=policy_name,
        observation_space=vec_env.observation_space,
        std_init=std_init,
        log_std_init=log_std_init,
    )
    model = PPO(
        policy=policy_name,
        env=vec_env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        verbose=verbose,
        tensorboard_log=tensorboard_log,
        device=device,
        policy_kwargs=policy_kwargs,
    )
    return model


def load_distributed_ppo(
    *,
    checkpoint_path: str,
    config_path: str,
    n_envs: int,
    n_steps: int,
    batch_size: int,
    n_epochs: int,
    learning_rate: float,
    gamma: float,
    gae_lambda: float,
    device: str,
    verbose: int = 0,
    tensorboard_log: str | None = None,
):
    if PPO is None:  # pragma: no cover - runtime guard
        raise ModuleNotFoundError("stable-baselines3 and torch are required") from _SB3_IMPORT_ERROR

    vec_env = build_space_only_vec_env(config_path, n_envs=n_envs)
    model = PPO.load(
        checkpoint_path,
        env=vec_env,
        device=device,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        gamma=gamma,
        gae_lambda=gae_lambda,
        verbose=verbose,
        tensorboard_log=tensorboard_log,
    )
    return model


def dump_policy_state(model) -> tuple[bytes, str]:
    if th is None:  # pragma: no cover - runtime guard
        raise ModuleNotFoundError("torch is required") from _SB3_IMPORT_ERROR
    buffer = io.BytesIO()
    th.save(model.policy.state_dict(), buffer)
    payload = buffer.getvalue()
    return payload, hashlib.sha256(payload).hexdigest()


def load_policy_state(model, payload: bytes) -> None:
    if th is None:  # pragma: no cover - runtime guard
        raise ModuleNotFoundError("torch is required") from _SB3_IMPORT_ERROR
    state_dict = th.load(io.BytesIO(payload), map_location=model.device)
    model.policy.load_state_dict(state_dict)


def _resolve_policy_name(policy: str, observation_space) -> str:
    if policy != "auto":
        return policy
    if isinstance(observation_space, gym.spaces.Dict):
        return "MultiInputPolicy"
    if len(getattr(observation_space, "shape", ())) == 3:
        return "CnnPolicy"
    return "MlpPolicy"


def _build_policy_kwargs(
    *,
    features_extractor: str,
    policy_name: str,
    observation_space,
    std_init: float,
    log_std_init: float | None,
) -> dict[str, Any]:
    policy_kwargs: dict[str, Any] = {}
    use_roach = features_extractor == "roach" or (
        features_extractor == "auto"
        and policy_name == "MultiInputPolicy"
        and isinstance(observation_space, gym.spaces.Dict)
    )
    if use_roach:
        from morai_rl.policies.roach_extractor import RoachCombinedExtractor

        policy_kwargs["features_extractor_class"] = RoachCombinedExtractor
    if log_std_init is None and std_init is not None and std_init > 0.0:
        log_std_init = math.log(float(std_init))
    if log_std_init is not None:
        policy_kwargs["log_std_init"] = float(log_std_init)
    return policy_kwargs


def _zero_observation(space):
    if isinstance(space, gym.spaces.Dict):
        return {key: _zero_observation(subspace) for key, subspace in space.spaces.items()}
    return np.zeros(space.shape, dtype=space.dtype)
