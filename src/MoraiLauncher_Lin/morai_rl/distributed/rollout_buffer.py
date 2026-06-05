from __future__ import annotations

import numpy as np

try:
    import torch as th
except ModuleNotFoundError as exc:  # pragma: no cover - runtime guard
    th = None
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None


def fill_rollout_buffer(model, rollouts: list[dict]) -> None:
    if th is None:  # pragma: no cover - runtime guard
        raise ModuleNotFoundError("torch is required") from _TORCH_IMPORT_ERROR
    if not rollouts:
        raise ValueError("at least one rollout is required")

    rollout_steps = int(rollouts[0]["steps"])
    n_envs = len(rollouts)
    for rollout in rollouts:
        if int(rollout["steps"]) != rollout_steps:
            raise ValueError("all rollouts must have the same number of steps")

    buffer = model.rollout_buffer
    buffer.reset()
    for step_index in range(rollout_steps):
        obs = _stack_observations([rollout["obs"][step_index] for rollout in rollouts])
        actions = np.stack([rollout["actions"][step_index] for rollout in rollouts]).astype(np.float32)
        rewards = np.asarray([rollout["rewards"][step_index] for rollout in rollouts], dtype=np.float32)
        episode_starts = np.asarray(
            [rollout["episode_starts"][step_index] for rollout in rollouts],
            dtype=np.float32,
        )
        values = th.as_tensor(
            [rollout["values"][step_index] for rollout in rollouts],
            dtype=th.float32,
            device=model.device,
        )
        log_probs = th.as_tensor(
            [rollout["log_probs"][step_index] for rollout in rollouts],
            dtype=th.float32,
            device=model.device,
        )
        buffer.add(obs, actions, rewards, episode_starts, values, log_probs)

    last_values = th.as_tensor(
        [rollout["last_value"] for rollout in rollouts],
        dtype=th.float32,
        device=model.device,
    )
    dones = np.asarray([rollout["last_episode_start"] for rollout in rollouts], dtype=np.float32)
    buffer.compute_returns_and_advantage(last_values=last_values, dones=dones)
    if not buffer.full:
        raise RuntimeError("rollout buffer was not filled")
    model.num_timesteps += rollout_steps * n_envs


def _stack_observations(observations: list):
    first = observations[0]
    if isinstance(first, dict):
        return {
            key: np.stack([obs[key] for obs in observations])
            for key in first.keys()
        }
    return np.stack(observations)

