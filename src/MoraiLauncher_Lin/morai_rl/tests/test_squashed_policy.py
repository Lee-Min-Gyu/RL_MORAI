from __future__ import annotations

import numpy as np
import torch as th
from gymnasium import spaces

from morai_rl.policies.squashed_policy import SquashedActorCriticPolicy


def _constant_schedule(_progress_remaining: float) -> float:
    return 1e-4


def test_squashed_policy_actions_and_log_prob_are_bounded_and_finite():
    obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
    action_space = spaces.Box(
        low=np.array([-1.0, -1.0], dtype=np.float32),
        high=np.array([1.0, 1.0], dtype=np.float32),
        dtype=np.float32,
    )
    policy = SquashedActorCriticPolicy(
        obs_space,
        action_space,
        _constant_schedule,
        net_arch=dict(pi=[8], vf=[8]),
        log_std_init=-0.5,
    )
    obs = th.zeros((32, 4), dtype=th.float32)

    actions, _values, log_prob = policy(obs, deterministic=False)
    assert th.all(actions <= 1.0)
    assert th.all(actions >= -1.0)
    assert th.isfinite(log_prob).all()

    deterministic_actions, _values, deterministic_log_prob = policy(obs, deterministic=True)
    assert th.all(deterministic_actions <= 1.0)
    assert th.all(deterministic_actions >= -1.0)
    assert th.isfinite(deterministic_log_prob).all()

    values, evaluated_log_prob, entropy = policy.evaluate_actions(obs, actions)
    assert values.shape == (32, 1)
    assert entropy is None
    assert th.isfinite(evaluated_log_prob).all()
