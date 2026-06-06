from __future__ import annotations

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.distributions import SquashedDiagGaussianDistribution
from stable_baselines3.common.policies import (
    ActorCriticCnnPolicy,
    ActorCriticPolicy,
    MultiInputActorCriticPolicy,
)
from stable_baselines3.common.preprocessing import get_action_dim


class SquashedActionDistributionMixin:
    """Use tanh-squashed Gaussian actions with PPO's existing actor-critic policy."""

    def _build(self, lr_schedule):
        if not isinstance(self.action_space, spaces.Box):
            raise TypeError("tanh-squashed actions require a continuous Box action space")
        if not (
            np.allclose(self.action_space.low, -1.0)
            and np.allclose(self.action_space.high, 1.0)
        ):
            raise ValueError("tanh-squashed PPO policy currently expects Box(-1, 1) actions")
        self.action_dist = SquashedDiagGaussianDistribution(get_action_dim(self.action_space))
        super()._build(lr_schedule)

    def raw_action_params(self, obs: th.Tensor | dict[str, th.Tensor]) -> tuple[th.Tensor, th.Tensor]:
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, _ = self.mlp_extractor(features)
        else:
            pi_features, _ = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
        raw_mean = self.action_net(latent_pi)
        raw_std = th.ones_like(raw_mean) * self.log_std.exp()
        return raw_mean, raw_std


class SquashedActorCriticPolicy(SquashedActionDistributionMixin, ActorCriticPolicy):
    pass


class SquashedActorCriticCnnPolicy(SquashedActionDistributionMixin, ActorCriticCnnPolicy):
    pass


class SquashedMultiInputActorCriticPolicy(SquashedActionDistributionMixin, MultiInputActorCriticPolicy):
    pass
