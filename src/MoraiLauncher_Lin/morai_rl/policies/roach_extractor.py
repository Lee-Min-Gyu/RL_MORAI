from __future__ import annotations

import gymnasium as gym
import torch as th
from torch import nn

from stable_baselines3.common.preprocessing import get_flattened_obs_dim, is_image_space
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import TensorDict


class RoachBeVEncoder(nn.Module):
    """Lightweight BeV encoder for 192x192 channel-first observations."""

    def __init__(
        self,
        n_input_channels: int,
        input_height: int,
        input_width: int,
        feature_dim: int = 256,
    ) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        with th.no_grad():
            sample = th.zeros((1, n_input_channels, input_height, input_width), dtype=th.float32)
            flattened_dim = int(th.flatten(self.cnn(sample), start_dim=1).shape[1])
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_dim, int(feature_dim)),
            nn.ReLU(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))


class RoachCombinedExtractor(BaseFeaturesExtractor):
    """
    Dict observation extractor that uses a ROACH-style CNN for BeV image keys.

    Non-image keys, such as the vector observation, are flattened and concatenated
    with the BeV features.
    """

    def __init__(self, observation_space: gym.spaces.Dict, normalized_image: bool = False) -> None:
        super().__init__(observation_space, features_dim=1)

        extractors: dict[str, nn.Module] = {}
        total_concat_size = 0
        for key, subspace in observation_space.spaces.items():
            if is_image_space(subspace, normalized_image=normalized_image):
                n_input_channels = int(subspace.shape[0])
                encoder = RoachBeVEncoder(
                    n_input_channels=n_input_channels,
                    input_height=int(subspace.shape[1]),
                    input_width=int(subspace.shape[2]),
                    feature_dim=256,
                )
                with th.no_grad():
                    sample = th.zeros((1, *subspace.shape), dtype=th.float32)
                    encoded_dim = int(encoder(sample).shape[1])
                extractors[key] = encoder
                total_concat_size += encoded_dim
            else:
                extractors[key] = nn.Flatten()
                total_concat_size += get_flattened_obs_dim(subspace)

        self.extractors = nn.ModuleDict(extractors)
        self._features_dim = int(total_concat_size)

    def forward(self, observations: TensorDict) -> th.Tensor:
        encoded_tensor_list = []
        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        return th.cat(encoded_tensor_list, dim=1)
