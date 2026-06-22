from __future__ import annotations

from morai_rl.policies.bev_extractor import BeVLightCombinedExtractor, BeVLightEncoder


# Backward-compatible aliases for older checkpoints/configs that referenced
# morai_rl.policies.roach_extractor.
RoachBeVEncoder = BeVLightEncoder
RoachCombinedExtractor = BeVLightCombinedExtractor
