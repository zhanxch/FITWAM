"""DEWO v6: same frozen-adapter freeze as v5, recoverability-event recipe.

The trainable parameter group is identical to v5. Sampling, CFG text, and
per-sample video BC / residual locks live in the dataset, Hydra CFG, and
``training_loss``.
"""

from __future__ import annotations

from fastwam.models.wan22.dewo_v5_train_mode import (
    GROUP_ADAPTER,
    GROUP_FROZEN,
    apply_dewo_v5_uncond_adapter_mode,
    classify_dewo_v5_parameter,
    collect_dewo_v5_param_groups,
)

apply_dewo_v6_uncond_adapter_mode = apply_dewo_v5_uncond_adapter_mode
classify_dewo_v6_parameter = classify_dewo_v5_parameter
collect_dewo_v6_param_groups = collect_dewo_v5_param_groups

__all__ = [
    "GROUP_ADAPTER",
    "GROUP_FROZEN",
    "apply_dewo_v6_uncond_adapter_mode",
    "classify_dewo_v6_parameter",
    "collect_dewo_v6_param_groups",
]
