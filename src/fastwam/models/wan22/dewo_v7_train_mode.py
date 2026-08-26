"""DEWO v7: v5 freeze + v6 pool, CFG residual is posi minus fail.

Trainable parameters match v5/v6. Pair contrast lives in the mix
``ε_0 + w (ε_+ − ε_-)`` and in D_fail action BC on the failure text.
"""

from __future__ import annotations

from fastwam.models.wan22.dewo_v5_train_mode import (
    GROUP_ADAPTER,
    GROUP_FROZEN,
    UNCOND_ADAPTER_TRAIN_MODES,
    apply_dewo_v5_uncond_adapter_mode,
    classify_dewo_v5_parameter,
    collect_dewo_v5_param_groups,
)

apply_dewo_v7_uncond_adapter_mode = apply_dewo_v5_uncond_adapter_mode
classify_dewo_v7_parameter = classify_dewo_v5_parameter
collect_dewo_v7_param_groups = collect_dewo_v5_param_groups

__all__ = [
    "GROUP_ADAPTER",
    "GROUP_FROZEN",
    "UNCOND_ADAPTER_TRAIN_MODES",
    "apply_dewo_v7_uncond_adapter_mode",
    "classify_dewo_v7_parameter",
    "collect_dewo_v7_param_groups",
]
