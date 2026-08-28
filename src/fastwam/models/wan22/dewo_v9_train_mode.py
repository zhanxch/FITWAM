"""DEWO v9 train mode: freeze the pretrained MoT, train CFG adapter + value head.

Trainable groups: text-side cross-attn K/V adapter **and** the VideoDiT
value head. Mix is ``ε_cfg = ε_0 + g w (ε_+ − ε_0)``. On-disk adapter
payloads still use format ``dewo_v5_uncond_adapter_v1`` with ``recipe=v9``.
"""

from __future__ import annotations

from typing import Iterable

import torch.nn as nn

from fastwam.models.wan22.uncond_adapter import (
    is_uncond_adapter_parameter_name,
    iter_uncond_adapter_parameters,
    set_uncond_adapter_enabled,
)
from fastwam.models.wan22.value_head import is_value_head_parameter_name
from fastwam.utils.logging_config import get_logger

logger = get_logger(__name__)

GROUP_FROZEN = "frozen"
GROUP_ADAPTER = "uncond_adapter"
GROUP_VALUE = "value_head"
UNCOND_ADAPTER_TRAIN_MODES = frozenset({"dewo_v9_uncond_adapter"})


def _unique_named_parameters(model: nn.Module) -> Iterable[tuple[str, nn.Parameter]]:
    seen: set[int] = set()
    for name, param in model.named_parameters():
        ident = id(param)
        if ident in seen:
            continue
        seen.add(ident)
        yield name, param


def classify_dewo_v9_parameter(name: str) -> str:
    if is_uncond_adapter_parameter_name(name):
        return GROUP_ADAPTER
    if is_value_head_parameter_name(name):
        return GROUP_VALUE
    return GROUP_FROZEN


def apply_dewo_v9_uncond_adapter_mode(model: nn.Module) -> dict[str, int]:
    """Freeze 本体; train CFG adapter + value head."""

    if not getattr(model, "uncond_adapter_injected", False):
        raise ValueError(
            "DEWO v9 train mode requires `model.uncond_adapter.enabled=true` "
            "so the CFG adapter is injected before freeze."
        )
    if getattr(model, "value_head", None) is None:
        raise ValueError("DEWO v9 train mode requires `model.value_head`.")

    model.eval()
    model.requires_grad_(False)
    dit = getattr(model, "dit", None)
    if dit is None:
        raise ValueError("DEWO v9 train mode requires `model.dit`.")
    dit.train()
    model.value_head.train()

    counts = {GROUP_FROZEN: 0, GROUP_ADAPTER: 0, GROUP_VALUE: 0}
    for name, param in _unique_named_parameters(model):
        group = classify_dewo_v9_parameter(name)
        counts[group] += param.numel()
        param.requires_grad = group in {GROUP_ADAPTER, GROUP_VALUE}

    proprio = getattr(model, "proprio_encoder", None)
    if proprio is not None:
        proprio.eval()
        proprio.requires_grad_(False)
    outcome = getattr(model, "outcome_encoder", None)
    if outcome is not None:
        outcome.eval()
        outcome.requires_grad_(False)

    set_uncond_adapter_enabled(model, True)
    logger.info(
        "DEWO v9 CFG-adapter+value: adapter=%.2fM value=%.2fM frozen=%.2fM",
        counts[GROUP_ADAPTER] / 1e6,
        counts[GROUP_VALUE] / 1e6,
        counts[GROUP_FROZEN] / 1e6,
    )
    if counts[GROUP_ADAPTER] < 1 or counts[GROUP_VALUE] < 1:
        raise ValueError(
            "DEWO v9 train mode produced an empty group: "
            f"adapter={counts[GROUP_ADAPTER]} value={counts[GROUP_VALUE]} "
            f"frozen={counts[GROUP_FROZEN]}"
        )
    return counts


def collect_dewo_v9_param_groups(
    model: nn.Module,
    *,
    lr: float,
    weight_decay: float,
    value_lr_scale: float = 1.0,
) -> list[dict]:
    adapter_params = [
        param for param in iter_uncond_adapter_parameters(model) if param.requires_grad
    ]
    value_params = [
        param
        for name, param in _unique_named_parameters(model)
        if classify_dewo_v9_parameter(name) == GROUP_VALUE and param.requires_grad
    ]
    if not adapter_params:
        raise ValueError("DEWO v9 optimizer found no trainable adapter parameters.")
    if not value_params:
        raise ValueError("DEWO v9 optimizer found no trainable value-head parameters.")
    named_trainable = [
        name for name, param in _unique_named_parameters(model) if param.requires_grad
    ]
    allowed = {GROUP_ADAPTER, GROUP_VALUE}
    bad = [name for name in named_trainable if classify_dewo_v9_parameter(name) not in allowed]
    if bad:
        raise ValueError(
            "Trainable non-adapter/value parameters in DEWO v9: "
            f"{bad[:8]}{'...' if len(bad) > 8 else ''}"
        )
    value_lr = float(lr) * float(value_lr_scale)
    logger.info(
        "DEWO v9 AdamW: adapter_n=%d value_n=%d adapter_lr=%.2e value_lr=%.2e scale=%.2f",
        len(adapter_params),
        len(value_params),
        lr,
        value_lr,
        float(value_lr_scale),
    )
    return [
        {
            "name": GROUP_ADAPTER,
            "params": adapter_params,
            "lr": float(lr),
            "weight_decay": float(weight_decay),
        },
        {
            "name": GROUP_VALUE,
            "params": value_params,
            "lr": value_lr,
            "weight_decay": float(weight_decay),
        },
    ]
