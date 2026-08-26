"""DEWO v5: freeze the pretrained MoT, train a bounded CFG residual.

Video text stays on the S0 base prompt. Action text gets CFG dropout.
Video K/V residual is trained by action loss plus an identity penalty,
not video BC. Adapter modules are text-side cross-attn K/V on video and
action. Saved as `dewo_v5_uncond_adapter_v1`.
"""

from __future__ import annotations

from typing import Iterable

import torch.nn as nn

from fastwam.models.wan22.uncond_adapter import (
    is_uncond_adapter_parameter_name,
    iter_uncond_adapter_parameters,
    set_uncond_adapter_enabled,
)
from fastwam.utils.logging_config import get_logger

logger = get_logger(__name__)

GROUP_FROZEN = "frozen"
GROUP_ADAPTER = "uncond_adapter"
UNCOND_ADAPTER_TRAIN_MODES = frozenset(
    {
        "dewo_v5_uncond_adapter",
        "dewo_v6_uncond_adapter",
        "dewo_v7_uncond_adapter",
        "dewo_v8_uncond_adapter",
        "dewo_v9_uncond_adapter",
    }
)


def _unique_named_parameters(model: nn.Module) -> Iterable[tuple[str, nn.Parameter]]:
    seen: set[int] = set()
    for name, param in model.named_parameters():
        ident = id(param)
        if ident in seen:
            continue
        seen.add(ident)
        yield name, param


def classify_dewo_v5_parameter(name: str) -> str:
    if is_uncond_adapter_parameter_name(name):
        return GROUP_ADAPTER
    return GROUP_FROZEN


def apply_dewo_v5_uncond_adapter_mode(model: nn.Module) -> dict[str, int]:
    """Freeze 本体; train the CFG adapter; leave the adapter **on** for BC."""

    if not getattr(model, "uncond_adapter_injected", False):
        raise ValueError(
            "DEWO v5 train mode requires `model.uncond_adapter.enabled=true` "
            "so the CFG adapter is injected before freeze."
        )

    model.eval()
    model.requires_grad_(False)
    dit = getattr(model, "dit", None)
    if dit is None:
        raise ValueError("DEWO v5 train mode requires `model.dit`.")
    dit.train()

    counts = {GROUP_FROZEN: 0, GROUP_ADAPTER: 0}
    for name, param in _unique_named_parameters(model):
        group = classify_dewo_v5_parameter(name)
        counts[group] += param.numel()
        param.requires_grad = group == GROUP_ADAPTER

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
        "DEWO v5 CFG-adapter: trainable=%.2fM frozen=%.2fM (adapter on)",
        counts[GROUP_ADAPTER] / 1e6,
        counts[GROUP_FROZEN] / 1e6,
    )
    if counts[GROUP_ADAPTER] < 1:
        raise ValueError(
            "DEWO v5 train mode produced an empty adapter group: "
            f"adapter={counts[GROUP_ADAPTER]} frozen={counts[GROUP_FROZEN]}"
        )
    return counts


def collect_dewo_v5_param_groups(
    model: nn.Module,
    *,
    lr: float,
    weight_decay: float,
) -> list[dict]:
    train_params = [param for param in iter_uncond_adapter_parameters(model) if param.requires_grad]
    if not train_params:
        raise ValueError("DEWO v5 optimizer found no trainable adapter parameters.")
    named_trainable = [
        name
        for name, param in _unique_named_parameters(model)
        if param.requires_grad
    ]
    bad = [name for name in named_trainable if classify_dewo_v5_parameter(name) != GROUP_ADAPTER]
    if bad:
        raise ValueError(
            "Trainable non-adapter parameters in DEWO v5: "
            f"{bad[:8]}{'...' if len(bad) > 8 else ''}"
        )
    logger.info("DEWO v5 AdamW: n=%d lr=%.2e (CFG adapter only)", len(train_params), lr)
    return [
        {
            "name": GROUP_ADAPTER,
            "params": train_params,
            "lr": float(lr),
            "weight_decay": float(weight_decay),
        }
    ]
