"""Frozen-backbone CFG residual for DEWO v5.

Frozen uncond + learned residual, not Ho joint CFG training. Action noise
is the guided target; video is first-frame context only. Video text stays
on the S0 **base** prompt so the observation encoding is not rewritten by
an OOD success suffix. A rank-r residual on text-side K/V can still
nudge that encoding, but only via action loss plus an identity penalty
on the video residual. Historical parameter names stay `uncond_down` /
`uncond_up`.

CFG inference branches:

- 本体 / base: adapter **off**, video **and** action on base prompt
- CFG / posi: adapter **on**, video still base, action on success prompt
- mix (only when two branches run): ``ε_cfg = ε_base + w · (ε_posi − ε_base)``

Do not confuse these two "ones":

- ``text_cfg_scale=1`` (eval/CLI): **本体 bypass**. Skip the mix, remap onto
  ``cfg_base_prompt``, adapter off. This is the on-distribution S0 policy.
- mix weight ``w=1``: ``ε_posi`` (adapter **on** + success text). **Not** 本体.
- mix weight ``w=0``: ``ε_base``. Same policy as the ``text_cfg_scale=1`` bypass
  once both branches have already been computed.
- mix weight ``w=2`` (default guided): ``ε_base + 2(ε_posi − ε_base)``.
"""

from __future__ import annotations

import json
import math
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from fastwam.utils.logging_config import get_logger

logger = get_logger(__name__)

CHECKPOINT_FORMAT_UNCOND_ADAPTER_V1 = "dewo_v5_uncond_adapter_v1"

DEFAULT_UNCOND_ADAPTER_TARGET_MODULES: tuple[str, ...] = (
    "cross_attn.k",
    "cross_attn.v",
)

DEFAULT_UNCOND_ADAPTER_EXPERTS: tuple[str, ...] = ("video", "action")

# ``epsilon_l`` is an inference-time trust-region radius for the action CFG
# residual.  Keep the names short because they are also used by CLI/eval
# metadata.  ``rms`` preserves the residual direction while bounding each
# action token; ``elementwise`` is useful when individual action coordinates
# have a known noise scale.
CFG_RESIDUAL_CLIP_MODE_RMS = "rms"
CFG_RESIDUAL_CLIP_MODE_ELEMENTWISE = "elementwise"
CFG_RESIDUAL_CLIP_MODES: tuple[str, ...] = (
    CFG_RESIDUAL_CLIP_MODE_RMS,
    CFG_RESIDUAL_CLIP_MODE_ELEMENTWISE,
)


class UncondAdapterGate:
    """Shared on/off switch for every injected linear. Not an nn.Module."""

    def __init__(self, enabled: bool = False) -> None:
        self.enabled = bool(enabled)


class LinearWithUncondAdapter(nn.Linear):
    """`nn.Linear` plus a residual adapter that can be disabled per forward.

    Backbone `weight` / `bias` keep the original parameter objects so a frozen
    本体 checkpoint still loads into the same keys.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        *,
        rank: int,
        alpha: float,
        gate: UncondAdapterGate,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(
            in_features,
            out_features,
            bias=bias,
            device=device,
            dtype=dtype,
        )
        if rank < 1:
            raise ValueError(f"`rank` must be >= 1, got {rank}")
        self.uncond_rank = int(rank)
        self.uncond_alpha = float(alpha)
        self.uncond_down = nn.Parameter(
            torch.empty(rank, in_features, device=device, dtype=dtype)
        )
        self.uncond_up = nn.Parameter(
            torch.empty(out_features, rank, device=device, dtype=dtype)
        )
        self._uncond_gate = gate
        self._last_delta_mse: torch.Tensor | None = None
        self.reset_uncond_parameters()

    def reset_uncond_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.uncond_down, a=math.sqrt(5))
        nn.init.zeros_(self.uncond_up)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out = F.linear(input, self.weight, self.bias)
        if not self._uncond_gate.enabled:
            self._last_delta_mse = None
            return out
        scale = self.uncond_alpha / float(self.uncond_rank)
        delta = F.linear(F.linear(input, self.uncond_down), self.uncond_up)
        scaled = scale * delta
        if self.training:
            reduce_dims = tuple(range(1, scaled.ndim))
            if reduce_dims:
                self._last_delta_mse = scaled.float().square().mean(dim=reduce_dims)
            else:
                self._last_delta_mse = scaled.float().square()
        else:
            self._last_delta_mse = None
        return out + scaled

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        *,
        rank: int,
        alpha: float,
        gate: UncondAdapterGate,
    ) -> LinearWithUncondAdapter:
        if isinstance(linear, cls):
            linear._uncond_gate = gate
            return linear
        adapted = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            rank=rank,
            alpha=alpha,
            gate=gate,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
        )
        adapted.weight = linear.weight
        adapted.bias = linear.bias
        return adapted


def _parse_target_modules(value: Any) -> list[str]:
    if value is None:
        return list(DEFAULT_UNCOND_ADAPTER_TARGET_MODULES)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return list(DEFAULT_UNCOND_ADAPTER_TARGET_MODULES)
        return [part.strip() for part in stripped.split(",") if part.strip()]
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    raise TypeError(f"`target_modules` must be str or list[str], got {type(value)}")


def _parse_experts(value: Any) -> list[str]:
    if value is None:
        return list(DEFAULT_UNCOND_ADAPTER_EXPERTS)
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
        return parts or list(DEFAULT_UNCOND_ADAPTER_EXPERTS)
    if isinstance(value, (list, tuple)):
        parts = [str(item).strip() for item in value if str(item).strip()]
        return parts or list(DEFAULT_UNCOND_ADAPTER_EXPERTS)
    raise TypeError(f"`experts` must be str or list[str], got {type(value)}")


def normalize_uncond_adapter_config(uncond_adapter: Any) -> dict[str, Any]:
    if uncond_adapter is None:
        return {"enabled": False}
    if not isinstance(uncond_adapter, dict):
        raise TypeError(
            f"`uncond_adapter` must be a dict or null, got {type(uncond_adapter)}"
        )
    enabled = bool(uncond_adapter.get("enabled", False))
    rank = int(uncond_adapter.get("rank", 16))
    if rank < 1:
        raise ValueError(f"`uncond_adapter.rank` must be >= 1, got {rank}")
    alpha_raw = uncond_adapter.get("alpha")
    alpha = float(rank if alpha_raw is None else alpha_raw)
    checkpoint = uncond_adapter.get("checkpoint")
    if checkpoint is not None:
        checkpoint = str(checkpoint).strip() or None
    source_checkpoint = uncond_adapter.get("source_checkpoint")
    if source_checkpoint is not None:
        source_checkpoint = str(source_checkpoint).strip() or None
    identity_lock = float(uncond_adapter.get("identity_lock_lambda", 0.0) or 0.0)
    if not math.isfinite(identity_lock) or identity_lock < 0.0:
        raise ValueError(
            f"`uncond_adapter.identity_lock_lambda` must be finite and >= 0, got {identity_lock}."
        )
    action_residual_lock = float(
        uncond_adapter.get("action_residual_lock_lambda", 0.0) or 0.0
    )
    if not math.isfinite(action_residual_lock) or action_residual_lock < 0.0:
        raise ValueError(
            "`uncond_adapter.action_residual_lock_lambda` must be finite and >= 0, "
            f"got {action_residual_lock}."
        )
    recipe = str(uncond_adapter.get("recipe") or "v5").strip() or "v5"
    from fastwam.models.wan22.value_head import normalize_value_head_config

    return {
        "enabled": enabled,
        "rank": rank,
        "alpha": alpha,
        "target_modules": _parse_target_modules(uncond_adapter.get("target_modules")),
        "experts": _parse_experts(uncond_adapter.get("experts")),
        "checkpoint": checkpoint,
        "source_checkpoint": source_checkpoint,
        "pin_video_context_to_base": bool(
            uncond_adapter.get("pin_video_context_to_base", False)
        ),
        "identity_lock_lambda": identity_lock,
        "action_residual_lock_lambda": action_residual_lock,
        "video_bc_on_zero_action": bool(
            uncond_adapter.get("video_bc_on_zero_action", False)
        ),
        "recipe": recipe,
        "value_head": normalize_value_head_config(
            uncond_adapter.get("value_head"), recipe=recipe
        ),
    }


def is_uncond_adapter_parameter_name(name: str) -> bool:
    lowered = name.replace(".", "_")
    return lowered.endswith("uncond_down") or lowered.endswith("uncond_up")


def is_uncond_adapter_checkpoint(payload: Any) -> bool:
    return (
        isinstance(payload, dict)
        and str(payload.get("format", "")) == CHECKPOINT_FORMAT_UNCOND_ADAPTER_V1
    )


def _matches_target(qualified_name: str, targets: Sequence[str]) -> bool:
    for target in targets:
        if not target:
            continue
        if qualified_name == target or qualified_name.endswith("." + target):
            return True
    return False


def _replace_submodule(root: nn.Module, qualified_name: str, new: nn.Module) -> None:
    parts = qualified_name.split(".")
    parent: Any = root
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    last = parts[-1]
    if last.isdigit():
        parent[int(last)] = new
    else:
        setattr(parent, last, new)


def inject_uncond_adapter_into_module(
    module: nn.Module,
    *,
    rank: int,
    alpha: float,
    target_modules: Sequence[str],
    gate: UncondAdapterGate,
) -> int:
    """Replace matching `nn.Linear` children. Returns the number of wraps."""

    replaced = 0
    for name, child in list(module.named_modules()):
        if not name or not isinstance(child, nn.Linear):
            continue
        if isinstance(child, LinearWithUncondAdapter):
            child._uncond_gate = gate
            continue
        if not _matches_target(name, target_modules):
            continue
        _replace_submodule(
            module,
            name,
            LinearWithUncondAdapter.from_linear(
                child, rank=rank, alpha=alpha, gate=gate
            ),
        )
        replaced += 1
    return replaced


def _expert_module(model: nn.Module, expert_name: str) -> nn.Module:
    mot = getattr(model, "mot", None) or getattr(model, "dit", None)
    if mot is not None:
        mixtures = getattr(mot, "mixtures", None)
        if mixtures is not None and expert_name in mixtures:
            return mixtures[expert_name]
    attr = f"{expert_name}_expert"
    expert = getattr(model, attr, None)
    if expert is None:
        raise ValueError(f"Cannot find expert {expert_name!r} on model.")
    return expert


def inject_uncond_adapter(
    model: nn.Module,
    *,
    rank: int,
    alpha: float,
    target_modules: Sequence[str] | None = None,
    experts: Sequence[str] | None = None,
    checkpoint: str | None = None,
) -> UncondAdapterGate:
    targets = list(target_modules or DEFAULT_UNCOND_ADAPTER_TARGET_MODULES)
    expert_names = list(experts or DEFAULT_UNCOND_ADAPTER_EXPERTS)
    gate = UncondAdapterGate(enabled=False)
    n_wrapped = 0
    for expert_name in expert_names:
        n_wrapped += inject_uncond_adapter_into_module(
            _expert_module(model, expert_name),
            rank=rank,
            alpha=alpha,
            target_modules=targets,
            gate=gate,
        )
    if n_wrapped < 1:
        raise ValueError(
            "Uncond adapter injection matched no Linear modules. "
            f"experts={expert_names} targets={targets}"
        )
    model.uncond_adapter_gate = gate
    model.uncond_adapter_injected = True
    model.uncond_adapter_config = {
        "rank": int(rank),
        "alpha": float(alpha),
        "target_modules": targets,
        "experts": expert_names,
    }
    n_train = sum(p.numel() for p in iter_uncond_adapter_parameters(model))
    logger.info(
        "Injected uncond adapter: rank=%d alpha=%s wraps=%d params=%.2fM experts=%s",
        rank,
        alpha,
        n_wrapped,
        n_train / 1e6,
        expert_names,
    )
    if checkpoint:
        load_uncond_adapter_state_dict(model, checkpoint)
    return gate


def iter_uncond_adapter_parameters(model: nn.Module) -> Iterator[nn.Parameter]:
    seen: set[int] = set()
    for name, param in model.named_parameters():
        if not is_uncond_adapter_parameter_name(name):
            continue
        ident = id(param)
        if ident in seen:
            continue
        seen.add(ident)
        yield param


def export_uncond_adapter_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    mot = getattr(model, "mot", None) or getattr(model, "dit", None) or model
    return {
        key: value.detach().cpu()
        for key, value in mot.state_dict().items()
        if is_uncond_adapter_parameter_name(key)
    }


def uncond_adapter_payload(
    model: nn.Module,
    *,
    step: int | None = None,
    source_checkpoint: str | None = None,
) -> dict[str, Any]:
    cfg = dict(getattr(model, "uncond_adapter_config", {}) or {})
    state = export_uncond_adapter_state_dict(model)
    if not state:
        raise ValueError("No uncond adapter parameters to export.")
    payload = {
        "format": CHECKPOINT_FORMAT_UNCOND_ADAPTER_V1,
        "step": step,
        "rank": int(cfg.get("rank", 0)),
        "alpha": float(cfg.get("alpha", 0.0)),
        "target_modules": list(cfg.get("target_modules", [])),
        "experts": list(cfg.get("experts", [])),
        "source_checkpoint": source_checkpoint,
        "recipe": str(cfg.get("recipe") or "v5"),
        "uncond_adapter": state,
        "n_tensors": len(state),
        "n_params": int(sum(t.numel() for t in state.values())),
    }
    from fastwam.models.wan22.value_head import export_value_head_state_dict

    value_state = export_value_head_state_dict(model)
    if value_state:
        payload["value_head"] = value_state
        payload["n_value_head_tensors"] = len(value_state)
        payload["n_value_head_params"] = int(
            sum(t.numel() for t in value_state.values())
        )
    return payload


def _adapter_state_from_payload(payload: dict[str, Any]) -> dict[str, torch.Tensor]:
    if is_uncond_adapter_checkpoint(payload):
        state = payload.get("uncond_adapter")
        if not isinstance(state, dict) or not state:
            raise ValueError("Adapter checkpoint is missing `uncond_adapter` weights.")
        return state
    if "uncond_adapter" in payload and isinstance(payload["uncond_adapter"], dict):
        return payload["uncond_adapter"]
    # Bare state dict of adapter tensors.
    adapter_only = {
        key: value
        for key, value in payload.items()
        if isinstance(value, torch.Tensor) and is_uncond_adapter_parameter_name(str(key))
    }
    if adapter_only:
        return adapter_only
    raise ValueError(
        "Not a DEWO v5 uncond-adapter checkpoint. Expected format "
        f"{CHECKPOINT_FORMAT_UNCOND_ADAPTER_V1!r}."
    )


def load_uncond_adapter_state_dict(
    model: nn.Module,
    state_dict_or_path: dict[str, Any] | str,
) -> None:
    if not getattr(model, "uncond_adapter_injected", False):
        raise ValueError(
            "load_uncond_adapter_state_dict() requires inject_uncond_adapter() first."
        )
    if isinstance(state_dict_or_path, str):
        payload = torch.load(state_dict_or_path, map_location="cpu")
        if not isinstance(payload, dict):
            raise ValueError(f"Unsupported adapter checkpoint type: {type(payload)}")
        state = _adapter_state_from_payload(payload)
        src = state_dict_or_path
    else:
        payload = state_dict_or_path
        state = (
            _adapter_state_from_payload(payload)
            if not all(is_uncond_adapter_parameter_name(k) for k in payload)
            else payload
        )
        src = "<dict>"

    mot = getattr(model, "mot", None) or getattr(model, "dit", None) or model
    incompatible = mot.load_state_dict(state, strict=False)
    adapter_missing = [
        key
        for key in incompatible.missing_keys
        if is_uncond_adapter_parameter_name(key)
    ]
    if adapter_missing:
        raise ValueError(
            "Uncond adapter checkpoint is missing tensors: "
            f"{adapter_missing[:8]}{'...' if len(adapter_missing) > 8 else ''}"
        )
    logger.info(
        "Loaded uncond adapter from %s: keys=%d unexpected=%d",
        src,
        len(state),
        len(incompatible.unexpected_keys),
    )
    value_state = payload.get("value_head") if isinstance(payload, dict) else None
    if isinstance(value_state, dict) and value_state:
        if getattr(model, "value_head", None) is None:
            logger.warning(
                "Adapter checkpoint has value_head weights but the model has "
                "no value_head module; skipping."
            )
        else:
            from fastwam.models.wan22.value_head import load_value_head_state_dict

            load_value_head_state_dict(model, value_state)


def peek_checkpoint_payload(path: str) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint is not a dict: {type(payload)}")
    return payload


def resolve_backbone_and_adapter_paths(
    checkpoint: str,
    *,
    backbone_checkpoint: str | None = None,
    adapter_checkpoint: str | None = None,
    config_resume: str | None = None,
) -> tuple[str, str]:
    """Pair a frozen base ckpt with a DEWO v5 adapter file.

    If `adapter_checkpoint` is set, `checkpoint` is treated as the backbone
    unless `backbone_checkpoint` is also set. Otherwise the checkpoint is
    peeked: adapter-format files look up the backbone from `backbone_checkpoint`,
    the payload `source_checkpoint`, or `config_resume`.
    """

    if adapter_checkpoint:
        backbone = backbone_checkpoint or checkpoint
        if not backbone:
            raise ValueError("Missing frozen backbone checkpoint for DEWO v5.")
        return str(backbone), str(adapter_checkpoint)

    payload = peek_checkpoint_payload(checkpoint)
    if is_uncond_adapter_checkpoint(payload):
        source = payload.get("source_checkpoint")
        source_text = str(source).strip() if source not in (None, "") else ""
        backbone = backbone_checkpoint or source_text or config_resume
        if not backbone:
            raise ValueError(
                f"{checkpoint} is a DEWO v5 uncond adapter. Pass "
                "--backbone-checkpoint or keep config.resume pointing at the frozen base."
            )
        return str(backbone), str(checkpoint)

    if backbone_checkpoint:
        return str(backbone_checkpoint), str(checkpoint)
    raise ValueError(
        "DEWO v5 requires an uncond-adapter checkpoint. Pass --uncond-adapter "
        "or an adapter-format --checkpoint."
    )


def uncond_adapter_residual_mse(
    model: nn.Module,
    *,
    expert: str = "video",
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """Mean squared adapter delta on one expert, if any wrap recorded it.

    Each wrap stores a per-sample vector ``[B]`` (mean over non-batch dims) or
    a scalar. When ``sample_weight`` is set, the batch reduction is a weighted
    mean over samples (DEWO v6: identity / action lock on D+ only).
    """

    squares: list[torch.Tensor] = []
    try:
        module = _expert_module(model, expert)
    except ValueError:
        return None
    for child in module.modules():
        mse = getattr(child, "_last_delta_mse", None)
        if mse is None:
            continue
        squares.append(mse.float())
    if not squares:
        return None
    per_sample: list[torch.Tensor] = []
    scalars: list[torch.Tensor] = []
    for mse in squares:
        flat = mse.reshape(-1)
        if flat.numel() == 1:
            scalars.append(flat.reshape(()))
        else:
            per_sample.append(flat)
    if per_sample:
        parts = list(per_sample)
        ref = per_sample[0]
        parts.extend(value.expand_as(ref) for value in scalars)
        reduced = torch.stack(parts).mean(dim=0)
        if sample_weight is None:
            return reduced.mean()
        weight = sample_weight.to(device=reduced.device, dtype=reduced.dtype).reshape(-1)
        if weight.shape[0] != reduced.shape[0]:
            raise ValueError(
                "`sample_weight` must match residual batch, got "
                f"{tuple(weight.shape)} vs {tuple(reduced.shape)}."
            )
        return (reduced * weight).sum() / weight.sum().clamp(min=1.0)
    stacked = torch.stack(scalars)
    return stacked.mean()


def pin_video_context_per_sample(
    context: torch.Tensor,
    context_mask: torch.Tensor,
    base_context: torch.Tensor,
    base_mask: torch.Tensor,
    action_loss_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pin rows with action BC (D0 / D+) to base video text; leave D_fail on failure text."""

    pin = action_loss_weight.to(device=context.device).reshape(-1) > 0
    if bool(pin.all()):
        return base_context, base_mask
    if not bool(pin.any()):
        return context, context_mask
    out_context = context.clone()
    out_mask = context_mask.clone()
    out_context[pin] = base_context[pin]
    out_mask[pin] = base_mask[pin]
    return out_context, out_mask


def recommend_adaptive_cfg_tau(
    e_plus: Sequence[float],
    e_zero: Sequence[float] | None = None,
    *,
    recall: float = 0.90,
    max_fpr0: float = 0.05,
) -> dict[str, Any]:
    """Choose the highest tau that still covers ``recall`` of D+ windows.

    ``tau = quantile(E+, 1-recall)``. Default ``recall=0.90`` → ``q_0.10(E+)``.
    ``FPR0 = P(E0 > tau)`` is a check, not a search target. If it exceeds
    ``max_fpr0``, the residual is not event-specific: ``separable=False``.
    """

    if not math.isfinite(float(recall)) or not 0.0 < float(recall) <= 1.0:
        raise ValueError(f"`recall` must be in (0, 1], got {recall}.")
    if not math.isfinite(float(max_fpr0)) or not 0.0 <= float(max_fpr0) <= 1.0:
        raise ValueError(f"`max_fpr0` must be in [0, 1], got {max_fpr0}.")

    plus = [float(value) for value in e_plus]
    zero = [] if e_zero is None else [float(value) for value in e_zero]
    if not plus:
        return {
            "tau": None,
            "recall_target": float(recall),
            "recall_plus": None,
            "fpr0": None,
            "max_fpr0": float(max_fpr0),
            "separable": False,
            "n_plus": 0,
            "n_zero": len(zero),
            "quantile": float(round(1.0 - float(recall), 12)),
            "reason": "empty_e_plus",
        }
    if any(not math.isfinite(value) for value in plus + zero):
        raise ValueError("E+ / E0 values must be finite.")

    # Normalize the derived probability before exposing it in metrics/JSON.
    # This keeps values such as ``1 - 0.90`` stable across Python's binary
    # floating-point representation while preserving the requested precision
    # for the quantile calculation.
    quantile = float(round(1.0 - float(recall), 12))
    plus_t = torch.tensor(plus, dtype=torch.float64)
    tau = float(torch.quantile(plus_t, quantile).item())
    recall_plus = float((plus_t >= tau).to(dtype=torch.float64).mean().item())
    fpr0 = None
    separable = True
    reason = "ok"
    if zero:
        zero_t = torch.tensor(zero, dtype=torch.float64)
        fpr0 = float((zero_t > tau).to(dtype=torch.float64).mean().item())
        if fpr0 > float(max_fpr0):
            separable = False
            reason = "fpr0_above_max"
    return {
        "tau": tau,
        "recall_target": float(recall),
        "recall_plus": recall_plus,
        "fpr0": fpr0,
        "max_fpr0": float(max_fpr0),
        "separable": bool(separable),
        "n_plus": len(plus),
        "n_zero": len(zero),
        "quantile": quantile,
        "reason": reason,
    }


def write_adaptive_cfg_tau_json(
    path: str | Path,
    e_plus: Sequence[float],
    e_zero: Sequence[float] | None = None,
    *,
    recall: float = 0.90,
    max_fpr0: float = 0.05,
    recipe: str = "v6",
) -> dict[str, Any]:
    """Write ``RUN_DIR/adaptive_cfg_tau.json`` from E+ / E0 RMS arrays.

    Does not grid-search tau and does not touch official eval seeds 0–49.
    ``ADAPTIVE_CFG_TAU=auto`` reads ``separable`` and ``tau`` from this file.
    """

    payload = recommend_adaptive_cfg_tau(
        e_plus,
        e_zero,
        recall=recall,
        max_fpr0=max_fpr0,
    )
    payload["recipe"] = str(recipe)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def v5_infer_video_uses_base_context(
    *,
    adapter_injected: bool,
    pin_video_context_to_base: bool,
    has_negative_context: bool,
) -> bool:
    """Video prefill should stay on cfg_base_prompt when the pin is on."""

    return (
        bool(adapter_injected)
        and bool(pin_video_context_to_base)
        and bool(has_negative_context)
    )


def v5_infer_use_adapter(*, branch: str, use_text_cfg: bool) -> bool:
    """Whether the v5 adapter is on for an `infer_action` branch.

    Frozen S0 was trained on base text. The adapter is the CFG-cond residual
    (success / outcome suffix).

    - ``base``: always off (本体)
    - ``posi``: on iff the two-branch mix is active (`text_cfg_scale != 1`).
      When ``text_cfg_scale=1``, the single branch is remapped to 本体, so
      the adapter stays off. Mix weight ``w=1`` is a different object.
    """
    if branch == "base":
        return False
    if branch == "posi":
        return bool(use_text_cfg)
    raise ValueError(f"unknown infer branch {branch!r}; expected 'posi' or 'base'")


def v5_infer_remap_to_base_context(
    *,
    adapter_injected: bool,
    use_text_cfg: bool,
    has_negative_context: bool,
) -> bool:
    """`text_cfg_scale=1` + `cfg_base_prompt` is the 本体 bypass, not mix w=1."""
    return bool(adapter_injected) and (not use_text_cfg) and bool(has_negative_context)


# Mix scalar in ε_base + w · δ. 0 is 本体.
# v5/v6: δ = ε_posi − ε_base, so mix w=1 is ε_posi.
# v7: δ = ε_posi − ε_fail, so mix w=1 is ε_base + (ε_posi − ε_fail), not ε_posi.
CFG_MIX_WEIGHT_BASE = 0.0
CFG_MIX_SUBTRACT_BASE = "base"
CFG_MIX_SUBTRACT_FAIL = "fail"


def cfg_mix_subtract_branch(recipe: str | None) -> str:
    """Which score the guided mix subtracts from ε_posi."""

    if str(recipe or "").strip() == "v7":
        return CFG_MIX_SUBTRACT_FAIL
    return CFG_MIX_SUBTRACT_BASE


def mix_guided_action_epsilon(
    epsilon_base: torch.Tensor,
    epsilon_posi: torch.Tensor,
    *,
    mix_weight: float,
    subtract: str = CFG_MIX_SUBTRACT_BASE,
    epsilon_fail: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(ε_cfg, δ)`` with origin ``ε_base``.

    ``subtract='base'``: δ = ε_posi − ε_base (v5/v6).
    ``subtract='fail'``: δ = ε_posi − ε_fail (v7). Never uses ε_fail as origin.
    """

    mode = str(subtract or CFG_MIX_SUBTRACT_BASE).strip() or CFG_MIX_SUBTRACT_BASE
    if mode == CFG_MIX_SUBTRACT_FAIL:
        if epsilon_fail is None:
            raise ValueError("v7 CFG mix requires `epsilon_fail`.")
        delta = epsilon_posi - epsilon_fail
    elif mode == CFG_MIX_SUBTRACT_BASE:
        delta = epsilon_posi - epsilon_base
    else:
        raise ValueError(
            f"`subtract` must be {CFG_MIX_SUBTRACT_BASE!r} or "
            f"{CFG_MIX_SUBTRACT_FAIL!r}, got {subtract!r}."
        )
    return epsilon_base + float(mix_weight) * delta, delta



def adaptive_cfg_mix_weight(
    *,
    exec_rms: float,
    tau: float,
    guided_scale: float,
) -> float:
    """Per-chunk mix weight from residual energy.

    High ``exec_rms`` keeps the guided mix (``guided_scale``, typically 2.0).
    Low energy falls back to ``CFG_MIX_WEIGHT_BASE`` (本体 / ``ε_base``).

    ``guided_scale`` must be the two-branch mix scale (``text_cfg_scale != 1``).
    Never encode 本体 as mix weight 1.0.
    """
    if not math.isfinite(float(tau)) or float(tau) < 0.0:
        raise ValueError(f"`tau` must be finite and >= 0, got {tau}.")
    scale = float(guided_scale)
    if not math.isfinite(scale) or scale == 1.0:
        raise ValueError(
            "`guided_scale` must be the CFG mix (e.g. 2.0), not 1.0. "
            "text_cfg_scale=1 is the 本体 bypass; low-E fallback is mix weight 0."
        )
    if scale <= 0.0:
        raise ValueError(f"`guided_scale` must be > 0 and != 1, got {guided_scale}.")
    if float(exec_rms) > float(tau):
        return scale
    return float(CFG_MIX_WEIGHT_BASE)


def normalize_cfg_residual_clip_mode(mode: str | None) -> str:
    """Normalize the action-CFG residual clipping mode.

    ``None`` intentionally resolves to token RMS clipping.  The mode is only
    observable when ``epsilon_l`` is set, so leaving it unspecified preserves
    the historical no-clipping path exactly.
    """

    value = CFG_RESIDUAL_CLIP_MODE_RMS if mode is None else str(mode).strip().lower()
    aliases = {
        "norm": CFG_RESIDUAL_CLIP_MODE_RMS,
        "token_norm": CFG_RESIDUAL_CLIP_MODE_RMS,
        "token_rms": CFG_RESIDUAL_CLIP_MODE_RMS,
        "abs": CFG_RESIDUAL_CLIP_MODE_ELEMENTWISE,
        "absolute": CFG_RESIDUAL_CLIP_MODE_ELEMENTWISE,
        "coordinate": CFG_RESIDUAL_CLIP_MODE_ELEMENTWISE,
    }
    value = aliases.get(value, value)
    if value not in CFG_RESIDUAL_CLIP_MODES:
        choices = ", ".join(CFG_RESIDUAL_CLIP_MODES)
        raise ValueError(f"`cfg_residual_clip_mode` must be one of {choices}, got {mode!r}.")
    return value


def bound_cfg_residual(
    delta: torch.Tensor,
    epsilon_l: float | None,
    *,
    mode: str | None = CFG_RESIDUAL_CLIP_MODE_RMS,
) -> torch.Tensor:
    """Apply an ``epsilon_l`` trust-region bound to a CFG residual.

    ``delta`` is normally ``[B, H, D]`` and represents
    ``epsilon_posi - epsilon_base``.  With ``mode='rms'`` each token is
    rescaled only when its feature RMS exceeds ``epsilon_l``; this keeps the
    direction of the learned correction and avoids pushing all coordinates to
    their support limits.  ``mode='elementwise'`` performs a coordinate-wise
    absolute clamp.  ``None`` returns the original tensor object, which makes
    the disabled path bit-for-bit compatible with the pre-threshold code.
    """

    if not isinstance(delta, torch.Tensor) or delta.ndim < 1:
        shape = getattr(delta, "shape", None)
        raise ValueError(f"`delta` must be a tensor with a feature dimension, got {shape!r}")
    if epsilon_l is None:
        return delta
    try:
        epsilon = float(epsilon_l)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"`epsilon_l` must be a finite float >= 0, got {epsilon_l!r}.") from exc
    if not math.isfinite(epsilon) or epsilon < 0.0:
        raise ValueError(f"`epsilon_l` must be a finite float >= 0, got {epsilon_l!r}.")
    clip_mode = normalize_cfg_residual_clip_mode(mode)
    if epsilon == 0.0:
        return torch.zeros_like(delta)
    if clip_mode == CFG_RESIDUAL_CLIP_MODE_ELEMENTWISE:
        return delta.clamp(min=-epsilon, max=epsilon)

    # Compute the norm in fp32 even when the model runs in bf16/fp16.  The
    # scale is cast back before multiplication so the returned dtype/shape
    # exactly match ``delta``.
    token_rms = delta.float().square().mean(dim=-1, keepdim=True).sqrt()
    tiny = torch.finfo(torch.float32).tiny
    scale = (epsilon / token_rms.clamp_min(tiny)).clamp(max=1.0)
    return delta * scale.to(dtype=delta.dtype)


def threshold_cfg_residual(
    delta: torch.Tensor,
    epsilon_l: float | None,
    *,
    mode: str | None = CFG_RESIDUAL_CLIP_MODE_RMS,
) -> torch.Tensor:
    """Backward-compatible/readable alias for :func:`bound_cfg_residual`."""

    return bound_cfg_residual(delta, epsilon_l, mode=mode)


def set_uncond_adapter_enabled(model: nn.Module, enabled: bool) -> None:
    gate = getattr(model, "uncond_adapter_gate", None)
    if gate is None:
        return
    gate.enabled = bool(enabled)


def uncond_adapter_is_enabled(model: nn.Module) -> bool:
    gate = getattr(model, "uncond_adapter_gate", None)
    if gate is None:
        return False
    return bool(gate.enabled)


@contextmanager
def uncond_adapter_enabled(model: nn.Module, enabled: bool):
    gate = getattr(model, "uncond_adapter_gate", None)
    if gate is None:
        yield
        return
    prev = bool(gate.enabled)
    gate.enabled = bool(enabled)
    try:
        yield
    finally:
        gate.enabled = prev
