"""Recoverability value head for DEWO v8 / v9.

v8 pools the current VAE frame. v9 pools frozen VideoDiT tokens of the
current observation (adapter off, S0 base text). Training fits \(V\) to a
progress return \(G_t=\gamma^{T-t}\) on success and \(G=0\) on the fail
cliff. Do not train \(V\) toward a floor near the event.

Do not gate CFG with ``σ(V − τ)``: that fires on high-V D0 from replan 0.
v8 keeps a once-only **drop**: ``V_{t-1} − V_t > δ`` (optional ``v_high`` floor).
v9's progress \(V\) rarely drops 0.15 across a 24-frame replan, so infer may
instead use **relative growth** (``cfg_gate_mode=value_growth``): from the
2nd/3rd replan, fire CFG when \((V_t-V_{t-1})/|V_{t-1}| < \tau\). Firing on a
would-be success is allowed; keep ``text_cfg_scale`` small (e.g. 1.1) so the
mix is a nudge. Per-replan, not once-fire. Not multi-sample ranking: \(V\) is
a function of the current frame only.
"""

from __future__ import annotations

from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from fastwam.utils.logging_config import get_logger

logger = get_logger(__name__)

DEFAULT_VALUE_HEAD_IN_CHANNELS = 48
DEFAULT_VALUE_DIT_IN_CHANNELS = 3072
DEFAULT_VALUE_HEAD_HIDDEN = 256
DEFAULT_VALUE_V_HIGH = 0.5
DEFAULT_VALUE_DROP_DELTA = 0.15
DEFAULT_VALUE_GROWTH_TAU = 0.05
DEFAULT_VALUE_GROWTH_START_REPLAN = 2
DEFAULT_VALUE_GROWTH_EPS = 1e-4
DEFAULT_VALUE_CLIFF_MARGIN = 0.2
DEFAULT_VALUE_GAMMA = 0.99
VALUE_ENCODER_VAE = "vae_latents"
VALUE_ENCODER_DIT = "video_dit"
VALUE_LOSS_BCE = "bce"
VALUE_LOSS_HUBER = "huber"


def is_value_head_parameter_name(name: str) -> bool:
    return name == "value_head" or name.startswith("value_head.")


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"", "null", "none"}:
        return None
    return float(value)


def progress_return(
    t: int,
    horizon: int,
    *,
    gamma: float = DEFAULT_VALUE_GAMMA,
    failed: bool = False,
) -> float:
    """Monte Carlo progress return. Fail frames are 0; do not floor \(G\)."""

    if failed:
        return 0.0
    t_horizon = max(int(horizon), 1)
    idx = min(max(int(t), 0), t_horizon - 1)
    return float(gamma ** (t_horizon - 1 - idx))


def drop_edge_gate(
    v_prev: float | None,
    v_curr: float,
    *,
    v_high: float | None = None,
    delta: float = DEFAULT_VALUE_DROP_DELTA,
    fired: bool = False,
) -> float:
    """Return ``g ∈ {0, 1}``. High \(V\) alone must not open the gate.

    ``v_high is None`` (v9): fire on the drop only.
    ``v_high`` set (v8): also require ``V_{t-1} >= v_high``.
    """

    if fired:
        return 0.0
    if v_prev is None:
        return 0.0
    prev = float(v_prev)
    curr = float(v_curr)
    if (prev - curr) <= float(delta):
        return 0.0
    if v_high is not None and prev < float(v_high):
        return 0.0
    return 1.0


def relative_growth(
    v_prev: float | None,
    v_curr: float,
    *,
    eps: float = DEFAULT_VALUE_GROWTH_EPS,
) -> float | None:
    """``(V_t - V_{t-1}) / max(|V_{t-1}|, eps)``. ``None`` if there is no previous V."""

    if v_prev is None:
        return None
    prev = float(v_prev)
    curr = float(v_curr)
    return (curr - prev) / max(abs(prev), float(eps))


def relative_growth_gate(
    v_prev: float | None,
    v_curr: float,
    *,
    tau: float = DEFAULT_VALUE_GROWTH_TAU,
    replan_index: int = 0,
    start_replan: int = DEFAULT_VALUE_GROWTH_START_REPLAN,
    eps: float = DEFAULT_VALUE_GROWTH_EPS,
) -> float:
    """Return ``g ∈ {0, 1}``. ``g=1`` means apply CFG this replan.

    Skip until ``replan_index >= start_replan`` (0-based; default 2 = third
    node at ``t=48`` when ``replan_steps=24``). Fire when relative growth is
    below ``tau``. Not once-fire and not drop-edge. A dip on a success
    trajectory may fire; that is intended.
    """

    if int(replan_index) < int(start_replan):
        return 0.0
    rel = relative_growth(v_prev, v_curr, eps=eps)
    if rel is None:
        return 0.0
    return 1.0 if rel < float(tau) else 0.0


def recoverability_cliff_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    pair_ids: Sequence[Any],
    *,
    margin: float = DEFAULT_VALUE_CLIFF_MARGIN,
) -> torch.Tensor:
    """``ReLU(δ − (V(s_t) − V(s_M)))`` when a pair has both labels in-batch.

    Always returns a tensor connected to ``pred`` so every rank has the same
    value-head graph (DeepSpeed unused-parameter NaNs otherwise).
    v9 keeps ``lambda_cliff=0``: the progress return already defines the
    cliff, and a ranking margin would inflate \(V\) near the event.
    """

    if pred.ndim != 1:
        pred = pred.view(-1)
    if target.ndim != 1:
        target = target.view(-1)
    if hasattr(pair_ids, "tolist") and not isinstance(pair_ids, (str, bytes)):
        pair_ids = list(pair_ids.tolist())
    else:
        pair_ids = list(pair_ids)
    if pred.shape[0] != target.shape[0] or pred.shape[0] != len(pair_ids):
        raise ValueError(
            "cliff loss requires pred/target/pair_ids of equal length, got "
            f"{tuple(pred.shape)}, {tuple(target.shape)}, {len(pair_ids)}"
        )
    pred = pred.float()
    target = target.float()
    groups: dict[str, list[int]] = {}
    for index, pair_id in enumerate(pair_ids):
        key = str(pair_id or "").strip()
        if not key:
            continue
        groups.setdefault(key, []).append(index)
    terms: list[torch.Tensor] = []
    margin_t = pred.new_tensor(float(margin))
    for idxs in groups.values():
        idx_t = torch.tensor(idxs, device=pred.device, dtype=torch.long)
        grouped_pred = pred[idx_t]
        grouped_target = target[idx_t]
        plus = grouped_pred[grouped_target >= 0.5]
        fail = grouped_pred[grouped_target < 0.5]
        if plus.numel() == 0 or fail.numel() == 0:
            continue
        terms.append(F.relu(margin_t - (plus.mean() - fail.mean())))
    if not terms:
        return pred.sum() * 0.0
    return torch.stack(terms).mean()


class RecoverabilityValueHead(nn.Module):
    """MLP on pooled current-frame features → ``σ(V) ∈ (0, 1)``."""

    def __init__(
        self,
        in_channels: int = DEFAULT_VALUE_HEAD_IN_CHANNELS,
        hidden: int = DEFAULT_VALUE_HEAD_HIDDEN,
    ) -> None:
        super().__init__()
        if in_channels < 1:
            raise ValueError(f"`in_channels` must be >= 1, got {in_channels}")
        if hidden < 1:
            raise ValueError(f"`hidden` must be >= 1, got {hidden}")
        self.in_channels = int(in_channels)
        self.hidden = int(hidden)
        self.mlp = nn.Sequential(
            nn.Linear(self.in_channels, self.hidden),
            nn.GELU(),
            nn.Linear(self.hidden, self.hidden),
            nn.GELU(),
            nn.Linear(self.hidden, 1),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    @staticmethod
    def pool_current_frame(latents: torch.Tensor) -> torch.Tensor:
        """``[B, C]`` from ``[B, C, T, H, W]`` or ``[B, C, H, W]`` (current frame)."""

        if latents.ndim == 5:
            current = latents[:, :, 0]
        elif latents.ndim == 4:
            current = latents
        else:
            raise ValueError(
                "`latents` must be [B,C,T,H,W] or [B,C,H,W], "
                f"got {tuple(latents.shape)}"
            )
        return current.float().mean(dim=(-2, -1))

    @staticmethod
    def pool_tokens(tokens: torch.Tensor) -> torch.Tensor:
        """``[B, D]`` from VideoDiT tokens ``[B, S, D]``."""

        if tokens.ndim != 3:
            raise ValueError(
                "`tokens` must be [B,S,D], "
                f"got {tuple(tokens.shape)}"
            )
        return tokens.float().mean(dim=1)

    @staticmethod
    def _linear_fp32(module: nn.Linear, inputs: torch.Tensor) -> torch.Tensor:
        return F.linear(
            inputs.float(),
            module.weight.float(),
            None if module.bias is None else module.bias.float(),
        )

    def logits(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim in {4, 5}:
            pooled = self.pool_current_frame(features)
        elif features.ndim == 3:
            pooled = self.pool_tokens(features)
        elif features.ndim == 2:
            pooled = features.float()
        else:
            raise ValueError(
                "value head expects [B,C,T,H,W], [B,C,H,W], [B,S,D], or [B,D], "
                f"got {tuple(features.shape)}"
            )
        if pooled.shape[-1] != self.in_channels:
            raise ValueError(
                f"value head expects C={self.in_channels}, got {tuple(pooled.shape)}"
            )
        # DeepSpeed/bf16 may recast this module; compute the critic in fp32.
        hidden_features = F.layer_norm(
            pooled.float(), (self.in_channels,), None, None, 1e-5
        )
        hidden = F.gelu(self._linear_fp32(self.mlp[0], hidden_features))
        hidden = F.gelu(self._linear_fp32(self.mlp[2], hidden))
        logit = self._linear_fp32(self.mlp[4], hidden).squeeze(-1)
        return logit.float().clamp(-16.0, 16.0)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.logits(features))


def normalize_value_head_config(value_head: Any, *, recipe: str) -> dict[str, Any]:
    if value_head is None:
        value_head = {}
    if not isinstance(value_head, dict):
        raise TypeError(f"`value_head` must be a dict or null, got {type(value_head)}")
    recipe_s = str(recipe or "").strip()
    is_v8 = recipe_s == "v8"
    is_v9 = recipe_s == "v9"
    enabled = bool(value_head.get("enabled", is_v8 or is_v9))
    encoder = str(
        value_head.get("encoder")
        or (VALUE_ENCODER_DIT if is_v9 else VALUE_ENCODER_VAE)
    ).strip()
    if encoder not in {VALUE_ENCODER_VAE, VALUE_ENCODER_DIT}:
        raise ValueError(
            f"`value_head.encoder` must be {VALUE_ENCODER_VAE!r} or "
            f"{VALUE_ENCODER_DIT!r}, got {encoder!r}"
        )
    default_c = (
        DEFAULT_VALUE_DIT_IN_CHANNELS
        if encoder == VALUE_ENCODER_DIT
        else DEFAULT_VALUE_HEAD_IN_CHANNELS
    )
    in_channels = int(value_head.get("in_channels", default_c))
    hidden = int(value_head.get("hidden", DEFAULT_VALUE_HEAD_HIDDEN))
    lambda_value = float(value_head.get("lambda_value", 1.0) or 0.0)
    lambda_cliff = float(
        value_head.get("lambda_cliff", 0.0 if is_v9 else 0.5) or 0.0
    )
    cliff_margin = float(
        value_head.get("cliff_margin", DEFAULT_VALUE_CLIFF_MARGIN) or 0.0
    )
    if "v_high" in value_head:
        v_high = _optional_float(value_head.get("v_high"))
    else:
        v_high = None if is_v9 else DEFAULT_VALUE_V_HIGH
    drop_delta = float(value_head.get("drop_delta", DEFAULT_VALUE_DROP_DELTA))
    gamma = float(value_head.get("gamma", DEFAULT_VALUE_GAMMA))
    loss = str(
        value_head.get("loss") or (VALUE_LOSS_HUBER if is_v9 else VALUE_LOSS_BCE)
    ).strip().lower()
    if loss not in {VALUE_LOSS_BCE, VALUE_LOSS_HUBER}:
        raise ValueError(
            f"`value_head.loss` must be {VALUE_LOSS_BCE!r} or "
            f"{VALUE_LOSS_HUBER!r}, got {loss!r}"
        )
    if in_channels < 1 or hidden < 1:
        raise ValueError("value_head in_channels/hidden must be >= 1")
    if lambda_value < 0.0 or lambda_cliff < 0.0 or cliff_margin < 0.0:
        raise ValueError("value_head lambdas/margin must be >= 0")
    if not (0.0 < gamma <= 1.0):
        raise ValueError(f"`value_head.gamma` must be in (0, 1], got {gamma}")
    if drop_delta < 0.0:
        raise ValueError("`value_head.drop_delta` must be >= 0")
    return {
        "enabled": enabled,
        "encoder": encoder,
        "in_channels": in_channels,
        "hidden": hidden,
        "lambda_value": lambda_value,
        "lambda_cliff": lambda_cliff,
        "cliff_margin": cliff_margin,
        "v_high": v_high,
        "drop_delta": drop_delta,
        "gamma": gamma,
        "loss": loss,
    }


def attach_recoverability_value_head(
    model: nn.Module,
    value_head_cfg: dict[str, Any] | None = None,
    *,
    recipe: str = "v8",
) -> RecoverabilityValueHead:
    cfg = normalize_value_head_config(value_head_cfg, recipe=recipe)
    if not cfg["enabled"]:
        raise ValueError("attach_recoverability_value_head requires value_head.enabled")
    in_channels = int(cfg["in_channels"])
    if cfg["encoder"] == VALUE_ENCODER_DIT:
        expert = getattr(model, "video_expert", None)
        hidden_dim = getattr(expert, "hidden_dim", None)
        if hidden_dim is not None:
            in_channels = int(hidden_dim)
    existing = getattr(model, "value_head", None)
    if isinstance(existing, RecoverabilityValueHead):
        head = existing
        if int(head.in_channels) != in_channels:
            raise ValueError(
                "Existing value_head in_channels="
                f"{head.in_channels} disagrees with encoder dim {in_channels}"
            )
    else:
        device = getattr(model, "device", torch.device("cpu"))
        head = RecoverabilityValueHead(
            in_channels=in_channels,
            hidden=int(cfg["hidden"]),
        )
        # Keep the critic in fp32; bf16 sigmoid-BCE saturates to 0/1 then inf.
        head = head.to(device=device, dtype=torch.float32)
        model.value_head = head
    model.value_head_lambda = float(cfg["lambda_value"])
    model.value_cliff_lambda = float(cfg["lambda_cliff"])
    model.value_cliff_margin = float(cfg["cliff_margin"])
    model.value_v_high = cfg["v_high"]
    model.value_drop_delta = float(cfg["drop_delta"])
    model.value_head_encoder = str(cfg["encoder"])
    model.value_head_loss = str(cfg["loss"])
    model.value_gamma = float(cfg["gamma"])
    logger.info(
        "Attached recoverability value head: encoder=%s C=%d hidden=%d "
        "loss=%s γ=%.4f λ_V=%.3f λ_cliff=%.3f v_high=%s δ=%.3f",
        cfg["encoder"],
        in_channels,
        cfg["hidden"],
        cfg["loss"],
        cfg["gamma"],
        cfg["lambda_value"],
        cfg["lambda_cliff"],
        cfg["v_high"],
        cfg["drop_delta"],
    )
    return head


def export_value_head_state_dict(model: nn.Module) -> dict[str, torch.Tensor] | None:
    head = getattr(model, "value_head", None)
    if head is None:
        return None
    return {key: value.detach().cpu() for key, value in head.state_dict().items()}


def load_value_head_state_dict(model: nn.Module, state: dict[str, Any]) -> None:
    head = getattr(model, "value_head", None)
    if head is None:
        raise ValueError("Cannot load value_head weights: model has no value_head.")
    tensors = {
        key: value
        for key, value in state.items()
        if isinstance(value, torch.Tensor)
    }
    if not tensors:
        raise ValueError("value_head checkpoint is empty.")
    head.load_state_dict(tensors, strict=True)
    logger.info("Loaded value head tensors=%d", len(tensors))
