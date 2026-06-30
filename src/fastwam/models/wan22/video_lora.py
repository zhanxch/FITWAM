"""Video DiT LoRA helpers (DiffSynth-style PEFT adapters on the video expert only)."""

from __future__ import annotations

from typing import Any, Iterable, Optional, Sequence

import torch
import torch.nn as nn

from fastwam.utils.logging_config import get_logger

logger = get_logger(__name__)

CHECKPOINT_FORMAT_VIDEO_LORA_V1 = "video_lora_v1"

DEFAULT_VIDEO_LORA_TARGET_MODULES: tuple[str, ...] = (
    "self_attn.q",
    "self_attn.k",
    "self_attn.v",
    "self_attn.o",
    "ffn.0",
    "ffn.2",
)


def _parse_target_modules(value: Any) -> list[str]:
    if value is None:
        return list(DEFAULT_VIDEO_LORA_TARGET_MODULES)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return list(DEFAULT_VIDEO_LORA_TARGET_MODULES)
        return [part.strip() for part in stripped.split(",") if part.strip()]
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    raise TypeError(f"`target_modules` must be str or list[str], got {type(value)}")


def normalize_video_lora_config(video_lora: Any) -> dict[str, Any]:
    if video_lora is None:
        return {"enabled": False}
    if not isinstance(video_lora, dict):
        raise TypeError(f"`video_lora` must be a dict or null, got {type(video_lora)}")
    enabled = bool(video_lora.get("enabled", False))
    rank = int(video_lora.get("rank", 32))
    if rank < 1:
        raise ValueError(f"`video_lora.rank` must be >= 1, got {rank}")
    alpha_raw = video_lora.get("alpha")
    alpha = int(alpha_raw) if alpha_raw is not None else rank
    target_modules = _parse_target_modules(video_lora.get("target_modules"))
    checkpoint = video_lora.get("checkpoint")
    if checkpoint is not None:
        checkpoint = str(checkpoint).strip() or None
    return {
        "enabled": enabled,
        "rank": rank,
        "alpha": alpha,
        "target_modules": target_modules,
        "checkpoint": checkpoint,
    }


def is_lora_parameter_name(name: str) -> bool:
    lowered = name.lower()
    return "lora_" in lowered or ".lora" in lowered


def inject_video_lora(
    video_expert: nn.Module,
    *,
    rank: int,
    alpha: int,
    target_modules: Sequence[str],
    checkpoint: str | None = None,
) -> nn.Module:
    try:
        from peft import LoraConfig, inject_adapter_in_model
    except ImportError as exc:
        raise ImportError(
            "Video LoRA training requires the `peft` package. "
            "Install project dependencies (`pip install -e .`) or `pip install peft`."
        ) from exc

    lora_config = LoraConfig(
        r=int(rank),
        lora_alpha=int(alpha),
        target_modules=list(target_modules),
        init_lora_weights="gaussian",
    )
    inject_adapter_in_model(lora_config, video_expert)
    trainable = sum(p.numel() for p in video_expert.parameters() if p.requires_grad)
    total = sum(p.numel() for p in video_expert.parameters())
    logger.info(
        "Injected video LoRA: rank=%d alpha=%d targets=%s trainable=%.2fM / total=%.2fM",
        rank,
        alpha,
        target_modules,
        trainable / 1e6,
        total / 1e6,
    )
    if checkpoint:
        load_video_lora_state_dict(video_expert, checkpoint)
    return video_expert


def export_video_lora_state_dict(video_expert: nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value
        for key, value in video_expert.state_dict().items()
        if is_lora_parameter_name(key)
    }


def load_video_lora_state_dict(
    video_expert: nn.Module,
    state_dict_or_path: dict[str, torch.Tensor] | str,
) -> None:
    if isinstance(state_dict_or_path, str):
        payload = torch.load(state_dict_or_path, map_location="cpu")
        if isinstance(payload, dict) and "video_lora" in payload:
            state_dict = payload["video_lora"]
        elif isinstance(payload, dict):
            state_dict = payload
        else:
            raise ValueError(f"Unsupported LoRA checkpoint type: {type(payload)}")
    else:
        state_dict = state_dict_or_path

    if not state_dict:
        raise ValueError("Video LoRA state dict is empty.")

    incompatible = video_expert.load_state_dict(state_dict, strict=False)
    logger.info(
        "Loaded video LoRA weights: keys=%d missing=%d unexpected=%d",
        len(state_dict),
        len(incompatible.missing_keys),
        len(incompatible.unexpected_keys),
    )


def export_action_expert_state_dict(mot: nn.Module) -> dict[str, torch.Tensor]:
    prefix = "mixtures.action."
    return {
        key[len(prefix) :]: value
        for key, value in mot.state_dict().items()
        if key.startswith(prefix)
    }


def load_action_expert_state_dict(mot: nn.Module, state_dict: dict[str, torch.Tensor]) -> None:
    prefixed = {f"mixtures.action.{key}": value for key, value in state_dict.items()}
    incompatible = mot.load_state_dict(prefixed, strict=False)
    logger.info(
        "Loaded action expert weights: keys=%d missing=%d unexpected=%d",
        len(prefixed),
        len(incompatible.missing_keys),
        len(incompatible.unexpected_keys),
    )


def apply_video_lora_training_mode(model: nn.Module) -> None:
    """Freeze everything except video LoRA adapters, full action expert, and proprio."""
    if not getattr(model, "video_lora_enabled", False):
        raise ValueError("apply_video_lora_training_mode() requires `model.video_lora_enabled=True`.")

    model.eval()
    model.requires_grad_(False)

    model.dit.train()
    model.dit.requires_grad_(False)

    model.video_expert.train()
    for name, param in model.video_expert.named_parameters():
        param.requires_grad = is_lora_parameter_name(name)

    model.action_expert.train()
    model.action_expert.requires_grad_(True)

    proprio_encoder = getattr(model, "proprio_encoder", None)
    if proprio_encoder is not None:
        proprio_encoder.train()
        proprio_encoder.requires_grad_(True)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Video-LoRA train mode: trainable params=%.2fM", trainable / 1e6)


def apply_full_dit_training_mode(model: nn.Module) -> None:
    """Default FastWAM training: full MoT (video + action experts) is trainable."""
    model.eval()
    model.requires_grad_(False)
    model.dit.train()
    model.dit.requires_grad_(True)
    proprio_encoder = getattr(model, "proprio_encoder", None)
    if proprio_encoder is not None:
        proprio_encoder.train()
        proprio_encoder.requires_grad_(True)


def apply_training_mode(model: nn.Module) -> None:
    if getattr(model, "video_lora_enabled", False):
        apply_video_lora_training_mode(model)
    else:
        apply_full_dit_training_mode(model)


def collect_trainable_parameters(model: nn.Module) -> list[nn.Parameter]:
    params = [param for param in model.parameters() if param.requires_grad]
    if not params:
        raise ValueError("No trainable parameters found after applying training mode.")
    return params
