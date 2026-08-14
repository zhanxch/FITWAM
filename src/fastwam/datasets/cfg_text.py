"""Ternary CFG text channels: outcome / FAST(action) / base."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import numpy as np

DEFAULT_CFG_CHANNEL_PROBS: dict[str, float] = {
    "outcome": 0.4,
    "fast": 0.2,
    "base": 0.4,
}

FAST_SUFFIX_PREFIX = " Action codes:"


def normalize_cfg_channel_probs(
    probs: Optional[Mapping[str, Any]],
) -> Optional[dict[str, float]]:
    """Return normalized {outcome, fast, base} or None if disabled."""

    if probs is None:
        return None
    if not isinstance(probs, Mapping):
        raise TypeError(f"cfg_channel_probs must be a mapping, got {type(probs)}")
    out = {
        "outcome": float(probs.get("outcome", 0.0)),
        "fast": float(probs.get("fast", 0.0)),
        "base": float(probs.get("base", 0.0)),
    }
    for key, value in out.items():
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"cfg_channel_probs[{key}] must be finite and >= 0, got {value}")
    total = float(sum(out.values()))
    if total <= 0.0:
        raise ValueError(f"cfg_channel_probs must sum to > 0, got {out}")
    return {key: value / total for key, value in out.items()}


def sample_cfg_channel(
    probs: Mapping[str, float],
    *,
    rng: np.random.RandomState | None = None,
) -> str:
    keys = ("outcome", "fast", "base")
    weights = np.asarray([float(probs[k]) for k in keys], dtype=np.float64)
    weights = weights / weights.sum()
    if rng is None:
        idx = int(np.random.choice(len(keys), p=weights))
    else:
        idx = int(rng.choice(len(keys), p=weights))
    return keys[idx]


def outcome_suffix_for_flag(
    outcome_flag: int,
    *,
    success_suffix: str,
    failure_suffix: str,
) -> str:
    return failure_suffix if int(outcome_flag) == 1 else success_suffix


_FAST_PROCESSOR = None


def _get_fast_processor(model_id: str = "physical-intelligence/fast"):
    global _FAST_PROCESSOR
    if _FAST_PROCESSOR is None:
        from transformers import AutoProcessor

        _FAST_PROCESSOR = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    return _FAST_PROCESSOR


def actions_to_fast_token_ids(
    actions: np.ndarray | Any,
    *,
    model_id: str = "physical-intelligence/fast",
    max_tokens: int = 32,
) -> list[int]:
    """Encode an action chunk with the public FAST tokenizer; truncate for T5."""

    arr = np.asarray(actions, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2:
        raise ValueError(f"actions must be [T,D] or [D], got shape {arr.shape}")
    if arr.shape[0] < 1 or arr.shape[1] < 1:
        raise ValueError(f"actions must be non-empty, got shape {arr.shape}")

    processor = _get_fast_processor(model_id)
    token_ids = processor(arr[None])[0]
    token_ids = [int(x) for x in np.asarray(token_ids).reshape(-1).tolist()]
    if max_tokens > 0:
        token_ids = token_ids[: int(max_tokens)]
    return token_ids


def format_fast_action_suffix(
    actions: np.ndarray | Any,
    *,
    model_id: str = "physical-intelligence/fast",
    max_tokens: int = 32,
    prefix: str = FAST_SUFFIX_PREFIX,
) -> str:
    token_ids = actions_to_fast_token_ids(
        actions, model_id=model_id, max_tokens=max_tokens
    )
    return f"{prefix} {' '.join(str(t) for t in token_ids)}"


def apply_ternary_cfg_suffix(
    task_text: str,
    *,
    outcome_flag: int,
    success_suffix: Optional[str],
    failure_suffix: Optional[str],
    channel_probs: Optional[Mapping[str, float]],
    failure_channel_probs: Optional[Mapping[str, float]] = None,
    actions: Optional[np.ndarray | Any] = None,
    is_training: bool = True,
    fast_model_id: str = "physical-intelligence/fast",
    fast_max_tokens: int = 32,
    fast_fail_closed: bool = False,
    legacy_dropout_prob: float = 0.0,
) -> tuple[str, str]:
    """Return (task_text_with_optional_suffix, channel_name).

    channel_name in {base, outcome, fast}.
    """

    selected_probs = (
        failure_channel_probs
        if int(outcome_flag) == 1 and failure_channel_probs is not None
        else channel_probs
    )
    probs = normalize_cfg_channel_probs(selected_probs)
    if probs is None:
        # Legacy binary outcome CFG with dropout → base.
        if success_suffix is None or failure_suffix is None:
            return task_text, "base"
        if (
            is_training
            and float(legacy_dropout_prob) > 0.0
            and float(np.random.random()) < float(legacy_dropout_prob)
        ):
            return task_text, "base"
        suffix = failure_suffix if int(outcome_flag) == 1 else success_suffix
        if suffix is None:
            return task_text, "base"
        return f"{task_text}{suffix}", "outcome"

    # Train and val/test both sample ternary CFG channels. Deployment CFG is a
    # separate success-vs-base inference schedule; do not collapse val to the
    # success/outcome condition alone.
    channel = sample_cfg_channel(probs)
    if channel == "base":
        return task_text, "base"
    if channel == "outcome":
        suffix = failure_suffix if int(outcome_flag) == 1 else success_suffix
        if suffix is None:
            raise ValueError(
                "Outcome CFG channel was sampled but its suffix is null."
            )
        return f"{task_text}{suffix}", "outcome"

    # Historical recipes fall back to outcome. Formal FAST experiments can
    # fail closed so their configured channel probability remains auditable.
    if actions is None:
        if fast_fail_closed:
            raise ValueError("FAST CFG channel was sampled but actions are missing.")
        suffix = failure_suffix if int(outcome_flag) == 1 else success_suffix
        return (task_text, "base") if suffix is None else (f"{task_text}{suffix}", "outcome")
    try:
        fast_suffix = format_fast_action_suffix(
            actions,
            model_id=fast_model_id,
            max_tokens=fast_max_tokens,
        )
    except Exception as exc:
        if fast_fail_closed:
            raise RuntimeError("FAST CFG action encoding failed.") from exc
        suffix = failure_suffix if int(outcome_flag) == 1 else success_suffix
        return (task_text, "base") if suffix is None else (f"{task_text}{suffix}", "outcome")
    return f"{task_text}{fast_suffix}", "fast"


def expand_prompts_with_outcome_suffixes(
    base_prompts: Sequence[str],
    *,
    success_suffix: Optional[str],
    failure_suffix: Optional[str],
) -> list[str]:
    prompts: list[str] = []
    seen: set[str] = set()
    for prompt in base_prompts:
        for candidate in (
            prompt,
            f"{prompt}{success_suffix}" if success_suffix else None,
            f"{prompt}{failure_suffix}" if failure_suffix else None,
        ):
            if candidate is None or candidate in seen:
                continue
            seen.add(candidate)
            prompts.append(candidate)
    return prompts
