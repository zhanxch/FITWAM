"""Standard FastWAM policy-server observation contract (matches training / model.infer_action)."""

from __future__ import annotations

from typing import Any

import torch

# Keys accepted by run_fastwam_server / model.infer_action (do not add sim-specific aliases here).
KEY_INPUT_IMAGE = "input_image"
KEY_PROPRIO = "proprio"
KEY_PROMPT = "prompt"
KEY_CONTEXT = "context"
KEY_CONTEXT_MASK = "context_mask"
KEY_NEGATIVE_PROMPT = "negative_prompt"
KEY_NEGATIVE_CONTEXT = "negative_context"
KEY_NEGATIVE_CONTEXT_MASK = "negative_context_mask"
KEY_FAILURE_PROMPT = "failure_prompt"
KEY_FAILURE_CONTEXT = "failure_context"
KEY_FAILURE_CONTEXT_MASK = "failure_context_mask"

# Returned by get_action after denormalization.
KEY_ACTION = "action"


def validate_policy_observation(observation: dict[str, Any]) -> None:
    """Reject sim/GR00T-style payloads; clients must send training-aligned tensors."""
    if not isinstance(observation, dict):
        raise TypeError(f"observation must be a dict, got {type(observation)}")

    forbidden = {"rgb", "video", "state", "language", "instruction"}
    overlap = forbidden & observation.keys()
    if overlap:
        raise ValueError(
            f"observation contains sim-only keys {sorted(overlap)}. "
            "Convert via fastwam_sim_agent / sim_adapter before calling the policy server."
        )

    if KEY_INPUT_IMAGE not in observation:
        raise ValueError(f"observation must contain '{KEY_INPUT_IMAGE}'")

    img = observation[KEY_INPUT_IMAGE]
    if isinstance(img, dict):
        raise ValueError(
            "observation['input_image'] must be a tensor/ndarray [1,3,H,W] in [-1,1], not a nested dict."
        )

    has_text = KEY_CONTEXT in observation and KEY_CONTEXT_MASK in observation
    has_prompt = KEY_PROMPT in observation
    if not has_text and not has_prompt:
        raise ValueError(
            f"observation must contain '{KEY_PROMPT}' or ('{KEY_CONTEXT}', '{KEY_CONTEXT_MASK}')."
        )
    if has_text and has_prompt:
        raise ValueError(f"Provide either '{KEY_PROMPT}' or cached text tensors, not both.")

    has_negative_context = (
        KEY_NEGATIVE_CONTEXT in observation
        or KEY_NEGATIVE_CONTEXT_MASK in observation
    )
    has_negative_prompt = KEY_NEGATIVE_PROMPT in observation
    if has_negative_context and not (
        KEY_NEGATIVE_CONTEXT in observation
        and KEY_NEGATIVE_CONTEXT_MASK in observation
    ):
        raise ValueError(
            f"Provide both '{KEY_NEGATIVE_CONTEXT}' and "
            f"'{KEY_NEGATIVE_CONTEXT_MASK}' together."
        )
    if has_negative_context and has_negative_prompt:
        raise ValueError(
            f"Provide either '{KEY_NEGATIVE_PROMPT}' or cached negative text tensors, not both."
        )
    if has_prompt and has_negative_context:
        raise ValueError("Prompt input requires a negative prompt, not cached negative context.")
    if has_text and has_negative_prompt:
        raise ValueError("Cached context input requires cached negative context, not a negative prompt.")

    has_failure_context = (
        KEY_FAILURE_CONTEXT in observation or KEY_FAILURE_CONTEXT_MASK in observation
    )
    has_failure_prompt = KEY_FAILURE_PROMPT in observation
    if has_failure_context and not (
        KEY_FAILURE_CONTEXT in observation and KEY_FAILURE_CONTEXT_MASK in observation
    ):
        raise ValueError(
            f"Provide both '{KEY_FAILURE_CONTEXT}' and "
            f"'{KEY_FAILURE_CONTEXT_MASK}' together."
        )
    if has_failure_context and has_failure_prompt:
        raise ValueError(
            f"Provide either '{KEY_FAILURE_PROMPT}' or cached failure text tensors, not both."
        )
    if has_prompt and has_failure_context:
        raise ValueError("Prompt input requires a failure prompt, not cached failure context.")
    if has_text and has_failure_prompt:
        raise ValueError("Cached context input requires cached failure context, not a failure prompt.")


def to_inference_tensors(observation: dict[str, Any], *, device: torch.device, dtype: torch.dtype) -> dict[str, Any]:
    """Convert msgpack/numpy payloads to torch tensors for model.infer_action."""
    validate_policy_observation(observation)

    input_image = observation[KEY_INPUT_IMAGE]
    if not isinstance(input_image, torch.Tensor):
        input_image = torch.as_tensor(input_image, dtype=torch.float32)
    if input_image.ndim == 3:
        input_image = input_image.unsqueeze(0)
    if input_image.ndim != 4 or input_image.shape[1] != 3:
        raise ValueError(
            f"input_image must be [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
        )
    input_image = input_image.to(device=device, dtype=dtype)

    out: dict[str, Any] = {KEY_INPUT_IMAGE: input_image}

    proprio = observation.get(KEY_PROPRIO)
    if proprio is not None:
        if not isinstance(proprio, torch.Tensor):
            proprio = torch.as_tensor(proprio, dtype=torch.float32)
        if proprio.ndim == 1:
            proprio = proprio.unsqueeze(0)
        out[KEY_PROPRIO] = proprio

    if KEY_CONTEXT in observation:
        context = observation[KEY_CONTEXT]
        if not isinstance(context, torch.Tensor):
            context = torch.as_tensor(context, dtype=torch.float32)
        out[KEY_CONTEXT] = context
        mask = observation[KEY_CONTEXT_MASK]
        if not isinstance(mask, torch.Tensor):
            mask = torch.as_tensor(mask, dtype=torch.bool)
        out[KEY_CONTEXT_MASK] = mask
        if KEY_NEGATIVE_CONTEXT in observation:
            negative_context = observation[KEY_NEGATIVE_CONTEXT]
            if not isinstance(negative_context, torch.Tensor):
                negative_context = torch.as_tensor(negative_context, dtype=torch.float32)
            out[KEY_NEGATIVE_CONTEXT] = negative_context
            negative_mask = observation[KEY_NEGATIVE_CONTEXT_MASK]
            if not isinstance(negative_mask, torch.Tensor):
                negative_mask = torch.as_tensor(negative_mask, dtype=torch.bool)
            out[KEY_NEGATIVE_CONTEXT_MASK] = negative_mask
        if KEY_FAILURE_CONTEXT in observation:
            failure_context = observation[KEY_FAILURE_CONTEXT]
            if not isinstance(failure_context, torch.Tensor):
                failure_context = torch.as_tensor(failure_context, dtype=torch.float32)
            out[KEY_FAILURE_CONTEXT] = failure_context
            failure_mask = observation[KEY_FAILURE_CONTEXT_MASK]
            if not isinstance(failure_mask, torch.Tensor):
                failure_mask = torch.as_tensor(failure_mask, dtype=torch.bool)
            out[KEY_FAILURE_CONTEXT_MASK] = failure_mask
    else:
        out[KEY_PROMPT] = str(observation[KEY_PROMPT])
        if KEY_NEGATIVE_PROMPT in observation:
            out[KEY_NEGATIVE_PROMPT] = str(observation[KEY_NEGATIVE_PROMPT])
        if KEY_FAILURE_PROMPT in observation:
            out[KEY_FAILURE_PROMPT] = str(observation[KEY_FAILURE_PROMPT])

    return out
