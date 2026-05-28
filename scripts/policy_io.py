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
    else:
        out[KEY_PROMPT] = str(observation[KEY_PROMPT])

    return out
