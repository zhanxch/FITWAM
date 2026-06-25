"""Wuji/Astribot real-robot ↔ FastWAM policy-server I/O adapter."""

from __future__ import annotations

from typing import Any

import numpy as np

from dexjoco_fastwam_adapter import (
    DEFAULT_PROMPT,
    concat_robotwin_rgb,
    hwc_rgb_to_input_image_np,
)

WUJI_ACTION_DIM = 58
WUJI_LEFT_EEF_DIM = 9
WUJI_RIGHT_EEF_DIM = 9
WUJI_LEFT_HAND_DIM = 20
WUJI_RIGHT_HAND_DIM = 20

WUJI_ACTION_KEYS = ("left_eef", "right_eef", "left_hand_joints", "right_hand_joints")
WUJI_VIDEO_KEYS = ("head_view", "left_wrist_view", "right_wrist_view")
GR00T_LANGUAGE_KEY = "annotation.human.action.task_description"

_WUJI_SLICES = {
    "left_eef": slice(0, 9),
    "right_eef": slice(9, 18),
    "left_hand_joints": slice(18, 38),
    "right_hand_joints": slice(38, 58),
}


def is_gr00t_observation(observation: dict[str, Any]) -> bool:
    return isinstance(observation, dict) and "video" in observation


def extract_gr00t_task(observation: dict[str, Any], *, fallback: str = "") -> str:
    language = observation.get("language")
    if isinstance(language, dict) and GR00T_LANGUAGE_KEY in language:
        value = language[GR00T_LANGUAGE_KEY]
        if isinstance(value, list) and value and isinstance(value[0], list) and value[0]:
            return str(value[0][0])
        if isinstance(value, list) and value:
            return str(value[0])
        if isinstance(value, str):
            return value
    return str(fallback)


def _extract_latest_rgb_frame(array: Any, *, view_name: str) -> np.ndarray:
    frame = np.asarray(array)
    if frame.ndim == 5:
        frame = frame[0, 0]
    elif frame.ndim == 4:
        frame = frame[0]
    elif frame.ndim == 3:
        pass
    else:
        raise ValueError(
            f"video.{view_name} must be (B,T,H,W,C), (T,H,W,C), or (H,W,C), got {frame.shape}"
        )
    if frame.ndim != 3 or frame.shape[-1] != 3:
        raise ValueError(f"video.{view_name} frame must be (H,W,3), got {frame.shape}")
    return np.ascontiguousarray(frame.astype(np.uint8))


def gr00t_obs_to_policy_obs(
    observation: dict[str, Any],
    *,
    task: str | None = None,
    include_proprio: bool = False,
) -> dict[str, Any]:
    """Convert GR00T Wuji observation to FastWAM policy-server payload."""
    video = observation.get("video")
    if not isinstance(video, dict):
        raise ValueError("GR00T observation must contain a 'video' dict")

    missing = [key for key in WUJI_VIDEO_KEYS if key not in video]
    if missing:
        raise KeyError(f"GR00T observation video missing required views: {missing}")

    head = _extract_latest_rgb_frame(video["head_view"], view_name="head_view")
    left = _extract_latest_rgb_frame(video["left_wrist_view"], view_name="left_wrist_view")
    right = _extract_latest_rgb_frame(video["right_wrist_view"], view_name="right_wrist_view")
    mosaic = concat_robotwin_rgb(head, left, right)
    input_image = hwc_rgb_to_input_image_np(mosaic)

    task_text = extract_gr00t_task(observation, fallback=task or "")
    if not task_text.strip():
        raise ValueError("GR00T observation is missing a language task instruction")

    policy_obs: dict[str, Any] = {
        "input_image": input_image,
        "prompt": task_text,
    }

    if include_proprio:
        state = observation.get("state")
        if not isinstance(state, dict):
            raise ValueError("GR00T observation must contain 'state' when proprio is enabled")
        proprio = merge_wuji_state(state)
        policy_obs["proprio"] = proprio

    return policy_obs


def merge_wuji_state(state: dict[str, Any]) -> np.ndarray:
    """Merge GR00T state dict into a single 58-d vector (latest timestep)."""
    parts: list[np.ndarray] = []
    for key in WUJI_ACTION_KEYS:
        if key not in state:
            raise KeyError(f"GR00T state missing required key: {key}")
        value = np.asarray(state[key], dtype=np.float32)
        if value.ndim == 3:
            value = value[0, 0]
        elif value.ndim == 2:
            value = value[0]
        elif value.ndim != 1:
            raise ValueError(f"state.{key} must be (D,), (1,D), or (1,1,D), got {value.shape}")
        parts.append(value.reshape(-1))

    merged = np.concatenate(parts, axis=0)
    if merged.shape != (WUJI_ACTION_DIM,):
        raise ValueError(f"Merged Wuji state must be ({WUJI_ACTION_DIM},), got {merged.shape}")
    return merged


def split_wuji_action(action: Any) -> dict[str, np.ndarray]:
    """Split [T,58] action into GR00T-style keys with shape (1, T, dim)."""
    action_np = np.asarray(action, dtype=np.float32)
    if action_np.ndim == 1:
        action_np = action_np[np.newaxis, :]
    if action_np.ndim != 2 or action_np.shape[-1] != WUJI_ACTION_DIM:
        raise ValueError(f"Expected action [T,{WUJI_ACTION_DIM}], got {action_np.shape}")

    return {
        key: action_np[:, slc][np.newaxis, ...].astype(np.float32, copy=False)
        for key, slc in _WUJI_SLICES.items()
    }
