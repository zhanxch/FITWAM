"""Shared RoboTwin 3-camera mosaic + image helpers for FastWAM policy serving.

Used by the Wuji/Astribot real-robot adapter (wuji_fastwam_adapter.py) and any
other client that needs to build the robotwin-style 384x320 RGB mosaic expected
by `RobotVideoDataset` with `concat_multi_camera="robotwin"`.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

DEFAULT_PROMPT = (
    "A video recorded from a robot's point of view executing the following instruction: {task}"
)

# Robotwin 3-cam mosaic (matches robot_video_dataset.concat_multi_camera="robotwin").
ROBOTWIN_TOP_SIZE_WH = (320, 256)
ROBOTWIN_WRIST_SIZE_WH = (160, 128)


def resize_rgb(rgb: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
    image = Image.fromarray(rgb.astype(np.uint8), mode="RGB")
    return np.asarray(image.resize(size_wh, resample=Image.BILINEAR), dtype=np.uint8)


def hwc_rgb_to_input_image_np(rgb: np.ndarray) -> np.ndarray:
    """HWC uint8 -> [1,3,H,W] float32 in [-1, 1] (no resize)."""
    rgb = np.ascontiguousarray(rgb.astype(np.uint8))
    tensor = rgb.transpose(2, 0, 1).astype(np.float32)
    tensor = tensor * (2.0 / 255.0) - 1.0
    return tensor[np.newaxis, ...]


def rgb_to_input_image_np(rgb: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
    """HWC uint8 -> [1,3,H,W] float32 in [-1, 1]."""
    return hwc_rgb_to_input_image_np(resize_rgb(rgb, size_wh))


def concat_robotwin_rgb(top: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Stack ego + wrist cameras into a 384x320 RGB mosaic (HWC uint8)."""
    head = resize_rgb(top, ROBOTWIN_TOP_SIZE_WH)
    wrist_left = resize_rgb(left, ROBOTWIN_WRIST_SIZE_WH)
    wrist_right = resize_rgb(right, ROBOTWIN_WRIST_SIZE_WH)
    bottom = np.concatenate([wrist_left, wrist_right], axis=1)
    return np.ascontiguousarray(np.concatenate([head, bottom], axis=0), dtype=np.uint8)
