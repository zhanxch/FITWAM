from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ConversionConfig:
    output_root: Path
    sim_root: Path
    fps: int
    overwrite: bool
    include_hand_pose: bool
    overlay_hand_pose: bool


@dataclass(frozen=True)
class TaskSource:
    root: Path
    task_text: str


@dataclass(frozen=True)
class EpisodeArrays:
    action: np.ndarray
    state: np.ndarray
    frames: np.ndarray
    hand: dict[str, np.ndarray | None]


@dataclass(frozen=True)
class DatasetSummary:
    total_episodes: int
    total_frames: int
    action_dim: int
    state_dim: int
    image_shape: tuple[int, int, int]

