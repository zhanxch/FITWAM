"""EveRobot dataset format: episode-level sampling with first-frame dropout.

Each episode (one mp4 per camera + one parquet) is treated as a single training
sample, inspired by DiffSynth-Studio's WAN fine-tuning where each video is one
sample. Instead of DiffSynth's ``dataset_repeat`` (seeing the same clip N times),
EveRobot drops the first N frames to create diverse sub-clips from the same
episode.

This module reuses ``RobotVideoDataset`` for all frame loading, multi-camera
concatenation, normalization, and text-embedding caching — only the indexing
strategy changes from frame-level to episode-level.
"""

from .dataset import EveRobotDataset
from .full_episode_dataset import EveRobotFullEpisodeDataset

__all__ = ["EveRobotDataset", "EveRobotFullEpisodeDataset"]
