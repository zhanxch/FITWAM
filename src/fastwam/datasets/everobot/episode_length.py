"""Length utilities for full-episode EveRobot training.

Aligns variable-length episodes with FastWAM / WAN constraints:

* Video frames are subsampled every ``action_video_freq_ratio`` control steps.
* ``T_video % 4 == 1`` (WAN temporal tokenization).
* ``action_horizon == action_video_freq_ratio * (T_video - 1)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class EpisodeTrim:
    drop_start: int
    num_raw_steps: int
    t_video: int
    action_horizon: int
    video_sample_indices: List[int]


def compute_episode_trim(
    raw_length: int,
    *,
    drop_start: int = 0,
    action_video_freq_ratio: int = 4,
    min_t_video: int = 2,
) -> Optional[EpisodeTrim]:
    """Compute trim lengths for a full episode after optional leading dropout.

    Args:
        raw_length: Total control steps in the episode.
        drop_start: Number of leading control steps to skip.
        action_video_freq_ratio: Subsample one video frame every N control steps.
        min_t_video: Minimum video frames required (WAN needs T > 1).

    Returns:
        ``EpisodeTrim`` when the episode is long enough, otherwise ``None``.
    """
    if drop_start < 0:
        raise ValueError(f"`drop_start` must be >= 0, got {drop_start}")
    if action_video_freq_ratio < 1:
        raise ValueError(
            f"`action_video_freq_ratio` must be >= 1, got {action_video_freq_ratio}"
        )

    available = int(raw_length) - int(drop_start)
    if available < action_video_freq_ratio + 1:
        return None

    max_t_video = (available - 1) // action_video_freq_ratio + 1
    t_video = int(max_t_video)
    while t_video > min_t_video and t_video % 4 != 1:
        t_video -= 1
    if t_video < min_t_video:
        return None

    num_raw_steps = action_video_freq_ratio * (t_video - 1) + 1
    action_horizon = action_video_freq_ratio * (t_video - 1)
    video_sample_indices = list(range(0, num_raw_steps, action_video_freq_ratio))

    assert len(video_sample_indices) == t_video
    assert num_raw_steps == video_sample_indices[-1] + 1
    assert action_horizon == num_raw_steps - 1

    return EpisodeTrim(
        drop_start=int(drop_start),
        num_raw_steps=int(num_raw_steps),
        t_video=int(t_video),
        action_horizon=int(action_horizon),
        video_sample_indices=video_sample_indices,
    )
