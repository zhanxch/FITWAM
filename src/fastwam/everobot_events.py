"""Pure NumPy post-processing for state-line event candidates.

All frame intervals use half-open semantics: ``[start_frame, end_frame)``.
The functions are stateless so extraction parameters can be versioned by the
EveRobot sidecar without coupling the algorithm to storage code.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EventCandidate:
    """One sustained state-transition candidate within an episode."""

    core_start_frame: int
    core_end_frame: int
    start_frame: int
    end_frame: int
    confidence: float
    peak_score: float
    episode_weight: float


@dataclass(frozen=True)
class CandidateExtraction:
    """Intermediate signals and final candidates from one episode."""

    smoothed_scores: np.ndarray
    active_mask: np.ndarray
    candidates: tuple[EventCandidate, ...]


def _as_1d_float(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array")
    return array


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or int(value) != value or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _nonnegative_int(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or int(value) != value or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)


def trailing_median(values: np.ndarray, window_size: int) -> np.ndarray:
    """Return a finite-only trailing median while preserving empty windows.

    NaN and infinite values are ignored. Position ``i`` uses only
    ``[max(0, i-window_size+1), i]``, so this operation does not look ahead.
    The result is NaN only when that complete trailing window has no finite
    value.
    """

    values = _as_1d_float(values, "values")
    window_size = _positive_int(window_size, "window_size")
    output = np.full(values.shape, np.nan, dtype=np.float64)

    for index in range(values.size):
        start = max(0, index - window_size + 1)
        window = values[start : index + 1]
        finite = window[np.isfinite(window)]
        if finite.size:
            output[index] = float(np.median(finite))
    return output


def exponential_moving_average(values: np.ndarray, alpha: float) -> np.ndarray:
    """Return a finite-aware EMA without inventing values at invalid frames.

    Invalid positions remain NaN and do not reset the previous finite EMA
    state. The next finite position resumes from that state.
    """

    values = _as_1d_float(values, "values")
    alpha = float(alpha)
    if not np.isfinite(alpha) or not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must be finite and in (0, 1]")

    output = np.full(values.shape, np.nan, dtype=np.float64)
    state: float | None = None
    for index, value in enumerate(values):
        if not np.isfinite(value):
            continue
        state = float(value) if state is None else alpha * float(value) + (1.0 - alpha) * state
        output[index] = state
    return output


def smooth_scores(
    scores: np.ndarray,
    *,
    median_window: int = 1,
    ema_alpha: float = 1.0,
) -> np.ndarray:
    """Apply the two smoothers while preserving the raw invalid-frame mask."""

    scores = _as_1d_float(scores, "scores")
    smoothed = exponential_moving_average(
        trailing_median(scores, median_window),
        ema_alpha,
    )
    smoothed[~np.isfinite(scores)] = np.nan
    return smoothed


def hysteresis_mask(
    scores: np.ndarray,
    *,
    high_threshold: float,
    low_threshold: float,
) -> np.ndarray:
    """Convert scores into sustained activity with high/low hysteresis.

    A finite score at or above ``high_threshold`` starts an event. The event
    remains active until the score falls below ``low_threshold``. Invalid
    scores stop the event so missing spans cannot be bridged implicitly.
    """

    scores = _as_1d_float(scores, "scores")
    high_threshold = float(high_threshold)
    low_threshold = float(low_threshold)
    if not np.isfinite(high_threshold) or not np.isfinite(low_threshold):
        raise ValueError("hysteresis thresholds must be finite")
    if low_threshold > high_threshold:
        raise ValueError("low_threshold must not exceed high_threshold")

    output = np.zeros(scores.shape, dtype=bool)
    active = False
    for index, score in enumerate(scores):
        if not np.isfinite(score):
            active = False
        elif active:
            active = bool(score >= low_threshold)
        else:
            active = bool(score >= high_threshold)
        output[index] = active
    return output


def merge_short_gaps(
    mask: np.ndarray,
    max_gap: int,
    *,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Fill internal false runs no longer than ``max_gap``.

    Gaps touching an invalid frame are never merged. Leading and trailing
    false runs are also left unchanged.
    """

    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 1:
        raise ValueError("mask must be a one-dimensional array")
    max_gap = _nonnegative_int(max_gap, "max_gap")
    if valid_mask is None:
        valid = np.ones(mask.shape, dtype=bool)
    else:
        valid = np.asarray(valid_mask, dtype=bool)
        if valid.shape != mask.shape:
            raise ValueError("valid_mask must have the same shape as mask")

    output = mask.copy()
    index = 0
    while index < output.size:
        if output[index]:
            index += 1
            continue
        start = index
        while index < output.size and not output[index]:
            index += 1
        end = index
        bounded = start > 0 and end < output.size and output[start - 1] and output[end]
        if bounded and end - start <= max_gap and bool(valid[start:end].all()):
            output[start:end] = True
    return output


def remove_short_runs(mask: np.ndarray, min_run: int) -> np.ndarray:
    """Remove true runs shorter than ``min_run`` frames."""

    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 1:
        raise ValueError("mask must be a one-dimensional array")
    min_run = _positive_int(min_run, "min_run")

    output = mask.copy()
    index = 0
    while index < output.size:
        if not output[index]:
            index += 1
            continue
        start = index
        while index < output.size and output[index]:
            index += 1
        if index - start < min_run:
            output[start:index] = False
    return output


def _true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    index = 0
    while index < mask.size:
        if not mask[index]:
            index += 1
            continue
        start = index
        while index < mask.size and mask[index]:
            index += 1
        runs.append((start, index))
    return runs


def _expand_window(
    core_start: int,
    core_end: int,
    *,
    episode_length: int,
    pre_padding: int,
    post_padding: int,
    min_window: int,
) -> tuple[int, int]:
    start = max(0, core_start - pre_padding)
    end = min(episode_length, core_end + post_padding)
    target = min(min_window, episode_length)
    deficit = max(0, target - (end - start))

    add_left = min(start, deficit // 2)
    start -= add_left
    deficit -= add_left

    add_right = min(episode_length - end, deficit)
    end += add_right
    deficit -= add_right

    if deficit:
        start -= min(start, deficit)
    return start, end


def extract_candidate_windows(
    scores: np.ndarray,
    *,
    median_window: int = 1,
    ema_alpha: float = 1.0,
    high_threshold: float = 0.5,
    low_threshold: float = 0.5,
    max_gap: int = 0,
    min_run: int = 1,
    pre_padding: int = 0,
    post_padding: int = 0,
    min_window: int = 1,
    max_candidates_per_episode: int | None = None,
) -> CandidateExtraction:
    """Extract deterministic state-line candidates from one episode.

    Candidate confidence is the finite mean smoothed score over the sustained
    core. Episode weights normalize confidences to sum to one; a uniform
    fallback is used only when every confidence is zero.
    """

    scores = _as_1d_float(scores, "scores")
    max_gap = _nonnegative_int(max_gap, "max_gap")
    min_run = _positive_int(min_run, "min_run")
    pre_padding = _nonnegative_int(pre_padding, "pre_padding")
    post_padding = _nonnegative_int(post_padding, "post_padding")
    min_window = _positive_int(min_window, "min_window")
    if max_candidates_per_episode is not None:
        max_candidates_per_episode = _positive_int(
            max_candidates_per_episode, "max_candidates_per_episode"
        )

    smoothed = smooth_scores(
        scores,
        median_window=median_window,
        ema_alpha=ema_alpha,
    )
    valid = np.isfinite(smoothed)
    active = hysteresis_mask(
        smoothed,
        high_threshold=high_threshold,
        low_threshold=low_threshold,
    )
    active = remove_short_runs(active, min_run)
    active = merge_short_gaps(active, max_gap, valid_mask=valid)

    provisional: list[tuple[int, int, int, int, float, float]] = []
    for core_start, core_end in _true_runs(active):
        core_scores = smoothed[core_start:core_end]
        finite = core_scores[np.isfinite(core_scores)]
        confidence = float(np.clip(finite.mean(), 0.0, 1.0))
        peak_score = float(np.clip(finite.max(), 0.0, 1.0))
        start, end = _expand_window(
            core_start,
            core_end,
            episode_length=scores.size,
            pre_padding=pre_padding,
            post_padding=post_padding,
            min_window=min_window,
        )
        provisional.append(
            (core_start, core_end, start, end, confidence, peak_score)
        )

    if (
        max_candidates_per_episode is not None
        and len(provisional) > max_candidates_per_episode
    ):
        retained_indices = set(
            sorted(
                range(len(provisional)),
                key=lambda index: (
                    -provisional[index][4],
                    -provisional[index][5],
                    provisional[index][0],
                    provisional[index][1],
                ),
            )[:max_candidates_per_episode]
        )
        provisional = [
            candidate
            for index, candidate in enumerate(provisional)
            if index in retained_indices
        ]
        active = np.zeros(active.shape, dtype=bool)
        for core_start, core_end, *_ in provisional:
            active[core_start:core_end] = True

    if provisional:
        confidences = np.asarray([row[4] for row in provisional], dtype=np.float64)
        confidence_sum = float(confidences.sum())
        if confidence_sum > 0.0:
            weights = confidences / confidence_sum
        else:
            weights = np.full(confidences.shape, 1.0 / confidences.size)
    else:
        weights = np.empty(0, dtype=np.float64)

    candidates = tuple(
        EventCandidate(
            core_start_frame=core_start,
            core_end_frame=core_end,
            start_frame=start,
            end_frame=end,
            confidence=confidence,
            peak_score=peak_score,
            episode_weight=float(weight),
        )
        for (core_start, core_end, start, end, confidence, peak_score), weight in zip(
            provisional, weights, strict=True
        )
    )
    return CandidateExtraction(
        smoothed_scores=smoothed,
        active_mask=active,
        candidates=candidates,
    )


__all__ = [
    "CandidateExtraction",
    "EventCandidate",
    "exponential_moving_average",
    "extract_candidate_windows",
    "hysteresis_mask",
    "merge_short_gaps",
    "remove_short_runs",
    "smooth_scores",
    "trailing_median",
]
