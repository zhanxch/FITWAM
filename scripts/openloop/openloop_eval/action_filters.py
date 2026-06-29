"""Optional post-processing filters for open-loop action trajectories.

These utilities operate on stitched raw action series after model inference. They do
not affect the model rollout itself unless a caller explicitly feeds the filtered
series back into another system.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def ema_low_pass(series: np.ndarray, *, alpha: float) -> np.ndarray:
    """Apply per-dimension exponential moving average to a [T, D] series.

    NaN gaps are preserved and reset the filter state for that dimension. Smaller
    alpha means stronger smoothing; alpha=1 returns the input unchanged.
    """
    if not 0.0 < float(alpha) <= 1.0:
        raise ValueError(f"alpha must be in (0, 1], got {alpha}")
    arr = np.asarray(series, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected [T, D] action series, got shape {arr.shape}")

    out = np.full_like(arr, np.nan, dtype=np.float32)
    prev = np.full((arr.shape[1],), np.nan, dtype=np.float32)
    alpha_f = np.float32(alpha)
    for t in range(arr.shape[0]):
        current = arr[t]
        finite = np.isfinite(current)
        reset = ~np.isfinite(prev)
        out[t, finite & reset] = current[finite & reset]
        keep = finite & ~reset
        out[t, keep] = alpha_f * current[keep] + (np.float32(1.0) - alpha_f) * prev[keep]
        prev[finite] = out[t, finite]
        prev[~finite] = np.nan
    return out


def _finite_diff_values(series: np.ndarray, replan_frames: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = np.asarray(series, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected [T, D] action series, got shape {arr.shape}")
    boundaries = {int(x) for x in np.asarray(replan_frames).reshape(-1).tolist() if 0 < int(x) < arr.shape[0]}
    boundary_diffs: list[np.ndarray] = []
    within_diffs: list[np.ndarray] = []
    all_diffs: list[np.ndarray] = []
    for t in range(1, arr.shape[0]):
        prev = arr[t - 1]
        cur = arr[t]
        valid = np.isfinite(prev) & np.isfinite(cur)
        diff = np.full((arr.shape[1],), np.nan, dtype=np.float32)
        diff[valid] = np.abs(cur[valid] - prev[valid])
        all_diffs.append(diff)
        if t in boundaries:
            boundary_diffs.append(diff)
        else:
            within_diffs.append(diff)

    def stack_or_empty(values: list[np.ndarray]) -> np.ndarray:
        if not values:
            return np.empty((0, arr.shape[1]), dtype=np.float32)
        return np.stack(values, axis=0)

    return stack_or_empty(all_diffs), stack_or_empty(boundary_diffs), stack_or_empty(within_diffs)


def _nan_mean_max(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if values.size == 0:
        width = values.shape[1] if values.ndim == 2 else 0
        return np.full((width,), np.nan, dtype=np.float32), np.full((width,), np.nan, dtype=np.float32)
    return np.nanmean(values, axis=0), np.nanmax(values, axis=0)


def action_series_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, Any]:
    pred_arr = np.asarray(pred, dtype=np.float32)
    gt_arr = np.asarray(gt, dtype=np.float32)
    if pred_arr.shape != gt_arr.shape:
        raise ValueError(f"pred/gt shape mismatch: {pred_arr.shape} vs {gt_arr.shape}")
    valid = np.isfinite(pred_arr) & np.isfinite(gt_arr)
    diff = np.abs(pred_arr - gt_arr)
    sq = (pred_arr - gt_arr) ** 2
    diff_valid = diff[valid]
    sq_valid = sq[valid]
    if diff_valid.size == 0:
        raise ValueError("No valid action entries for metrics")
    return {
        "action_l1": float(np.mean(diff_valid)),
        "action_mse": float(np.mean(sq_valid)),
        "action_rmse": float(np.sqrt(np.mean(sq_valid))),
        "action_max_abs": float(np.max(diff_valid)),
        "action_l1_per_dim": np.nanmean(diff, axis=0),
        "action_mse_per_dim": np.nanmean(sq, axis=0),
    }


def jump_statistics(series: np.ndarray, replan_frames: np.ndarray) -> dict[str, Any]:
    all_diffs, boundary_diffs, within_diffs = _finite_diff_values(series, replan_frames)
    all_mean, all_max = _nan_mean_max(all_diffs)
    boundary_mean, boundary_max = _nan_mean_max(boundary_diffs)
    within_mean, within_max = _nan_mean_max(within_diffs)
    return {
        "all_mean_per_dim": all_mean,
        "all_max_per_dim": all_max,
        "boundary_mean_per_dim": boundary_mean,
        "boundary_max_per_dim": boundary_max,
        "within_mean_per_dim": within_mean,
        "within_max_per_dim": within_max,
        "all_max": float(np.nanmax(all_max)) if all_max.size else float("nan"),
        "boundary_max": float(np.nanmax(boundary_max)) if boundary_max.size else float("nan"),
        "within_max": float(np.nanmax(within_max)) if within_max.size else float("nan"),
    }
