"""Prefix-conditional future metrics shared by action chunks and replayed video.

S = RMS between success and fail centroids.
U = mean pairwise RMS among the M samples.
AUROC = leave-one-out centroid scoring of the M labels.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def centroid_rms(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) == 0 or len(right) == 0:
        return float("nan")
    delta = (
        np.asarray(left, dtype=np.float64).reshape(len(left), -1).mean(0)
        - np.asarray(right, dtype=np.float64).reshape(len(right), -1).mean(0)
    )
    return float(np.sqrt(np.mean(delta * delta)))


def mean_pairwise_rms(group: np.ndarray) -> float:
    flat = np.asarray(group, dtype=np.float64).reshape(len(group), -1)
    n, dim = flat.shape
    if n < 2:
        return float("nan")
    # Broadcast is fine for action chunks; video vectors are millions of dims.
    if n * n * dim > 4_000_000:
        total = 0.0
        count = 0
        for i in range(n - 1):
            delta = flat[i + 1 :] - flat[i]
            total += float(np.sum(np.sqrt(np.mean(delta * delta, axis=1))))
            count += n - 1 - i
        return total / max(count, 1)
    diff = flat[:, None, :] - flat[None, :, :]
    dist = np.sqrt(np.mean(diff * diff, axis=-1))
    iu = np.triu_indices(n, k=1)
    return float(np.mean(dist[iu]))


def _rms_row(vector: np.ndarray, centroid: np.ndarray) -> float:
    delta = np.asarray(vector, dtype=np.float64) - np.asarray(centroid, dtype=np.float64)
    return float(np.sqrt(np.mean(delta * delta)))


def mann_whitney_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    pos = scores[labels]
    neg = scores[~labels]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    greater = np.sum(pos[:, None] > neg[None, :])
    equal = np.sum(pos[:, None] == neg[None, :])
    return float((greater + 0.5 * equal) / (pos.size * neg.size))


def loo_centroid_scores(samples: np.ndarray, success: np.ndarray) -> np.ndarray | None:
    """Higher score = closer to the success centroid than the fail centroid."""

    flat = np.asarray(samples, dtype=np.float64).reshape(len(samples), -1)
    success = np.asarray(success, dtype=bool)
    if flat.shape[0] != success.shape[0]:
        raise ValueError("samples and success must have the same length")
    n_ok = int(success.sum())
    n_fail = int((~success).sum())
    if n_ok < 2 or n_fail < 2:
        return None
    scores = np.empty(len(flat), dtype=np.float64)
    for index in range(len(flat)):
        mask = np.ones(len(flat), dtype=bool)
        mask[index] = False
        left = success[mask]
        others = flat[mask]
        if not left.any() or np.all(left):
            return None
        centroid_ok = others[left].mean(axis=0)
        centroid_fail = others[~left].mean(axis=0)
        scores[index] = _rms_row(flat[index], centroid_fail) - _rms_row(
            flat[index], centroid_ok
        )
    return scores


def loo_centroid_auroc(samples: np.ndarray, success: np.ndarray) -> float:
    scores = loo_centroid_scores(samples, success)
    if scores is None:
        return float("nan")
    return mann_whitney_auroc(scores, success)


def roc_curve(labels: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.size != scores.size or not np.isfinite(scores).all():
        return None
    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(-scores, kind="mergesort")
    ranked = labels[order]
    tps = np.concatenate([[0], np.cumsum(ranked)])
    fps = np.concatenate([[0], np.cumsum(~ranked)])
    return fps / n_neg, tps / n_pos


def _strict_increasing(fpr: np.ndarray, tpr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xs = [float(fpr[0])]
    ys = [float(tpr[0])]
    for x, y in zip(fpr[1:], tpr[1:]):
        if x == xs[-1]:
            ys[-1] = float(y)
        else:
            xs.append(float(x))
            ys.append(float(y))
    return np.asarray(xs), np.asarray(ys)


def mean_roc(
    groups: list[tuple[np.ndarray, np.ndarray]], *, grid: np.ndarray | None = None
) -> dict[str, Any]:
    grid = np.linspace(0.0, 1.0, 51) if grid is None else np.asarray(grid, dtype=np.float64)
    curves: list[np.ndarray] = []
    aucs: list[float] = []
    n_takeovers = 0
    for labels, scores in groups:
        points = roc_curve(labels, scores)
        if points is None:
            continue
        fpr, tpr = _strict_increasing(*points)
        curves.append(np.interp(grid, fpr, tpr))
        aucs.append(mann_whitney_auroc(scores, labels))
        n_takeovers += int(len(labels))
    if not curves:
        return {"n": 0}
    stack = np.stack(curves, axis=0)
    mean = stack.mean(axis=0)
    sem = (
        stack.std(axis=0, ddof=1) / np.sqrt(len(curves))
        if len(curves) > 1
        else np.zeros_like(mean)
    )
    return {
        "n": int(len(curves)),
        "n_takeovers": n_takeovers,
        "grid": grid,
        "mean": mean,
        "sem": sem,
        "median_auc": float(np.median(aucs)),
        "mean_auc": float(np.mean(aucs)),
    }


def pooled_zscore_auroc(groups: list[tuple[np.ndarray, np.ndarray]]) -> float:
    scores: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for lab, sc in groups:
        sc = np.asarray(sc, dtype=np.float64)
        lab = np.asarray(lab, dtype=bool)
        std = float(np.std(sc))
        if std <= 0 or int(lab.sum()) == 0 or int((~lab).sum()) == 0:
            continue
        scores.append((sc - float(np.mean(sc))) / std)
        labels.append(lab)
    if not scores:
        return float("nan")
    return mann_whitney_auroc(np.concatenate(scores), np.concatenate(labels))


def branch_metrics(samples: np.ndarray, success: np.ndarray) -> dict[str, Any]:
    success = np.asarray(success, dtype=bool)
    samples = np.asarray(samples)
    n_ok = int(success.sum())
    n_fail = int((~success).sum())
    gap = centroid_rms(samples[success], samples[~success])
    spread = mean_pairwise_rms(samples)
    ratio = (
        float(gap / spread)
        if np.isfinite(gap) and np.isfinite(spread) and spread > 0
        else float("nan")
    )
    loo_scores = loo_centroid_scores(samples, success)
    auroc = (
        float(mann_whitney_auroc(loo_scores, success))
        if loo_scores is not None
        else float("nan")
    )
    return {
        "n": int(len(success)),
        "n_success": n_ok,
        "n_fail": n_fail,
        "s": float(gap),
        "u": float(spread),
        "s_over_u": ratio,
        "auroc": auroc,
        "loo_scores": None if loo_scores is None else [float(v) for v in loo_scores.tolist()],
    }
