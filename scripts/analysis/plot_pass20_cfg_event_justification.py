#!/usr/bin/env python3
"""Why crop the recoverability event for CFG: success/fail futures diverge there.

Event is defined by Pass@K. From the same prefix the policy emits both
recovering and failing action chunks. CFG can use that paired difference
only if it is larger at the critical interval than at earlier events.

S = RMS between success and fail action centroids (first 24 steps, z-score).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analysis.pass20_scan_data import (  # noqa: E402
    DEFAULT_STATS,
    first_pass_zero,
    last_recoverable_before,
    load_chunks,
    node_metrics,
    sorted_nodes,
)
from scripts.analysis._plot_style import (
    MUTED,
    PASS,
    WINDOW,
    apply_style,
    paired_boxes,
    panel_label,
    polish,
    save_figure,
    stat_note,
)
from scripts.fold_glasses.discover_seedpair_branch_events import (  # noqa: E402
    load_global_zscore,
    normalize_actions,
)

HORIZON = 24
STRIDE = 24
EVENT_NUM_FRAMES = 33


def _centroid_rms(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) == 0 or len(right) == 0:
        return float("nan")
    delta = left.reshape(len(left), -1).mean(0) - right.reshape(len(right), -1).mean(0)
    return float(np.sqrt(np.mean(delta * delta)))


def collect_episodes(
    *,
    data: dict[str, np.ndarray],
    z: np.ndarray,
    metrics: dict[str, np.ndarray],
    min_other: int,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for ep in np.unique(metrics["episode"]):
        sel = metrics["episode"] == ep
        frames, pass_rate, spread = sorted_nodes(
            metrics["frame"][sel], metrics["pass_rate"][sel], metrics["spread"][sel]
        )
        t_zero = first_pass_zero(frames, pass_rate)
        t_star = (
            last_recoverable_before(frames, pass_rate, t_zero)
            if t_zero is not None
            else None
        )
        if t_star is None or int((frames < t_star).sum()) < min_other:
            continue
        gaps: list[float] = []
        n_ok: list[int] = []
        n_fail: list[int] = []
        for frame in frames.tolist():
            idx = np.where(
                (data["episode_index"] == ep) & (data["prefix_frame"] == frame)
            )[0]
            success = data["success"][idx]
            chunk = z[idx][:, :HORIZON]
            n_success = int(success.sum())
            n_failure = int(np.logical_not(success).sum())
            gaps.append(_centroid_rms(chunk[success], chunk[np.logical_not(success)]))
            n_ok.append(n_success)
            n_fail.append(n_failure)
        tau = (frames.astype(np.float64) - float(t_star)) / float(STRIDE)
        before = frames < t_star
        star = frames == t_star
        gap = np.asarray(gaps, dtype=np.float64)
        out.append(
            {
                "episode": int(ep),
                "t_star": int(t_star),
                "t_zero": int(t_zero),
                "frames": frames,
                "tau": tau,
                "pass_rate": pass_rate,
                "uncertainty": spread,
                "branch_gap": gap,
                "n_success": np.asarray(n_ok, dtype=np.int32),
                "n_fail": np.asarray(n_fail, dtype=np.int32),
                "pass_star": float(pass_rate[star][0]),
                "u_star": float(spread[star][0]),
                "u_before": float(np.median(spread[before])),
                "gap_star": float(gap[star][0]),
                "gap_before": float(np.nanmedian(gap[before])),
            }
        )
    return out


def _paired(rows: list[dict[str, Any]], before_key: str, star_key: str) -> dict[str, Any]:
    from scipy.stats import wilcoxon

    a = np.asarray([row[before_key] for row in rows], dtype=np.float64)
    b = np.asarray([row[star_key] for row in rows], dtype=np.float64)
    ok = np.isfinite(a) & np.isfinite(b) & (a > 0)
    a, b = a[ok], b[ok]
    if a.size == 0:
        return {"n": 0}
    ratio = b / a
    p_value = None
    if a.size >= 3:
        try:
            p_value = float(wilcoxon(b, a, alternative="greater").pvalue)
        except ValueError:
            p_value = None
    return {
        "n": int(a.size),
        "n_star_gt_before": int(np.sum(ratio > 1.0)),
        "median_before": float(np.median(a)),
        "median_star": float(np.median(b)),
        "median_ratio": float(np.median(ratio)),
        "wilcoxon_p": p_value,
        "before": a,
        "star": b,
    }


def _align(
    rows: list[dict[str, Any]], key: str, taus: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    means = np.full(len(taus), np.nan)
    sems = np.full(len(taus), np.nan)
    for i, tau in enumerate(taus):
        values = []
        for row in rows:
            hit = np.isclose(row["tau"], tau, atol=1e-6)
            if not hit.any():
                continue
            value = float(row[key][hit][0])
            if np.isfinite(value):
                values.append(value)
        if values:
            arr = np.asarray(values, dtype=np.float64)
            means[i] = float(np.mean(arr))
            sems[i] = float(np.std(arr, ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
    return means, sems


def _stat_line(stats: dict[str, Any]) -> str:
    p_txt = (
        f"$p$={stats['wilcoxon_p']:.3g}" if stats.get("wilcoxon_p") is not None else ""
    )
    return (
        f"{stats['n_star_gt_before']}/{stats['n']} higher at $t^*$\n"
        f"median {stats['median_ratio']:.2f}×   {p_txt}"
    ).strip()


def _paired_panel(
    ax,
    stats: dict[str, Any],
    *,
    ylabel: str,
    title: str,
    letter: str,
    labels: tuple[str, str] = ("earlier", r"$t^*$"),
    event_color: str | None = None,
) -> None:
    if stats.get("n", 0) == 0:
        ax.set_title(title, loc="left")
        return
    kwargs: dict[str, Any] = {"labels": labels}
    if event_color is not None:
        kwargs["event_color"] = event_color
    paired_boxes(ax, stats["before"], stats["star"], **kwargs)
    y_hi = float(np.quantile(np.concatenate([stats["before"], stats["star"]]), 0.9)) * 1.45
    y_hi = max(y_hi, float(np.median(stats["star"])) * 2.2, 1e-3)
    ax.set_ylim(0.0, y_hi)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", pad=8)
    panel_label(ax, letter)
    stat_note(ax, _stat_line(stats))


def plot_justification(rows: list[dict[str, Any]], *, output: Path) -> dict[str, Any]:
    apply_style()
    gap_stats = _paired(rows, "gap_before", "gap_star")
    taus = np.arange(-8.0, 4.0, 1.0)
    pass_mean, pass_sem = _align(rows, "pass_rate", taus)
    lo = -EVENT_NUM_FRAMES / STRIDE + 1.0

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2))
    ax_pass, ax_s = axes

    ok = np.isfinite(pass_mean)
    ax_pass.axvspan(-8.4, lo, color="#f5f5f4", lw=0, zorder=0)
    ax_pass.axvspan(lo, 1.0, color=WINDOW, lw=0, zorder=0)
    ax_pass.axvline(0.0, color="#c4b8b0", ls="--", lw=0.9, zorder=1)
    ax_pass.axvline(1.0, color="#c4b8b0", ls=":", lw=0.9, zorder=1)
    ax_pass.fill_between(
        taus[ok],
        pass_mean[ok] - pass_sem[ok],
        pass_mean[ok] + pass_sem[ok],
        color=PASS,
        alpha=0.16,
        linewidth=0,
        zorder=2,
    )
    ax_pass.plot(taus[ok], pass_mean[ok], color=PASS, lw=2.15, zorder=3, solid_capstyle="round")
    ax_pass.scatter(
        taus[ok],
        pass_mean[ok],
        s=18,
        c=PASS,
        zorder=4,
        edgecolors="white",
        linewidths=0.5,
    )
    ax_pass.set_xlabel(r"replan steps from $t^*$")
    ax_pass.set_ylabel("Pass@20")
    ax_pass.set_title("Recoverability drops at the critical event", loc="left", pad=8)
    ax_pass.set_ylim(0.0, 1.02)
    ax_pass.set_yticks([0.0, 0.5, 1.0])
    polish(ax_pass)
    panel_label(ax_pass, "a")
    ax_pass.text(
        (-8.0 + lo) / 2.0,
        0.98,
        "non-critical",
        ha="center",
        va="top",
        fontsize=8.2,
        color=MUTED,
    )
    ax_pass.text(
        (lo + 1.0) / 2.0,
        0.98,
        "critical",
        ha="center",
        va="top",
        fontsize=8.2,
        color=MUTED,
    )

    _paired_panel(
        ax_s,
        gap_stats,
        ylabel="success vs fail RMS",
        title=r"$S$ is larger at the critical event",
        letter="b",
    )
    fig.suptitle("Why crop the recoverability event for CFG", fontsize=13.5, color="#1c1917", y=1.02)
    fig.tight_layout(w_pad=2.4)
    save_figure(fig, output)

    def compact(stats: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in stats.items() if k not in {"before", "star"}}

    return {
        "png": str(output.resolve()),
        "pdf": str(output.with_suffix(".pdf").resolve()),
        "n_episodes": len(rows),
        "median_pass_at_tstar": float(np.median([row["pass_star"] for row in rows])),
        "branch_gap": compact(gap_stats),
        "episodes": [
            {
                "episode": row["episode"],
                "t_star": row["t_star"],
                "t_zero": row["t_zero"],
                "pass_star": row["pass_star"],
                "gap_star": row["gap_star"],
                "gap_before": row["gap_before"],
            }
            for row in rows
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-stats", type=Path, default=DEFAULT_STATS)
    parser.add_argument("--min-replicates", type=int, default=20)
    parser.add_argument("--min-other", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data = load_chunks(args.scan_root.expanduser().resolve())
    mean, std = load_global_zscore(args.dataset_stats.expanduser().resolve())
    z = normalize_actions(data["chunk"], mean, std)
    metrics = node_metrics(
        chunks=data["chunk"],
        episode_index=data["episode_index"],
        prefix_frame=data["prefix_frame"],
        success=data["success"],
        mean=mean,
        std=std,
        min_replicates=int(args.min_replicates),
    )
    rows = collect_episodes(
        data=data, z=z, metrics=metrics, min_other=int(args.min_other)
    )
    if not rows:
        raise SystemExit("No completed recoverability events")
    meta = plot_justification(rows, output=args.output.expanduser().resolve())
    print(json.dumps(meta, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
