#!/usr/bin/env python3
"""Video analogue of the Pass@20 CFG-event justification, from replayed chunks.

S_V / U_V / AUROC_V are computed on the 24 executed camera frames of each
Pass@20 first-action chunk. Action metrics use the same 20 samples.
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

from scripts.analysis._plot_style import (
    EVENT,
    INK,
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
from scripts.analysis.pass20_future_metrics import branch_metrics
from scripts.analysis.pass20_scan_data import DEFAULT_STATS, load_chunks, node_metrics
from scripts.analysis.plot_pass20_cfg_event_justification import (
    EVENT_NUM_FRAMES,
    HORIZON,
    STRIDE,
    _align,
    _paired,
    _stat_line,
    collect_episodes,
)
from scripts.fold_glasses.discover_seedpair_branch_events import (
    load_global_zscore,
    normalize_actions,
)
from scripts.fold_glasses.validate_factual_replay import read_json


def load_video_metrics(replay_root: Path) -> dict[tuple[int, int], dict[str, Any]]:
    out: dict[tuple[int, int], dict[str, Any]] = {}
    for path in sorted(replay_root.glob("ep*_f*/metrics.json")):
        payload = read_json(path)
        if payload.get("status") != "complete":
            continue
        key = (int(payload["episode_index"]), int(payload["prefix_frame"]))
        out[key] = payload
    return out


def _action_prefix_table(
    *,
    data: dict[str, np.ndarray],
    z: np.ndarray,
) -> dict[tuple[int, int], dict[str, float]]:
    table: dict[tuple[int, int], dict[str, float]] = {}
    keys = np.unique(
        np.stack(
            [
                data["episode_index"].astype(np.int64),
                data["prefix_frame"].astype(np.int64),
            ],
            axis=1,
        ),
        axis=0,
    )
    for episode, frame in keys:
        idx = np.where(
            (data["episode_index"] == episode) & (data["prefix_frame"] == frame)
        )[0]
        metrics = branch_metrics(z[idx][:, :HORIZON], data["success"][idx])
        table[(int(episode), int(frame))] = metrics
    return table


def collect_video_rows(
    action_rows: list[dict[str, Any]],
    *,
    action_table: dict[tuple[int, int], dict[str, float]],
    video_table: dict[tuple[int, int], dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in action_rows:
        episode = int(row["episode"])
        frames = np.asarray(row["frames"], dtype=np.int32)
        s_v = np.full(frames.shape, np.nan)
        u_v = np.full(frames.shape, np.nan)
        su_v = np.full(frames.shape, np.nan)
        auc_v = np.full(frames.shape, np.nan)
        s_a = np.full(frames.shape, np.nan)
        u_a = np.full(frames.shape, np.nan)
        su_a = np.full(frames.shape, np.nan)
        auc_a = np.full(frames.shape, np.nan)
        missing = False
        for i, frame in enumerate(frames.tolist()):
            key = (episode, int(frame))
            action = action_table.get(key)
            video = video_table.get(key)
            if action is not None:
                s_a[i] = action["s"]
                u_a[i] = action["u"]
                su_a[i] = action["s_over_u"]
                auc_a[i] = action["auroc"]
            if video is not None:
                s_v[i] = video["s"]
                u_v[i] = video["u"]
                su_v[i] = video["s_over_u"]
                auc_v[i] = video["auroc"]
            elif int(frame) == int(row["t_star"]):
                missing = True
        if missing:
            continue
        before = frames < int(row["t_star"])
        star = frames == int(row["t_star"])
        if int(before.sum()) < 1 or not star.any():
            continue
        rows.append(
            {
                **row,
                "s_video": s_v,
                "u_video": u_v,
                "su_video": su_v,
                "auroc_video": auc_v,
                "s_action": s_a,
                "u_action": u_a,
                "su_action": su_a,
                "auroc_action": auc_a,
                "s_video_star": float(s_v[star][0]),
                "s_video_before": float(np.nanmedian(s_v[before])),
                "su_video_star": float(su_v[star][0]),
                "su_action_star": float(su_a[star][0]),
                "auroc_video_star": float(auc_v[star][0]),
                "auroc_action_star": float(auc_a[star][0]),
            }
        )
    return rows


def _curve(ax, taus, means, sems, *, color, label=None) -> None:
    ok = np.isfinite(means)
    ax.fill_between(
        taus[ok],
        means[ok] - sems[ok],
        means[ok] + sems[ok],
        color=color,
        alpha=0.16,
        linewidth=0,
        zorder=2,
    )
    ax.plot(taus[ok], means[ok], color=color, lw=2.15, zorder=3, solid_capstyle="round", label=label)
    ax.scatter(taus[ok], means[ok], s=18, c=color, zorder=4, edgecolors="white", linewidths=0.5)


def _stat_compare(stats: dict[str, Any], *, left: str, right: str) -> str:
    p_txt = (
        f"$p$={stats['wilcoxon_p']:.3g}" if stats.get("wilcoxon_p") is not None else ""
    )
    return (
        f"{stats['n_star_gt_before']}/{stats['n']} {right} $> $ {left}\n"
        f"median {stats['median_ratio']:.2f}×   {p_txt}"
    ).strip()


def plot_video_justification(rows: list[dict[str, Any]], *, output: Path) -> dict[str, Any]:
    apply_style()
    s_stats = _paired(rows, "s_video_before", "s_video_star")
    su_stats = _paired(rows, "su_action_star", "su_video_star")
    auc_stats = _paired(rows, "auroc_action_star", "auroc_video_star")
    taus = np.arange(-8.0, 4.0, 1.0)
    su_a_mean, su_a_sem = _align(rows, "su_action", taus)
    su_v_mean, su_v_sem = _align(rows, "su_video", taus)
    lo = -EVENT_NUM_FRAMES / STRIDE + 1.0

    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.4))
    ax = axes[0, 0]
    ax.axvspan(lo, 1.0, color=WINDOW, lw=0, zorder=0)
    ax.axvline(0.0, color="#c4b8b0", ls="--", lw=0.9, zorder=1)
    ax.axvline(1.0, color="#c4b8b0", ls=":", lw=0.9, zorder=1)
    _curve(ax, taus, su_a_mean, su_a_sem, color=PASS, label="action")
    _curve(ax, taus, su_v_mean, su_v_sem, color=EVENT, label="video")
    ax.set_xlabel(r"replan steps from $t^*$")
    ax.set_ylabel(r"between / within  ($S/U$)")
    ax.set_title(r"$S/U$ of paired futures", loc="left", pad=8)
    ax.set_ylim(0.0, None)
    polish(ax)
    panel_label(ax, "a")
    ax.legend(loc="upper left", handlelength=1.6)
    ax.text(
        (lo + 1.0) / 2.0,
        ax.get_ylim()[1] * 0.96 if ax.get_ylim()[1] > 0 else 1.0,
        "CFG window",
        ha="center",
        va="top",
        fontsize=8.2,
        color=MUTED,
    )

    if s_stats.get("n", 0):
        paired_boxes(axes[0, 1], s_stats["before"], s_stats["star"])
        y_hi = float(np.quantile(np.concatenate([s_stats["before"], s_stats["star"]]), 0.9)) * 1.45
        y_hi = max(y_hi, float(np.median(s_stats["star"])) * 2.2, 1e-3)
        axes[0, 1].set_ylim(0.0, y_hi)
        stat_note(axes[0, 1], _stat_line(s_stats))
    axes[0, 1].set_ylabel("video success vs fail RMS")
    axes[0, 1].set_title(r"$S_V$ at the event", loc="left", pad=8)
    panel_label(axes[0, 1], "b")

    if su_stats.get("n", 0):
        paired_boxes(
            axes[1, 0],
            su_stats["before"],
            su_stats["star"],
            labels=("action $S/U$", "video $S/U$"),
        )
        axes[1, 0].set_ylim(0.0, max(1.05, float(np.quantile(su_stats["star"], 0.95)) * 1.25))
        stat_note(axes[1, 0], _stat_compare(su_stats, left="action", right="video"))
    axes[1, 0].set_ylabel(r"between / within  ($S/U$)")
    axes[1, 0].set_title(r"At $t^*$: video modes are cleaner", loc="left", pad=8)
    panel_label(axes[1, 0], "c")

    if auc_stats.get("n", 0):
        paired_boxes(
            axes[1, 1],
            auc_stats["before"],
            auc_stats["star"],
            labels=("action", "video"),
        )
        axes[1, 1].axhline(0.5, color="#d6d3d1", ls="--", lw=1.0, zorder=1)
        axes[1, 1].set_ylim(0.35, 1.02)
        stat_note(axes[1, 1], _stat_compare(auc_stats, left="action", right="video"))
    axes[1, 1].set_ylabel("within-prefix AUROC")
    axes[1, 1].set_title(r"At $t^*$: video ranks the 20 takeovers", loc="left", pad=8)
    panel_label(axes[1, 1], "d")

    fig.suptitle(
        "Why crop the recoverability event for video CFG",
        fontsize=13.5,
        color=INK,
        y=0.995,
    )
    fig.text(
        0.5,
        -0.01,
        r"Video = 24 executed camera frames of the same Pass@20 first-action chunks. Overlay is dimensionless $S/U$.",
        ha="center",
        va="top",
        fontsize=8.4,
        color=MUTED,
    )
    fig.tight_layout(rect=(0.0, 0.02, 1.0, 0.96), w_pad=2.2, h_pad=2.4)
    save_figure(fig, output)

    def compact(stats: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in stats.items() if k not in {"before", "star"}}

    return {
        "png": str(output.resolve()),
        "pdf": str(output.with_suffix(".pdf").resolve()),
        "n_episodes": len(rows),
        "s_video": compact(s_stats),
        "s_over_u": compact(su_stats),
        "auroc": compact(auc_stats),
        "episodes": [
            {
                "episode": row["episode"],
                "t_star": row["t_star"],
                "s_video_star": row["s_video_star"],
                "s_video_before": row["s_video_before"],
                "su_video_star": row["su_video_star"],
                "su_action_star": row["su_action_star"],
                "auroc_video_star": row["auroc_video_star"],
                "auroc_action_star": row["auroc_action_star"],
            }
            for row in rows
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan-root", type=Path, required=True)
    parser.add_argument("--replay-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-stats", type=Path, default=DEFAULT_STATS)
    parser.add_argument("--min-replicates", type=int, default=20)
    parser.add_argument("--min-other", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    video_table = load_video_metrics(args.replay_root.expanduser().resolve())
    if not video_table:
        raise SystemExit(f"No video metrics under {args.replay_root}")
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
    action_rows = collect_episodes(
        data=data, z=z, metrics=metrics, min_other=int(args.min_other)
    )
    action_table = _action_prefix_table(data=data, z=z)
    rows = collect_video_rows(
        action_rows, action_table=action_table, video_table=video_table
    )
    if not rows:
        raise SystemExit("No episodes have a replayed video at $t^*$ yet")
    meta = plot_video_justification(rows, output=args.output.expanduser().resolve())
    meta["n_video_prefixes"] = len(video_table)
    print(json.dumps(meta, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
