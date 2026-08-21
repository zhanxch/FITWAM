#!/usr/bin/env python3
"""Why video CFG rather than ReCap: both use failures, different interfaces.

Not a classifier figure. CFG needs two generative modes in the space that is
denoised. At the recoverability event those modes exist in the future world
(S_V), not as an AUROC over the 20 takeovers.

a  Same failed transition (o, a, v). ReCap compresses the world fork into a
   scalar on overlapping actions. Video CFG conditions the world itself.
b  On video's own scale, success/fail futures separate at t*. That is the
   mode video CFG can steer; it is not compared to action RMS.
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
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analysis._plot_style import (
    EVENT,
    INK,
    MUTED,
    PASS,
    apply_style,
    panel_label,
    polish,
    save_figure,
    stat_note,
)
from scripts.analysis.pass20_scan_data import DEFAULT_STATS, load_chunks, node_metrics
from scripts.analysis.plot_pass20_cfg_event_justification import (
    _paired,
    _paired_panel,
    collect_episodes,
)
from scripts.analysis.plot_pass20_cfg_video_justification import (
    _action_prefix_table,
    collect_video_rows,
    load_video_metrics,
)
from scripts.fold_glasses.discover_seedpair_branch_events import (
    load_global_zscore,
    normalize_actions,
)

DEFAULT_SCAN = (
    ROOT
    / "data"
    / "fold_glasses_opensource_s0_collect_4x50_20260812_112113"
    / "pass20_action_chunk_analysis_20260819"
)
DEFAULT_OUTPUT = DEFAULT_SCAN / "figures" / "fig_pass20_video_cfg_interface.png"


def _box(ax, xy, w, h, text, *, facecolor, edgecolor, fontsize=9.2) -> None:
    patch = FancyBboxPatch(
        xy,
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=1.15,
        zorder=3,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + w / 2.0,
        xy[1] + h / 2.0,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=INK,
        zorder=4,
        linespacing=1.25,
    )


def _arrow(ax, start, end, *, color, lw=1.35) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=9.5,
            lw=lw,
            color=color,
            zorder=2,
        )
    )


def draw_interface(ax) -> None:
    ax.set_xlim(0.0, 10.0)
    ax.set_ylim(0.0, 6.2)
    ax.axis("off")
    ax.set_title("Both methods use the failed transition; the statistic differs", loc="left", pad=8)

    _box(ax, (0.25, 2.55), 1.35, 1.1, r"$o$", facecolor="#f5f5f4", edgecolor="#a8a29e")
    _box(ax, (2.45, 2.55), 1.55, 1.1, r"$a$", facecolor="#d7e4ec", edgecolor=PASS)
    _box(ax, (5.05, 2.55), 1.55, 1.1, r"$v=T(o,a)$", facecolor="#f4d6ce", edgecolor=EVENT)
    _box(ax, (7.65, 2.55), 1.55, 1.1, r"$Y=g(v)$", facecolor="#f5f5f4", edgecolor="#a8a29e")
    _arrow(ax, (1.60, 3.10), (2.45, 3.10), color="#a8a29e")
    _arrow(ax, (4.00, 3.10), (5.05, 3.10), color="#a8a29e")
    _arrow(ax, (6.60, 3.10), (7.65, 3.10), color="#a8a29e")
    ax.text(2.02, 3.42, r"$\pi$", ha="center", fontsize=8.2, color=MUTED)
    ax.text(4.52, 3.42, r"$T$", ha="center", fontsize=8.2, color=MUTED)
    ax.text(7.12, 3.42, r"$g$", ha="center", fontsize=8.2, color=MUTED)

    _box(
        ax,
        (2.15, 0.35),
        2.15,
        1.35,
        "ReCap\n" r"scalar $\hat A$ on $a$",
        facecolor="#eef3f6",
        edgecolor=PASS,
        fontsize=8.6,
    )
    _box(
        ax,
        (5.15, 4.35),
        2.35,
        1.35,
        "video CFG\n" r"modes of $v$",
        facecolor="#fbf0ec",
        edgecolor=EVENT,
        fontsize=8.6,
    )
    _arrow(ax, (3.22, 1.70), (3.22, 2.55), color=PASS, lw=1.5)
    _arrow(ax, (6.32, 4.35), (6.32, 3.65), color=EVENT, lw=1.5)
    ax.text(
        0.25,
        5.85,
        r"knife-edge: nearby $a$  $\rightarrow$  distinct $v$",
        ha="left",
        va="top",
        fontsize=8.2,
        color=MUTED,
    )
    panel_label(ax, "a")


def plot_interface(
    video_rows: list[dict[str, Any]],
    *,
    output: Path,
) -> dict[str, Any]:
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.35), gridspec_kw={"width_ratios": [1.15, 1.0]})
    draw_interface(axes[0])
    video_stats = _paired(video_rows, "s_video_before", "s_video_star")
    _paired_panel(
        axes[1],
        video_stats,
        ylabel="video success vs fail RMS",
        title=r"World modes appear at $t^*$",
        letter="b",
        labels=("earlier", r"$S_V$"),
        event_color=EVENT,
    )
    fig.suptitle(
        "Failure data enters the world, not a scalar on overlapping actions",
        fontsize=13.2,
        color=INK,
        y=1.03,
    )
    fig.tight_layout(w_pad=2.4)
    save_figure(fig, output)
    def compact(stats: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in stats.items() if k not in {"before", "star"}}

    return {
        "png": str(output.resolve()),
        "n_video_episodes": len(video_rows),
        "video_s": compact(video_stats),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan-root", type=Path, default=DEFAULT_SCAN)
    parser.add_argument("--replay-root", type=Path, default=DEFAULT_SCAN / "video_replay")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
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
    action_rows = collect_episodes(
        data=data, z=z, metrics=metrics, min_other=int(args.min_other)
    )
    action_table = _action_prefix_table(data=data, z=z)
    video_table = load_video_metrics(args.replay_root.expanduser().resolve())
    video_rows = collect_video_rows(
        action_rows, action_table=action_table, video_table=video_table
    )
    if not video_rows:
        raise SystemExit("No replayed video at t*")
    meta = plot_interface(video_rows, output=args.output.expanduser().resolve())
    print(json.dumps(meta, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
