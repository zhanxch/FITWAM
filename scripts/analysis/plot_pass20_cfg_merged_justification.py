#!/usr/bin/env python3
"""Merged CFG crop figure: recoverability, then action S and video S separately.

a  Pass@20 drops at the critical interval, so that is the CFG crop.
b  Action centroid RMS is larger at t* than at earlier prefixes.
c  Video centroid RMS, on its own scale, is larger at t* than earlier.
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
    panel_label,
    polish,
    save_figure,
)
from scripts.analysis.pass20_scan_data import DEFAULT_STATS, load_chunks, node_metrics
from scripts.analysis.plot_pass20_cfg_event_justification import (
    EVENT_NUM_FRAMES,
    STRIDE,
    _align,
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


def plot_merged(
    action_rows: list[dict[str, Any]],
    video_rows: list[dict[str, Any]],
    *,
    output: Path,
) -> dict[str, Any]:
    apply_style()
    action_stats = _paired(action_rows, "gap_before", "gap_star")
    video_stats = _paired(video_rows, "s_video_before", "s_video_star")
    taus = np.arange(-8.0, 4.0, 1.0)
    pass_mean, pass_sem = _align(action_rows, "pass_rate", taus)
    lo = -EVENT_NUM_FRAMES / STRIDE + 1.0

    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.2))
    ax_pass, ax_a, ax_v = axes

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
        ax_a,
        action_stats,
        ylabel="action success vs fail RMS",
        title=r"$S_A$ is larger at the critical event",
        letter="b",
        labels=("earlier", r"$S_A$"),
        event_color=PASS,
    )
    _paired_panel(
        ax_v,
        video_stats,
        ylabel="video success vs fail RMS",
        title=r"$S_V$ is larger at the critical event",
        letter="c",
        labels=("earlier", r"$S_V$"),
        event_color=EVENT,
    )

    fig.suptitle("Crop the recoverability event for CFG", fontsize=13.5, color=INK, y=1.02)
    fig.tight_layout(w_pad=2.2)
    save_figure(fig, output)

    def compact(stats: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in stats.items() if k not in {"before", "star"}}

    return {
        "png": str(output.resolve()),
        "pdf": str(output.with_suffix(".pdf").resolve()),
        "n_action_episodes": len(action_rows),
        "n_video_episodes": len(video_rows),
        "action_s": compact(action_stats),
        "video_s": compact(video_stats),
        "episodes": [
            {
                "episode": row["episode"],
                "t_star": row["t_star"],
                "s_action_earlier": row["gap_before"],
                "s_action": row["gap_star"],
                "s_video_earlier": row["s_video_before"],
                "s_video": row["s_video_star"],
            }
            for row in video_rows
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
    if not action_rows:
        raise SystemExit("No completed recoverability events")
    action_table = _action_prefix_table(data=data, z=z)
    video_table = load_video_metrics(args.replay_root.expanduser().resolve())
    video_rows = collect_video_rows(
        action_rows, action_table=action_table, video_table=video_table
    )
    if not video_rows:
        raise SystemExit("No episodes have a replayed video at $t^*$ yet")
    meta = plot_merged(
        action_rows,
        video_rows,
        output=args.output.expanduser().resolve(),
    )
    meta["n_video_prefixes"] = len(video_table)
    print(json.dumps(meta, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
