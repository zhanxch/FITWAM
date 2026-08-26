#!/usr/bin/env python3
"""AlphaGo-style eval of a failed fold-glasses episode.

Pass@20 from each replan prefix is the Monte Carlo win rate V^π(s).
The factual failure is the played game. The K continuations are the
search tree. The CFG crop is the blunder: the chunk that collapsed
win rate to zero.

This is a diagnostic figure, not a new training recipe.
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
from matplotlib.patches import FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analysis._plot_style import (
    EVENT,
    INK,
    MUTED,
    PASS,
    STONE,
    WINDOW,
    apply_style,
    panel_label,
    polish,
    save_figure,
)
from scripts.analysis.pass20_scan_data import DEFAULT_STATS, load_chunks, node_metrics
from scripts.analysis.plot_pass20_cfg_event_justification import (
    STRIDE,
    _align,
    collect_episodes,
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
DEFAULT_OUTPUT = DEFAULT_SCAN / "figures" / "fig_pass20_alphago_eval.png"
DEFAULT_EPISODE = 113


def _row_for_episode(rows: list[dict[str, Any]], episode: int) -> dict[str, Any]:
    for row in rows:
        if int(row["episode"]) == int(episode):
            return row
    available = ", ".join(str(int(r["episode"])) for r in rows)
    raise SystemExit(f"Episode {episode} has no recoverability event. Have: {available}")


def _winrate_panel(ax, row: dict[str, Any]) -> None:
    frames = np.asarray(row["frames"], dtype=np.int32)
    v = np.asarray(row["pass_rate"], dtype=np.float64)
    n_ok = np.asarray(row["n_success"], dtype=np.int32)
    moves = np.arange(len(frames), dtype=np.float64)
    t_star = int(row["t_star"])
    t_zero = int(row["t_zero"])
    i_star = int(np.where(frames == t_star)[0][0])
    i_zero = int(np.where(frames == t_zero)[0][0])
    drop = float(v[i_star] - v[i_zero])

    ax.axhspan(0.5, 1.02, color="#eef4f7", lw=0, zorder=0)
    ax.axhspan(-0.02, 0.5, color="#f7f1ef", lw=0, zorder=0)
    ax.axhline(0.5, color=STONE, ls="--", lw=0.9, zorder=1)
    ax.axvspan(i_star - 0.5, i_zero + 0.35, color=WINDOW, lw=0, zorder=1)
    ax.axvspan(i_zero + 0.35, moves[-1] + 0.6, color="#f4f4f3", lw=0, zorder=1)

    ax.fill_between(moves, 0.5, v, where=v >= 0.5, color=PASS, alpha=0.22, lw=0, zorder=2)
    ax.fill_between(moves, v, 0.5, where=v < 0.5, color=EVENT, alpha=0.20, lw=0, zorder=2)
    ax.plot(moves, v, color=INK, lw=2.35, zorder=4, solid_capstyle="round")
    ax.scatter(
        moves,
        v,
        s=28,
        c=[PASS if x >= 0.5 else EVENT for x in v],
        zorder=5,
        edgecolors="white",
        linewidths=0.7,
    )

    ax.annotate(
        f"blunder  −{drop:.0%} win rate",
        xy=(i_zero, float(v[i_zero])),
        xytext=(i_zero + 0.15, 0.78),
        color=EVENT,
        fontsize=8.4,
        fontweight="medium",
        arrowprops={
            "arrowstyle": "-|>",
            "color": EVENT,
            "lw": 1.0,
            "shrinkA": 0,
            "shrinkB": 4,
        },
        zorder=6,
    )
    ax.text(
        i_star,
        1.0,
        r"$t^*$",
        ha="center",
        va="bottom",
        fontsize=8.4,
        color=MUTED,
    )
    ax.text(
        (i_zero + moves[-1]) / 2.0,
        0.14,
        "resign / prune",
        ha="center",
        va="bottom",
        fontsize=8.0,
        color=MUTED,
    )
    ax.text(
        -0.15,
        0.96,
        "ahead",
        ha="right",
        va="top",
        fontsize=7.6,
        color=PASS,
        transform=ax.get_yaxis_transform(),
        clip_on=False,
    )
    ax.text(
        -0.15,
        0.04,
        "behind",
        ha="right",
        va="bottom",
        fontsize=7.6,
        color=EVENT,
        transform=ax.get_yaxis_transform(),
        clip_on=False,
    )

    ax.set_xlim(-0.55, moves[-1] + 0.65)
    ax.set_ylim(-0.04, 1.06)
    ax.set_yticks([0.0, 0.5, 1.0], ["0%", "50%", "100%"])
    ax.set_xticks(moves, [str(int(f)) for f in frames], fontsize=7.6)
    ax.set_xlabel("prefix frame (replan grid, one 'move' = 24 executed actions)")
    ax.set_ylabel(r"win rate  $V^\pi$  (Pass@20)")
    ax.set_title(
        f"Episode {int(row['episode'])}: played game vs search eval",
        loc="left",
        pad=8,
    )
    polish(ax)
    panel_label(ax, "a")
    ax.text(
        0.99,
        0.90,
        f"search at $t^*$: {int(n_ok[i_star])}/20 recover",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.0,
        color=MUTED,
    )


def _trial_row(ax, x0: float, y0: float, n_ok: int, n_fail: int, size: float = 0.22) -> None:
    total = int(n_ok + n_fail)
    for i in range(total):
        color = PASS if i < int(n_ok) else EVENT
        ax.add_patch(
            Rectangle(
                (x0 + i * (size + 0.04), y0),
                size,
                size,
                facecolor=color,
                edgecolor="white",
                linewidth=0.4,
                zorder=3,
            )
        )


def _search_panel(ax, row: dict[str, Any]) -> None:
    frames = np.asarray(row["frames"], dtype=np.int32)
    v = np.asarray(row["pass_rate"], dtype=np.float64)
    n_ok = np.asarray(row["n_success"], dtype=np.int32)
    n_fail = np.asarray(row["n_fail"], dtype=np.int32)
    t_star = int(row["t_star"])
    i_star = int(np.where(frames == t_star)[0][0])
    indices = [i_star - 1, i_star, i_star + 1]
    labels = [r"$t^*-24$", r"$t^*$  fork", r"$t^*+24$  dead"]
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4.2)
    ax.axis("off")
    ax.set_title("Search tree at three consecutive moves", loc="left", pad=8)
    panel_label(ax, "b")

    for col, idx, label in zip(range(3), indices, labels):
        x = 0.45 + col * 4.0
        ok = int(n_ok[idx])
        fail = int(n_fail[idx])
        rate = float(v[idx])
        root_color = PASS if rate >= 0.5 else EVENT
        circle = plt.Circle((x + 1.55, 3.35), 0.38, facecolor=root_color, edgecolor=INK, lw=0.7, zorder=4)
        ax.add_patch(circle)
        ax.text(
            x + 1.55,
            3.35,
            f"{rate:.0%}",
            ha="center",
            va="center",
            fontsize=8.2,
            color="white",
            fontweight="medium",
            zorder=5,
        )
        ax.text(x + 1.55, 3.88, label, ha="center", va="bottom", fontsize=8.0, color=INK)
        ax.plot([x + 1.15, x + 0.55], [3.05, 2.45], color=STONE, lw=0.9, zorder=2)
        ax.plot([x + 1.95, x + 2.55], [3.05, 2.45], color=STONE, lw=0.9, zorder=2)
        ax.add_patch(
            FancyBboxPatch(
                (x + 0.15, 1.85),
                0.95,
                0.48,
                boxstyle="round,pad=0.04,rounding_size=0.08",
                facecolor="#d7e4ec",
                edgecolor=PASS,
                lw=0.8,
                zorder=3,
            )
        )
        ax.add_patch(
            FancyBboxPatch(
                (x + 2.05, 1.85),
                0.95,
                0.48,
                boxstyle="round,pad=0.04,rounding_size=0.08",
                facecolor="#f4d6ce",
                edgecolor=EVENT,
                lw=0.8,
                zorder=3,
            )
        )
        ax.text(x + 0.62, 2.09, f"a+  {ok}", ha="center", va="center", fontsize=7.6, color=PASS, zorder=4)
        ax.text(x + 2.52, 2.09, f"a−  {fail}", ha="center", va="center", fontsize=7.6, color=EVENT, zorder=4)
        _trial_row(ax, x + 0.12, 1.35, ok, fail, size=0.13)
        if col == 1:
            ax.text(x + 0.62, 1.08, "PV / CFG +", ha="center", va="top", fontsize=7.2, color=PASS)
            ax.text(x + 2.52, 1.08, "played", ha="center", va="top", fontsize=7.2, color=EVENT)
        elif col == 2:
            ax.text(x + 1.55, 1.08, "no recovery branch — resign", ha="center", va="top", fontsize=7.2, color=MUTED)
        else:
            ax.text(
                x + 1.55,
                1.08,
                "still mixed, not the crop",
                ha="center",
                va="top",
                fontsize=7.2,
                color=MUTED,
            )
    ax.text(
        0.0,
        0.28,
        "Each square is one closed-loop continuation (Pass@20). "
        "CFG uses the fork where both children exist; it does not search the dead node.",
        fontsize=7.8,
        color=MUTED,
        wrap=True,
    )


def _aggregate_panel(ax, rows: list[dict[str, Any]]) -> None:
    taus = np.arange(-8.0, 4.0, 1.0)
    mean, sem = _align(rows, "pass_rate", taus)
    ok = np.isfinite(mean)
    ax.axhspan(0.5, 1.02, color="#eef4f7", lw=0, zorder=0)
    ax.axhspan(-0.02, 0.5, color="#f7f1ef", lw=0, zorder=0)
    ax.axhline(0.5, color=STONE, ls="--", lw=0.9, zorder=1)
    ax.axvline(0.0, color="#c4b8b0", ls="--", lw=0.9, zorder=1)
    ax.axvline(1.0, color="#c4b8b0", ls=":", lw=0.9, zorder=1)
    ax.fill_between(taus[ok], 0.5, mean[ok], where=mean[ok] >= 0.5, color=PASS, alpha=0.20, lw=0, zorder=2)
    ax.fill_between(taus[ok], mean[ok], 0.5, where=mean[ok] < 0.5, color=EVENT, alpha=0.18, lw=0, zorder=2)
    ax.fill_between(
        taus[ok],
        mean[ok] - sem[ok],
        mean[ok] + sem[ok],
        color=INK,
        alpha=0.10,
        lw=0,
        zorder=3,
    )
    ax.plot(taus[ok], mean[ok], color=INK, lw=2.1, zorder=4, solid_capstyle="round")
    ax.scatter(taus[ok], mean[ok], s=18, c=INK, zorder=5, edgecolors="white", linewidths=0.5)
    ax.set_xlim(-8.4, 3.4)
    ax.set_ylim(-0.04, 1.06)
    ax.set_yticks([0.0, 0.5, 1.0], ["0%", "50%", "100%"])
    ax.set_xlabel(r"replan steps from $t^*$")
    ax.set_ylabel(r"mean $V^\pi$")
    ax.set_title(f"26 failed games aligned at the blunder", loc="left", pad=8)
    polish(ax)
    panel_label(ax, "c")
    ax.text(
        -4.0,
        0.92,
        "ahead: keep S0",
        ha="center",
        fontsize=7.8,
        color=PASS,
    )
    ax.text(
        0.5,
        0.92,
        "close: CFG",
        ha="center",
        fontsize=7.8,
        color=EVENT,
    )
    ax.text(
        2.2,
        0.12,
        "resign",
        ha="center",
        fontsize=7.8,
        color=MUTED,
    )


def plot_alphago(
    *,
    row: dict[str, Any],
    rows: list[dict[str, Any]],
    output: Path,
) -> dict[str, Any]:
    apply_style()
    fig = plt.figure(figsize=(12.6, 8.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0], hspace=0.38, wspace=0.28)
    ax_win = fig.add_subplot(gs[0, :])
    ax_search = fig.add_subplot(gs[1, 0])
    ax_agg = fig.add_subplot(gs[1, 1])
    _winrate_panel(ax_win, row)
    _search_panel(ax_search, row)
    _aggregate_panel(ax_agg, rows)
    fig.suptitle(
        r"Pass@20 is AlphaGo's win-rate graph: search eval, not the played return",
        fontsize=13.4,
        color=INK,
        y=1.02,
    )
    save_figure(fig, output)
    frames = np.asarray(row["frames"])
    v = np.asarray(row["pass_rate"])
    i_star = int(np.where(frames == int(row["t_star"]))[0][0])
    i_zero = int(np.where(frames == int(row["t_zero"]))[0][0])
    return {
        "png": str(output.resolve()),
        "pdf": str(output.with_suffix(".pdf").resolve()),
        "episode": int(row["episode"]),
        "t_star": int(row["t_star"]),
        "t_zero": int(row["t_zero"]),
        "win_rate_at_tstar": float(v[i_star]),
        "blunder_drop": float(v[i_star] - v[i_zero]),
        "n_games_in_aggregate": len(rows),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan-root", type=Path, default=DEFAULT_SCAN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dataset-stats", type=Path, default=DEFAULT_STATS)
    parser.add_argument("--episode", type=int, default=DEFAULT_EPISODE)
    parser.add_argument("--min-replicates", type=int, default=20)
    parser.add_argument("--min-other", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scan = args.scan_root.expanduser().resolve()
    data = load_chunks(scan)
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
        raise SystemExit("No recoverability events")
    row = _row_for_episode(rows, int(args.episode))
    output = args.output.expanduser().resolve()
    meta = plot_alphago(row=row, rows=rows, output=output)
    output.with_suffix(".json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(meta, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
