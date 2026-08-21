#!/usr/bin/env python3
"""Schematic of the *expected* video-CFG result, if the hypothesis holds.

Synthetic data only. Layout mirrors the action justification figure:
the critical interval is special, and video separates success/fail futures
more cleanly than action on the same 20 takeovers.
"""

from __future__ import annotations

import sys
from pathlib import Path

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

OUTPUT = (
    ROOT
    / "data"
    / "fold_glasses_opensource_s0_collect_4x50_20260812_112113"
    / "pass20_action_chunk_analysis_20260819"
    / "figures"
    / "fig_pass20_cfg_video_schematic_expected.png"
)


def _paired_draw(rng: np.random.Generator, n: int, loc_before: float, loc_star: float, scale: float) -> tuple[np.ndarray, np.ndarray]:
    before = np.clip(rng.normal(loc_before, scale, n), 0.02, None)
    star = np.clip(before * (loc_star / loc_before) + rng.normal(0.0, scale * 0.7, n), before + 0.02, None)
    return before, star


def main() -> int:
    rng = np.random.default_rng(7)
    n = 17
    taus = np.arange(-8.0, 4.0, 1.0)
    lo = -33 / 24 + 1.0

    # Dimensionless S/U over time: overlay is valid. Action has a modest peak
    # (as in the real justification); video stays flat then jumps at t*.
    su_a_curve = np.array([0.28, 0.41, 0.36, 0.27, 0.26, 0.25, 0.32, 0.29, 0.48, 0.42, 0.35, 0.26])
    su_a_sem = su_a_curve * 0.16
    su_v_curve = np.array([0.22, 0.23, 0.21, 0.22, 0.24, 0.25, 0.27, 0.30, 0.78, 0.69, 0.48, 0.31])
    su_v_sem = su_v_curve * 0.14

    s_v_before, s_v_star = _paired_draw(rng, n, 0.024, 0.110, 0.008)
    su_a = np.clip(rng.normal(0.42, 0.08, n), 0.15, 0.7)
    su_v = np.clip(su_a + rng.normal(0.28, 0.07, n), su_a + 0.08, 1.15)
    auc_a = np.clip(rng.normal(0.56, 0.05, n), 0.48, 0.70)
    auc_v = np.clip(auc_a + rng.normal(0.18, 0.05, n), auc_a + 0.06, 0.92)

    apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.4))

    def _curve(ax, means, sems, *, color, ylabel, title, letter, ylim=None, label=None):
        ax.axvspan(lo, 1.0, color=WINDOW, lw=0, zorder=0)
        ax.axvline(0.0, color="#c4b8b0", ls="--", lw=0.9, zorder=1)
        ax.axvline(1.0, color="#c4b8b0", ls=":", lw=0.9, zorder=1)
        ax.fill_between(taus, means - sems, means + sems, color=color, alpha=0.16, linewidth=0, zorder=2)
        ax.plot(taus, means, color=color, lw=2.15, zorder=3, solid_capstyle="round", label=label)
        ax.scatter(taus, means, s=18, c=color, zorder=4, edgecolors="white", linewidths=0.5)
        ax.set_xlabel(r"replan steps from $t^*$")
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", pad=8)
        polish(ax)
        panel_label(ax, letter)
        if ylim is not None:
            ax.set_ylim(*ylim)

    _curve(
        axes[0, 0],
        su_a_curve,
        su_a_sem,
        color=PASS,
        ylabel=r"between / within  ($S/U$)",
        title=r"$S/U$ of paired futures",
        letter="a",
        label="action",
        ylim=(0.0, 1.05),
    )
    axes[0, 0].fill_between(
        taus, su_v_curve - su_v_sem, su_v_curve + su_v_sem,
        color=EVENT, alpha=0.16, linewidth=0, zorder=2,
    )
    axes[0, 0].plot(taus, su_v_curve, color=EVENT, lw=2.15, zorder=3, solid_capstyle="round", label="video")
    axes[0, 0].scatter(taus, su_v_curve, s=18, c=EVENT, zorder=4, edgecolors="white", linewidths=0.5)
    axes[0, 0].legend(loc="upper left", handlelength=1.6)
    axes[0, 0].text(
        (lo + 1.0) / 2.0,
        0.98,
        "CFG window",
        ha="center",
        va="top",
        fontsize=8.2,
        color=MUTED,
    )

    paired_boxes(
        axes[0, 1],
        s_v_before,
        s_v_star,
        labels=("earlier", r"$t^*$"),
    )
    axes[0, 1].set_ylim(0.0, float(np.quantile(np.concatenate([s_v_before, s_v_star]), 0.95)) * 1.35)
    axes[0, 1].set_ylabel("video success vs fail RMS")
    axes[0, 1].set_title(r"$S_V$ at the event  (video can CFG)", loc="left", pad=8)
    panel_label(axes[0, 1], "b")
    stat_note(axes[0, 1], rf"{n}/{n} larger at $t^*$" + "\nmedian 4.6×")

    paired_boxes(
        axes[1, 0],
        su_a,
        su_v,
        labels=("action $S/U$", "video $S/U$"),
        event_color=EVENT,
    )
    axes[1, 0].set_ylim(0.0, 1.25)
    axes[1, 0].set_ylabel(r"between / within  ($S/U$)")
    axes[1, 0].set_title(r"At $t^*$: video modes are cleaner", loc="left", pad=8)
    panel_label(axes[1, 0], "c")
    stat_note(axes[1, 0], rf"{n}/{n} video $> $ action" + "\nmedian 1.7×")

    paired_boxes(
        axes[1, 1],
        auc_a,
        auc_v,
        labels=("action", "video"),
        event_color=EVENT,
    )
    axes[1, 1].axhline(0.5, color="#d6d3d1", ls="--", lw=1.0, zorder=1)
    axes[1, 1].set_ylim(0.35, 1.02)
    axes[1, 1].set_ylabel("within-prefix AUROC")
    axes[1, 1].set_title(r"At $t^*$: video ranks the 20 takeovers", loc="left", pad=8)
    panel_label(axes[1, 1], "d")
    stat_note(axes[1, 1], "action 0.56   video 0.74")

    fig.suptitle(
        "Schematic (synthetic): expected pattern if video CFG should be cropped at the event, and is better than action",
        fontsize=12.2,
        color=INK,
        y=0.995,
    )
    fig.text(
        0.5,
        -0.01,
        "Not real data. Overlay uses dimensionless $S/U$ (raw $S_A$ and $S_V$ have different units). Video = executed 24-frame futures on the same 20 takeovers.",
        ha="center",
        va="top",
        fontsize=8.4,
        color=MUTED,
    )
    fig.tight_layout(rect=(0.0, 0.02, 1.0, 0.96), w_pad=2.2, h_pad=2.4)
    save_figure(fig, OUTPUT)
    print(OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
