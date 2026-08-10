#!/usr/bin/env python3
"""Side-by-side S0 | B1 open-loop interval: WIDTH ONLY (centered ribbons).

No absolute level / residual / GT. Each panel shows a symmetric band
  ± w(t) about 0, where w(t)=||σ(a0)||₂,
so the only visual is how thick the ribbon is along progress.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIR = ROOT / "results/openloop_action_interval_expert_ep22_20260808"
DEFAULT_STEM = DEFAULT_DIR / "fig_openloop_action_interval_S0_vs_B1"

C_S0 = "#4C78A8"
C_B1 = "#54A24B"
C_TEXT = "#1F2A33"
C_MUTED = "#5B6B7A"
C_GRID = "#E8EBEE"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", type=Path, default=DEFAULT_DIR)
    p.add_argument("--output-stem", type=Path, default=DEFAULT_STEM)
    p.add_argument("--dpi", type=int, default=300)
    return p.parse_args()


def style_ax(ax: plt.Axes) -> None:
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#C5CCD3")
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(labelsize=7.5, colors=C_MUTED, length=2.5, pad=1.5)
    ax.grid(True, axis="x", color=C_GRID, linewidth=0.65, alpha=0.9)
    ax.axhline(0.0, color="#D0D5DB", lw=0.8, zorder=1)
    ax.set_axisbelow(True)


def panel(ax: plt.Axes, progress: np.ndarray, w: np.ndarray, *, color: str, title: str) -> None:
    # ribbon thickness = 2w; only thickness matters
    ax.fill_between(progress, -w, w, color=color, alpha=0.28, linewidth=0, zorder=2)
    ax.plot(progress, w, color=color, lw=1.5, zorder=3)
    ax.plot(progress, -w, color=color, lw=1.5, zorder=3)
    style_ax(ax)
    ax.set_title(title, fontsize=9.5, color=color, loc="left", fontweight="semibold", pad=5)
    ax.set_xlabel("episode progress", fontsize=8, color=C_MUTED, labelpad=2)
    ax.set_xlim(0.0, 1.0)
    # hide y tick labels — scale is shared for compare, but values are ±width
    ax.set_yticks([])


def main() -> None:
    args = parse_args()
    ddir = args.input_dir
    z = np.load(ddir / "openloop_interval.npz")
    progress = np.asarray(z["progress"], dtype=np.float64)
    w_s0 = np.asarray(z["S0_sigma_l2"], dtype=np.float64)
    w_b1 = np.asarray(z["B1_sigma_l2"], dtype=np.float64)
    # optional outer envelope from sample radius p90
    r_s0 = np.asarray(z["S0_radius_p90"], dtype=np.float64)
    r_b1 = np.asarray(z["B1_radius_p90"], dtype=np.float64)

    source = "expert ep22 (water_plant_fastwam)"
    report_path = ddir / "openloop_report.json"
    if report_path.exists():
        rep = json.loads(report_path.read_text())
        source = rep.get("episode", {}).get("source_label", source)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.2), dpi=args.dpi, sharey=True)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.78, bottom=0.18, wspace=0.08)

    fig.suptitle(
        "Open-loop action-interval width on a fixed episode",
        fontsize=12,
        fontweight="bold",
        color=C_TEXT,
        y=0.98,
        x=0.52,
    )
    fig.text(
        0.52,
        0.905,
        f"{source}  ·  {int(z['n_frames_full'])} steps  ·  "
        f"K={int(z['num_samples'])} samples/frame  ·  stride={int(z['stride'])}  ·  "
        r"ribbon = $\pm\|\sigma(a_0)\|_2$ (centered; only thickness matters)",
        ha="center",
        fontsize=7.6,
        color=C_MUTED,
    )

    # faint outer envelope (p90 radius) — still width, no level
    for ax, r, color in [(axes[0], r_s0, C_S0), (axes[1], r_b1, C_B1)]:
        ax.fill_between(progress, -r, r, color=color, alpha=0.10, linewidth=0, zorder=1)

    panel(axes[0], progress, w_s0, color=C_S0, title="Expert-only Baseline (S0)")
    panel(axes[1], progress, w_b1, color=C_B1, title="Rollout-Retrained (B1-remap-cfg)")

    ymax = max(float(np.nanmax(r_s0)), float(np.nanmax(r_b1)), float(np.nanmax(w_s0)), float(np.nanmax(w_b1)))
    for ax in axes:
        ax.set_ylim(-ymax * 1.08, ymax * 1.08)

    # single shared width legend cue on left
    axes[0].text(
        0.02,
        0.95,
        "thicker = wider action samples",
        transform=axes[0].transAxes,
        ha="left",
        va="top",
        fontsize=6.8,
        color=C_MUTED,
    )
    axes[1].text(
        0.02,
        0.95,
        "thicker = wider action samples",
        transform=axes[1].transAxes,
        ha="left",
        va="top",
        fontsize=6.8,
        color=C_MUTED,
    )

    fig.text(
        0.52,
        0.04,
        "Same y-scale on both panels. Dark ribbon: ±‖σ(a₀)‖₂ across K open-loop seeds; "
        "light ribbon: ± sample-radius p90. Absolute action level is removed — compare thickness only.",
        ha="center",
        fontsize=6.5,
        color=C_MUTED,
    )

    stem = args.output_stem
    fig.savefig(f"{stem}.png", dpi=args.dpi)
    fig.savefig(f"{stem}.pdf")
    plt.close(fig)
    print(f"[done] {stem}.png", flush=True)


if __name__ == "__main__":
    main()
