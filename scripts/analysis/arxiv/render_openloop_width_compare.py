#!/usr/bin/env python3
"""Clean figure: action-interval WIDTH over progress (S0 vs B1).

Story
-----
S0's open-loop sample width is relatively high and uniform along the episode;
B1 is overall tighter and especially narrow in some progress regions.

Reads:
  results/openloop_action_interval_expert_ep22_20260808/openloop_interval.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_NPZ = ROOT / "results/openloop_action_interval_expert_ep22_20260808/openloop_interval.npz"
DEFAULT_STEM = ROOT / "results/openloop_action_interval_expert_ep22_20260808/fig_openloop_width_S0_vs_B1"

C_S0 = "#4C78A8"
C_B1 = "#54A24B"
C_RATIO = "#F58518"
C_TEXT = "#1F2A33"
C_MUTED = "#5B6B7A"
C_GRID = "#E6E9ED"
C_PANEL = "#F7F8FA"
C_SHADE = "#E8F0E8"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--npz", type=Path, default=DEFAULT_NPZ)
    p.add_argument("--output-stem", type=Path, default=DEFAULT_STEM)
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--early-cut", type=float, default=0.05, help="mark / soft-focus before this progress")
    return p.parse_args()


def style_ax(ax: plt.Axes) -> None:
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#C5CCD3")
        ax.spines[side].set_linewidth(0.85)
    ax.tick_params(labelsize=8, colors=C_MUTED, length=3, pad=2)
    ax.grid(True, which="major", color=C_GRID, linewidth=0.7, alpha=0.95)
    ax.set_axisbelow(True)


def smooth(y: np.ndarray, win: int = 3) -> np.ndarray:
    if win <= 1 or len(y) < win:
        return y
    k = np.ones(win, dtype=np.float64) / win
    pad = win // 2
    yp = np.pad(y.astype(np.float64), (pad, pad), mode="edge")
    return np.convolve(yp, k, mode="valid")


def main() -> None:
    args = parse_args()
    z = np.load(args.npz, allow_pickle=True)
    p = np.asarray(z["progress"], dtype=np.float64)
    s0 = np.asarray(z["S0_sigma_l2"], dtype=np.float64)
    b1 = np.asarray(z["B1_sigma_l2"], dtype=np.float64)
    # secondary width: p90-p10 of ||a-μ|| radius
    s0_band = np.asarray(z["S0_radius_p90"] - z["S0_radius_p10"], dtype=np.float64)
    b1_band = np.asarray(z["B1_radius_p90"] - z["B1_radius_p10"], dtype=np.float64)

    mid = p >= args.early_cut
    s0_mid_mean = float(s0[mid].mean())
    b1_mid_mean = float(b1[mid].mean())
    s0_mid_cv = float(s0[mid].std() / max(s0_mid_mean, 1e-12))
    b1_mid_cv = float(b1[mid].std() / max(b1_mid_mean, 1e-12))
    ratio = s0 / np.maximum(b1, 1e-8)
    ratio_mid_mean = float(ratio[mid].mean())

    # highlight where B1 is most compressed relative to S0 (mid trajectory)
    ratio_s = smooth(ratio, 3)
    # top 25% ratio in mid region → "B1 narrower"
    thr = float(np.quantile(ratio[mid], 0.70))
    narrow = mid & (ratio >= thr)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig = plt.figure(figsize=(9.6, 6.6), dpi=args.dpi)
    gs = GridSpec(
        2,
        1,
        figure=fig,
        height_ratios=[1.55, 1.0],
        hspace=0.32,
        left=0.10,
        right=0.97,
        top=0.86,
        bottom=0.10,
    )

    fig.text(
        0.10,
        0.945,
        "Open-loop action-interval width along an expert trajectory",
        fontsize=13.5,
        fontweight="bold",
        color=C_TEXT,
    )
    fig.text(
        0.10,
        0.905,
        "expert ep22  ·  K=8 samples/frame  ·  "
        r"$w(t)=\|\sigma(a_0)\|_2$ from multi-seed open-loop infer_action  ·  "
        "S0 wide & relatively uniform; B1 selectively tighter",
        fontsize=8.0,
        color=C_MUTED,
    )

    # ---- Panel A: width overlay ----
    ax = fig.add_subplot(gs[0, 0])
    # shade B1-narrow regions
    if np.any(narrow):
        # merge contiguous True runs
        idx = np.flatnonzero(narrow)
        starts = [idx[0]]
        ends = []
        for i in range(1, len(idx)):
            if idx[i] != idx[i - 1] + 1:
                ends.append(idx[i - 1])
                starts.append(idx[i])
        ends.append(idx[-1])
        for a0, a1 in zip(starts, ends):
            ax.axvspan(
                p[a0] - 0.5 * (p[1] - p[0]) if a0 > 0 else p[a0],
                p[a1] + 0.5 * (p[1] - p[0]) if a1 + 1 < len(p) else p[a1],
                color=C_SHADE,
                alpha=0.85,
                zorder=0,
                linewidth=0,
            )

    # soft fill under curves for visual weight = width
    ax.fill_between(p, 0.0, s0, color=C_S0, alpha=0.12, linewidth=0, zorder=1)
    ax.fill_between(p, 0.0, b1, color=C_B1, alpha=0.14, linewidth=0, zorder=2)
    ax.plot(p, s0, color=C_S0, lw=2.1, label=r"S0  $w(t)=\|\sigma(a_0)\|_2$", zorder=3)
    ax.plot(p, b1, color=C_B1, lw=2.1, label=r"B1  $w(t)=\|\sigma(a_0)\|_2$", zorder=4)

    # light dashed: p90-p10 radius width (secondary)
    ax.plot(p, s0_band, color=C_S0, lw=1.0, ls="--", alpha=0.55, label=r"S0  p90−p10 radius", zorder=3)
    ax.plot(p, b1_band, color=C_B1, lw=1.0, ls="--", alpha=0.55, label=r"B1  p90−p10 radius", zorder=4)

    # early cut marker
    ax.axvline(args.early_cut, color="#B0B8C0", ls=":", lw=1.0, zorder=1)
    ax.text(
        args.early_cut + 0.01,
        0.92,
        "early transient →",
        transform=ax.get_xaxis_transform(),
        fontsize=6.8,
        color=C_MUTED,
        va="top",
    )

    style_ax(ax)
    ax.set_xlim(0.0, 1.0)
    # focus y on mid-trajectory dynamics; annotate early spike if large
    y_hi = max(float(np.quantile(s0[mid], 0.98)), float(np.quantile(b1[mid], 0.98))) * 1.35
    y_hi = max(y_hi, 0.85)
    ax.set_ylim(0.0, y_hi)
    if float(s0.max()) > y_hi * 1.05:
        # marker for clipped early peak
        i_peak = int(np.argmax(s0))
        ax.annotate(
            f"S0 early peak\n{s0[i_peak]:.2f} (clipped)",
            xy=(p[i_peak], y_hi * 0.98),
            xytext=(p[i_peak] + 0.08, y_hi * 0.78),
            fontsize=6.8,
            color=C_S0,
            arrowprops=dict(arrowstyle="->", color=C_S0, lw=0.8),
        )

    ax.set_xlabel("episode progress", fontsize=9, color=C_MUTED, labelpad=3)
    ax.set_ylabel("action-interval width", fontsize=9, color=C_MUTED, labelpad=3)
    ax.set_title(
        "A   Interval width over progress  (solid = ‖σ‖₂; dashed = p90−p10)",
        fontsize=10,
        fontweight="semibold",
        color=C_TEXT,
        loc="left",
        pad=6,
    )
    ax.legend(frameon=False, fontsize=7.4, loc="upper right", ncol=2, columnspacing=1.1, handlelength=1.8)

    # note box
    ax.text(
        0.02,
        0.96,
        f"progress≥{args.early_cut:.2f}:  "
        f"mean w  S0={s0_mid_mean:.2f} · B1={b1_mid_mean:.2f}  "
        f"({s0_mid_mean/max(b1_mid_mean,1e-12):.1f}×)   ·   "
        f"CV  S0={s0_mid_cv:.2f} · B1={b1_mid_cv:.2f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.2,
        color=C_MUTED,
        bbox=dict(boxstyle="round,pad=0.28", facecolor=C_PANEL, edgecolor="#E2E6EA", linewidth=0.6),
    )
    # legend for green shade
    ax.text(
        0.98,
        0.08,
        "shaded = B1 relatively narrow vs S0",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.8,
        color="#3F6F45",
        style="italic",
    )

    # ---- Panel B: ratio ----
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.axhline(1.0, color="#B0B8C0", ls="--", lw=1.0, zorder=1)
    ax2.fill_between(p, 1.0, ratio_s, where=(ratio_s >= 1.0), color=C_RATIO, alpha=0.18, linewidth=0, zorder=2)
    ax2.plot(p, ratio_s, color=C_RATIO, lw=2.0, zorder=3, label=r"$w_{\mathrm{S0}}(t)\,/\,w_{\mathrm{B1}}(t)$")
    ax2.axvline(args.early_cut, color="#B0B8C0", ls=":", lw=1.0, zorder=1)
    style_ax(ax2)
    ax2.set_xlim(0.0, 1.0)
    # clip extreme early ratio for readability
    r_hi = float(np.quantile(ratio[mid], 0.98)) * 1.25
    r_hi = max(r_hi, 6.0)
    ax2.set_ylim(0.0, r_hi)
    if float(ratio.max()) > r_hi:
        i_peak = int(np.argmax(ratio))
        ax2.annotate(
            f"peak {ratio[i_peak]:.1f}×",
            xy=(p[i_peak], r_hi * 0.98),
            xytext=(min(p[i_peak] + 0.12, 0.75), r_hi * 0.72),
            fontsize=6.8,
            color=C_RATIO,
            arrowprops=dict(arrowstyle="->", color=C_RATIO, lw=0.8),
        )
    ax2.set_xlabel("episode progress", fontsize=9, color=C_MUTED, labelpad=3)
    ax2.set_ylabel("width ratio", fontsize=9, color=C_MUTED, labelpad=3)
    ax2.set_title(
        "B   Where B1 is tighter than S0   (>1 ⇒ B1 narrower)",
        fontsize=10,
        fontweight="semibold",
        color=C_TEXT,
        loc="left",
        pad=6,
    )
    ax2.legend(frameon=False, fontsize=7.6, loc="upper right")
    ax2.text(
        0.02,
        0.92,
        f"mean ratio (progress≥{args.early_cut:.2f}):  {ratio_mid_mean:.1f}×",
        transform=ax2.transAxes,
        ha="left",
        va="top",
        fontsize=7.2,
        color=C_MUTED,
        bbox=dict(boxstyle="round,pad=0.25", facecolor=C_PANEL, edgecolor="#E2E6EA", linewidth=0.6),
    )

    fig.text(
        0.10,
        0.02,
        "Width from denormalized a0 open-loop samples on fixed expert observations (ep22).  "
        "Solid: ‖σ_k(a0)‖₂ across K seeds.  Dashed: p90−p10 of ‖a−μ‖.  "
        "Green bands mark progress where S0/B1 width ratio is in the top 30% (mid-trajectory).",
        fontsize=6.4,
        color=C_MUTED,
    )

    stem = args.output_stem
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{stem}.png", dpi=args.dpi)
    fig.savefig(f"{stem}.pdf")
    plt.close(fig)

    meta = {
        "early_cut": args.early_cut,
        "mid_mean_width": {"S0": s0_mid_mean, "B1": b1_mid_mean, "S0_over_B1": s0_mid_mean / max(b1_mid_mean, 1e-12)},
        "mid_cv": {"S0": s0_mid_cv, "B1": b1_mid_cv},
        "mid_mean_ratio": ratio_mid_mean,
        "figure": f"{stem}.png",
    }
    Path(f"{stem}_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))
    print(f"[done] {stem}.png", flush=True)


if __name__ == "__main__":
    main()
