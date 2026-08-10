#!/usr/bin/env python3
"""Side-by-side S0 | B1 open-loop RESIDUAL intervals (early transient dropped).

Per frame t, across K seeds:
  r_k = ||a_k - a_GT||_2
Show median(r) and [p10, p90] band.

Early progress < cut is removed (startup spike / disturbance).
Panels use independent y-scales so relative width changes are visible
(B1 is much tighter absolutely than S0).
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
    p.add_argument("--early-cut", type=float, default=0.10, help="drop progress < cut")
    p.add_argument("--dpi", type=int, default=300)
    return p.parse_args()


def residual_stats(actions: np.ndarray, gt: np.ndarray) -> dict[str, np.ndarray]:
    """actions: [T,K,D], gt: [T,D] → residual norm stats over K."""
    r = np.linalg.norm(actions - gt[:, None, :], axis=-1)  # [T,K]
    return {
        "median": np.median(r, axis=1),
        "p10": np.quantile(r, 0.10, axis=1),
        "p90": np.quantile(r, 0.90, axis=1),
        "std": r.std(axis=1),
    }


def style_ax(ax: plt.Axes) -> None:
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#C5CCD3")
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(labelsize=7.5, colors=C_MUTED, length=2.5, pad=1.5)
    ax.grid(True, axis="y", color=C_GRID, linewidth=0.65, alpha=0.95)
    ax.set_axisbelow(True)


def panel(
    ax: plt.Axes,
    progress: np.ndarray,
    stats: dict[str, np.ndarray],
    *,
    color: str,
    title: str,
) -> dict[str, float]:
    lo, hi, mid = stats["p10"], stats["p90"], stats["median"]
    width = hi - lo
    ax.fill_between(progress, lo, hi, color=color, alpha=0.28, linewidth=0, zorder=2, label=r"residual [p10, p90]")
    ax.plot(progress, mid, color=color, lw=1.8, zorder=3, label=r"median $\|a-a^{\mathrm{GT}}\|_2$")
    # light outline of band edges for width readability
    ax.plot(progress, lo, color=color, lw=0.7, alpha=0.55, zorder=3)
    ax.plot(progress, hi, color=color, lw=0.7, alpha=0.55, zorder=3)

    style_ax(ax)
    ax.set_title(title, fontsize=9.5, color=color, loc="left", fontweight="semibold", pad=5)
    ax.set_xlabel("episode progress", fontsize=8, color=C_MUTED, labelpad=2)
    ax.set_xlim(float(progress.min()), float(progress.max()))
    # independent scale with a little headroom
    y0 = max(0.0, float(np.nanmin(lo)) * 0.85)
    y1 = float(np.nanmax(hi)) * 1.12
    ax.set_ylim(y0, max(y1, y0 + 1e-3))
    ax.legend(frameon=False, fontsize=6.8, loc="upper right")

    mean_w = float(np.mean(width))
    mean_med = float(np.mean(mid))
    ax.text(
        0.02,
        0.95,
        f"mean band width {mean_w:.2f}   ·   mean residual {mean_med:.2f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.8,
        color=C_MUTED,
        bbox=dict(boxstyle="round,pad=0.22", facecolor="#F7F8FA", edgecolor="#E2E6EA", linewidth=0.6),
    )
    return {"mean_band_width": mean_w, "mean_residual": mean_med}


def main() -> None:
    args = parse_args()
    ddir = args.input_dir
    z = np.load(ddir / "openloop_interval.npz")
    progress = np.asarray(z["progress"], dtype=np.float64)
    gt = np.asarray(z["executed_at_frames"], dtype=np.float64)
    keep = progress >= args.early_cut
    progress = progress[keep]
    gt = gt[keep]

    s0 = residual_stats(np.asarray(z["S0_actions_a0"])[keep], gt)
    b1 = residual_stats(np.asarray(z["B1_actions_a0"])[keep], gt)

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

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.35), dpi=args.dpi, sharey=False)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.76, bottom=0.20, wspace=0.22)

    fig.suptitle(
        "Open-loop residual to expert action  ·  interval width along the trajectory",
        fontsize=12,
        fontweight="bold",
        color=C_TEXT,
        y=0.98,
        x=0.52,
    )
    fig.text(
        0.52,
        0.895,
        f"{source}  ·  drop progress < {args.early_cut:.2f} (early transient)  ·  "
        f"K={int(z['num_samples'])} samples/frame  ·  "
        r"$r=\|a-a^{\mathrm{GT}}\|_2$  ·  panels use independent y-scales",
        ha="center",
        fontsize=7.5,
        color=C_MUTED,
    )

    m0 = panel(axes[0], progress, s0, color=C_S0, title="Expert-only Baseline (S0)")
    m1 = panel(axes[1], progress, b1, color=C_B1, title="Rollout-Retrained (B1-remap-cfg)")
    axes[0].set_ylabel(r"residual $\|a-a^{\mathrm{GT}}\|_2$", fontsize=8, color=C_MUTED, labelpad=3)

    fig.text(
        0.52,
        0.035,
        "Shaded band = [p10, p90] of residual norms across open-loop seeds (width = how spread the residuals are).  "
        "Y-scales differ on purpose so B1’s relative narrowing is visible; compare absolute width via the annotations.  "
        f"S0 mean width {m0['mean_band_width']:.2f} vs B1 {m1['mean_band_width']:.2f} "
        f"({m0['mean_band_width']/max(m1['mean_band_width'],1e-12):.1f}×).",
        ha="center",
        fontsize=6.4,
        color=C_MUTED,
    )

    stem = args.output_stem
    fig.savefig(f"{stem}.png", dpi=args.dpi)
    fig.savefig(f"{stem}.pdf")
    plt.close(fig)

    meta = {
        "early_cut": args.early_cut,
        "metric": "residual_norm ||a-a_GT||_2 over K seeds",
        "independent_ylim": True,
        "S0": m0,
        "B1": m1,
        "figure": f"{stem}.png",
    }
    Path(f"{stem}_residual_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))
    print(f"[done] {stem}.png", flush=True)


if __name__ == "__main__":
    main()
