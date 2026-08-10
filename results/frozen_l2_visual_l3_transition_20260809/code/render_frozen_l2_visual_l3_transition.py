#!/usr/bin/env python3
"""Frozen paper figures: L3 transition 1NN hist + L2-visual frame-level coverage.

Canonical approved artifacts live under:
  results/frozen_l2_visual_l3_transition_20260809/

This script is style-locked. Prefer --mode replot (from cached distances).
Do not casually restyle; if a paper figure must change, bump a new frozen dir.

Panels
------
1. L3_transition_nn_hist.png
   Source run: experience_distribution_coverage_20260808
   Feature: [s_t, a_t, s_{t+5}] z-scored by Expert
   Data: Expert vs S0 rollout (b0/b1 20260718), stride=5, seed=20260808,
         50 sampled success + all 45 failure episodes

2. L2_visual_frame_all.png
   Source: render_l2_visual_frame_all.py (cfg10086 4×50, all frames, no trim)
   Feature: concat(s, z_VAE, a), global 1NN to Expert

Example:
  conda activate web
  python scripts/analysis/render_frozen_l2_visual_l3_transition.py --mode replot
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FROZEN = ROOT / "results/frozen_l2_visual_l3_transition_20260809"
ARCHIVE = ROOT / "archive/l2l3_experience_coverage_20260809"
L2_SCRIPT = ARCHIVE / "scripts/render_l2_visual_frame_all.py"
L2_FEATURES = FROZEN / "L2_visual_features_fulltraj.npz"

# L3 hist colors — must match experience_distribution_coverage.hist_plot
L3_COLORS = {
    "expert_self": "#4C78A8",
    "rollout_success": "#72B7B2",
    "rollout_failure": "#E45756",
}

# L2-visual colors — must match render_l2_visual_frame_all.py
C_EXPERT = "#4C78A8"
C_SUCC = "#54A24B"
C_FAIL = "#E45756"
C_MUTED = "#5B6B7A"
C_TEXT = "#1F2A33"
C_GRID = "#E6E9ED"
C_PANEL = "#F7F8FA"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--mode",
        choices=("replot", "recompute-l2", "snapshot-code"),
        default="replot",
        help="replot=from frozen distances; recompute-l2=rerun L2 NN; snapshot-code=refresh code/",
    )
    p.add_argument("--frozen-dir", type=Path, default=FROZEN)
    p.add_argument("--dpi-l3", type=int, default=160)
    p.add_argument("--dpi-l2", type=int, default=300)
    p.add_argument(
        "--overwrite-canonical",
        action="store_true",
        help="Replace frozen PNGs/PDFs (default: write *_regen.* beside them)",
    )
    return p.parse_args()


def out_path(stem: Path, *, overwrite: bool, suffix: str) -> Path:
    if overwrite:
        return stem.with_suffix(suffix)
    return stem.with_name(stem.name + "_regen").with_suffix(suffix)


def plot_l3_hist(
    distances: Path,
    out_png: Path,
    *,
    dpi: int,
) -> dict[str, Any]:
    """Exact style of experience_distribution_coverage.hist_plot for L3."""
    z = np.load(distances)
    series = [
        ("expert_self", np.asarray(z["d_self"])),
        ("rollout_success", np.asarray(z["d_succ"])),
        ("rollout_failure", np.asarray(z["d_fail"])),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for name, d in series:
        if len(d) == 0:
            continue
        ax.hist(d, bins=40, density=True, alpha=0.45, label=name, color=L3_COLORS[name])
    ax.set_title("L3 transition novelty: 1NN to expert (o_t,a_t,o_{t+k})")
    ax.set_xlabel("euclidean 1NN")
    ax.set_ylabel("density")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)
    return {
        "n": {name: int(len(d)) for name, d in series},
        "median": {name: float(np.median(d)) for name, d in series},
        "mean": {name: float(np.mean(d)) for name, d in series},
        "figure": str(out_png),
    }


def kde_curve(
    d: np.ndarray,
    *,
    x_grid: np.ndarray,
    bw_method: float | str = "scott",
) -> np.ndarray:
    from scipy.stats import gaussian_kde

    d = np.asarray(d, dtype=np.float64)
    d = d[np.isfinite(d)]
    if len(d) < 2:
        return np.zeros_like(x_grid)
    # Slight floor so long-tail modes don't get over-smoothed into invisibility.
    kde = gaussian_kde(d, bw_method=bw_method)
    y = kde(x_grid)
    return np.maximum(y, 0.0)


def plot_l3_style_smooth(
    series: list[tuple[str, np.ndarray, str]],
    out_png: Path,
    *,
    title: str,
    xlabel: str,
    dpi: int,
    x_max: float | None = None,
    bw_method: float | str = 0.35,
    figsize: tuple[float, float] = (7.2, 4.8),
) -> dict[str, Any]:
    """L3 aesthetic (simple axes/grid/legend) with filled KDE curves instead of bins."""
    cleaned: list[tuple[str, np.ndarray, str]] = []
    for name, d, color in series:
        d = np.asarray(d, dtype=np.float64)
        d = d[np.isfinite(d)]
        if len(d) == 0:
            continue
        cleaned.append((name, d, color))
    if not cleaned:
        raise ValueError("no finite distances to plot")

    hi = max(float(np.quantile(d, 0.995)) for _, d, _ in cleaned)
    if x_max is None:
        x_max = max(hi * 1.05, 1.0)
    x_grid = np.linspace(0.0, float(x_max), 512)

    fig, ax = plt.subplots(figsize=figsize)
    stats: dict[str, Any] = {"n": {}, "median": {}, "mean": {}}
    for name, d, color in cleaned:
        y = kde_curve(d, x_grid=x_grid, bw_method=bw_method)
        ax.fill_between(x_grid, y, alpha=0.35, color=color, linewidth=0)
        ax.plot(x_grid, y, color=color, linewidth=1.8, label=name, alpha=0.95)
        stats["n"][name] = int(len(d))
        stats["median"][name] = float(np.median(d))
        stats["mean"][name] = float(np.mean(d))

    ax.set_xlim(0.0, float(x_max))
    ax.set_ylim(bottom=0.0)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("density")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi)
    # Also write PDF sibling for paper use
    fig.savefig(out_png.with_suffix(".pdf"))
    plt.close(fig)
    stats["figure"] = str(out_png)
    stats["bw_method"] = bw_method
    stats["x_max"] = float(x_max)
    return stats


def plot_l3_smooth(distances: Path, out_png: Path, *, dpi: int) -> dict[str, Any]:
    z = np.load(distances)
    return plot_l3_style_smooth(
        [
            ("expert_self", np.asarray(z["d_self"]), L3_COLORS["expert_self"]),
            ("rollout_success", np.asarray(z["d_succ"]), L3_COLORS["rollout_success"]),
            ("rollout_failure", np.asarray(z["d_fail"]), L3_COLORS["rollout_failure"]),
        ],
        out_png,
        title="L3 transition novelty: 1NN to expert (o_t,a_t,o_{t+k})",
        xlabel="euclidean 1NN",
        dpi=dpi,
        bw_method=0.35,
    )


def plot_l2_linear_l3style_smooth(
    distances: Path,
    out_png: Path,
    *,
    dpi: int,
) -> dict[str, Any]:
    z = np.load(distances)
    return plot_l3_style_smooth(
        [
            ("expert_self", np.asarray(z["d_self"]), L3_COLORS["expert_self"]),
            ("rollout_success", np.asarray(z["d_succ"]), L3_COLORS["rollout_success"]),
            ("rollout_failure", np.asarray(z["d_fail"]), L3_COLORS["rollout_failure"]),
        ],
        out_png,
        title="L2–Visual: 1NN to expert  ·  frame-level (all frames, linear)",
        xlabel="1-NN distance to Expert",
        dpi=dpi,
        bw_method=0.35,
    )



def hist_density_l2(
    ax: plt.Axes,
    series: list[tuple[np.ndarray, str, str]],
    *,
    note: str,
    log_x: bool = True,
) -> None:
    positives = [d[np.isfinite(d) & (d > 0)] for d, _, _ in series if len(d)]
    lo = min(float(np.quantile(d, 0.02)) for d in positives)
    hi = max(float(np.quantile(d, 0.98)) for d in positives)
    if log_x:
        bins = np.geomspace(max(lo, 1e-2), max(hi * 1.05, lo * 1.2), 42)
    else:
        bins = np.linspace(0.0, max(hi * 1.05, lo * 1.2), 42)
    for d, color, name in series:
        d = d[np.isfinite(d)]
        ax.hist(
            np.clip(d, bins[0], bins[-1]),
            bins=bins,
            density=True,
            histtype="stepfilled",
            alpha=0.28,
            color=color,
            linewidth=0,
            label=name,
        )
        ax.hist(
            np.clip(d, bins[0], bins[-1]),
            bins=bins,
            density=True,
            histtype="step",
            alpha=0.95,
            color=color,
            linewidth=1.4,
        )
        med = float(np.median(d))
        ax.axvline(med, color=color, ls="--", lw=1.1, alpha=0.75)

    if log_x:
        ax.set_xscale("log")
        xlabel = "1-NN distance to Expert  (log scale)"
    else:
        xlabel = "1-NN distance to Expert  (linear)"
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#C5CCD3")
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(labelsize=8, colors=C_MUTED, length=3, pad=2)
    ax.grid(True, which="major", color=C_GRID, linewidth=0.65, alpha=0.95)
    ax.set_axisbelow(True)
    ax.set_xlabel(xlabel, fontsize=9, color=C_MUTED, labelpad=3)
    ax.set_ylabel("Density", fontsize=9, color=C_MUTED, labelpad=3)
    ax.legend(frameon=False, fontsize=8.2, loc="upper right", handlelength=1.2)
    ax.text(
        0.02,
        0.96,
        note,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.0,
        color=C_MUTED,
        bbox=dict(boxstyle="round,pad=0.28", facecolor=C_PANEL, edgecolor="#E2E6EA", linewidth=0.6),
    )


def plot_l2_visual_from_cache(
    distances: Path,
    meta_path: Path,
    out_stem: Path,
    *,
    dpi: int,
    log_x: bool = True,
) -> dict[str, Any]:
    z = np.load(distances)
    d_self = np.asarray(z["d_self"])
    d_succ = np.asarray(z["d_succ"])
    d_fail = np.asarray(z["d_fail"])
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    r_s = float(meta.get("median_over_expert", {}).get("success", np.median(d_succ) / max(np.median(d_self), 1e-12)))
    r_f = float(meta.get("median_over_expert", {}).get("failure", np.median(d_fail) / max(np.median(d_self), 1e-12)))
    # Episode counts from the approved figure caption
    n_e, n_s, n_f = 100, 144, 56

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
    fig, ax = plt.subplots(figsize=(8.6, 5.4), dpi=dpi)
    fig.subplots_adjust(left=0.10, right=0.97, top=0.82, bottom=0.14)
    fig.text(
        0.10,
        0.93,
        "L2–Visual Experience Coverage  ·  Frame-level (all frames)",
        fontsize=13.5,
        fontweight="bold",
        color=C_TEXT,
    )
    fig.text(
        0.10,
        0.875,
        "Water Plant  ·  Expert vs S0 4×50 (seed 10086)  ·  "
        r"$x=\mathrm{concat}(s,\;z_{\mathrm{VAE}},\;a)$  ·  "
        r"$d(x)=\min_{e}\|x-e\|$",
        fontsize=8.2,
        color=C_MUTED,
    )
    hist_density_l2(
        ax,
        [
            (d_self, C_EXPERT, f"Expert self  ({n_e} eps)"),
            (d_succ, C_SUCC, rf"$R_{{\mathrm{{succ}}}}$  ({n_s} eps)"),
            (d_fail, C_FAIL, rf"$R_{{\mathrm{{fail}}}}$  ({n_f} eps)"),
        ],
        note=f"median / Expert:  succ {r_s:.2f}×   fail {r_f:.2f}×",
        log_x=log_x,
    )
    ax.set_title(
        "All frames (no trim)" if log_x else "All frames (no trim)  ·  linear-x",
        fontsize=10.5,
        fontweight="semibold",
        color=C_TEXT,
        loc="left",
        pad=8,
    )
    fig.text(
        0.10,
        0.035,
        "Feature: state ⊕ S0 VAE pooled front|wrist ⊕ action (each z-scored on Expert, then concat).  "
        "Expert↔Expert = leave-one-point self-1NN; rollouts = 1NN into Expert gallery.  "
        "Aligned to visual frame grid; all frames, no trim.",
        fontsize=6.5,
        color=C_MUTED,
    )
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    png = out_stem.with_suffix(".png")
    pdf = out_stem.with_suffix(".pdf")
    fig.savefig(png, dpi=dpi)
    fig.savefig(pdf)
    plt.close(fig)
    return {
        "figure_png": str(png),
        "figure_pdf": str(pdf),
        "x_scale": "log" if log_x else "linear",
        "median_over_expert": {"success": r_s, "failure": r_f},
    }

def snapshot_code(frozen: Path) -> None:
    code = frozen / "code"
    code.mkdir(parents=True, exist_ok=True)
    # Prefer live script if present; else archived copy.
    l2_src = ROOT / "scripts/analysis/render_l2_visual_frame_all.py"
    if not l2_src.exists():
        l2_src = L2_SCRIPT
    if l2_src.exists():
        shutil.copy2(l2_src, code / "render_l2_visual_frame_all.py")
    shutil.copy2(
        ROOT / "scripts/analysis/render_frozen_l2_visual_l3_transition.py",
        code / "render_frozen_l2_visual_l3_transition.py",
    )
    (code / "L3_SOURCE.txt").write_text(
        "L3 panel originally produced by experience_distribution_coverage.py\n"
        "(now archived under archive/l2l3_experience_coverage_20260809/scripts/).\n"
        "hist_plot() + Layer-3 transition gallery (stride=5, lag=5, seed=20260808).\n"
        "Frozen replot uses L3_distances.npz via render_frozen_l2_visual_l3_transition.py.\n"
    )
    (code / "ARCHIVE.txt").write_text(
        f"Historical L2/L3 iteration code+results: {ARCHIVE}\n"
        f"Active figure dir: {FROZEN}\n"
    )


def recompute_l2(frozen: Path) -> None:
    import subprocess

    if not L2_SCRIPT.exists():
        raise FileNotFoundError(f"Missing archived L2 script: {L2_SCRIPT}")
    if not L2_FEATURES.exists():
        raise FileNotFoundError(f"Missing frozen visual features: {L2_FEATURES}")

    stem = frozen / "L2_visual_frame_all"
    cmd = [
        sys.executable,
        str(L2_SCRIPT),
        "--visual-features",
        str(L2_FEATURES),
        "--output-stem",
        str(stem),
    ]
    print("[recompute-l2]", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)
    dist = Path(f"{stem}_distances.npz")
    meta = Path(f"{stem}_meta.json")
    if dist.exists():
        shutil.copy2(dist, frozen / "L2_visual_distances.npz")
    if meta.exists():
        shutil.copy2(meta, frozen / "L2_visual_meta.json")


def main() -> None:
    args = parse_args()
    frozen = args.frozen_dir
    frozen.mkdir(parents=True, exist_ok=True)

    if args.mode == "snapshot-code":
        snapshot_code(frozen)
        print(f"[done] code snapshot → {frozen / 'code'}")
        return

    if args.mode == "recompute-l2":
        recompute_l2(frozen)
        snapshot_code(frozen)
        print(f"[done] recompute-l2 → {frozen}")
        return

    # replot both from cache
    l3_png = out_path(frozen / "L3_transition_nn_hist", overwrite=args.overwrite_canonical, suffix=".png")
    l3_stats = plot_l3_hist(frozen / "L3_distances.npz", l3_png, dpi=args.dpi_l3)
    print("[L3]", json.dumps(l3_stats, indent=2))

    l2_stem = out_path(frozen / "L2_visual_frame_all", overwrite=args.overwrite_canonical, suffix=".png").with_suffix("")
    l2_stats = plot_l2_visual_from_cache(
        frozen / "L2_visual_distances.npz",
        frozen / "L2_visual_meta.json",
        l2_stem,
        dpi=args.dpi_l2,
        log_x=True,
    )
    print("[L2-log]", json.dumps(l2_stats, indent=2))

    # Always write the linear-x companion next to the frozen log-x figure.
    l2_lin = plot_l2_visual_from_cache(
        frozen / "L2_visual_distances.npz",
        frozen / "L2_visual_meta.json",
        frozen / "L2_visual_frame_all_linear",
        dpi=args.dpi_l2,
        log_x=False,
    )
    print("[L2-linear]", json.dumps(l2_lin, indent=2))

    # Smooth L3-style companions (do not overwrite the original binned L3 hist).
    l3_smooth = plot_l3_smooth(
        frozen / "L3_distances.npz",
        frozen / "L3_transition_nn_smooth.png",
        dpi=args.dpi_l3,
    )
    print("[L3-smooth]", json.dumps(l3_smooth, indent=2))
    l2_l3s = plot_l2_linear_l3style_smooth(
        frozen / "L2_visual_distances.npz",
        frozen / "L2_visual_frame_all_linear_l3style_smooth.png",
        dpi=max(args.dpi_l3, 200),
    )
    print("[L2-linear-l3style-smooth]", json.dumps(l2_l3s, indent=2))

    snapshot_code(frozen)
    print(f"[done] replot → {frozen}  (overwrite_canonical={args.overwrite_canonical})")

if __name__ == "__main__":
    main()
