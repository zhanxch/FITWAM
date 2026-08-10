#!/usr/bin/env python3
"""Render NeurIPS-style 4×2 interaction-sensitivity evidence figure.

Reads artifacts from build_interaction_sensitivity_evidence.py:

  results/interaction_sensitivity_evidence_20260808/
    report.json
    panel_a_uncertainty.npz
    panel_b_action_deviation.npz
    panel_c_stage_failure.npz
    panel_d_latent_probe.npz

Layout (rows A–D × columns S0 | B1-remap-cfg):
  A  criticality + success action-deviation band over progress
  B  interaction Δa densities vs failure region
  C  failure rate contribution by episode stage
  D  interaction-centric motif latent PCA + probe stats

Example:
  conda activate web
  python scripts/analysis/render_interaction_sensitivity_figure.py
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
DEFAULT_DIR = ROOT / "results/interaction_sensitivity_evidence_20260808"
DEFAULT_STEM = DEFAULT_DIR / "fig_interaction_sensitivity_S0_vs_B1"

C_S0 = "#4C78A8"
C_B1 = "#54A24B"
C_FAIL = "#E45756"
C_SAFE = "#9DC183"
C_GRAY = "#6B7280"
C_GRID = "#E5E7EB"
C_TEXT = "#111827"
C_MUTED = "#4B5563"
C_BAND = "#DBEAFE"
C_BAND2 = "#DCFCE7"
C_EVENT = "#FEE2E2"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", type=Path, default=DEFAULT_DIR)
    p.add_argument("--output-stem", type=Path, default=DEFAULT_STEM)
    p.add_argument("--dpi", type=int, default=300)
    return p.parse_args()


def style_ax(ax, *, ylabel: str | None = None, xlabel: str | None = None, title: str | None = None):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=7, colors=C_MUTED)
    ax.grid(True, axis="y", color=C_GRID, linewidth=0.6, alpha=0.9)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8, color=C_TEXT)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=8, color=C_TEXT)
    if title:
        ax.set_title(title, fontsize=9, color=C_TEXT, pad=4, loc="left", fontweight="semibold")


def shade_interaction(ax, onset: float, offset: float, color: str = C_EVENT):
    ax.axvspan(onset, offset, color=color, alpha=0.55, zorder=0, lw=0)
    ax.axvline(onset, color=C_FAIL, lw=0.7, ls="--", alpha=0.7)
    ax.axvline(offset, color=C_FAIL, lw=0.7, ls="--", alpha=0.7)


def panel_a(ax, pack: dict, color: str, band_color: str, label: str):
    x = pack["progress_centers"]
    onset = float(pack["interaction_onset_q"][1])
    offset = float(pack["interaction_offset_q"][1])
    shade_interaction(ax, onset, offset)
    # success Δa band
    lo, mid, hi = pack["delta_p10"], pack["delta_p50"], pack["delta_p90"]
    ax.fill_between(x, lo, hi, color=band_color, alpha=0.85, linewidth=0, label="success Δa [p10,p90]", zorder=2)
    ax.plot(x, mid, color=color, lw=1.6, label="success Δa median", zorder=3)
    # soft-event on twin axis
    ax2 = ax.twinx()
    ax2.plot(x, pack["soft_event_mean"], color=C_GRAY, lw=1.1, ls=":", label="soft-event", zorder=4)
    ax2.set_ylim(0, max(0.35, float(np.nanmax(pack["soft_event_mean"])) * 1.25))
    ax2.set_ylabel("soft-event", fontsize=7, color=C_GRAY)
    ax2.tick_params(labelsize=6, colors=C_GRAY)
    ax2.spines["top"].set_visible(False)
    style_ax(ax, ylabel=r"$\|\Delta a\|$ vs expert", xlabel="episode progress", title=label)
    ax.set_xlim(0, 1)
    ymax = np.nanpercentile(hi[np.isfinite(hi)], 95) if np.any(np.isfinite(hi)) else 1.0
    ax.set_ylim(0, max(1.0, ymax * 1.15))
    # compact legend
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=6, frameon=False, loc="upper right")
    ax.text(
        (onset + offset) / 2,
        ax.get_ylim()[1] * 0.92,
        "interaction",
        ha="center",
        va="top",
        fontsize=6.5,
        color=C_FAIL,
    )


def panel_b(ax, pack: dict, color: str, label: str):
    x = pack["delta_centers"]
    ds = pack["density_success"]
    df = pack["density_failure"]
    boundary = pack["failure_boundary"]
    width = float(x[1] - x[0]) if len(x) > 1 else 0.05
    ax.bar(x, ds, width=width * 0.95, color=color, alpha=0.55, label="success", linewidth=0)
    ax.plot(x, df, color=C_FAIL, lw=1.5, label="failure")
    if np.isfinite(boundary):
        ax.axvspan(float(boundary), float(x[-1] + width), color=C_FAIL, alpha=0.12, lw=0)
        ax.axvline(float(boundary), color=C_FAIL, lw=1.0, ls="--")
        ax.text(
            float(boundary),
            ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1.0,
            " failure region",
            fontsize=6.5,
            color=C_FAIL,
            va="bottom",
        )
    style_ax(ax, ylabel="density", xlabel=r"action deviation $\Delta a=\|a-a_{\mathrm{expert}}\|$", title=label)
    ax.legend(fontsize=6, frameon=False, loc="upper right")
    # annotate means
    sm = pack["success_mean_delta"]
    fm = pack["failure_mean_delta"]
    if np.isfinite(sm) and np.isfinite(fm):
        ax.text(
            0.98,
            0.72,
            f"succ μ={sm:.2f}\nfail μ={fm:.2f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=6.5,
            color=C_MUTED,
        )


def panel_c(ax, stages: list[str], vals: np.ndarray, overall: float, color: str, which: str, ymax: float):
    x = np.arange(len(stages))
    bars = ax.bar(x, vals, color=color, width=0.72, alpha=0.9, edgecolor="white", linewidth=0.6)
    bars[2].set_edgecolor(C_FAIL)
    bars[2].set_linewidth(1.2)
    style_ax(
        ax,
        ylabel="failure rate contribution",
        xlabel=None,
        title=f"C  Failure rate by stage  ({which} overall {overall:.1%})",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(["Approach", "Pre-contact", "Interaction", "Post-contact"], fontsize=6.5, rotation=15)
    ax.set_ylim(0, ymax)
    for i, v in enumerate(vals):
        ax.text(i, v + ymax * 0.02, f"{v:.1%}", ha="center", va="bottom", fontsize=6.5, color=C_MUTED)


def panel_d(ax, z: np.ndarray, crit: np.ndarray, fail: np.ndarray, probe: dict, color: str, label: str):
    # subsample already done in compute; color by criticality
    sc = ax.scatter(
        z[:, 0],
        z[:, 1],
        c=crit,
        s=4,
        cmap="YlOrRd",
        alpha=0.55,
        linewidths=0,
        vmin=0.0,
        vmax=max(0.3, float(np.nanquantile(crit, 0.95))),
    )
    # mark failures lightly
    fmask = fail.astype(bool)
    if np.any(fmask):
        ax.scatter(
            z[fmask, 0],
            z[fmask, 1],
            s=10,
            facecolors="none",
            edgecolors=C_FAIL,
            linewidths=0.5,
            alpha=0.7,
            label="failure frames",
        )
    style_ax(ax, ylabel="PC2", xlabel="PC1", title=label)
    auc = probe.get("failure_auc")
    r2 = probe.get("criticality_r2")
    txt = []
    if auc is not None:
        txt.append(f"fail probe AUC={auc:.2f}")
    if r2 is not None:
        txt.append(f"crit. $R^2$={r2:.2f}")
    if txt:
        ax.text(
            0.02,
            0.98,
            "\n".join(txt),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=6.5,
            color=C_MUTED,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=C_GRID, alpha=0.9),
        )
    cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
    cbar.ax.tick_params(labelsize=6)
    cbar.set_label("soft-event", fontsize=6.5)
    if np.any(fmask):
        ax.legend(fontsize=6, frameon=False, loc="lower right")


def main() -> None:
    args = parse_args()
    d = args.input_dir
    report = json.loads((d / "report.json").read_text())
    a = np.load(d / "panel_a_uncertainty.npz")
    b = np.load(d / "panel_b_action_deviation.npz")
    c = np.load(d / "panel_c_stage_failure.npz")
    dd = np.load(d / "panel_d_latent_probe.npz")

    def apack(m: str) -> dict:
        return {
            "progress_centers": a[f"{m}_progress_centers"],
            "sigma_mean": a[f"{m}_sigma_mean"],
            "soft_event_mean": a[f"{m}_soft_event_mean"],
            "delta_p10": a[f"{m}_delta_p10"],
            "delta_p50": a[f"{m}_delta_p50"],
            "delta_p90": a[f"{m}_delta_p90"],
            "interaction_onset_q": a[f"{m}_interaction_onset_q"],
            "interaction_offset_q": a[f"{m}_interaction_offset_q"],
        }

    def bpack(m: str) -> dict:
        return {
            "delta_centers": b[f"{m}_delta_centers"],
            "density_success": b[f"{m}_density_success"],
            "density_failure": b[f"{m}_density_failure"],
            "failure_boundary": float(b[f"{m}_failure_boundary"]),
            "success_mean_delta": float(b[f"{m}_success_mean_delta"]),
            "failure_mean_delta": float(b[f"{m}_failure_mean_delta"]),
        }

    fig = plt.figure(figsize=(10.8, 11.4), dpi=args.dpi, facecolor="white")
    gs = GridSpec(
        5,
        2,
        figure=fig,
        height_ratios=[1.1, 1.0, 0.9, 1.15, 0.32],
        hspace=0.48,
        wspace=0.30,
        left=0.09,
        right=0.97,
        top=0.90,
        bottom=0.035,
    )
    fig.suptitle(
        "Expert demonstrations teach what action to take;\n"
        "rollout experience teaches when action precision matters",
        fontsize=12,
        fontweight="semibold",
        color=C_TEXT,
        y=0.98,
    )
    fig.text(
        0.5,
        0.935,
        "Water Plant  ·  official eval seeds 0–49 × 4 runs  ·  "
        f"S0 {report['counts']['S0']['success']}/{sum(report['counts']['S0'].values())}  vs  "
        f"B1 {report['counts']['B1']['success']}/{sum(report['counts']['B1'].values())}  ·  "
        "peak-centered soft-event stages",
        ha="center",
        fontsize=7.5,
        color=C_MUTED,
    )

    # Column titles
    fig.text(0.29, 0.905, "Expert-only Baseline (S0)", ha="center", fontsize=10, color=C_S0, fontweight="semibold")
    fig.text(0.73, 0.905, "Rollout-Retrained (B1-remap-cfg)", ha="center", fontsize=10, color=C_B1, fontweight="semibold")

    # A
    ax = fig.add_subplot(gs[0, 0])
    panel_a(ax, apack("S0"), C_S0, C_BAND, "A  Criticality & success action interval")
    ax = fig.add_subplot(gs[0, 1])
    panel_a(ax, apack("B1"), C_B1, C_BAND2, "A  Criticality & success action interval")

    # B
    ax = fig.add_subplot(gs[1, 0])
    panel_b(ax, bpack("S0"), C_S0, "B  Interaction-event action deviation")
    ax = fig.add_subplot(gs[1, 1])
    panel_b(ax, bpack("B1"), C_B1, "B  Interaction-event action deviation")

    # C
    stages = [str(s) for s in c["stage_names"]]
    rate_s0 = c["S0_failure_rate_contribution"]
    rate_b1 = c["B1_failure_rate_contribution"]
    rates = {
        "S0": float(c["S0_overall_failure_rate"]),
        "B1": float(c["B1_overall_failure_rate"]),
    }
    ymax = max(0.14, float(np.max(np.r_[rate_s0, rate_b1])) * 1.25)
    ax = fig.add_subplot(gs[2, 0])
    panel_c(ax, stages, rate_s0, rates["S0"], C_S0, "S0", ymax)
    ax = fig.add_subplot(gs[2, 1])
    panel_c(ax, stages, rate_b1, rates["B1"], C_B1, "B1", ymax)
    ax.text(
        0.98,
        0.98,
        f"Δ overall {rates['B1']-rates['S0']:+.1%}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7,
        color=C_B1,
        fontweight="semibold",
    )

    # D
    ax = fig.add_subplot(gs[3, 0])
    panel_d(
        ax,
        dd["S0_scatter_z"],
        dd["S0_scatter_crit"],
        dd["S0_scatter_fail"],
        report["panel_d"]["S0"]["probe"],
        C_S0,
        "D  Interaction-centric motif latent",
    )
    ax = fig.add_subplot(gs[3, 1])
    panel_d(
        ax,
        dd["B1_scatter_z"],
        dd["B1_scatter_crit"],
        dd["B1_scatter_fail"],
        report["panel_d"]["B1"]["probe"],
        C_B1,
        "D  Interaction-centric motif latent",
    )

    # Takeaway box
    ax = fig.add_subplot(gs[4, :])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    box = mpatches.FancyBboxPatch(
        (0.01, 0.08),
        0.98,
        0.84,
        boxstyle="round,pad=0.015,rounding_size=0.02",
        linewidth=1.0,
        edgecolor="#CBD5E1",
        facecolor="#F8FAFC",
        transform=ax.transAxes,
        clip_on=False,
    )
    ax.add_patch(box)
    ax.text(
        0.03,
        0.70,
        "Takeaway",
        fontsize=8,
        fontweight="semibold",
        color=C_TEXT,
        transform=ax.transAxes,
        va="center",
    )
    ax.text(
        0.03,
        0.35,
        report["takeaway"],
        fontsize=7.5,
        color=C_MUTED,
        transform=ax.transAxes,
        va="center",
        wrap=True,
    )

    # Row tags on left
    for y, tag in zip([0.78, 0.58, 0.40, 0.20], ["A", "B", "C", "D"]):
        pass  # titles already include letters

    args.output_stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        path = Path(str(args.output_stem) + f".{ext}")
        fig.savefig(path, dpi=args.dpi, bbox_inches="tight", facecolor="white")
        print("wrote", path)
    plt.close(fig)


if __name__ == "__main__":
    main()
