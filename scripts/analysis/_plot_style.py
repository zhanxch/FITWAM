"""Shared publication style for Pass@20 / CFG event figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

INK = "#1c1917"
MUTED = "#57534e"
STONE = "#a8a29e"
GRID = "#f3f1ef"
BEFORE = "#7c756f"
BEFORE_FILL = "#e8e4df"
EVENT = "#b4533a"
EVENT_FILL = "#f4d6ce"
RECOVER = "#3f6f8a"
RECOVER_FILL = "#d7e4ec"
PASS = "#3f6f8a"
WINDOW = "#f7efe9"


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Liberation Sans", "Arial"],
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titlecolor": INK,
            "axes.labelsize": 10,
            "axes.labelcolor": INK,
            "axes.edgecolor": "#d6d3d1",
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "legend.frameon": False,
            "legend.fontsize": 8.5,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def polish(ax) -> None:
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRID, linewidth=0.9, zorder=0)
    ax.xaxis.grid(False)
    ax.tick_params(length=3, color="#d6d3d1")


def panel_label(ax, letter: str) -> None:
    ax.text(
        -0.12,
        1.08,
        letter,
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        color=INK,
        va="bottom",
        ha="left",
    )


def stat_note(ax, text: str, *, loc: str = "upper right") -> None:
    ha, va, x, y = {
        "upper right": ("right", "top", 0.97, 0.96),
        "upper left": ("left", "top", 0.04, 0.96),
        "lower right": ("right", "bottom", 0.97, 0.05),
    }[loc]
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=8.2,
        color=MUTED,
        linespacing=1.35,
        bbox={
            "boxstyle": "round,pad=0.32",
            "facecolor": "#fffcfa",
            "edgecolor": "#eadfd8",
            "linewidth": 0.6,
        },
        zorder=6,
    )


def save_figure(fig, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=240, bbox_inches="tight", pad_inches=0.12)
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def paired_boxes(
    ax,
    before: np.ndarray,
    event: np.ndarray,
    *,
    labels: tuple[str, str] = ("earlier", r"$t^*$"),
    event_color: str = EVENT,
) -> None:
    rng = np.random.default_rng(1)
    data = [before, event]
    box = ax.boxplot(
        data,
        positions=[1, 2],
        widths=0.42,
        showfliers=False,
        patch_artist=True,
        medianprops={"color": INK, "linewidth": 1.6},
        whiskerprops={"color": STONE, "linewidth": 1.0},
        capprops={"color": STONE, "linewidth": 1.0},
        boxprops={"linewidth": 0.0},
        zorder=2,
    )
    fills = [BEFORE_FILL, EVENT_FILL if event_color == EVENT else RECOVER_FILL]
    edges = [BEFORE, event_color]
    for patch, fill, edge in zip(box["boxes"], fills, edges):
        patch.set_facecolor(fill)
        patch.set_edgecolor(edge)
        patch.set_linewidth(1.1)
        patch.set_alpha(0.95)
    for left, right in zip(before, event):
        ax.plot(
            [1.16, 1.84],
            [left, right],
            color=STONE,
            alpha=0.35,
            lw=0.7,
            zorder=1,
            solid_capstyle="round",
        )
    ax.scatter(
        1 + rng.uniform(-0.06, 0.06, before.size),
        before,
        s=22,
        c=BEFORE,
        alpha=0.9,
        zorder=3,
        edgecolors="white",
        linewidths=0.4,
    )
    ax.scatter(
        2 + rng.uniform(-0.06, 0.06, event.size),
        event,
        s=26,
        c=event_color,
        alpha=0.95,
        zorder=3,
        edgecolors="white",
        linewidths=0.45,
    )
    ax.set_xticks([1, 2], list(labels))
    polish(ax)


def triple_boxes(
    ax,
    left: np.ndarray,
    mid: np.ndarray,
    right: np.ndarray,
    *,
    labels: tuple[str, str, str] = ("earlier", r"$S_A$", r"$S_V$"),
    colors: tuple[str, str, str] = (BEFORE, PASS, EVENT),
    fills: tuple[str, str, str] = (BEFORE_FILL, RECOVER_FILL, EVENT_FILL),
) -> None:
    rng = np.random.default_rng(1)
    data = [left, mid, right]
    positions = [1, 2, 3]
    box = ax.boxplot(
        data,
        positions=positions,
        widths=0.42,
        showfliers=False,
        patch_artist=True,
        medianprops={"color": INK, "linewidth": 1.6},
        whiskerprops={"color": STONE, "linewidth": 1.0},
        capprops={"color": STONE, "linewidth": 1.0},
        boxprops={"linewidth": 0.0},
        zorder=2,
    )
    for patch, fill, edge in zip(box["boxes"], fills, colors):
        patch.set_facecolor(fill)
        patch.set_edgecolor(edge)
        patch.set_linewidth(1.1)
        patch.set_alpha(0.95)
    for a, b, c in zip(left, mid, right):
        ax.plot(
            [1.16, 1.84, 2.16, 2.84],
            [a, b, b, c],
            color=STONE,
            alpha=0.35,
            lw=0.7,
            zorder=1,
            solid_capstyle="round",
        )
    sizes = (22, 24, 26)
    for pos, values, color, size in zip(positions, data, colors, sizes):
        ax.scatter(
            pos + rng.uniform(-0.06, 0.06, values.size),
            values,
            s=size,
            c=color,
            alpha=0.95,
            zorder=3,
            edgecolors="white",
            linewidths=0.45,
        )
    ax.set_xticks(positions, list(labels))
    polish(ax)
