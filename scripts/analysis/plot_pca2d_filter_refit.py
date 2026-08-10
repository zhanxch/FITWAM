#!/usr/bin/env python3
"""PCA-plane filter → drop Success∩Expert → refit PCA (circles + scatter).

Protocol
--------
1. PCA#1 on Expert + Success + Failure (joint).
2. In PCA#1 plane, drop Success that falls inside Expert coverage:
     --mode rms      : ||z - μ_E|| ≤ r_rms(E)
     --mode ellipse2 : Mahalanobis² ≤ χ²_2(0.95) under Expert cov
3. PCA#2 refit on Expert + Success_remain + Failure.
4. Emit per-layer 2×2 panels (stage1/2 × scatter/circles) and an overview.

Canonical outputs:
  results/frozen_l2_visual_l3_transition_20260809/pca_probe/
    fig_pca2d_filter_refit_{rms,ellipse}_{L2,L3,overview}.{png,pdf}
    pca2d_filter_refit_report.json

Example:
  conda activate web
  python scripts/analysis/plot_pca2d_filter_refit.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse
import numpy as np
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[2]
ANALYSIS = Path(__file__).resolve().parent
FROZEN = ROOT / "results/frozen_l2_visual_l3_transition_20260809"
DEFAULT_OUT = FROZEN / "pca_probe"
DEFAULT_EXPERT = ROOT / "data/water_plant_fastwam"
DEFAULT_L2_ROLLOUT = ROOT / "data/water_plant_s0_b1_video_cfg_20260808_152243/rollout_raw_200"
DEFAULT_L3_ROLLOUT = ROOT / "data/water_plant_s0_rollout_b0_b1_20260718/rollout"
DEFAULT_VIS = FROZEN / "L2_visual_features_fulltraj.npz"
DEFAULT_L3_META = FROZEN / "L3_meta.json"

C = {"expert": "#4C78A8", "success": "#54A24B", "failure": "#E45756"}
C_MUTED, C_TEXT, C_GRID = "#5B6B7A", "#1F2A33", "#E6E9ED"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--expert", type=Path, default=DEFAULT_EXPERT)
    p.add_argument("--l2-rollout", type=Path, default=DEFAULT_L2_ROLLOUT)
    p.add_argument("--l3-rollout", type=Path, default=DEFAULT_L3_ROLLOUT)
    p.add_argument("--visual-features", type=Path, default=DEFAULT_VIS)
    p.add_argument("--l3-meta", type=Path, default=DEFAULT_L3_META)
    p.add_argument(
        "--modes",
        nargs="+",
        default=["rms", "ellipse2"],
        choices=("rms", "ellipse2", "p90"),
        help="PCA#1 membership rules to run",
    )
    p.add_argument("--dpi", type=int, default=210)
    return p.parse_args()


def class_geom(Z: np.ndarray, y: np.ndarray, c: int) -> dict[str, Any]:
    z = Z[y == c]
    mu = z.mean(0)
    d = np.linalg.norm(z - mu, axis=1)
    cov = np.cov(z.T) if len(z) > 1 else np.eye(2)
    cov = cov + 1e-6 * np.eye(2)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    evals, evecs = evals[order], evecs[:, order]
    return {
        "mu": mu,
        "n": int(len(z)),
        "r_rms": float(np.sqrt(np.mean(d**2))),
        "r_p90": float(np.quantile(d, 0.9)),
        "cov": cov,
        "cov_evals": evals,
        "cov_evecs": evecs,
    }


def inside_expert_pca2d(Z: np.ndarray, y: np.ndarray, mode: str) -> tuple[np.ndarray, dict[str, Any]]:
    ge = class_geom(Z, y, 0)
    zs = Z[y == 1]
    mu = ge["mu"]
    if mode == "rms":
        r = ge["r_rms"]
        return np.linalg.norm(zs - mu, axis=1) <= r, {"eps_r": r, "rule": "||z-μ_E|| ≤ r_rms(E)"}
    if mode == "p90":
        r = ge["r_p90"]
        return np.linalg.norm(zs - mu, axis=1) <= r, {"eps_r": r, "rule": "||z-μ_E|| ≤ r_p90(E)"}
    if mode == "ellipse2":
        cov_inv = np.linalg.inv(ge["cov"])
        d2 = np.einsum("ij,jk,ik->i", zs - mu, cov_inv, zs - mu)
        thr = 5.991  # χ²_2(0.95)
        return d2 <= thr, {"eps_r": float(np.sqrt(thr)), "rule": "Mahalanobis² ≤ χ²_2(0.95) Expert cov"}
    raise ValueError(mode)


def run_pipeline(X: np.ndarray, y: np.ndarray, layer: str, mode: str) -> dict[str, Any]:
    pca1 = PCA(n_components=2, random_state=0)
    Z1 = pca1.fit_transform(X)
    in_E, rule = inside_expert_pca2d(Z1, y, mode=mode)

    succ_idx = np.flatnonzero(y == 1)
    drop_idx = succ_idx[in_E]
    keep_mask = np.ones(len(y), dtype=bool)
    keep_mask[drop_idx] = False
    assert keep_mask[y == 0].all() and keep_mask[y == 2].all()

    Xf, yf = X[keep_mask], y[keep_mask]
    pca2 = PCA(n_components=2, random_state=0)
    Z2 = pca2.fit_transform(Xf)

    g1 = {name: class_geom(Z1, y, c) for c, name in [(0, "expert"), (1, "success"), (2, "failure")]}
    g2 = {name: class_geom(Z2, yf, c) for c, name in [(0, "expert"), (1, "success"), (2, "failure")]}
    stats = {
        "layer": layer,
        "mode": mode,
        "rule": rule["rule"],
        "eps_r_stage1": rule["eps_r"],
        "n_success_total": int((y == 1).sum()),
        "n_success_in_expert_pca2d_dropped": int(in_E.sum()),
        "n_success_remain": int((yf == 1).sum()),
        "frac_dropped": float(in_E.mean()) if len(in_E) else 0.0,
        "pc_var_stage1": pca1.explained_variance_ratio_.tolist(),
        "pc_var_stage2": pca2.explained_variance_ratio_.tolist(),
        "shift_es_s2": float(np.linalg.norm(g2["success"]["mu"] - g2["expert"]["mu"])),
        "shift_ef_s2": float(np.linalg.norm(g2["failure"]["mu"] - g2["expert"]["mu"])),
        "r_ratio_s_s2": g2["success"]["r_rms"] / max(g2["expert"]["r_rms"], 1e-8),
        "r_ratio_f_s2": g2["failure"]["r_rms"] / max(g2["expert"]["r_rms"], 1e-8),
    }
    return {
        "stats": stats,
        "Z1": Z1,
        "y1": y,
        "g1": g1,
        "in_E_success": in_E,
        "Z2": Z2,
        "y2": yf,
        "g2": g2,
        "drop_rule_r": rule["eps_r"],
    }


def style(ax: plt.Axes) -> None:
    ax.grid(True, color=C_GRID, lw=0.7)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(labelsize=7.5, colors=C_MUTED)


def draw_circles(ax: plt.Axes, g: dict[str, Any], var: np.ndarray, title: str, note: str, success_label: str) -> None:
    alphas = {"failure": 0.12, "expert": 0.22, "success": 0.22}
    for name in ("failure", "expert", "success"):
        cls = g[name]
        color = C[name]
        mu, r = cls["mu"], cls["r_rms"]
        ax.add_patch(Circle(mu, r, facecolor=color, edgecolor=color, alpha=alphas[name], lw=0, zorder=2))
        ax.add_patch(Circle(mu, r, facecolor="none", edgecolor=color, alpha=0.95, lw=2.1, zorder=5))
        ax.add_patch(Circle(mu, cls["r_p90"], facecolor="none", edgecolor=color, alpha=0.4, lw=1.1, ls="--", zorder=4))
        ax.plot(mu[0], mu[1], marker="o", ms=5.5, color=color, markeredgecolor="white", markeredgewidth=1.0, zorder=10, linestyle="None")
        lab = success_label if name == "success" else name.capitalize()
        ax.plot([], [], color=color, lw=2.1, label=f"{lab}  r={cls['r_rms']:.2f} (n={cls['n']})")
    e = g["expert"]["mu"]
    for name in ("success", "failure"):
        ax.annotate("", xy=g[name]["mu"], xytext=e, arrowprops=dict(arrowstyle="->", color=C_MUTED, lw=1.05), zorder=8)
    xs, ys = [], []
    for cls in g.values():
        r = cls["r_p90"]
        xs += [cls["mu"][0] - r, cls["mu"][0] + r]
        ys += [cls["mu"][1] - r, cls["mu"][1] + r]
    pad = 0.06 * max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, fontsize=10, fontweight="semibold", color=C_TEXT, loc="left", pad=6)
    ax.set_xlabel(f"PC1 ({100 * var[0]:.1f}%)", fontsize=8, color=C_MUTED)
    ax.set_ylabel(f"PC2 ({100 * var[1]:.1f}%)", fontsize=8, color=C_MUTED)
    style(ax)
    ax.text(0.02, 0.98, note, transform=ax.transAxes, ha="left", va="top", fontsize=6.8, color=C_MUTED,
            bbox=dict(boxstyle="round,pad=0.25", fc="#F7F8FA", ec="#E2E6EA", lw=0.6))
    ax.legend(frameon=False, fontsize=6.6, loc="lower right")


def draw_scatter_stage1(ax: plt.Axes, pack: dict[str, Any], var: np.ndarray, title: str, mode: str) -> None:
    rng = np.random.default_rng(3)
    Z, y, in_E = pack["Z1"], pack["y1"], pack["in_E_success"]
    max_n = 2200
    for c, name, color, alpha in [(2, "Failure", C["failure"], 0.14), (0, "Expert", C["expert"], 0.26)]:
        idx = np.flatnonzero(y == c)
        n = int(len(idx))
        if len(idx) > max_n:
            idx = rng.choice(idx, max_n, replace=False)
        ax.scatter(Z[idx, 0], Z[idx, 1], s=7, alpha=alpha, c=color, lw=0, rasterized=True, label=f"{name} (n={n})", zorder=2)
        ax.scatter([Z[y == c, 0].mean()], [Z[y == c, 1].mean()], s=70, c=color, marker="X", edgecolors="white", linewidths=1.0, zorder=9)
    succ = np.flatnonzero(y == 1)
    inside, outside = succ[in_E], succ[~in_E]
    for idx, color, name, alpha, z in [
        (inside, "#A8ABB0", f"Success∩Expert (drop n={len(inside)})", 0.28, 1),
        (outside, C["success"], f"Success outside (keep n={len(outside)})", 0.38, 3),
    ]:
        ii = idx if len(idx) <= max_n else rng.choice(idx, max_n, replace=False)
        ax.scatter(Z[ii, 0], Z[ii, 1], s=7, alpha=alpha, c=color, lw=0, rasterized=True, label=name, zorder=z)

    ge = pack["g1"]["expert"]
    if mode in ("rms", "p90"):
        r = pack["drop_rule_r"]
        ax.add_patch(Circle(ge["mu"], r, facecolor=C["expert"], edgecolor="none", alpha=0.06, zorder=0))
        ax.add_patch(Circle(ge["mu"], r, facecolor="none", edgecolor=C["expert"], lw=1.6, zorder=7))
    else:
        evals, evecs = ge["cov_evals"], ge["cov_evecs"]
        ang = float(np.degrees(np.arctan2(evecs[1, 0], evecs[0, 0])))
        thr = 5.991
        for face, lw, z in [(C["expert"], 0.0, 0), ("none", 1.6, 7)]:
            ax.add_patch(Ellipse(
                ge["mu"],
                width=2 * np.sqrt(evals[0] * thr),
                height=2 * np.sqrt(evals[1] * thr),
                angle=ang,
                facecolor=face if face != "none" else "none",
                edgecolor=C["expert"],
                alpha=0.06 if face != "none" else 0.95,
                lw=lw if face == "none" else 0.0,
                zorder=z,
            ))
    ax.set_title(title, fontsize=10, fontweight="semibold", color=C_TEXT, loc="left", pad=6)
    ax.set_xlabel(f"PC1 ({100 * var[0]:.1f}%)", fontsize=8, color=C_MUTED)
    ax.set_ylabel(f"PC2 ({100 * var[1]:.1f}%)", fontsize=8, color=C_MUTED)
    style(ax)
    st = pack["stats"]
    ax.text(
        0.02, 0.98,
        f"PCA#1 decide: drop {st['n_success_in_expert_pca2d_dropped']}/{st['n_success_total']} "
        f"({100 * st['frac_dropped']:.0f}%)\n{st['rule']}",
        transform=ax.transAxes, ha="left", va="top", fontsize=6.8, color=C_MUTED,
        bbox=dict(boxstyle="round,pad=0.25", fc="#F7F8FA", ec="#E2E6EA", lw=0.6),
    )
    ax.legend(frameon=False, fontsize=6.4, loc="best", markerscale=1.3)


def draw_scatter_stage2(ax: plt.Axes, pack: dict[str, Any], title: str) -> None:
    rng = np.random.default_rng(5)
    Z, y = pack["Z2"], pack["y2"]
    var = np.asarray(pack["stats"]["pc_var_stage2"])
    max_n = 2200
    for c, name, key in [(2, "Failure", "failure"), (0, "Expert", "expert"), (1, "Success remain", "success")]:
        idx = np.flatnonzero(y == c)
        n = int(len(idx))
        if n == 0:
            continue
        ii = idx if len(idx) <= max_n else rng.choice(idx, max_n, replace=False)
        ax.scatter(Z[ii, 0], Z[ii, 1], s=7, alpha=0.16 if c == 2 else 0.32, c=C[key], lw=0,
                   rasterized=True, label=f"{name} (n={n})", zorder=2 + c)
        zc = Z[y == c]
        ax.scatter([zc[:, 0].mean()], [zc[:, 1].mean()], s=70, c=C[key], marker="X",
                   edgecolors="white", linewidths=1.0, zorder=8)
    ax.set_title(title, fontsize=10, fontweight="semibold", color=C_TEXT, loc="left", pad=6)
    ax.set_xlabel(f"PC1 ({100 * var[0]:.1f}%)", fontsize=8, color=C_MUTED)
    ax.set_ylabel(f"PC2 ({100 * var[1]:.1f}%)", fontsize=8, color=C_MUTED)
    style(ax)
    st = pack["stats"]
    ax.text(
        0.02, 0.98,
        f"PCA#2 refit after drop  ·  Success remain={st['n_success_remain']}\n"
        f"shift E→S {st['shift_es_s2']:.2f}  E→F {st['shift_ef_s2']:.2f}",
        transform=ax.transAxes, ha="left", va="top", fontsize=6.8, color=C_MUTED,
        bbox=dict(boxstyle="round,pad=0.25", fc="#F7F8FA", ec="#E2E6EA", lw=0.6),
    )
    ax.legend(frameon=False, fontsize=6.5, loc="best", markerscale=1.3)


def make_figures(packs: dict[str, dict[str, Any]], mode: str, out_dir: Path, dpi: int, blurb: str) -> dict[str, str]:
    tag = "ellipse" if mode == "ellipse2" else mode
    paths: dict[str, str] = {}
    for layer, pack in packs.items():
        fig, axes = plt.subplots(2, 2, figsize=(12.8, 10.2), dpi=dpi)
        fig.subplots_adjust(left=0.06, right=0.98, top=0.90, bottom=0.06, wspace=0.22, hspace=0.28)
        fig.text(0.06, 0.955, f"{layer}: PCA-plane filter → drop Success∩Expert → refit PCA",
                 fontsize=13.5, fontweight="bold", color=C_TEXT)
        fig.text(0.06, 0.925, blurb, fontsize=7.8, color=C_MUTED)
        v1 = np.asarray(pack["stats"]["pc_var_stage1"])
        v2 = np.asarray(pack["stats"]["pc_var_stage2"])
        draw_scatter_stage1(axes[0, 0], pack, v1, "① PCA#1 scatter  ·  decide membership", mode)
        draw_circles(axes[0, 1], pack["g1"], v1, "① PCA#1 circles  ·  before drop",
                     note=f"all Success n={pack['stats']['n_success_total']}", success_label="Success (all)")
        draw_scatter_stage2(axes[1, 0], pack, "② PCA#2 scatter  ·  after drop + refit")
        draw_circles(axes[1, 1], pack["g2"], v2, "② PCA#2 circles  ·  after drop + refit",
                     note=f"Success remain n={pack['stats']['n_success_remain']}  (dropped {100 * pack['stats']['frac_dropped']:.0f}%)",
                     success_label="Success remain")
        stem = out_dir / f"fig_pca2d_filter_refit_{tag}_{layer}"
        fig.savefig(f"{stem}.png", dpi=dpi)
        fig.savefig(f"{stem}.pdf")
        plt.close(fig)
        paths[f"{tag}_{layer}"] = f"{stem}.png"

    fig, axes = plt.subplots(2, 2, figsize=(12.8, 10.0), dpi=dpi)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.90, bottom=0.06, wspace=0.22, hspace=0.28)
    fig.text(0.06, 0.955, "After PCA-plane filter  ·  refit PCA  ·  scatter + circles",
             fontsize=13.5, fontweight="bold", color=C_TEXT)
    fig.text(0.06, 0.925, blurb, fontsize=7.8, color=C_MUTED)
    for row, (layer, pack) in enumerate(packs.items()):
        v2 = np.asarray(pack["stats"]["pc_var_stage2"])
        draw_scatter_stage2(axes[row, 0], pack, f"{layer}  ·  PCA#2 scatter")
        draw_circles(
            axes[row, 1], pack["g2"], v2, f"{layer}  ·  PCA#2 circles",
            note=(f"drop {pack['stats']['n_success_in_expert_pca2d_dropped']}/"
                  f"{pack['stats']['n_success_total']} ({100 * pack['stats']['frac_dropped']:.0f}%) in PCA#1"),
            success_label="Success remain",
        )
    stem = out_dir / f"fig_pca2d_filter_refit_{tag}_overview"
    fig.savefig(f"{stem}.png", dpi=dpi)
    fig.savefig(f"{stem}.pdf")
    plt.close(fig)
    paths[f"{tag}_overview"] = f"{stem}.png"
    return paths


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    import sys

    sys.path.insert(0, str(ANALYSIS))
    from plot_frozen_l2_l3_pca import build_l2_concat, build_l3_transitions

    print("[data] rebuild L2/L3 features…", flush=True)
    X2, y2 = build_l2_concat(visual_features=args.visual_features, expert=args.expert, rollout=args.l2_rollout)
    X3, y3 = build_l3_transitions(expert=args.expert, rollout=args.l3_rollout, l3_meta=args.l3_meta)

    report: dict[str, Any] = {"figures": {}}
    blurbs = {
        "rms": "Membership in PCA#1: Success inside Expert RMS circle dropped; then PCA#2 refit on Expert + Success_remain + Failure.",
        "p90": "Membership in PCA#1: Success inside Expert p90 circle dropped; then PCA#2 refit.",
        "ellipse2": "Membership in PCA#1: Success inside Expert Mahalanobis χ²_2(0.95) ellipse dropped; then PCA#2 refit.",
    }
    for mode in args.modes:
        print(f"[run] mode={mode}", flush=True)
        packs = {
            "L2": run_pipeline(X2, y2, "L2", mode=mode),
            "L3": run_pipeline(X3, y3, "L3", mode=mode),
        }
        key = "ellipse" if mode == "ellipse2" else mode
        report[key] = {layer: packs[layer]["stats"] for layer in packs}
        paths = make_figures(packs, mode=mode, out_dir=out_dir, dpi=args.dpi, blurb=blurbs[mode])
        report["figures"].update(paths)
        for layer in packs:
            print(f"  {layer}: drop {packs[layer]['stats']['frac_dropped']*100:.0f}% "
                  f"remain={packs[layer]['stats']['n_success_remain']}", flush=True)

    (out_dir / "pca2d_filter_refit_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["figures"], indent=2))


if __name__ == "__main__":
    main()
