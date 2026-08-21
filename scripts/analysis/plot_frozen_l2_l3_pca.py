#!/usr/bin/env python3
"""PCA probe on the frozen L2-visual / L3-transition diversity features.

Rebuilds the same feature spaces used by
  results/frozen_l2_visual_l3_transition_20260809/
then projects Expert / Success / Failure into PC1–PC2.

Does not overwrite the frozen 1NN figures.

Example:
  conda activate web
  python scripts/analysis/plot_frozen_l2_l3_pca.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[2]
FROZEN = ROOT / "results/frozen_l2_visual_l3_transition_20260809"
DEFAULT_EXPERT = ROOT / "data/dexjoco/dexjoco_lerobot_datasets/water_plant"
DEFAULT_L2_ROLLOUT = ROOT / "data/water_plant_s0_b1_video_cfg_20260808_152243/rollout_raw_200"
DEFAULT_L3_ROLLOUT = ROOT / "data/water_plant_s0_rollout_b0_b1_20260718/rollout"
DEFAULT_VIS = FROZEN / "L2_visual_features_fulltraj.npz"
DEFAULT_L3_META = FROZEN / "L3_meta.json"

C_EXPERT = "#4C78A8"
C_SUCC = "#54A24B"
C_FAIL = "#E45756"
C_MUTED = "#5B6B7A"
C_TEXT = "#1F2A33"
C_GRID = "#E6E9ED"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--frozen-dir", type=Path, default=FROZEN)
    p.add_argument("--expert", type=Path, default=DEFAULT_EXPERT)
    p.add_argument("--l2-rollout", type=Path, default=DEFAULT_L2_ROLLOUT)
    p.add_argument("--l3-rollout", type=Path, default=DEFAULT_L3_ROLLOUT)
    p.add_argument("--visual-features", type=Path, default=DEFAULT_VIS)
    p.add_argument("--l3-meta", type=Path, default=DEFAULT_L3_META)
    p.add_argument("--out-dir", type=Path, default=None, help="default: <frozen>/pca_probe")
    p.add_argument("--max-points-per-class", type=int, default=3500, help="scatter subsample for clarity")
    p.add_argument("--seed", type=int, default=20260810)
    p.add_argument("--dpi", type=int, default=220)
    return p.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_episode(path: Path) -> dict[str, np.ndarray]:
    table = pq.read_table(path, columns=["observation.state", "action", "frame_index"])
    return {
        "state": np.asarray(table.column("observation.state").to_pylist(), dtype=np.float32),
        "action": np.asarray(table.column("action").to_pylist(), dtype=np.float32),
        "frame": np.asarray(table.column("frame_index").to_pylist(), dtype=np.int64),
    }


def fit_norm(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = X.mean(0)
    sd = X.std(0)
    sd = np.where(sd < 1e-6, 1.0, sd)
    return mu.astype(np.float32), sd.astype(np.float32)


def apply_norm(X: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return ((X - mu) / sd).astype(np.float32)


def build_sa_lookup(root: Path) -> dict[tuple[int, int], tuple[np.ndarray, np.ndarray]]:
    chunk = root / "data" / "chunk-000"
    out: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    for path in sorted(chunk.glob("episode_*.parquet")):
        ep = int(path.stem.split("_")[-1])
        arr = load_episode(path)
        for i, f in enumerate(arr["frame"]):
            out[(ep, int(f))] = (arr["state"][i], arr["action"][i])
    return out


def episode_paths(root: Path) -> dict[int, Path]:
    chunk = root / "data" / "chunk-000"
    return {int(p.stem.split("_")[-1]): p for p in sorted(chunk.glob("episode_*.parquet"))}


def build_l2_concat(
    *,
    visual_features: Path,
    expert: Path,
    rollout: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (X_concat_zscored, labels) with labels in {0,1,2}=expert/succ/fail."""
    pack = np.load(visual_features, allow_pickle=True)
    src = np.asarray(pack["source"]).astype(str)
    feat_v = np.asarray(pack["feat"], dtype=np.float32)
    episode = np.asarray(pack["episode"], dtype=np.int64)
    frame = np.asarray(pack["frame"], dtype=np.int64)
    success = np.asarray(pack["success"], dtype=bool)

    print("[L2] building (s,a) lookups…", flush=True)
    sa_expert = build_sa_lookup(expert)
    sa_roll = build_sa_lookup(rollout)

    s_list, a_list, v_list, lab = [], [], [], []
    n_miss = 0
    for i in range(len(src)):
        key = (int(episode[i]), int(frame[i]))
        sa = sa_expert.get(key) if src[i] == "expert" else sa_roll.get(key)
        if sa is None:
            n_miss += 1
            continue
        s, a = sa
        s_list.append(s)
        a_list.append(a)
        v_list.append(feat_v[i])
        if src[i] == "expert":
            lab.append(0)
        elif bool(success[i]):
            lab.append(1)
        else:
            lab.append(2)

    S = np.asarray(s_list, dtype=np.float32)
    A = np.asarray(a_list, dtype=np.float32)
    V = np.asarray(v_list, dtype=np.float32)
    y = np.asarray(lab, dtype=np.int8)
    print(f"[L2] kept {len(S)} / {len(src)} (dropped {n_miss})", flush=True)

    m_e = y == 0
    mu_s, sd_s = fit_norm(S[m_e])
    mu_v, sd_v = fit_norm(V[m_e])
    mu_a, sd_a = fit_norm(A[m_e])
    X = np.concatenate(
        [apply_norm(S, mu_s, sd_s), apply_norm(V, mu_v, sd_v), apply_norm(A, mu_a, sd_a)],
        axis=1,
    )
    return X, y


def build_l3_transitions(
    *,
    expert: Path,
    rollout: Path,
    l3_meta: Path,
) -> tuple[np.ndarray, np.ndarray]:
    meta = json.loads(l3_meta.read_text())
    proto = meta["protocol"]
    stride = int(proto["stride"])
    lag = int(proto["transition_lag"])
    succ_ids = set(int(x) for x in proto["rollout_success_episodes_sampled"])
    fail_ids = set(int(x) for x in proto["rollout_failure_episodes"])

    def collect(root: Path, ep_filter: set[int] | None, label: int) -> tuple[list[np.ndarray], list[int]]:
        paths = episode_paths(root)
        feats: list[np.ndarray] = []
        labs: list[int] = []
        for ep, path in paths.items():
            if ep_filter is not None and ep not in ep_filter:
                continue
            arr = load_episode(path)
            n = len(arr["state"])
            idx = np.arange(0, n, stride, dtype=np.int64)
            for t in idx:
                t2 = int(t) + lag
                if t2 >= n:
                    continue
                e = np.concatenate([arr["state"][t], arr["action"][t], arr["state"][t2]], axis=0)
                feats.append(e)
                labs.append(label)
        return feats, labs

    print("[L3] collecting transitions…", flush=True)
    fe, ye = collect(expert, None, 0)
    fs, ys = collect(rollout, succ_ids, 1)
    ff, yf = collect(rollout, fail_ids, 2)
    raw = np.asarray(fe + fs + ff, dtype=np.float32)
    y = np.asarray(ye + ys + yf, dtype=np.int8)
    print(f"[L3] n expert/succ/fail = {(y==0).sum()}/{(y==1).sum()}/{(y==2).sum()}", flush=True)

    mu, sd = fit_norm(raw[y == 0])
    X = apply_norm(raw, mu, sd)
    return X, y


def subsample_mask(y: np.ndarray, max_per_class: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    keep = np.zeros(len(y), dtype=bool)
    for c in (0, 1, 2):
        idx = np.flatnonzero(y == c)
        if len(idx) > max_per_class:
            idx = rng.choice(idx, size=max_per_class, replace=False)
        keep[idx] = True
    return keep


def class_stats(Z: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    names = {0: "expert", 1: "success", 2: "failure"}
    cents: dict[str, np.ndarray] = {}
    out: dict[str, Any] = {"n": {}, "mean_pc": {}, "std_pc": {}, "centroid_l2": {}}
    for c, name in names.items():
        m = y == c
        z = Z[m]
        out["n"][name] = int(m.sum())
        out["mean_pc"][name] = [float(z[:, 0].mean()), float(z[:, 1].mean())]
        out["std_pc"][name] = [float(z[:, 0].std()), float(z[:, 1].std())]
        cents[name] = z.mean(0)
    pairs = [("expert", "success"), ("expert", "failure"), ("success", "failure")]
    for a, b in pairs:
        d = float(np.linalg.norm(cents[a] - cents[b]))
        # normalize by pooled within-class RMS radius in PC1–2
        ra = float(np.sqrt(np.mean(np.sum((Z[y == {"expert": 0, "success": 1, "failure": 2}[a]] - cents[a]) ** 2, 1))))
        rb = float(np.sqrt(np.mean(np.sum((Z[y == {"expert": 0, "success": 1, "failure": 2}[b]] - cents[b]) ** 2, 1))))
        out["centroid_l2"][f"{a}_vs_{b}"] = {
            "distance": d,
            "sep_over_pooled_rms": float(d / max(0.5 * (ra + rb), 1e-8)),
            "rms_a": ra,
            "rms_b": rb,
        }
    return out


def plot_pca_panel(
    ax: plt.Axes,
    Z: np.ndarray,
    y: np.ndarray,
    *,
    title: str,
    var: np.ndarray,
    max_per_class: int,
    seed: int,
) -> None:
    keep = subsample_mask(y, max_per_class, seed)
    order = [(2, C_FAIL, "Failure"), (1, C_SUCC, "Success"), (0, C_EXPERT, "Expert")]
    for c, color, name in order:
        m = keep & (y == c)
        ax.scatter(
            Z[m, 0],
            Z[m, 1],
            s=8,
            alpha=0.18 if c == 2 else 0.28,
            c=color,
            linewidths=0,
            rasterized=True,
            label=f"{name} (n={int((y == c).sum())})",
            zorder=2 if c else 3,
        )
    # centroids
    for c, color, name in order:
        m = y == c
        ax.scatter(
            [Z[m, 0].mean()],
            [Z[m, 1].mean()],
            s=90,
            c=color,
            edgecolors="white",
            linewidths=1.2,
            marker="X",
            zorder=5,
        )
    ax.set_title(title, fontsize=11, fontweight="semibold", color=C_TEXT, loc="left", pad=8)
    ax.set_xlabel(f"PC 1 ({100 * var[0]:.1f}%)", fontsize=9, color=C_MUTED)
    ax.set_ylabel(f"PC 2 ({100 * var[1]:.1f}%)", fontsize=9, color=C_MUTED)
    ax.grid(True, color=C_GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(labelsize=8, colors=C_MUTED)
    ax.legend(frameon=False, fontsize=8, loc="best", markerscale=1.8)


def plot_density_panel(
    ax: plt.Axes,
    Z: np.ndarray,
    y: np.ndarray,
    *,
    title: str,
    var: np.ndarray,
) -> None:
    """2D histogram contours / filled density per class for region readability."""
    # shared grid from 2–98% quantiles
    lo = np.quantile(Z, 0.02, axis=0)
    hi = np.quantile(Z, 0.98, axis=0)
    xedges = np.linspace(lo[0], hi[0], 70)
    yedges = np.linspace(lo[1], hi[1], 70)
    xx = 0.5 * (xedges[:-1] + xedges[1:])
    yy = 0.5 * (yedges[:-1] + yedges[1:])
    XX, YY = np.meshgrid(xx, yy)

    for c, color, name, levels in [
        (0, C_EXPERT, "Expert", [0.35, 0.55, 0.75]),
        (1, C_SUCC, "Success", [0.35, 0.55, 0.75]),
        (2, C_FAIL, "Failure", [0.25, 0.45, 0.65]),
    ]:
        m = y == c
        H, _, _ = np.histogram2d(Z[m, 0], Z[m, 1], bins=[xedges, yedges], density=True)
        H = H.T
        if H.max() <= 0:
            continue
        Hn = H / H.max()
        ax.contourf(XX, YY, Hn, levels=[0.15, 0.35, 0.55, 0.8, 1.01], colors=[color], alpha=0.18, zorder=1)
        ax.contour(XX, YY, Hn, levels=levels, colors=[color], linewidths=1.4, alpha=0.95, zorder=2)
        ax.plot([], [], color=color, lw=2.0, label=name)

    ax.set_title(title, fontsize=11, fontweight="semibold", color=C_TEXT, loc="left", pad=8)
    ax.set_xlabel(f"PC 1 ({100 * var[0]:.1f}%)", fontsize=9, color=C_MUTED)
    ax.set_ylabel(f"PC 2 ({100 * var[1]:.1f}%)", fontsize=9, color=C_MUTED)
    ax.grid(True, color=C_GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(labelsize=8, colors=C_MUTED)
    ax.legend(frameon=False, fontsize=8, loc="best")


def run_pca_views(X: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    """Joint PCA (all points) and Expert-fitted PCA (project others)."""
    views: dict[str, Any] = {}

    pca_all = PCA(n_components=2, random_state=0)
    Z_all = pca_all.fit_transform(X)
    views["joint"] = {
        "Z": Z_all,
        "var": pca_all.explained_variance_ratio_,
        "stats": class_stats(Z_all, y),
    }

    pca_e = PCA(n_components=2, random_state=0)
    pca_e.fit(X[y == 0])
    Z_e = pca_e.transform(X)
    views["expert_fit"] = {
        "Z": Z_e,
        "var": pca_e.explained_variance_ratio_,
        "stats": class_stats(Z_e, y),
    }
    return views


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir or (args.frozen_dir / "pca_probe")
    out_dir.mkdir(parents=True, exist_ok=True)

    X2, y2 = build_l2_concat(
        visual_features=args.visual_features,
        expert=args.expert,
        rollout=args.l2_rollout,
    )
    X3, y3 = build_l3_transitions(
        expert=args.expert,
        rollout=args.l3_rollout,
        l3_meta=args.l3_meta,
    )

    views2 = run_pca_views(X2, y2)
    views3 = run_pca_views(X3, y3)

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

    # Main figure: joint PCA scatter for L2 + L3
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.6), dpi=args.dpi)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.82, bottom=0.12, wspace=0.22)
    fig.text(
        0.07,
        0.93,
        "PCA on frozen diversity features  ·  Expert / Success / Failure",
        fontsize=14,
        fontweight="bold",
        color=C_TEXT,
    )
    fig.text(
        0.07,
        0.875,
        "Same protocols as frozen 1NN figures.  PCA fit on all points jointly.  "
        "Markers × = class centroids.  Points subsampled for visibility.",
        fontsize=8.2,
        color=C_MUTED,
    )
    plot_pca_panel(
        axes[0],
        views2["joint"]["Z"],
        y2,
        title=r"L2–Visual  ·  $\mathrm{concat}(s,z_{\mathrm{VAE}},a)$",
        var=views2["joint"]["var"],
        max_per_class=args.max_points_per_class,
        seed=args.seed,
    )
    plot_pca_panel(
        axes[1],
        views3["joint"]["Z"],
        y3,
        title=r"L3 transition  ·  $[s_t,a_t,s_{t+5}]$",
        var=views3["joint"]["var"],
        max_per_class=args.max_points_per_class,
        seed=args.seed + 1,
    )
    stem = out_dir / "fig_pca_joint_scatter_L2_L3"
    fig.savefig(f"{stem}.png", dpi=args.dpi)
    fig.savefig(f"{stem}.pdf")
    plt.close(fig)

    # Density / region figure
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.6), dpi=args.dpi)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.82, bottom=0.12, wspace=0.22)
    fig.text(
        0.07,
        0.93,
        "PCA density regions  ·  do classes occupy distinct areas?",
        fontsize=14,
        fontweight="bold",
        color=C_TEXT,
    )
    fig.text(
        0.07,
        0.875,
        "Filled/outline contours = relative 2D density in PC1–PC2 (joint PCA).  "
        "Clear 2–3 blobs would appear as non-overlapping high-density cores.",
        fontsize=8.2,
        color=C_MUTED,
    )
    plot_density_panel(
        axes[0],
        views2["joint"]["Z"],
        y2,
        title="L2–Visual density",
        var=views2["joint"]["var"],
    )
    plot_density_panel(
        axes[1],
        views3["joint"]["Z"],
        y3,
        title="L3 transition density",
        var=views3["joint"]["var"],
    )
    stem_d = out_dir / "fig_pca_joint_density_L2_L3"
    fig.savefig(f"{stem_d}.png", dpi=args.dpi)
    fig.savefig(f"{stem_d}.pdf")
    plt.close(fig)

    # Expert-fit PCA (aligned with 1NN gallery)
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.6), dpi=args.dpi)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.82, bottom=0.12, wspace=0.22)
    fig.text(
        0.07,
        0.93,
        "PCA fit on Expert only  ·  then project Success / Failure",
        fontsize=14,
        fontweight="bold",
        color=C_TEXT,
    )
    fig.text(
        0.07,
        0.875,
        "Closer to the frozen 1NN setup (Expert defines the coordinate system).",
        fontsize=8.2,
        color=C_MUTED,
    )
    plot_pca_panel(
        axes[0],
        views2["expert_fit"]["Z"],
        y2,
        title="L2–Visual  ·  Expert-fit PCA",
        var=views2["expert_fit"]["var"],
        max_per_class=args.max_points_per_class,
        seed=args.seed + 2,
    )
    plot_pca_panel(
        axes[1],
        views3["expert_fit"]["Z"],
        y3,
        title="L3 transition  ·  Expert-fit PCA",
        var=views3["expert_fit"]["var"],
        max_per_class=args.max_points_per_class,
        seed=args.seed + 3,
    )
    stem_e = out_dir / "fig_pca_expertfit_scatter_L2_L3"
    fig.savefig(f"{stem_e}.png", dpi=args.dpi)
    fig.savefig(f"{stem_e}.pdf")
    plt.close(fig)

    report = {
        "L2": {
            "feature": "concat(s, z_VAE, a) z-scored by Expert",
            "dim": int(X2.shape[1]),
            "joint": {
                "explained_variance_ratio": views2["joint"]["var"].tolist(),
                "stats": views2["joint"]["stats"],
            },
            "expert_fit": {
                "explained_variance_ratio": views2["expert_fit"]["var"].tolist(),
                "stats": views2["expert_fit"]["stats"],
            },
        },
        "L3": {
            "feature": "[s_t, a_t, s_{t+5}] z-scored by Expert",
            "dim": int(X3.shape[1]),
            "joint": {
                "explained_variance_ratio": views3["joint"]["var"].tolist(),
                "stats": views3["joint"]["stats"],
            },
            "expert_fit": {
                "explained_variance_ratio": views3["expert_fit"]["var"].tolist(),
                "stats": views3["expert_fit"]["stats"],
            },
        },
        "figures": {
            "joint_scatter": f"{stem}.png",
            "joint_density": f"{stem_d}.png",
            "expertfit_scatter": f"{stem_e}.png",
        },
        "interpretation_hint": (
            "sep_over_pooled_rms ≫ 1 suggests well-separated centroids relative to "
            "within-class spread; values near ≤1 mean heavy overlap in PC1–PC2."
        ),
    }
    (out_dir / "pca_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("figures",)}, indent=2))
    for layer in ("L2", "L3"):
        s = report[layer]["joint"]["stats"]["centroid_l2"]
        print(f"[{layer} joint] centroid sep / pooled RMS:")
        for k, v in s.items():
            print(f"  {k}: dist={v['distance']:.2f}  sep={v['sep_over_pooled_rms']:.2f}")
    print(f"[out] {out_dir}")


if __name__ == "__main__":
    main()
