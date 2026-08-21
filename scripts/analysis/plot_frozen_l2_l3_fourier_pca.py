#!/usr/bin/env python3
"""Temporal Fourier spectrum + episode-level PCA on frozen L2-visual / L3 features.

Same feature protocols as plot_frozen_l2_l3_pca.py:
  L2: concat(s, z_VAE, a) z-scored on Expert
  L3: [s_t, a_t, s_{t+5}] z-scored on Expert (stride=5)

Pipeline per layer (unit = episode, never frame scatter):
  1. Rebuild per-episode feature trajectories.
  2. Mean PSD vs frequency by class.
  3. Episode log-PSD (full band) → PC1–PC2.
  4. Episode high-pass: log-PSD restricted to f >= cutoff → PC1–PC2.

Example:
  conda activate web
  python scripts/analysis/plot_frozen_l2_l3_fourier_pca.py
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

C_EXPERT = "#2077B4"  # strong blue
C_SUCC = "#E08B00"  # amber/orange — distinct from blue & red
C_FAIL = "#C51B8A"  # magenta — distinct from amber
C_MUTED = "#5B6B7A"
C_TEXT = "#1F2A33"
C_GRID = "#E6E9ED"
C_CUT = "#6B6B6B"

CLASS_NAMES = {0: "expert", 1: "success", 2: "failure"}
CLASS_COLORS = {0: C_EXPERT, 1: C_SUCC, 2: C_FAIL}
CLASS_LABELS = {0: "Expert", 1: "Success", 2: "Failure"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--frozen-dir", type=Path, default=FROZEN)
    p.add_argument("--expert", type=Path, default=DEFAULT_EXPERT)
    p.add_argument("--l2-rollout", type=Path, default=DEFAULT_L2_ROLLOUT)
    p.add_argument("--l3-rollout", type=Path, default=DEFAULT_L3_ROLLOUT)
    p.add_argument("--visual-features", type=Path, default=DEFAULT_VIS)
    p.add_argument("--l3-meta", type=Path, default=DEFAULT_L3_META)
    p.add_argument("--out-dir", type=Path, default=None, help="default: <frozen>/fourier_pca")
    p.add_argument(
        "--highpass-frac",
        type=float,
        default=0.25,
        help="keep FFT bins with f >= frac * Nyquist (0.5 cycles/frame). Default 0.25 → f>=0.125",
    )
    p.add_argument("--freq-grid", type=int, default=128, help="common frequency grid size for PSD mean")
    p.add_argument("--min-frames", type=int, default=16, help="skip short episodes for FFT/highpass")
    p.add_argument("--max-points-per-class", type=int, default=3500)
    p.add_argument("--seed", type=int, default=20260812)
    p.add_argument("--dpi", type=int, default=220)
    return p.parse_args()


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


def build_l2_trajectories(
    *,
    visual_features: Path,
    expert: Path,
    rollout: Path,
) -> list[dict[str, Any]]:
    """Return list of {label, episode, X:(T,D)} trajectories."""
    pack = np.load(visual_features, allow_pickle=True)
    src = np.asarray(pack["source"]).astype(str)
    feat_v = np.asarray(pack["feat"], dtype=np.float32)
    episode = np.asarray(pack["episode"], dtype=np.int64)
    frame = np.asarray(pack["frame"], dtype=np.int64)
    success = np.asarray(pack["success"], dtype=bool)

    print("[L2] building (s,a) lookups…", flush=True)
    sa_expert = build_sa_lookup(expert)
    sa_roll = build_sa_lookup(rollout)

    # accumulate raw rows then group
    rows: dict[tuple[str, int], list[tuple[int, np.ndarray, np.ndarray, np.ndarray, int]]] = {}
    n_miss = 0
    for i in range(len(src)):
        key = (int(episode[i]), int(frame[i]))
        sa = sa_expert.get(key) if src[i] == "expert" else sa_roll.get(key)
        if sa is None:
            n_miss += 1
            continue
        s, a = sa
        if src[i] == "expert":
            lab = 0
            group = ("expert", int(episode[i]))
        elif bool(success[i]):
            lab = 1
            group = ("success", int(episode[i]))
        else:
            lab = 2
            group = ("failure", int(episode[i]))
        rows.setdefault(group, []).append((int(frame[i]), s, feat_v[i], a, lab))

    # expert z-score from all expert frames
    S_e, V_e, A_e = [], [], []
    for (kind, _), items in rows.items():
        if kind != "expert":
            continue
        for _, s, v, a, _ in items:
            S_e.append(s)
            V_e.append(v)
            A_e.append(a)
    mu_s, sd_s = fit_norm(np.asarray(S_e, dtype=np.float32))
    mu_v, sd_v = fit_norm(np.asarray(V_e, dtype=np.float32))
    mu_a, sd_a = fit_norm(np.asarray(A_e, dtype=np.float32))

    trajs: list[dict[str, Any]] = []
    for (kind, ep), items in rows.items():
        items = sorted(items, key=lambda t: t[0])
        S = np.asarray([t[1] for t in items], dtype=np.float32)
        V = np.asarray([t[2] for t in items], dtype=np.float32)
        A = np.asarray([t[3] for t in items], dtype=np.float32)
        lab = int(items[0][4])
        X = np.concatenate(
            [apply_norm(S, mu_s, sd_s), apply_norm(V, mu_v, sd_v), apply_norm(A, mu_a, sd_a)],
            axis=1,
        )
        trajs.append({"label": lab, "episode": ep, "kind": kind, "X": X})
    print(
        f"[L2] trajectories={len(trajs)} frames={sum(t['X'].shape[0] for t in trajs)} "
        f"(dropped {n_miss})",
        flush=True,
    )
    return trajs


def build_l3_trajectories(
    *,
    expert: Path,
    rollout: Path,
    l3_meta: Path,
) -> list[dict[str, Any]]:
    meta = json.loads(l3_meta.read_text())
    proto = meta["protocol"]
    stride = int(proto["stride"])
    lag = int(proto["transition_lag"])
    succ_ids = set(int(x) for x in proto["rollout_success_episodes_sampled"])
    fail_ids = set(int(x) for x in proto["rollout_failure_episodes"])

    def collect(root: Path, ep_filter: set[int] | None, label: int, kind: str) -> list[dict[str, Any]]:
        paths = episode_paths(root)
        out: list[dict[str, Any]] = []
        for ep, path in paths.items():
            if ep_filter is not None and ep not in ep_filter:
                continue
            arr = load_episode(path)
            n = len(arr["state"])
            feats = []
            for t in np.arange(0, n, stride, dtype=np.int64):
                t2 = int(t) + lag
                if t2 >= n:
                    continue
                feats.append(np.concatenate([arr["state"][t], arr["action"][t], arr["state"][t2]], 0))
            if not feats:
                continue
            out.append({"label": label, "episode": ep, "kind": kind, "X_raw": np.asarray(feats, np.float32)})
        return out

    print("[L3] collecting transitions…", flush=True)
    raw = (
        collect(expert, None, 0, "expert")
        + collect(rollout, succ_ids, 1, "success")
        + collect(rollout, fail_ids, 2, "failure")
    )
    Xe = np.concatenate([t["X_raw"] for t in raw if t["label"] == 0], axis=0)
    mu, sd = fit_norm(Xe)
    trajs = []
    for t in raw:
        trajs.append(
            {
                "label": t["label"],
                "episode": t["episode"],
                "kind": t["kind"],
                "X": apply_norm(t["X_raw"], mu, sd),
            }
        )
    counts = {c: sum(1 for t in trajs if t["label"] == c) for c in (0, 1, 2)}
    print(f"[L3] trajectories expert/succ/fail = {counts[0]}/{counts[1]}/{counts[2]}", flush=True)
    return trajs


def episode_psd(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One-sided PSD averaged over feature dims. Returns (freq cycles/frame, psd)."""
    T, D = X.shape
    # remove per-dim mean so DC does not dominate comparison of shape
    Y = X - X.mean(axis=0, keepdims=True)
    # Hann window
    w = np.hanning(T).astype(np.float64)
    w = w / np.sqrt((w**2).mean())
    Yw = Y * w[:, None]
    F = np.fft.rfft(Yw, axis=0)
    # power per cycle; normalize by T so different lengths comparable in density sense
    psd = (np.abs(F) ** 2).mean(axis=1) / T
    freq = np.fft.rfftfreq(T, d=1.0)
    return freq.astype(np.float64), psd.astype(np.float64)


def aggregate_psd(
    trajs: list[dict[str, Any]],
    *,
    freq_grid: int,
    min_frames: int,
) -> dict[str, Any]:
    f_common = np.linspace(0.0, 0.5, freq_grid)
    by_class: dict[int, list[np.ndarray]] = {0: [], 1: [], 2: []}
    n_used = {0: 0, 1: 0, 2: 0}
    for t in trajs:
        X = t["X"]
        if X.shape[0] < min_frames:
            continue
        f, p = episode_psd(X)
        # interpolate onto common grid (exclude exact Nyquist duplicates)
        p_i = np.interp(f_common, f, p)
        by_class[int(t["label"])].append(p_i)
        n_used[int(t["label"])] += 1

    out: dict[str, Any] = {"freq": f_common, "n_episodes": n_used, "mean": {}, "std": {}, "cum_energy": {}}
    for c in (0, 1, 2):
        name = CLASS_NAMES[c]
        stack = np.asarray(by_class[c], dtype=np.float64)
        mean = stack.mean(0)
        std = stack.std(0)
        out["mean"][name] = mean
        out["std"][name] = std
        # cumulative energy fraction vs frequency
        cum = np.cumsum(mean)
        cum = cum / max(cum[-1], 1e-12)
        out["cum_energy"][name] = cum
    return out


def highpass_trajectory(X: np.ndarray, cutoff_cycles: float) -> np.ndarray:
    """Zero FFT bins with f < cutoff; return real residual (same length)."""
    T, D = X.shape
    Y = X.astype(np.float64) - X.mean(axis=0, keepdims=True)
    F = np.fft.rfft(Y, axis=0)
    freq = np.fft.rfftfreq(T, d=1.0)
    F[freq < cutoff_cycles, :] = 0.0
    # keep DC removed; residual is high-frequency content
    R = np.fft.irfft(F, n=T, axis=0)
    return R.astype(np.float32)


def episode_hf_log_psd_matrix(
    trajs: list[dict[str, Any]],
    *,
    freq_grid: int,
    min_frames: int,
    cutoff_cycles: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Each episode → log10(PSD) on HF band only (f >= cutoff). Returns X, y, hf_frac, f_hf."""
    f_common = np.linspace(0.0, 0.5, freq_grid)
    hf_mask = f_common >= cutoff_cycles
    f_hf = f_common[hf_mask]
    xs, ys, hfs = [], [], []
    for t in trajs:
        X = t["X"]
        if X.shape[0] < min_frames:
            continue
        f, p = episode_psd(X)
        p_i = np.maximum(np.interp(f_common, f, p), 1e-16)
        xs.append(np.log10(p_i[hf_mask]))
        ys.append(int(t["label"]))
        hfs.append(float(p_i[hf_mask].sum() / p_i.sum()))
    return (
        np.asarray(xs, dtype=np.float64),
        np.asarray(ys, dtype=np.int8),
        np.asarray(hfs, dtype=np.float64),
        f_hf.astype(np.float64),
    )


def episode_highpass_rms_matrix(
    trajs: list[dict[str, Any]],
    *,
    cutoff_cycles: float,
    min_frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Each episode → per-dim RMS of FFT high-pass residual. Returns X, y."""
    xs, ys = [], []
    for t in trajs:
        X = t["X"]
        if X.shape[0] < min_frames:
            continue
        R = highpass_trajectory(X, cutoff_cycles)
        xs.append(np.sqrt(np.mean(R.astype(np.float64) ** 2, axis=0)))
        ys.append(int(t["label"]))
    return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.int8)


def class_stats(Z: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    cents: dict[str, np.ndarray] = {}
    out: dict[str, Any] = {"n": {}, "mean_pc": {}, "std_pc": {}, "centroid_l2": {}}
    for c, name in CLASS_NAMES.items():
        m = y == c
        z = Z[m]
        out["n"][name] = int(m.sum())
        out["mean_pc"][name] = [float(z[:, 0].mean()), float(z[:, 1].mean())]
        out["std_pc"][name] = [float(z[:, 0].std()), float(z[:, 1].std())]
        cents[name] = z.mean(0)
    for a, b in [("expert", "success"), ("expert", "failure"), ("success", "failure")]:
        d = float(np.linalg.norm(cents[a] - cents[b]))
        ia = {"expert": 0, "success": 1, "failure": 2}[a]
        ib = {"expert": 0, "success": 1, "failure": 2}[b]
        ra = float(np.sqrt(np.mean(np.sum((Z[y == ia] - cents[a]) ** 2, 1))))
        rb = float(np.sqrt(np.mean(np.sum((Z[y == ib] - cents[b]) ** 2, 1))))
        out["centroid_l2"][f"{a}_vs_{b}"] = {
            "distance": d,
            "sep_over_pooled_rms": float(d / max(0.5 * (ra + rb), 1e-8)),
            "rms_a": ra,
            "rms_b": rb,
        }
    return out


def hf_energy_fraction(trajs: list[dict[str, Any]], cutoff: float, min_frames: int) -> dict[str, float]:
    """Mean fraction of trajectory energy in f >= cutoff, by class."""
    acc = {0: [], 1: [], 2: []}
    for t in trajs:
        X = t["X"]
        if X.shape[0] < min_frames:
            continue
        f, p = episode_psd(X)
        total = float(p.sum())
        if total <= 0:
            continue
        frac = float(p[f >= cutoff].sum() / total)
        acc[int(t["label"])].append(frac)
    return {CLASS_NAMES[c]: float(np.mean(v)) if v else float("nan") for c, v in acc.items()}


def plot_spectrum_panel(ax: plt.Axes, psd: dict[str, Any], *, title: str, cutoff: float) -> None:
    f = psd["freq"]
    for c in (0, 1, 2):
        name = CLASS_NAMES[c]
        mean = psd["mean"][name]
        std = psd["std"][name]
        n = psd["n_episodes"][c]
        ax.plot(f, mean, color=CLASS_COLORS[c], lw=2.0, label=f"{CLASS_LABELS[c]} (n_ep={n})")
        ax.fill_between(f, np.maximum(mean - std, 1e-16), mean + std, color=CLASS_COLORS[c], alpha=0.12, linewidth=0)
    ax.axvline(cutoff, color=C_CUT, ls="--", lw=1.2, label=f"high-pass cut f≥{cutoff:.3f}")
    ax.set_yscale("log")
    ax.set_xlim(0.0, 0.5)
    ax.set_title(title, fontsize=11, fontweight="semibold", color=C_TEXT, loc="left", pad=8)
    ax.set_xlabel("Frequency (cycles / frame)", fontsize=9, color=C_MUTED)
    ax.set_ylabel("Mean PSD (log)", fontsize=9, color=C_MUTED)
    ax.grid(True, color=C_GRID, linewidth=0.7, which="both")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(labelsize=8, colors=C_MUTED)
    ax.legend(frameon=False, fontsize=7.5, loc="upper right")


def plot_cum_energy_panel(ax: plt.Axes, psd: dict[str, Any], *, title: str, cutoff: float) -> None:
    f = psd["freq"]
    for c in (0, 1, 2):
        name = CLASS_NAMES[c]
        ax.plot(f, psd["cum_energy"][name], color=CLASS_COLORS[c], lw=2.0, label=CLASS_LABELS[c])
    ax.axvline(cutoff, color=C_CUT, ls="--", lw=1.2)
    ax.set_xlim(0.0, 0.5)
    ax.set_ylim(0.0, 1.02)
    ax.set_title(title, fontsize=11, fontweight="semibold", color=C_TEXT, loc="left", pad=8)
    ax.set_xlabel("Frequency (cycles / frame)", fontsize=9, color=C_MUTED)
    ax.set_ylabel("Cumulative energy fraction", fontsize=9, color=C_MUTED)
    ax.grid(True, color=C_GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(labelsize=8, colors=C_MUTED)
    ax.legend(frameon=False, fontsize=8, loc="lower right")


def episode_log_psd_matrix(
    trajs: list[dict[str, Any]],
    *,
    freq_grid: int,
    min_frames: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Each episode → log10(PSD) on common freq grid. Returns X, y, freq."""
    f_common = np.linspace(0.0, 0.5, freq_grid)
    xs, ys = [], []
    for t in trajs:
        X = t["X"]
        if X.shape[0] < min_frames:
            continue
        f, p = episode_psd(X)
        p_i = np.interp(f_common, f, p)
        p_i = np.maximum(p_i, 1e-16)
        xs.append(np.log10(p_i))
        ys.append(int(t["label"]))
    return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.int8), f_common


def plot_episode_pca_panel(
    ax: plt.Axes,
    Z: np.ndarray,
    y: np.ndarray,
    *,
    title: str,
    var: np.ndarray,
    color_by: np.ndarray | None = None,
    cbar_label: str | None = None,
) -> None:
    order = [(2, C_FAIL, "Failure"), (1, C_SUCC, "Success"), (0, C_EXPERT, "Expert")]
    if color_by is None:
        for c, color, name in order:
            m = y == c
            ax.scatter(
                Z[m, 0],
                Z[m, 1],
                s=28,
                alpha=0.75,
                c=color,
                edgecolors="white",
                linewidths=0.3,
                label=f"{name} (n={int(m.sum())})",
                zorder=2 if c else 3,
            )
        for c, color, _ in order:
            m = y == c
            ax.scatter(
                [Z[m, 0].mean()],
                [Z[m, 1].mean()],
                s=110,
                c=color,
                edgecolors="white",
                linewidths=1.2,
                marker="X",
                zorder=5,
            )
        ax.legend(frameon=False, fontsize=8, loc="best")
    else:
        sc = ax.scatter(
            Z[:, 0],
            Z[:, 1],
            s=28,
            c=color_by,
            cmap="viridis",
            alpha=0.85,
            edgecolors="white",
            linewidths=0.3,
        )
        cb = ax.figure.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        if cbar_label:
            cb.set_label(cbar_label, fontsize=8, color=C_MUTED)
        cb.ax.tick_params(labelsize=7, colors=C_MUTED)
    ax.set_title(title, fontsize=11, fontweight="semibold", color=C_TEXT, loc="left", pad=8)
    ax.set_xlabel(f"PC 1 ({100 * var[0]:.1f}%)", fontsize=9, color=C_MUTED)
    ax.set_ylabel(f"PC 2 ({100 * var[1]:.1f}%)", fontsize=9, color=C_MUTED)
    ax.grid(True, color=C_GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(labelsize=8, colors=C_MUTED)


def run_layer(
    name: str,
    title_feat: str,
    trajs: list[dict[str, Any]],
    *,
    args: argparse.Namespace,
    out_dir: Path,
) -> dict[str, Any]:
    cutoff = float(args.highpass_frac) * 0.5  # frac of Nyquist
    print(f"[{name}] episode PSD + episode HF PCA (cutoff={cutoff:.4f} cyc/frame)…", flush=True)

    psd = aggregate_psd(trajs, freq_grid=args.freq_grid, min_frames=args.min_frames)
    hf_frac = hf_energy_fraction(trajs, cutoff, args.min_frames)

    # Full-band episode log-PSD PCA
    Xs, ys, _f_grid = episode_log_psd_matrix(
        trajs, freq_grid=args.freq_grid, min_frames=args.min_frames
    )
    Xhf, yhf, hfs, _f_hf = episode_hf_log_psd_matrix(
        trajs,
        freq_grid=args.freq_grid,
        min_frames=args.min_frames,
        cutoff_cycles=cutoff,
    )
    # sanity: same episode set / order
    assert len(ys) == len(yhf) and np.array_equal(ys, yhf)

    pca_s = PCA(n_components=2, random_state=0)
    Zs = pca_s.fit_transform(Xs)
    stats_s = class_stats(Zs, ys)

    pca_h = PCA(n_components=2, random_state=0)
    Zh = pca_h.fit_transform(Xhf)
    stats_h = class_stats(Zh, yhf)

    # Spectrum figure
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8), dpi=args.dpi)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.82, bottom=0.14, wspace=0.28)
    fig.text(0.07, 0.93, f"{name} temporal Fourier spectrum  ·  {title_feat}", fontsize=13, fontweight="bold", color=C_TEXT)
    fig.text(
        0.07,
        0.875,
        "Per-episode rFFT PSD (Hann, mean over feature dims, mean-centered). "
        f"Shaded = ±1 std across episodes. Episode HF PCA uses f ≥ {cutoff:.3f} (= {args.highpass_frac:.0%} Nyquist).",
        fontsize=8.0,
        color=C_MUTED,
    )
    plot_spectrum_panel(axes[0], psd, title="Mean power spectral density", cutoff=cutoff)
    plot_cum_energy_panel(axes[1], psd, title="Cumulative energy vs frequency", cutoff=cutoff)
    stem = out_dir / f"fig_fourier_spectrum_{name}"
    fig.savefig(f"{stem}.png", dpi=args.dpi)
    fig.savefig(f"{stem}.pdf")
    plt.close(fig)

    # Episode log-PSD (full) PCA
    fig3, axes3 = plt.subplots(1, 2, figsize=(12.4, 5.2), dpi=args.dpi)
    fig3.subplots_adjust(left=0.07, right=0.98, top=0.82, bottom=0.12, wspace=0.28)
    fig3.text(
        0.07,
        0.93,
        f"{name} episode log-PSD PCA  ·  full spectrum",
        fontsize=13,
        fontweight="bold",
        color=C_TEXT,
    )
    fig3.text(
        0.07,
        0.875,
        "One point = one episode. Feature = log10(PSD) over all frequencies. "
        "Right: color = high-frequency energy fraction.",
        fontsize=8.0,
        color=C_MUTED,
    )
    plot_episode_pca_panel(
        axes3[0],
        Zs,
        ys,
        title="Episode log-PSD → PCA (by class)",
        var=pca_s.explained_variance_ratio_,
    )
    plot_episode_pca_panel(
        axes3[1],
        Zs,
        ys,
        title="Same embedding · color = HF energy fraction",
        var=pca_s.explained_variance_ratio_,
        color_by=hfs,
        cbar_label=f"energy fraction f≥{cutoff:.3f}",
    )
    stem3 = out_dir / f"fig_fourier_episode_spectral_pca_{name}"
    fig3.savefig(f"{stem3}.png", dpi=args.dpi)
    fig3.savefig(f"{stem3}.pdf")
    plt.close(fig3)

    # Episode high-pass (HF-band log-PSD) PCA
    fig2, axes2 = plt.subplots(1, 2, figsize=(12.4, 5.2), dpi=args.dpi)
    fig2.subplots_adjust(left=0.07, right=0.98, top=0.82, bottom=0.12, wspace=0.28)
    fig2.text(
        0.07,
        0.93,
        f"{name} episode high-pass PCA  ·  log10(PSD) for f≥{cutoff:.3f}",
        fontsize=13,
        fontweight="bold",
        color=C_TEXT,
    )
    fig2.text(
        0.07,
        0.875,
        "One point = one episode. Same PSD pipeline, but only high-frequency bins enter the feature vector.",
        fontsize=8.0,
        color=C_MUTED,
    )
    plot_episode_pca_panel(
        axes2[0],
        Zh,
        yhf,
        title="Episode HF-band log-PSD → PCA",
        var=pca_h.explained_variance_ratio_,
    )
    plot_episode_pca_panel(
        axes2[1],
        Zh,
        yhf,
        title="Same embedding · color = HF energy fraction",
        var=pca_h.explained_variance_ratio_,
        color_by=hfs,
        cbar_label=f"energy fraction f≥{cutoff:.3f}",
    )
    stem2 = out_dir / f"fig_fourier_highpass_pca_{name}"
    fig2.savefig(f"{stem2}.png", dpi=args.dpi)
    fig2.savefig(f"{stem2}.pdf")
    plt.close(fig2)

    return {
        "layer": name,
        "feature": title_feat,
        "cutoff_cycles_per_frame": cutoff,
        "highpass_frac_of_nyquist": args.highpass_frac,
        "hf_energy_fraction_mean": hf_frac,
        "psd_n_episodes": psd["n_episodes"],
        "episode_n": {CLASS_NAMES[c]: int((ys == c).sum()) for c in (0, 1, 2)},
        "episode_spectral_pca_var": [float(x) for x in pca_s.explained_variance_ratio_],
        "episode_spectral_pca_stats": stats_s,
        "episode_highpass_pca_var": [float(x) for x in pca_h.explained_variance_ratio_],
        "episode_highpass_pca_stats": stats_h,
        "figures": {
            "spectrum_png": str(stem.with_suffix(".png")),
            "spectrum_pdf": str(stem.with_suffix(".pdf")),
            "episode_spectral_pca_png": str(stem3.with_suffix(".png")),
            "episode_spectral_pca_pdf": str(stem3.with_suffix(".pdf")),
            "highpass_pca_png": str(stem2.with_suffix(".png")),
            "highpass_pca_pdf": str(stem2.with_suffix(".pdf")),
        },
        "psd_freq": psd["freq"].tolist(),
        "psd_mean": {k: v.tolist() for k, v in psd["mean"].items()},
        "psd_cum": {k: v.tolist() for k, v in psd["cum_energy"].items()},
        "episode_pca_points": {
            "pc1": Zs[:, 0].tolist(),
            "pc2": Zs[:, 1].tolist(),
            "label": ys.astype(int).tolist(),
            "hf_frac": hfs.tolist(),
        },
        "episode_highpass_pca_points": {
            "pc1": Zh[:, 0].tolist(),
            "pc2": Zh[:, 1].tolist(),
            "label": yhf.astype(int).tolist(),
            "hf_frac": hfs.tolist(),
        },
    }


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir or (args.frozen_dir / "fourier_pca")
    out_dir.mkdir(parents=True, exist_ok=True)

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

    trajs2 = build_l2_trajectories(
        visual_features=args.visual_features,
        expert=args.expert,
        rollout=args.l2_rollout,
    )
    trajs3 = build_l3_trajectories(
        expert=args.expert,
        rollout=args.l3_rollout,
        l3_meta=args.l3_meta,
    )

    rep2 = run_layer("L2", r"concat(s, z_VAE, a)", trajs2, args=args, out_dir=out_dir)
    rep3 = run_layer("L3", r"[s_t, a_t, s_{t+5}]", trajs3, args=args, out_dir=out_dir)

    # Overview 2×2: episode log-PSD PCA | episode HF PCA  (L2 / L3)
    cutoff = float(args.highpass_frac) * 0.5
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 10.0), dpi=args.dpi)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.90, bottom=0.07, wspace=0.26, hspace=0.30)
    fig.text(
        0.07,
        0.955,
        "L2-visual / L3  ·  episode Fourier PCA (full spectrum vs high-pass band)",
        fontsize=14,
        fontweight="bold",
        color=C_TEXT,
    )
    fig.text(
        0.07,
        0.915,
        f"One point = one episode. Left: log10(PSD) all freqs. Right: log10(PSD) for f≥{cutoff:.3f} "
        f"({args.highpass_frac:.0%} Nyquist). Same frozen L2/L3 protocols.",
        fontsize=8.2,
        color=C_MUTED,
    )

    for row, (layer, rep) in enumerate((("L2", rep2), ("L3", rep3))):
        pts = rep["episode_pca_points"]
        pth = rep["episode_highpass_pca_points"]
        plot_episode_pca_panel(
            axes[row, 0],
            np.column_stack([pts["pc1"], pts["pc2"]]),
            np.asarray(pts["label"], dtype=np.int8),
            title=f"{layer} episode log-PSD PCA (full)",
            var=np.asarray(rep["episode_spectral_pca_var"]),
        )
        plot_episode_pca_panel(
            axes[row, 1],
            np.column_stack([pth["pc1"], pth["pc2"]]),
            np.asarray(pth["label"], dtype=np.int8),
            title=f"{layer} episode high-pass PCA (f≥{cutoff:.3f})",
            var=np.asarray(rep["episode_highpass_pca_var"]),
        )

    stem_ov = out_dir / "fig_fourier_highpass_overview_L2_L3"
    fig.savefig(f"{stem_ov}.png", dpi=args.dpi)
    fig.savefig(f"{stem_ov}.pdf")
    plt.close(fig)

    drop_keys = (
        "psd_freq",
        "psd_mean",
        "psd_cum",
        "episode_pca_points",
        "episode_highpass_pca_points",
    )
    report = {
        "protocol": {
            "highpass_frac_of_nyquist": args.highpass_frac,
            "cutoff_cycles_per_frame": cutoff,
            "min_frames": args.min_frames,
            "unit": "episode (not frame)",
            "psd": "per-episode rFFT after per-dim mean removal + Hann; mean over dims; interp to common f-grid",
            "episode_spectral": "log10(PSD) full band per episode → joint PCA",
            "episode_highpass": "log10(PSD) restricted to f >= cutoff per episode → joint PCA",
        },
        "L2": {k: v for k, v in rep2.items() if k not in drop_keys},
        "L3": {k: v for k, v in rep3.items() if k not in drop_keys},
        "overview": str(stem_ov.with_suffix(".png")),
        "curves": {
            "L2": {"freq": rep2["psd_freq"], "psd_mean": rep2["psd_mean"], "psd_cum": rep2["psd_cum"]},
            "L3": {"freq": rep3["psd_freq"], "psd_mean": rep3["psd_mean"], "psd_cum": rep3["psd_cum"]},
        },
        "episode_pca": {
            "L2": rep2["episode_pca_points"],
            "L3": rep3["episode_pca_points"],
        },
        "episode_highpass_pca": {
            "L2": rep2["episode_highpass_pca_points"],
            "L3": rep3["episode_highpass_pca_points"],
        },
    }
    (out_dir / "fourier_pca_report.json").write_text(json.dumps(report, indent=2))
    print(f"[done] wrote {out_dir}", flush=True)
    for layer in ("L2", "L3"):
        sep_full = report[layer]["episode_spectral_pca_stats"]["centroid_l2"]["expert_vs_failure"][
            "sep_over_pooled_rms"
        ]
        sep_hf = report[layer]["episode_highpass_pca_stats"]["centroid_l2"]["expert_vs_failure"][
            "sep_over_pooled_rms"
        ]
        print(
            f"  {layer} episode-full E↔F sep={sep_full:.3f}  episode-HF E↔F sep={sep_hf:.3f} "
            f"hf_energy={report[layer]['hf_energy_fraction_mean']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
