#!/usr/bin/env python3
"""Single-panel L2-visual: concat(s, z_VAE, a) → frame-level 1NN to Expert.

Uses ALL frames (no failure head/tail trim). Distance:
  d(x) = min_{e in Expert} ||x - e||   (global 1-NN, no progress align)

Feature per frame t:
  s = observation.state   (z-scored by Expert)
  v = VAE pooled front|wrist  (z-scored by Expert)
  a = action             (z-scored by Expert)
  x = concat(s, v, a)

Frozen approved figure (do not casually restyle):
  results/frozen_l2_visual_l3_transition_20260809/L2_visual_frame_all.png
  Regenerate via: scripts/analysis/render_frozen_l2_visual_l3_transition.py
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

ROOT = Path(__file__).resolve().parents[3]  # archive/l2l3_.../scripts → repo root
DEFAULT_EXPERT = ROOT / "data/dexjoco/dexjoco_lerobot_datasets/water_plant"
DEFAULT_ROLLOUT = ROOT / "data/water_plant_s0_b1_video_cfg_20260808_152243/rollout_raw_200"
DEFAULT_VIS = ROOT / "results/frozen_l2_visual_l3_transition_20260809/L2_visual_features_fulltraj.npz"
DEFAULT_OUT = ROOT / "results/frozen_l2_visual_l3_transition_20260809/L2_visual_frame_all"

C_EXPERT = "#4C78A8"
C_SUCC = "#54A24B"
C_FAIL = "#E45756"
C_MUTED = "#5B6B7A"
C_TEXT = "#1F2A33"
C_GRID = "#E6E9ED"
C_PANEL = "#F7F8FA"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--expert", type=Path, default=DEFAULT_EXPERT)
    p.add_argument("--rollout", type=Path, default=DEFAULT_ROLLOUT)
    p.add_argument("--visual-features", type=Path, default=DEFAULT_VIS)
    p.add_argument("--output-stem", type=Path, default=DEFAULT_OUT)
    p.add_argument("--dpi", type=int, default=300)
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


def self_nn(X: np.ndarray, chunk: int = 2048) -> np.ndarray:
    X = X.astype(np.float64, copy=False)
    n = len(X)
    out = np.empty(n)
    q = np.sum(X * X, 1)
    for i0 in range(0, n, chunk):
        i1 = min(n, i0 + chunk)
        d2 = q[i0:i1, None] + q[None, :] - 2.0 * (X[i0:i1] @ X.T)
        for r, g in enumerate(range(i0, i1)):
            d2[r, g] = np.inf
        out[i0:i1] = np.sqrt(np.maximum(d2.min(1), 0.0))
    return out


def cross_nn(G: np.ndarray, Q: np.ndarray, chunk: int = 2048) -> np.ndarray:
    if len(Q) == 0:
        return np.zeros(0)
    G = G.astype(np.float64, copy=False)
    Q = Q.astype(np.float64, copy=False)
    g2 = np.sum(G * G, 1)
    out = np.empty(len(Q))
    for i0 in range(0, len(Q), chunk):
        i1 = min(len(Q), i0 + chunk)
        q = Q[i0:i1]
        q2 = np.sum(q * q, 1)
        d2 = q2[:, None] + g2[None, :] - 2.0 * (q @ G.T)
        out[i0:i1] = np.sqrt(np.maximum(d2.min(1), 0.0))
    return out


def build_sa_lookup(root: Path) -> dict[tuple[int, int], tuple[np.ndarray, np.ndarray]]:
    """Map (episode, frame_index) → (state, action)."""
    chunk = root / "data" / "chunk-000"
    out: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    for path in sorted(chunk.glob("episode_*.parquet")):
        ep = int(path.stem.split("_")[-1])
        arr = load_episode(path)
        for i, f in enumerate(arr["frame"]):
            out[(ep, int(f))] = (arr["state"][i], arr["action"][i])
    return out


def hist_density(ax: plt.Axes, series: list[tuple[np.ndarray, str, str]], *, note: str) -> None:
    positives = [d[np.isfinite(d) & (d > 0)] for d, _, _ in series if len(d)]
    lo = min(float(np.quantile(d, 0.02)) for d in positives)
    hi = max(float(np.quantile(d, 0.98)) for d in positives)
    bins = np.geomspace(max(lo, 1e-2), max(hi * 1.05, lo * 1.2), 42)
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

    ax.set_xscale("log")
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#C5CCD3")
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(labelsize=8, colors=C_MUTED, length=3, pad=2)
    ax.grid(True, which="major", color=C_GRID, linewidth=0.65, alpha=0.95)
    ax.set_axisbelow(True)
    ax.set_xlabel("1-NN distance to Expert  (log scale)", fontsize=9, color=C_MUTED, labelpad=3)
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


def main() -> None:
    args = parse_args()
    pack = np.load(args.visual_features, allow_pickle=True)
    src = np.asarray(pack["source"]).astype(str)
    feat_v = np.asarray(pack["feat"], dtype=np.float32)
    episode = np.asarray(pack["episode"], dtype=np.int64)
    frame = np.asarray(pack["frame"], dtype=np.int64)
    success = np.asarray(pack["success"], dtype=bool)

    print("[L2] building expert (s,a) lookup…", flush=True)
    sa_expert = build_sa_lookup(args.expert)
    print("[L2] building rollout (s,a) lookup…", flush=True)
    sa_roll = build_sa_lookup(args.rollout)

    s_list, a_list, v_list, src_list, succ_list = [], [], [], [], []
    n_miss = 0
    for i in range(len(src)):
        ep = int(episode[i])
        fr = int(frame[i])
        key = (ep, fr)
        sa = sa_expert.get(key) if src[i] == "expert" else sa_roll.get(key)
        if sa is None:
            n_miss += 1
            continue
        s, a = sa
        s_list.append(s)
        a_list.append(a)
        v_list.append(feat_v[i])
        src_list.append(src[i])
        succ_list.append(bool(success[i]))

    S = np.asarray(s_list, dtype=np.float32)
    A = np.asarray(a_list, dtype=np.float32)
    V = np.asarray(v_list, dtype=np.float32)
    src_a = np.asarray(src_list)
    succ_a = np.asarray(succ_list, dtype=bool)
    print(f"[align] kept {len(S)} / {len(src)}  (dropped {n_miss})", flush=True)

    m_e = src_a == "expert"
    m_s = (src_a == "rollout") & succ_a
    m_f = (src_a == "rollout") & (~succ_a)

    mu_s, sd_s = fit_norm(S[m_e])
    mu_v, sd_v = fit_norm(V[m_e])
    mu_a, sd_a = fit_norm(A[m_e])
    Sn = apply_norm(S, mu_s, sd_s)
    Vn = apply_norm(V, mu_v, sd_v)
    An = apply_norm(A, mu_a, sd_a)
    X = np.concatenate([Sn, Vn, An], axis=1)  # s, v, a

    Ge = X[m_e]
    Qs = X[m_s]
    Qf = X[m_f]
    print(f"[nn] expert={len(Ge)} succ={len(Qs)} fail={len(Qf)} dim={X.shape[1]}", flush=True)

    d_self = self_nn(Ge)
    d_succ = cross_nn(Ge, Qs)
    d_fail = cross_nn(Ge, Qf)
    r_s = float(np.median(d_succ) / max(np.median(d_self), 1e-12))
    r_f = float(np.median(d_fail) / max(np.median(d_self), 1e-12))

    outcomes = {
        int(o["episode_index"]): o
        for o in load_jsonl(args.rollout / "meta" / "episode_outcomes.jsonl")
    }
    n_e = len({int(e["episode_index"]) for e in load_jsonl(args.expert / "meta" / "episodes.jsonl")})
    n_s = sum(1 for o in outcomes.values() if o["success"])
    n_f = sum(1 for o in outcomes.values() if not o["success"])

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
    fig, ax = plt.subplots(figsize=(8.6, 5.4), dpi=args.dpi)
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

    hist_density(
        ax,
        [
            (d_self, C_EXPERT, f"Expert self  ({n_e} eps)"),
            (d_succ, C_SUCC, rf"$R_{{\mathrm{{succ}}}}$  ({n_s} eps)"),
            (d_fail, C_FAIL, rf"$R_{{\mathrm{{fail}}}}$  ({n_f} eps)"),
        ],
        note=f"median / Expert:  succ {r_s:.2f}×   fail {r_f:.2f}×",
    )
    ax.set_title("All frames (no trim)", fontsize=10.5, fontweight="semibold", color=C_TEXT, loc="left", pad=8)

    fig.text(
        0.10,
        0.035,
        "Feature: state ⊕ S0 VAE pooled front|wrist ⊕ action (each z-scored on Expert, then concat).  "
        "Expert↔Expert = leave-one-point self-1NN; rollouts = 1NN into Expert gallery.  "
        "Aligned to visual frame grid; all frames, no trim.",
        fontsize=6.5,
        color=C_MUTED,
    )

    stem = args.output_stem
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{stem}.png", dpi=args.dpi)
    fig.savefig(f"{stem}.pdf")
    plt.close(fig)

    meta = {
        "feature": "concat(s, z_VAE, a)",
        "protocol": "frame-level all frames; global 1NN to Expert; no trim",
        "n_frames": {
            "expert": int(m_e.sum()),
            "success": int(m_s.sum()),
            "failure": int(m_f.sum()),
            "dropped": n_miss,
        },
        "dim": {"s": int(S.shape[1]), "visual": int(V.shape[1]), "a": int(A.shape[1]), "concat": int(X.shape[1])},
        "median_over_expert": {"success": r_s, "failure": r_f},
        "figure": f"{stem}.png",
    }
    Path(f"{stem}_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    np.savez_compressed(f"{stem}_distances.npz", d_self=d_self, d_succ=d_succ, d_fail=d_fail)
    print(json.dumps(meta, indent=2))
    print(f"[done] {stem}.png", flush=True)


if __name__ == "__main__":
    main()
