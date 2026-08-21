#!/usr/bin/env python3
"""Why video CFG, not ReCap-style advantage-conditioned action CFG.

S0 Pass@20 on failed fold-glasses rollouts. No new training.

a  Episode Monte Carlo return is 0 on every prefix, but Pass@20 stays
   high until t*. A value fit to that return labels recoverable states
   as hopeless.
b  Inventory: the same prefixes are almost all still salvageable.
c  Even with correct pair labels at t*, actions do not rank outcomes
   within a prefix or across episodes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analysis._plot_style import (
    EVENT,
    INK,
    MUTED,
    PASS,
    WINDOW,
    apply_style,
    panel_label,
    polish,
    save_figure,
    stat_note,
)
from scripts.analysis.pass20_scan_data import (
    DEFAULT_STATS,
    first_pass_zero,
    last_recoverable_before,
    load_chunks,
    node_metrics,
    sorted_nodes,
)
from scripts.analysis.plot_pass20_cfg_event_justification import (
    EVENT_NUM_FRAMES,
    HORIZON,
    STRIDE,
    _align,
    collect_episodes,
)
from scripts.fold_glasses.discover_seedpair_branch_events import (
    load_global_zscore,
    normalize_actions,
)

DEFAULT_SCAN = (
    ROOT
    / "data"
    / "fold_glasses_opensource_s0_collect_4x50_20260812_112113"
    / "pass20_action_chunk_analysis_20260819"
)
DEFAULT_OUTPUT = DEFAULT_SCAN / "figures" / "fig_pass20_why_not_recap.png"


def auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    pos = scores[labels]
    neg = scores[~labels]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    greater = np.mean(pos[:, None] > neg[None, :])
    equal = np.mean(pos[:, None] == neg[None, :])
    return float(greater + 0.5 * equal)


def _rms(left: np.ndarray, right: np.ndarray) -> float:
    delta = left.reshape(-1).astype(np.float64) - right.reshape(-1).astype(np.float64)
    return float(np.sqrt(np.mean(delta * delta)))


def within_prefix_action_auroc(
    *,
    data: dict[str, np.ndarray],
    z: np.ndarray,
    rows: list[dict[str, Any]],
) -> np.ndarray:
    values: list[float] = []
    for row in rows:
        idx = np.where(
            (data["episode_index"] == row["episode"])
            & (data["prefix_frame"] == row["t_star"])
        )[0]
        chunks = z[idx][:, :HORIZON]
        success = data["success"][idx]
        if int(success.sum()) < 2 or int((~success).sum()) < 2:
            continue
        scores = np.empty(len(idx), dtype=np.float64)
        for i in range(len(idx)):
            s_mask = success.copy()
            f_mask = ~success
            if success[i]:
                s_mask[i] = False
            else:
                f_mask[i] = False
            if int(s_mask.sum()) < 1 or int(f_mask.sum()) < 1:
                scores[i] = np.nan
                continue
            c_s = chunks[s_mask].mean(axis=0)
            c_f = chunks[f_mask].mean(axis=0)
            scores[i] = _rms(chunks[i], c_f) - _rms(chunks[i], c_s)
        ok = np.isfinite(scores)
        if ok.sum() < 6 or success[ok].sum() < 2 or (~success[ok]).sum() < 2:
            continue
        values.append(auroc(success[ok], scores[ok]))
    return np.asarray(values, dtype=np.float64)


def cross_episode_action_auroc(
    *,
    data: dict[str, np.ndarray],
    z: np.ndarray,
    rows: list[dict[str, Any]],
) -> dict[str, float]:
    succ_means: list[np.ndarray] = []
    fail_means: list[np.ndarray] = []
    for row in rows:
        idx = np.where(
            (data["episode_index"] == row["episode"])
            & (data["prefix_frame"] == row["t_star"])
        )[0]
        chunks = z[idx][:, :HORIZON]
        success = data["success"][idx]
        if int(success.sum()) < 1 or int((~success).sum()) < 1:
            continue
        succ_means.append(chunks[success].mean(axis=0).reshape(-1))
        fail_means.append(chunks[~success].mean(axis=0).reshape(-1))
    if len(succ_means) < 4:
        return {"n": 0, "auroc": float("nan"), "pair_ranking": float("nan")}
    succ = np.stack(succ_means)
    fail = np.stack(fail_means)
    n = len(succ)
    s_scores = np.empty(n, dtype=np.float64)
    f_scores = np.empty(n, dtype=np.float64)
    for i in range(n):
        keep = np.ones(n, dtype=bool)
        keep[i] = False
        c_s = succ[keep].mean(axis=0)
        c_f = fail[keep].mean(axis=0)
        s_scores[i] = _rms(succ[i], c_f) - _rms(succ[i], c_s)
        f_scores[i] = _rms(fail[i], c_f) - _rms(fail[i], c_s)
    labels = np.concatenate([np.ones(n, dtype=bool), np.zeros(n, dtype=bool)])
    scores = np.concatenate([s_scores, f_scores])
    return {
        "n": int(n),
        "auroc": auroc(labels, scores),
        "pair_ranking": float(np.mean(s_scores > f_scores)),
    }


def prefix_inventory(
    *,
    metrics: dict[str, np.ndarray],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    n = int(len(metrics["pass_rate"]))
    n_rec = int(np.sum(metrics["pass_rate"] > 0))
    before_pass: list[float] = []
    star_pass: list[float] = []
    for row in rows:
        sel = metrics["episode"] == row["episode"]
        frames, rates, _ = sorted_nodes(
            metrics["frame"][sel], metrics["pass_rate"][sel], metrics["spread"][sel]
        )
        t_star = int(row["t_star"])
        for frame, rate in zip(frames.tolist(), rates.tolist()):
            if int(frame) < t_star:
                before_pass.append(float(rate))
            elif int(frame) == t_star:
                star_pass.append(float(rate))
    before = np.asarray(before_pass, dtype=np.float64)
    star = np.asarray(star_pass, dtype=np.float64)
    return {
        "n_prefixes": n,
        "n_recoverable": n_rec,
        "frac_recoverable": float(n_rec / n) if n else float("nan"),
        "n_before": int(before.size),
        "n_before_recoverable": int(np.sum(before > 0)) if before.size else 0,
        "mean_pass_before": float(np.mean(before)) if before.size else float("nan"),
        "mean_pass_star": float(np.mean(star)) if star.size else float("nan"),
        "median_pass_star": float(np.median(star)) if star.size else float("nan"),
        "n_star": int(star.size),
    }


def plot_why_not_recap(
    *,
    rows: list[dict[str, Any]],
    inventory: dict[str, Any],
    within: np.ndarray,
    cross: dict[str, float],
    output: Path,
) -> None:
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.25))
    ax_v, ax_inv, ax_auc = axes
    taus = np.arange(-8.0, 4.0, 1.0)
    pass_mean, pass_sem = _align(rows, "pass_rate", taus)
    lo = -EVENT_NUM_FRAMES / STRIDE + 1.0

    ok = np.isfinite(pass_mean)
    ax_v.axvspan(-8.4, lo, color="#f5f5f4", lw=0, zorder=0)
    ax_v.axvspan(lo, 1.0, color=WINDOW, lw=0, zorder=0)
    ax_v.axvline(0.0, color="#c4b8b0", ls="--", lw=0.9, zorder=1)
    ax_v.axvline(1.0, color="#c4b8b0", ls=":", lw=0.9, zorder=1)
    ax_v.fill_between(
        taus[ok],
        np.zeros_like(pass_mean[ok]),
        pass_mean[ok],
        color=EVENT,
        alpha=0.10,
        linewidth=0,
        zorder=2,
    )
    ax_v.fill_between(
        taus[ok],
        pass_mean[ok] - pass_sem[ok],
        pass_mean[ok] + pass_sem[ok],
        color=PASS,
        alpha=0.16,
        linewidth=0,
        zorder=3,
    )
    ax_v.plot(
        taus[ok],
        pass_mean[ok],
        color=PASS,
        lw=2.15,
        zorder=4,
        solid_capstyle="round",
        label=r"Pass@20  (oracle $V^\pi$)",
    )
    ax_v.plot(
        taus[ok],
        np.zeros_like(pass_mean[ok]),
        color=EVENT,
        lw=2.0,
        ls=(0, (3.2, 1.8)),
        zorder=4,
        label="episode return (MC)",
    )
    ax_v.set_xlabel(r"replan steps from $t^*$")
    ax_v.set_ylabel("value / success rate")
    ax_v.set_title("Return labels recoverable states as fail", loc="left", pad=8)
    ax_v.set_ylim(-0.04, 1.05)
    ax_v.set_yticks([0.0, 0.5, 1.0])
    ax_v.legend(loc="center right", fontsize=8)
    polish(ax_v)
    panel_label(ax_v, "a")
    ax_v.text(
        (-8.0 + lo) / 2.0,
        1.0,
        "non-critical",
        ha="center",
        va="top",
        fontsize=8.2,
        color=MUTED,
    )
    ax_v.text(
        (lo + 1.0) / 2.0,
        1.0,
        "critical",
        ha="center",
        va="top",
        fontsize=8.2,
        color=MUTED,
    )

    rec = inventory["n_recoverable"]
    groups = ["all prefixes", r"before $t^*$"]
    rec_vals = [rec, inventory["n_before_recoverable"]]
    tot_vals = [inventory["n_prefixes"], inventory["n_before"]]
    dead_vals = [tot - r for tot, r in zip(tot_vals, rec_vals)]
    ax_inv.bar(
        groups,
        rec_vals,
        color="#d7e4ec",
        edgecolor=PASS,
        linewidth=1.1,
        width=0.55,
        zorder=3,
        label="still recoverable",
    )
    ax_inv.bar(
        groups,
        dead_vals,
        bottom=rec_vals,
        color="#f4d6ce",
        edgecolor=EVENT,
        linewidth=1.1,
        width=0.55,
        zorder=3,
        label="Pass@20 = 0",
    )
    ax_inv.set_ylabel("prefixes (MC return = fail)")
    ax_inv.set_title("The state was not hopeless", loc="left", pad=8)
    ax_inv.set_ylim(0, max(tot_vals) * 1.18)
    ax_inv.legend(loc="upper right", fontsize=8)
    polish(ax_inv)
    panel_label(ax_inv, "b")
    stat_note(
        ax_inv,
        f"{inventory['frac_recoverable']:.0%} prefixes still salvageable\n"
        rf"before $t^*$: {inventory['mean_pass_before']:.2f} Pass@20",
        loc="upper left",
    )

    ax_auc.axhline(0.5, color="#d6d3d1", ls="--", lw=1.0, zorder=1)
    names = ["within prefix", "across episodes"]
    vals = [
        float(np.mean(within)) if within.size else float("nan"),
        float(cross.get("auroc", np.nan)),
    ]
    colors = [PASS, EVENT]
    fills = ["#d7e4ec", "#f4d6ce"]
    ax_auc.bar(
        [0, 1],
        vals,
        color=fills,
        edgecolor=colors,
        linewidth=1.15,
        width=0.58,
        zorder=2,
    )
    rng = np.random.default_rng(2)
    if within.size:
        ax_auc.scatter(
            rng.uniform(-0.12, 0.12, within.size),
            within,
            s=22,
            c=PASS,
            zorder=3,
            edgecolors="white",
            linewidths=0.4,
        )
    ax_auc.set_xticks([0, 1], names)
    ax_auc.set_ylim(0.35, 1.02)
    ax_auc.set_ylabel("AUROC of action score")
    ax_auc.set_title("Advantage cannot rank the two actions", loc="left", pad=8)
    polish(ax_auc)
    panel_label(ax_auc, "c")
    pair = cross.get("pair_ranking", float("nan"))
    stat_note(
        ax_auc,
        f"within $n$={within.size}  mean {vals[0]:.2f}\n"
        f"cross $n$={cross.get('n', 0)}  {vals[1]:.2f}"
        + (f"\npair rank {pair:.0%}" if np.isfinite(pair) else ""),
    )

    fig.suptitle(
        "ReCap labels the state and the action; the outcome lives in the future world",
        fontsize=13.2,
        color=INK,
        y=1.03,
    )
    fig.tight_layout(w_pad=2.2)
    save_figure(fig, output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan-root", type=Path, default=DEFAULT_SCAN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dataset-stats", type=Path, default=DEFAULT_STATS)
    parser.add_argument("--min-replicates", type=int, default=20)
    parser.add_argument("--min-other", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scan = args.scan_root.expanduser().resolve()
    data = load_chunks(scan)
    mean, std = load_global_zscore(args.dataset_stats.expanduser().resolve())
    z = normalize_actions(data["chunk"], mean, std)
    metrics = node_metrics(
        chunks=data["chunk"],
        episode_index=data["episode_index"],
        prefix_frame=data["prefix_frame"],
        success=data["success"],
        mean=mean,
        std=std,
        min_replicates=int(args.min_replicates),
    )
    rows = collect_episodes(
        data=data, z=z, metrics=metrics, min_other=int(args.min_other)
    )
    if not rows:
        raise SystemExit("No recoverability events")
    inventory = prefix_inventory(metrics=metrics, rows=rows)
    within = within_prefix_action_auroc(data=data, z=z, rows=rows)
    cross = cross_episode_action_auroc(data=data, z=z, rows=rows)
    output = args.output.expanduser().resolve()
    plot_why_not_recap(
        rows=rows,
        inventory=inventory,
        within=within,
        cross=cross,
        output=output,
    )
    meta = {
        "png": str(output),
        "n_episodes_with_tstar": len(rows),
        "inventory": inventory,
        "within_prefix_action_auroc": {
            "n": int(within.size),
            "mean": float(np.mean(within)) if within.size else None,
            "median": float(np.median(within)) if within.size else None,
            "values": [float(x) for x in within],
        },
        "cross_episode_action_auroc": cross,
        "claim": (
            "Failed-episode Monte Carlo return is 0 on prefixes that remain "
            "recoverable. Action scores do not separate success/fail takeovers."
        ),
    }
    output.with_suffix(".json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(meta, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
