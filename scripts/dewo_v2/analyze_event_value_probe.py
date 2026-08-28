#!/usr/bin/env python3
"""Cluster / probe MLP: can replan-level V features identify recoverability events?

Two label sources:
  1) benti 4x50: many replans, proxy labels (fail vs succ; stall vs not).
  2) pair dataset dump (optional): true t* on 38 failure prefixes vs negatives.

  python scripts/dewo_v2/analyze_event_value_probe.py \\
    --benti-root evaluate_results/dexjoco/fold_glasses_dewo_v9_step_005000_benti_cfg1_4x50_20260827_162324 \\
    --pair-index data/fold_glasses_dewo_v9_pair_full_lerobot/pair_index.json \\
    --out-dir evaluate_results/dexjoco/fold_glasses_v9_base_cfg_eval_step_005000_oracle_once_20260827_200336/event_probe
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class ReplanRow:
    episode_id: str
    success: bool
    step: int
    replan_k: int
    values: np.ndarray
    rels: np.ndarray
    label_event: int | None = None
    pair_id: str | None = None


def _load_benti_rows(out_root: Path, replan: int = 24) -> list[ReplanRow]:
    rows: list[ReplanRow] = []
    for path in sorted(out_root.glob("run*/shard_*/**/*_actions.npz")):
        z = np.load(path, allow_pickle=True)
        if "cfg_values" not in z.files:
            continue
        values = np.asarray(z["cfg_values"], dtype=np.float64).reshape(-1)
        rels = (
            np.asarray(z["cfg_value_rels"], dtype=np.float64).reshape(-1)
            if "cfg_value_rels" in z.files
            else np.full_like(values, np.nan)
        )
        if values.size > 1:
            for i in range(1, values.size):
                if not np.isfinite(rels[i]):
                    prev = float(values[i - 1])
                    cur = float(values[i])
                    rels[i] = (cur - prev) / max(abs(prev), 1e-6)
        if "policy_query_steps" in z.files:
            steps = np.asarray(z["policy_query_steps"], dtype=np.int32).reshape(-1)
        else:
            steps = np.arange(values.size, dtype=np.int32) * replan
        n = min(values.size, steps.size, rels.size)
        ok = "success" in path.name
        ep = path.parent.name
        for k in range(n):
            rows.append(
                ReplanRow(
                    episode_id=f"{path.parent.parent.parent.name}/{ep}",
                    success=ok,
                    step=int(steps[k]),
                    replan_k=k,
                    values=values[: k + 1],
                    rels=rels[: k + 1],
                )
            )
    return rows


def _feat_window(values: np.ndarray, rels: np.ndarray, k: int, step: int, ep_len: int) -> np.ndarray:
    """Feature vector at replan index k (inclusive history length 4)."""
    hist_v = []
    hist_r = []
    for lag in range(4):
        idx = k - lag
        if idx >= 0:
            hist_v.append(float(values[idx]))
            hist_r.append(float(rels[idx]) if np.isfinite(rels[idx]) else 0.0)
        else:
            hist_v.append(0.0)
            hist_r.append(0.0)
    v_t = hist_v[0]
    v_prev = hist_v[1] if hist_v[1] != 0.0 or k >= 1 else v_t
    delta = v_t - v_prev
    rel_t = hist_r[0]
    vmax = float(np.nanmax(values[: k + 1])) if k >= 0 else v_t
    return np.array(
        [
            v_t,
            v_prev,
            delta,
            rel_t,
            hist_r[1],
            float(step),
            float(step / max(ep_len, 1)),
            vmax,
            v_t - vmax,
            float(np.nanstd(values[: k + 1])) if k >= 0 else 0.0,
            float(np.nanmean(values[: k + 1])) if k >= 0 else v_t,
            float(k),
        ],
        dtype=np.float64,
    )


def _rows_to_xy(rows: list[ReplanRow]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    xs, ys, tags = [], [], []
    for row in rows:
        ep_len = int(max(row.step + 24, row.step + 1))
        xs.append(_feat_window(row.values, row.rels, row.replan_k, row.step, ep_len))
        if row.label_event is not None:
            ys.append(int(row.label_event))
            tags.append("pair")
        else:
            ys.append(1 if row.success else 0)
            tags.append("benti_fail_succ")
    return np.stack(xs), np.asarray(ys), tags


def _standardize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return (X - mean) / std, mean, std


def _roc_auc(y: np.ndarray, prob: np.ndarray) -> float:
    order = np.argsort(prob)
    y_sorted = y[order]
    n_pos = float((y == 1).sum())
    n_neg = float((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = np.arange(1, y.size + 1, dtype=np.float64)
    sum_ranks_pos = float(ranks[y_sorted == 1].sum())
    return (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _average_precision(y: np.ndarray, prob: np.ndarray) -> float:
    order = np.argsort(-prob)
    y_sorted = y[order]
    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)
    prec = tp / np.maximum(tp + fp, 1)
    pos = int((y == 1).sum())
    if pos == 0:
        return float("nan")
    return float(prec[y_sorted == 1].sum() / pos)


class _ProbeMLP(nn.Module):
    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _mlp_cv(X: np.ndarray, y: np.ndarray, *, n_splits: int = 5, epochs: int = 30) -> dict:
    torch.set_num_threads(4)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    n_splits = min(n_splits, n_pos, n_neg)
    if n_splits < 2:
        return {"n": int(y.size), "n_pos": n_pos, "n_neg": n_neg, "error": "too_few_samples"}

    idx_pos = np.where(y == 1)[0]
    idx_neg = np.where(y == 0)[0]
    rng = np.random.default_rng(0)
    folds: list[np.ndarray] = []
    for s in range(n_splits):
        hold_pos = np.array_split(rng.permutation(idx_pos), n_splits)[s]
        hold_neg = np.array_split(rng.permutation(idx_neg), n_splits)[s]
        folds.append(np.concatenate([hold_pos, hold_neg]))

    probs = np.zeros(y.size, dtype=np.float64)
    for hold in folds:
        train = np.setdiff1d(np.arange(y.size), hold)
        X_tr, mean, std = _standardize(X[train])
        X_te = (X[hold] - mean) / std
        model = _ProbeMLP(X.shape[1])
        opt = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-4)
        loss_fn = nn.BCEWithLogitsLoss()
        x_tr = torch.from_numpy(X_tr.astype(np.float32))
        y_tr = torch.from_numpy(y[train].astype(np.float32))
        model.train()
        for _ in range(epochs):
            opt.zero_grad()
            loss = loss_fn(model(x_tr), y_tr)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            logits = model(torch.from_numpy(X_te.astype(np.float32)))
            probs[hold] = torch.sigmoid(logits).cpu().numpy()

    pred = (probs >= 0.5).astype(np.int32)
    return {
        "n": int(y.size),
        "n_pos": n_pos,
        "n_neg": n_neg,
        "accuracy": float((pred == y).mean()),
        "roc_auc": float(_roc_auc(y, probs)),
        "ap": float(_average_precision(y, probs)),
    }


def _kmeans(X: np.ndarray, k: int, *, iters: int = 40) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    n = X.shape[0]
    centers = X[rng.choice(n, size=k, replace=False)].copy()
    labels = np.zeros(n, dtype=np.int32)
    for _ in range(iters):
        dist = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels = dist.argmin(axis=1).astype(np.int32)
        for j in range(k):
            m = labels == j
            if m.any():
                centers[j] = X[m].mean(axis=0)
    return labels, centers


def _silhouette(X: np.ndarray, labels: np.ndarray, *, max_n: int = 400) -> float:
    n = X.shape[0]
    if n > max_n:
        rng = np.random.default_rng(0)
        idx = rng.choice(n, size=max_n, replace=False)
        X = X[idx]
        labels = labels[idx]
        n = max_n
    if len(set(labels.tolist())) < 2:
        return -1.0
    dist = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2))
    scores = []
    for i in range(n):
        same = labels == labels[i]
        other = labels != labels[i]
        if same.sum() <= 1 or not other.any():
            continue
        a = dist[i, same].sum() / (same.sum() - 1)
        b = min(dist[i, labels[other] == lab].mean() for lab in set(labels[other].tolist()))
        scores.append((b - a) / max(a, b, 1e-8))
    return float(np.mean(scores)) if scores else -1.0


def _cluster_scan(X: np.ndarray, k_max: int = 6) -> dict:
    Xs, _, _ = _standardize(X)
    best = {"k": 2, "silhouette": -1.0}
    rows = []
    for k in range(2, k_max + 1):
        labels, _ = _kmeans(Xs, k)
        sil = _silhouette(Xs, labels)
        rows.append({"k": k, "silhouette": sil})
        if sil > best["silhouette"]:
            best = {"k": k, "silhouette": sil, "labels": labels}
    return {"scan": rows, "best": best}


def _load_pair_dump(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _build_pair_labeled_rows(
    dump_rows: list[dict],
    replan: int,
) -> list[ReplanRow]:
    """Build ReplanRow list from pair value dump jsonl."""
    by_ep: dict[tuple[str, int], list[dict]] = {}
    for row in dump_rows:
        key = (str(row["dataset_root"]), int(row["episode_index"]))
        by_ep.setdefault(key, []).append(row)
    out: list[ReplanRow] = []
    for (_root, _ep), group in by_ep.items():
        group = sorted(group, key=lambda r: int(r["frame_index"]))
        values = np.array([float(r["cfg_value"]) for r in group], dtype=np.float64)
        steps = np.array([int(r["frame_index"]) for r in group], dtype=np.int32)
        rels = np.zeros_like(values)
        for i in range(1, values.size):
            prev = values[i - 1]
            rels[i] = (values[i] - prev) / max(abs(prev), 1e-6)
        for i, row in enumerate(group):
            out.append(
                ReplanRow(
                    episode_id=f"pair_ep{row['episode_index']}",
                    success=False,
                    step=int(steps[i]),
                    replan_k=i,
                    values=values[: i + 1],
                    rels=rels[: i + 1],
                    label_event=1 if bool(row.get("is_event")) else 0,
                    pair_id=str(row.get("pair_id", "")),
                )
            )
    return out


def _plot_cluster(out_dir: Path, X: np.ndarray, labels: np.ndarray, title: str) -> None:
    Xs, _, _ = _standardize(X)
    # 2D projection via SVD
    u, s, _ = np.linalg.svd(Xs - Xs.mean(axis=0), full_matrices=False)
    xy = u[:, :2] * s[:2]
    fig, ax = plt.subplots(figsize=(7, 5))
    for lab in sorted(set(labels.tolist())):
        m = labels == lab
        ax.scatter(xy[m, 0], xy[m, 1], s=8, alpha=0.5, label=f"c{lab}")
    ax.legend(fontsize=8, markerscale=2)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_dir / "cluster_pca.png", dpi=130)
    plt.close(fig)


def _plot_pair_separation(out_dir: Path, dump_rows: list[dict]) -> None:
    ev = [float(r["cfg_value"]) for r in dump_rows if r.get("is_event")]
    neg = [float(r["cfg_value"]) for r in dump_rows if not r.get("is_event")]
    fig, ax = plt.subplots(figsize=(6, 4))
    if ev:
        ax.hist(ev, bins=20, alpha=0.65, label=f"event t* n={len(ev)}", color="#2ca02c")
    if neg:
        ax.hist(neg, bins=30, alpha=0.55, label=f"non-event n={len(neg)}", color="#7f7f7f")
    ax.set_xlabel("V")
    ax.set_ylabel("count")
    ax.set_title("Pair dump: V at t* vs negatives")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "pair_v_hist.png", dpi=130)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benti-root", type=Path, required=True)
    parser.add_argument(
        "--pair-index",
        type=Path,
        default=Path("data/fold_glasses_dewo_v9_pair_full_lerobot/pair_index.json"),
    )
    parser.add_argument(
        "--pair-dump",
        type=Path,
        default=None,
        help="jsonl from dump_pair_event_values.py (optional)",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--replan-steps", type=int, default=24)
    args = parser.parse_args()

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    replan = int(args.replan_steps)

    benti_rows = _load_benti_rows(args.benti_root, replan)
    X_all, y_fs, _ = _rows_to_xy(benti_rows)
    summary: dict = {"benti_n_replans": len(benti_rows), "feature_dim": int(X_all.shape[1])}

    # --- fail vs succ ---
    summary["mlp_fail_vs_succ"] = _mlp_cv(X_all, y_fs, n_splits=5)

    # --- stall / drop on fail episodes only (proxy for gate) ---
    fail_rows = [r for r in benti_rows if not r.success and r.replan_k >= 1]
    stall_y = []
    stall_x = []
    for row in fail_rows:
        rel = float(row.rels[row.replan_k])
        stall_y.append(1 if (np.isfinite(rel) and rel < 0.05) else 0)
        ep_len = int(max(row.step + 24, 1))
        stall_x.append(_feat_window(row.values, row.rels, row.replan_k, row.step, ep_len))
    if stall_x:
        summary["mlp_fail_stall_rel_lt_0.05"] = _mlp_cv(np.stack(stall_x), np.asarray(stall_y))

    # --- calendar t* match on benti (weak): step near any oracle t* ---
    pairs = json.loads(args.pair_index.read_text(encoding="utf-8"))
    pair_list = pairs["pairs"] if isinstance(pairs, dict) else pairs
    t_stars = sorted({int(p["t_star_last_recoverable"]) for p in pair_list})
    tol = replan // 2
    cal_y = []
    cal_x = []
    for row in benti_rows:
        if row.replan_k < 1:
            continue
        near = any(abs(row.step - t) <= tol for t in t_stars)
        cal_y.append(1 if near else 0)
        ep_len = int(max(row.step + 24, 1))
        cal_x.append(_feat_window(row.values, row.rels, row.replan_k, row.step, ep_len))
    summary["oracle_tstar_calendar"] = {
        "n_tstar": len(t_stars),
        "t_star_median": float(np.median(t_stars)),
        "n_pos": int(sum(cal_y)),
        "n_neg": int(len(cal_y) - sum(cal_y)),
    }
    summary["mlp_calendar_tstar_vs_rest"] = _mlp_cv(np.stack(cal_x), np.asarray(cal_y))

    # --- clustering on fail replans only ---
    fail_only = [r for r in benti_rows if not r.success and r.replan_k >= 2]
    Xf = np.stack(
        [
            _feat_window(r.values, r.rels, r.replan_k, r.step, int(max(r.step + 24, 1)))
            for r in fail_only
        ]
    )
    cl = _cluster_scan(Xf, k_max=6)
    summary["cluster_fail_replans"] = {"scan": cl["scan"], "best_k": cl["best"]["k"], "best_sil": cl["best"]["silhouette"]}
    _plot_cluster(out_dir, Xf, cl["best"]["labels"], f"KMeans k={cl['best']['k']} on fail replans")

    # --- pair dump: true t* labels ---
    pair_dump_path = args.pair_dump or (out_dir / "pair_event_values.jsonl")
    dump_rows = _load_pair_dump(pair_dump_path)
    if dump_rows:
        pair_rows = _build_pair_labeled_rows(dump_rows, replan)
        Xp, yp, _ = _rows_to_xy(pair_rows)
        summary["pair_dump_n"] = len(dump_rows)
        summary["mlp_pair_event_tstar"] = _mlp_cv(Xp, yp, n_splits=5)
        # LOPO by pair_id
        pair_ids = sorted({str(r["pair_id"]) for r in dump_rows if r.get("pair_id")})
        if len(pair_ids) >= 4:
            lopo_correct = 0
            lopo_total = 0
            for hold in pair_ids:
                train = [r for r in dump_rows if str(r.get("pair_id")) != hold]
                test = [r for r in dump_rows if str(r.get("pair_id")) == hold]
                if not test:
                    continue
                tr_rows = _build_pair_labeled_rows(train, replan)
                te_rows = _build_pair_labeled_rows(test, replan)
                X_tr, y_tr, _ = _rows_to_xy(tr_rows)
                X_te, y_te, _ = _rows_to_xy(te_rows)
                X_tr, mean, std = _standardize(X_tr)
                X_te = (X_te - mean) / std
                model = _ProbeMLP(X_tr.shape[1])
                opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
                loss_fn = nn.BCEWithLogitsLoss()
                ds = TensorDataset(
                    torch.from_numpy(X_tr.astype(np.float32)),
                    torch.from_numpy(y_tr.astype(np.float32)),
                )
                model.train()
                for _ in range(80):
                    for xb, yb in DataLoader(ds, batch_size=64, shuffle=True):
                        opt.zero_grad()
                        loss_fn(model(xb), yb).backward()
                        opt.step()
                model.eval()
                with torch.no_grad():
                    pred = (
                        torch.sigmoid(model(torch.from_numpy(X_te.astype(np.float32)))) >= 0.5
                    ).cpu().numpy().astype(np.int32)
                lopo_correct += int((pred == y_te).sum())
                lopo_total += int(y_te.size)
            summary["mlp_pair_lopo_acc"] = float(lopo_correct / max(lopo_total, 1))
        _plot_pair_separation(out_dir, dump_rows)
    else:
        summary["pair_dump"] = "missing — run dump_pair_event_values.py first for true t* labels"

    (out_dir / "probe_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
