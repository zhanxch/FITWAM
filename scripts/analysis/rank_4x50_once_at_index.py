#!/usr/bin/env python3
"""Rank once-CFG policies from a 4×50 early-index sweep.

Each ``sweep_root/at{i}`` is a 4×50 with CFG once at replan i, then 本体
(max_steps=500 search). 本体 4×50 is the no-CFG baseline.

Two-stage ranking. Optimal actions are a *set*, not a unique pick:

1. Per episode, every action that attains the best available outcome is
   optimal:
   - 本体 succeeds and CFG-at-k also succeeds → {none, k, …} are all optimal.
   - 本体 fails and CFG-at-k and CFG-at-n both succeed → {k, n} are optimal
     (none is not).
   - Every action fails → all actions are equally optimal (nothing to save).
   Intervening at a node that *fails* while 本体 succeeds is not optimal.

2. Approximate those sets with a deployable rule: drop the first N and last
   L early nodes, then fire once at the first remaining node whose 本体-prefix
   energy E lies in a contiguous interval [lo, hi] (optional: ``always`` =
   any finite E). Score by causal 4×50 lookup. A rule matches an episode if
   its action is in that episode's optimal set. If no single interval covers
   every episode (small-E and large-E both want CFG, middle must not), keep
   the feasible rule with the highest success count; leftover misses are OK.

Official 4×50 at max_steps=1000 is still required for the chosen rule.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

Key = tuple[int, int]


def _episodes(run_root: Path) -> dict[Key, dict[str, Any]]:
    out: dict[Key, dict[str, Any]] = {}
    for run_i in range(1, 5):
        summary_path = run_root / f"run{run_i}" / "summary.json"
        if not summary_path.is_file():
            summary_path = run_root / f"run{run_i}" / "shard_0" / "summary.json"
        if not summary_path.is_file():
            continue
        summary = json.loads(summary_path.read_text())
        for ep in summary["tasks"][0]["episode_results"]:
            key: Key = (run_i, int(ep["seed"]))
            row: dict[str, Any] = {
                "success": bool(ep["success"]),
                "steps": int(ep["steps"]),
                "actions_path": ep.get("actions_path"),
            }
            path = ep.get("actions_path")
            if path and Path(path).is_file():
                blob = np.load(path, allow_pickle=False)
                row["energy"] = np.asarray(blob["cfg_gate_exec_rms"], dtype=np.float64)
                row["mix"] = np.asarray(blob["cfg_mix_weights"], dtype=np.float64)
                row["query_steps"] = np.asarray(blob["policy_query_steps"], dtype=np.int32)
            out[key] = row
    return out


def _sr(rows: dict[Key, dict[str, Any]]) -> tuple[float, int, int]:
    n = len(rows)
    s = sum(1 for r in rows.values() if r["success"])
    return (s / n if n else float("nan")), s, n


def _energy_at(row: dict[str, Any] | None, index: int) -> float:
    if row is None or "energy" not in row:
        return float("nan")
    energy = row["energy"]
    if energy.size <= index:
        return float("nan")
    value = float(energy[index])
    return value if np.isfinite(value) else float("nan")


def _oracle_sets(
    base_ok: bool,
    node_ok: np.ndarray,
) -> tuple[bool, np.ndarray, str]:
    """All actions that attain the best available outcome.

    Returns (none_is_optimal, node_is_optimal, reason).
    """
    any_node = bool(node_ok.any())
    if not base_ok and not any_node:
        # Every action fails: all are equally optimal.
        return True, np.ones_like(node_ok, dtype=bool), "all_fail_any_action"
    none_ok = bool(base_ok)
    nodes = node_ok.copy()
    n_opt = int(none_ok) + int(nodes.sum())
    if base_ok and any_node:
        reason = "本体_and_some_k"
    elif base_ok:
        reason = "本体_only"
    elif n_opt > 1:
        reason = "multiple_k"
    else:
        reason = "single_k"
    return none_ok, nodes, reason


def _interval_endpoints(values: np.ndarray, *, max_unique: int = 120) -> np.ndarray:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.asarray([-np.inf, np.inf], dtype=np.float64)
    unique = np.unique(np.round(finite, 5))
    if unique.size > max_unique:
        pick = np.linspace(0, unique.size - 1, max_unique).round().astype(np.int32)
        unique = unique[pick]
    return np.unique(np.concatenate([[-np.inf], unique, [np.inf]]))


def _first_hit_success(
    energy: np.ndarray,
    node_ok: np.ndarray,
    base_ok: np.ndarray,
    lo: float,
    hi: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Causal success / fire mask / hit index (into the window) for E in [lo, hi]."""
    in_band = np.isfinite(energy) & (energy >= lo) & (energy <= hi)
    none = ~in_band.any(axis=1)
    wide = np.where(in_band, np.arange(energy.shape[1], dtype=np.int32), energy.shape[1] + 1)
    hit = wide.min(axis=1).astype(np.int32)
    hit = np.where(none, -1, hit)
    ok = base_ok.copy()
    fired = hit >= 0
    if fired.any():
        rows = np.flatnonzero(fired)
        ok[rows] = node_ok[rows, hit[rows]]
    return ok, fired, hit


def _agree_oracle(
    fired: np.ndarray,
    hit: np.ndarray,
    nodes: list[int],
    opt_none: np.ndarray,
    opt_node: np.ndarray,
) -> np.ndarray:
    agree = np.zeros(fired.shape[0], dtype=bool)
    stay = ~fired
    agree[stay] = opt_none[stay]
    rows = np.flatnonzero(fired)
    if rows.size:
        global_k = np.asarray(nodes, dtype=np.int32)[hit[rows]]
        agree[rows] = opt_node[rows, global_k]
    return agree


def _score_policy(
    *,
    name: str,
    skip_first: int,
    skip_last: int,
    lo: float | None,
    hi: float | None,
    x_label: str,
    ok: np.ndarray,
    fired: np.ndarray,
    hit: np.ndarray,
    nodes: list[int],
    opt_none: np.ndarray,
    opt_node: np.ndarray,
    base_rate: float,
) -> dict[str, Any]:
    n = int(ok.size)
    successes = int(ok.sum())
    n_fired = int(fired.sum())
    agree = _agree_oracle(fired, hit, nodes, opt_none, opt_node)
    n_agree = int(agree.sum())
    return {
        "policy": name,
        "skip_first": skip_first,
        "skip_last": skip_last,
        "lo": (None if lo is None or not np.isfinite(lo) else float(lo)),
        "hi": (None if hi is None or not np.isfinite(hi) else float(hi)),
        "X": x_label,
        "successes": successes,
        "episodes": n,
        "sr": successes / n if n else float("nan"),
        "delta_vs_本体": (successes / n if n else float("nan")) - base_rate,
        "n_fired": n_fired,
        "n_stayed_本体": n - n_fired,
        "n_agree_oracle_set": n_agree,
        "oracle_set_agree_rate": n_agree / n if n else float("nan"),
    }


def _batch_interval_scores(
    energy: np.ndarray,
    node_ok: np.ndarray,
    base_ok: np.ndarray,
    endpoints: np.ndarray,
) -> list[tuple[float, float, np.ndarray, np.ndarray, np.ndarray]]:
    pairs: list[tuple[float, float]] = []
    for i, lo in enumerate(endpoints):
        for hi in endpoints[i:]:
            pairs.append((float(lo), float(hi)))
    out: list[tuple[float, float, np.ndarray, np.ndarray, np.ndarray]] = []
    chunk = 512
    for start in range(0, len(pairs), chunk):
        block = pairs[start : start + chunk]
        los = np.asarray([p[0] for p in block], dtype=np.float64)
        his = np.asarray([p[1] for p in block], dtype=np.float64)
        in_band = (
            np.isfinite(energy)[None, :, :]
            & (energy[None, :, :] >= los[:, None, None])
            & (energy[None, :, :] <= his[:, None, None])
        )
        none = ~in_band.any(axis=2)
        width = energy.shape[1]
        wide = np.where(
            in_band,
            np.arange(width, dtype=np.int32)[None, None, :],
            width + 1,
        )
        hit = wide.min(axis=2).astype(np.int32)
        hit = np.where(none, -1, hit)
        fired = hit >= 0
        ok = np.broadcast_to(base_ok[None, :], hit.shape).copy()
        for p in range(hit.shape[0]):
            rows = np.flatnonzero(fired[p])
            if rows.size:
                ok[p, rows] = node_ok[rows, hit[p, rows]]
        for p, (lo, hi) in enumerate(block):
            out.append((lo, hi, ok[p], fired[p], hit[p]))
    return out


def _x_label(lo: float, hi: float) -> str:
    lo_s = "−∞" if not np.isfinite(lo) else f"{lo:.5f}"
    hi_s = "∞" if not np.isfinite(hi) else f"{hi:.5f}"
    return f"E in [{lo_s}, {hi_s}]"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--本体-agg", type=Path, required=True)
    parser.add_argument("--本体-root", type=Path, required=True)
    parser.add_argument("--sweep-root", type=Path, required=True)
    parser.add_argument("--max-replan", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    baseline = json.loads(args.本体_agg.read_text())
    base_rows = _episodes(args.本体_root)
    base_rate, base_s, base_n = _sr(base_rows)
    max_replan = int(args.max_replan)

    def _has_run_summary(root: Path) -> bool:
        return (root / "run1" / "summary.json").is_file() or (
            root / "run1" / "shard_0" / "summary.json"
        ).is_file()

    by_i: dict[int, dict[Key, dict[str, Any]]] = {}
    always_at = []
    for i in range(max_replan + 1):
        root = args.sweep_root / f"at{i}"
        if not _has_run_summary(root):
            continue
        rows = _episodes(root)
        by_i[i] = rows
        rate, successes, n_ep = _sr(rows)
        always_at.append(
            {
                "policy": f"always_at_{i}",
                "skip_first": i,
                "skip_last": max_replan - i,
                "X": "always",
                "successes": successes,
                "episodes": n_ep,
                "sr": rate,
                "delta_vs_本体": rate - base_rate,
            }
        )
    always_at.sort(key=lambda row: -float(row["sr"]))

    available = sorted(by_i)
    if not available:
        payload = {
            "baseline": {
                "sr": baseline.get("pooled_success_rate", base_rate),
                "pooled_successes": baseline.get("pooled_successes", base_s),
                "pooled_episodes": baseline.get("pooled_episodes", base_n),
                "note": "本体 4×50 max_steps=1000. Sweep uses max_steps=500.",
            },
            "always_at_k_ranked": always_at,
            "best": None,
            "note": "no at{i} summaries yet",
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(payload, indent=2))
        return

    keys = sorted(set(base_rows) & set.intersection(*(set(by_i[i]) for i in available)))
    n_ep = len(keys)
    n_nodes = max_replan + 1
    energy = np.full((n_ep, n_nodes), np.nan, dtype=np.float64)
    node_ok = np.zeros((n_ep, n_nodes), dtype=bool)
    base_ok = np.zeros(n_ep, dtype=bool)
    opt_none = np.zeros(n_ep, dtype=bool)
    opt_node = np.zeros((n_ep, n_nodes), dtype=bool)
    oracle_rows: list[dict[str, Any]] = []
    reason_counts: dict[str, int] = {}

    for row_i, key in enumerate(keys):
        base_ok[row_i] = bool(base_rows[key]["success"])
        energies_i = [None] * n_nodes
        success_i = [None] * n_nodes
        for node in available:
            rec = by_i[node][key]
            node_ok[row_i, node] = bool(rec["success"])
            energy[row_i, node] = _energy_at(rec, node)
            energies_i[node] = energy[row_i, node]
            success_i[node] = bool(rec["success"])
        none_ok, nodes_ok, reason = _oracle_sets(base_ok[row_i], node_ok[row_i])
        opt_none[row_i] = none_ok
        opt_node[row_i] = nodes_ok
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
        opt_actions: list[int | None] = []
        if none_ok:
            opt_actions.append(None)
        opt_actions.extend(int(k) for k in np.flatnonzero(nodes_ok))
        oracle_rows.append(
            {
                "run": key[0],
                "seed": key[1],
                "本体": bool(base_ok[row_i]),
                "at_success": success_i,
                "energy": energies_i,
                "oracle_actions": opt_actions,
                "n_optimal": len(opt_actions),
                "reason": reason,
            }
        )

    # Upper bound: pick any optimal action → success iff any action succeeds.
    oracle_ok = base_ok | node_ok.any(axis=1)
    oracle_successes = int(oracle_ok.sum())
    oracle_sr = oracle_successes / n_ep if n_ep else float("nan")
    n_opt = opt_none.astype(np.int32) + opt_node.sum(axis=1)
    n_multi = int(((n_opt > 1) & oracle_ok).sum())
    n_sizes = n_opt.astype(int)

    scored: list[dict[str, Any]] = []
    all_nodes = list(range(n_nodes))
    scored.append(
        _score_policy(
            name="never",
            skip_first=n_nodes,
            skip_last=n_nodes,
            lo=None,
            hi=None,
            x_label="never",
            ok=base_ok,
            fired=np.zeros(n_ep, dtype=bool),
            hit=np.full(n_ep, -1, dtype=np.int32),
            nodes=all_nodes,
            opt_none=opt_none,
            opt_node=opt_node,
            base_rate=base_rate,
        )
    )

    for skip_first in range(n_nodes):
        for skip_last in range(n_nodes - skip_first):
            i0 = skip_first
            i1 = n_nodes - skip_last
            nodes = [n for n in range(i0, i1) if n in by_i]
            if not nodes:
                continue
            e_win = energy[:, nodes]
            s_win = node_ok[:, nodes]
            ok, fired, hit = _first_hit_success(e_win, s_win, base_ok, -np.inf, np.inf)
            if skip_last == n_nodes - skip_first - 1:
                name = f"always_at_{skip_first}"
                x_label = "always"
            else:
                name = f"skip_{skip_first}_drop_last_{skip_last}_always"
                x_label = "always"
            scored.append(
                _score_policy(
                    name=name,
                    skip_first=skip_first,
                    skip_last=skip_last,
                    lo=None,
                    hi=None,
                    x_label=x_label,
                    ok=ok,
                    fired=fired,
                    hit=hit,
                    nodes=nodes,
                    opt_none=opt_none,
                    opt_node=opt_node,
                    base_rate=base_rate,
                )
            )

            endpoints = _interval_endpoints(e_win)
            pos = energy[opt_node] if opt_node.any() else np.asarray([], dtype=np.float64)
            pos = pos[np.isfinite(pos)]
            if pos.size:
                endpoints = np.unique(
                    np.concatenate([endpoints, np.round(pos, 5)])
                )

            local: list[dict[str, Any]] = []
            for lo, hi, ok_i, fired_i, hit_i in _batch_interval_scores(
                e_win, s_win, base_ok, endpoints
            ):
                if not np.isfinite(lo) and not np.isfinite(hi):
                    continue
                x_label = _x_label(lo, hi)
                local.append(
                    _score_policy(
                        name=f"skip_{skip_first}_drop_last_{skip_last}_{x_label}",
                        skip_first=skip_first,
                        skip_last=skip_last,
                        lo=lo,
                        hi=hi,
                        x_label=x_label,
                        ok=ok_i,
                        fired=fired_i,
                        hit=hit_i,
                        nodes=nodes,
                        opt_none=opt_none,
                        opt_node=opt_node,
                        base_rate=base_rate,
                    )
                )
            local.sort(
                key=lambda row: (
                    -int(row["successes"]),
                    -int(row["n_agree_oracle_set"]),
                    int(row["n_fired"]),
                )
            )
            scored.extend(local[:8])

    scored.sort(
        key=lambda row: (
            -int(row["successes"]),
            -int(row["n_agree_oracle_set"]),
            int(row["n_fired"]),
            int(row["skip_first"]),
        )
    )
    unique_rules: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for row in scored:
        sig = (
            row["skip_first"],
            row["skip_last"],
            row["X"],
            row["successes"],
            row["n_fired"],
        )
        if sig in seen:
            continue
        seen.add(sig)
        unique_rules.append(row)

    best = unique_rules[0] if unique_rules else None
    payload = {
        "method": (
            "oracle is the set of all outcome-optimal actions "
            "(none and/or any k that succeed when that is best); "
            "then skip_first N / drop_last L / optional contiguous E interval; "
            "score by causal at-k lookup; agree if the rule's action is in the set."
        ),
        "baseline": {
            "sr": baseline.get("pooled_success_rate", base_rate),
            "pooled_successes": baseline.get("pooled_successes", base_s),
            "pooled_episodes": baseline.get("pooled_episodes", base_n),
            "note": "本体 4×50 max_steps=1000. Sweep uses max_steps=500.",
        },
        "oracle": {
            "successes": oracle_successes,
            "episodes": n_ep,
            "sr": oracle_sr,
            "delta_vs_本体": oracle_sr - base_rate,
            "n_multi_optimal": n_multi,
            "mean_n_optimal_actions": float(n_sizes.mean()) if n_ep else float("nan"),
            "reason_counts": reason_counts,
            "note": (
                "Per-episode SET of best actions in {no CFG, CFG once at a node}. "
                "Not a deployable rule; upper bound for this sweep window."
            ),
        },
        "always_at_k_ranked": always_at,
        "ranked_top": unique_rules[:25],
        "best": best,
        "oracle_episodes": oracle_rows,
        "available_nodes": available,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    summary = {
        "oracle": payload["oracle"],
        "best": best,
        "ranked_top": payload["ranked_top"],
        "always_at_k_ranked": always_at,
        "available_nodes": available,
        "baseline": payload["baseline"],
        "method": payload["method"],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
