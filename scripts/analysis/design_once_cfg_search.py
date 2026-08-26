#!/usr/bin/env python3
"""Design / rank once-CFG search configs (order-aware, strata-filtered).

Strata (本体 × always-CFG on the same seeds/repeats):
  both_ok     — 全程 CFG 仍成功。搜参阶段剔除；终案再抽检。
  fail        — 本体失败。救援候选。
  fragile     — 本体成功、全程 CFG 失败。伤害约束：干预后仍须成功。
  cfg_rescued — 本体失败、全程 CFG 成功。可选对照，不进主搜。

Policy (online, causal):
  1. Skip replan indices ``0 .. K-1`` (pre-truncate; K=0 allows t=0).
  2. At the **first** replan with index>=K and gate energy E in band, fire CFG once.
  3. All later replans stay 本体. Later high-E nodes never matter if step 2 already fired
     (and success trajectories often end before late nodes).

Offline schedule from baseline E dumps approximates (2) on the 本体 prefix.
Actual rescue/damage must be measured by closed-loop re-roll (SCREEN=fail|fragile).

Example:
  python scripts/analysis/design_once_cfg_search.py \\
    --本体-root .../cfg1_本体_4x50_... \\
    --always-cfg-root .../cfg1.05_4x50_... \\
    --probe-root .../cfg_strength_probe_... \\
    --out-dir .../partitions/search_design
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def _load_success_map(root: Path) -> dict[tuple[int, int], bool]:
    out: dict[tuple[int, int], bool] = {}
    for run_i in range(1, 5):
        summary = json.loads((root / f"run{run_i}" / "summary.json").read_text(encoding="utf-8"))
        for ep in summary["tasks"][0]["episode_results"]:
            out[(run_i, int(ep["seed"]))] = bool(ep["success"])
    return out


def _energy(blob: np.lib.npyio.NpzFile) -> np.ndarray:
    if "cfg_gate_exec_rms" in blob.files:
        arr = np.asarray(blob["cfg_gate_exec_rms"], dtype=np.float64)
        if np.isfinite(arr).any():
            return arr
    return np.asarray(blob["cfg_exec_rms"], dtype=np.float64)


def _load_probe_rows(probe_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run_i in range(1, 5):
        residual = probe_root / f"run{run_i}" / "residual"
        if not residual.is_dir():
            continue
        for path in sorted(residual.glob("seed_*.npz")):
            blob = np.load(path, allow_pickle=False)
            query_steps = np.asarray(blob["query_steps"], dtype=np.int32)
            if "replan_indices" in blob.files:
                replan_indices = np.asarray(blob["replan_indices"], dtype=np.int32)
            else:
                replan_indices = np.arange(len(query_steps), dtype=np.int32)
            rows.append(
                {
                    "eval_repeat": run_i - 1,
                    "run": run_i,
                    "seed": int(blob["env_seed"]),
                    "success": bool(blob["success"]),
                    "energy": _energy(blob),
                    "query_steps": query_steps,
                    "replan_indices": replan_indices,
                }
            )
    return rows


def _first_hit(
    energy: np.ndarray,
    replan_indices: np.ndarray,
    *,
    lo: float,
    hi: float | None,
    min_replan: int,
) -> tuple[int, int, float] | None:
    """Return (local_i, replan_index, E) of first in-band node with replan>=min."""
    for local_i, (ri, e) in enumerate(
        zip(replan_indices.tolist(), energy.tolist(), strict=True)
    ):
        ri_i = int(ri)
        e_f = float(e)
        if ri_i < min_replan or not np.isfinite(e_f):
            continue
        if hi is None:
            if e_f >= lo:
                return local_i, ri_i, e_f
        elif lo <= e_f < hi:
            return local_i, ri_i, e_f
    return None


def _band_edges(values: np.ndarray, n_bands: int) -> list[tuple[float, float | None, str]]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("empty energy pool for bands")
    qs = np.linspace(0.0, 1.0, n_bands + 1)
    edges = np.quantile(values, qs)
    out: list[tuple[float, float | None, str]] = []
    for i in range(n_bands):
        lo = float(edges[i])
        hi: float | None
        if i == n_bands - 1:
            hi = None
            name = f"q{i * (100 // n_bands):02d}_100"
        else:
            hi = float(edges[i + 1])
            name = f"q{i * (100 // n_bands):02d}_{((i + 1) * (100 // n_bands)):02d}"
        out.append((lo, hi, name))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--本体-root", dest="base_root", type=Path, required=True)
    parser.add_argument("--always-cfg-root", type=Path, required=True)
    parser.add_argument("--probe-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--n-bands", type=int, default=4)
    parser.add_argument(
        "--min-replan-grid",
        type=str,
        default="0,1,2,3",
        help="Comma-separated K values (skip first K replans).",
    )
    args = parser.parse_args()

    base = _load_success_map(args.base_root.expanduser().resolve())
    cfg = _load_success_map(args.always_cfg_root.expanduser().resolve())
    strata: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for key, sa in base.items():
        sb = cfg[key]
        if sa and sb:
            strata["both_ok"].append(key)
        elif (not sa) and (not sb):
            strata["fail"].append(key)
        elif sa and (not sb):
            strata["fragile"].append(key)
        else:
            strata["cfg_rescued"].append(key)

    # Search pool: all 本体 failures ∪ fragile. Exclude only both_ok.
    本体_fail = list(strata["fail"]) + list(strata["cfg_rescued"])
    search_keys = set(本体_fail) | set(strata["fragile"])
    both_ok_n = len(strata["both_ok"])

    probe_rows = _load_probe_rows(args.probe_root.expanduser().resolve())

    pool: list[float] = []
    search_set = set(search_keys)
    for row in probe_rows:
        key = (row["run"], row["seed"])
        if key not in search_set:
            continue
        for e in row["energy"].tolist():
            if np.isfinite(float(e)):
                pool.append(float(e))

    if not pool:
        raise SystemExit(
            "No probe energies for 本体_fail∪fragile yet. Dump residuals on those strata first."
        )

    bands = _band_edges(np.asarray(pool, dtype=np.float64), int(args.n_bands))
    min_grid = [int(x) for x in str(args.min_replan_grid).split(",") if x.strip() != ""]

    fail_set = set(本体_fail)
    fragile_set = set(strata["fragile"])

    candidates: list[dict[str, Any]] = []
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    for min_replan in min_grid:
        for lo, hi, bname in bands:
            fail_sched: dict[str, int] = {}
            fragile_sched: dict[str, int] = {}
            fail_hit = fragile_hit = fail_miss = fragile_miss = 0
            fail_idx: list[int] = []
            fragile_idx: list[int] = []

            for row in probe_rows:
                key_t = (row["run"], row["seed"])
                rep_key = f"{row['eval_repeat']}:{row['seed']}"
                hit = _first_hit(
                    row["energy"],
                    row["replan_indices"],
                    lo=lo,
                    hi=hi,
                    min_replan=min_replan,
                )
                if key_t in fail_set:
                    if hit is None:
                        fail_miss += 1
                    else:
                        fail_hit += 1
                        fail_sched[rep_key] = hit[1]
                        fail_idx.append(hit[1])
                elif key_t in fragile_set:
                    if hit is None:
                        fragile_miss += 1
                    else:
                        fragile_hit += 1
                        fragile_sched[rep_key] = hit[1]
                        fragile_idx.append(hit[1])

            # Prefer configs that hit many fails but few fragiles (offline fire rate).
            # Final verdict still needs closed-loop SCREEN=fail|fragile.
            score = fail_hit - 1.5 * fragile_hit
            tag = f"{bname}_minr{min_replan}"
            fail_flat = out_dir / f"schedule_{tag}_fail_flat.json"
            frag_flat = out_dir / f"schedule_{tag}_fragile_flat.json"
            fail_flat.write_text(json.dumps(fail_sched, indent=2) + "\n", encoding="utf-8")
            frag_flat.write_text(json.dumps(fragile_sched, indent=2) + "\n", encoding="utf-8")
            cand = {
                "tag": tag,
                "band": bname,
                "lo": lo,
                "hi": hi,
                "min_replan_index": min_replan,
                "fail_hit": fail_hit,
                "fail_miss": fail_miss,
                "fragile_hit": fragile_hit,
                "fragile_miss": fragile_miss,
                "offline_score_fail_minus_1p5_fragile": score,
                "fail_replan_idx_hist": Counter(fail_idx).most_common(8),
                "fragile_replan_idx_hist": Counter(fragile_idx).most_common(8),
                "schedule_fail": str(fail_flat),
                "schedule_fragile": str(frag_flat),
            }
            candidates.append(cand)

    candidates.sort(
        key=lambda c: (
            -float(c["offline_score_fail_minus_1p5_fragile"]),
            -int(c["fail_hit"]),
            int(c["fragile_hit"]),
        )
    )

    strata_payload = {
        "protocol": {
            "exclude_from_search": ["both_ok"],
            "search_on": ["本体_fail", "fragile"],
            "本体_fail_includes": ["both_fail", "cfg_rescued"],
            "policy": (
                "Skip replan 0..K-1; fire CFG once at first E-in-band node; "
                "thereafter 本体. Later band nodes are irrelevant once fired "
                "(success paths often never reach them)."
            ),
            "metric": (
                "Closed-loop: rescue on 本体_fail SCREEN, break_rate on fragile SCREEN. "
                "Accept if rescued > broken. both_ok only in final full eval."
            ),
            "max_env_steps_search": 500,
        },
        "counts": {
            **{k: len(v) for k, v in strata.items()},
            "本体_fail": len(本体_fail),
            "search_pool": len(search_keys),
        },
        "both_ok_excluded_n": both_ok_n,
        "fail_seeds_by_eval_repeat": _by_repeat(本体_fail),
        "fragile_seeds_by_eval_repeat": _by_repeat(strata["fragile"]),
        "both_fail_seeds_by_eval_repeat": _by_repeat(strata["fail"]),
        "cfg_rescued_seeds_by_eval_repeat": _by_repeat(strata["cfg_rescued"]),
        "probe_coverage": {
            "n_probe_rows": len(probe_rows),
            "n_本体_fail_with_probe": sum(
                1 for r in probe_rows if (r["run"], r["seed"]) in fail_set
            ),
            "n_fragile_with_probe": sum(
                1 for r in probe_rows if (r["run"], r["seed"]) in fragile_set
            ),
            "note": "Dump 本体_fail+fragile early residuals before trusting fragile_hit ranks.",
        },
        "candidates_ranked": candidates,
    }
    out_path = out_dir / "search_design.json"
    out_path.write_text(json.dumps(strata_payload, indent=2) + "\n", encoding="utf-8")
    # Convenience copies for SCREEN= scripts
    (out_dir / "fail_seeds_by_eval_repeat.json").write_text(
        json.dumps(_by_repeat(strata["fail"]), indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / "fragile_seeds_by_eval_repeat.json").write_text(
        json.dumps(_by_repeat(strata["fragile"]), indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: strata_payload[k] for k in ("protocol", "counts", "probe_coverage")}, indent=2))
    print("top candidates:")
    for c in candidates[:8]:
        print(
            f"  {c['tag']}: fail_hit={c['fail_hit']} fragile_hit={c['fragile_hit']} "
            f"score={c['offline_score_fail_minus_1p5_fragile']}"
        )
    print(f"wrote {out_path}")


def _by_repeat(keys: list[tuple[int, int]]) -> dict[str, list[int]]:
    out: dict[str, list[int]] = defaultdict(list)
    for run, seed in keys:
        out[str(run - 1)].append(int(seed))
    return {k: sorted(set(v)) for k, v in sorted(out.items(), key=lambda kv: int(kv[0]))}


if __name__ == "__main__":
    main()
