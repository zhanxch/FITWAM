#!/usr/bin/env python3
"""Partition baseline replan nodes by CFG gate strength and build once-intervene schedules.

Inputs: residual dumps from ``dump_cfg_residual_on_eval_traj.py`` (prefer
``cfg_gate_exec_rms`` = NFE0 exec RMS; fall back to ``cfg_exec_rms``).

For each strength band (quartile of replan-node energies), write a schedule
that fires guided CFG **once** at the earliest in-band replan on each **failed**
baseline trajectory. Success episodes are omitted from the fail schedule.

``--min-replan-index N`` skips the first N replans (e.g. 1 = never intervene at
env step 0). Use when start-of-episode CFG damages fragile successes.

Example:
  python scripts/analysis/partition_cfg_strength_once_intervene.py \\
    --probe-root evaluate_results/.../cfg_strength_probe \\
    --out-dir evaluate_results/.../partitions --pool fail --min-replan-index 1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def _energy(blob: np.lib.npyio.NpzFile) -> np.ndarray:
    if "cfg_gate_exec_rms" in blob.files:
        arr = np.asarray(blob["cfg_gate_exec_rms"], dtype=np.float64)
        if np.isfinite(arr).any():
            return arr
    if "cfg_exec_rms" not in blob.files:
        raise KeyError("dump missing cfg_gate_exec_rms and cfg_exec_rms")
    return np.asarray(blob["cfg_exec_rms"], dtype=np.float64)


def _load_run(run_dir: Path, eval_repeat: int) -> list[dict[str, Any]]:
    residual = run_dir / "residual"
    if not residual.is_dir():
        raise FileNotFoundError(residual)
    rows: list[dict[str, Any]] = []
    for path in sorted(residual.glob("seed_*.npz")):
        blob = np.load(path, allow_pickle=False)
        energy = _energy(blob)
        query_steps = np.asarray(blob["query_steps"], dtype=np.int32)
        if "replan_indices" in blob.files:
            replan_indices = np.asarray(blob["replan_indices"], dtype=np.int32)
        else:
            replan_indices = np.arange(len(query_steps), dtype=np.int32)
        rows.append(
            {
                "eval_repeat": int(eval_repeat),
                "seed": int(blob["env_seed"]),
                "success": bool(blob["success"]),
                "energy": energy,
                "query_steps": query_steps,
                "replan_indices": replan_indices,
                "path": str(path),
            }
        )
    return rows


def _band_edges(values: np.ndarray, n_bands: int) -> list[tuple[float, float, str]]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("no finite energy values to partition")
    qs = np.linspace(0.0, 1.0, n_bands + 1)
    edges = np.quantile(values, qs)
    bands: list[tuple[float, float, str]] = []
    for i in range(n_bands):
        lo = float(edges[i])
        hi = float(edges[i + 1])
        if i == n_bands - 1:
            hi = float(edges[-1] + 1e-12)
        name = f"q{i * (100 // n_bands):02d}_{((i + 1) * (100 // n_bands)):02d}"
        bands.append((lo, hi, name))
    return bands


def _earliest_in_band(
    energy: np.ndarray, *, lo: float, hi: float, top_band: bool
) -> int | None:
    for i, e in enumerate(energy.tolist()):
        e = float(e)
        if not np.isfinite(e):
            continue
        if top_band:
            if e >= lo:
                return int(i)
        elif lo <= e < hi:
            return int(i)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--n-bands", type=int, default=4)
    parser.add_argument("--pool", choices=("all", "fail"), default="all")
    parser.add_argument(
        "--min-replan-index",
        type=int,
        default=0,
        help="Do not intervene before this replan index (0 = allow t=0).",
    )
    args = parser.parse_args()

    probe_root = args.probe_root.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    min_replan = int(args.min_replan_index)
    if min_replan < 0:
        raise ValueError(f"--min-replan-index must be >= 0, got {min_replan}")

    all_rows: list[dict[str, Any]] = []
    for run_i in range(1, 16):
        run_dir = probe_root / f"run{run_i}"
        if not run_dir.is_dir():
            break
        all_rows.extend(_load_run(run_dir, eval_repeat=run_i - 1))
    if not all_rows:
        raise FileNotFoundError(f"No run*/residual dumps under {probe_root}")

    pool_vals: list[float] = []
    for row in all_rows:
        if args.pool == "fail" and row["success"]:
            continue
        for local_i, e in enumerate(row["energy"].tolist()):
            if int(row["replan_indices"][local_i]) < min_replan:
                continue
            if np.isfinite(float(e)):
                pool_vals.append(float(e))
    bands = _band_edges(np.asarray(pool_vals, dtype=np.float64), int(args.n_bands))

    band_reports: list[dict[str, Any]] = []
    for bi, (lo, hi, name) in enumerate(bands):
        top = bi == len(bands) - 1
        by_key: dict[str, int] = {}
        details: list[dict[str, Any]] = []
        n_fail = 0
        n_scheduled = 0
        for row in all_rows:
            if row["success"]:
                continue
            n_fail += 1
            mask = [
                i
                for i, ri in enumerate(row["replan_indices"].tolist())
                if int(ri) >= min_replan
            ]
            if not mask:
                details.append(
                    {
                        "eval_repeat": row["eval_repeat"],
                        "seed": row["seed"],
                        "scheduled": False,
                        "reason": "no_replan_after_min",
                    }
                )
                continue
            energy_sub = row["energy"][mask]
            idx_sub = _earliest_in_band(energy_sub, lo=lo, hi=hi, top_band=top)
            if idx_sub is None:
                details.append(
                    {
                        "eval_repeat": row["eval_repeat"],
                        "seed": row["seed"],
                        "scheduled": False,
                    }
                )
                continue
            local_i = int(mask[idx_sub])
            idx = int(row["replan_indices"][local_i])
            key = f"{row['eval_repeat']}:{row['seed']}"
            by_key[key] = int(idx)
            n_scheduled += 1
            details.append(
                {
                    "eval_repeat": row["eval_repeat"],
                    "seed": row["seed"],
                    "scheduled": True,
                    "replan_index": int(idx),
                    "env_step": int(row["query_steps"][local_i]),
                    "energy": float(row["energy"][local_i]),
                }
            )

        suffix = "" if min_replan == 0 else f"_minr{min_replan}"
        schedule = {
            "band": name,
            "band_lo": lo,
            "band_hi": hi if not top else float("inf"),
            "pool": args.pool,
            "min_replan_index": min_replan,
            "n_fail": n_fail,
            "n_scheduled": n_scheduled,
            "by_key": by_key,
            "details": details,
        }
        band_path = out_dir / f"schedule_{name}{suffix}.json"
        band_path.write_text(json.dumps(schedule, indent=2) + "\n", encoding="utf-8")
        flat_path = out_dir / f"schedule_{name}{suffix}_flat.json"
        flat_path.write_text(json.dumps(by_key, indent=2) + "\n", encoding="utf-8")
        band_reports.append(
            {
                "band": name,
                "lo": lo,
                "hi": hi if not top else None,
                "min_replan_index": min_replan,
                "n_scheduled": n_scheduled,
                "n_fail": n_fail,
                "schedule": str(flat_path),
            }
        )
        print(
            f"[partition] {name}{suffix} lo={lo:.5f} hi={hi:.5f} "
            f"min_replan={min_replan} scheduled={n_scheduled}/{n_fail} -> {flat_path}",
            flush=True,
        )

    fail_by_repeat: dict[str, list[int]] = {}
    for row in all_rows:
        if row["success"]:
            continue
        key = str(row["eval_repeat"])
        fail_by_repeat.setdefault(key, []).append(int(row["seed"]))
    for key, seeds in fail_by_repeat.items():
        fail_by_repeat[key] = sorted(set(seeds))

    summary = {
        "probe_root": str(probe_root),
        "n_episodes": len(all_rows),
        "n_fail": sum(1 for r in all_rows if not r["success"]),
        "pool": args.pool,
        "n_bands": int(args.n_bands),
        "min_replan_index": min_replan,
        "energy_mean": float(np.mean(pool_vals)),
        "energy_p50": float(np.median(pool_vals)),
        "energy_p90": float(np.quantile(pool_vals, 0.9)),
        "bands": band_reports,
        "fail_seeds_by_eval_repeat": fail_by_repeat,
    }
    summary_path = out_dir / (
        "partition_summary.json"
        if min_replan == 0
        else f"partition_summary_minr{min_replan}.json"
    )
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
