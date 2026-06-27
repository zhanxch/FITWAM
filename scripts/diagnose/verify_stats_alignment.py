#!/usr/bin/env python3
"""Verify that FastWAM-recomputed stats match GR00T's meta/stats.json.

Computes global statistics directly from parquet (mirroring GR00T's
calculate_dataset_statistics in gr00t/data/stats.py) and compares against
the precomputed meta/stats.json, sliced per modality key via modality.json.

Also recomputes stats the FastWAM way (per-episode then aggregate) to
quantify any divergence from the global parquet approach.

Usage:
    PYTHONPATH=src python scripts/diagnose/verify_stats_alignment.py \
        --dataset-dir data/spray_water_rot6d_rosbag_ts_filter
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


META_STATS = "meta/stats.json"
META_MODALITY = "meta/modality.json"
PARQUET_GLOB = "data/*/*.parquet"
LOWDIM_COLS = ("action", "observation.state")


def load_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def gr00t_style_global_stats(parquet_dir: Path, cols: tuple[str, ...]) -> dict:
    """Mirror GR00T calculate_dataset_statistics: concat all parquet, global axis=0 stats."""
    files = sorted(parquet_dir.glob(PARQUET_GLOB))
    print(f"Loading {len(files)} parquet files...")
    frames = [pd.read_parquet(p) for p in files]
    all_data = pd.concat(frames, axis=0)
    stats = {}
    for col in cols:
        print(f"  Computing global stats for {col}...")
        arr = np.vstack([np.asarray(x, dtype=np.float32) for x in all_data[col]])
        stats[col] = {
            "mean": np.mean(arr, axis=0),
            "std": np.std(arr, axis=0),
            "min": np.min(arr, axis=0),
            "max": np.max(arr, axis=0),
            "q01": np.quantile(arr, 0.01, axis=0),
            "q99": np.quantile(arr, 0.99, axis=0),
        }
    return stats


def fastwam_style_episode_stats(parquet_dir: Path, cols: tuple[str, ...]) -> dict:
    """Mirror FastWAM get_dataset_stats: per-episode stats then aggregate.

    global_min = stack(per_ep_min).amin(0)
    global_mean = means.mean((0,1))  (mean of per-episode means, then over steps)
    global_std = (vars + (means - global_mean)^2).mean((0,1)).sqrt()  (pooled)
    """
    files = sorted(parquet_dir.glob(PARQUET_GLOB))
    print(f"Loading {len(files)} parquet files (per-episode)...")
    per_ep = {c: {"min": [], "max": [], "mean": [], "var": [], "q01": [], "q99": []} for c in cols}
    for p in files:
        df = pd.read_parquet(p)
        for col in cols:
            arr = np.vstack([np.asarray(x, dtype=np.float32) for x in df[col]])
            per_ep[col]["min"].append(np.min(arr, axis=0))
            per_ep[col]["max"].append(np.max(arr, axis=0))
            per_ep[col]["mean"].append(np.mean(arr, axis=0))
            per_ep[col]["var"].append(np.var(arr, axis=0))
            per_ep[col]["q01"].append(np.quantile(arr, 0.01, axis=0))
            per_ep[col]["q99"].append(np.quantile(arr, 0.99, axis=0))

    stats = {}
    for col in cols:
        mins = np.stack(per_ep[col]["min"])      # (num_ep, D)
        maxs = np.stack(per_ep[col]["max"])
        means = np.stack(per_ep[col]["mean"])
        vars_ = np.stack(per_ep[col]["var"])
        q01s = np.stack(per_ep[col]["q01"])
        q99s = np.stack(per_ep[col]["q99"])
        global_min = mins.min(0)
        global_max = maxs.max(0)
        global_mean = means.mean(0)
        global_std = np.sqrt((vars_ + (means - global_mean) ** 2).mean(0))
        global_q01 = q01s.min(0)
        global_q99 = q99s.max(0)
        stats[col] = {
            "mean": global_mean,
            "std": global_std,
            "min": global_min,
            "max": global_max,
            "q01": global_q01,
            "q99": global_q99,
        }
    return stats


def compare(a: dict, b: dict, label_a: str, label_b: str, keys: tuple[str, ...]) -> dict:
    """Compare two stats dicts keyed by stat-name -> ndarray. Returns max abs diff per stat."""
    report = {}
    for stat in ("min", "max", "mean", "std", "q01", "q99"):
        if stat not in a or stat not in b:
            continue
        va, vb = np.asarray(a[stat], dtype=np.float64), np.asarray(b[stat], dtype=np.float64)
        if va.shape != vb.shape:
            report[stat] = f"shape mismatch {va.shape} vs {vb.shape}"
            continue
        diff = np.abs(va - vb)
        report[stat] = {
            "max_abs_diff": float(diff.max()),
            "max_rel_diff": float((diff / (np.abs(vb) + 1e-8)).max()),
            "argmax_dim": int(diff.argmax()),
        }
    return report


def slice_by_modality(arr: np.ndarray, modality_meta: dict, modality_type: str) -> dict:
    """Slice a flat (58,) array into per-key sub-arrays using modality.json start/end."""
    out = {}
    for key, info in modality_meta[modality_type].items():
        s, e = info["start"], info["end"]
        out[key] = np.asarray(arr[s:e], dtype=np.float64)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", default="data/spray_water_rot6d_rosbag_ts_filter")
    args = ap.parse_args()
    ds = Path(args.dataset_dir)

    modality_meta = load_json(ds / META_MODALITY)
    meta_stats = load_json(ds / META_STATS)

    print("\n=== (1) GR00T-style global stats vs meta/stats.json ===")
    gr00t_stats = gr00t_style_global_stats(ds, LOWDIM_COLS)
    all_match = True
    for col in LOWDIM_COLS:
        print(f"\n  Column: {col}")
        rep = compare(gr00t_stats[col], meta_stats[col], "gr00t-global", "meta/stats.json", LOWDIM_COLS)
        for stat, info in rep.items():
            if isinstance(info, str):
                print(f"    {stat}: {info}")
                all_match = False
            else:
                ok = info["max_abs_diff"] < 1e-4
                tag = "OK " if ok else "DIFF"
                print(f"    {stat}: {tag} max_abs={info['max_abs_diff']:.2e} "
                      f"max_rel={info['max_rel_diff']:.2e} @dim{info['argmax_dim']}")
                if not ok:
                    all_match = False
        # per-modality-key slice comparison
        print("    per-modality-key slices:")
        for stat in ("min", "max", "mean", "std"):
            g = slice_by_modality(gr00t_stats[col][stat], modality_meta,
                                  "action" if col == "action" else "state")
            m = slice_by_modality(np.asarray(meta_stats[col][stat], dtype=np.float64),
                                  modality_meta, "action" if col == "action" else "state")
            for key in g:
                d = np.abs(g[key] - m[key]).max()
                ok = d < 1e-4
                if not ok:
                    all_match = False
                print(f"      {key} {stat}: {'OK ' if ok else 'DIFF'} max_abs={d:.2e}")

    print("\n=== (2) FastWAM per-episode stats vs meta/stats.json ===")
    fw_stats = fastwam_style_episode_stats(ds, LOWDIM_COLS)
    for col in LOWDIM_COLS:
        print(f"\n  Column: {col}")
        rep = compare(fw_stats[col], meta_stats[col], "fastwam-episode", "meta/stats.json", LOWDIM_COLS)
        for stat, info in rep.items():
            if isinstance(info, str):
                print(f"    {stat}: {info}")
            else:
                ok = info["max_abs_diff"] < 1e-4
                tag = "OK " if ok else "DIFF"
                print(f"    {stat}: {tag} max_abs={info['max_abs_diff']:.2e} "
                      f"max_rel={info['max_rel_diff']:.2e} @dim{info['argmax_dim']}")

    print("\n=== Summary ===")
    print(f"GR00T-global vs meta/stats.json: {'ALL MATCH (<1e-4)' if all_match else 'MISMATCH detected'}")
    if all_match:
        print("Conclusion: meta/stats.json == global parquet stats. Safe to use meta/ directly.")
    else:
        print("Conclusion: divergence detected. Inspect diffs above before using meta/.")


if __name__ == "__main__":
    main()
