#!/usr/bin/env python3
"""Build datasets with probe-event p prepended to the action vector."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from probe_event_transition import low_pass_filter, moving_average, robust_probability, trend_errors


DEFAULT_DATASETS = [
    "data/pinch_tongs_fastwam",
    "data/hammer_nail_fastwam",
    "data/fold_glasses_fastwam",
    "data/water_plant_fastwam",
]


def read_actions(path: Path) -> np.ndarray:
    table = pq.ParquetFile(path).read(columns=["action"])
    return np.asarray(table["action"].combine_chunks().to_pylist(), dtype=np.float32)


def episode_index_from_path(path: Path) -> int:
    return int(path.stem.split("_")[-1])


def p_data_stats(values: np.ndarray) -> dict[str, float | list[int]]:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        finite = np.asarray([0.0], dtype=np.float32)
    return {
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "count": [int(len(values))],
        "q01": float(np.quantile(finite, 0.01)),
        "q10": float(np.quantile(finite, 0.10)),
        "q50": float(np.quantile(finite, 0.50)),
        "q90": float(np.quantile(finite, 0.90)),
        "q99": float(np.quantile(finite, 0.99)),
    }


def p_normalization_stats(count: int) -> dict[str, float | list[int]]:
    """Fixed stats for p so original action normalization is only shifted, not recomputed.

    p is already clipped to [0, 1]. With min/max normalization this maps p to
    [-1, 1], while action[1:23] keeps the exact original 22-D action stats.
    """
    return {
        "min": 0.0,
        "max": 1.0,
        "mean": 0.0,
        "std": 1.0,
        "count": [int(count)],
        "q01": 0.0,
        "q10": 0.0,
        "q50": 0.5,
        "q90": 1.0,
        "q99": 1.0,
    }


def prepend_action_stats(action_stats: dict[str, Any], p_stat: dict[str, Any]) -> dict[str, Any]:
    updated = {}
    for key, value in action_stats.items():
        if isinstance(value, list) and len(value) > 1:
            updated[key] = [p_stat[key]] + value
        else:
            updated[key] = value
    return updated


def copy_dataset_shell(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for child in src.iterdir():
        target = dst / child.name
        if child.name == "data":
            target.mkdir(exist_ok=True)
        elif child.name == "videos":
            if target.exists() or target.is_symlink():
                continue
            os.symlink(os.path.abspath(child), target, target_is_directory=True)
        elif child.name == "meta" and child.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(child, target)
        elif child.is_file():
            shutil.copy2(child, target)


def write_episode_with_p(src_path: Path, dst_path: Path, p: np.ndarray) -> None:
    table = pq.ParquetFile(src_path).read()
    actions = np.asarray(table["action"].combine_chunks().to_pylist(), dtype=np.float32)
    p = np.nan_to_num(p, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)
    p = np.clip(p, 0.0, 1.0)
    actions_with_p = np.concatenate([p[:, None], actions], axis=1).astype(np.float32)

    flat = pa.array(actions_with_p.reshape(-1), type=pa.float32())
    action_array = pa.FixedSizeListArray.from_arrays(flat, actions_with_p.shape[1])
    action_field = pa.field("action", pa.list_(pa.float32(), actions_with_p.shape[1]))
    action_idx = table.schema.get_field_index("action")
    table = table.set_column(action_idx, action_field, action_array)

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, dst_path)


def update_meta(src: Path, dst: Path, p_by_episode: dict[int, np.ndarray], p_all: np.ndarray) -> None:
    p_norm_stat = p_normalization_stats(len(p_all))
    p_actual_stat = p_data_stats(p_all)

    info_path = dst / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info["features"]["action"]["shape"] = [23]
    info_path.write_text(json.dumps(info, indent=4), encoding="utf-8")

    modality_path = dst / "meta" / "modality.json"
    modality = json.loads(modality_path.read_text())
    modality["action"] = {
        "event_transition_p": {"start": 0, "end": 1},
        "eef_target": {"start": 1, "end": 7},
        "hand_joints": {"start": 7, "end": 23},
    }
    modality_path.write_text(json.dumps(modality, indent=2), encoding="utf-8")

    for name in ["stats.json", "relative_stats.json"]:
        path = dst / "meta" / name
        if not path.exists():
            continue
        stats = json.loads(path.read_text())
        if "action" in stats:
            stats["action"] = prepend_action_stats(stats["action"], p_norm_stat)
            path.write_text(json.dumps(stats, indent=4), encoding="utf-8")

    src_eps_stats = src / "meta" / "episodes_stats.jsonl"
    dst_eps_stats = dst / "meta" / "episodes_stats.jsonl"
    if src_eps_stats.exists():
        with src_eps_stats.open() as fin, dst_eps_stats.open("w") as fout:
            for line in fin:
                row = json.loads(line)
                ep = int(row["episode_index"])
                if "action" in row.get("stats", {}):
                    row["stats"]["action"] = prepend_action_stats(
                        row["stats"]["action"],
                        p_normalization_stats(len(p_by_episode[ep])),
                    )
                fout.write(json.dumps(row) + "\n")

    (dst / "meta" / "probe_event_p_stats.json").write_text(
        json.dumps(
            {
                "actual_p_stats": p_actual_stat,
                "normalization_p_stats": p_norm_stat,
                "note": (
                    "stats.json action[0] uses fixed [0, 1] p stats. "
                    "stats.json action[1:23] is copied from the original 22-D action stats unchanged."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def build_one_dataset(src: Path, dst_root: Path, args: argparse.Namespace) -> Path:
    dst = dst_root / f"{src.name}_probe_event"
    parquet_paths = sorted((src / "data").glob("chunk-*/*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"No parquet files found under {src / 'data'}")

    actions_by_path = {path: read_actions(path) for path in parquet_paths}
    all_actions = np.concatenate(list(actions_by_path.values()), axis=0)
    scale = all_actions.std(axis=0).astype(np.float32)
    scale[scale < 1e-6] = 1.0

    errors_by_path = {}
    all_errors = []
    for path, actions in actions_by_path.items():
        err = trend_errors(actions, scale, args.history, args.action_groups)
        errors_by_path[path] = err
        all_errors.append(err["error"])

    p_all, calibration = robust_probability(
        np.concatenate(all_errors),
        low_q=args.low_quantile,
        high_q=args.high_quantile,
    )

    copy_dataset_shell(src, dst)

    cursor = 0
    p_by_episode: dict[int, np.ndarray] = {}
    p_final_all = []
    for path in parquet_paths:
        n = len(actions_by_path[path])
        p = p_all[cursor : cursor + n]
        cursor += n
        p_lpf = low_pass_filter(p, args.lpf_alpha, args.lpf_release_alpha)
        p_smooth = moving_average(p_lpf, args.smooth)
        p_final = np.nan_to_num(p_smooth, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)
        p_final[: args.history] = 0.0

        rel = path.relative_to(src)
        dst_path = dst / rel
        write_episode_with_p(path, dst_path, p_final)
        ep = episode_index_from_path(path)
        p_by_episode[ep] = p_final
        p_final_all.append(p_final)

    update_meta(src, dst, p_by_episode, np.concatenate(p_final_all))
    (dst / ".fastwam_prepared").touch()

    summary = {
        "source_dataset": str(src),
        "output_dataset": str(dst),
        "num_episodes": len(parquet_paths),
        "num_frames": int(sum(len(v) for v in actions_by_path.values())),
        "history": args.history,
        "action_groups": args.action_groups,
        "p_column": "action[0]",
        "original_action_columns": "action[1:23]",
        "p_filter": {
            "low_quantile": args.low_quantile,
            "high_quantile": args.high_quantile,
            "lpf_alpha": args.lpf_alpha,
            "lpf_release_alpha": args.lpf_release_alpha,
            "smooth": args.smooth,
        },
        "error_to_p_calibration": calibration,
        "video_storage": "symlink",
        "normalization_note": (
            "Original 22-D action stats are copied unchanged and shifted to action[1:23]. "
            "The inserted p column has fixed [0, 1] stats at action[0], so p never changes the "
            "normalization of the original action dimensions."
        ),
    }
    (dst / "meta" / "probe_event_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return dst


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", type=Path, default=[Path(p) for p in DEFAULT_DATASETS])
    parser.add_argument("--output-root", type=Path, default=Path("data/probe_event_fastwam"))
    parser.add_argument("--history", type=int, default=2)
    parser.add_argument(
        "--action-groups",
        nargs="+",
        choices=["eef", "hand"],
        default=["eef"],
    )
    parser.add_argument("--low-quantile", type=float, default=0.10)
    parser.add_argument("--high-quantile", type=float, default=0.95)
    parser.add_argument("--lpf-alpha", type=float, default=0.28)
    parser.add_argument("--lpf-release-alpha", type=float, default=0.08)
    parser.add_argument("--smooth", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    outputs = []
    for src in args.datasets:
        dst = build_one_dataset(src, args.output_root, args)
        outputs.append(str(dst))
        print(dst)
    (args.output_root / "manifest.json").write_text(
        json.dumps({"datasets": outputs}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
