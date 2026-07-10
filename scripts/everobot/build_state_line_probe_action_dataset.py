#!/usr/bin/env python3
"""Build an EveRobot dataset variant with state-line probe prepended to action.

The source dataset is left unchanged. The output dataset is still LeRobot
compatible and keeps the EveRobot sidecar under ``meta/eve``. Only the action
column is changed from [D] to [D+1], where action[0] is the state-line
event-transition score and action[1:] is the original action vector.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.probe_state_line_distance import line_distances, load_state_span, mean_finite, robust_score


STATE_COLUMN = "observation.state"
SCHEMA_VERSION = "0.2"


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def dataset_id_from_path(path: Path) -> str:
    return path.expanduser().resolve().name


def action_dim_from_info(info: dict[str, Any]) -> int:
    shape = info.get("features", {}).get("action", {}).get("shape")
    if not shape:
        raise KeyError("meta/info.json is missing features.action.shape")
    return int(shape[0])


def copy_shell(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    for child in src.iterdir():
        target = dst / child.name
        if child.name == "data":
            target.mkdir(parents=True, exist_ok=True)
        elif child.name == "videos":
            os.symlink(os.path.abspath(child), target, target_is_directory=True)
        elif child.is_dir():
            shutil.copytree(child, target)
        elif child.is_file():
            shutil.copy2(child, target)


def fixed_size_float_array(values: np.ndarray, dim: int) -> pa.FixedSizeListArray:
    values = np.asarray(values, dtype=np.float32)
    flat = pa.array(values.reshape(-1), type=pa.float32())
    return pa.FixedSizeListArray.from_arrays(flat, dim)


def read_episode_state(path: Path) -> np.ndarray:
    table = pq.ParquetFile(path).read(columns=[STATE_COLUMN])
    return np.asarray(table[STATE_COLUMN].combine_chunks().to_pylist(), dtype=np.float32)


def write_episode_with_probe(src_path: Path, dst_path: Path, probe: np.ndarray) -> None:
    table = pq.ParquetFile(src_path).read()
    actions = np.asarray(table["action"].combine_chunks().to_pylist(), dtype=np.float32)
    probe = np.nan_to_num(probe, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)
    probe = np.clip(probe, 0.0, 1.0)
    if len(probe) != len(actions):
        raise ValueError(f"Probe/action length mismatch for {src_path}: {len(probe)} vs {len(actions)}")
    actions_with_probe = np.concatenate([probe[:, None], actions], axis=1).astype(np.float32)

    action_idx = table.schema.get_field_index("action")
    action_field = pa.field("action", pa.list_(pa.float32(), actions_with_probe.shape[1]))
    table = table.set_column(
        action_idx,
        action_field,
        fixed_size_float_array(actions_with_probe, actions_with_probe.shape[1]),
    )
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table.replace_schema_metadata(), dst_path)


def vector_stat(values: np.ndarray) -> dict[str, Any]:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        finite = np.asarray([0.0], dtype=np.float32)
    return {
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "q01": float(np.quantile(finite, 0.01)),
        "q10": float(np.quantile(finite, 0.10)),
        "q50": float(np.quantile(finite, 0.50)),
        "q90": float(np.quantile(finite, 0.90)),
        "q99": float(np.quantile(finite, 0.99)),
    }


def fixed_probe_stat() -> dict[str, Any]:
    return {
        "min": 0.0,
        "max": 1.0,
        "mean": 0.5,
        "std": 0.25,
        "q01": 0.0,
        "q10": 0.0,
        "q50": 0.5,
        "q90": 1.0,
        "q99": 1.0,
    }


def flatten_stat(value: Any) -> list[float]:
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        value = value[0]
    if not isinstance(value, list):
        raise TypeError(f"Expected list stat, got {type(value)}")
    return [float(item) for item in value]


def prepend_action_stats(action_stats: dict[str, Any], probe_stat: dict[str, Any], count: int) -> dict[str, Any]:
    out = {}
    for key, value in action_stats.items():
        if key == "count":
            out[key] = [int(count)]
            continue
        if key in probe_stat:
            out[key] = [float(probe_stat[key])] + flatten_stat(value)
        else:
            out[key] = value
    return out


def update_meta(
    *,
    src: Path,
    dst: Path,
    dataset_id: str,
    reference_stats_meta_dir: Path,
    probe_values: np.ndarray,
    probe_by_episode: dict[int, np.ndarray],
    calibration: dict[str, float],
    probe_stats_mode: str,
) -> None:
    info = read_json(dst / "meta" / "info.json")
    original_action_dim = action_dim_from_info(info)
    policy_action_dim = original_action_dim + 1
    info["features"]["action"]["shape"] = [policy_action_dim]
    write_json(dst / "meta" / "info.json", info)

    modality = read_json(dst / "meta" / "modality.json")
    shifted_action = {
        name: {"start": int(bounds["start"]) + 1, "end": int(bounds["end"]) + 1}
        for name, bounds in modality["action"].items()
    }
    modality["action"] = {"event_transition_score": {"start": 0, "end": 1}, **shifted_action}
    write_json(dst / "meta" / "modality.json", modality)

    reference_stats = read_json(reference_stats_meta_dir / "stats.json")
    actual_probe_stat = vector_stat(probe_values)
    norm_probe_stat = fixed_probe_stat() if probe_stats_mode == "fixed_0_1" else actual_probe_stat
    output_stats = dict(reference_stats)
    output_stats["action"] = prepend_action_stats(reference_stats["action"], norm_probe_stat, len(probe_values))
    write_json(dst / "meta" / "stats.json", output_stats)

    reference_relative = reference_stats_meta_dir / "relative_stats.json"
    if reference_relative.exists():
        relative_stats = read_json(reference_relative)
        if "action" in relative_stats:
            relative_stats["action"] = prepend_action_stats(
                relative_stats["action"],
                norm_probe_stat,
                len(probe_values),
            )
        write_json(dst / "meta" / "relative_stats.json", relative_stats)

    eps_stats_path = dst / "meta" / "episodes_stats.jsonl"
    if eps_stats_path.exists():
        rows = []
        src_rows = load_jsonl(src / "meta" / "episodes_stats.jsonl")
        for row in src_rows:
            ep = int(row["episode_index"])
            if "action" in row.get("stats", {}):
                row["stats"]["action"] = prepend_action_stats(
                    row["stats"]["action"],
                    vector_stat(probe_by_episode[ep]),
                    len(probe_by_episode[ep]),
                )
            rows.append(row)
        write_jsonl(eps_stats_path, rows)

    summary = {
        "source_dataset": str(src),
        "reference_stats_meta_dir": str(reference_stats_meta_dir),
        "probe_source": "observation.state",
        "probe_column": "action[0]",
        "original_action_columns": f"action[1:{policy_action_dim}]",
        "probe_stats_mode": probe_stats_mode,
        "actual_probe_stats": actual_probe_stat,
        "normalization_probe_stats": norm_probe_stat,
        "calibration": calibration,
        "policy_action_dim": policy_action_dim,
        "environment_action_dim": original_action_dim,
        "policy_action_prefix_dim": 1,
        "control_action_slice": [1, policy_action_dim],
        "normalization_note": (
            "The inserted state-line event_transition_score is normalized with its own prepended stats. "
            "The original action normalization stats are copied unchanged into action[1:]."
        ),
    }
    eve_root = dst / "meta" / "eve"
    write_json(eve_root / "state_line_probe_action_summary.json", summary)
    write_json(
        eve_root / "action_schema.json",
        {
            "format": "EveRobotActionSchema",
            "schema_version": SCHEMA_VERSION,
            "dataset_id": dataset_id,
            "policy_action_dim": policy_action_dim,
            "environment_action_dim": original_action_dim,
            "policy_action_prefix_dim": 1,
            "policy_action_prefix": [
                {
                    "name": "event_transition_score",
                    "source": "state_line_distance",
                    "start": 0,
                    "end": 1,
                    "normalization": "own_stats_in_meta_stats_action_0",
                }
            ],
            "control_action_slice": [1, policy_action_dim],
            "normalization": {
                "meta_dir": str(dst / "meta"),
                "stats_path": str(dst / "meta" / "stats.json"),
                "modality_path": str(dst / "meta" / "modality.json"),
                "note": "action[0] uses probe stats; action[1:] keeps the original action stats unchanged.",
            },
        },
    )


def rewrite_eve_paths(src: Path, dst: Path, dataset_id: str) -> None:
    eve_root = dst / "meta" / "eve"
    if not eve_root.exists():
        return

    old_dataset_ids: set[str] = set()
    old_schema = eve_root / "action_schema.json"
    if old_schema.exists():
        old_id = read_json(old_schema).get("dataset_id")
        if old_id:
            old_dataset_ids.add(str(old_id))
    manifests_dir = eve_root / "manifests"
    for path in manifests_dir.glob("*.json"):
        old_dataset_ids.update(str(key) for key in read_json(path).get("dataset_roots", {}).keys())
    if not old_dataset_ids:
        old_dataset_ids.add(src.name)

    def rewrite_identifier(value: Any) -> Any:
        if not isinstance(value, str):
            return value
        for old_id in sorted(old_dataset_ids, key=len, reverse=True):
            if value == old_id:
                return dataset_id
            if value.startswith(f"{old_id}_"):
                return f"{dataset_id}_{value[len(old_id) + 1:]}"
        return value

    def rewrite_row(row: dict[str, Any], *, is_event: bool) -> dict[str, Any]:
        row = dict(row)
        if "dataset_root" in row:
            row["dataset_root"] = str(dst)
        if is_event:
            row["dataset_id"] = dataset_id
            if "event_id" in row:
                row["event_id"] = rewrite_identifier(row["event_id"])
        return row

    episode_meta = eve_root / "episode_meta.jsonl"
    if episode_meta.exists():
        write_jsonl(episode_meta, [rewrite_row(row, is_event=False) for row in load_jsonl(episode_meta)])

    event_meta = eve_root / "event_meta.jsonl"
    if event_meta.exists():
        write_jsonl(event_meta, [rewrite_row(row, is_event=True) for row in load_jsonl(event_meta)])

    for path in manifests_dir.glob("*.json"):
        manifest = read_json(path)
        manifest["eve_root"] = str(eve_root)
        manifest["dataset_roots"] = {dataset_id: str(dst)}
        for sample in manifest.get("samples", []):
            sample["dataset_root"] = str(dst)
            sample["dataset_id"] = dataset_id
            for key in ("event_id", "sample_id", "paired_success_event_id"):
                if key in sample and sample[key] is not None:
                    sample[key] = rewrite_identifier(sample[key])
        write_json(path, manifest)

    for name in ["schema_version.json", "collection_iters.json"]:
        path = eve_root / name
        if path.exists():
            payload = read_json(path)
            payload["dataset_id"] = dataset_id
            write_json(path, payload)


def build(args: argparse.Namespace) -> None:
    src = args.source_dataset.expanduser().resolve()
    dst = args.output_dataset.expanduser().resolve()
    dataset_id = args.dataset_id or dataset_id_from_path(dst)
    reference_stats_meta_dir = (
        args.reference_stats_meta_dir.expanduser().resolve()
        if args.reference_stats_meta_dir is not None
        else src / "meta"
    )
    if dst.exists():
        if not args.overwrite:
            raise FileExistsError(dst)
        shutil.rmtree(dst)
    copy_shell(src, dst)

    state_start, state_end, state_modalities = load_state_span(src)
    parquet_paths = sorted((src / "data").glob("chunk-*/*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"No parquet files found under {src / 'data'}")

    states_by_path = {path: read_episode_state(path) for path in parquet_paths}
    all_state = np.concatenate(list(states_by_path.values()), axis=0)
    scale = all_state.std(axis=0).astype(np.float32)
    scale[scale < 1e-6] = 1.0

    distances = []
    distance_by_path = {}
    for path, state in states_by_path.items():
        per_dim_distance = line_distances(state, scale)
        distance = mean_finite(per_dim_distance[:, state_start:state_end], axis=1).astype(np.float32)
        distance_by_path[path] = distance
        distances.append(distance)

    score_all, calibration = robust_score(
        np.concatenate(distances),
        args.low_quantile,
        args.high_quantile,
    )

    cursor = 0
    probe_by_episode: dict[int, np.ndarray] = {}
    probe_values = []
    for path in parquet_paths:
        n = len(states_by_path[path])
        probe = score_all[cursor : cursor + n]
        cursor += n
        probe = np.nan_to_num(probe, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)
        probe = np.clip(probe, 0.0, 1.0)
        probe[:2] = 0.0

        rel = path.relative_to(src)
        write_episode_with_probe(path, dst / rel, probe)
        ep = int(path.stem.split("_")[-1])
        probe_by_episode[ep] = probe
        probe_values.append(probe)

    all_probe = np.concatenate(probe_values)
    update_meta(
        src=src,
        dst=dst,
        dataset_id=dataset_id,
        reference_stats_meta_dir=reference_stats_meta_dir,
        probe_values=all_probe,
        probe_by_episode=probe_by_episode,
        calibration=calibration,
        probe_stats_mode=args.probe_stats_mode,
    )
    rewrite_eve_paths(src, dst, dataset_id)

    output_info = read_json(dst / "meta" / "info.json")
    policy_action_dim = action_dim_from_info(output_info)
    environment_action_dim = policy_action_dim - 1

    write_json(
        dst / "meta" / "eve" / "state_line_probe_action_build.json",
        {
            "source_dataset": str(src),
            "output_dataset": str(dst),
            "dataset_id": dataset_id,
            "num_episodes": len(parquet_paths),
            "num_frames": int(len(all_probe)),
            "state_modalities": state_modalities,
            "policy_action_dim": policy_action_dim,
            "environment_action_dim": environment_action_dim,
        },
    )
    print(f"[state-line-probe-action] wrote {dst}")
    print(f"[state-line-probe-action] action_dim={policy_action_dim} frames={len(all_probe)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dataset", type=Path, required=True)
    parser.add_argument("--output-dataset", type=Path, required=True)
    parser.add_argument(
        "--reference-stats-meta-dir",
        type=Path,
        default=None,
        help="Meta dir whose original action stats should be preserved. Defaults to <source-dataset>/meta.",
    )
    parser.add_argument("--dataset-id", type=str, default=None)
    parser.add_argument("--low-quantile", type=float, default=0.10)
    parser.add_argument("--high-quantile", type=float, default=0.95)
    parser.add_argument("--probe-stats-mode", choices=["actual", "fixed_0_1"], default="actual")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    build(parse_args())


if __name__ == "__main__":
    main()
