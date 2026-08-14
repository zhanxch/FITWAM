#!/usr/bin/env python3
"""Build a LeRobot-compatible EveRobot dataset from base and rollout rounds.

The output root is a normal LeRobot dataset with ``meta/``, ``data/`` and
``videos/``.  EveRobot metadata is stored as an additive sidecar under
``meta/eve/`` and records iteration order plus trainable event windows.
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

from scripts.build_rollout_datasets import (
    aggregate_stats,
    compute_episode_stats,
    load_jsonl,
    read_json,
    serialize,
    video_keys,
    write_json,
    write_jsonl,
)


SCHEMA_VERSION = "0.2"
FAILURE_PHRASE = "Failed to finish the whole process."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dataset", type=Path, required=True)
    parser.add_argument("--rollout-dataset", type=Path, default=None)
    parser.add_argument("--output-dataset", type=Path, required=True)
    parser.add_argument(
        "--dataset-id",
        type=str,
        default=None,
        help="EveRobot dataset id. Defaults to the output dataset directory name.",
    )
    parser.add_argument(
        "--base-dataset-id",
        type=str,
        default=None,
        help="Source id for iter0. Defaults to the base dataset directory name.",
    )
    parser.add_argument(
        "--rollout-dataset-id",
        type=str,
        default=None,
        help="Source id for the appended rollout dataset. Defaults to the rollout dataset directory name.",
    )
    parser.add_argument("--trimmed-reference-dataset", type=Path, default=None)
    parser.add_argument(
        "--task-name",
        type=str,
        default=None,
        help="Task name stored in meta/eve. Defaults to the base dataset name without a _fastwam suffix.",
    )
    parser.add_argument("--source-policy", type=str, default="fastwam_step6500")
    parser.add_argument("--source-checkpoint", type=str, default=None)
    parser.add_argument("--trim-failure-seconds", type=float, default=8.0)
    parser.add_argument("--trim-only-length", type=int, default=600)
    parser.add_argument("--failure-phrase", type=str, default=FAILURE_PHRASE)
    parser.add_argument("--base-iter", type=int, default=0)
    parser.add_argument("--rollout-iter", type=int, default=1)
    parser.add_argument("--manifest-name", type=str, default="train_all_iters_events")
    parser.add_argument("--failure-action-loss", choices=["enabled", "disabled"], default="disabled")
    parser.add_argument("--copy-mode", choices=["copy", "hardlink"], default="hardlink")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def first_task(ep: dict[str, Any]) -> str:
    tasks = ep.get("tasks", [])
    if isinstance(tasks, list) and tasks:
        return str(tasks[0])
    return str(ep.get("task", ""))


def strip_failure_phrase(task: str, failure_phrase: str) -> str:
    return " ".join(str(task).replace(failure_phrase, "").split()).strip()


def is_failure_task(task: str, failure_phrase: str) -> bool:
    return failure_phrase in str(task)


def dataset_id_from_path(path: Path) -> str:
    return path.expanduser().resolve().name


def task_name_from_dataset(path: Path) -> str:
    name = dataset_id_from_path(path)
    for suffix in ("_fastwam", "_EveRobot"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def action_dim_from_info(info: dict[str, Any]) -> int:
    shape = info.get("features", {}).get("action", {}).get("shape")
    if not shape:
        raise KeyError("meta/info.json is missing features.action.shape")
    return int(shape[0])


def feature_dim_from_info(info: dict[str, Any], key: str) -> int:
    shape = info.get("features", {}).get(key, {}).get("shape")
    if not shape:
        raise KeyError(f"meta/info.json is missing features.{key}.shape")
    return int(shape[0])


def fixed_size_float_array(values: np.ndarray, dim: int) -> pa.FixedSizeListArray:
    values = np.asarray(values, dtype=np.float32)
    flat = pa.array(values.reshape(-1), type=pa.float32())
    return pa.FixedSizeListArray.from_arrays(flat, dim)


def table_to_numpy_dict_for_info(table: pa.Table, info: dict[str, Any]) -> dict[str, np.ndarray]:
    def col_to_2d(name: str, dim: int) -> np.ndarray:
        col = table[name].combine_chunks()
        flat = col.flatten().to_numpy(zero_copy_only=False)
        return flat.reshape(-1, dim)

    return {
        "action": col_to_2d("action", action_dim_from_info(info)),
        "observation.state": col_to_2d("observation.state", feature_dim_from_info(info, "observation.state")),
        "timestamp": table["timestamp"].combine_chunks().to_numpy(zero_copy_only=False),
        "frame_index": table["frame_index"].combine_chunks().to_numpy(zero_copy_only=False),
        "episode_index": table["episode_index"].combine_chunks().to_numpy(zero_copy_only=False),
        "index": table["index"].combine_chunks().to_numpy(zero_copy_only=False),
        "task_index": table["task_index"].combine_chunks().to_numpy(zero_copy_only=False),
    }


def write_episode_parquet_for_info(path: Path, data: dict[str, np.ndarray], info: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "action": fixed_size_float_array(data["action"], action_dim_from_info(info)),
            "observation.state": fixed_size_float_array(
                data["observation.state"],
                feature_dim_from_info(info, "observation.state"),
            ),
            "timestamp": pa.array(data["timestamp"].astype(np.float32), type=pa.float32()),
            "frame_index": pa.array(data["frame_index"].astype(np.int64), type=pa.int64()),
            "episode_index": pa.array(data["episode_index"].astype(np.int64), type=pa.int64()),
            "index": pa.array(data["index"].astype(np.int64), type=pa.int64()),
            "task_index": pa.array(data["task_index"].astype(np.int64), type=pa.int64()),
        }
    )
    pq.write_table(table.replace_schema_metadata(), path)


def flatten_leading_singleton_lists(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: flatten_leading_singleton_lists(item) for key, item in value.items()}
    if (
        isinstance(value, list)
        and len(value) == 1
        and isinstance(value[0], list)
    ):
        return flatten_leading_singleton_lists(value[0])
    return value


def link_or_copy(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "hardlink":
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def copy_static_meta(datasets: list[Path], output: Path) -> dict[str, Any]:
    if not datasets:
        raise ValueError("At least one source dataset is required.")
    base = datasets[0]
    info = read_json(base / "meta" / "info.json")
    output.mkdir(parents=True, exist_ok=True)
    (output / "meta").mkdir(parents=True, exist_ok=True)
    (output / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    for key in video_keys(info):
        (output / "videos" / "chunk-000" / key).mkdir(parents=True, exist_ok=True)

    modality = base / "meta" / "modality.json"
    if modality.exists():
        shutil.copy2(modality, output / "meta" / "modality.json")

    tasks: list[str] = []
    for dataset in datasets:
        for row in load_jsonl(dataset / "meta" / "tasks.jsonl"):
            task = str(row["task"])
            if task not in tasks:
                tasks.append(task)
    write_jsonl(output / "meta" / "tasks.jsonl", [{"task_index": i, "task": task} for i, task in enumerate(tasks)])
    return info


def trim_report_by_episode(trimmed_reference: Path | None) -> dict[int, dict[str, Any]]:
    if trimmed_reference is None:
        return {}
    summary_path = trimmed_reference / "collection_summary.json"
    if not summary_path.exists():
        return {}
    summary = read_json(summary_path)
    return {int(row["episode_index"]): dict(row) for row in summary.get("trim_report", [])}


def event_end_frame(
    *,
    rollout_episode_index: int,
    length: int,
    fps: int,
    trim_seconds: float,
    trim_only_length: int,
    trim_report: dict[int, dict[str, Any]],
) -> tuple[int, str]:
    row = trim_report.get(int(rollout_episode_index))
    if row is not None and "trimmed_length" in row:
        return min(int(row["trimmed_length"]), int(length)), "trimmed_reference_window"
    if trim_only_length <= 0 or int(length) == int(trim_only_length):
        trim_frames = int(round(float(trim_seconds) * int(fps)))
        return max(1, int(length) - trim_frames), f"drop_last_{trim_seconds:g}s_window"
    return int(length), "full_short_failure_window"


def manifest_from_events(
    *,
    manifest_name: str,
    eve_root: Path,
    output_root: Path,
    events: list[dict[str, Any]],
    collection_iters: set[int] | None,
    dataset_id: str,
) -> dict[str, Any]:
    selected = [
        dict(event, sample_type="event", sample_id=event["event_id"], episode_outcome=event["event_outcome"], sample_stride=1)
        for event in events
        if collection_iters is None or int(event["collection_iter"]) in collection_iters
    ]
    return {
        "format": "EveRobotTrainManifest",
        "schema_version": SCHEMA_VERSION,
        "manifest_name": manifest_name,
        "eve_root": str(eve_root),
        "include_outcomes": sorted({str(sample["event_outcome"]) for sample in selected}),
        "failure_sample_mode": "event_only",
        "dataset_roots": {dataset_id: str(output_root)},
        "collection_iters": None if collection_iters is None else sorted(collection_iters),
        "num_samples": len(selected),
        "samples": selected,
    }


def write_action_schema(
    *,
    eve_root: Path,
    output_root: Path,
    info: dict[str, Any],
    dataset_id: str,
) -> None:
    action_dim = action_dim_from_info(info)
    write_json(
        eve_root / "action_schema.json",
        {
            "format": "EveRobotActionSchema",
            "schema_version": SCHEMA_VERSION,
            "dataset_id": dataset_id,
            "policy_action_dim": action_dim,
            "environment_action_dim": action_dim,
            "policy_action_prefix_dim": 0,
            "policy_action_prefix": [],
            "control_action_slice": [0, action_dim],
            "normalization": {
                "meta_dir": str(output_root / "meta"),
                "stats_path": str(output_root / "meta" / "stats.json"),
                "modality_path": str(output_root / "meta" / "modality.json"),
            },
        },
    )


def append_dataset(
    *,
    source_root: Path,
    output_root: Path,
    info: dict[str, Any],
    source_dataset_id: str,
    source_type: str,
    source_policy: str,
    source_checkpoint: str | None,
    collection_iter: int,
    start_episode_index: int,
    start_global_index: int,
    copy_mode: str,
    failure_phrase: str,
    task_name: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int, int]:
    source_info = read_json(source_root / "meta" / "info.json")
    out_episodes: list[dict[str, Any]] = []
    out_stats: list[dict[str, Any]] = []
    episode_meta: list[dict[str, Any]] = []
    global_index = int(start_global_index)
    next_episode_index = int(start_episode_index)
    task_to_index = {row["task"]: int(row["task_index"]) for row in load_jsonl(output_root / "meta" / "tasks.jsonl")}

    for ep in load_jsonl(source_root / "meta" / "episodes.jsonl"):
        old_ep_idx = int(ep["episode_index"])
        new_ep_idx = next_episode_index
        old_chunk = old_ep_idx // int(source_info["chunks_size"])
        new_chunk = new_ep_idx // int(info["chunks_size"])
        task_raw = first_task(ep)
        task_clean = strip_failure_phrase(task_raw, failure_phrase)
        outcome = "failure" if is_failure_task(task_raw, failure_phrase) else "success"

        src_parquet = source_root / source_info["data_path"].format(
            episode_chunk=old_chunk,
            episode_index=old_ep_idx,
        )
        data = table_to_numpy_dict_for_info(pq.read_table(src_parquet), info)
        length = int(len(data["action"]))
        data["frame_index"] = np.arange(length, dtype=np.int64)
        data["timestamp"] = (data["frame_index"] / float(info["fps"])).astype(np.float32)
        data["episode_index"] = np.full(length, new_ep_idx, dtype=np.int64)
        data["index"] = np.arange(global_index, global_index + length, dtype=np.int64)
        data["task_index"] = np.full(length, task_to_index[task_raw], dtype=np.int64)

        dst_parquet = output_root / info["data_path"].format(episode_chunk=new_chunk, episode_index=new_ep_idx)
        write_episode_parquet_for_info(dst_parquet, data, info)
        for key in video_keys(info):
            src_video = source_root / source_info["video_path"].format(
                episode_chunk=old_chunk,
                video_key=key,
                episode_index=old_ep_idx,
            )
            dst_video = output_root / info["video_path"].format(
                episode_chunk=new_chunk,
                video_key=key,
                episode_index=new_ep_idx,
            )
            link_or_copy(src_video, dst_video, copy_mode)

        out_episodes.append({"episode_index": new_ep_idx, "tasks": [task_raw], "length": length})
        out_stats.append(compute_episode_stats(data, info["features"]))
        episode_meta.append(
            {
                "schema_version": SCHEMA_VERSION,
                "dataset_id": source_dataset_id,
                "dataset_root": str(output_root),
                "episode_index": new_ep_idx,
                "source_episode_index": old_ep_idx,
                "task_name": task_name,
                "task": task_raw,
                "task_clean": task_clean,
                "source_type": source_type,
                "source_policy": source_policy,
                "source_checkpoint": source_checkpoint,
                "collection_iter": int(collection_iter),
                "collection_round": int(collection_iter),
                "episode_outcome": outcome,
                "failure_type": None if outcome == "success" else "unknown_failure",
                "length": length,
                "fps": int(info["fps"]),
                "split": "train",
            }
        )
        global_index += length
        next_episode_index += 1

    return out_episodes, out_stats, episode_meta, next_episode_index, global_index


def build(args: argparse.Namespace) -> None:
    base = args.base_dataset.expanduser().resolve()
    rollout = args.rollout_dataset.expanduser().resolve() if args.rollout_dataset is not None else None
    output = args.output_dataset.expanduser().resolve()
    trimmed_reference = args.trimmed_reference_dataset.expanduser().resolve() if args.trimmed_reference_dataset else None
    dataset_id = args.dataset_id or dataset_id_from_path(output)
    base_dataset_id = args.base_dataset_id or dataset_id_from_path(base)
    rollout_dataset_id = args.rollout_dataset_id or (dataset_id_from_path(rollout) if rollout is not None else None)
    task_name = args.task_name or task_name_from_dataset(base)

    if output.exists():
        if not args.overwrite:
            raise FileExistsError(output)
        shutil.rmtree(output)

    info = copy_static_meta([base] + ([] if rollout is None else [rollout]), output)
    all_episodes: list[dict[str, Any]] = []
    all_stats: list[dict[str, Any]] = []
    all_episode_meta: list[dict[str, Any]] = []

    base_eps, base_stats, base_meta, next_ep, global_index = append_dataset(
        source_root=base,
        output_root=output,
        info=info,
        source_dataset_id=base_dataset_id,
        source_type="expert_success",
        source_policy="human_or_expert",
        source_checkpoint=None,
        collection_iter=args.base_iter,
        start_episode_index=0,
        start_global_index=0,
        copy_mode=args.copy_mode,
        failure_phrase=args.failure_phrase,
        task_name=task_name,
    )
    all_episodes.extend(base_eps)
    all_stats.extend(base_stats)
    all_episode_meta.extend(base_meta)

    if rollout is not None:
        rollout_eps, rollout_stats, rollout_meta, next_ep, global_index = append_dataset(
            source_root=rollout,
            output_root=output,
            info=info,
            source_dataset_id=str(rollout_dataset_id),
            source_type="policy_rollout",
            source_policy=args.source_policy,
            source_checkpoint=args.source_checkpoint,
            collection_iter=args.rollout_iter,
            start_episode_index=next_ep,
            start_global_index=global_index,
            copy_mode=args.copy_mode,
            failure_phrase=args.failure_phrase,
            task_name=task_name,
        )
        all_episodes.extend(rollout_eps)
        all_stats.extend(rollout_stats)
        all_episode_meta.extend(rollout_meta)
    else:
        rollout_eps = []

    info["total_episodes"] = len(all_episodes)
    info["total_frames"] = int(global_index)
    info["total_videos"] = len(all_episodes) * len(video_keys(info))
    info["total_chunks"] = 1
    info["splits"] = {"train": f"0:{len(all_episodes)}"}
    write_json(output / "meta" / "info.json", info)
    write_jsonl(output / "meta" / "episodes.jsonl", all_episodes)
    write_jsonl(
        output / "meta" / "episodes_stats.jsonl",
        [{"episode_index": i, "stats": serialize(stats)} for i, stats in enumerate(all_stats)],
    )
    source_stats_path = base / "meta" / "stats.json"
    if rollout is None and source_stats_path.exists():
        output_stats = read_json(source_stats_path)
    else:
        output_stats = flatten_leading_singleton_lists(serialize(aggregate_stats(all_stats)))
    write_json(output / "meta" / "stats.json", output_stats)

    eve_root = output / "meta" / "eve"
    write_json(
        eve_root / "schema_version.json",
        {
            "format": "EveRobot",
            "schema_version": SCHEMA_VERSION,
            "compatible_base_format": "LeRobot",
            "layout": "LeRobot dataset root with additive meta/eve/ sidecar",
            "dataset_id": dataset_id,
            "iteration_semantics": {
                "collection_iter": "0 is the initial dataset; larger values are appended rollout/data-collection rounds.",
                "iter0": "base expert/success dataset copied from --base-dataset",
                "iter1": "first appended rollout dataset copied from --rollout-dataset when provided",
            },
        },
    )
    write_action_schema(eve_root=eve_root, output_root=output, info=info, dataset_id=dataset_id)
    iter_rows = [
        {
            "collection_iter": int(args.base_iter),
            "role": "iter0",
            "source_dataset": str(base),
            "source_dataset_id": base_dataset_id,
            "source_type": "expert_success",
            "num_episodes": len(base_eps),
        }
    ]
    if rollout is not None:
        iter_rows.append(
            {
                "collection_iter": int(args.rollout_iter),
                "role": f"iter{int(args.rollout_iter)}",
                "source_dataset": str(rollout),
                "source_dataset_id": rollout_dataset_id,
                "source_type": "policy_rollout",
                "source_policy": args.source_policy,
                "source_checkpoint": args.source_checkpoint,
                "num_episodes": len(rollout_eps),
            }
        )
    write_json(
        eve_root / "collection_iters.json",
        {
            "format": "EveRobotCollectionIters",
            "schema_version": SCHEMA_VERSION,
            "dataset_id": dataset_id,
            "iters": iter_rows,
        },
    )
    write_jsonl(eve_root / "episode_meta.jsonl", all_episode_meta)

    trim_report = trim_report_by_episode(trimmed_reference)
    fps = int(info["fps"])
    events: list[dict[str, Any]] = []
    for row in all_episode_meta:
        if row["episode_outcome"] == "failure":
            end_frame, rule = event_end_frame(
                rollout_episode_index=int(row["source_episode_index"]),
                length=int(row["length"]),
                fps=fps,
                trim_seconds=args.trim_failure_seconds,
                trim_only_length=args.trim_only_length,
                trim_report=trim_report,
            )
            event_type = "failure_event"
            event_outcome = "failure"
            action_loss = args.failure_action_loss
            sample_role = "failure_context"
            failure_frame = int(row["length"]) - 1
        else:
            end_frame = int(row["length"])
            rule = "full_success_episode"
            event_type = "success_event"
            event_outcome = "success"
            action_loss = "enabled"
            sample_role = "success_full_event"
            failure_frame = None

        event_id = f"{dataset_id}_iter{int(row['collection_iter'])}_ep{int(row['episode_index']):06d}_{event_type}"
        events.append(
            {
                "schema_version": SCHEMA_VERSION,
                "event_id": event_id,
                "dataset_id": dataset_id,
                "dataset_root": str(output),
                "episode_index": int(row["episode_index"]),
                "source_dataset_id": row["dataset_id"],
                "source_episode_index": int(row["source_episode_index"]),
                "source_type": row["source_type"],
                "task_name": task_name,
                "task": row["task"],
                "task_clean": row["task_clean"],
                "event_type": event_type,
                "event_outcome": event_outcome,
                "failure_type": row["failure_type"],
                "source_policy": row["source_policy"],
                "source_checkpoint": row["source_checkpoint"],
                "collection_iter": int(row["collection_iter"]),
                "collection_round": int(row["collection_iter"]),
                "start_frame": 0,
                "end_frame": int(end_frame),
                "failure_frame": failure_frame,
                "source_window_rule": rule,
                "action_loss": action_loss,
                "sample_role": sample_role,
                "paired_success_event_id": None,
                "paired_trimmed_dataset": str(trimmed_reference) if event_outcome == "failure" and trimmed_reference else None,
                "split": "train",
            }
        )
    write_jsonl(eve_root / "event_meta.jsonl", events)

    manifest_specs = [
        (args.manifest_name, None),
        ("train_iter0_events", {int(args.base_iter)}),
    ]
    if rollout is not None:
        manifest_specs.extend(
            [
                ("train_iter1_events", {int(args.rollout_iter)}),
                ("train_iter0_iter1_events", {int(args.base_iter), int(args.rollout_iter)}),
            ]
        )
    seen_manifest_names = set()
    for manifest_name, collection_iters in manifest_specs:
        if manifest_name in seen_manifest_names:
            continue
        seen_manifest_names.add(manifest_name)
        manifest = manifest_from_events(
            manifest_name=manifest_name,
            eve_root=eve_root,
            output_root=output,
            events=events,
            collection_iters=collection_iters,
            dataset_id=dataset_id,
        )
        write_json(eve_root / "manifests" / f"{manifest_name}.json", manifest)
    write_json(
        output / "collection_summary.json",
        {
            "status": "complete",
            "mode": "lerobot_plus_eve_sidecar",
            "dataset_id": dataset_id,
            "base_dataset": str(base),
            "rollout_dataset": str(rollout) if rollout is not None else None,
            "trimmed_reference_dataset": str(trimmed_reference) if trimmed_reference else None,
            "base_iter": int(args.base_iter),
            "rollout_iter": int(args.rollout_iter),
            "task_name": task_name,
            "episodes": len(all_episodes),
            "base_episodes": len(base_eps),
            "rollout_episodes": len(rollout_eps),
            "successes": sum(1 for row in all_episode_meta if row["episode_outcome"] == "success"),
            "failures": sum(1 for row in all_episode_meta if row["episode_outcome"] == "failure"),
            "events": len(events),
            "success_events": sum(1 for event in events if event["event_outcome"] == "success"),
            "failure_events": sum(1 for event in events if event["event_outcome"] == "failure"),
            "manifest": str(eve_root / "manifests" / f"{args.manifest_name}.json"),
        },
    )
    print(f"[eve-lerobot] wrote {output}")
    print(
        f"[eve-lerobot] episodes={len(all_episodes)} events={len(events)} "
        f"failure_events={sum(1 for event in events if event['event_outcome'] == 'failure')}"
    )


def main() -> None:
    build(parse_args())


if __name__ == "__main__":
    main()
