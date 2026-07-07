#!/usr/bin/env python3
"""Build raw and failure-tail-trimmed LeRobot rollout datasets from shards."""

from __future__ import annotations

import argparse
import copy
import json
import shutil
from pathlib import Path
from typing import Any

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


FAILURE_PHRASE = "Failed to finish the whole process."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    merge = subparsers.add_parser("merge-shards")
    merge.add_argument("--shard-datasets", type=Path, nargs="+", required=True)
    merge.add_argument("--output-dataset", type=Path, required=True)
    merge.add_argument("--failure-phrase", type=str, default=FAILURE_PHRASE)
    merge.add_argument("--overwrite", action="store_true")

    trim = subparsers.add_parser("trim-failures")
    trim.add_argument("--source-dataset", type=Path, required=True)
    trim.add_argument("--output-dataset", type=Path, required=True)
    trim.add_argument("--trim-failure-seconds", type=float, default=8.0)
    trim.add_argument(
        "--trim-only-length",
        type=int,
        default=600,
        help="Only trim failed episodes with this original frame length. Use <=0 to trim all failures.",
    )
    trim.add_argument("--failure-phrase", type=str, default=FAILURE_PHRASE)
    trim.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def video_keys(info: dict[str, Any]) -> list[str]:
    return [key for key, spec in info["features"].items() if spec.get("dtype") == "video"]


def fixed_size_float_array(values: np.ndarray, dim: int) -> pa.FixedSizeListArray:
    values = np.asarray(values, dtype=np.float32)
    flat = pa.array(values.reshape(-1), type=pa.float32())
    return pa.FixedSizeListArray.from_arrays(flat, dim)


def table_to_numpy_dict(table: pa.Table) -> dict[str, np.ndarray]:
    def col_to_2d(name: str, dim: int) -> np.ndarray:
        col = table[name].combine_chunks()
        flat = col.flatten().to_numpy(zero_copy_only=False)
        return flat.reshape(-1, dim)

    return {
        "action": col_to_2d("action", 22),
        "observation.state": col_to_2d("observation.state", 23),
        "timestamp": table["timestamp"].combine_chunks().to_numpy(zero_copy_only=False),
        "frame_index": table["frame_index"].combine_chunks().to_numpy(zero_copy_only=False),
        "episode_index": table["episode_index"].combine_chunks().to_numpy(zero_copy_only=False),
        "index": table["index"].combine_chunks().to_numpy(zero_copy_only=False),
        "task_index": table["task_index"].combine_chunks().to_numpy(zero_copy_only=False),
    }


def write_episode_parquet(path: Path, data: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "action": fixed_size_float_array(data["action"], 22),
            "observation.state": fixed_size_float_array(data["observation.state"], 23),
            "timestamp": pa.array(data["timestamp"].astype(np.float32), type=pa.float32()),
            "frame_index": pa.array(data["frame_index"].astype(np.int64), type=pa.int64()),
            "episode_index": pa.array(data["episode_index"].astype(np.int64), type=pa.int64()),
            "index": pa.array(data["index"].astype(np.int64), type=pa.int64()),
            "task_index": pa.array(data["task_index"].astype(np.int64), type=pa.int64()),
        }
    )
    pq.write_table(table.replace_schema_metadata(), path)


def feature_stats(array: np.ndarray, keepdims: bool) -> dict[str, np.ndarray]:
    return {
        "min": np.min(array, axis=0, keepdims=keepdims),
        "max": np.max(array, axis=0, keepdims=keepdims),
        "mean": np.mean(array, axis=0, keepdims=keepdims),
        "std": np.std(array, axis=0, keepdims=keepdims),
        "count": np.array([len(array)]),
    }


def compute_episode_stats(data: dict[str, np.ndarray], features: dict[str, Any]) -> dict[str, Any]:
    stats = {}
    for key in ("action", "observation.state", "timestamp", "frame_index", "episode_index", "index", "task_index"):
        if key not in features:
            continue
        arr = data[key]
        stats[key] = feature_stats(arr, keepdims=arr.ndim == 1)
    return stats


def aggregate_feature_stats(items: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    counts = np.stack([item["count"] for item in items], axis=0).squeeze(-1)
    total = counts.sum(axis=0, keepdims=True)
    mean = sum(item["mean"] * item["count"] for item in items) / total
    var = sum((item["std"] ** 2 + (item["mean"] - mean) ** 2) * item["count"] for item in items) / total
    return {
        "min": np.min(np.stack([item["min"] for item in items], axis=0), axis=0, keepdims=True),
        "max": np.max(np.stack([item["max"] for item in items], axis=0), axis=0, keepdims=True),
        "mean": mean,
        "std": np.sqrt(var),
        "count": total,
    }


def aggregate_stats(stats_list: list[dict[str, Any]]) -> dict[str, Any]:
    keys = sorted({key for stats in stats_list for key in stats.keys()})
    return {key: aggregate_feature_stats([stats[key] for stats in stats_list if key in stats]) for key in keys}


def serialize(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: serialize(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def copy_video(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def read_video_frames(path: Path) -> list[np.ndarray]:
    frames = []
    container = av.open(str(path), mode="r")
    try:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            frames.append(frame.to_ndarray(format="rgb24"))
    finally:
        container.close()
    return frames


def save_video_frames(frames: list[np.ndarray], path: Path, fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not frames:
        raise ValueError(f"Cannot write empty video: {path}")
    height, width = frames[0].shape[:2]
    container = av.open(str(path), mode="w")
    try:
        stream = container.add_stream("libx264", rate=fps)
        stream.width = int(width)
        stream.height = int(height)
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "21"}
        for frame in frames:
            video_frame = av.VideoFrame.from_ndarray(np.asarray(frame, dtype=np.uint8), format="rgb24")
            for packet in stream.encode(video_frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()


def prepare_output(source_info: dict[str, Any], output_root: Path, overwrite: bool) -> dict[str, Any]:
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(output_root)
        shutil.rmtree(output_root)
    (output_root / "meta").mkdir(parents=True)
    (output_root / "data" / "chunk-000").mkdir(parents=True)
    for key in video_keys(source_info):
        (output_root / "videos" / "chunk-000" / key).mkdir(parents=True)
    return copy.deepcopy(source_info)


def finalize_dataset(
    output_root: Path,
    info: dict[str, Any],
    episodes: list[dict[str, Any]],
    episode_stats: list[dict[str, Any]],
    *,
    total_frames: int,
    extra_summary: dict[str, Any],
) -> None:
    info["total_episodes"] = len(episodes)
    info["total_frames"] = int(total_frames)
    info["total_videos"] = len(episodes) * len(video_keys(info))
    info["total_chunks"] = 1 if episodes else 0
    info["splits"] = {"train": f"0:{len(episodes)}"}
    for key in video_keys(info):
        if episodes:
            path = output_root / info["video_path"].format(episode_chunk=0, video_key=key, episode_index=0)
            try:
                container = av.open(str(path), mode="r")
                try:
                    stream = container.streams.video[0]
                    info["features"][key]["info"] = {
                        "video.fps": float(stream.average_rate),
                        "video.height": int(stream.height),
                        "video.width": int(stream.width),
                    }
                finally:
                    container.close()
            except Exception:
                pass
    write_json(output_root / "meta" / "info.json", info)
    write_jsonl(output_root / "meta" / "episodes.jsonl", episodes)
    write_jsonl(
        output_root / "meta" / "episodes_stats.jsonl",
        [{"episode_index": i, "stats": serialize(stats)} for i, stats in enumerate(episode_stats)],
    )
    write_json(output_root / "meta" / "stats.json", serialize(aggregate_stats(episode_stats)))
    write_json(output_root / "collection_summary.json", extra_summary)


def merge_shards(shard_datasets: list[Path], output_dataset: Path, overwrite: bool, failure_phrase: str) -> None:
    first_info = read_json(shard_datasets[0] / "meta" / "info.json")
    info = prepare_output(first_info, output_dataset, overwrite)
    shutil.copy2(shard_datasets[0] / "meta" / "tasks.jsonl", output_dataset / "meta" / "tasks.jsonl")
    modality = shard_datasets[0] / "meta" / "modality.json"
    if modality.exists():
        shutil.copy2(modality, output_dataset / "meta" / "modality.json")

    out_episodes = []
    out_stats = []
    global_index = 0
    new_ep_idx = 0
    attempt_log = []
    for shard_id, shard_root in enumerate(shard_datasets):
        shard_info = read_json(shard_root / "meta" / "info.json")
        for ep in load_jsonl(shard_root / "meta" / "episodes.jsonl"):
            old_ep_idx = int(ep["episode_index"])
            old_chunk = old_ep_idx // int(shard_info["chunks_size"])
            src_parquet = shard_root / shard_info["data_path"].format(
                episode_chunk=old_chunk,
                episode_index=old_ep_idx,
            )
            data = table_to_numpy_dict(pq.read_table(src_parquet))
            length = len(data["action"])
            data["frame_index"] = np.arange(length, dtype=np.int64)
            data["timestamp"] = (data["frame_index"] / float(info["fps"])).astype(np.float32)
            data["episode_index"] = np.full(length, new_ep_idx, dtype=np.int64)
            data["index"] = np.arange(global_index, global_index + length, dtype=np.int64)

            chunk = new_ep_idx // int(info["chunks_size"])
            dst_parquet = output_dataset / info["data_path"].format(episode_chunk=chunk, episode_index=new_ep_idx)
            write_episode_parquet(dst_parquet, data)
            for key in video_keys(info):
                src_video = shard_root / shard_info["video_path"].format(
                    episode_chunk=old_chunk,
                    video_key=key,
                    episode_index=old_ep_idx,
                )
                dst_video = output_dataset / info["video_path"].format(
                    episode_chunk=chunk,
                    video_key=key,
                    episode_index=new_ep_idx,
                )
                copy_video(src_video, dst_video)

            out_episodes.append({"episode_index": new_ep_idx, "tasks": ep["tasks"], "length": length})
            out_stats.append(compute_episode_stats(data, info["features"]))
            global_index += length
            new_ep_idx += 1

        summary_path = shard_root / "collection_summary.json"
        if summary_path.exists():
            summary = read_json(summary_path)
            for item in summary.get("attempt_log", []):
                item = dict(item)
                item["shard_id"] = shard_id
                attempt_log.append(item)

    finalize_dataset(
        output_dataset,
        info,
        out_episodes,
        out_stats,
        total_frames=global_index,
        extra_summary={
            "status": "complete",
            "mode": "raw_merged_save_all",
            "episodes": len(out_episodes),
            "failures": sum(1 for ep in out_episodes if any(failure_phrase in task for task in ep["tasks"])),
            "successes_saved": sum(1 for ep in out_episodes if not any(failure_phrase in task for task in ep["tasks"])),
            "shard_datasets": [str(path) for path in shard_datasets],
            "attempt_log": attempt_log,
        },
    )


def trim_failures(
    source_dataset: Path,
    output_dataset: Path,
    trim_seconds: float,
    trim_only_length: int,
    failure_phrase: str,
    overwrite: bool,
) -> None:
    source_info = read_json(source_dataset / "meta" / "info.json")
    info = prepare_output(source_info, output_dataset, overwrite)
    shutil.copy2(source_dataset / "meta" / "tasks.jsonl", output_dataset / "meta" / "tasks.jsonl")
    modality = source_dataset / "meta" / "modality.json"
    if modality.exists():
        shutil.copy2(modality, output_dataset / "meta" / "modality.json")

    fps = int(info["fps"])
    trim_steps = int(round(trim_seconds * fps))
    out_episodes = []
    out_stats = []
    trim_report = []
    global_index = 0
    for ep in load_jsonl(source_dataset / "meta" / "episodes.jsonl"):
        ep_idx = int(ep["episode_index"])
        chunk = ep_idx // int(source_info["chunks_size"])
        is_failure = any(failure_phrase in str(task) for task in ep["tasks"])
        data = table_to_numpy_dict(
            pq.read_table(
                source_dataset / source_info["data_path"].format(episode_chunk=chunk, episode_index=ep_idx)
            )
        )
        orig_len = len(data["action"])
        should_trim = is_failure and (trim_only_length <= 0 or orig_len == trim_only_length)
        keep = max(1, orig_len - trim_steps) if should_trim else orig_len
        trimmed = {key: value[:keep] for key, value in data.items()}
        trimmed["frame_index"] = np.arange(keep, dtype=np.int64)
        trimmed["timestamp"] = (trimmed["frame_index"] / float(fps)).astype(np.float32)
        trimmed["episode_index"] = np.full(keep, ep_idx, dtype=np.int64)
        trimmed["index"] = np.arange(global_index, global_index + keep, dtype=np.int64)

        dst_parquet = output_dataset / info["data_path"].format(episode_chunk=chunk, episode_index=ep_idx)
        write_episode_parquet(dst_parquet, trimmed)
        for key in video_keys(info):
            src_video = source_dataset / source_info["video_path"].format(
                episode_chunk=chunk,
                video_key=key,
                episode_index=ep_idx,
            )
            dst_video = output_dataset / info["video_path"].format(
                episode_chunk=chunk,
                video_key=key,
                episode_index=ep_idx,
            )
            if should_trim:
                save_video_frames(read_video_frames(src_video)[:keep], dst_video, fps)
            else:
                copy_video(src_video, dst_video)

        out_episodes.append({"episode_index": ep_idx, "tasks": ep["tasks"], "length": keep})
        out_stats.append(compute_episode_stats(trimmed, info["features"]))
        trim_report.append(
            {
                "episode_index": ep_idx,
                "failure": is_failure,
                "trimmed": should_trim,
                "original_length": orig_len,
                "trimmed_length": keep,
                "trimmed_tail_steps": orig_len - keep,
            }
        )
        global_index += keep

    finalize_dataset(
        output_dataset,
        info,
        out_episodes,
        out_stats,
        total_frames=global_index,
        extra_summary={
            "status": "complete",
            "mode": "trimmed_failures",
            "source_dataset": str(source_dataset),
            "trim_failure_seconds": trim_seconds,
            "trim_only_length": int(trim_only_length),
            "episodes": len(out_episodes),
            "failures": sum(1 for item in trim_report if item["failure"]),
            "successes_saved": sum(1 for item in trim_report if not item["failure"]),
            "trimmed_failures": sum(1 for item in trim_report if item["trimmed"]),
            "trim_report": trim_report,
        },
    )


def main() -> None:
    args = parse_args()
    if args.command == "merge-shards":
        merge_shards(
            [path.expanduser().resolve() for path in args.shard_datasets],
            args.output_dataset.expanduser().resolve(),
            args.overwrite,
            args.failure_phrase,
        )
    elif args.command == "trim-failures":
        trim_failures(
            args.source_dataset.expanduser().resolve(),
            args.output_dataset.expanduser().resolve(),
            args.trim_failure_seconds,
            args.trim_only_length,
            args.failure_phrase,
            args.overwrite,
        )


if __name__ == "__main__":
    main()
