#!/usr/bin/env python3
"""Trim a LeRobot v2.1 dataset temporally and write a new dataset copy."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def get_feature_stats(array: np.ndarray, axis: tuple, keepdims: bool) -> dict[str, np.ndarray]:
    return {
        "min": np.min(array, axis=axis, keepdims=keepdims),
        "max": np.max(array, axis=axis, keepdims=keepdims),
        "mean": np.mean(array, axis=axis, keepdims=keepdims),
        "std": np.std(array, axis=axis, keepdims=keepdims),
        "count": np.array([len(array)]),
    }


def compute_episode_stats(
    episode_data: dict[str, np.ndarray],
    features: dict,
) -> dict:
    ep_stats = {}
    for key, data in episode_data.items():
        if key not in features:
            continue
        if features[key]["dtype"] in ("image", "video", "string"):
            continue
        ep_ft_array = data
        axes_to_reduce = 0
        keepdims = data.ndim == 1
        ep_stats[key] = get_feature_stats(ep_ft_array, axis=axes_to_reduce, keepdims=keepdims)
    return ep_stats


def aggregate_feature_stats(stats_ft_list: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    counts = np.stack([s["count"] for s in stats_ft_list], axis=0).squeeze(-1)
    total_count = counts.sum(axis=0, keepdims=True)
    weighted_mean = sum(s["mean"] * s["count"] for s in stats_ft_list) / total_count
    weighted_var = sum(
        (s["std"] ** 2 + (s["mean"] - weighted_mean) ** 2) * s["count"]
        for s in stats_ft_list
    ) / total_count
    return {
        "min": np.min(np.stack([s["min"] for s in stats_ft_list], axis=0), axis=0, keepdims=True),
        "max": np.max(np.stack([s["max"] for s in stats_ft_list], axis=0), axis=0, keepdims=True),
        "mean": weighted_mean,
        "std": np.sqrt(weighted_var),
        "count": total_count,
    }


def aggregate_stats(stats_list: list[dict[str, dict[str, np.ndarray]]]) -> dict[str, dict[str, np.ndarray]]:
    if not stats_list:
        return {}
    keys = stats_list[0].keys()
    return {key: aggregate_feature_stats([stats[key] for stats in stats_list]) for key in keys}


def serialize_dict(data: dict) -> dict:
    out = {}
    for key, value in data.items():
        if isinstance(value, dict):
            out[key] = serialize_dict(value)
        elif isinstance(value, np.ndarray):
            out[key] = value.tolist()
        else:
            out[key] = value
    return out


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def fixed_size_float_array(values: np.ndarray, dim: int) -> pa.FixedSizeListArray:
    values = np.asarray(values, dtype=np.float32)
    flat = pa.array(values.reshape(-1), type=pa.float32())
    return pa.FixedSizeListArray.from_arrays(flat, dim)


def write_episode_parquet(path: Path, episode_data: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "action": fixed_size_float_array(episode_data["action"], 22),
            "observation.state": fixed_size_float_array(episode_data["observation.state"], 23),
            "timestamp": pa.array(episode_data["timestamp"].astype(np.float32), type=pa.float32()),
            "frame_index": pa.array(episode_data["frame_index"].astype(np.int64), type=pa.int64()),
            "episode_index": pa.array(episode_data["episode_index"].astype(np.int64), type=pa.int64()),
            "index": pa.array(episode_data["index"].astype(np.int64), type=pa.int64()),
            "task_index": pa.array(episode_data["task_index"].astype(np.int64), type=pa.int64()),
        }
    )
    pq.write_table(table.replace_schema_metadata(), path)


def table_to_numpy_dict(table: pa.Table) -> dict[str, np.ndarray]:
    def col_to_2d(name: str, dim: int) -> np.ndarray:
        col = table[name].combine_chunks()
        if pa.types.is_fixed_size_list(col.type):
            flat = col.flatten().to_numpy(zero_copy_only=False)
            return flat.reshape(-1, dim)
        raise TypeError(f"Unexpected column type for {name}: {col.type}")

    return {
        "action": col_to_2d("action", 22),
        "observation.state": col_to_2d("observation.state", 23),
        "timestamp": table["timestamp"].combine_chunks().to_numpy(zero_copy_only=False),
        "frame_index": table["frame_index"].combine_chunks().to_numpy(zero_copy_only=False),
        "episode_index": table["episode_index"].combine_chunks().to_numpy(zero_copy_only=False),
        "index": table["index"].combine_chunks().to_numpy(zero_copy_only=False),
        "task_index": table["task_index"].combine_chunks().to_numpy(zero_copy_only=False),
    }


def compute_trim_range(
    length: int,
    fps: int,
    trim_start_s: float,
    trim_end_s_if_20s: float | None,
    full_episode_frames_for_tail_trim: int,
) -> tuple[int, int]:
    start = int(round(trim_start_s * fps))
    start = min(max(start, 0), max(length - 1, 0))
    end = length
    if length == full_episode_frames_for_tail_trim and trim_end_s_if_20s is not None:
        end = length - int(round(trim_end_s_if_20s * fps))
    end = max(end, start + 1)
    return start, end


def trim_video_ffmpeg(
    src: Path,
    dst: Path,
    *,
    fps: int,
    trim_start_s: float,
    keep_duration_s: float | None,
) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(src),
        "-ss",
        f"{trim_start_s:.6f}",
    ]
    if keep_duration_s is not None:
        cmd.extend(["-t", f"{keep_duration_s:.6f}"])
    cmd.extend(
        [
            "-r",
            str(fps),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "21",
            str(dst),
        ]
    )
    subprocess.run(cmd, check=True)


def trim_dataset(
    source_root: Path,
    output_root: Path,
    *,
    trim_start_s: float = 1.0,
    trim_end_s_if_20s: float = 8.0,
    full_episode_frames_for_tail_trim: int = 600,
    overwrite: bool,
) -> dict[str, Any]:
    if not source_root.exists():
        raise FileNotFoundError(source_root)
    if output_root.exists():
        if overwrite:
            shutil.rmtree(output_root)
        else:
            raise FileExistsError(f"Output already exists: {output_root}")

    info = read_json(source_root / "meta" / "info.json")
    fps = int(info["fps"])
    episodes = load_jsonl(source_root / "meta" / "episodes.jsonl")
    video_keys = [
        key
        for key, spec in info["features"].items()
        if spec.get("dtype") == "video"
    ]

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "meta").mkdir(parents=True, exist_ok=True)
    for rel in ("meta/modality.json", "meta/tasks.jsonl"):
        src = source_root / rel
        if src.exists():
            shutil.copy2(src, output_root / rel)

    new_episodes: list[dict[str, Any]] = []
    new_episode_stats: list[dict[str, Any]] = []
    stats_list = []
    global_index = 0
    trim_report = []

    for ep in episodes:
        ep_idx = int(ep["episode_index"])
        orig_len = int(ep["length"])
        start, end = compute_trim_range(
            orig_len,
            fps,
            trim_start_s,
            trim_end_s_if_20s,
            full_episode_frames_for_tail_trim,
        )
        new_len = end - start

        parquet_src = source_root / "data" / f"chunk-{ep_idx // info['chunks_size']:03d}" / f"episode_{ep_idx:06d}.parquet"
        table = pq.read_table(parquet_src)
        data = table_to_numpy_dict(table)
        trimmed = {k: v[start:end] for k, v in data.items()}
        trimmed["frame_index"] = np.arange(new_len, dtype=np.int64)
        trimmed["timestamp"] = (trimmed["frame_index"] / fps).astype(np.float32)
        trimmed["episode_index"] = np.full(new_len, ep_idx, dtype=np.int64)
        trimmed["index"] = np.arange(global_index, global_index + new_len, dtype=np.int64)
        trimmed["task_index"] = np.full(new_len, 0, dtype=np.int64)

        parquet_dst = output_root / parquet_src.relative_to(source_root)
        write_episode_parquet(parquet_dst, trimmed)

        for video_key in video_keys:
            rel_video = info["video_path"].format(
                episode_chunk=ep_idx // info["chunks_size"],
                video_key=video_key,
                episode_index=ep_idx,
            )
            src_video = source_root / rel_video
            dst_video = output_root / rel_video
            keep_duration_s = None
            if orig_len == full_episode_frames_for_tail_trim:
                keep_duration_s = (orig_len / fps) - trim_start_s - trim_end_s_if_20s
            trim_video_ffmpeg(
                src_video,
                dst_video,
                fps=fps,
                trim_start_s=trim_start_s,
                keep_duration_s=keep_duration_s,
            )

        new_episodes.append(
            {
                "episode_index": ep_idx,
                "tasks": ep["tasks"],
                "length": new_len,
            }
        )
        ep_stats = compute_episode_stats(trimmed, info["features"])
        stats_list.append(ep_stats)
        new_episode_stats.append(
            {
                "episode_index": ep_idx,
                "stats": serialize_dict(ep_stats),
            }
        )
        trim_report.append(
            {
                "episode_index": ep_idx,
                "orig_length": orig_len,
                "new_length": new_len,
                "trim_start_frame": start,
                "trim_end_frame": end,
                "tail_trim_applied": orig_len == full_episode_frames_for_tail_trim,
            }
        )
        global_index += new_len

    new_info = dict(info)
    new_info["total_frames"] = global_index
    new_info["total_episodes"] = len(new_episodes)
    new_info["total_videos"] = len(new_episodes) * len(video_keys)
    write_json(output_root / "meta" / "info.json", new_info)
    write_jsonl(output_root / "meta" / "episodes.jsonl", new_episodes)
    write_jsonl(output_root / "meta" / "episodes_stats.jsonl", new_episode_stats)
    write_json(output_root / "meta" / "stats.json", serialize_dict(aggregate_stats(stats_list)))

    summary = {
        "source_dataset": str(source_root.resolve()),
        "output_dataset": str(output_root.resolve()),
        "fps": fps,
        "trim_start_s": trim_start_s,
        "trim_end_s_if_20s": trim_end_s_if_20s,
        "full_episode_frames_for_tail_trim": full_episode_frames_for_tail_trim,
        "total_episodes": len(new_episodes),
        "total_frames_before": sum(int(ep["length"]) for ep in episodes),
        "total_frames_after": global_index,
        "episodes_with_tail_trim": sum(1 for row in trim_report if row["tail_trim_applied"]),
        "episodes": trim_report,
    }
    write_json(output_root / "trim_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--trim-start-s", type=float, default=1.0)
    parser.add_argument("--trim-end-s-if-20s", type=float, default=8.0)
    parser.add_argument("--full-episode-frames-for-tail-trim", type=int, default=600)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = trim_dataset(
        args.source_root.expanduser().resolve(),
        args.output_root.expanduser().resolve(),
        trim_start_s=args.trim_start_s,
        trim_end_s_if_20s=args.trim_end_s_if_20s,
        full_episode_frames_for_tail_trim=args.full_episode_frames_for_tail_trim,
        overwrite=args.overwrite,
    )
    print(
        f"[trim] wrote {summary['output_dataset']} "
        f"episodes={summary['total_episodes']} "
        f"frames {summary['total_frames_before']} -> {summary['total_frames_after']} "
        f"tail_trim_episodes={summary['episodes_with_tail_trim']}"
    )


if __name__ == "__main__":
    main()
