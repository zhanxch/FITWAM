#!/usr/bin/env python3
"""Merge Unihand G1 LeRobot datasets into one FastWAM-trainable dataset."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


DEFAULT_SOURCES = (
    ("g1u_box", "Put the dexterous hand on the table into the box, then close the box."),
    ("g1u_package", "Turn the package over and scan it with the barcode scanner."),
)
SOURCE_VIDEO_KEY = "observation.camera_0.rgb"
DEST_VIDEO_KEY = "observation.images.camera_0"
CHUNKS_SIZE = 1000


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _link_or_copy(src: Path, dst: Path, *, copy_videos: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    if copy_videos:
        shutil.copy2(src, dst)
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _combine_stats(stats_with_counts: list[tuple[dict[str, Any], int]]) -> dict[str, Any]:
    total = sum(count for _, count in stats_with_counts)
    combined: dict[str, Any] = {}

    for key in stats_with_counts[0][0]:
        combined[key] = {}
        for stat_name in ("mean", "std", "min", "max", "q01", "q99"):
            values = [
                np.asarray(stats[key][stat_name], dtype=np.float64)
                for stats, _ in stats_with_counts
            ]
            counts = np.asarray([count for _, count in stats_with_counts], dtype=np.float64)

            if stat_name == "mean":
                value = sum(v * c for v, c in zip(values, counts, strict=True)) / total
            elif stat_name == "std":
                means = [
                    np.asarray(stats[key]["mean"], dtype=np.float64)
                    for stats, _ in stats_with_counts
                ]
                global_mean = sum(m * c for m, c in zip(means, counts, strict=True)) / total
                variance = sum(
                    c * (np.square(s) + np.square(m - global_mean))
                    for s, m, c in zip(values, means, counts, strict=True)
                ) / total
                value = np.sqrt(np.maximum(variance, 0.0))
            elif stat_name in {"min", "q01"}:
                value = np.minimum.reduce(values)
            else:
                value = np.maximum.reduce(values)
            combined[key][stat_name] = value.astype(float).tolist()
        combined[key]["count"] = [total]

    if "task_index" in combined:
        task_mean = sum(i * count for i, (_, count) in enumerate(stats_with_counts)) / total
        combined["task_index"] = {
            "mean": [task_mean],
            "std": [
                math.sqrt(
                    sum(
                        count * (i - task_mean) ** 2
                        for i, (_, count) in enumerate(stats_with_counts)
                    )
                    / total
                )
            ],
            "min": [0.0],
            "max": [float(len(stats_with_counts) - 1)],
            "q01": [0.0],
            "q99": [float(len(stats_with_counts) - 1)],
            "count": [total],
        }

    return combined


def _feature_with_fastwam_video_key(info: dict[str, Any]) -> dict[str, Any]:
    features = dict(info["features"])
    video_feature = dict(features.pop(SOURCE_VIDEO_KEY))
    video_feature["names"] = ["height", "width", "rgb"]
    if "video_info" in video_feature and "info" not in video_feature:
        video_feature["info"] = video_feature.pop("video_info")
    features[DEST_VIDEO_KEY] = video_feature
    return features


def prepare_dataset(source_root: Path, output_root: Path, *, copy_videos: bool, overwrite: bool) -> None:
    sources = [(source_root / name, task) for name, task in DEFAULT_SOURCES]
    for src, _ in sources:
        if not src.exists():
            raise FileNotFoundError(f"Missing source dataset: {src}")

    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {output_root}. Pass --overwrite to replace it.")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    tasks_rows = [{"task_index": idx, "task": task} for idx, (_, task) in enumerate(sources)]
    episodes_rows: list[dict[str, Any]] = []
    stats_with_counts: list[tuple[dict[str, Any], int]] = []

    total_frames = 0
    next_episode_index = 0
    for task_index, (src_root, task) in enumerate(sources):
        info = _read_json(src_root / "meta" / "info.json")
        source_stats = _read_json(src_root / "meta" / "stats.json")
        stats_with_counts.append((source_stats, int(info["total_frames"])))

        with (src_root / "meta" / "episodes.jsonl").open("r", encoding="utf-8") as f:
            source_episodes = [json.loads(line) for line in f if line.strip()]

        for source_episode in source_episodes:
            source_episode_index = int(source_episode["episode_index"])
            dest_episode_index = next_episode_index
            dest_chunk = dest_episode_index // CHUNKS_SIZE
            source_chunk = source_episode_index // int(info["chunks_size"])

            source_data = src_root / info["data_path"].format(
                episode_chunk=source_chunk,
                episode_index=source_episode_index,
            )
            dest_data = output_root / "data" / f"chunk-{dest_chunk:03d}" / f"episode_{dest_episode_index:06d}.parquet"

            table = pq.read_table(source_data)
            columns = []
            for name in table.column_names:
                if name == "episode_index":
                    columns.append(pa.array([dest_episode_index] * table.num_rows, type=table[name].type))
                elif name == "index":
                    columns.append(pa.array(range(total_frames, total_frames + table.num_rows), type=table[name].type))
                elif name == "task_index":
                    columns.append(pa.array([task_index] * table.num_rows, type=table[name].type))
                else:
                    columns.append(table[name])
            dest_data.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(pa.Table.from_arrays(columns, names=table.column_names), dest_data)

            source_video = src_root / info["video_path"].format(
                episode_chunk=source_chunk,
                video_key=SOURCE_VIDEO_KEY,
                episode_index=source_episode_index,
            )
            dest_video = (
                output_root
                / "videos"
                / f"chunk-{dest_chunk:03d}"
                / DEST_VIDEO_KEY
                / f"episode_{dest_episode_index:06d}.mp4"
            )
            _link_or_copy(source_video, dest_video, copy_videos=copy_videos)

            episodes_rows.append(
                {
                    "episode_index": dest_episode_index,
                    "tasks": [task],
                    "length": int(source_episode["length"]),
                }
            )
            total_frames += table.num_rows
            next_episode_index += 1

    first_info = _read_json(sources[0][0] / "meta" / "info.json")
    info = {
        "codebase_version": first_info.get("codebase_version", "v2.0"),
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "fps": int(first_info["fps"]),
        "chunks_size": CHUNKS_SIZE,
        "total_episodes": next_episode_index,
        "total_frames": total_frames,
        "total_tasks": len(tasks_rows),
        "total_videos": next_episode_index,
        "total_chunks": math.ceil(next_episode_index / CHUNKS_SIZE),
        "splits": {"train": f"0:{next_episode_index}"},
        "features": _feature_with_fastwam_video_key(first_info),
        "robot_type": first_info.get("robot_type", "MyDexHand"),
    }

    _write_json(output_root / "meta" / "info.json", info)
    _write_json(output_root / "meta" / "stats.json", _combine_stats(stats_with_counts))
    _write_jsonl(output_root / "meta" / "tasks.jsonl", tasks_rows)
    _write_jsonl(output_root / "meta" / "episodes.jsonl", episodes_rows)

    print(f"Wrote {next_episode_index} episodes and {total_frames} frames to {output_root}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("data/Unihand_Preview/robot_data/g1_lerobot/G1U"),
        help="Directory containing g1u_box and g1u_package.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/g1"),
        help="Destination LeRobot dataset directory.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output directory.")
    parser.add_argument(
        "--copy-videos",
        action="store_true",
        help="Copy videos instead of hard-linking them when possible.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepare_dataset(
        source_root=args.source_root,
        output_root=args.output_root,
        copy_videos=args.copy_videos,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
