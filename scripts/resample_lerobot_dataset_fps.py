#!/usr/bin/env python3
"""Resample a LeRobot dataset to a lower target FPS."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from fastwam.datasets.lerobot.lerobot.datasets.compute_stats import (
    aggregate_stats,
    compute_episode_stats,
)

DEFAULT_SOURCE_ROOT = Path("data/dexjoco_single")
DEFAULT_OUTPUT_ROOT = Path("data/dexjoco_single_fps5")
DEFAULT_TARGET_FPS = 5


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


def _stats_to_json(stats: dict[str, dict[str, np.ndarray]]) -> dict[str, dict[str, list[float]]]:
    return {key: {name: value.tolist() for name, value in ft_stats.items()} for key, ft_stats in stats.items()}


def _subsample_indices(num_rows: int, stride: int) -> list[int]:
    return list(range(0, num_rows, stride))


def _subsample_table(
    table: pa.Table,
    *,
    stride: int,
    target_fps: int,
    dest_episode_index: int,
    task_index: int,
    global_index_start: int,
) -> pa.Table:
    row_indices = _subsample_indices(table.num_rows, stride)
    sub = table.take(row_indices)
    num_rows = sub.num_rows

    columns: list[pa.Array] = []
    names: list[str] = []
    for name in sub.column_names:
        if name == "timestamp":
            values = [frame_idx / float(target_fps) for frame_idx in range(num_rows)]
            columns.append(pa.array(values, type=pa.float32()))
        elif name == "frame_index":
            columns.append(pa.array(list(range(num_rows)), type=sub[name].type))
        elif name == "episode_index":
            columns.append(pa.array([dest_episode_index] * num_rows, type=sub[name].type))
        elif name == "index":
            columns.append(
                pa.array(range(global_index_start, global_index_start + num_rows), type=sub[name].type)
            )
        elif name == "task_index":
            columns.append(pa.array([task_index] * num_rows, type=sub[name].type))
        else:
            columns.append(sub[name])
        names.append(name)

    return pa.Table.from_arrays(columns, names=names)


def _episode_stats_from_table(table: pa.Table, features: dict[str, Any]) -> dict[str, dict[str, np.ndarray]]:
    episode_data = {
        "action": np.asarray([table["action"][i].as_py() for i in range(table.num_rows)], dtype=np.float32),
        "observation.state": np.asarray(
            [table["observation.state"][i].as_py() for i in range(table.num_rows)],
            dtype=np.float32,
        ),
    }
    return compute_episode_stats(episode_data, features, is_compute_episode_stats_image=False)


def _ffmpeg_encoder(vcodec: str) -> str:
    if vcodec in {"libsvtav1", "svt_av1"}:
        return "libsvtav1"
    if vcodec in {"av1", "libaom-av1"}:
        return "libaom-av1"
    return vcodec


def _subsample_video(
    src_video: Path,
    dst_video: Path,
    *,
    stride: int,
    target_fps: int,
    vcodec: str,
    pix_fmt: str,
) -> int:
    dst_video.parent.mkdir(parents=True, exist_ok=True)
    encoder = _ffmpeg_encoder(vcodec)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(src_video),
        "-vf",
        f"select=not(mod(n\\,{stride}))",
        "-vsync",
        "vfr",
        "-r",
        str(target_fps),
        "-c:v",
        encoder,
        "-pix_fmt",
        pix_fmt,
    ]
    if encoder == "libaom-av1":
        cmd.extend(["-cpu-used", "8"])
    cmd.extend(["-crf", "30", str(dst_video)])

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed for {src_video}:\n{result.stderr[-2000:]}"
        )

    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "csv=p=0",
            str(dst_video),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return int(probe.stdout.strip())


def _global_index_start(output_root: Path, info: dict[str, Any], episode_index: int) -> int:
    total = 0
    chunks_size = int(info["chunks_size"])
    for ep_idx in range(episode_index):
        chunk = ep_idx // chunks_size
        data_path = output_root / "data" / f"chunk-{chunk:03d}" / f"episode_{ep_idx:06d}.parquet"
        if not data_path.exists():
            raise FileNotFoundError(
                f"Cannot compute global index for episode {episode_index}: missing {data_path}"
            )
        total += pq.read_table(data_path).num_rows
    return total


def _video_keys(features: dict[str, Any]) -> list[str]:
    return sorted(key for key, value in features.items() if value.get("dtype") == "video")


def _write_dataset_meta(
    *,
    output_root: Path,
    source_info: dict[str, Any],
    output_features: dict[str, Any],
    tasks_rows: list[dict[str, Any]],
    episodes_rows: list[dict[str, Any]],
    episodes_stats_rows: list[dict[str, Any]],
    target_fps: int,
) -> None:
    output_info = dict(source_info)
    output_info["fps"] = target_fps
    output_info["total_episodes"] = len(episodes_rows)
    output_info["total_frames"] = sum(int(row["length"]) for row in episodes_rows)
    output_info["total_videos"] = len(episodes_rows)
    output_info["total_chunks"] = math.ceil(len(episodes_rows) / int(source_info["chunks_size"]))
    output_info["splits"] = {"train": f"0:{len(episodes_rows)}"}
    output_info["features"] = output_features

    _write_json(output_root / "meta" / "info.json", output_info)
    _write_jsonl(output_root / "meta" / "tasks.jsonl", tasks_rows)
    _write_jsonl(output_root / "meta" / "episodes.jsonl", episodes_rows)
    _write_jsonl(
        output_root / "meta" / "episodes_stats.jsonl",
        [{"episode_index": row["episode_index"], "stats": _stats_to_json(row["stats"])} for row in episodes_stats_rows],
    )
    _write_json(
        output_root / "meta" / "stats.json",
        _stats_to_json(aggregate_stats([row["stats"] for row in episodes_stats_rows])),
    )


def finalize_dataset_meta(source_root: Path, output_root: Path, *, target_fps: int) -> None:
    info = _read_json(source_root / "meta" / "info.json")
    video_keys = _video_keys(info["features"])
    video_key = video_keys[0]
    video_feature = dict(info["features"][video_key])
    video_info = dict(video_feature.get("info", {}))
    video_info["video.fps"] = target_fps
    video_feature["info"] = video_info
    output_features = dict(info["features"])
    output_features[video_key] = video_feature

    tasks_rows = []
    with (source_root / "meta" / "tasks.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                tasks_rows.append(json.loads(line))

    with (source_root / "meta" / "episodes.jsonl").open("r", encoding="utf-8") as f:
        source_episodes = [json.loads(line) for line in f if line.strip()]

    episodes_rows: list[dict[str, Any]] = []
    episodes_stats_rows: list[dict[str, Any]] = []
    total_frames = 0

    for source_episode in source_episodes:
        episode_index = int(source_episode["episode_index"])
        chunk = episode_index // int(info["chunks_size"])
        data_path = output_root / "data" / f"chunk-{chunk:03d}" / f"episode_{episode_index:06d}.parquet"
        if not data_path.exists():
            raise FileNotFoundError(f"Missing resampled parquet: {data_path}")

        table = pq.read_table(data_path)
        episodes_rows.append(
            {
                "episode_index": episode_index,
                "tasks": source_episode.get("tasks", []),
                "length": table.num_rows,
            }
        )
        episodes_stats_rows.append(
            {
                "episode_index": episode_index,
                "stats": _episode_stats_from_table(table, output_features),
            }
        )
        total_frames += table.num_rows

    _write_dataset_meta(
        output_root=output_root,
        source_info=info,
        output_features=output_features,
        tasks_rows=tasks_rows,
        episodes_rows=episodes_rows,
        episodes_stats_rows=episodes_stats_rows,
        target_fps=target_fps,
    )
    print(f"Finalized meta for {len(episodes_rows)} episodes and {total_frames} frames at {output_root}")


def resample_dataset(
    source_root: Path,
    output_root: Path,
    *,
    target_fps: int,
    overwrite: bool,
    episode_start: int | None = None,
    episode_end: int | None = None,
) -> None:
    if not source_root.exists():
        raise FileNotFoundError(f"Missing source root: {source_root}")

    info = _read_json(source_root / "meta" / "info.json")
    source_fps = int(info["fps"])
    if source_fps % target_fps != 0:
        raise ValueError(
            f"source_fps ({source_fps}) must be evenly divisible by target_fps ({target_fps})"
        )
    stride = source_fps // target_fps

    if output_root.exists():
        if overwrite:
            shutil.rmtree(output_root)
            output_root.mkdir(parents=True, exist_ok=True)
        elif episode_start is None and episode_end is None:
            raise FileExistsError(f"Output already exists: {output_root}. Pass --overwrite to replace it.")
    else:
        output_root.mkdir(parents=True, exist_ok=True)

    video_keys = _video_keys(info["features"])
    if len(video_keys) != 1:
        raise ValueError(f"Expected exactly one video feature, found {video_keys}")
    video_key = video_keys[0]
    video_feature = dict(info["features"][video_key])
    video_info = dict(video_feature.get("info", {}))
    video_info["video.fps"] = target_fps
    video_feature["info"] = video_info

    output_features = dict(info["features"])
    output_features[video_key] = video_feature

    tasks_rows = []
    with (source_root / "meta" / "tasks.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                tasks_rows.append(json.loads(line))

    episodes_rows: list[dict[str, Any]] = []
    episodes_stats_rows: list[dict[str, Any]] = []
    total_frames = 0
    next_episode_index = 0

    with (source_root / "meta" / "episodes.jsonl").open("r", encoding="utf-8") as f:
        all_source_episodes = [json.loads(line) for line in f if line.strip()]

    source_episodes = all_source_episodes
    if episode_start is not None or episode_end is not None:
        start = 0 if episode_start is None else episode_start
        end = len(all_source_episodes) - 1 if episode_end is None else episode_end
        source_episodes = [
            episode
            for episode in all_source_episodes
            if start <= int(episode["episode_index"]) <= end
        ]

    for source_episode in source_episodes:
        source_episode_index = int(source_episode["episode_index"])
        dest_episode_index = source_episode_index
        dest_chunk = dest_episode_index // int(info["chunks_size"])
        source_chunk = source_episode_index // int(info["chunks_size"])
        global_index_start = (
            _global_index_start(output_root, info, dest_episode_index)
            if episode_start is not None or episode_end is not None
            else total_frames
        )

        source_data = source_root / info["data_path"].format(
            episode_chunk=source_chunk,
            episode_index=source_episode_index,
        )
        dest_data = output_root / "data" / f"chunk-{dest_chunk:03d}" / f"episode_{dest_episode_index:06d}.parquet"

        table = pq.read_table(source_data)
        transformed = _subsample_table(
            table,
            stride=stride,
            target_fps=target_fps,
            dest_episode_index=dest_episode_index,
            task_index=int(table["task_index"][0].as_py()),
            global_index_start=total_frames,
        )
        dest_data.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(transformed, dest_data)

        source_video = source_root / info["video_path"].format(
            episode_chunk=source_chunk,
            video_key=video_key,
            episode_index=source_episode_index,
        )
        dest_video = (
            output_root
            / "videos"
            / f"chunk-{dest_chunk:03d}"
            / video_key
            / f"episode_{dest_episode_index:06d}.mp4"
        )
        vcodec = video_info.get("video.codec", "libsvtav1")
        pix_fmt = video_info.get("video.pix_fmt", "yuv420p")
        video_num_frames = _subsample_video(
            source_video,
            dest_video,
            stride=stride,
            target_fps=target_fps,
            vcodec=vcodec,
            pix_fmt=pix_fmt,
        )
        if video_num_frames != transformed.num_rows:
            raise RuntimeError(
                f"Frame count mismatch for episode {source_episode_index}: "
                f"parquet={transformed.num_rows}, video={video_num_frames}"
            )

        episodes_rows.append(
            {
                "episode_index": dest_episode_index,
                "tasks": source_episode.get("tasks", []),
                "length": transformed.num_rows,
            }
        )
        episode_stats = _episode_stats_from_table(transformed, output_features)
        episodes_stats_rows.append({"episode_index": dest_episode_index, "stats": episode_stats})

        total_frames = max(total_frames, global_index_start + transformed.num_rows)
        next_episode_index = max(next_episode_index, dest_episode_index + 1)

        if dest_episode_index % 50 == 0 or dest_episode_index == int(source_episodes[-1]["episode_index"]):
            print(f"Resampled episode {dest_episode_index + 1}/{len(all_source_episodes)}")

    if episode_start is not None or episode_end is not None:
        finalize_dataset_meta(source_root, output_root, target_fps=target_fps)
        return

    _write_dataset_meta(
        output_root=output_root,
        source_info=info,
        output_features=output_features,
        tasks_rows=tasks_rows,
        episodes_rows=episodes_rows,
        episodes_stats_rows=episodes_stats_rows,
        target_fps=target_fps,
    )

    print(
        f"Wrote {next_episode_index} episodes and {total_frames} frames "
        f"to {output_root} at {target_fps} fps (stride={stride})"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=DEFAULT_SOURCE_ROOT,
        help="Source LeRobot dataset directory.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Destination LeRobot dataset directory.",
    )
    parser.add_argument(
        "--target-fps",
        type=int,
        default=DEFAULT_TARGET_FPS,
        help="Target dataset FPS (must evenly divide source FPS).",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output directory.")
    parser.add_argument(
        "--finalize-only",
        action="store_true",
        help="Only rebuild meta files from existing resampled parquet data.",
    )
    parser.add_argument("--episode-start", type=int, default=None, help="First episode index to resample.")
    parser.add_argument("--episode-end", type=int, default=None, help="Last episode index to resample.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.finalize_only:
        finalize_dataset_meta(args.source_root, args.output_root, target_fps=args.target_fps)
        return
    resample_dataset(
        source_root=args.source_root,
        output_root=args.output_root,
        target_fps=args.target_fps,
        overwrite=args.overwrite,
        episode_start=args.episode_start,
        episode_end=args.episode_end,
    )


if __name__ == "__main__":
    main()
