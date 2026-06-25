#!/usr/bin/env python3
"""Split a DexJoCo LeRobot dataset into subtask segments using dual-hand annotations."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if SRC_ROOT.exists() and str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fastwam.datasets.lerobot.lerobot.datasets.compute_stats import (  # noqa: E402
    aggregate_stats,
    compute_episode_stats,
)

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from dexjoco_subtask_annotator import (  # noqa: E402
    ANNOTATION_FILENAME,
    DEFAULT_TASK,
    LEFT_SUBTASKS,
    RIGHT_SUBTASKS,
)

DEFAULT_SOURCE = Path("data/dexjoco_microwave_cook")
DEFAULT_OUTPUT = Path("data/dexjoco_microwave_cook_subtasks")
CHUNKS_SIZE = 1000


@dataclass(frozen=True)
class VideoTrimJob:
    src_video: str
    dst_video: str
    start_frame: int
    end_frame: int
    fps: int
    vcodec: str
    pix_fmt: str
    crf: int
    preset: str


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


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _stats_to_json(stats: dict[str, dict[str, np.ndarray]]) -> dict[str, dict[str, list[float]]]:
    return {key: {name: value.tolist() for name, value in ft_stats.items()} for key, ft_stats in stats.items()}


def _list_video_keys(features: dict[str, Any]) -> list[str]:
    return sorted(key for key, value in features.items() if value.get("dtype") == "video")


def _ffmpeg_encoder(vcodec: str) -> str:
    if vcodec in {"libsvtav1", "svt_av1"}:
        return "libsvtav1"
    if vcodec in {"av1", "libaom-av1"}:
        return "libaom-av1"
    return vcodec


def _combined_task(left_subtask: str, right_subtask: str) -> str:
    return f"Left arm: {left_subtask}. Right arm: {right_subtask}."


def _build_task_catalog(coarse_task: str) -> tuple[list[dict[str, Any]], dict[tuple[str, str], int]]:
    tasks_rows: list[dict[str, Any]] = [{"task_index": 0, "task": coarse_task}]
    pair_to_index: dict[tuple[str, str], int] = {}
    next_index = 1
    for left in LEFT_SUBTASKS:
        for right in RIGHT_SUBTASKS:
            pair = (left, right)
            if pair in pair_to_index:
                continue
            pair_to_index[pair] = next_index
            tasks_rows.append(
                {
                    "task_index": next_index,
                    "task": _combined_task(left, right),
                }
            )
            next_index += 1
    return tasks_rows, pair_to_index


def _build_output_features(source_features: dict[str, Any]) -> dict[str, Any]:
    features = dict(source_features)
    for key in ("left_subtask_index", "right_subtask_index", "coarse_task_index"):
        features[key] = {"dtype": "int64", "shape": [1], "names": None}
    return features


def _slice_table(
    table: pa.Table,
    start_frame: int,
    end_frame: int,
    *,
    dest_episode_index: int,
    task_index: int,
    left_subtask_index: int,
    right_subtask_index: int,
    coarse_task_index: int,
    global_index_start: int,
    fps: int,
) -> pa.Table:
    length = end_frame - start_frame + 1
    sub = table.slice(start_frame, length)
    num_rows = sub.num_rows

    columns: list[pa.Array] = []
    names: list[str] = []
    for name in sub.column_names:
        if name == "timestamp":
            values = [frame_idx / float(fps) for frame_idx in range(num_rows)]
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

    columns.extend(
        [
            pa.array([left_subtask_index] * num_rows, type=pa.int64()),
            pa.array([right_subtask_index] * num_rows, type=pa.int64()),
            pa.array([coarse_task_index] * num_rows, type=pa.int64()),
        ]
    )
    names.extend(["left_subtask_index", "right_subtask_index", "coarse_task_index"])
    return pa.Table.from_arrays(columns, names=names)


def _trim_video(job: VideoTrimJob) -> tuple[str, str | None]:
    src_video = Path(job.src_video)
    dst_video = Path(job.dst_video)
    dst_video.parent.mkdir(parents=True, exist_ok=True)
    encoder = _ffmpeg_encoder(job.vcodec)
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(src_video),
        "-vf",
        f"select='between(n\\,{job.start_frame}\\,{job.end_frame})',setpts=N/{job.fps}/TB",
        "-vsync",
        "vfr",
        "-r",
        str(job.fps),
        "-c:v",
        encoder,
        "-pix_fmt",
        job.pix_fmt,
    ]
    if encoder == "libx264":
        cmd.extend(["-preset", job.preset, "-crf", str(job.crf)])
    elif encoder == "libaom-av1":
        cmd.extend(["-cpu-used", "8", "-crf", str(job.crf)])
    else:
        cmd.extend(["-crf", str(job.crf)])
    cmd.append(str(dst_video))

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return str(dst_video), result.stderr[-1000:]
    return str(dst_video), None


def _run_video_jobs(jobs: list[VideoTrimJob], *, workers: int) -> None:
    if not jobs:
        return
    workers = max(1, min(workers, len(jobs)))
    print(f"Trimming {len(jobs)} videos with {workers} workers...")
    failures: list[str] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_trim_video, job) for job in jobs]
        done = 0
        for future in as_completed(futures):
            dst, err = future.result()
            done += 1
            if err is not None:
                failures.append(f"{dst}: {err}")
            if done % 100 == 0 or done == len(jobs):
                print(f"  trimmed {done}/{len(jobs)}")
    if failures:
        raise RuntimeError("Video trimming failed for some files:\n" + "\n".join(failures[:5]))


def _apply_output_video_codec(features: dict[str, Any], *, vcodec: str) -> dict[str, Any]:
    updated = dict(features)
    for key, feature in updated.items():
        if feature.get("dtype") != "video":
            continue
        new_feature = dict(feature)
        video_info = dict(new_feature.get("info", {}))
        video_info["video.codec"] = vcodec
        new_feature["info"] = video_info
        updated[key] = new_feature
    return updated


def _episode_stats_from_table(table: pa.Table, features: dict[str, Any]) -> dict[str, dict[str, np.ndarray]]:
    episode_data = {
        "action": np.asarray([table["action"][i].as_py() for i in range(table.num_rows)], dtype=np.float32),
        "observation.state": np.asarray(
            [table["observation.state"][i].as_py() for i in range(table.num_rows)],
            dtype=np.float32,
        ),
    }
    return compute_episode_stats(episode_data, features, is_compute_episode_stats_image=False)


def prepare_subtask_dataset(
    source_root: Path,
    output_root: Path,
    *,
    annotation_path: Path | None = None,
    overwrite: bool = False,
    skip_videos: bool = False,
    video_workers: int = 8,
    output_vcodec: str = "libx264",
    video_crf: int = 23,
    video_preset: str = "veryfast",
) -> None:
    source_root = source_root.resolve()
    output_root = output_root.resolve()
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output already exists: {output_root}. Pass --overwrite to replace it."
            )
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    info = _read_json(source_root / "meta" / "info.json")
    fps = int(info.get("fps", 30))
    chunks_size = int(info.get("chunks_size", CHUNKS_SIZE))
    video_keys = _list_video_keys(info["features"])
    if not video_keys:
        raise ValueError("Source dataset has no video features.")

    annotation_path = annotation_path or (source_root / "annotations" / ANNOTATION_FILENAME)
    annotations = _read_jsonl(annotation_path)
    if not annotations:
        raise FileNotFoundError(f"No annotations found at {annotation_path}")

    source_episodes = _read_jsonl(source_root / "meta" / "episodes.jsonl")
    annotation_by_episode = {int(row["episode_index"]): row for row in annotations}

    coarse_task = annotations[0].get("task", DEFAULT_TASK)
    tasks_rows, pair_to_index = _build_task_catalog(coarse_task)
    left_to_index = {label: idx for idx, label in enumerate(LEFT_SUBTASKS)}
    right_to_index = {label: idx for idx, label in enumerate(RIGHT_SUBTASKS)}
    output_features = _build_output_features(info["features"])
    pix_fmt = "yuv420p"

    episodes_rows: list[dict[str, Any]] = []
    episodes_stats_rows: list[dict[str, Any]] = []
    video_jobs: list[VideoTrimJob] = []
    total_frames = 0
    dest_episode_index = 0

    for source_episode in source_episodes:
        source_episode_index = int(source_episode["episode_index"])
        annotation = annotation_by_episode.get(source_episode_index)
        if annotation is None:
            print(f"Skipping episode {source_episode_index}: missing annotation")
            continue

        source_chunk = source_episode_index // chunks_size
        source_parquet = source_root / info["data_path"].format(
            episode_chunk=source_chunk,
            episode_index=source_episode_index,
        )
        table = pq.read_table(source_parquet)

        for segment_index, segment in enumerate(annotation.get("segments", [])):
            start_frame = int(segment["start_frame"])
            end_frame = int(segment["end_frame"])
            if end_frame < start_frame:
                continue

            left_subtask = segment.get("left_subtask", "")
            right_subtask = segment.get("right_subtask", "")
            if not left_subtask or not right_subtask:
                print(
                    f"Skipping episode {source_episode_index} segment {segment_index}: "
                    "missing subtask labels"
                )
                continue

            task_index = pair_to_index[(left_subtask, right_subtask)]
            combined_task = _combined_task(left_subtask, right_subtask)
            dest_chunk = dest_episode_index // CHUNKS_SIZE

            transformed = _slice_table(
                table,
                start_frame,
                end_frame,
                dest_episode_index=dest_episode_index,
                task_index=task_index,
                left_subtask_index=left_to_index[left_subtask],
                right_subtask_index=right_to_index[right_subtask],
                coarse_task_index=0,
                global_index_start=total_frames,
                fps=fps,
            )

            dest_parquet = (
                output_root
                / "data"
                / f"chunk-{dest_chunk:03d}"
                / f"episode_{dest_episode_index:06d}.parquet"
            )
            dest_parquet.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(transformed, dest_parquet)

            if not skip_videos:
                for video_key in video_keys:
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
                    video_jobs.append(
                        VideoTrimJob(
                            src_video=str(source_video),
                            dst_video=str(dest_video),
                            start_frame=start_frame,
                            end_frame=end_frame,
                            fps=fps,
                            vcodec=output_vcodec,
                            pix_fmt=pix_fmt,
                            crf=video_crf,
                            preset=video_preset,
                        )
                    )

            episodes_rows.append(
                {
                    "episode_index": dest_episode_index,
                    "tasks": [combined_task],
                    "length": transformed.num_rows,
                    "source_episode_index": source_episode_index,
                    "source_segment_index": segment_index,
                    "left_subtask": left_subtask,
                    "right_subtask": right_subtask,
                    "source_start_frame": start_frame,
                    "source_end_frame": end_frame,
                }
            )
            episode_stats = _episode_stats_from_table(transformed, output_features)
            episodes_stats_rows.append(
                {"episode_index": dest_episode_index, "stats": episode_stats}
            )

            total_frames += transformed.num_rows
            dest_episode_index += 1

    if dest_episode_index == 0:
        raise RuntimeError("No subtask episodes were written.")

    if not skip_videos:
        _run_video_jobs(video_jobs, workers=video_workers)
        output_features = _apply_output_video_codec(output_features, vcodec=output_vcodec)

    output_info = {
        "codebase_version": info.get("codebase_version", "v2.1"),
        "robot_type": info.get("robot_type", "dexjoco"),
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "fps": fps,
        "chunks_size": CHUNKS_SIZE,
        "total_episodes": dest_episode_index,
        "total_frames": total_frames,
        "total_tasks": len(tasks_rows),
        "total_videos": dest_episode_index * len(video_keys),
        "total_chunks": math.ceil(dest_episode_index / CHUNKS_SIZE),
        "splits": {"train": f"0:{dest_episode_index}"},
        "features": output_features,
        "source_dataset": str(source_root),
        "annotation_file": str(annotation_path),
    }

    _write_json(output_root / "meta" / "info.json", output_info)
    _write_jsonl(output_root / "meta" / "tasks.jsonl", tasks_rows)
    _write_jsonl(output_root / "meta" / "episodes.jsonl", episodes_rows)
    _write_jsonl(
        output_root / "meta" / "episodes_stats.jsonl",
        [
            {"episode_index": row["episode_index"], "stats": _stats_to_json(row["stats"])}
            for row in episodes_stats_rows
        ],
    )
    _write_json(
        output_root / "meta" / "stats.json",
        _stats_to_json(aggregate_stats([row["stats"] for row in episodes_stats_rows])),
    )

    ann_out = output_root / "annotations" / ANNOTATION_FILENAME
    ann_out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(annotation_path, ann_out)

    print(
        f"Wrote {dest_episode_index} subtask episodes ({total_frames} frames) "
        f"from {len(source_episodes)} source episodes to {output_root}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--annotations",
        type=Path,
        default=None,
        help="Path to dual_hand_subtasks.jsonl (default: <source>/annotations/...).",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--skip-videos",
        action="store_true",
        help="Only write parquet/meta; skip ffmpeg video trimming.",
    )
    parser.add_argument("--video-workers", type=int, default=8)
    parser.add_argument("--output-vcodec", type=str, default="libx264")
    parser.add_argument("--video-crf", type=int, default=23)
    parser.add_argument("--video-preset", type=str, default="veryfast")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepare_subtask_dataset(
        source_root=args.source,
        output_root=args.output,
        annotation_path=args.annotations,
        overwrite=args.overwrite,
        skip_videos=args.skip_videos,
        video_workers=args.video_workers,
        output_vcodec=args.output_vcodec,
        video_crf=args.video_crf,
        video_preset=args.video_preset,
    )


if __name__ == "__main__":
    main()
