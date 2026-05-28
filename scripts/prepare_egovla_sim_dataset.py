#!/usr/bin/env python3
"""Convert EgoVLA simulator HDF5 episodes to FastWAM's LeRobot-style format."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from pathlib import Path
from typing import Any

import h5py
import imageio.v2 as imageio
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


VIDEO_KEY = "observation.images.camera_0"
CHUNKS_SIZE = 1000

EGOVLA_SIM_ROOT = Path("data/EgoVLA_SIM")

LONG_TASKS: tuple[tuple[str, str], ...] = (
    ("Insert-And-Unload-Cans", "Insert and unload cans."),
    ("Stack-Can-Into-Drawer", "Stack the can into the drawer."),
    ("Sort-Cans", "Sort the cans."),
    ("Unload-Cans", "Unload the cans."),
    ("Insert-Cans", "Insert the cans."),
)

SHORT_TASKS: tuple[tuple[str, str], ...] = (
    ("Close-Drawer", "Close the drawer."),
    ("Flip-Mug", "Flip the mug."),
    ("Open-Drawer", "Open the drawer."),
    ("Open-Laptop", "Open the laptop."),
    ("Pour-Balls", "Pour the balls."),
    ("Push-Box", "Push the box."),
    ("Stack-Can", "Stack the can."),
)


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


def _episode_index(path: Path) -> int:
    stem = path.stem
    try:
        return int(stem.rsplit("_", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Unexpected episode file name: {path.name}") from exc


def _task_from_dir(path: Path) -> str:
    words = path.name.replace("_", "-").split("-")
    return " ".join(word.lower() for word in words if word).capitalize() + "."


def _feature_names(prefix: str, dim: int) -> list[str]:
    return [f"{prefix}_{i}" for i in range(dim)]


def _stats(values: np.ndarray) -> dict[str, list[float] | list[int]]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    return {
        "mean": arr.mean(axis=0).astype(float).tolist(),
        "std": arr.std(axis=0).astype(float).tolist(),
        "min": arr.min(axis=0).astype(float).tolist(),
        "max": arr.max(axis=0).astype(float).tolist(),
        "q01": np.quantile(arr, 0.01, axis=0).astype(float).tolist(),
        "q99": np.quantile(arr, 0.99, axis=0).astype(float).tolist(),
        "count": [int(arr.shape[0])],
    }


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


def _write_video(path: Path, frames_rgb: np.ndarray, fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(
        path,
        frames_rgb,
        fps=fps,
        codec="libx264",
        pixelformat="yuv420p",
        macro_block_size=1,
    )


def _write_episode_parquet(
    path: Path,
    *,
    state: np.ndarray,
    action: np.ndarray,
    fps: int,
    episode_index: int,
    global_start_index: int,
    task_index: int,
) -> None:
    num_frames = int(action.shape[0])
    timestamps = np.arange(num_frames, dtype=np.float32) / float(fps)
    frame_indices = np.arange(num_frames, dtype=np.int64)
    episode_indices = np.full(num_frames, episode_index, dtype=np.int64)
    global_indices = np.arange(global_start_index, global_start_index + num_frames, dtype=np.int64)
    task_indices = np.full(num_frames, task_index, dtype=np.int64)

    table = pa.Table.from_arrays(
        [
            pa.array(state.astype(np.float32).tolist(), type=pa.list_(pa.float32())),
            pa.array(action.astype(np.float32).tolist(), type=pa.list_(pa.float32())),
            pa.array(timestamps, type=pa.float32()),
            pa.array(frame_indices, type=pa.int64()),
            pa.array(episode_indices, type=pa.int64()),
            pa.array(global_indices, type=pa.int64()),
            pa.array(task_indices, type=pa.int64()),
        ],
        names=[
            "observation.state",
            "action",
            "timestamp",
            "frame_index",
            "episode_index",
            "index",
            "task_index",
        ],
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def _read_episode_arrays(hdf5_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(hdf5_path, "r") as f:
        action = np.asarray(f["action"], dtype=np.float32)
        state = np.asarray(f["observations/qpos"], dtype=np.float32)
        frames = np.asarray(f["observations/images/main"], dtype=np.uint8)

    if action.ndim != 2:
        raise ValueError(f"`action` must be 2D in {hdf5_path}, got {action.shape}")
    if state.shape != action.shape:
        raise ValueError(f"`observations/qpos` shape {state.shape} does not match action {action.shape}")
    if frames.ndim != 4 or frames.shape[0] != action.shape[0] or frames.shape[-1] != 3:
        raise ValueError(
            f"`observations/images/main` shape {frames.shape} is incompatible with action {action.shape}"
        )
    return action, state, frames


def _validate_dims(
    hdf5_path: Path,
    action: np.ndarray,
    state: np.ndarray,
    frames: np.ndarray,
    *,
    action_dim: int | None,
    state_dim: int | None,
    image_shape: tuple[int, int, int] | None,
) -> tuple[int, int, tuple[int, int, int]]:
    ad = int(action.shape[1])
    sd = int(state.shape[1])
    ish = tuple(int(x) for x in frames.shape[1:])
    if action_dim is None:
        action_dim, state_dim, image_shape = ad, sd, ish
    else:
        assert action_dim is not None and state_dim is not None and image_shape is not None
        if ad != action_dim or sd != state_dim:
            raise ValueError(f"Inconsistent action/state dims in {hdf5_path}")
        if ish != image_shape:
            raise ValueError(f"Inconsistent image shape in {hdf5_path}")
    return action_dim, state_dim, image_shape


def _build_info(
    *,
    action_dim: int,
    state_dim: int,
    image_shape: tuple[int, int, int],
    fps: int,
    total_episodes: int,
    total_frames: int,
    total_tasks: int,
) -> dict[str, Any]:
    height, width, channels = image_shape
    if channels != 3:
        raise ValueError(f"Expected RGB images with 3 channels, got {image_shape}")

    return {
        "codebase_version": "v2.0",
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "fps": fps,
        "chunks_size": CHUNKS_SIZE,
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": total_tasks,
        "total_videos": total_episodes,
        "total_chunks": math.ceil(total_episodes / CHUNKS_SIZE),
        "splits": {"train": f"0:{total_episodes}"},
        "features": {
            "observation.state": {
                "dtype": "float32",
                "shape": [state_dim],
                "names": _feature_names("qpos", state_dim),
            },
            "action": {
                "dtype": "float32",
                "shape": [action_dim],
                "names": _feature_names("action", action_dim),
            },
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
            VIDEO_KEY: {
                "dtype": "video",
                "shape": [height, width, channels],
                "names": ["height", "width", "rgb"],
                "info": {
                    "video.fps": fps,
                    "video.codec": "h264",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "has_audio": False,
                },
            },
        },
        "robot_type": "EgoVLA_SIM",
    }


def prepare_dataset(
    source_root: Path,
    output_root: Path,
    *,
    task: str | None,
    fps: int,
    overwrite: bool,
) -> None:
    task_text = task or _task_from_dir(source_root)
    prepare_merged_dataset(
        [(source_root, task_text)],
        output_root,
        sim_root=source_root.parent,
        fps=fps,
        overwrite=overwrite,
    )


def prepare_merged_dataset(
    sources: list[tuple[Path, str]],
    output_root: Path,
    *,
    sim_root: Path,
    fps: int,
    overwrite: bool,
) -> None:
    if not sources:
        raise ValueError("At least one source task directory is required.")

    for src, _ in sources:
        if not src.exists():
            raise FileNotFoundError(f"Missing source dataset: {src}")
        if not list(src.glob("episode_*.hdf5")):
            raise FileNotFoundError(f"No episode_*.hdf5 files found in: {src}")

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
    action_dim: int | None = None
    state_dim: int | None = None
    image_shape: tuple[int, int, int] | None = None

    for task_index, (src_root, task_text) in enumerate(sources):
        episode_files = sorted(src_root.glob("episode_*.hdf5"), key=_episode_index)
        print(f"Task {task_index}: {src_root.name} ({len(episode_files)} episodes)")

        task_state: list[np.ndarray] = []
        task_action: list[np.ndarray] = []
        task_timestamp: list[np.ndarray] = []
        task_frame_index: list[np.ndarray] = []
        task_episode_index: list[np.ndarray] = []
        task_index_arr: list[np.ndarray] = []
        task_global_index: list[np.ndarray] = []
        task_frame_count = 0

        for hdf5_path in episode_files:
            action, state, frames = _read_episode_arrays(hdf5_path)
            action_dim, state_dim, image_shape = _validate_dims(
                hdf5_path, action, state, frames,
                action_dim=action_dim, state_dim=state_dim, image_shape=image_shape,
            )

            dest_episode_index = next_episode_index
            chunk = dest_episode_index // CHUNKS_SIZE
            parquet_path = output_root / "data" / f"chunk-{chunk:03d}" / f"episode_{dest_episode_index:06d}.parquet"
            video_path = (
                output_root
                / "videos"
                / f"chunk-{chunk:03d}"
                / VIDEO_KEY
                / f"episode_{dest_episode_index:06d}.mp4"
            )

            _write_episode_parquet(
                parquet_path,
                state=state,
                action=action,
                fps=fps,
                episode_index=dest_episode_index,
                global_start_index=total_frames,
                task_index=task_index,
            )
            _write_video(video_path, frames, fps=fps)

            num_frames = int(action.shape[0])
            timestamps = np.arange(num_frames, dtype=np.float32) / float(fps)
            frame_indices = np.arange(num_frames, dtype=np.int64)
            episode_indices = np.full(num_frames, dest_episode_index, dtype=np.int64)
            global_indices = np.arange(total_frames, total_frames + num_frames, dtype=np.int64)
            task_indices = np.full(num_frames, task_index, dtype=np.int64)

            task_state.append(state)
            task_action.append(action)
            task_timestamp.append(timestamps)
            task_frame_index.append(frame_indices)
            task_episode_index.append(episode_indices)
            task_index_arr.append(task_indices)
            task_global_index.append(global_indices)

            episodes_rows.append(
                {
                    "episode_index": dest_episode_index,
                    "tasks": [task_text],
                    "length": num_frames,
                }
            )
            total_frames += num_frames
            task_frame_count += num_frames
            next_episode_index += 1

            if dest_episode_index % 50 == 0 or dest_episode_index == next_episode_index - 1:
                print(f"  episode {dest_episode_index}: {hdf5_path.name} ({num_frames} frames)")

        if task_frame_count > 0:
            stats_with_counts.append(
                (
                    {
                        "observation.state": _stats(np.concatenate(task_state, axis=0)),
                        "action": _stats(np.concatenate(task_action, axis=0)),
                        "timestamp": _stats(np.concatenate(task_timestamp, axis=0)),
                        "frame_index": _stats(np.concatenate(task_frame_index, axis=0)),
                        "episode_index": _stats(np.concatenate(task_episode_index, axis=0)),
                        "index": _stats(np.concatenate(task_global_index, axis=0)),
                        "task_index": _stats(np.concatenate(task_index_arr, axis=0)),
                    },
                    task_frame_count,
                )
            )

    assert action_dim is not None and state_dim is not None and image_shape is not None
    info = _build_info(
        action_dim=action_dim,
        state_dim=state_dim,
        image_shape=image_shape,
        fps=fps,
        total_episodes=next_episode_index,
        total_frames=total_frames,
        total_tasks=len(sources),
    )

    _write_json(output_root / "meta" / "info.json", info)
    _write_json(output_root / "meta" / "stats.json", _combine_stats(stats_with_counts))
    _write_jsonl(output_root / "meta" / "tasks.jsonl", tasks_rows)
    _write_jsonl(output_root / "meta" / "episodes.jsonl", episodes_rows)
    print(f"Wrote {next_episode_index} episodes and {total_frames} frames to {output_root}")


def _resolve_split_sources(sim_root: Path, split: str) -> list[tuple[Path, str]]:
    if split == "long":
        spec = LONG_TASKS
    elif split == "short":
        spec = SHORT_TASKS
    else:
        raise ValueError(f"Unknown split: {split}")
    return [(sim_root / name, task) for name, task in spec]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=None, help="Single task HDF5 directory.")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--sim-root", type=Path, default=EGOVLA_SIM_ROOT)
    parser.add_argument(
        "--split",
        choices=("long", "short"),
        default=None,
        help="Merge preset EgoVLA_SIM task groups into one dataset.",
    )
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.split is not None:
        sources = _resolve_split_sources(args.sim_root, args.split)
        prepare_merged_dataset(
            sources,
            args.output_root,
            sim_root=args.sim_root,
            fps=args.fps,
            overwrite=args.overwrite,
        )
        return

    if args.source_root is None:
        raise ValueError("Either --split or --source-root must be provided.")

    prepare_dataset(
        source_root=args.source_root,
        output_root=args.output_root,
        task=args.task,
        fps=args.fps,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
