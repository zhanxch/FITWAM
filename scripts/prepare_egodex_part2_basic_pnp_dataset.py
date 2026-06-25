#!/usr/bin/env python3
"""Convert EgoDex basic_pick_place episodes to FastWAM LeRobot-style video-pretrain data."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

DEFAULT_SOURCE_ROOT = Path("data/egodex/part2/basic_pick_place")
DEFAULT_OUTPUT_ROOT = Path("data/egodex_part2_basic_pnp_fastwam_video_pretrain")
CHUNKS_SIZE = 1000
DEST_VIDEO_KEY = "observation.images.ego"
FPS = 30
DUMMY_ACTION_DIM = 1
DUMMY_STATE_DIM = 1


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


def _decode_attr(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _read_language(hdf5_path: Path) -> str:
    with h5py.File(hdf5_path, "r") as root:
        llm_type = _decode_attr(root.attrs.get("llm_type", "reset"))
        if llm_type == "reversible":
            direction = _decode_attr(root.attrs["which_llm_description"])
            key = "llm_description" if direction == "1" else "llm_description2"
            return _decode_attr(root.attrs[key])
        return _decode_attr(root.attrs["llm_description"])


def _read_num_frames(hdf5_path: Path) -> int:
    with h5py.File(hdf5_path, "r") as root:
        return int(len(root["transforms/camera"]))


def _probe_video_shape(mp4_path: Path) -> tuple[int, int]:
    payload = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "json",
            str(mp4_path),
        ],
        text=True,
    )
    stream = json.loads(payload)["streams"][0]
    return int(stream["height"]), int(stream["width"])


def _zero_stats(dim: int, count: int) -> dict[str, list[float] | list[int]]:
    zeros = [0.0] * dim
    return {
        "mean": zeros,
        "std": zeros,
        "min": zeros,
        "max": zeros,
        "q01": zeros,
        "q99": zeros,
        "count": [count],
    }


def _episode_stats(length: int) -> dict[str, dict[str, list[float] | list[int]]]:
    return {
        "action": _zero_stats(DUMMY_ACTION_DIM, length),
        "observation.state": _zero_stats(DUMMY_STATE_DIM, length),
    }


def write_episodes_stats(output_root: Path, episodes_rows: list[dict[str, Any]]) -> None:
    episodes_stats_rows = [
        {
            "episode_index": int(row["episode_index"]),
            "stats": _episode_stats(int(row["length"])),
        }
        for row in episodes_rows
    ]
    _write_jsonl(output_root / "meta" / "episodes_stats.jsonl", episodes_stats_rows)


def _build_features(height: int, width: int) -> dict[str, Any]:
    return {
        DEST_VIDEO_KEY: {
            "dtype": "video",
            "shape": [height, width, 3],
            "names": ["height", "width", "channel"],
            "info": {
                "video.height": height,
                "video.width": width,
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "video.fps": FPS,
                "video.channels": 3,
                "has_audio": False,
            },
        },
        "action": {
            "dtype": "float32",
            "shape": [DUMMY_ACTION_DIM],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": [DUMMY_STATE_DIM],
        },
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
    }


def _write_episode_parquet(
    path: Path,
    *,
    num_frames: int,
    episode_index: int,
    global_start_index: int,
    task_index: int,
) -> None:
    timestamps = (np.arange(num_frames, dtype=np.float32) / FPS).tolist()
    frame_indices = np.arange(num_frames, dtype=np.int64).tolist()
    episode_indices = [episode_index] * num_frames
    global_indices = list(range(global_start_index, global_start_index + num_frames))
    task_indices = [task_index] * num_frames
    zero_action = [[0.0] * DUMMY_ACTION_DIM for _ in range(num_frames)]
    zero_state = [[0.0] * DUMMY_STATE_DIM for _ in range(num_frames)]

    table = pa.table(
        {
            "action": pa.array(zero_action, type=pa.list_(pa.float32(), DUMMY_ACTION_DIM)),
            "observation.state": pa.array(zero_state, type=pa.list_(pa.float32(), DUMMY_STATE_DIM)),
            "timestamp": pa.array(timestamps, type=pa.float32()),
            "frame_index": pa.array(frame_indices, type=pa.int64()),
            "episode_index": pa.array(episode_indices, type=pa.int64()),
            "index": pa.array(global_indices, type=pa.int64()),
            "task_index": pa.array(task_indices, type=pa.int64()),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def _list_episode_stems(source_root: Path) -> list[str]:
    mp4_stems = {path.stem for path in source_root.glob("*.mp4")}
    hdf5_stems = {path.stem for path in source_root.glob("*.hdf5")}
    paired = sorted(mp4_stems & hdf5_stems, key=lambda name: int(name))
    if not paired:
        raise FileNotFoundError(f"No paired mp4/hdf5 episodes found under {source_root}")
    return paired


def prepare_dataset(
    source_root: Path,
    output_root: Path,
    *,
    copy_videos: bool,
    overwrite: bool,
    max_episodes: int | None,
) -> None:
    if not source_root.exists():
        raise FileNotFoundError(f"Missing source root: {source_root}")

    episode_stems = _list_episode_stems(source_root)
    if max_episodes is not None:
        episode_stems = episode_stems[:max_episodes]

    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {output_root}. Pass --overwrite to replace it.")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    sample_mp4 = source_root / f"{episode_stems[0]}.mp4"
    height, width = _probe_video_shape(sample_mp4)
    features = _build_features(height, width)

    tasks_rows: list[dict[str, Any]] = []
    episodes_rows: list[dict[str, Any]] = []
    total_frames = 0

    for episode_index, stem in enumerate(episode_stems):
        hdf5_path = source_root / f"{stem}.hdf5"
        mp4_path = source_root / f"{stem}.mp4"
        num_frames = _read_num_frames(hdf5_path)
        if num_frames <= 0:
            continue

        task_text = _read_language(hdf5_path)
        tasks_rows.append({"task_index": episode_index, "task": task_text})

        chunk = episode_index // CHUNKS_SIZE
        parquet_path = output_root / "data" / f"chunk-{chunk:03d}" / f"episode_{episode_index:06d}.parquet"
        video_path = (
            output_root
            / "videos"
            / f"chunk-{chunk:03d}"
            / DEST_VIDEO_KEY
            / f"episode_{episode_index:06d}.mp4"
        )

        _write_episode_parquet(
            parquet_path,
            num_frames=num_frames,
            episode_index=episode_index,
            global_start_index=total_frames,
            task_index=episode_index,
        )
        _link_or_copy(mp4_path, video_path, copy_videos=copy_videos)

        episodes_rows.append(
            {
                "episode_index": episode_index,
                "tasks": [task_text],
                "length": num_frames,
            }
        )
        total_frames += num_frames

        if (episode_index + 1) % 500 == 0 or episode_index + 1 == len(episode_stems):
            print(f"Processed {episode_index + 1}/{len(episode_stems)} episodes")

    total_episodes = len(episodes_rows)
    info = {
        "codebase_version": "v2.1",
        "robot_type": "egodex",
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "fps": FPS,
        "chunks_size": CHUNKS_SIZE,
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": total_episodes,
        "total_videos": total_episodes,
        "total_chunks": math.ceil(total_episodes / CHUNKS_SIZE) if total_episodes else 0,
        "splits": {"train": f"0:{total_episodes}"},
        "features": features,
    }
    stats = {
        "action": _zero_stats(DUMMY_ACTION_DIM, total_frames),
        "observation.state": _zero_stats(DUMMY_STATE_DIM, total_frames),
        "timestamp": _zero_stats(1, total_frames),
        "frame_index": _zero_stats(1, total_frames),
        "episode_index": _zero_stats(1, total_frames),
        "index": _zero_stats(1, total_frames),
        "task_index": _zero_stats(1, total_frames),
    }

    _write_json(output_root / "meta" / "info.json", info)
    _write_json(output_root / "meta" / "stats.json", stats)
    _write_jsonl(output_root / "meta" / "tasks.jsonl", tasks_rows)
    _write_jsonl(output_root / "meta" / "episodes.jsonl", episodes_rows)
    write_episodes_stats(output_root, episodes_rows)
    print(f"Wrote {total_episodes} episodes and {total_frames} frames to {output_root}")


def repair_episodes_stats(output_root: Path) -> None:
    episodes_path = output_root / "meta" / "episodes.jsonl"
    if not episodes_path.exists():
        raise FileNotFoundError(f"Missing episodes file: {episodes_path}")
    episodes_rows = [
        json.loads(line)
        for line in episodes_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    write_episodes_stats(output_root, episodes_rows)
    print(f"Wrote episodes_stats.jsonl for {len(episodes_rows)} episodes to {output_root}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--copy-videos",
        action="store_true",
        help="Copy videos instead of hard-linking them when possible.",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Optional cap for quick smoke tests.",
    )
    parser.add_argument(
        "--repair-episodes-stats",
        action="store_true",
        help="Only write missing meta/episodes_stats.jsonl for an existing output dataset.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.repair_episodes_stats:
        repair_episodes_stats(args.output_root)
        return
    prepare_dataset(
        source_root=args.source_root,
        output_root=args.output_root,
        copy_videos=args.copy_videos,
        overwrite=args.overwrite,
        max_episodes=args.max_episodes,
    )


if __name__ == "__main__":
    main()
