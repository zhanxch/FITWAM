#!/usr/bin/env python3
"""Merge DexJoCo LeRobot task datasets into one ego-camera FastWAM dataset."""

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

from fastwam.datasets.lerobot.lerobot.datasets.compute_stats import (
    aggregate_stats,
    compute_episode_stats,
)

DEFAULT_SOURCE_ROOT = Path("data/dexjoco/dexjoco_lerobot_datasets")
DEFAULT_OUTPUT_ROOT = Path("data/dexjoco_ego")
CHUNKS_SIZE = 1000
DUAL_ARM_ACTION_DIM = 44
DUAL_ARM_STATE_DIM = 46
SINGLE_ARM_ACTION_DIM = 22
SINGLE_ARM_STATE_DIM = 23
DEST_VIDEO_KEY = "observation.images.ego"
EGO_SOURCE_KEYS = (
    "observation.images.ego",
    "observation.images.ego_right",
    "observation.images.front",
)


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


def _fit_vector(values: list[float] | np.ndarray, target_dim: int, *, allow_pad: bool) -> list[float]:
    arr = np.asarray(values, dtype=np.float32)
    if arr.shape[-1] >= target_dim:
        return arr[:target_dim].astype(np.float32).tolist()
    if not allow_pad:
        raise ValueError(
            f"Vector dim {arr.shape[-1]} is smaller than target dim {target_dim}; "
            "single-arm mode does not zero-pad missing dimensions."
        )
    padded = np.zeros(target_dim, dtype=np.float32)
    padded[: arr.shape[-1]] = arr
    return padded.tolist()


def _detect_ego_source_key(features: dict[str, Any]) -> str:
    for key in EGO_SOURCE_KEYS:
        if key in features:
            return key
    raise KeyError(f"No ego camera found in features. Expected one of {EGO_SOURCE_KEYS}")


def _list_image_feature_keys(features: dict[str, Any]) -> list[str]:
    return sorted(key for key in features if key.startswith("observation.images."))


def _build_output_features(
    template_info: dict[str, Any],
    *,
    action_dim: int,
    state_dim: int,
    keep_all_cameras: bool = False,
    ego_source_key: str | None = None,
) -> dict[str, Any]:
    features: dict[str, Any] = {
        "action": {
            "dtype": "float32",
            "shape": [action_dim],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": [state_dim],
        },
    }
    if keep_all_cameras:
        for key in _list_image_feature_keys(template_info["features"]):
            features[key] = dict(template_info["features"][key])
        if DEST_VIDEO_KEY not in features:
            if ego_source_key is None:
                ego_source_key = _detect_ego_source_key(template_info["features"])
            features[DEST_VIDEO_KEY] = dict(template_info["features"][ego_source_key])
    else:
        if ego_source_key is None:
            ego_source_key = _detect_ego_source_key(template_info["features"])
        ego_feature = dict(template_info["features"][ego_source_key])
        if ego_source_key != DEST_VIDEO_KEY:
            ego_feature = dict(template_info["features"][ego_source_key])
        features[DEST_VIDEO_KEY] = ego_feature

    for key, value in template_info["features"].items():
        if key in features or key.startswith("observation.images."):
            continue
        features[key] = value
    return features


def _transform_table(
    table: pa.Table,
    *,
    dest_episode_index: int,
    task_index: int,
    global_index_start: int,
    action_dim: int,
    state_dim: int,
    allow_pad: bool,
) -> pa.Table:
    columns: list[pa.Array] = []
    names: list[str] = []
    num_rows = table.num_rows

    for name in table.column_names:
        if name == "action":
            values = [
                _fit_vector(table[name][row_idx].as_py(), action_dim, allow_pad=allow_pad)
                for row_idx in range(num_rows)
            ]
            columns.append(pa.array(values, type=pa.list_(pa.float32(), action_dim)))
        elif name == "observation.state":
            values = [
                _fit_vector(table[name][row_idx].as_py(), state_dim, allow_pad=allow_pad)
                for row_idx in range(num_rows)
            ]
            columns.append(pa.array(values, type=pa.list_(pa.float32(), state_dim)))
        elif name == "episode_index":
            columns.append(pa.array([dest_episode_index] * num_rows, type=table[name].type))
        elif name == "index":
            columns.append(
                pa.array(range(global_index_start, global_index_start + num_rows), type=table[name].type)
            )
        elif name == "task_index":
            columns.append(pa.array([task_index] * num_rows, type=table[name].type))
        else:
            columns.append(table[name])
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


def _stats_to_json(stats: dict[str, dict[str, np.ndarray]]) -> dict[str, dict[str, list[float]]]:
    return {key: {name: value.tolist() for name, value in ft_stats.items()} for key, ft_stats in stats.items()}


def prepare_dataset(
    source_root: Path,
    output_root: Path,
    *,
    copy_videos: bool,
    overwrite: bool,
    exclude_substr: str | None = None,
    include_substr: str | None = None,
    single_arm: bool = False,
    keep_all_cameras: bool = False,
) -> None:
    if not source_root.exists():
        raise FileNotFoundError(f"Missing source root: {source_root}")

    source_dirs = sorted(path for path in source_root.iterdir() if path.is_dir())
    if exclude_substr:
        source_dirs = [path for path in source_dirs if exclude_substr not in path.name]
    if include_substr:
        source_dirs = [path for path in source_dirs if include_substr in path.name]
    if not source_dirs:
        raise FileNotFoundError(f"No task datasets found under {source_root}")

    action_dim = SINGLE_ARM_ACTION_DIM if single_arm else DUAL_ARM_ACTION_DIM
    state_dim = SINGLE_ARM_STATE_DIM if single_arm else DUAL_ARM_STATE_DIM
    allow_pad = not single_arm

    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {output_root}. Pass --overwrite to replace it.")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    tasks_rows: list[dict[str, Any]] = []
    episodes_rows: list[dict[str, Any]] = []
    episodes_stats_rows: list[dict[str, Any]] = []

    total_frames = 0
    next_episode_index = 0
    output_features: dict[str, Any] | None = None
    output_video_keys: list[str] | None = None

    for task_index, src_root in enumerate(source_dirs):
        info = _read_json(src_root / "meta" / "info.json")
        ego_source_key = _detect_ego_source_key(info["features"])
        if output_features is None:
            template_info = dict(info)
            template_info["features"] = dict(info["features"])
            if not keep_all_cameras and ego_source_key != DEST_VIDEO_KEY:
                template_info["features"][DEST_VIDEO_KEY] = dict(info["features"][ego_source_key])
            output_features = _build_output_features(
                template_info,
                action_dim=action_dim,
                state_dim=state_dim,
                keep_all_cameras=keep_all_cameras,
                ego_source_key=ego_source_key,
            )
            output_video_keys = _list_image_feature_keys(output_features)
            if not output_video_keys:
                raise ValueError("Output dataset must contain at least one camera feature.")

        with (src_root / "meta" / "tasks.jsonl").open("r", encoding="utf-8") as f:
            source_tasks = [json.loads(line) for line in f if line.strip()]
        task_text = source_tasks[0]["task"]
        tasks_rows.append({"task_index": task_index, "task": task_text})

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
            transformed = _transform_table(
                table,
                dest_episode_index=dest_episode_index,
                task_index=task_index,
                global_index_start=total_frames,
                action_dim=action_dim,
                state_dim=state_dim,
                allow_pad=allow_pad,
            )
            dest_data.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(transformed, dest_data)

            assert output_video_keys is not None
            for video_key in output_video_keys:
                source_video_key = ego_source_key if (
                    not keep_all_cameras and video_key == DEST_VIDEO_KEY
                ) else video_key
                source_video = src_root / info["video_path"].format(
                    episode_chunk=source_chunk,
                    video_key=source_video_key,
                    episode_index=source_episode_index,
                )
                dest_video = (
                    output_root
                    / "videos"
                    / f"chunk-{dest_chunk:03d}"
                    / video_key
                    / f"episode_{dest_episode_index:06d}.mp4"
                )
                _link_or_copy(source_video, dest_video, copy_videos=copy_videos)

            episodes_rows.append(
                {
                    "episode_index": dest_episode_index,
                    "tasks": [task_text],
                    "length": int(source_episode["length"]),
                }
            )
            episode_stats = _episode_stats_from_table(transformed, output_features)
            episodes_stats_rows.append(
                {
                    "episode_index": dest_episode_index,
                    "stats": episode_stats,
                }
            )

            total_frames += transformed.num_rows
            next_episode_index += 1

        print(f"Merged {src_root.name}: {len(source_episodes)} episodes")

    assert output_features is not None
    info = {
        "codebase_version": "v2.1",
        "robot_type": "dexjoco",
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "fps": 30,
        "chunks_size": CHUNKS_SIZE,
        "total_episodes": next_episode_index,
        "total_frames": total_frames,
        "total_tasks": len(tasks_rows),
        "total_videos": next_episode_index * len(output_video_keys or [DEST_VIDEO_KEY]),
        "total_chunks": math.ceil(next_episode_index / CHUNKS_SIZE),
        "splits": {"train": f"0:{next_episode_index}"},
        "features": output_features,
    }

    _write_json(output_root / "meta" / "info.json", info)
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

    print(f"Wrote {next_episode_index} episodes and {total_frames} frames to {output_root}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=DEFAULT_SOURCE_ROOT,
        help="Directory containing per-task DexJoCo LeRobot datasets.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Destination LeRobot dataset directory.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output directory.")
    parser.add_argument(
        "--copy-videos",
        action="store_true",
        help="Copy videos instead of hard-linking them when possible.",
    )
    parser.add_argument(
        "--exclude-substr",
        type=str,
        default=None,
        help="Skip task directories whose name contains this substring (e.g. bimanual).",
    )
    parser.add_argument(
        "--include-substr",
        type=str,
        default=None,
        help="Keep only task directories whose name contains this substring (e.g. water_plant).",
    )
    parser.add_argument(
        "--single-arm",
        action="store_true",
        help="Keep native single-arm dims (action=22, state=23) without zero-padding to dual-arm.",
    )
    parser.add_argument(
        "--keep-all-cameras",
        action="store_true",
        help="Preserve every observation.images.* video stream instead of exporting ego only.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepare_dataset(
        source_root=args.source_root,
        output_root=args.output_root,
        copy_videos=args.copy_videos,
        overwrite=args.overwrite,
        exclude_substr=args.exclude_substr,
        include_substr=args.include_substr,
        single_arm=args.single_arm,
        keep_all_cameras=args.keep_all_cameras,
    )


if __name__ == "__main__":
    main()
