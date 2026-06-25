from pathlib import Path
from typing import Any

import h5py
import imageio.v2 as imageio
import json
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .types import EpisodeArrays


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_video(path: Path, frames_rgb: np.ndarray, fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(
        path,
        frames_rgb,
        fps=fps,
        codec="libx264",
        pixelformat="yuv420p",
        macro_block_size=1,
    )


def write_episode_parquet(
    path: Path,
    *,
    state: np.ndarray,
    action: np.ndarray,
    fps: int,
    episode_index: int,
    global_start_index: int,
    task_index: int,
    left_ee_pose: np.ndarray | None = None,
    right_ee_pose: np.ndarray | None = None,
    left_finger_tip_pos: np.ndarray | None = None,
    right_finger_tip_pos: np.ndarray | None = None,
) -> None:
    num_frames = int(action.shape[0])
    timestamps = np.arange(num_frames, dtype=np.float32) / float(fps)
    frame_indices = np.arange(num_frames, dtype=np.int64)
    episode_indices = np.full(num_frames, episode_index, dtype=np.int64)
    global_indices = np.arange(global_start_index, global_start_index + num_frames, dtype=np.int64)
    task_indices = np.full(num_frames, task_index, dtype=np.int64)

    arrays = [
        pa.array(state.astype(np.float32).tolist(), type=pa.list_(pa.float32())),
        pa.array(action.astype(np.float32).tolist(), type=pa.list_(pa.float32())),
        pa.array(timestamps, type=pa.float32()),
        pa.array(frame_indices, type=pa.int64()),
        pa.array(episode_indices, type=pa.int64()),
        pa.array(global_indices, type=pa.int64()),
        pa.array(task_indices, type=pa.int64()),
    ]
    names = [
        "observation.state",
        "action",
        "timestamp",
        "frame_index",
        "episode_index",
        "index",
        "task_index",
    ]
    if left_ee_pose is not None:
        arrays.append(pa.array(left_ee_pose.astype(np.float32).tolist(), type=pa.list_(pa.float32())))
        names.append("observation.state.left_ee_pose")
    if right_ee_pose is not None:
        arrays.append(pa.array(right_ee_pose.astype(np.float32).tolist(), type=pa.list_(pa.float32())))
        names.append("observation.state.right_ee_pose")
    if left_finger_tip_pos is not None:
        arrays.append(
            pa.array(
                left_finger_tip_pos.reshape(left_finger_tip_pos.shape[0], -1).astype(np.float32).tolist(),
                type=pa.list_(pa.float32()),
            )
        )
        names.append("observation.state.left_finger_tip_pos")
    if right_finger_tip_pos is not None:
        arrays.append(
            pa.array(
                right_finger_tip_pos.reshape(right_finger_tip_pos.shape[0], -1).astype(np.float32).tolist(),
                type=pa.list_(pa.float32()),
            )
        )
        names.append("observation.state.right_finger_tip_pos")

    table = pa.Table.from_arrays(arrays, names=names)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def read_episode_arrays(hdf5_path: Path) -> EpisodeArrays:
    with h5py.File(hdf5_path, "r") as f:
        action = np.asarray(f["action"], dtype=np.float32)
        state = np.asarray(f["observations/qpos"], dtype=np.float32)
        frames = np.asarray(f["observations/images/main"], dtype=np.uint8)
        hand = {
            "left_ee_pose": np.asarray(f["observations/left_ee_pose"], dtype=np.float32)
            if "observations/left_ee_pose" in f
            else None,
            "right_ee_pose": np.asarray(f["observations/right_ee_pose"], dtype=np.float32)
            if "observations/right_ee_pose" in f
            else None,
            "left_finger_tip_pos": np.asarray(f["observations/left_finger_tip_pos"], dtype=np.float32)
            if "observations/left_finger_tip_pos" in f
            else None,
            "right_finger_tip_pos": np.asarray(f["observations/right_finger_tip_pos"], dtype=np.float32)
            if "observations/right_finger_tip_pos" in f
            else None,
        }

    if action.ndim != 2:
        raise ValueError(f"`action` must be 2D in {hdf5_path}, got {action.shape}")
    if state.shape != action.shape:
        raise ValueError(f"`observations/qpos` shape {state.shape} does not match action {action.shape}")
    if frames.ndim != 4 or frames.shape[0] != action.shape[0] or frames.shape[-1] != 3:
        raise ValueError(
            f"`observations/images/main` shape {frames.shape} is incompatible with action {action.shape}"
        )
    return EpisodeArrays(action=action, state=state, frames=frames, hand=hand)

