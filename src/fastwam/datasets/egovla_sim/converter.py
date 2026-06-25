from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import numpy as np

from .constants import CHUNKS_SIZE, LONG_TASKS, SHORT_TASKS, VIDEO_KEY
from .io import read_episode_arrays, write_episode_parquet, write_json, write_jsonl, write_video
from .schema import build_info, combine_stats, stats
from .types import ConversionConfig, DatasetSummary, EpisodeArrays, TaskSource
from .visualization import overlay_hand_pose


def episode_index(path: Path) -> int:
    stem = path.stem
    try:
        return int(stem.rsplit("_", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Unexpected episode file name: {path.name}") from exc


def task_from_dir(path: Path) -> str:
    words = path.name.replace("_", "-").split("-")
    return " ".join(word.lower() for word in words if word).capitalize() + "."


def resolve_split_sources(sim_root: Path, split: str) -> list[tuple[Path, str]]:
    if split == "long":
        spec = LONG_TASKS
    elif split == "short":
        spec = SHORT_TASKS
    else:
        raise ValueError(f"Unknown split: {split}")
    return [(sim_root / name, task) for name, task in spec]


def prepare_dataset(
    source_root: Path,
    output_root: Path,
    *,
    task: str | None,
    fps: int,
    overwrite: bool,
    include_hand_pose: bool,
    overlay_hand_pose: bool,
) -> None:
    task_text = task or task_from_dir(source_root)
    prepare_merged_dataset(
        [(source_root, task_text)],
        output_root,
        sim_root=source_root.parent,
        fps=fps,
        overwrite=overwrite,
        include_hand_pose=include_hand_pose,
        overlay_hand_pose=overlay_hand_pose,
    )


def prepare_merged_dataset(
    sources: list[tuple[Path, str]],
    output_root: Path,
    *,
    sim_root: Path,
    fps: int,
    overwrite: bool,
    include_hand_pose: bool,
    overlay_hand_pose: bool,
) -> None:
    task_sources = [TaskSource(root=source_root, task_text=task_text) for source_root, task_text in sources]
    config = ConversionConfig(
        output_root=output_root,
        sim_root=sim_root,
        fps=fps,
        overwrite=overwrite,
        include_hand_pose=include_hand_pose,
        overlay_hand_pose=overlay_hand_pose,
    )
    convert_dataset(task_sources, config)


def convert_dataset(task_sources: list[TaskSource], config: ConversionConfig) -> DatasetSummary:
    _validate_sources(task_sources)
    _prepare_output_root(config.output_root, config.overwrite)

    tasks_rows = [{"task_index": idx, "task": source.task_text} for idx, source in enumerate(task_sources)]
    episodes_rows: list[dict[str, Any]] = []
    stats_with_counts: list[tuple[dict[str, Any], int]] = []

    total_frames = 0
    next_episode_index = 0
    action_dim: int | None = None
    state_dim: int | None = None
    image_shape: tuple[int, int, int] | None = None

    for task_index, task_source in enumerate(task_sources):
        task_result = _convert_task(
            task_index=task_index,
            task_source=task_source,
            config=config,
            next_episode_index=next_episode_index,
            total_frames=total_frames,
            action_dim=action_dim,
            state_dim=state_dim,
            image_shape=image_shape,
        )
        episodes_rows.extend(task_result["episodes_rows"])
        stats_with_counts.append((task_result["stats"], task_result["frame_count"]))
        total_frames = task_result["total_frames"]
        next_episode_index = task_result["next_episode_index"]
        action_dim = task_result["action_dim"]
        state_dim = task_result["state_dim"]
        image_shape = task_result["image_shape"]

    assert action_dim is not None and state_dim is not None and image_shape is not None
    info = build_info(
        action_dim=action_dim,
        state_dim=state_dim,
        image_shape=image_shape,
        fps=config.fps,
        total_episodes=next_episode_index,
        total_frames=total_frames,
        total_tasks=len(task_sources),
        include_hand_pose=config.include_hand_pose,
    )

    write_json(config.output_root / "meta" / "info.json", info)
    write_json(config.output_root / "meta" / "stats.json", combine_stats(stats_with_counts))
    write_jsonl(config.output_root / "meta" / "tasks.jsonl", tasks_rows)
    write_jsonl(config.output_root / "meta" / "episodes.jsonl", episodes_rows)
    print(f"Wrote {next_episode_index} episodes and {total_frames} frames to {config.output_root}")
    return DatasetSummary(
        total_episodes=next_episode_index,
        total_frames=total_frames,
        action_dim=action_dim,
        state_dim=state_dim,
        image_shape=image_shape,
    )


def _validate_sources(task_sources: list[TaskSource]) -> None:
    if not task_sources:
        raise ValueError("At least one source task directory is required.")

    for source in task_sources:
        if not source.root.exists():
            raise FileNotFoundError(f"Missing source dataset: {source.root}")
        if not list(source.root.glob("episode_*.hdf5")):
            raise FileNotFoundError(f"No episode_*.hdf5 files found in: {source.root}")


def _prepare_output_root(output_root: Path, overwrite: bool) -> None:
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {output_root}. Pass --overwrite to replace it.")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)


def _convert_task(
    *,
    task_index: int,
    task_source: TaskSource,
    config: ConversionConfig,
    next_episode_index: int,
    total_frames: int,
    action_dim: int | None,
    state_dim: int | None,
    image_shape: tuple[int, int, int] | None,
) -> dict[str, Any]:
    episode_files = sorted(task_source.root.glob("episode_*.hdf5"), key=episode_index)
    print(f"Task {task_index}: {task_source.root.name} ({len(episode_files)} episodes)")

    task_arrays = _new_task_arrays()
    episodes_rows: list[dict[str, Any]] = []
    task_frame_count = 0

    for hdf5_path in episode_files:
        episode = read_episode_arrays(hdf5_path)
        action_dim, state_dim, image_shape = _validate_dims(
            hdf5_path,
            episode,
            action_dim=action_dim,
            state_dim=state_dim,
            image_shape=image_shape,
        )
        _validate_hand_requirements(hdf5_path, episode, config)

        dest_episode_index = next_episode_index
        chunk = dest_episode_index // CHUNKS_SIZE
        parquet_path = config.output_root / "data" / f"chunk-{chunk:03d}" / f"episode_{dest_episode_index:06d}.parquet"
        video_path = (
            config.output_root
            / "videos"
            / f"chunk-{chunk:03d}"
            / VIDEO_KEY
            / f"episode_{dest_episode_index:06d}.mp4"
        )

        _write_episode_outputs(
            parquet_path=parquet_path,
            video_path=video_path,
            episode=episode,
            fps=config.fps,
            dest_episode_index=dest_episode_index,
            total_frames=total_frames,
            task_index=task_index,
            include_hand_pose=config.include_hand_pose,
            should_overlay_hand_pose=config.overlay_hand_pose,
        )

        num_frames = int(episode.action.shape[0])
        _append_task_arrays(
            task_arrays=task_arrays,
            episode=episode,
            fps=config.fps,
            dest_episode_index=dest_episode_index,
            total_frames=total_frames,
            task_index=task_index,
            include_hand_pose=config.include_hand_pose,
        )
        episodes_rows.append(
            {
                "episode_index": dest_episode_index,
                "tasks": [task_source.task_text],
                "length": num_frames,
            }
        )

        total_frames += num_frames
        task_frame_count += num_frames
        next_episode_index += 1

        if dest_episode_index % 50 == 0 or dest_episode_index == next_episode_index - 1:
            print(f"  episode {dest_episode_index}: {hdf5_path.name} ({num_frames} frames)")

    return {
        "episodes_rows": episodes_rows,
        "stats": _build_task_stats(task_arrays, config.include_hand_pose),
        "frame_count": task_frame_count,
        "total_frames": total_frames,
        "next_episode_index": next_episode_index,
        "action_dim": action_dim,
        "state_dim": state_dim,
        "image_shape": image_shape,
    }


def _new_task_arrays() -> dict[str, list[np.ndarray]]:
    return {
        "state": [],
        "action": [],
        "left_ee_pose": [],
        "right_ee_pose": [],
        "left_finger_tip_pos": [],
        "right_finger_tip_pos": [],
        "timestamp": [],
        "frame_index": [],
        "episode_index": [],
        "task_index": [],
        "global_index": [],
    }


def _validate_dims(
    hdf5_path: Path,
    episode: EpisodeArrays,
    *,
    action_dim: int | None,
    state_dim: int | None,
    image_shape: tuple[int, int, int] | None,
) -> tuple[int, int, tuple[int, int, int]]:
    current_action_dim = int(episode.action.shape[1])
    current_state_dim = int(episode.state.shape[1])
    current_image_shape = tuple(int(x) for x in episode.frames.shape[1:])
    if action_dim is None:
        action_dim, state_dim, image_shape = current_action_dim, current_state_dim, current_image_shape
    else:
        assert state_dim is not None and image_shape is not None
        if current_action_dim != action_dim or current_state_dim != state_dim:
            raise ValueError(f"Inconsistent action/state dims in {hdf5_path}")
        if current_image_shape != image_shape:
            raise ValueError(f"Inconsistent image shape in {hdf5_path}")
    return action_dim, state_dim, image_shape


def _validate_hand_requirements(hdf5_path: Path, episode: EpisodeArrays, config: ConversionConfig) -> None:
    if config.include_hand_pose and any(episode.hand[key] is None for key in episode.hand):
        raise ValueError(f"Missing hand pose keys in {hdf5_path} but --include-hand-pose was requested.")
    if config.overlay_hand_pose and any(episode.hand[key] is None for key in episode.hand):
        raise ValueError(f"Missing hand pose keys in {hdf5_path} but --overlay-hand-pose was requested.")


def _write_episode_outputs(
    *,
    parquet_path: Path,
    video_path: Path,
    episode: EpisodeArrays,
    fps: int,
    dest_episode_index: int,
    total_frames: int,
    task_index: int,
    include_hand_pose: bool,
    should_overlay_hand_pose: bool,
) -> None:
    write_episode_parquet(
        parquet_path,
        state=episode.state,
        action=episode.action,
        fps=fps,
        episode_index=dest_episode_index,
        global_start_index=total_frames,
        task_index=task_index,
        left_ee_pose=episode.hand["left_ee_pose"] if include_hand_pose else None,
        right_ee_pose=episode.hand["right_ee_pose"] if include_hand_pose else None,
        left_finger_tip_pos=episode.hand["left_finger_tip_pos"] if include_hand_pose else None,
        right_finger_tip_pos=episode.hand["right_finger_tip_pos"] if include_hand_pose else None,
    )
    video_frames = overlay_hand_pose(episode.frames, episode.hand) if should_overlay_hand_pose else episode.frames
    write_video(video_path, video_frames, fps=fps)


def _append_task_arrays(
    *,
    task_arrays: dict[str, list[np.ndarray]],
    episode: EpisodeArrays,
    fps: int,
    dest_episode_index: int,
    total_frames: int,
    task_index: int,
    include_hand_pose: bool,
) -> None:
    num_frames = int(episode.action.shape[0])
    timestamps = np.arange(num_frames, dtype=np.float32) / float(fps)
    frame_indices = np.arange(num_frames, dtype=np.int64)
    episode_indices = np.full(num_frames, dest_episode_index, dtype=np.int64)
    global_indices = np.arange(total_frames, total_frames + num_frames, dtype=np.int64)
    task_indices = np.full(num_frames, task_index, dtype=np.int64)

    task_arrays["state"].append(episode.state)
    task_arrays["action"].append(episode.action)
    if include_hand_pose:
        task_arrays["left_ee_pose"].append(np.asarray(episode.hand["left_ee_pose"], dtype=np.float32))
        task_arrays["right_ee_pose"].append(np.asarray(episode.hand["right_ee_pose"], dtype=np.float32))
        task_arrays["left_finger_tip_pos"].append(
            np.asarray(episode.hand["left_finger_tip_pos"], dtype=np.float32).reshape(-1, 15)
        )
        task_arrays["right_finger_tip_pos"].append(
            np.asarray(episode.hand["right_finger_tip_pos"], dtype=np.float32).reshape(-1, 15)
        )
    task_arrays["timestamp"].append(timestamps)
    task_arrays["frame_index"].append(frame_indices)
    task_arrays["episode_index"].append(episode_indices)
    task_arrays["task_index"].append(task_indices)
    task_arrays["global_index"].append(global_indices)


def _build_task_stats(task_arrays: dict[str, list[np.ndarray]], include_hand_pose: bool) -> dict[str, Any]:
    base_stats = {
        "observation.state": stats(np.concatenate(task_arrays["state"], axis=0)),
        "action": stats(np.concatenate(task_arrays["action"], axis=0)),
        "timestamp": stats(np.concatenate(task_arrays["timestamp"], axis=0)),
        "frame_index": stats(np.concatenate(task_arrays["frame_index"], axis=0)),
        "episode_index": stats(np.concatenate(task_arrays["episode_index"], axis=0)),
        "index": stats(np.concatenate(task_arrays["global_index"], axis=0)),
        "task_index": stats(np.concatenate(task_arrays["task_index"], axis=0)),
    }
    if not include_hand_pose:
        return base_stats

    return {
        **base_stats,
        "observation.state.left_ee_pose": stats(np.concatenate(task_arrays["left_ee_pose"], axis=0)),
        "observation.state.right_ee_pose": stats(np.concatenate(task_arrays["right_ee_pose"], axis=0)),
        "observation.state.left_finger_tip_pos": stats(
            np.concatenate(task_arrays["left_finger_tip_pos"], axis=0)
        ),
        "observation.state.right_finger_tip_pos": stats(
            np.concatenate(task_arrays["right_finger_tip_pos"], axis=0)
        ),
    }

