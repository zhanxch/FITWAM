#!/usr/bin/env python3
"""Validate the four-episode S0 rollout before formal collection."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Callable

import numpy as np


EXPECTED_ACTION_KEYS = {
    "initial_state",
    "executed_actions",
    "raw_policy_actions",
    "low_pass_actions",
    "executed_is_fallback",
    "policy_query_steps",
    "policy_arrival_steps",
    "policy_latencies",
    "policy_chunks",
    "replan_steps",
    "action_horizon",
    "action_dim",
}
EXPECTED_MODEL_PROPRIO_DIM = 23
EXPECTED_WATER_PLANT_RAW_STATE_DIM = 38


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate S0 sanity summary, videos, and action archives."
    )
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--expected-episodes", type=int, default=4)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.incomplete-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def require_finite(name: str, value: np.ndarray) -> None:
    if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
        raise ValueError(f"{name} contains NaN or infinite values")


def scalar_int(archive: Any, key: str) -> int:
    value = np.asarray(archive[key])
    if value.size != 1:
        raise ValueError(f"{key} must be scalar, got shape {value.shape}")
    return int(value.reshape(()))


def inspect_action_archive(path: Path, *, expected_steps: int) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(f"Missing action archive: {path}")
    with np.load(path, allow_pickle=False) as archive:
        missing = sorted(EXPECTED_ACTION_KEYS.difference(archive.files))
        if missing:
            raise ValueError(f"{path}: missing action fields {missing}")

        action_dim = scalar_int(archive, "action_dim")
        action_horizon = scalar_int(archive, "action_horizon")
        replan_steps = scalar_int(archive, "replan_steps")
        if (action_dim, action_horizon, replan_steps) != (22, 32, 25):
            raise ValueError(
                f"{path}: expected action_dim/horizon/replan=22/32/25, got "
                f"{action_dim}/{action_horizon}/{replan_steps}"
            )

        initial_state = np.asarray(archive["initial_state"])
        expected_initial_state_shape = (EXPECTED_WATER_PLANT_RAW_STATE_DIM,)
        if initial_state.shape != expected_initial_state_shape:
            raise ValueError(
                f"{path}: raw initial_state must have shape "
                f"{expected_initial_state_shape}, got {initial_state.shape}"
            )
        require_finite(f"{path}:initial_state", initial_state)

        for key in ("executed_actions", "raw_policy_actions", "low_pass_actions"):
            values = np.asarray(archive[key])
            if values.shape != (expected_steps, 22):
                raise ValueError(
                    f"{path}:{key} must have shape ({expected_steps}, 22), got {values.shape}"
                )
            require_finite(f"{path}:{key}", values)

        fallback = np.asarray(archive["executed_is_fallback"])
        if fallback.shape != (expected_steps,):
            raise ValueError(
                f"{path}:executed_is_fallback must have shape ({expected_steps},), "
                f"got {fallback.shape}"
            )

        chunks = np.asarray(archive["policy_chunks"])
        if chunks.ndim != 3 or chunks.shape[0] < 1 or chunks.shape[1:] != (32, 22):
            raise ValueError(
                f"{path}:policy_chunks must have shape (N, 32, 22), got {chunks.shape}"
            )
        require_finite(f"{path}:policy_chunks", chunks)

        query_steps = np.asarray(archive["policy_query_steps"])
        arrival_steps = np.asarray(archive["policy_arrival_steps"])
        latencies = np.asarray(archive["policy_latencies"])
        expected_queries = (chunks.shape[0],)
        for key, values in (
            ("policy_query_steps", query_steps),
            ("policy_arrival_steps", arrival_steps),
            ("policy_latencies", latencies),
        ):
            if values.shape != expected_queries:
                raise ValueError(
                    f"{path}:{key} must have shape {expected_queries}, got {values.shape}"
                )
            require_finite(f"{path}:{key}", values)
        if np.any(latencies < 0):
            raise ValueError(f"{path}:policy_latencies contains negative values")

    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "steps": expected_steps,
        "raw_initial_state_dim": int(initial_state.shape[0]),
        "action_dim": action_dim,
        "action_horizon": action_horizon,
        "replan_steps": replan_steps,
        "policy_queries": int(chunks.shape[0]),
    }


def inspect_video(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(f"Missing video: {path}")
    import av

    frame_count = 0
    height = 0
    width = 0
    maximum_dynamic_range = 0
    with av.open(str(path), mode="r") as container:
        for frame in container.decode(video=0):
            image = frame.to_ndarray(format="rgb24")
            if image.ndim != 3 or image.shape[2] != 3:
                raise ValueError(f"{path}: invalid decoded RGB shape {image.shape}")
            frame_count += 1
            height, width = int(image.shape[0]), int(image.shape[1])
            maximum_dynamic_range = max(
                maximum_dynamic_range,
                int(image.max()) - int(image.min()),
            )
    if frame_count < 1:
        raise ValueError(f"{path}: no decodable frames")
    if maximum_dynamic_range < 5:
        raise ValueError(f"{path}: decoded frames are blank or nearly uniform")
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "decoded_frames": frame_count,
        "height": height,
        "width": width,
        "maximum_dynamic_range": maximum_dynamic_range,
    }


def resolve_artifact(raw_path: Any, *, summary_path: Path, field: str) -> Path:
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError(f"{field} must be a non-empty path")
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = summary_path.parent / path
    return path.resolve()


def validate_sanity_outputs(
    summary_path: Path,
    protocol_path: Path,
    *,
    expected_episodes: int,
    video_inspector: Callable[[Path], dict[str, Any]] = inspect_video,
) -> dict[str, Any]:
    summary_path = summary_path.expanduser().resolve()
    protocol_path = protocol_path.expanduser().resolve()
    summary = load_json(summary_path)
    protocol = load_json(protocol_path)

    if int(summary.get("total_episodes", -1)) != expected_episodes:
        raise ValueError(
            f"Expected {expected_episodes} sanity episodes, got "
            f"{summary.get('total_episodes')}"
        )
    if summary.get("save_actions") is not True:
        raise ValueError("S0 sanity must save action archives")
    if summary.get("randomize") is not False or summary.get("randomize_dynamics") is not False:
        raise ValueError("S0 sanity must disable object and dynamics randomization")
    if summary.get("action_clip") is not False:
        raise ValueError("S0 sanity must disable action clipping")

    config = protocol["model"]["config"]
    if config.get("camera_keys") != ["front", "wrist"]:
        raise ValueError(
            f"S0 protocol must use front+wrist cameras, got {config.get('camera_keys')}"
        )
    if int(config.get("proprio_dim", -1)) != EXPECTED_MODEL_PROPRIO_DIM:
        raise ValueError(
            f"S0 protocol model proprio_dim must be {EXPECTED_MODEL_PROPRIO_DIM}, "
            f"got {config.get('proprio_dim')}"
        )
    collection = protocol["collection"]
    base_seed = int(collection["base_seed"])
    expected_seeds = list(range(base_seed, base_seed + expected_episodes))

    tasks = summary.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != 1:
        raise ValueError("S0 sanity summary must contain exactly one task")
    task = tasks[0]
    if task.get("env_name") != "water_plant":
        raise ValueError(f"Expected water_plant, got {task.get('env_name')}")
    episodes = task.get("episode_results")
    if not isinstance(episodes, list) or len(episodes) != expected_episodes:
        raise ValueError(
            f"Expected {expected_episodes} episode results, got "
            f"{len(episodes) if isinstance(episodes, list) else type(episodes).__name__}"
        )
    actual_seeds = sorted(int(episode["seed"]) for episode in episodes)
    if actual_seeds != expected_seeds:
        raise ValueError(f"Expected seeds {expected_seeds}, got {actual_seeds}")

    reports = []
    for episode in episodes:
        steps = int(episode.get("steps", -1))
        if steps < 1 or steps > int(collection["max_env_steps"]):
            raise ValueError(f"Invalid episode step count: {steps}")
        video_path = resolve_artifact(
            episode.get("video_path"),
            summary_path=summary_path,
            field="video_path",
        )
        actions_path = resolve_artifact(
            episode.get("actions_path"),
            summary_path=summary_path,
            field="actions_path",
        )
        video_report = video_inspector(video_path)
        if int(video_report["decoded_frames"]) != steps + 1:
            raise ValueError(
                f"{video_path}: expected {steps + 1} frames for {steps} steps, "
                f"got {video_report['decoded_frames']}"
            )
        reports.append(
            {
                "episode": int(episode["episode"]),
                "seed": int(episode["seed"]),
                "success": bool(episode["success"]),
                "steps": steps,
                "video": video_report,
                "actions": inspect_action_archive(actions_path, expected_steps=steps),
            }
        )

    return {
        "status": "valid",
        "summary": str(summary_path),
        "protocol": str(protocol_path),
        "expected_episodes": expected_episodes,
        "camera_keys": ["front", "wrist"],
        "model_proprio_dim": EXPECTED_MODEL_PROPRIO_DIM,
        "raw_initial_state_dim": EXPECTED_WATER_PLANT_RAW_STATE_DIM,
        "episodes": reports,
    }


def main() -> None:
    args = parse_args()
    report = validate_sanity_outputs(
        args.summary,
        args.protocol,
        expected_episodes=args.expected_episodes,
    )
    atomic_write_json(args.report.expanduser().resolve(), report)
    print(f"[s0-sanity-validation] status={report['status']}")
    print(f"[s0-sanity-validation] report={args.report.expanduser().resolve()}")


if __name__ == "__main__":
    main()
