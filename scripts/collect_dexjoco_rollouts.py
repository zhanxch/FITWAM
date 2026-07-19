#!/usr/bin/env python3
"""Collect DexJoCo rollouts as a two-camera LeRobot dataset.

By default this keeps the historical water_plant behavior: save failed rollouts only. With
``--save-all-trajectories`` it saves both successes and failures into one dataset.
Every saved rollout has a structured outcome row in ``meta/episode_outcomes.jsonl``.
The historical failure marker in task text remains the default; formal collection can
instead keep clean task text with ``--outcome-task-mode clean``.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import sys
import tempfile
import time
from collections import deque
from pathlib import Path
from typing import Any

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

os.environ.setdefault("MUJOCO_GL", "egl")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
SRC_ROOT = PROJECT_ROOT / "src"
DEXJOCO_ASYNC_DIR = SCRIPTS_ROOT / "dexjoco_async"
DEXJOCO_REPO_ROOT = PROJECT_ROOT / "third_party" / "dexjoco"
DEXJOCO_PY_ROOT = DEXJOCO_REPO_ROOT / "dexjoco"
for path in (
    PROJECT_ROOT,
    SRC_ROOT,
    SCRIPTS_ROOT,
    DEXJOCO_ASYNC_DIR,
    DEXJOCO_REPO_ROOT,
    DEXJOCO_PY_ROOT,
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dexjoco_fastwam_adapter import (
    DEFAULT_TASK_CONFIG_DIR,
    ActionConstraintConfig,
    DexJoCoFastWAMAdapter,
    DexJoCoFastWAMEvalEnv,
    DexJoCoTaskConfig,
    _safe_rgb_uint8,
    constrain_rotvec_action,
    load_dexjoco_eval_settings,
)
from policy_client_async import PolicyClientAsync


DEFAULT_SUCCESS_PROMPT = "Grasp the watering can and apply water to the plant."
FAILURE_PHRASE = "Failed to finish the whole process."
OUTCOME_LEDGER_NAME = "episode_outcomes.jsonl"
OUTCOME_SOURCE = "dexjoco_env"
VIDEO_KEYS = (
    ("observation.images.front", "front"),
    ("observation.images.wrist", "wrist"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect FastWAM DexJoCo rollouts in LeRobot v2.1 format."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="FastWAM training run directory containing config.yaml.",
    )
    parser.add_argument(
        "--text-embedding-cache-dir",
        type=Path,
        default=None,
        help="Optional runtime relocation of cached task contexts.",
    )
    parser.add_argument("--policy-host", type=str, default="127.0.0.1")
    parser.add_argument("--policy-port", type=int, default=5560)
    parser.add_argument("--policy-timeout-ms", type=int, default=300000)
    parser.add_argument(
        "--task-config",
        type=Path,
        default=DEFAULT_TASK_CONFIG_DIR / "water_plant.yaml",
        help="DexJoCo task yaml. Defaults to rand_obj/water_plant.yaml for compatibility.",
    )
    parser.add_argument(
        "--source-dataset",
        type=Path,
        default=PROJECT_ROOT / "data" / "dexjoco" / "dexjoco_lerobot_datasets" / "water_plant",
        help="Existing successful LeRobot dataset used as schema/template. Defaults to water_plant for compatibility.",
    )
    parser.add_argument(
        "--output-dataset",
        type=Path,
        default=PROJECT_ROOT / "data" / "dexjoco" / "rollouts" / "water_plant_failure_fastwam_2cam_text",
    )
    parser.add_argument("--target-failures", type=int, default=100)
    parser.add_argument(
        "--target-episodes",
        type=int,
        default=None,
        help="Total rollout attempts to save when --save-all-trajectories is set.",
    )
    parser.add_argument("--max-attempts", type=int, default=260)
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--replan-steps", type=int, default=None)
    parser.add_argument("--max-env-steps", type=int, default=600)
    parser.add_argument("--video-fps", type=int, default=30)
    parser.add_argument(
        "--success-prompt",
        type=str,
        default=DEFAULT_SUCCESS_PROMPT,
        help="Instruction text assigned to successful trajectories.",
    )
    parser.add_argument("--failure-phrase", type=str, default=FAILURE_PHRASE)
    parser.add_argument(
        "--save-all-trajectories",
        action="store_true",
        help="Save both successful and failed rollouts instead of keeping failures only.",
    )
    parser.add_argument(
        "--outcome-task-mode",
        choices=("task-marker", "clean"),
        default="task-marker",
        help=(
            "task-marker keeps the legacy failure phrase in failed task text; "
            "clean uses the same instruction text for successful and failed rollouts."
        ),
    )
    parser.add_argument(
        "--trim-failure-seconds",
        type=float,
        default=0.0,
        help="Drop this many seconds from the end of failed trajectories before saving.",
    )
    parser.add_argument("--randomize", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--randomize-dynamics", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--action-clip", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--clip-max-xyz-step", type=float, default=0.05)
    parser.add_argument("--clip-max-dz-down", type=float, default=0.03)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Append to an existing output dataset using collection_summary.json to continue seeds.",
    )
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            pass
    finally:
        tmp_path.unlink(missing_ok=True)


def write_json(path: Path, payload: Any) -> None:
    _atomic_write_text(
        path,
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
    )


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    if existing and not existing.endswith("\n"):
        existing += "\n"
    _atomic_write_text(path, existing + json.dumps(payload, ensure_ascii=False) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    _atomic_write_text(path, text)


def make_outcome_row(
    *,
    episode_index: int,
    success: bool,
    attempt_index: int,
    seed: int,
) -> dict[str, Any]:
    return {
        "episode_index": int(episode_index),
        "outcome": "success" if success else "failure",
        "success": bool(success),
        "attempt_index": int(attempt_index),
        "seed": int(seed),
        "source": OUTCOME_SOURCE,
    }


def validate_outcome_rows(
    rows: list[dict[str, Any]],
    *,
    expected_episode_count: int,
    attempts: list[dict[str, Any]] | None = None,
) -> None:
    by_episode: dict[int, dict[str, Any]] = {}
    required = {
        "episode_index",
        "outcome",
        "success",
        "attempt_index",
        "seed",
        "source",
    }
    for row_number, row in enumerate(rows, start=1):
        missing = required.difference(row)
        if missing:
            raise ValueError(f"Outcome row {row_number} is missing fields: {sorted(missing)}")
        episode_index = row["episode_index"]
        attempt_index = row["attempt_index"]
        seed = row["seed"]
        success = row["success"]
        if isinstance(episode_index, bool) or not isinstance(episode_index, int) or episode_index < 0:
            raise ValueError(f"Invalid episode_index in outcome row {row_number}: {episode_index!r}")
        if episode_index in by_episode:
            raise ValueError(f"Duplicate outcome row for episode_index={episode_index}")
        if isinstance(success, bool) is False:
            raise ValueError(f"Outcome row {row_number} has non-boolean success={success!r}")
        expected_outcome = "success" if success else "failure"
        if row["outcome"] != expected_outcome:
            raise ValueError(
                f"Outcome row {row_number} is inconsistent: "
                f"outcome={row['outcome']!r}, success={success!r}"
            )
        if isinstance(attempt_index, bool) or not isinstance(attempt_index, int) or attempt_index < 0:
            raise ValueError(
                f"Invalid attempt_index in outcome row {row_number}: {attempt_index!r}"
            )
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError(f"Invalid seed in outcome row {row_number}: {seed!r}")
        if row["source"] != OUTCOME_SOURCE:
            raise ValueError(
                f"Outcome row {row_number} has source={row['source']!r}; "
                f"expected {OUTCOME_SOURCE!r}"
            )
        by_episode[episode_index] = row

    expected = set(range(expected_episode_count))
    actual = set(by_episode)
    if actual != expected:
        raise ValueError(
            "Outcome ledger must contain exactly one row per saved episode: "
            f"missing={sorted(expected - actual)} extra={sorted(actual - expected)}"
        )

    if attempts is None:
        return
    attempt_rows: dict[int, dict[str, Any]] = {}
    for attempt in attempts:
        saved_episode_index = attempt.get("saved_episode_index")
        if saved_episode_index is None:
            continue
        saved_episode_index = int(saved_episode_index)
        if saved_episode_index in attempt_rows:
            raise ValueError(
                f"Attempt log maps multiple attempts to saved episode {saved_episode_index}"
            )
        attempt_rows[saved_episode_index] = attempt
    if set(attempt_rows) != expected:
        raise ValueError(
            "Attempt log and outcome ledger disagree on saved episodes: "
            f"attempt_only={sorted(set(attempt_rows) - expected)} "
            f"ledger_only={sorted(expected - set(attempt_rows))}"
        )
    for episode_index, row in by_episode.items():
        attempt = attempt_rows[episode_index]
        if (
            int(attempt["attempt_index"]) != row["attempt_index"]
            or int(attempt["seed"]) != row["seed"]
            or bool(attempt["success"]) != row["success"]
        ):
            raise ValueError(
                f"Outcome ledger row for episode {episode_index} disagrees with attempt log"
            )


def _legacy_outcome_rows_from_attempts(
    attempts: list[dict[str, Any]],
    *,
    expected_episode_count: int,
) -> list[dict[str, Any]]:
    rows = []
    for attempt in attempts:
        saved_episode_index = attempt.get("saved_episode_index")
        if saved_episode_index is None:
            continue
        rows.append(
            make_outcome_row(
                episode_index=int(saved_episode_index),
                success=bool(attempt["success"]),
                attempt_index=int(attempt["attempt_index"]),
                seed=int(attempt["seed"]),
            )
        )
    rows.sort(key=lambda row: row["episode_index"])
    validate_outcome_rows(
        rows,
        expected_episode_count=expected_episode_count,
        attempts=attempts,
    )
    return rows


def save_episode_video(frames: list[np.ndarray], path: Path, fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not frames:
        raise ValueError(f"Cannot save empty video: {path}")
    first = _safe_rgb_uint8(frames[0])
    height, width = first.shape[:2]
    container = av.open(str(path), mode="w")
    try:
        stream = container.add_stream("libx264", rate=fps)
        stream.width = int(width)
        stream.height = int(height)
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "21"}
        for frame in frames:
            video_frame = av.VideoFrame.from_ndarray(_safe_rgb_uint8(frame), format="rgb24")
            for packet in stream.encode(video_frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()


def fixed_size_float_array(values: np.ndarray, dim: int) -> pa.FixedSizeListArray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != dim:
        raise ValueError(f"Expected [N,{dim}] float array, got {values.shape}")
    flat = pa.array(values.reshape(-1), type=pa.float32())
    return pa.FixedSizeListArray.from_arrays(flat, dim)


def write_episode_parquet(
    path: Path,
    *,
    actions: np.ndarray,
    states: np.ndarray,
    timestamps: np.ndarray,
    frame_indices: np.ndarray,
    episode_indices: np.ndarray,
    global_indices: np.ndarray,
    task_indices: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "action": fixed_size_float_array(actions, 22),
            "observation.state": fixed_size_float_array(states, 23),
            "timestamp": pa.array(timestamps.astype(np.float32), type=pa.float32()),
            "frame_index": pa.array(frame_indices.astype(np.int64), type=pa.int64()),
            "episode_index": pa.array(episode_indices.astype(np.int64), type=pa.int64()),
            "index": pa.array(global_indices.astype(np.int64), type=pa.int64()),
            "task_index": pa.array(task_indices.astype(np.int64), type=pa.int64()),
        }
    )
    pq.write_table(table, path)


def load_task(task_config: Path) -> DexJoCoTaskConfig:
    with task_config.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return DexJoCoTaskConfig.from_yaml(cfg)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _load_jsonl_through_boundary(
    path: Path,
    *,
    boundary: int,
    label: str,
) -> list[dict[str, Any]]:
    if not path.exists():
        if boundary:
            raise ValueError(f"{label} is missing before committed boundary {boundary}: {path}")
        return []

    rows: list[dict[str, Any]] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    for line_number, line in enumerate(lines, start=1):
        if len(rows) == boundary:
            break
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"{label} is corrupt inside committed prefix at line {line_number}"
            ) from exc
        if not isinstance(row, dict):
            raise ValueError(f"{label} row {line_number} must be a JSON object")
        rows.append(row)

    if len(rows) < boundary:
        raise ValueError(
            f"{label} has {len(rows)} rows, shorter than committed boundary {boundary}"
        )
    return rows


def _validate_indexed_prefix(
    rows: list[dict[str, Any]],
    *,
    boundary: int,
    label: str,
) -> list[dict[str, Any]]:
    prefix = rows[:boundary]
    for expected_index, row in enumerate(prefix):
        actual_index = row.get("episode_index")
        if actual_index != expected_index:
            raise ValueError(
                f"{label} committed row {expected_index} has "
                f"episode_index={actual_index!r}"
            )
    return prefix


def _parquet_num_rows(path: Path) -> int:
    return int(pq.read_metadata(path).num_rows)


def _episode_file_index(path: Path) -> int | None:
    stem = path.stem
    if not stem.startswith("episode_"):
        return None
    suffix = stem.removeprefix("episode_")
    return int(suffix) if suffix.isdigit() else None


def _reconcile_episode_files(
    output_dataset: Path,
    info: dict[str, Any],
    episodes: list[dict[str, Any]],
    *,
    boundary: int,
) -> None:
    for episode_index, row in enumerate(episodes):
        chunk = episode_index // int(info["chunks_size"])
        parquet_path = output_dataset / info["data_path"].format(
            episode_chunk=chunk,
            episode_index=episode_index,
        )
        if not parquet_path.is_file() or parquet_path.stat().st_size <= 0:
            raise ValueError(
                f"Committed episode {episode_index} is missing parquet: {parquet_path}"
            )
        expected_length = int(row["length"])
        actual_length = _parquet_num_rows(parquet_path)
        if actual_length != expected_length:
            raise ValueError(
                f"Committed episode {episode_index} parquet length mismatch: "
                f"{actual_length} != {expected_length}"
            )
        for video_key, _ in VIDEO_KEYS:
            video_path = output_dataset / info["video_path"].format(
                episode_chunk=chunk,
                video_key=video_key,
                episode_index=episode_index,
            )
            if not video_path.is_file() or video_path.stat().st_size <= 0:
                raise ValueError(
                    f"Committed episode {episode_index} is missing video {video_key}: "
                    f"{video_path}"
                )

    artifact_roots = (
        (output_dataset / "data", "*.parquet"),
        (output_dataset / "videos", "*.mp4"),
    )
    for root, pattern in artifact_roots:
        if not root.exists():
            continue
        for path in root.rglob(pattern):
            episode_index = _episode_file_index(path)
            if episode_index is not None and episode_index >= boundary:
                path.unlink()


def _next_attempt_index(attempts: list[dict[str, Any]]) -> int:
    if not attempts:
        return 0
    indices = [int(item["attempt_index"]) for item in attempts]
    if len(indices) != len(set(indices)):
        raise ValueError("collection_summary attempt_log has duplicate attempt_index values")
    if indices != sorted(indices):
        raise ValueError("collection_summary attempt_log is not ordered by attempt_index")
    return indices[-1] + 1


def _validate_summary_boundary(
    summary: dict[str, Any],
    *,
    requested_base_seed: int | None,
    expected_mode: str,
    expected_outcome_task_mode: str,
) -> tuple[int, list[dict[str, Any]]]:
    if "mode" in summary and summary["mode"] != expected_mode:
        raise ValueError(
            f"Resume mode mismatch: summary={summary['mode']!r}, requested={expected_mode!r}"
        )
    if (
        "outcome_task_mode" in summary
        and summary["outcome_task_mode"] != expected_outcome_task_mode
    ):
        raise ValueError(
            "Resume outcome_task_mode mismatch: "
            f"summary={summary['outcome_task_mode']!r}, "
            f"requested={expected_outcome_task_mode!r}"
        )
    attempts = summary.get("attempt_log", [])
    if not isinstance(attempts, list) or any(not isinstance(row, dict) for row in attempts):
        raise ValueError("collection_summary.attempt_log must be a list of objects")

    saved_indices = [
        int(row["saved_episode_index"])
        for row in attempts
        if row.get("saved_episode_index") is not None
    ]
    if "episodes" in summary:
        boundary = int(summary["episodes"])
    else:
        boundary = len(saved_indices)
    if boundary < 0:
        raise ValueError(f"Invalid collection_summary episode boundary: {boundary}")
    if saved_indices != list(range(boundary)):
        raise ValueError(
            "collection_summary attempt_log does not map exactly to committed episodes "
            f"0..{boundary - 1}: {saved_indices}"
        )
    if "attempts" in summary and int(summary["attempts"]) != len(attempts):
        raise ValueError(
            "collection_summary attempts count disagrees with attempt_log: "
            f"{summary['attempts']} != {len(attempts)}"
        )
    if expected_mode == "save_all" and len(saved_indices) != len(attempts):
        raise ValueError("save_all summary contains an attempt without a saved episode")

    saved_attempts = [
        row for row in attempts if row.get("saved_episode_index") is not None
    ]
    committed_failures = sum(not bool(row["success"]) for row in saved_attempts)
    committed_successes = sum(bool(row["success"]) for row in saved_attempts)
    if "failures" in summary and int(summary["failures"]) != committed_failures:
        raise ValueError(
            "collection_summary failures count disagrees with committed attempts: "
            f"{summary['failures']} != {committed_failures}"
        )
    if (
        "successes_saved" in summary
        and int(summary["successes_saved"]) != committed_successes
    ):
        raise ValueError(
            "collection_summary successes_saved count disagrees with committed attempts: "
            f"{summary['successes_saved']} != {committed_successes}"
        )

    next_attempt = _next_attempt_index(attempts)
    summary_base_seed = summary.get("base_seed")
    if summary_base_seed is not None:
        summary_base_seed = int(summary_base_seed)
        if requested_base_seed is not None and summary_base_seed != requested_base_seed:
            raise ValueError(
                f"Resume seed mismatch: summary base_seed={summary_base_seed}, "
                f"requested={requested_base_seed}"
            )
    elif attempts:
        inferred = {
            int(row["seed"]) - int(row["attempt_index"])
            for row in attempts
        }
        if len(inferred) != 1:
            raise ValueError("Cannot infer one base seed from collection_summary attempt_log")
        summary_base_seed = inferred.pop()
        if requested_base_seed is not None and summary_base_seed != requested_base_seed:
            raise ValueError(
                f"Resume seed mismatch: inferred base_seed={summary_base_seed}, "
                f"requested={requested_base_seed}"
            )
    elif requested_base_seed is not None:
        summary_base_seed = requested_base_seed

    if summary_base_seed is not None:
        for row in attempts:
            expected_seed = int(summary_base_seed) + int(row["attempt_index"])
            if int(row["seed"]) != expected_seed:
                raise ValueError(
                    f"Attempt {row['attempt_index']} seed mismatch: "
                    f"{row['seed']} != {expected_seed}"
                )
    if "next_attempt_index" in summary and int(summary["next_attempt_index"]) != next_attempt:
        raise ValueError(
            "collection_summary next_attempt_index disagrees with attempt_log: "
            f"{summary['next_attempt_index']} != {next_attempt}"
        )
    return boundary, attempts


def reconcile_resume_dataset(
    output_dataset: Path,
    *,
    requested_base_seed: int | None,
    total_tasks: int,
    expected_mode: str,
    expected_outcome_task_mode: str,
) -> tuple[dict[str, Any], int, int, list[dict[str, dict]], list[dict[str, Any]]]:
    summary_path = output_dataset / "collection_summary.json"
    if not summary_path.exists():
        raise ValueError(
            "Cannot safely resume without collection_summary.json commit boundary"
        )
    summary = read_json(summary_path)
    boundary, attempts = _validate_summary_boundary(
        summary,
        requested_base_seed=requested_base_seed,
        expected_mode=expected_mode,
        expected_outcome_task_mode=expected_outcome_task_mode,
    )
    info = read_json(output_dataset / "meta" / "info.json")

    episodes_path = output_dataset / "meta" / "episodes.jsonl"
    stats_path = output_dataset / "meta" / "episodes_stats.jsonl"
    ledger_path = output_dataset / "meta" / OUTCOME_LEDGER_NAME
    episodes = _validate_indexed_prefix(
        _load_jsonl_through_boundary(
            episodes_path,
            boundary=boundary,
            label="episodes.jsonl",
        ),
        boundary=boundary,
        label="episodes.jsonl",
    )
    stats_rows = _validate_indexed_prefix(
        _load_jsonl_through_boundary(
            stats_path,
            boundary=boundary,
            label="episodes_stats.jsonl",
        ),
        boundary=boundary,
        label="episodes_stats.jsonl",
    )

    if ledger_path.exists():
        outcome_rows = _validate_indexed_prefix(
            _load_jsonl_through_boundary(
                ledger_path,
                boundary=boundary,
                label=OUTCOME_LEDGER_NAME,
            ),
            boundary=boundary,
            label=OUTCOME_LEDGER_NAME,
        )
    elif boundary:
        outcome_rows = _legacy_outcome_rows_from_attempts(
            attempts,
            expected_episode_count=boundary,
        )
    else:
        outcome_rows = []
    validate_outcome_rows(
        outcome_rows,
        expected_episode_count=boundary,
        attempts=attempts,
    )

    _reconcile_episode_files(
        output_dataset,
        info,
        episodes,
        boundary=boundary,
    )
    write_jsonl(episodes_path, episodes)
    write_jsonl(stats_path, stats_rows)
    write_jsonl(ledger_path, outcome_rows)

    total_frames = sum(int(row["length"]) for row in episodes)
    if "frames" in summary and int(summary["frames"]) != total_frames:
        raise ValueError(
            f"collection_summary frames={summary['frames']} disagrees with "
            f"committed episode lengths={total_frames}"
        )
    episode_stats = [cast_stats_to_numpy(item["stats"]) for item in stats_rows]
    stats_file = output_dataset / "meta" / "stats.json"
    if episode_stats:
        write_json(stats_file, serialize_dict(aggregate_stats(episode_stats)))
    else:
        stats_file.unlink(missing_ok=True)
    update_info(
        output_dataset,
        info,
        num_episodes=boundary,
        total_frames=total_frames,
        total_tasks=total_tasks,
    )
    return info, boundary, total_frames, episode_stats, attempts


def _feature_stats(array: np.ndarray, *, keepdims: bool) -> dict[str, np.ndarray]:
    return {
        "min": np.min(array, axis=0, keepdims=keepdims),
        "max": np.max(array, axis=0, keepdims=keepdims),
        "mean": np.mean(array, axis=0, keepdims=keepdims),
        "std": np.std(array, axis=0, keepdims=keepdims),
        "count": np.array([len(array)], dtype=np.int64),
    }


def compute_episode_stats(
    episode_data: dict[str, np.ndarray],
    features: dict[str, Any],
    *,
    is_compute_episode_stats_image: bool = False,
) -> dict[str, dict[str, np.ndarray]]:
    del is_compute_episode_stats_image
    stats: dict[str, dict[str, np.ndarray]] = {}
    for key, data in episode_data.items():
        spec = features.get(key)
        if spec is None or spec.get("dtype") in {"image", "video", "string"}:
            continue
        arr = np.asarray(data)
        stats[key] = _feature_stats(arr, keepdims=arr.ndim == 1)
    return stats


def _aggregate_feature_stats(items: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    counts = np.stack([item["count"] for item in items], axis=0).squeeze(-1)
    total = counts.sum(axis=0, keepdims=True)
    mean = sum(item["mean"] * item["count"] for item in items) / total
    var = sum((item["std"] ** 2 + (item["mean"] - mean) ** 2) * item["count"] for item in items) / total
    return {
        "min": np.min(np.stack([item["min"] for item in items], axis=0), axis=0, keepdims=True),
        "max": np.max(np.stack([item["max"] for item in items], axis=0), axis=0, keepdims=True),
        "mean": mean,
        "std": np.sqrt(var),
        "count": total,
    }


def aggregate_stats(stats_list: list[dict[str, dict[str, np.ndarray]]]) -> dict[str, dict[str, np.ndarray]]:
    if not stats_list:
        return {}
    keys = sorted({key for stats in stats_list for key in stats.keys()})
    return {
        key: _aggregate_feature_stats([stats[key] for stats in stats_list if key in stats])
        for key in keys
    }


def serialize_dict(data: Any) -> Any:
    if isinstance(data, dict):
        return {key: serialize_dict(value) for key, value in data.items()}
    if isinstance(data, np.ndarray):
        return data.tolist()
    return data


def cast_stats_to_numpy(data: Any) -> Any:
    if isinstance(data, dict):
        return {key: cast_stats_to_numpy(value) for key, value in data.items()}
    if isinstance(data, list):
        return np.asarray(data)
    return data


def get_video_info(path: Path) -> dict[str, Any]:
    container = av.open(str(path), mode="r")
    try:
        stream = container.streams.video[0]
        fps = float(stream.average_rate) if stream.average_rate is not None else None
        return {
            "video.fps": fps,
            "video.height": int(stream.height),
            "video.width": int(stream.width),
            "video.codec": stream.codec_context.name,
        }
    finally:
        container.close()


def prepare_dataset(
    source_dataset: Path,
    output_dataset: Path,
    success_task: str,
    failure_task: str,
    *,
    overwrite: bool,
    resume: bool,
    save_all_trajectories: bool,
    outcome_task_mode: str,
    base_seed: int | None = None,
) -> tuple[dict, int, int, list[dict[str, dict]], list[dict[str, Any]]]:
    if overwrite and resume:
        raise ValueError("--overwrite and --resume are mutually exclusive")

    if resume:
        if not output_dataset.exists():
            raise FileNotFoundError(f"Cannot resume missing dataset: {output_dataset}")
        info = read_json(output_dataset / "meta" / "info.json")
        task_lines = load_jsonl(output_dataset / "meta" / "tasks.jsonl")
        if outcome_task_mode == "clean":
            expected_tasks = [success_task]
        else:
            expected_tasks = (
                [success_task, failure_task] if save_all_trajectories else [failure_task]
            )
        existing_tasks = [item.get("task") for item in task_lines]
        if existing_tasks != expected_tasks:
            raise ValueError(
                "Existing dataset task text does not match requested collection mode: "
                f"existing={existing_tasks!r} expected={expected_tasks!r}"
            )
        total_tasks = 1 if outcome_task_mode == "clean" else (
            2 if save_all_trajectories else 1
        )
        return reconcile_resume_dataset(
            output_dataset,
            requested_base_seed=base_seed,
            total_tasks=total_tasks,
            expected_mode="save_all" if save_all_trajectories else "failures_only",
            expected_outcome_task_mode=outcome_task_mode,
        )

    if output_dataset.exists():
        if not overwrite:
            raise FileExistsError(f"Output dataset already exists: {output_dataset}")
        shutil.rmtree(output_dataset)

    (output_dataset / "meta").mkdir(parents=True, exist_ok=False)
    (output_dataset / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    for video_key, _ in VIDEO_KEYS:
        (output_dataset / "videos" / "chunk-000" / video_key).mkdir(parents=True, exist_ok=True)

    info = copy.deepcopy(read_json(source_dataset / "meta" / "info.json"))
    info["total_episodes"] = 0
    info["total_frames"] = 0
    total_tasks = 1 if outcome_task_mode == "clean" else (2 if save_all_trajectories else 1)
    info["total_tasks"] = total_tasks
    info["total_videos"] = 0
    info["total_chunks"] = 1
    info["splits"] = {"train": "0:0"}
    info["fps"] = int(info.get("fps", 30))
    info["data_path"] = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
    info["video_path"] = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
    write_json(output_dataset / "meta" / "info.json", info)
    if outcome_task_mode == "clean":
        append_jsonl(output_dataset / "meta" / "tasks.jsonl", {"task_index": 0, "task": success_task})
    elif save_all_trajectories:
        append_jsonl(output_dataset / "meta" / "tasks.jsonl", {"task_index": 0, "task": success_task})
        append_jsonl(output_dataset / "meta" / "tasks.jsonl", {"task_index": 1, "task": failure_task})
    else:
        append_jsonl(output_dataset / "meta" / "tasks.jsonl", {"task_index": 0, "task": failure_task})
    write_jsonl(output_dataset / "meta" / OUTCOME_LEDGER_NAME, [])

    modality_path = source_dataset / "meta" / "modality.json"
    if modality_path.exists():
        shutil.copy2(modality_path, output_dataset / "meta" / "modality.json")

    return info, 0, 0, [], []


def update_info(
    output_dataset: Path,
    info: dict,
    *,
    num_episodes: int,
    total_frames: int,
    total_tasks: int | None = None,
) -> None:
    info["total_episodes"] = int(num_episodes)
    info["total_frames"] = int(total_frames)
    if total_tasks is not None:
        info["total_tasks"] = int(total_tasks)
    info["total_videos"] = int(num_episodes * len(VIDEO_KEYS))
    info["total_chunks"] = 1 if num_episodes > 0 else 0
    info["splits"] = {"train": f"0:{num_episodes}"}
    if num_episodes > 0:
        for video_key, _ in VIDEO_KEYS:
            video_path = output_dataset / info["video_path"].format(
                episode_chunk=0,
                video_key=video_key,
                episode_index=0,
            )
            try:
                info["features"][video_key]["info"] = get_video_info(video_path)
            except Exception as exc:
                print(f"[warn] get_video_info failed for {video_path}: {exc}", flush=True)
    write_json(output_dataset / "meta" / "info.json", info)


def run_attempt(
    task: DexJoCoTaskConfig,
    *,
    policy: PolicyClientAsync,
    adapter: DexJoCoFastWAMAdapter,
    seed: int,
    replan_steps: int,
    max_env_steps: int,
    randomize: bool,
    randomize_dynamics: bool,
    action_clip_config: ActionConstraintConfig | None,
) -> dict[str, Any]:
    env = DexJoCoFastWAMEvalEnv(
        task,
        seed=seed,
        randomize=randomize,
        randomize_dynamics=randomize_dynamics,
    )
    action_queue: deque[np.ndarray] = deque()
    frames: dict[str, list[np.ndarray]] = {video_key: [] for video_key, _ in VIDEO_KEYS}
    actions: list[np.ndarray] = []
    states: list[np.ndarray] = []
    policy_query_steps: list[int] = []
    t0 = time.perf_counter()

    try:
        env.reset()
        policy.reset()
        env.click_mouse_warmup()

        steps = 0
        while steps < max_env_steps:
            if not action_queue:
                policy_obs = env.build_policy_obs(adapter)
                response = policy.get_action(policy_obs)
                chunk = adapter.parse_policy_response(response)
                n_exec = max(1, min(replan_steps, chunk.shape[0]))
                policy_query_steps.append(steps)
                for idx in range(n_exec):
                    action_queue.append(chunk[idx])

            rotvec_action = np.asarray(action_queue.popleft(), dtype=np.float32)
            if action_clip_config is not None:
                current_state = np.asarray(env._latest_obs["state"], dtype=np.float32).reshape(-1)
                rotvec_action = constrain_rotvec_action(
                    rotvec_action,
                    current_state,
                    dual_arm=env.task.dual_arm,
                    config=action_clip_config,
                )

            current_obs = env._latest_obs
            states.append(np.asarray(current_obs["state"], dtype=np.float32).reshape(-1)[:23])
            actions.append(rotvec_action.reshape(-1)[:22].astype(np.float32))
            for video_key, env_key in VIDEO_KEYS:
                frames[video_key].append(_safe_rgb_uint8(current_obs[env_key]))

            env.step_rotvec(rotvec_action)
            steps += 1
            if env.is_done:
                break

        return {
            "success": bool(env.is_success),
            "done": bool(env.is_done),
            "steps": int(steps),
            "elapsed_s": time.perf_counter() - t0,
            "actions": np.stack(actions).astype(np.float32) if actions else np.zeros((0, 22), dtype=np.float32),
            "states": np.stack(states).astype(np.float32) if states else np.zeros((0, 23), dtype=np.float32),
            "frames": frames,
            "policy_query_steps": policy_query_steps,
        }
    finally:
        env.close()


def trim_episode_tail(episode: dict[str, Any], trim_steps: int) -> dict[str, Any]:
    if trim_steps <= 0:
        return episode
    length = int(episode["actions"].shape[0])
    keep = max(1, length - trim_steps)
    trimmed = dict(episode)
    trimmed["actions"] = episode["actions"][:keep]
    trimmed["states"] = episode["states"][:keep]
    trimmed["frames"] = {
        video_key: list(frames[:keep]) for video_key, frames in episode["frames"].items()
    }
    trimmed["steps"] = keep
    trimmed["trimmed_tail_steps"] = int(length - keep)
    return trimmed


def save_lerobot_episode(
    output_dataset: Path,
    info: dict,
    stats_list: list[dict[str, dict]],
    *,
    episode_index: int,
    global_start_index: int,
    episode: dict[str, Any],
    task_text: str,
    task_index: int,
    fps: int,
) -> int:
    length = int(episode["actions"].shape[0])
    if length <= 0:
        raise ValueError("Cannot save empty episode")

    chunk = episode_index // int(info["chunks_size"])
    actions = np.asarray(episode["actions"], dtype=np.float32)
    states = np.asarray(episode["states"], dtype=np.float32)
    timestamps = np.arange(length, dtype=np.float32) / float(fps)
    frame_indices = np.arange(length, dtype=np.int64)
    episode_indices = np.full((length,), episode_index, dtype=np.int64)
    global_indices = np.arange(global_start_index, global_start_index + length, dtype=np.int64)
    task_indices = np.full((length,), int(task_index), dtype=np.int64)

    for video_key, _ in VIDEO_KEYS:
        video_path = output_dataset / info["video_path"].format(
            episode_chunk=chunk,
            video_key=video_key,
            episode_index=episode_index,
        )
        save_episode_video(episode["frames"][video_key], video_path, fps)

    parquet_path = output_dataset / info["data_path"].format(
        episode_chunk=chunk,
        episode_index=episode_index,
    )
    write_episode_parquet(
        parquet_path,
        actions=actions,
        states=states,
        timestamps=timestamps,
        frame_indices=frame_indices,
        episode_indices=episode_indices,
        global_indices=global_indices,
        task_indices=task_indices,
    )

    append_jsonl(
        output_dataset / "meta" / "episodes.jsonl",
        {
            "episode_index": episode_index,
            "tasks": [task_text],
            "length": length,
        },
    )

    episode_data_for_stats = {
        "action": actions,
        "observation.state": states,
        "timestamp": timestamps,
        "frame_index": frame_indices,
        "episode_index": episode_indices,
        "index": global_indices,
        "task_index": task_indices,
    }
    ep_stats = compute_episode_stats(
        episode_data_for_stats,
        info["features"],
        is_compute_episode_stats_image=False,
    )
    stats_list.append(ep_stats)
    append_jsonl(
        output_dataset / "meta" / "episodes_stats.jsonl",
        {
            "episode_index": episode_index,
            "stats": serialize_dict(ep_stats),
        },
    )
    return length


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    source_dataset = args.source_dataset.expanduser().resolve()
    output_dataset = args.output_dataset.expanduser().resolve()
    task_config = args.task_config.expanduser().resolve()

    if not source_dataset.exists():
        raise FileNotFoundError(source_dataset)
    if not task_config.exists():
        raise FileNotFoundError(task_config)

    source_info = read_json(source_dataset / "meta" / "info.json")
    fps = int(args.video_fps or source_info.get("fps", 30))
    base_task = args.success_prompt.strip()
    if not base_task:
        raise ValueError("--success-prompt must not be empty")
    failure_task = f"{base_task} {args.failure_phrase.strip()}".strip()
    save_all = bool(args.save_all_trajectories)
    target_episodes = int(args.target_episodes or args.max_attempts)
    if save_all and args.target_episodes is None:
        target_episodes = int(args.max_attempts)
    if save_all and args.max_attempts < target_episodes:
        args.max_attempts = target_episodes
    info, saved_episodes, global_index, stats_list, attempts = prepare_dataset(
        source_dataset,
        output_dataset,
        base_task,
        failure_task,
        overwrite=args.overwrite,
        resume=args.resume,
        save_all_trajectories=save_all,
        outcome_task_mode=args.outcome_task_mode,
        base_seed=args.seed,
    )
    total_tasks = 1 if args.outcome_task_mode == "clean" else (2 if save_all else 1)
    if save_all:
        failures = sum(
            1 for item in attempts if not item.get("success", False) and item.get("saved_episode_index") is not None
        )
        successes_saved = sum(
            1 for item in attempts if item.get("success", False) and item.get("saved_episode_index") is not None
        )
    else:
        failures = saved_episodes
        successes_saved = 0

    eval_settings = load_dexjoco_eval_settings(
        run_dir,
        text_embedding_cache_dir_override=args.text_embedding_cache_dir,
    )
    adapter = DexJoCoFastWAMAdapter(eval_settings)
    replan_steps = args.replan_steps
    if replan_steps is None:
        replan_steps = max(1, int(0.8 * adapter.action_horizon))

    task = load_task(task_config)
    action_clip_config = None
    if args.action_clip:
        action_clip_config = ActionConstraintConfig(
            max_xyz_step=args.clip_max_xyz_step,
            max_dz_down=args.clip_max_dz_down,
            clip_to_dataset_bounds=False,
        )

    print(f"[collect] run_dir={run_dir}", flush=True)
    print(f"[collect] output_dataset={output_dataset}", flush=True)
    if save_all:
        print(f"[collect] mode=save_all target_episodes={target_episodes} max_attempts={args.max_attempts}", flush=True)
    else:
        print(f"[collect] mode=failures_only target_failures={args.target_failures} max_attempts={args.max_attempts}", flush=True)
    if args.resume:
        print(
            f"[collect] resume existing episodes={saved_episodes} failures={failures} "
            f"attempts={len(attempts)} frames={global_index}",
            flush=True,
        )
    print(f"[collect] policy={args.policy_host}:{args.policy_port} replan_steps={replan_steps}", flush=True)
    print(f"[collect] outcome_task_mode={args.outcome_task_mode}", flush=True)
    print(f"[collect] failure_task={failure_task}", flush=True)

    def persist_summary(status: str) -> None:
        write_json(
            output_dataset / "collection_summary.json",
            {
                "status": status,
                "mode": "save_all" if save_all else "failures_only",
                "outcome_task_mode": args.outcome_task_mode,
                "target_episodes": target_episodes if save_all else None,
                "target_failures": args.target_failures,
                "max_attempts": args.max_attempts,
                "base_seed": int(args.seed),
                "next_attempt_index": _next_attempt_index(attempts),
                "episodes": saved_episodes,
                "frames": global_index,
                "failures": failures,
                "successes_saved": successes_saved,
                "attempts": len(attempts),
                "successes_discarded": sum(
                    1
                    for item in attempts
                    if item["success"] and item.get("saved_episode_index") is None
                ),
                "success_task": base_task,
                "failure_task": failure_task,
                "output_dataset": str(output_dataset),
                "attempt_log": attempts,
            },
        )

    persist_summary("running")
    policy = PolicyClientAsync(
        host=args.policy_host,
        port=args.policy_port,
        timeout_ms=args.policy_timeout_ms,
        identity=b"dexjoco-collect",
    )
    if not policy.ping():
        raise RuntimeError(f"Policy server ping failed at {args.policy_host}:{args.policy_port}")

    try:
        for attempt_idx in range(_next_attempt_index(attempts), args.max_attempts):
            if save_all and saved_episodes >= target_episodes:
                break
            if (not save_all) and failures >= args.target_failures:
                break
            seed = int(args.seed + attempt_idx)
            print(f"[collect] attempt {attempt_idx + 1}/{args.max_attempts} seed={seed}", flush=True)
            episode = run_attempt(
                task,
                policy=policy,
                adapter=adapter,
                seed=seed,
                replan_steps=replan_steps,
                max_env_steps=args.max_env_steps,
                randomize=args.randomize,
                randomize_dynamics=args.randomize_dynamics,
                action_clip_config=action_clip_config,
            )
            attempts.append(
                {
                    "attempt_index": attempt_idx,
                    "seed": seed,
                    "success": bool(episode["success"]),
                    "done": bool(episode["done"]),
                    "steps": int(episode["steps"]),
                    "elapsed_s": float(episode["elapsed_s"]),
                    "saved_failure_index": failures if (not save_all and not episode["success"]) else None,
                    "saved_episode_index": saved_episodes if (save_all or not episode["success"]) else None,
                }
            )
            print(
                f"[collect] result success={episode['success']} done={episode['done']} "
                f"steps={episode['steps']} elapsed={episode['elapsed_s']:.1f}s",
                flush=True,
            )
            if episode["success"] and not save_all:
                persist_summary("running")
                continue

            if args.outcome_task_mode == "clean":
                task_text = base_task
                task_index = 0
            else:
                task_text = base_task if episode["success"] else failure_task
                task_index = 0 if (save_all and episode["success"]) else (1 if save_all else 0)
            if not episode["success"]:
                trim_steps = int(round(float(args.trim_failure_seconds) * float(fps)))
                episode = trim_episode_tail(episode, trim_steps)
                if trim_steps > 0:
                    attempts[-1]["trim_failure_seconds"] = float(args.trim_failure_seconds)
                    attempts[-1]["trimmed_tail_steps"] = int(episode.get("trimmed_tail_steps", 0))
                    attempts[-1]["saved_steps"] = int(episode["actions"].shape[0])

            length = save_lerobot_episode(
                output_dataset,
                info,
                stats_list,
                episode_index=saved_episodes,
                global_start_index=global_index,
                episode=episode,
                task_text=task_text,
                task_index=task_index,
                fps=fps,
            )
            append_jsonl(
                output_dataset / "meta" / OUTCOME_LEDGER_NAME,
                make_outcome_row(
                    episode_index=saved_episodes,
                    success=bool(episode["success"]),
                    attempt_index=attempt_idx,
                    seed=seed,
                ),
            )
            global_index += length
            saved_episodes += 1
            if episode["success"]:
                successes_saved += 1
            else:
                failures += 1
            update_info(
                output_dataset,
                info,
                num_episodes=saved_episodes,
                total_frames=global_index,
                total_tasks=total_tasks,
            )
            persist_summary("running")
            if save_all:
                print(f"[collect] saved episode {saved_episodes}/{target_episodes} failures={failures}", flush=True)
            else:
                print(f"[collect] saved failure {failures}/{args.target_failures}", flush=True)
    finally:
        policy.close()

    outcome_rows = load_jsonl(output_dataset / "meta" / OUTCOME_LEDGER_NAME)
    validate_outcome_rows(
        outcome_rows,
        expected_episode_count=saved_episodes,
        attempts=attempts,
    )
    if stats_list:
        write_json(output_dataset / "meta" / "stats.json", serialize_dict(aggregate_stats(stats_list)))
    update_info(
        output_dataset,
        info,
        num_episodes=saved_episodes,
        total_frames=global_index,
        total_tasks=total_tasks,
    )
    persist_summary(
        "complete"
        if (
            (save_all and saved_episodes >= target_episodes)
            or ((not save_all) and failures >= args.target_failures)
        )
        else "incomplete"
    )
    print(
        f"[collect] finished episodes={saved_episodes} failures={failures} "
        f"successes_saved={successes_saved} attempts={len(attempts)} frames={global_index}",
        flush=True,
    )


if __name__ == "__main__":
    main()
