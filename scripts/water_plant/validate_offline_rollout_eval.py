#!/usr/bin/env python3
"""Strictly validate a formal Water Plant offline checkpoint rollout.

The wrapper-created protocol is the source of truth. This validator accepts a
combined summary only when its four shards, 50 episode rows, seeds, settings,
inference seed, and saved artifacts all agree with that frozen protocol.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import io
import json
import math
import os
import stat
import statistics
import tempfile
from pathlib import Path
from typing import Any


ALLOWED_VARIANTS = {"B0", "B1", "C", "M"}
EXPECTED_EPISODES = 50
EXPECTED_GPUS = [0, 1, 2, 3]
EXPECTED_REPLAN_STEPS = 25
EXPECTED_MAX_ENV_STEPS = 1500
EXPECTED_TASK = "water_plant"
HORIZONS = (600, 1000, 1500)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a 50-episode FITWAM Water Plant checkpoint rollout."
    )
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--report-csv", type=Path, required=True)
    parser.add_argument("--episodes-csv", type=Path, required=True)
    return parser.parse_args()


def _read_json_object(path: Path, *, label: str) -> tuple[dict[str, Any], bytes]:
    path = path.expanduser().resolve()
    payload = path.read_bytes()
    value = json.loads(payload.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return value, payload


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _require_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer, got {value!r}")
    return value


def _require_bool(value: Any, *, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be boolean, got {value!r}")
    return value


def _require_number(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric, got {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite, got {value!r}")
    return result


def _require_equal(actual: Any, expected: Any, *, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label} must be {expected!r}, got {actual!r}")


def _require_rate(actual: Any, expected: float, *, label: str) -> None:
    value = _require_number(actual, label=label)
    if not math.isclose(value, expected, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{label} must be {expected!r}, got {value!r}")


def _protocol_value(protocol: dict[str, Any], key: str) -> Any:
    """Read a setting from the protocol root or its ``evaluation`` object.

    Supporting both locations keeps the validator compatible with a flat launch
    card and a namespaced wrapper protocol while rejecting conflicting copies.
    """

    values: list[tuple[str, Any]] = []
    if key in protocol:
        values.append((key, protocol[key]))
    evaluation = protocol.get("evaluation")
    if evaluation is not None:
        if not isinstance(evaluation, dict):
            raise ValueError("protocol.evaluation must be a JSON object")
        if key in evaluation:
            values.append((f"evaluation.{key}", evaluation[key]))
    if not values:
        raise ValueError(f"protocol is missing required setting {key!r}")
    first = values[0][1]
    for location, value in values[1:]:
        if value != first:
            raise ValueError(
                f"protocol has conflicting {key!r} values: "
                f"{values[0][0]}={first!r}, {location}={value!r}"
            )
    return first


def _protocol_shards(protocol: dict[str, Any]) -> Any:
    candidates: list[tuple[str, Any]] = []
    for key in ("shards", "shard_assignments"):
        try:
            candidates.append((key, _protocol_value(protocol, key)))
        except ValueError as exc:
            if "missing required setting" not in str(exc):
                raise
    if not candidates:
        raise ValueError("protocol is missing required shard assignments")
    first = candidates[0][1]
    for location, value in candidates[1:]:
        if value != first:
            raise ValueError(
                "protocol has conflicting shard assignments: "
                f"{candidates[0][0]} and {location}"
            )
    return first


def _normalize_protocol(protocol: dict[str, Any]) -> dict[str, Any]:
    variant = _protocol_value(protocol, "variant")
    if variant not in ALLOWED_VARIANTS:
        raise ValueError(
            f"protocol.variant must be one of {sorted(ALLOWED_VARIANTS)}, got {variant!r}"
        )

    episodes = _require_int(_protocol_value(protocol, "episodes"), label="protocol.episodes")
    _require_equal(episodes, EXPECTED_EPISODES, label="protocol.episodes")

    raw_gpus = _protocol_value(protocol, "gpus")
    if not isinstance(raw_gpus, list):
        raise ValueError("protocol.gpus must be a list")
    gpus = [_require_int(value, label="protocol.gpus[]") for value in raw_gpus]
    _require_equal(gpus, EXPECTED_GPUS, label="protocol.gpus")

    replan_steps = _require_int(
        _protocol_value(protocol, "replan_steps"), label="protocol.replan_steps"
    )
    _require_equal(
        replan_steps, EXPECTED_REPLAN_STEPS, label="protocol.replan_steps"
    )
    max_env_steps = _require_int(
        _protocol_value(protocol, "max_env_steps"), label="protocol.max_env_steps"
    )
    _require_equal(
        max_env_steps, EXPECTED_MAX_ENV_STEPS, label="protocol.max_env_steps"
    )
    task = _protocol_value(protocol, "task")
    _require_equal(task, EXPECTED_TASK, label="protocol.task")

    boolean_expectations = {
        "save_video": True,
        "save_actions": True,
        "randomize": False,
        "randomize_dynamics": False,
        "action_clip": False,
    }
    booleans: dict[str, bool] = {}
    for key, expected in boolean_expectations.items():
        value = _require_bool(_protocol_value(protocol, key), label=f"protocol.{key}")
        _require_equal(value, expected, label=f"protocol.{key}")
        booleans[key] = value

    base_seed = _require_int(
        _protocol_value(protocol, "base_seed"), label="protocol.base_seed"
    )
    inference_seed = _require_int(
        _protocol_value(protocol, "inference_seed"),
        label="protocol.inference_seed",
    )
    expected_seeds = list(range(base_seed, base_seed + episodes))

    if "seed_stop_exclusive" in protocol or (
        isinstance(protocol.get("evaluation"), dict)
        and "seed_stop_exclusive" in protocol["evaluation"]
    ):
        seed_stop = _require_int(
            _protocol_value(protocol, "seed_stop_exclusive"),
            label="protocol.seed_stop_exclusive",
        )
        _require_equal(
            seed_stop,
            base_seed + episodes,
            label="protocol.seed_stop_exclusive",
        )
    if "seeds" in protocol or (
        isinstance(protocol.get("evaluation"), dict)
        and "seeds" in protocol["evaluation"]
    ):
        raw_seeds = _protocol_value(protocol, "seeds")
        if not isinstance(raw_seeds, list):
            raise ValueError("protocol.seeds must be a list")
        seeds = [_require_int(value, label="protocol.seeds[]") for value in raw_seeds]
        _require_equal(seeds, expected_seeds, label="protocol.seeds")

    raw_assignments = _protocol_shards(protocol)
    if not isinstance(raw_assignments, list) or len(raw_assignments) != len(gpus):
        raise ValueError("protocol shard assignments must contain exactly four rows")

    base_count, remainder = divmod(episodes, len(gpus))
    assignments: list[dict[str, int]] = []
    seed_cursor = base_seed
    episode_cursor = 0
    seen_ids: set[int] = set()
    for expected_id, raw in enumerate(raw_assignments):
        if not isinstance(raw, dict):
            raise ValueError("each protocol shard assignment must be an object")
        shard_id = _require_int(raw.get("shard_id"), label="protocol.shards[].shard_id")
        if shard_id in seen_ids:
            raise ValueError(f"protocol contains duplicate shard_id {shard_id}")
        seen_ids.add(shard_id)
        _require_equal(shard_id, expected_id, label="protocol.shards[].shard_id")

        count = base_count + (1 if expected_id < remainder else 0)
        gpu = _require_int(raw.get("gpu"), label=f"protocol.shards[{shard_id}].gpu")
        _require_equal(gpu, gpus[expected_id], label=f"protocol.shards[{shard_id}].gpu")
        assigned_episodes = _require_int(
            raw.get("episodes"), label=f"protocol.shards[{shard_id}].episodes"
        )
        _require_equal(
            assigned_episodes, count, label=f"protocol.shards[{shard_id}].episodes"
        )

        raw_seed_start = raw.get("base_seed", raw.get("seed_start"))
        seed_start = _require_int(
            raw_seed_start, label=f"protocol.shards[{shard_id}].base_seed"
        )
        _require_equal(
            seed_start, seed_cursor, label=f"protocol.shards[{shard_id}].base_seed"
        )
        if "base_seed" in raw and "seed_start" in raw:
            _require_equal(
                raw["seed_start"],
                raw["base_seed"],
                label=f"protocol.shards[{shard_id}].seed_start",
            )
        seed_stop = _require_int(
            raw.get("seed_stop_exclusive"),
            label=f"protocol.shards[{shard_id}].seed_stop_exclusive",
        )
        _require_equal(
            seed_stop,
            seed_cursor + count,
            label=f"protocol.shards[{shard_id}].seed_stop_exclusive",
        )
        global_start = _require_int(
            raw.get("global_episode_start"),
            label=f"protocol.shards[{shard_id}].global_episode_start",
        )
        _require_equal(
            global_start,
            episode_cursor,
            label=f"protocol.shards[{shard_id}].global_episode_start",
        )
        assignments.append(
            {
                "shard_id": shard_id,
                "gpu": gpu,
                "episodes": count,
                "base_seed": seed_start,
                "seed_stop_exclusive": seed_stop,
                "global_episode_start": global_start,
            }
        )
        seed_cursor += count
        episode_cursor += count

    _require_equal(seed_cursor, base_seed + episodes, label="protocol shard seed coverage")
    _require_equal(episode_cursor, episodes, label="protocol shard episode coverage")

    provenance = protocol.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("protocol.provenance must be a JSON object")
    provenance = copy.deepcopy(provenance)
    if "inference_seed" in provenance:
        stored = _require_int(
            provenance["inference_seed"], label="protocol.provenance.inference_seed"
        )
        _require_equal(
            stored, inference_seed, label="protocol.provenance.inference_seed"
        )
    provenance["inference_seed"] = inference_seed

    return {
        "variant": variant,
        "episodes": episodes,
        "gpus": gpus,
        "replan_steps": replan_steps,
        "max_env_steps": max_env_steps,
        "task": task,
        "base_seed": base_seed,
        "inference_seed": inference_seed,
        "expected_seeds": expected_seeds,
        "assignments": assignments,
        "provenance": provenance,
        **booleans,
    }


def _absolute_artifact_path(raw_path: Any, *, summary_path: Path, field: str) -> Path:
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError(f"{field} must be a non-empty path")
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = summary_path.parent / path
    return Path(os.path.abspath(path))


def _require_nonempty_regular_file(path: Path, *, label: str) -> int:
    metadata = os.lstat(path)
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{label} must be a regular file: {path}")
    if metadata.st_size <= 0:
        raise ValueError(f"{label} must be nonempty: {path}")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise ValueError(f"{label} must be a regular file: {path}")
        if (metadata.st_dev, metadata.st_ino) != (opened.st_dev, opened.st_ino):
            raise RuntimeError(f"{label} was replaced while opening: {path}")
        if opened.st_size <= 0:
            raise ValueError(f"{label} must be nonempty: {path}")
        return opened.st_size
    finally:
        os.close(descriptor)


def _validate_summary_setting(
    summary: dict[str, Any], key: str, expected: Any, *, required: bool = True
) -> None:
    if key not in summary:
        if required:
            raise ValueError(f"combined summary is missing setting {key!r}")
        return
    actual = summary[key]
    if isinstance(expected, bool):
        actual = _require_bool(actual, label=f"summary.{key}")
    elif isinstance(expected, int):
        actual = _require_int(actual, label=f"summary.{key}")
    _require_equal(actual, expected, label=f"summary.{key}")


def validate_offline_rollout_eval(
    summary_path: Path, protocol_path: Path
) -> dict[str, Any]:
    summary_path = summary_path.expanduser().resolve()
    protocol_path = protocol_path.expanduser().resolve()
    summary, summary_bytes = _read_json_object(summary_path, label="summary")
    protocol, protocol_bytes = _read_json_object(protocol_path, label="protocol")
    frozen = _normalize_protocol(protocol)

    required_summary_settings = {
        "total_episodes": frozen["episodes"],
        "episodes_per_task": frozen["episodes"],
        "num_tasks": 1,
        "num_shards": 4,
        "gpus": frozen["gpus"],
        "replan_steps": frozen["replan_steps"],
        "seed": frozen["base_seed"],
        "inference_seed": frozen["inference_seed"],
        "save_actions": frozen["save_actions"],
        "randomize": frozen["randomize"],
        "randomize_dynamics": frozen["randomize_dynamics"],
        "action_clip": frozen["action_clip"],
    }
    for key, expected in required_summary_settings.items():
        _validate_summary_setting(summary, key, expected)
    _validate_summary_setting(
        summary, "max_env_steps", frozen["max_env_steps"]
    )
    _validate_summary_setting(summary, "save_video", frozen["save_video"])
    _validate_summary_setting(summary, "task", frozen["task"])

    tasks = summary.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != 1:
        raise ValueError("combined summary must contain exactly one task")
    task = tasks[0]
    if not isinstance(task, dict):
        raise ValueError("combined summary task must be an object")
    _require_equal(task.get("env_name"), frozen["task"], label="summary task")
    _require_equal(
        _require_int(task.get("episodes"), label="summary task episodes"),
        frozen["episodes"],
        label="summary task episodes",
    )

    raw_episodes = task.get("episode_results")
    if not isinstance(raw_episodes, list) or len(raw_episodes) != frozen["episodes"]:
        count = len(raw_episodes) if isinstance(raw_episodes, list) else type(raw_episodes).__name__
        raise ValueError(
            f"combined summary must contain exactly {frozen['episodes']} episode rows, got {count}"
        )

    episode_ids: list[int] = []
    seeds: list[int] = []
    episode_reports: list[dict[str, Any]] = []
    rows_by_shard: dict[int, list[dict[str, Any]]] = {index: [] for index in range(4)}
    for row_number, raw in enumerate(raw_episodes, 1):
        if not isinstance(raw, dict):
            raise ValueError(f"episode row {row_number} must be an object")
        episode = _require_int(raw.get("episode"), label=f"episode row {row_number}.episode")
        seed = _require_int(raw.get("seed"), label=f"episode row {row_number}.seed")
        shard = _require_int(raw.get("shard"), label=f"episode row {row_number}.shard")
        if shard not in rows_by_shard:
            raise ValueError(f"episode row {row_number}.shard must be 0..3, got {shard}")
        success = _require_bool(raw.get("success"), label=f"episode row {row_number}.success")
        steps = _require_int(raw.get("steps"), label=f"episode row {row_number}.steps")
        if not 1 <= steps <= frozen["max_env_steps"]:
            raise ValueError(
                f"episode row {row_number}.steps must be in 1..{frozen['max_env_steps']}, got {steps}"
            )

        video_path = _absolute_artifact_path(
            raw.get("video_path"), summary_path=summary_path, field="video_path"
        )
        actions_path = _absolute_artifact_path(
            raw.get("actions_path"), summary_path=summary_path, field="actions_path"
        )
        video_size = _require_nonempty_regular_file(video_path, label="video_path")
        actions_size = _require_nonempty_regular_file(actions_path, label="actions_path")

        report = {
            "episode": episode,
            "seed": seed,
            "shard": shard,
            "success": success,
            "outcome": "success" if success else "failure",
            "steps": steps,
            "success_step": steps if success else None,
            "video_path": str(video_path),
            "video_size_bytes": video_size,
            "actions_path": str(actions_path),
            "actions_size_bytes": actions_size,
        }
        episode_ids.append(episode)
        seeds.append(seed)
        rows_by_shard[shard].append(report)
        episode_reports.append(report)

    if len(set(episode_ids)) != frozen["episodes"]:
        raise ValueError("combined summary contains duplicate episode values")
    expected_episode_ids = list(range(frozen["episodes"]))
    if sorted(episode_ids) != expected_episode_ids:
        raise ValueError(
            f"combined summary episode values must be 0..{frozen['episodes'] - 1}"
        )
    if len(set(seeds)) != frozen["episodes"]:
        raise ValueError("combined summary contains duplicate seed values")
    actual_seed_set = set(seeds)
    expected_seed_set = set(frozen["expected_seeds"])
    if actual_seed_set != expected_seed_set:
        missing = sorted(expected_seed_set - actual_seed_set)
        unexpected = sorted(actual_seed_set - expected_seed_set)
        raise ValueError(
            f"combined summary seed set mismatch; missing={missing}, unexpected={unexpected}"
        )

    raw_shards = summary.get("shards")
    if not isinstance(raw_shards, list) or len(raw_shards) != 4:
        raise ValueError("combined summary must contain exactly four shard metadata rows")
    summary_shards: dict[int, dict[str, Any]] = {}
    for raw in raw_shards:
        if not isinstance(raw, dict):
            raise ValueError("each combined summary shard row must be an object")
        shard_id = _require_int(raw.get("shard_id"), label="summary.shards[].shard_id")
        if shard_id in summary_shards:
            raise ValueError(f"combined summary contains duplicate shard_id {shard_id}")
        if shard_id not in rows_by_shard:
            raise ValueError(f"combined summary shard_id must be 0..3, got {shard_id}")
        summary_shards[shard_id] = raw

    for assignment in frozen["assignments"]:
        shard_id = assignment["shard_id"]
        rows = rows_by_shard[shard_id]
        expected_shard_seeds = list(
            range(assignment["base_seed"], assignment["seed_stop_exclusive"])
        )
        actual_shard_seeds = sorted(row["seed"] for row in rows)
        _require_equal(
            len(rows), assignment["episodes"], label=f"shard {shard_id} episode row count"
        )
        _require_equal(
            actual_shard_seeds,
            expected_shard_seeds,
            label=f"shard {shard_id} seed range",
        )

        metadata = summary_shards[shard_id]
        _require_equal(
            _require_int(metadata.get("episodes"), label=f"summary shard {shard_id}.episodes"),
            assignment["episodes"],
            label=f"summary shard {shard_id}.episodes",
        )
        _require_equal(
            _require_int(metadata.get("base_seed"), label=f"summary shard {shard_id}.base_seed"),
            assignment["base_seed"],
            label=f"summary shard {shard_id}.base_seed",
        )
        if "seed_stop_exclusive" in metadata:
            _require_equal(
                _require_int(
                    metadata["seed_stop_exclusive"],
                    label=f"summary shard {shard_id}.seed_stop_exclusive",
                ),
                assignment["seed_stop_exclusive"],
                label=f"summary shard {shard_id}.seed_stop_exclusive",
            )
        if "global_episode_start" in metadata:
            _require_equal(
                _require_int(
                    metadata["global_episode_start"],
                    label=f"summary shard {shard_id}.global_episode_start",
                ),
                assignment["global_episode_start"],
                label=f"summary shard {shard_id}.global_episode_start",
            )
        if "gpu" in metadata:
            _require_equal(
                _require_int(metadata["gpu"], label=f"summary shard {shard_id}.gpu"),
                assignment["gpu"],
                label=f"summary shard {shard_id}.gpu",
            )
        shard_successes = sum(1 for row in rows if row["success"])
        _require_equal(
            _require_int(
                metadata.get("successes"), label=f"summary shard {shard_id}.successes"
            ),
            shard_successes,
            label=f"summary shard {shard_id}.successes",
        )
        _require_rate(
            metadata.get("success_rate"),
            shard_successes / assignment["episodes"],
            label=f"summary shard {shard_id}.success_rate",
        )

    final_successes = sum(1 for row in episode_reports if row["success"])
    _require_equal(
        _require_int(task.get("successes"), label="summary task successes"),
        final_successes,
        label="summary task successes",
    )
    _require_rate(
        task.get("success_rate"),
        final_successes / frozen["episodes"],
        label="summary task success_rate",
    )
    _require_equal(
        _require_int(summary.get("total_successes"), label="summary.total_successes"),
        final_successes,
        label="summary.total_successes",
    )
    _require_rate(
        summary.get("overall_success_rate"),
        final_successes / frozen["episodes"],
        label="summary.overall_success_rate",
    )

    episode_reports.sort(key=lambda row: row["seed"])
    successful_steps = [row["steps"] for row in episode_reports if row["success"]]
    median_successful_step = (
        float(statistics.median(successful_steps)) if successful_steps else None
    )
    horizons: dict[str, dict[str, Any]] = {}
    for horizon in HORIZONS:
        successes = sum(
            1
            for row in episode_reports
            if row["success"] and row["steps"] <= horizon
        )
        horizons[str(horizon)] = {
            "horizon": horizon,
            "successes": successes,
            "episodes": frozen["episodes"],
            "rate": successes / frozen["episodes"],
        }

    return {
        "status": "valid",
        "variant": frozen["variant"],
        "provenance": frozen["provenance"],
        "protocol": {
            "path": str(protocol_path),
            "sha256": _sha256(protocol_bytes),
        },
        "summary": {
            "path": str(summary_path),
            "sha256": _sha256(summary_bytes),
        },
        "settings": {
            "task": frozen["task"],
            "episodes": frozen["episodes"],
            "gpus": frozen["gpus"],
            "base_seed": frozen["base_seed"],
            "inference_seed": frozen["inference_seed"],
            "replan_steps": frozen["replan_steps"],
            "max_env_steps": frozen["max_env_steps"],
            "save_video": frozen["save_video"],
            "save_actions": frozen["save_actions"],
            "randomize": frozen["randomize"],
            "randomize_dynamics": frozen["randomize_dynamics"],
            "action_clip": frozen["action_clip"],
            "shards": frozen["assignments"],
        },
        "horizons": horizons,
        "final_successes": final_successes,
        "median_successful_step": median_successful_step,
        "episodes": episode_reports,
    }


def _render_horizon_csv(report: dict[str, Any]) -> str:
    stream = io.StringIO(newline="")
    fields = [
        "variant",
        "inference_seed",
        "horizon",
        "successes",
        "episodes",
        "rate",
        "median_successful_step",
    ]
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for horizon in HORIZONS:
        metric = report["horizons"][str(horizon)]
        writer.writerow(
            {
                "variant": report["variant"],
                "inference_seed": report["settings"]["inference_seed"],
                "horizon": horizon,
                "successes": metric["successes"],
                "episodes": metric["episodes"],
                "rate": metric["rate"],
                "median_successful_step": report["median_successful_step"],
            }
        )
    return stream.getvalue()


def _render_episode_csv(report: dict[str, Any]) -> str:
    stream = io.StringIO(newline="")
    fields = [
        "variant",
        "inference_seed",
        "episode",
        "seed",
        "shard",
        "success",
        "outcome",
        "steps",
        "success_step",
        "video_path",
        "actions_path",
    ]
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for episode in report["episodes"]:
        writer.writerow(
            {
                "variant": report["variant"],
                "inference_seed": report["settings"]["inference_seed"],
                **{key: episode[key] for key in fields if key in episode},
            }
        )
    return stream.getvalue()


def _stage_atomic_text(path: Path, content: str) -> Path:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.incomplete-", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return temporary


def _atomic_write_many(outputs: list[tuple[Path, str]]) -> None:
    staged: list[tuple[Path, Path]] = []
    try:
        for target, content in outputs:
            target = target.expanduser().resolve()
            staged.append((target, _stage_atomic_text(target, content)))
        for target, temporary in staged:
            os.replace(temporary, target)
    finally:
        for _, temporary in staged:
            temporary.unlink(missing_ok=True)


def validate_and_write(
    summary_path: Path,
    protocol_path: Path,
    report_json_path: Path,
    report_csv_path: Path,
    episodes_csv_path: Path,
) -> dict[str, Any]:
    outputs = [
        report_json_path.expanduser().resolve(),
        report_csv_path.expanduser().resolve(),
        episodes_csv_path.expanduser().resolve(),
    ]
    if len(set(outputs)) != len(outputs):
        raise ValueError("report-json, report-csv, and episodes-csv must be distinct paths")
    input_paths = {
        summary_path.expanduser().resolve(),
        protocol_path.expanduser().resolve(),
    }
    if input_paths.intersection(outputs):
        raise ValueError("output paths must not overwrite summary or protocol inputs")

    # Remove stale validity artifacts before revalidation so a failed run cannot
    # leave an older valid report beside incomplete rollout outputs.
    for path in outputs:
        path.unlink(missing_ok=True)

    report = validate_offline_rollout_eval(summary_path, protocol_path)
    report_json = json.dumps(
        report, ensure_ascii=False, indent=2, sort_keys=True
    ) + "\n"
    _atomic_write_many(
        [
            (outputs[0], report_json),
            (outputs[1], _render_horizon_csv(report)),
            (outputs[2], _render_episode_csv(report)),
        ]
    )
    return report


def main() -> None:
    args = parse_args()
    report = validate_and_write(
        args.summary,
        args.protocol,
        args.report_json,
        args.report_csv,
        args.episodes_csv,
    )
    print(f"[offline-rollout-validation] status={report['status']}")
    print(f"[offline-rollout-validation] variant={report['variant']}")
    print(f"[offline-rollout-validation] report={args.report_json.expanduser().resolve()}")


if __name__ == "__main__":
    main()
