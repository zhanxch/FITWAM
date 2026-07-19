#!/usr/bin/env python3
"""Build deterministic success/failure event pairs for EveRobot.

The pair ledger is append-only.  A pair is emitted only when both events belong
to the same task and satisfy all configured progress, pre-state, and action
constraints.  Missing features make an event ineligible instead of triggering
a heuristic fallback.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


PAIR_LEDGER_FORMAT = "EveRobotEventPairs"
PAIR_SCHEMA_VERSION = "0.1"
DEFAULT_PAIRING_VERSION = "event_pair_v1"
PAIR_DIAGNOSTICS_FORMAT = "EveRobotEventPairDiagnostics"
PAIR_CALIBRATION_FORMAT = "EveRobotEventPairCalibration"
PAIR_CALIBRATION_VERSION = "event_pair_calibration_v1"

_PROGRESS_FIELDS = ("progress", "event_progress", "phase_progress")
_PRE_STATE_FIELDS = (
    "pre_state_embedding",
    "pre_state_feature",
    "pre_state",
)
_ACTION_FIELDS = (
    "action_embedding",
    "action_feature",
    "action_summary",
)


def canonical_json(payload: Any) -> str:
    return json.dumps(
        payload,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_json(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def file_sha256(path: Path | None) -> str | None:
    if path is None:
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            line = line.strip()
            if not line:
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} must contain a JSON object")
            rows.append(value)
    return rows


def _parse_csv_value(value: str) -> Any:
    stripped = value.strip()
    if not stripped:
        return None
    if stripped[0] in "[{\"" or stripped in {"true", "false", "null"}:
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass
    try:
        return float(stripped)
    except ValueError:
        return value


def read_feature_table(path: Path | None) -> list[dict[str, Any]]:
    """Read optional per-event features from JSONL, JSON, CSV, or Parquet."""

    if path is None:
        return []
    suffix = path.suffix.lower()
    if suffix in {".jsonl", ".ndjson"}:
        return read_jsonl(path)
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list) or not all(
            isinstance(row, dict) for row in payload
        ):
            raise ValueError(f"{path} must contain a JSON list of objects")
        return payload
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as stream:
            return [
                {key: _parse_csv_value(value) for key, value in row.items()}
                for row in csv.DictReader(stream)
            ]
    if suffix == ".parquet":
        try:
            import pandas as pd
        except ImportError as error:
            raise RuntimeError(
                "Reading Parquet features requires pandas with a Parquet engine"
            ) from error
        return pd.read_parquet(path).to_dict(orient="records")
    raise ValueError(
        f"Unsupported feature table extension {suffix!r}; "
        "use .jsonl, .json, .csv, or .parquet"
    )


def write_jsonl_atomic(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(canonical_json(row))
                stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def write_json_atomic_frozen(path: Path, payload: Mapping[str, Any]) -> bool:
    """Atomically create an immutable JSON artifact or verify exact reuse."""

    path.parent.mkdir(parents=True, exist_ok=True)
    content = (canonical_json(dict(payload)) + "\n").encode("utf-8")
    if path.exists():
        existing = path.read_bytes()
        if existing != content:
            raise ValueError(f"Frozen artifact conflict: {path}")
        return False

    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary_path, path)
        except FileExistsError:
            existing = path.read_bytes()
            if existing != content:
                raise ValueError(f"Frozen artifact conflict: {path}")
            return False
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return True
    finally:
        temporary_path.unlink(missing_ok=True)


@contextmanager
def pair_ledger_lock(output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output_path.with_suffix(output_path.suffix + ".lock")
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _without_created_at(row: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key != "created_at"}


def prepare_immutable_pairs(
    existing_rows: Sequence[dict[str, Any]],
    new_rows: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    """Return an immutable append plan and the number of newly added rows."""

    merged = [dict(row) for row in existing_rows]
    by_id: dict[str, dict[str, Any]] = {}
    for row in merged:
        pair_id = row.get("pair_id")
        if not isinstance(pair_id, str) or not pair_id:
            raise ValueError("Every existing pair row must have a non-empty pair_id")
        if (
            pair_id in by_id
            and _without_created_at(by_id[pair_id])
            != _without_created_at(row)
        ):
            raise ValueError(f"Conflicting duplicate pair_id already exists: {pair_id}")
        by_id[pair_id] = row

    appended = 0
    for new_row in new_rows:
        pair_id = new_row.get("pair_id")
        if not isinstance(pair_id, str) or not pair_id:
            raise ValueError("Every new pair row must have a non-empty pair_id")
        existing = by_id.get(pair_id)
        if existing is not None:
            if _without_created_at(existing) != _without_created_at(new_row):
                raise ValueError(
                    f"Immutable pair_id collision for {pair_id}; "
                    "use a new pairing version or configuration"
                )
            continue
        copied = dict(new_row)
        by_id[pair_id] = copied
        merged.append(copied)
        appended += 1
    return merged, appended


def verify_frozen_pair_rows(
    existing_rows: Sequence[dict[str, Any]],
    selected_rows: Sequence[dict[str, Any]],
) -> None:
    """Require an existing formal pair ledger to match the current selection."""

    # Reuse the append validator to reject malformed or conflicting duplicate
    # IDs before comparing the frozen snapshots.
    prepare_immutable_pairs(existing_rows, ())
    existing = sorted(
        (_without_created_at(row) for row in existing_rows),
        key=lambda row: str(row["pair_id"]),
    )
    selected = sorted(
        (_without_created_at(row) for row in selected_rows),
        key=lambda row: str(row["pair_id"]),
    )
    if existing != selected:
        raise ValueError(
            "Frozen pair ledger conflicts with the current selection; "
            "use a new pairing version or EVE_ROOT for changed inputs."
        )


def _finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number")
    return result


def _vector(value: Any, label: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{label} must be a non-empty numeric vector")
    return tuple(_finite_float(item, label) for item in value)


def _first_value(row: Mapping[str, Any], fields: Sequence[str]) -> Any:
    for field in fields:
        if row.get(field) is not None:
            return row[field]
    return None


def _episode_identity(row: Mapping[str, Any]) -> tuple[str, int]:
    dataset_id = row.get("dataset_id")
    episode_index = row.get("episode_index")
    if not isinstance(dataset_id, str) or not dataset_id:
        raise ValueError("Episode/event row requires dataset_id")
    if isinstance(episode_index, bool) or not isinstance(episode_index, int):
        raise ValueError("Episode/event row requires integer episode_index")
    return dataset_id, episode_index


def _index_episodes(
    episode_rows: Sequence[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[tuple[str, int], dict[str, Any]]]:
    by_id: dict[str, dict[str, Any]] = {}
    by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for row in episode_rows:
        key = _episode_identity(row)
        if key in by_key and by_key[key] != row:
            raise ValueError(f"Conflicting episode identity: {key}")
        by_key[key] = row
        episode_id = row.get("episode_id")
        if isinstance(episode_id, str) and episode_id:
            if episode_id in by_id and by_id[episode_id] != row:
                raise ValueError(f"Conflicting episode_id: {episode_id}")
            by_id[episode_id] = row
    return by_id, by_key


def _index_features(
    feature_rows: Sequence[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in feature_rows:
        event_id = row.get("event_id")
        if not isinstance(event_id, str) or not event_id:
            raise ValueError("Every feature row must have a non-empty event_id")
        if event_id in indexed and indexed[event_id] != row:
            raise ValueError(f"Conflicting feature rows for event_id {event_id}")
        indexed[event_id] = row
    return indexed


def _linked_episode(
    event: Mapping[str, Any],
    by_id: Mapping[str, dict[str, Any]],
    by_key: Mapping[tuple[str, int], dict[str, Any]],
) -> dict[str, Any]:
    episode_id = event.get("episode_id")
    episode = by_id.get(episode_id) if isinstance(episode_id, str) else None
    if episode is None:
        episode = by_key.get(_episode_identity(event))
    if episode is None:
        raise ValueError(f"Event {event.get('event_id')} references a missing episode")
    if _episode_identity(event) != _episode_identity(episode):
        raise ValueError(f"Event {event.get('event_id')} does not match its episode")
    return episode


def _normalized_task(event: Mapping[str, Any], episode: Mapping[str, Any]) -> str:
    value = (
        event.get("task_name")
        or episode.get("task_name")
        or event.get("task")
        or episode.get("task")
    )
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Event {event.get('event_id')} has no task")
    return " ".join(value.casefold().split())


def _episode_outcome(
    event: Mapping[str, Any], episode: Mapping[str, Any]
) -> str:
    outcome = event.get("episode_outcome", episode.get("episode_outcome"))
    if outcome not in {"success", "failure"}:
        raise ValueError(
            f"Event {event.get('event_id')} requires success/failure episode_outcome"
        )
    episode_value = episode.get("episode_outcome")
    if episode_value is not None and episode_value != outcome:
        raise ValueError(
            f"Event {event.get('event_id')} outcome conflicts with episode"
        )
    return str(outcome)


def _absolute_confidence(event: Mapping[str, Any]) -> float:
    event_id = event.get("event_id")
    direct = event.get("absolute_confidence")
    annotation = event.get("annotation")
    nested = (
        annotation.get("confidence")
        if isinstance(annotation, Mapping)
        else None
    )
    if direct is None and nested is None:
        raise ValueError(
            f"Event {event_id} requires absolute_confidence; "
            "event_weight is only an episode-normalized sampling weight"
        )
    direct_value = (
        None
        if direct is None
        else _finite_float(direct, f"{event_id}.absolute_confidence")
    )
    nested_value = (
        None
        if nested is None
        else _finite_float(nested, f"{event_id}.annotation.confidence")
    )
    if (
        direct_value is not None
        and nested_value is not None
        and not math.isclose(direct_value, nested_value, rel_tol=0.0, abs_tol=1e-12)
    ):
        raise ValueError(
            f"Event {event_id} has conflicting absolute confidence values"
        )
    value = direct_value if direct_value is not None else nested_value
    assert value is not None
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{event_id}.absolute_confidence must be in [0, 1]")
    return value


def _derived_progress(event: Mapping[str, Any], episode: Mapping[str, Any]) -> float:
    start = event.get("core_start_frame", event.get("start_frame"))
    end = event.get("core_end_frame", event.get("end_frame"))
    length = episode.get("length")
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in (start, end, length)
    ):
        raise ValueError(f"Event {event.get('event_id')} cannot derive progress")
    if length <= 0 or start < 0 or start >= end or end > length:
        raise ValueError(f"Event {event.get('event_id')} has an invalid frame interval")
    return (start + end) / (2.0 * length)


def _event_record(
    event: dict[str, Any],
    episode: dict[str, Any],
    feature: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    event_id = event.get("event_id")
    if not isinstance(event_id, str) or not event_id:
        raise ValueError("Every event row must have a non-empty event_id")
    source: dict[str, Any] = dict(event)
    if feature is not None:
        source.update(
            {key: value for key, value in feature.items() if key != "event_id"}
        )

    progress_value = _first_value(source, _PROGRESS_FIELDS)
    progress = (
        _derived_progress(event, episode)
        if progress_value is None
        else _finite_float(progress_value, f"{event_id}.progress")
    )
    if not 0.0 <= progress <= 1.0:
        raise ValueError(f"{event_id}.progress must be in [0, 1]")

    absolute_confidence = _absolute_confidence(event)
    pre_state_value = _first_value(source, _PRE_STATE_FIELDS)
    action_value = _first_value(source, _ACTION_FIELDS)
    if pre_state_value is None or action_value is None:
        return None

    event_weight_value = event.get("event_weight", 1.0)
    event_weight = _finite_float(event_weight_value, f"{event_id}.event_weight")
    if not 0.0 <= event_weight <= 1.0:
        raise ValueError(f"{event_id}.event_weight must be in [0, 1]")
    event_split = str(event.get("split", episode.get("split", "train")))
    episode_split = str(episode.get("split", event_split))
    if event_split != episode_split:
        raise ValueError(
            f"{event_id}.split={event_split!r} conflicts with episode "
            f"split={episode_split!r}"
        )
    if event_split not in {"train", "val", "test"}:
        raise ValueError(f"{event_id}.split is invalid: {event_split!r}")
    return {
        "event_id": event_id,
        "task": _normalized_task(event, episode),
        "outcome": _episode_outcome(event, episode),
        "split": event_split,
        "progress": progress,
        "pre_state": _vector(pre_state_value, f"{event_id}.pre_state"),
        "action": _vector(action_value, f"{event_id}.action"),
        "absolute_confidence": absolute_confidence,
        "event_weight": event_weight,
    }


def rms_distance(left: Sequence[float], right: Sequence[float], label: str) -> float:
    if len(left) != len(right):
        raise ValueError(f"{label} vectors must have the same dimension")
    if not left:
        raise ValueError(f"{label} vectors must not be empty")
    return math.sqrt(
        sum(
            (left_value - right_value) ** 2
            for left_value, right_value in zip(left, right)
        )
        / len(left)
    )


def pairing_config(
    *,
    pairing_version: str,
    matching: str,
    max_success_uses: int,
    max_failure_uses: int,
    max_progress_delta: float,
    max_pre_state_distance: float,
    min_action_divergence: float,
    min_pair_weight: float = 0.0,
    tau_progress: float = 0.08,
    tau_state: float = 1.0,
    event_types: Sequence[str] | None = None,
    splits: Sequence[str] | None = None,
) -> dict[str, Any]:
    return {
        "pairing_version": pairing_version,
        "matching": matching,
        "max_success_uses": max_success_uses,
        "max_failure_uses": max_failure_uses,
        "max_progress_delta": max_progress_delta,
        "max_pre_state_distance": max_pre_state_distance,
        "min_action_divergence": min_action_divergence,
        "min_pair_weight": min_pair_weight,
        "tau_progress": tau_progress,
        "tau_state": tau_state,
        "event_types": sorted(event_types) if event_types else None,
        "splits": sorted(splits) if splits else None,
    }


def _validate_pairing_config(config: Mapping[str, Any]) -> None:
    if config["matching"] not in {"one_to_one", "bounded", "mutual_nearest"}:
        raise ValueError(
            "matching must be 'one_to_one', 'bounded', or 'mutual_nearest'"
        )
    for field in ("max_success_uses", "max_failure_uses"):
        value = config[field]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{field} must be a positive integer")
    for field in (
        "max_progress_delta",
        "max_pre_state_distance",
        "min_action_divergence",
        "min_pair_weight",
    ):
        value = _finite_float(config.get(field, 0.0), field)
        if value < 0:
            raise ValueError(f"{field} must be non-negative")
    for field in ("tau_progress", "tau_state"):
        value = _finite_float(config[field], field)
        if value <= 0:
            raise ValueError(f"{field} must be positive")


def _collect_event_records(
    event_rows: Sequence[dict[str, Any]],
    episode_rows: Sequence[dict[str, Any]],
    feature_rows: Sequence[dict[str, Any]],
    *,
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_episode_id, by_episode_key = _index_episodes(episode_rows)
    features = _index_features(feature_rows)
    allowed_event_types = (
        None if not config.get("event_types") else set(config["event_types"])
    )
    allowed_splits = None if not config.get("splits") else set(config["splits"])

    records: list[dict[str, Any]] = []
    missing_failure_features: list[dict[str, Any]] = []
    seen_event_ids: set[str] = set()
    for event in sorted(event_rows, key=lambda row: str(row.get("event_id", ""))):
        event_id = event.get("event_id")
        if not isinstance(event_id, str) or not event_id:
            raise ValueError("Every event row must have a non-empty event_id")
        if event_id in seen_event_ids:
            raise ValueError(f"Duplicate event_id: {event_id}")
        seen_event_ids.add(event_id)
        if (
            allowed_event_types is not None
            and event.get("event_type") not in allowed_event_types
        ):
            continue
        if (
            allowed_splits is not None
            and event.get("split", "train") not in allowed_splits
        ):
            continue
        episode = _linked_episode(event, by_episode_id, by_episode_key)
        record = _event_record(event, episode, features.get(event_id))
        if record is not None:
            records.append(record)
            continue
        outcome = _episode_outcome(event, episode)
        if outcome == "failure":
            missing_failure_features.append(
                {
                    "failure_event_id": event_id,
                    "task": _normalized_task(event, episode),
                    "split": str(event.get("split", episode.get("split", "train"))),
                    "selected": False,
                    "rejection_reason": "missing_feature",
                    "selected_pair_ids": [],
                    "candidate_counts": {
                        "same_task_split": 0,
                        "progress_gate": 0,
                        "pre_state_gate": 0,
                        "action_gate": 0,
                    },
                }
            )
    return records, missing_failure_features


def _calibration_input_payload(
    records: Sequence[Mapping[str, Any]],
    *,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    train_records = [
        {
            "event_id": record["event_id"],
            "task": record["task"],
            "outcome": record["outcome"],
            "split": record["split"],
            "progress": record["progress"],
            "pre_state": list(record["pre_state"]),
            "action": list(record["action"]),
        }
        for record in records
        if record["split"] == "train"
    ]
    return {
        "records": sorted(train_records, key=lambda row: row["event_id"]),
        "selection": {
            "max_progress_delta": float(config["max_progress_delta"]),
            "max_pre_state_distance": float(config["max_pre_state_distance"]),
            "event_types": sorted(config.get("event_types") or []),
        },
    }


def _distribution(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
        }
    ordered = sorted(float(value) for value in values)
    return {
        "count": len(ordered),
        "min": ordered[0],
        "max": ordered[-1],
        "mean": statistics.fmean(ordered),
        "median": statistics.median(ordered),
    }


def fit_pairing_calibration(
    records: Sequence[Mapping[str, Any]],
    *,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Fit pair thresholds from train records only."""

    train = [record for record in records if record["split"] == "train"]
    successes = [record for record in train if record["outcome"] == "success"]
    failures = [record for record in train if record["outcome"] == "failure"]
    if not successes or not failures:
        raise ValueError(
            "Pair calibration requires non-empty train success and failure events"
        )

    max_progress = float(config["max_progress_delta"])
    max_pre_state = float(config["max_pre_state_distance"])
    nearest: list[dict[str, Any]] = []
    for failure in failures:
        compatible: list[tuple[float, float, Mapping[str, Any]]] = []
        for success in successes:
            if success["task"] != failure["task"]:
                continue
            progress_delta = abs(
                float(success["progress"]) - float(failure["progress"])
            )
            if progress_delta > max_progress:
                continue
            pre_state_distance = rms_distance(
                success["pre_state"], failure["pre_state"], "pre_state"
            )
            if pre_state_distance > max_pre_state:
                continue
            compatible.append((pre_state_distance, progress_delta, success))
        if not compatible:
            continue
        pre_state_distance, progress_delta, success = min(
            compatible,
            key=lambda item: (
                item[0],
                item[1],
                str(item[2]["event_id"]),
            ),
        )
        nearest.append(
            {
                "failure_event_id": failure["event_id"],
                "success_event_id": success["event_id"],
                "pre_state_distance": pre_state_distance,
                "progress_delta": progress_delta,
                "action_divergence": rms_distance(
                    success["action"], failure["action"], "action"
                ),
            }
        )
    if not nearest:
        raise ValueError(
            "Pair calibration found no train failure with a compatible success event"
        )

    state_distances = [float(row["pre_state_distance"]) for row in nearest]
    action_distances = [float(row["action_divergence"]) for row in nearest]
    tau_state = statistics.median(state_distances)
    if tau_state <= 0.0:
        positive = [value for value in state_distances if value > 0.0]
        tau_state = statistics.median(positive) if positive else 1e-6
    input_payload = _calibration_input_payload(records, config=config)
    input_sha256 = sha256_json(input_payload)
    payload: dict[str, Any] = {
        "format": PAIR_CALIBRATION_FORMAT,
        "schema_version": PAIR_SCHEMA_VERSION,
        "algorithm_version": PAIR_CALIBRATION_VERSION,
        "calibration_split": "train",
        "input_sha256": input_sha256,
        "input_summary": {
            "num_train_records": len(train),
            "num_train_success_events": len(successes),
            "num_train_failure_events": len(failures),
            "num_calibration_matches": len(nearest),
        },
        "selection": input_payload["selection"],
        "statistics": {
            "nearest_pre_state_distance": _distribution(state_distances),
            "nearest_action_divergence": _distribution(action_distances),
        },
        "thresholds": {
            "min_action_divergence": statistics.median(action_distances),
            "tau_state": tau_state,
        },
    }
    payload["calibration_sha256"] = sha256_json(payload)
    payload["calibration_id"] = (
        f"pair-calibration-{payload['calibration_sha256'][:16]}"
    )
    return payload


def validate_pairing_calibration(
    payload: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    *,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    calibration = dict(payload)
    if calibration.get("format") != PAIR_CALIBRATION_FORMAT:
        raise ValueError("Unrecognized pair calibration format")
    if calibration.get("schema_version") != PAIR_SCHEMA_VERSION:
        raise ValueError("Unsupported pair calibration schema")
    if calibration.get("algorithm_version") != PAIR_CALIBRATION_VERSION:
        raise ValueError("Unsupported pair calibration algorithm")
    if calibration.get("calibration_split") != "train":
        raise ValueError("Pair calibration must use split='train'")
    expected_hash = calibration.get("calibration_sha256")
    unhashed = {
        key: value
        for key, value in calibration.items()
        if key not in {"calibration_sha256", "calibration_id"}
    }
    actual_hash = sha256_json(unhashed)
    if expected_hash != actual_hash:
        raise ValueError("Pair calibration hash does not match its payload")
    if calibration.get("calibration_id") != f"pair-calibration-{actual_hash[:16]}":
        raise ValueError("Pair calibration ID does not match its payload")
    current_input_hash = sha256_json(
        _calibration_input_payload(records, config=config)
    )
    if calibration.get("input_sha256") != current_input_hash:
        raise ValueError("Pair calibration conflicts with current train inputs")
    thresholds = calibration.get("thresholds")
    if not isinstance(thresholds, Mapping):
        raise ValueError("Pair calibration has no thresholds")
    min_action = _finite_float(
        thresholds.get("min_action_divergence"),
        "calibration.min_action_divergence",
    )
    if min_action < 0.0:
        raise ValueError("calibration.min_action_divergence must be non-negative")
    tau_state = _finite_float(
        thresholds.get("tau_state"), "calibration.tau_state"
    )
    if tau_state <= 0.0:
        raise ValueError("calibration.tau_state must be positive")
    return calibration


def _quality_below_maximum(value: float, maximum: float) -> float:
    if maximum == 0.0:
        return 1.0 if value == 0.0 else 0.0
    return max(0.0, 1.0 - value / maximum)


def _quality_above_minimum(value: float, minimum: float) -> float:
    if minimum == 0.0:
        return 1.0
    return value / (value + minimum)


def build_event_pairs_with_diagnostics(
    event_rows: Sequence[dict[str, Any]],
    episode_rows: Sequence[dict[str, Any]],
    feature_rows: Sequence[dict[str, Any]] = (),
    *,
    config: Mapping[str, Any],
    provenance: Mapping[str, Any] | None = None,
    created_at: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Create deterministic event pairs plus per-failure diagnostics."""

    _validate_pairing_config(config)
    records, missing_failure_features = _collect_event_records(
        event_rows,
        episode_rows,
        feature_rows,
        config=config,
    )

    successes = [record for record in records if record["outcome"] == "success"]
    failures = [record for record in records if record["outcome"] == "failure"]
    candidates: list[dict[str, Any]] = []
    failure_diagnostics: dict[str, dict[str, Any]] = {
        row["failure_event_id"]: row for row in missing_failure_features
    }
    max_progress = float(config["max_progress_delta"])
    max_pre_state = float(config["max_pre_state_distance"])
    min_action = float(config["min_action_divergence"])
    min_pair_weight = float(config.get("min_pair_weight", 0.0))
    tau_progress = float(config["tau_progress"])
    tau_state = float(config["tau_state"])
    for failure in failures:
        diagnostic = {
            "failure_event_id": failure["event_id"],
            "task": failure["task"],
            "split": failure["split"],
            "selected": False,
            "rejection_reason": None,
            "selected_pair_ids": [],
            "candidate_counts": {
                "same_task_split": 0,
                "progress_gate": 0,
                "pre_state_gate": 0,
                "action_gate": 0,
                "confidence_gate": 0,
                "pair_weight_gate": 0,
            },
        }
        same_task_split = [
            success
            for success in successes
            if success["task"] == failure["task"]
            and success["split"] == failure["split"]
        ]
        diagnostic["candidate_counts"]["same_task_split"] = len(same_task_split)
        for success in same_task_split:
            progress_delta = abs(success["progress"] - failure["progress"])
            if progress_delta > max_progress:
                continue
            diagnostic["candidate_counts"]["progress_gate"] += 1
            pre_state_distance = rms_distance(
                success["pre_state"], failure["pre_state"], "pre_state"
            )
            if pre_state_distance > max_pre_state:
                continue
            diagnostic["candidate_counts"]["pre_state_gate"] += 1
            action_divergence = rms_distance(
                success["action"], failure["action"], "action"
            )
            if action_divergence < min_action:
                continue
            diagnostic["candidate_counts"]["action_gate"] += 1
            components = {
                "progress_delta": progress_delta,
                "pre_state_distance": pre_state_distance,
                "action_divergence": action_divergence,
                "matching_distance": (
                    progress_delta / tau_progress
                    + pre_state_distance / tau_state
                ),
                "progress_quality": _quality_below_maximum(
                    progress_delta, max_progress
                ),
                "pre_state_quality": _quality_below_maximum(
                    pre_state_distance, max_pre_state
                ),
                "action_quality": _quality_above_minimum(
                    action_divergence, min_action
                ),
            }
            event_quality = (
                success["absolute_confidence"]
                * failure["absolute_confidence"]
            )
            if event_quality <= 0.0:
                continue
            diagnostic["candidate_counts"]["confidence_gate"] += 1
            pair_weight = (
                event_quality
                * math.exp(-progress_delta / tau_progress)
                * math.exp(-pre_state_distance / tau_state)
            )
            if pair_weight <= min_pair_weight:
                continue
            diagnostic["candidate_counts"]["pair_weight_gate"] += 1
            candidates.append(
                {
                    "success": success,
                    "failure": failure,
                    "components": components,
                    "pair_weight": min(1.0, pair_weight),
                }
            )
        failure_diagnostics[failure["event_id"]] = diagnostic

    candidates.sort(
        key=lambda candidate: (
            -candidate["pair_weight"],
            candidate["components"]["progress_delta"],
            candidate["components"]["pre_state_distance"],
            -candidate["components"]["action_divergence"],
            candidate["success"]["event_id"],
            candidate["failure"]["event_id"],
        )
    )
    if config["matching"] == "mutual_nearest":
        def nearest_key(candidate: Mapping[str, Any]) -> tuple[Any, ...]:
            return (
                candidate["components"]["matching_distance"],
                candidate["components"]["progress_delta"],
                candidate["components"]["pre_state_distance"],
                -candidate["components"]["action_divergence"],
                candidate["success"]["event_id"],
                candidate["failure"]["event_id"],
            )

        nearest_by_success: dict[str, dict[str, Any]] = {}
        nearest_by_failure: dict[str, dict[str, Any]] = {}
        for candidate in candidates:
            success_id = candidate["success"]["event_id"]
            failure_id = candidate["failure"]["event_id"]
            if (
                success_id not in nearest_by_success
                or nearest_key(candidate) < nearest_key(nearest_by_success[success_id])
            ):
                nearest_by_success[success_id] = candidate
            if (
                failure_id not in nearest_by_failure
                or nearest_key(candidate) < nearest_key(nearest_by_failure[failure_id])
            ):
                nearest_by_failure[failure_id] = candidate
        selected = [
            candidate
            for candidate in candidates
            if nearest_by_success[candidate["success"]["event_id"]] is candidate
            and nearest_by_failure[candidate["failure"]["event_id"]] is candidate
        ]
    else:
        success_cap = (
            1
            if config["matching"] == "one_to_one"
            else int(config["max_success_uses"])
        )
        failure_cap = (
            1
            if config["matching"] == "one_to_one"
            else int(config["max_failure_uses"])
        )
        success_uses: dict[str, int] = {}
        failure_uses: dict[str, int] = {}
        selected = []
        for candidate in candidates:
            success_id = candidate["success"]["event_id"]
            failure_id = candidate["failure"]["event_id"]
            if success_uses.get(success_id, 0) >= success_cap:
                continue
            if failure_uses.get(failure_id, 0) >= failure_cap:
                continue
            selected.append(candidate)
            success_uses[success_id] = success_uses.get(success_id, 0) + 1
            failure_uses[failure_id] = failure_uses.get(failure_id, 0) + 1

    config_hash = sha256_json(dict(config))
    timestamp = created_at or datetime.now(timezone.utc).isoformat()
    rows: list[dict[str, Any]] = []
    for candidate in selected:
        success = candidate["success"]
        failure = candidate["failure"]
        identity = {
            "pairing_version": config["pairing_version"],
            "pairing_config_sha256": config_hash,
            "success_event_id": success["event_id"],
            "failure_event_id": failure["event_id"],
        }
        pair_id = f"pair:{config['pairing_version']}:{sha256_json(identity)[:24]}"
        rows.append(
            {
                "format": PAIR_LEDGER_FORMAT,
                "schema_version": PAIR_SCHEMA_VERSION,
                "pairing_version": config["pairing_version"],
                "pairing_config_sha256": config_hash,
                "pair_id": pair_id,
                "task": success["task"],
                "split": success["split"],
                "success_event_id": success["event_id"],
                "failure_event_id": failure["event_id"],
                "pair_weight": candidate["pair_weight"],
                "components": {
                    **candidate["components"],
                    "success_progress": success["progress"],
                    "failure_progress": failure["progress"],
                    "success_absolute_confidence": success[
                        "absolute_confidence"
                    ],
                    "failure_absolute_confidence": failure[
                        "absolute_confidence"
                    ],
                    "success_event_weight": success["event_weight"],
                    "failure_event_weight": failure["event_weight"],
                },
                "provenance": dict(provenance or {}),
                "created_at": timestamp,
            }
        )
    rows.sort(key=lambda row: row["pair_id"])

    selected_by_failure: dict[str, list[str]] = {}
    for row in rows:
        selected_by_failure.setdefault(row["failure_event_id"], []).append(
            row["pair_id"]
        )
    for failure_id, diagnostic in failure_diagnostics.items():
        selected_pair_ids = sorted(selected_by_failure.get(failure_id, []))
        diagnostic["selected_pair_ids"] = selected_pair_ids
        diagnostic["selected"] = bool(selected_pair_ids)
        if selected_pair_ids:
            diagnostic["rejection_reason"] = None
            continue
        if diagnostic["rejection_reason"] is not None:
            continue
        counts = diagnostic["candidate_counts"]
        if counts["same_task_split"] == 0:
            reason = "no_success_same_task_split"
        elif counts["progress_gate"] == 0:
            reason = "progress_threshold"
        elif counts["pre_state_gate"] == 0:
            reason = "pre_state_threshold"
        elif counts["action_gate"] == 0:
            reason = "action_divergence_threshold"
        elif counts["confidence_gate"] == 0:
            reason = "zero_absolute_confidence"
        elif counts["pair_weight_gate"] == 0:
            reason = "pair_weight_threshold"
        elif config["matching"] == "mutual_nearest":
            reason = "not_mutual_nearest"
        else:
            reason = "capacity_conflict"
        diagnostic["rejection_reason"] = reason

    diagnostic_rows = sorted(
        failure_diagnostics.values(), key=lambda row: row["failure_event_id"]
    )
    selected_failure_count = sum(bool(row["selected"]) for row in diagnostic_rows)
    reason_counts: dict[str, int] = {}
    for row in diagnostic_rows:
        reason = "selected" if row["selected"] else str(row["rejection_reason"])
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
    total_failures = len(diagnostic_rows)
    diagnostics = {
        "format": PAIR_DIAGNOSTICS_FORMAT,
        "schema_version": PAIR_SCHEMA_VERSION,
        "pairing_version": config["pairing_version"],
        "pairing_config_sha256": config_hash,
        "pairing_config": dict(config),
        "provenance": dict(provenance or {}),
        "coverage": {
            "total_failure_events": total_failures,
            "selected_failure_events": selected_failure_count,
            "failure_event_coverage": (
                selected_failure_count / total_failures if total_failures else 0.0
            ),
            "selected_pairs": len(rows),
            "unique_success_events": len(
                {row["success_event_id"] for row in rows}
            ),
            "rejection_reason_counts": dict(sorted(reason_counts.items())),
        },
        "pair_weight_distribution": _distribution(
            [float(row["pair_weight"]) for row in rows]
        ),
        "failure_events": diagnostic_rows,
    }
    diagnostics["diagnostics_sha256"] = sha256_json(diagnostics)
    return rows, diagnostics


def build_event_pairs(
    event_rows: Sequence[dict[str, Any]],
    episode_rows: Sequence[dict[str, Any]],
    feature_rows: Sequence[dict[str, Any]] = (),
    *,
    config: Mapping[str, Any],
    provenance: Mapping[str, Any] | None = None,
    created_at: str | None = None,
) -> list[dict[str, Any]]:
    """Backward-compatible pair-only API."""

    rows, _ = build_event_pairs_with_diagnostics(
        event_rows,
        episode_rows,
        feature_rows,
        config=config,
        provenance=provenance,
        created_at=created_at,
    )
    return rows


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eve-root", type=Path, required=True)
    parser.add_argument("--event-ledger", type=Path)
    parser.add_argument("--episode-ledger", type=Path)
    parser.add_argument(
        "--features",
        type=Path,
        nargs="+",
        help=(
            "One or more train-calibrated feature tables. Formal train/val "
            "pairing passes both split-specific tables."
        ),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--pairing-version", default=DEFAULT_PAIRING_VERSION)
    parser.add_argument(
        "--matching",
        choices=("one_to_one", "bounded", "mutual_nearest"),
        default="one_to_one",
    )
    parser.add_argument("--max-success-uses", type=int, default=1)
    parser.add_argument("--max-failure-uses", type=int, default=1)
    parser.add_argument("--max-progress-delta", type=float, default=0.12)
    parser.add_argument("--max-pre-state-distance", type=float, default=1.0)
    parser.add_argument("--min-action-divergence", type=float, default=0.1)
    parser.add_argument("--min-pair-weight", type=float, default=0.0)
    parser.add_argument("--tau-progress", type=float, default=0.08)
    parser.add_argument("--tau-state", type=float, default=1.0)
    parser.add_argument("--event-types", nargs="+")
    parser.add_argument("--splits", nargs="+", default=["train"])
    calibration = parser.add_mutually_exclusive_group()
    calibration.add_argument(
        "--fit-calibration",
        type=Path,
        help=(
            "Fit train-only min_action_divergence and tau_state, then atomically "
            "create or exactly reuse this frozen JSON artifact."
        ),
    )
    calibration.add_argument(
        "--calibration",
        type=Path,
        help=(
            "Load a frozen train-only calibration and reject any train-input "
            "or pairing-selection conflict."
        ),
    )
    parser.add_argument(
        "--diagnostics-output",
        type=Path,
        help="Frozen JSON diagnostics; defaults beside the pair ledger.",
    )
    parser.add_argument("--created-at")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> tuple[Path, int, int]:
    eve_root = args.eve_root.resolve()
    event_path = (args.event_ledger or eve_root / "event_meta.jsonl").resolve()
    episode_path = (args.episode_ledger or eve_root / "episode_meta.jsonl").resolve()
    raw_feature_paths = args.features
    if raw_feature_paths is None:
        feature_paths: list[Path] = []
    elif isinstance(raw_feature_paths, Path):
        feature_paths = [raw_feature_paths.resolve()]
    else:
        feature_paths = [Path(path).resolve() for path in raw_feature_paths]
    output_path = (
        args.output or eve_root / "pairs" / f"{args.pairing_version}.jsonl"
    ).resolve()
    diagnostics_output_arg = getattr(args, "diagnostics_output", None)
    diagnostics_path = (
        Path(diagnostics_output_arg).resolve()
        if diagnostics_output_arg is not None
        else output_path.with_suffix(".diagnostics.json")
    )
    config = pairing_config(
        pairing_version=args.pairing_version,
        matching=args.matching,
        max_success_uses=args.max_success_uses,
        max_failure_uses=args.max_failure_uses,
        max_progress_delta=args.max_progress_delta,
        max_pre_state_distance=args.max_pre_state_distance,
        min_action_divergence=args.min_action_divergence,
        min_pair_weight=getattr(args, "min_pair_weight", 0.0),
        tau_progress=getattr(args, "tau_progress", 0.08),
        tau_state=getattr(args, "tau_state", 1.0),
        event_types=args.event_types,
        splits=args.splits,
    )
    event_rows = read_jsonl(event_path)
    episode_rows = read_jsonl(episode_path)
    feature_rows: list[dict[str, Any]] = []
    for feature_path in feature_paths:
        feature_rows.extend(read_feature_table(feature_path))

    calibration_path: Path | None = None
    calibration_payload: dict[str, Any] | None = None
    fit_calibration_arg = getattr(args, "fit_calibration", None)
    load_calibration_arg = getattr(args, "calibration", None)
    if fit_calibration_arg is not None or load_calibration_arg is not None:
        records, _ = _collect_event_records(
            event_rows,
            episode_rows,
            feature_rows,
            config=config,
        )
        if fit_calibration_arg is not None:
            calibration_path = Path(fit_calibration_arg).resolve()
            calibration_payload = fit_pairing_calibration(records, config=config)
            write_json_atomic_frozen(calibration_path, calibration_payload)
        else:
            calibration_path = Path(load_calibration_arg).resolve()
            if not calibration_path.is_file():
                raise FileNotFoundError(calibration_path)
            loaded = json.loads(calibration_path.read_text(encoding="utf-8"))
            if not isinstance(loaded, dict):
                raise ValueError(
                    f"Pair calibration must contain a JSON object: {calibration_path}"
                )
            calibration_payload = loaded
        calibration_payload = validate_pairing_calibration(
            calibration_payload,
            records,
            config=config,
        )
        thresholds = calibration_payload["thresholds"]
        config = {
            **config,
            "min_action_divergence": float(
                thresholds["min_action_divergence"]
            ),
            "tau_state": float(thresholds["tau_state"]),
            "threshold_source": "train_calibrated",
            "calibration_sha256": calibration_payload[
                "calibration_sha256"
            ],
        }
        _validate_pairing_config(config)

    provenance = {
        "method": "thresholded_event_feature_matching",
        "event_ledger": str(event_path),
        "event_ledger_sha256": file_sha256(event_path),
        "episode_ledger": str(episode_path),
        "episode_ledger_sha256": file_sha256(episode_path),
        "feature_tables": [
            {
                "path": str(path),
                "sha256": file_sha256(path),
            }
            for path in feature_paths
        ],
        "feature_table": (
            str(feature_paths[0]) if len(feature_paths) == 1 else None
        ),
        "feature_table_sha256": (
            file_sha256(feature_paths[0]) if len(feature_paths) == 1 else None
        ),
        "configuration": config,
        "calibration": (
            None
            if calibration_payload is None
            else {
                "path": str(calibration_path),
                "sha256": calibration_payload["calibration_sha256"],
                "id": calibration_payload["calibration_id"],
                "input_sha256": calibration_payload["input_sha256"],
            }
        ),
    }
    new_rows, diagnostics = build_event_pairs_with_diagnostics(
        event_rows,
        episode_rows,
        feature_rows,
        config=config,
        provenance=provenance,
        created_at=args.created_at,
    )
    with pair_ledger_lock(output_path):
        existing_rows = read_jsonl(output_path)
        if existing_rows:
            verify_frozen_pair_rows(existing_rows, new_rows)
            appended = 0
        else:
            appended = len(new_rows)
        # Diagnostics are written first. If a stale frozen artifact conflicts,
        # the pair ledger remains untouched; after a crash, an identical retry
        # can safely finish creating the ledger.
        write_json_atomic_frozen(diagnostics_path, diagnostics)
        if not existing_rows:
            write_jsonl_atomic(output_path, new_rows)
    return output_path, len(new_rows), appended


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_path, selected, appended = run(args)
    diagnostics_path = (
        args.diagnostics_output.resolve()
        if args.diagnostics_output is not None
        else output_path.with_suffix(".diagnostics.json")
    )
    print(
        canonical_json(
            {
                "output": str(output_path),
                "diagnostics": str(diagnostics_path),
                "selected_pairs": selected,
                "appended_pairs": appended,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
