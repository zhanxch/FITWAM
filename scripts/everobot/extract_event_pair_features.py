#!/usr/bin/env python3
"""Extract calibrated per-event features for EveRobot event pairing.

The extractor fits normalization statistics only from train-split episodes.
Validation and test extraction must load an existing train calibration.  Event
features are written as immutable, atomically published Parquet or JSONL files
and can be consumed by ``build_event_pairs.py``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


FEATURE_FORMAT = "EveRobotEventPairFeatures"
CALIBRATION_FORMAT = "EveRobotEventPairFeatureCalibration"
SCHEMA_VERSION = "0.1"
ALGORITHM_VERSION = "event_pair_features_v1"
STATE_COLUMN = "observation.state"
ACTION_COLUMN = "action"
STATE_DIM = 23
ACTION_DIM = 22
DEFAULT_PRE_STATE_FRAMES = 4
DEFAULT_STD_EPSILON = 1e-6
ALLOWED_SPLITS = ("train", "val", "test")


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


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number} is invalid JSON") from error
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} must contain a JSON object")
            rows.append(row)
    return rows


def _required_string(row: Mapping[str, Any], field: str, label: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label}.{field} must be a non-empty string")
    return value


def _required_int(
    row: Mapping[str, Any], field: str, label: str, *, minimum: int = 0
) -> int:
    value = row.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label}.{field} must be an integer >= {minimum}")
    return value


def _episode_key(row: Mapping[str, Any], label: str) -> tuple[str, int]:
    return (
        _required_string(row, "dataset_id", label),
        _required_int(row, "episode_index", label),
    )


def _validate_episode(row: Mapping[str, Any], label: str) -> None:
    _required_string(row, "episode_id", label)
    _episode_key(row, label)
    _required_string(row, "dataset_root", label)
    _required_string(row, "task_name", label)
    _required_string(row, "split", label)
    _required_int(row, "length", label, minimum=1)


def index_episodes(
    rows: Sequence[dict[str, Any]],
) -> dict[tuple[str, int], dict[str, Any]]:
    indexed: dict[tuple[str, int], dict[str, Any]] = {}
    episode_ids: set[str] = set()
    for index, row in enumerate(rows):
        label = f"episode_meta[{index}]"
        _validate_episode(row, label)
        key = _episode_key(row, label)
        episode_id = str(row["episode_id"])
        if key in indexed:
            raise ValueError(f"Duplicate episode identity: {key}")
        if episode_id in episode_ids:
            raise ValueError(f"Duplicate episode_id: {episode_id}")
        indexed[key] = row
        episode_ids.add(episode_id)
    return indexed


def locate_episode_parquet(episode: Mapping[str, Any]) -> Path:
    label = f"episode {episode.get('episode_id')}"
    _validate_episode(episode, label)
    dataset_root = Path(str(episode["dataset_root"])).expanduser().resolve()
    episode_index = int(episode["episode_index"])
    info_path = dataset_root / "meta" / "info.json"
    candidates: list[Path] = []
    if info_path.is_file():
        info = json.loads(info_path.read_text(encoding="utf-8"))
        if not isinstance(info, dict):
            raise ValueError(f"{info_path} must contain a JSON object")
        template = info.get(
            "data_path",
            "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        )
        chunks_size = info.get("chunks_size", 1000)
        if not isinstance(template, str) or not template:
            raise ValueError(f"{info_path}: data_path must be a non-empty string")
        if isinstance(chunks_size, bool) or not isinstance(chunks_size, int):
            raise ValueError(f"{info_path}: chunks_size must be a positive integer")
        if chunks_size <= 0:
            raise ValueError(f"{info_path}: chunks_size must be a positive integer")
        try:
            relative = template.format(
                episode_chunk=episode_index // chunks_size,
                episode_index=episode_index,
            )
        except (KeyError, ValueError) as error:
            raise ValueError(f"{info_path}: invalid data_path template") from error
        candidate = dataset_root / relative
        if candidate.is_file():
            candidates.append(candidate.resolve())

    candidates.extend(
        path.resolve()
        for path in sorted(
            (dataset_root / "data").glob(
                f"chunk-*/episode_{episode_index:06d}.parquet"
            )
        )
        if path.is_file() and path.resolve() not in candidates
    )
    if not candidates:
        raise FileNotFoundError(
            f"No Parquet file for episode {episode_index} under {dataset_root}"
        )
    if len(candidates) != 1:
        raise ValueError(
            f"Episode {episode_index} resolves to multiple Parquet files: "
            f"{[str(path) for path in candidates]}"
        )
    return candidates[0]


def _list_column_to_matrix(column: Any, *, name: str, dim: int, path: Path) -> np.ndarray:
    try:
        import pyarrow as pa
    except ImportError as error:
        raise RuntimeError("pyarrow is required for event feature extraction") from error

    combined = column.combine_chunks()
    column_type = combined.type
    if not (
        pa.types.is_fixed_size_list(column_type)
        or pa.types.is_list(column_type)
        or pa.types.is_large_list(column_type)
    ):
        raise ValueError(
            f"{path}:{name} must be a list or fixed-size-list column, "
            f"got {column_type}"
        )
    if pa.types.is_fixed_size_list(column_type) and column_type.list_size != dim:
        raise ValueError(
            f"{path}:{name} fixed-size-list dimension "
            f"{column_type.list_size} != {dim}"
        )
    values = combined.to_pylist()
    for row_index, value in enumerate(values):
        if not isinstance(value, list) or len(value) != dim:
            actual = None if value is None else len(value)
            raise ValueError(
                f"{path}:{name}[{row_index}] dimension {actual} != {dim}"
            )
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.shape != (len(values), dim):
        raise ValueError(f"{path}:{name} has invalid shape {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{path}:{name} contains NaN or infinite values")
    return matrix


def load_episode_arrays(episode: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError("pyarrow is required for event feature extraction") from error

    path = locate_episode_parquet(episode)
    try:
        table = pq.read_table(path, columns=[STATE_COLUMN, ACTION_COLUMN])
    except Exception as error:
        raise ValueError(
            f"Could not read {STATE_COLUMN!r} and {ACTION_COLUMN!r} from {path}"
        ) from error
    missing = {STATE_COLUMN, ACTION_COLUMN} - set(table.column_names)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    expected_length = _required_int(episode, "length", "episode", minimum=1)
    if table.num_rows != expected_length:
        raise ValueError(
            f"{path} row count {table.num_rows} != episode length {expected_length}"
        )
    states = _list_column_to_matrix(
        table[STATE_COLUMN], name=STATE_COLUMN, dim=STATE_DIM, path=path
    )
    actions = _list_column_to_matrix(
        table[ACTION_COLUMN], name=ACTION_COLUMN, dim=ACTION_DIM, path=path
    )
    return states, actions


@dataclass
class RunningMoments:
    dim: int
    count: int = 0
    total: np.ndarray | None = None
    total_square: np.ndarray | None = None

    def update(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.dim or len(values) == 0:
            raise ValueError(f"Expected non-empty [frames, {self.dim}] values")
        if not np.isfinite(values).all():
            raise ValueError("Calibration values must be finite")
        if self.total is None:
            self.total = np.zeros(self.dim, dtype=np.float64)
            self.total_square = np.zeros(self.dim, dtype=np.float64)
        self.count += len(values)
        self.total += values.sum(axis=0, dtype=np.float64)
        self.total_square += np.square(values).sum(axis=0, dtype=np.float64)

    def finalize(self, epsilon: float) -> tuple[np.ndarray, np.ndarray, list[int]]:
        if self.count <= 0 or self.total is None or self.total_square is None:
            raise ValueError("No calibration frames were accumulated")
        mean = self.total / self.count
        variance = np.maximum(self.total_square / self.count - np.square(mean), 0.0)
        raw_std = np.sqrt(variance)
        constant = np.flatnonzero(raw_std < epsilon).astype(int).tolist()
        std = raw_std.copy()
        std[raw_std < epsilon] = 1.0
        return mean, std, constant


def _array_input_digest(
    digest: Any,
    episode: Mapping[str, Any],
    states: np.ndarray,
    actions: np.ndarray,
) -> None:
    identity = {
        "dataset_id": episode["dataset_id"],
        "episode_index": episode["episode_index"],
        "episode_id": episode["episode_id"],
        "length": episode["length"],
        "split": episode["split"],
    }
    digest.update(canonical_json(identity).encode("utf-8"))
    digest.update(b"\0")
    for values in (states, actions):
        array = np.ascontiguousarray(values, dtype="<f8")
        digest.update(canonical_json(list(array.shape)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(array.tobytes(order="C"))


def fit_train_calibration(
    episode_rows: Sequence[dict[str, Any]],
    *,
    episode_ledger_sha256: str,
    std_epsilon: float = DEFAULT_STD_EPSILON,
) -> dict[str, Any]:
    if not math.isfinite(std_epsilon) or std_epsilon <= 0.0:
        raise ValueError("std_epsilon must be a finite positive number")
    train_rows = [
        row for row in episode_rows if _required_string(row, "split", "episode") == "train"
    ]
    if not train_rows:
        raise ValueError("No train episodes are available for calibration")
    train_rows.sort(key=lambda row: _episode_key(row, "episode"))
    state_moments = RunningMoments(STATE_DIM)
    action_moments = RunningMoments(ACTION_DIM)
    input_digest = hashlib.sha256(b"EveRobotEventPairCalibrationInputV1\0")
    episode_ids: list[str] = []
    for episode in train_rows:
        _validate_episode(episode, f"episode {episode.get('episode_id')}")
        states, actions = load_episode_arrays(episode)
        state_moments.update(states)
        action_moments.update(actions)
        _array_input_digest(input_digest, episode, states, actions)
        episode_ids.append(str(episode["episode_id"]))
    state_mean, state_std, constant_state = state_moments.finalize(std_epsilon)
    action_mean, action_std, constant_action = action_moments.finalize(std_epsilon)
    payload: dict[str, Any] = {
        "format": CALIBRATION_FORMAT,
        "schema_version": SCHEMA_VERSION,
        "algorithm_version": ALGORITHM_VERSION,
        "fit_split": "train",
        "state_column": STATE_COLUMN,
        "action_column": ACTION_COLUMN,
        "state_dim": STATE_DIM,
        "action_dim": ACTION_DIM,
        "std_epsilon": float(std_epsilon),
        "episode_ledger_sha256": episode_ledger_sha256,
        "calibration_input_sha256": input_digest.hexdigest(),
        "episode_ids": episode_ids,
        "num_episodes": len(episode_ids),
        "num_frames": state_moments.count,
        "state_mean": state_mean.tolist(),
        "state_std": state_std.tolist(),
        "action_mean": action_mean.tolist(),
        "action_std": action_std.tolist(),
        "constant_state_dimensions": constant_state,
        "constant_action_dimensions": constant_action,
    }
    payload["calibration_sha256"] = sha256_json(payload)
    payload["calibration_id"] = (
        f"event-pair-calibration-{payload['calibration_sha256'][:16]}"
    )
    return payload


def validate_calibration(payload: Mapping[str, Any]) -> dict[str, Any]:
    calibration = dict(payload)
    if calibration.get("format") != CALIBRATION_FORMAT:
        raise ValueError("Unrecognized event-pair feature calibration format")
    if calibration.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported event-pair feature calibration schema")
    if calibration.get("algorithm_version") != ALGORITHM_VERSION:
        raise ValueError("Unsupported event-pair feature calibration algorithm")
    if calibration.get("fit_split") != "train":
        raise ValueError("Calibration must have been fit on split='train'")
    if calibration.get("state_dim") != STATE_DIM:
        raise ValueError(f"Calibration state_dim must be {STATE_DIM}")
    if calibration.get("action_dim") != ACTION_DIM:
        raise ValueError(f"Calibration action_dim must be {ACTION_DIM}")
    expected_hash = calibration.get("calibration_sha256")
    if not isinstance(expected_hash, str):
        raise ValueError("Calibration is missing calibration_sha256")
    unhashed = {
        key: value
        for key, value in calibration.items()
        if key not in {"calibration_sha256", "calibration_id"}
    }
    actual_hash = sha256_json(unhashed)
    if expected_hash != actual_hash:
        raise ValueError("Calibration hash does not match its payload")
    expected_id = f"event-pair-calibration-{actual_hash[:16]}"
    if calibration.get("calibration_id") != expected_id:
        raise ValueError("Calibration ID does not match its payload")
    for field, dim in (
        ("state_mean", STATE_DIM),
        ("state_std", STATE_DIM),
        ("action_mean", ACTION_DIM),
        ("action_std", ACTION_DIM),
    ):
        try:
            values = np.asarray(calibration.get(field), dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Calibration {field} must contain {dim} finite values"
            ) from error
        if values.shape != (dim,) or not np.isfinite(values).all():
            raise ValueError(f"Calibration {field} must contain {dim} finite values")
        if field.endswith("_std") and np.any(values <= 0.0):
            raise ValueError(f"Calibration {field} must be positive")
    return calibration


def load_calibration(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return validate_calibration(payload)


def _publish_temp_without_overwrite(temporary: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(temporary, output)
    except FileExistsError:
        raise FileExistsError(f"Refusing to overwrite immutable artifact: {output}")
    directory_fd = os.open(output.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def write_bytes_atomic_immutable(path: Path, content: bytes) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        _publish_temp_without_overwrite(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json_atomic_immutable(path: Path, payload: Mapping[str, Any]) -> None:
    write_bytes_atomic_immutable(path, (canonical_json(payload) + "\n").encode("utf-8"))


def write_jsonl_atomic_immutable(
    path: Path, rows: Iterable[Mapping[str, Any]]
) -> None:
    content = "".join(canonical_json(dict(row)) + "\n" for row in rows)
    write_bytes_atomic_immutable(path, content.encode("utf-8"))


def write_parquet_atomic_immutable(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    metadata: Mapping[str, str],
) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError("pyarrow is required to write Parquet features") from error
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist([dict(row) for row in rows])
    encoded_metadata = {
        str(key).encode("utf-8"): str(value).encode("utf-8")
        for key, value in metadata.items()
    }
    table = table.replace_schema_metadata(
        {**(table.schema.metadata or {}), **encoded_metadata}
    )
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp.parquet", dir=path.parent
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        pq.write_table(table, temporary)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        _publish_temp_without_overwrite(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _event_interval(
    event: Mapping[str, Any], *, episode_length: int, label: str
) -> tuple[int, int]:
    start = _required_int(event, "core_start_frame", label)
    end = _required_int(event, "core_end_frame", label)
    if start >= end:
        raise ValueError(f"{label} core interval must be non-empty")
    if end > episode_length:
        raise ValueError(
            f"{label} core interval [{start}, {end}) exceeds episode length "
            f"{episode_length}"
        )
    return start, end


def extract_feature_rows(
    event_rows: Sequence[dict[str, Any]],
    episode_rows: Sequence[dict[str, Any]],
    calibration: Mapping[str, Any],
    *,
    split: str,
    episode_ledger_sha256: str,
    event_ledger_sha256: str,
    event_types: frozenset[str] | None = None,
    pre_state_frames: int = DEFAULT_PRE_STATE_FRAMES,
) -> tuple[list[dict[str, Any]], str, dict[str, Any]]:
    if split not in ALLOWED_SPLITS:
        raise ValueError(f"split must be one of {ALLOWED_SPLITS}")
    if isinstance(pre_state_frames, bool) or not isinstance(pre_state_frames, int):
        raise ValueError("pre_state_frames must be an integer")
    if pre_state_frames <= 0:
        raise ValueError("pre_state_frames must be positive")
    calibration = validate_calibration(calibration)
    episodes = index_episodes(episode_rows)
    parameters = {
        "algorithm_version": ALGORITHM_VERSION,
        "state_column": STATE_COLUMN,
        "action_column": ACTION_COLUMN,
        "state_dim": STATE_DIM,
        "action_dim": ACTION_DIM,
        "pre_state_frames": pre_state_frames,
        "pre_state_fallback": "core_start_frame_when_no_prior_frame",
        "action_summary": "standardized_mean_std_last_minus_first",
        "split": split,
        "event_types": (
            None if event_types is None else sorted(event_types)
        ),
    }
    method_payload = {
        "input": {
            "episode_ledger_sha256": episode_ledger_sha256,
            "event_ledger_sha256": event_ledger_sha256,
        },
        "calibration_sha256": calibration["calibration_sha256"],
        "parameters": parameters,
    }
    method_id = f"event-pair-features-{sha256_json(method_payload)[:16]}"
    provenance = {
        "method_id": method_id,
        **method_payload,
    }
    state_mean = np.asarray(calibration["state_mean"], dtype=np.float64)
    state_std = np.asarray(calibration["state_std"], dtype=np.float64)
    action_mean = np.asarray(calibration["action_mean"], dtype=np.float64)
    action_std = np.asarray(calibration["action_std"], dtype=np.float64)

    selected: list[tuple[dict[str, Any], dict[str, Any]]] = []
    seen_event_ids: set[str] = set()
    for index, event in enumerate(event_rows):
        label = f"event_meta[{index}]"
        event_id = _required_string(event, "event_id", label)
        if event_id in seen_event_ids:
            raise ValueError(f"Duplicate event_id: {event_id}")
        seen_event_ids.add(event_id)
        event_split = _required_string(event, "split", label)
        if event_split != split:
            continue
        if event_types is not None:
            event_type = _required_string(event, "event_type", label)
            if event_type not in event_types:
                continue
        key = _episode_key(event, label)
        episode = episodes.get(key)
        if episode is None:
            raise ValueError(f"{label} references missing episode {key}")
        episode_split = _required_string(episode, "split", "episode")
        if episode_split != event_split:
            raise ValueError(
                f"{label}.split {event_split!r} conflicts with episode split "
                f"{episode_split!r}"
            )
        event_episode_id = _required_string(event, "episode_id", label)
        if event_episode_id != episode["episode_id"]:
            raise ValueError(f"{label}.episode_id conflicts with linked episode")
        task_name = _required_string(event, "task_name", label)
        episode_task = episode.get("task_name")
        if episode_task is not None and episode_task != task_name:
            raise ValueError(f"{label}.task_name conflicts with linked episode")
        selected.append((event, episode))
    if not selected:
        raise ValueError(f"No event rows belong to split={split!r}")

    arrays_cache: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}
    output_rows: list[dict[str, Any]] = []
    for event, episode in sorted(selected, key=lambda item: str(item[0]["event_id"])):
        key = _episode_key(episode, "episode")
        if key not in arrays_cache:
            arrays_cache[key] = load_episode_arrays(episode)
        states, actions = arrays_cache[key]
        episode_length = int(episode["length"])
        label = f"event {event['event_id']}"
        core_start, core_end = _event_interval(
            event, episode_length=episode_length, label=label
        )
        pre_start = max(0, core_start - pre_state_frames)
        pre_end = core_start
        if pre_start == pre_end:
            pre_start = core_start
            pre_end = core_start + 1
        if pre_end > episode_length:
            raise ValueError(f"{label} has no available pre-state frame")
        pre_states = states[pre_start:pre_end]
        if len(pre_states) < 1:
            raise ValueError(f"{label} has no available pre-state frame")
        standardized_pre_states = (pre_states - state_mean) / state_std
        pre_state_embedding = standardized_pre_states.mean(axis=0)

        core_actions = actions[core_start:core_end]
        if len(core_actions) < 1:
            raise ValueError(f"{label} has an empty action interval")
        standardized_actions = (core_actions - action_mean) / action_std
        action_embedding = np.concatenate(
            (
                standardized_actions.mean(axis=0),
                standardized_actions.std(axis=0, ddof=0),
                standardized_actions[-1] - standardized_actions[0],
            )
        )
        if pre_state_embedding.shape != (STATE_DIM,):
            raise AssertionError("Unexpected pre-state embedding dimension")
        if action_embedding.shape != (ACTION_DIM * 3,):
            raise AssertionError("Unexpected action embedding dimension")
        if not (
            np.isfinite(pre_state_embedding).all()
            and np.isfinite(action_embedding).all()
        ):
            raise ValueError(f"{label} produced non-finite features")

        row_without_hash: dict[str, Any] = {
            "format": FEATURE_FORMAT,
            "schema_version": SCHEMA_VERSION,
            "method_id": method_id,
            "event_id": str(event["event_id"]),
            "episode_id": str(episode["episode_id"]),
            "dataset_id": str(episode["dataset_id"]),
            "episode_index": int(episode["episode_index"]),
            "task_name": str(event["task_name"]),
            "split": split,
            "progress": (core_start + core_end) / (2.0 * episode_length),
            "pre_state_embedding": pre_state_embedding.tolist(),
            "action_embedding": action_embedding.tolist(),
            "pre_state_window": [pre_start, pre_end],
            "core_interval": [core_start, core_end],
            "calibration_id": calibration["calibration_id"],
            "calibration_sha256": calibration["calibration_sha256"],
            "episode_ledger_sha256": episode_ledger_sha256,
            "event_ledger_sha256": event_ledger_sha256,
            "provenance": provenance,
        }
        row_without_hash["feature_sha256"] = sha256_json(row_without_hash)
        output_rows.append(row_without_hash)
    return output_rows, method_id, provenance


def _resolve_calibration(
    *,
    split: str,
    fit_calibration_path: Path | None,
    calibration_path: Path | None,
    episode_rows: Sequence[dict[str, Any]],
    episode_ledger_sha256: str,
    std_epsilon: float,
) -> tuple[dict[str, Any], Path]:
    if (fit_calibration_path is None) == (calibration_path is None):
        raise ValueError(
            "Specify exactly one of fit_calibration_path or calibration_path"
        )
    if fit_calibration_path is not None:
        if split != "train":
            raise ValueError("Calibration fitting is allowed only for split='train'")
        calibration = fit_train_calibration(
            episode_rows,
            episode_ledger_sha256=episode_ledger_sha256,
            std_epsilon=std_epsilon,
        )
        resolved = fit_calibration_path.expanduser().resolve()
        write_json_atomic_immutable(resolved, calibration)
        return calibration, resolved
    assert calibration_path is not None
    resolved = calibration_path.expanduser().resolve()
    return load_calibration(resolved), resolved


def run(
    *,
    eve_root: Path,
    split: str,
    output: Path | None,
    fit_calibration_path: Path | None,
    calibration_path: Path | None,
    episode_ledger: Path | None = None,
    event_ledger: Path | None = None,
    event_types: Sequence[str] | None = None,
    jsonl_output: Path | None = None,
    pre_state_frames: int = DEFAULT_PRE_STATE_FRAMES,
    std_epsilon: float = DEFAULT_STD_EPSILON,
) -> dict[str, Any]:
    eve_root = eve_root.expanduser().resolve()
    episode_path = (
        episode_ledger.expanduser().resolve()
        if episode_ledger is not None
        else eve_root / "episode_meta.jsonl"
    )
    event_path = (
        event_ledger.expanduser().resolve()
        if event_ledger is not None
        else eve_root / "event_meta.jsonl"
    )
    episode_hash = file_sha256(episode_path)
    event_hash = file_sha256(event_path)
    episode_rows = read_jsonl(episode_path)
    event_rows = read_jsonl(event_path)
    index_episodes(episode_rows)
    normalized_event_types = (
        None
        if event_types is None
        else frozenset(
            _required_string(
                {"event_type": event_type},
                "event_type",
                f"event_types[{index}]",
            )
            for index, event_type in enumerate(event_types)
        )
    )
    if normalized_event_types is not None and not normalized_event_types:
        raise ValueError("event_types must contain at least one value")
    calibration, resolved_calibration = _resolve_calibration(
        split=split,
        fit_calibration_path=fit_calibration_path,
        calibration_path=calibration_path,
        episode_rows=episode_rows,
        episode_ledger_sha256=episode_hash,
        std_epsilon=std_epsilon,
    )
    rows, method_id, provenance = extract_feature_rows(
        event_rows,
        episode_rows,
        calibration,
        split=split,
        episode_ledger_sha256=episode_hash,
        event_ledger_sha256=event_hash,
        event_types=normalized_event_types,
        pre_state_frames=pre_state_frames,
    )
    output_path = (
        output.expanduser().resolve()
        if output is not None
        else eve_root / "features" / f"{method_id}.parquet"
    )
    suffix = output_path.suffix.lower()
    resolved_jsonl = (
        jsonl_output.expanduser().resolve()
        if jsonl_output is not None
        else output_path.with_suffix(".jsonl") if suffix == ".parquet" else None
    )
    if resolved_jsonl == output_path:
        raise ValueError("jsonl_output must differ from output")
    if resolved_jsonl is not None and resolved_jsonl.suffix.lower() not in {
        ".jsonl",
        ".ndjson",
    }:
        raise ValueError("jsonl_output must end in .jsonl or .ndjson")
    immutable_outputs = [output_path]
    if resolved_jsonl is not None:
        immutable_outputs.append(resolved_jsonl)
    existing = [path for path in immutable_outputs if path.exists()]
    if existing:
        raise FileExistsError(
            f"Refusing to overwrite immutable artifact: {existing[0]}"
        )

    if suffix == ".parquet":
        write_parquet_atomic_immutable(
            output_path,
            rows,
            metadata={
                "format": FEATURE_FORMAT,
                "schema_version": SCHEMA_VERSION,
                "method_id": method_id,
                "provenance": canonical_json(provenance),
                "rows_sha256": sha256_json(rows),
            },
        )
    elif suffix in {".jsonl", ".ndjson"}:
        write_jsonl_atomic_immutable(output_path, rows)
    else:
        raise ValueError("output must end in .parquet, .jsonl, or .ndjson")
    if resolved_jsonl is not None:
        write_jsonl_atomic_immutable(resolved_jsonl, rows)
    pairing_input = resolved_jsonl or output_path
    return {
        "output": str(output_path),
        "jsonl_output": None if resolved_jsonl is None else str(resolved_jsonl),
        "pairing_input": str(pairing_input),
        "calibration": str(resolved_calibration),
        "calibration_sha256": calibration["calibration_sha256"],
        "method_id": method_id,
        "split": split,
        "num_events": len(rows),
        "rows_sha256": sha256_json(rows),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eve-root", type=Path, required=True)
    parser.add_argument("--split", choices=ALLOWED_SPLITS, required=True)
    parser.add_argument("--episode-ledger", type=Path)
    parser.add_argument("--event-ledger", type=Path)
    parser.add_argument(
        "--event-types",
        nargs="+",
        default=None,
        help=(
            "Only extract these event_type values. Use interaction_candidate "
            "for state-line event pairing so coarse episode-level failure "
            "records are not treated as candidate windows."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Immutable .parquet or .jsonl output; defaults to versioned Parquet.",
    )
    parser.add_argument(
        "--jsonl-output",
        type=Path,
        help=(
            "Immutable JSONL companion for build_event_pairs.py. Parquet output "
            "creates a same-stem companion by default."
        ),
    )
    calibration = parser.add_mutually_exclusive_group(required=True)
    calibration.add_argument(
        "--fit-calibration",
        type=Path,
        help="Fit train-only statistics and atomically create this JSON file.",
    )
    calibration.add_argument(
        "--calibration",
        type=Path,
        help="Load an existing train-only calibration JSON file.",
    )
    parser.add_argument(
        "--pre-state-frames", type=int, default=DEFAULT_PRE_STATE_FRAMES
    )
    parser.add_argument("--std-epsilon", type=float, default=DEFAULT_STD_EPSILON)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = run(
        eve_root=args.eve_root,
        split=args.split,
        output=args.output,
        fit_calibration_path=args.fit_calibration,
        calibration_path=args.calibration,
        episode_ledger=args.episode_ledger,
        event_ledger=args.event_ledger,
        event_types=args.event_types,
        jsonl_output=args.jsonl_output,
        pre_state_frames=args.pre_state_frames,
        std_epsilon=args.std_epsilon,
    )
    print(json.dumps(result, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
