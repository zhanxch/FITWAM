#!/usr/bin/env python3
"""Train the Offline Steer trajectory teacher and export frozen pair targets.

The script consumes EveRobot episode/event ledgers plus an immutable event-pair
ledger.  It reads variable-length action trajectories directly from the
referenced LeRobot Parquet files, keeps train/validation episodes disjoint, and
trains ``TrajectoryTeacher`` with an augmented InfoNCE objective:

* two independently masked/jittered views of the same trajectory are positives;
* the paired trajectory with the opposite outcome is an explicit hard negative;
* other trajectories in the batch are ordinary negatives.

The teacher never consumes observations or language.  Its normalized success
and failure embeddings are exported as fixed targets for the observation-side
Offline Steer student.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from functools import partial
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import tempfile
from typing import Any, Callable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
for source_root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))


DEFAULT_ACTION_DIM = 22
DEFAULT_HIDDEN_DIM = 256
DEFAULT_EMBEDDING_DIM = 256
DEFAULT_NUM_LAYERS = 2
DEFAULT_NUM_HEADS = 4
DEFAULT_DROPOUT = 0.1
CHECKPOINT_SCHEMA_VERSION = 2
METRIC_FIELDS = (
    "epoch",
    "train_loss",
    "val_loss",
    "val_positive_cosine",
    "val_negative_cosine",
    "val_cosine_gap",
    "val_embedding_variance",
    "val_top1_paired_retrieval_accuracy",
    "learning_rate",
)


def canonical_json(payload: Any) -> str:
    return json.dumps(
        payload,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} must contain a JSON object")
            rows.append(row)
    return rows


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(canonical_json(dict(row)))
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _write_jsonl_atomic(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(canonical_json(dict(row)))
                stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _write_metrics_csv_atomic(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=METRIC_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _finite_weight(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number in (0, 1]")
    result = float(value)
    if not math.isfinite(result) or not 0.0 < result <= 1.0:
        raise ValueError(f"{label} must be a finite number in (0, 1]")
    return result


def episode_key(row: Mapping[str, Any]) -> str:
    dataset_id = row.get("dataset_id")
    episode_index = row.get("episode_index")
    if (
        isinstance(dataset_id, str)
        and dataset_id.strip()
        and isinstance(episode_index, int)
        and not isinstance(episode_index, bool)
        and episode_index >= 0
    ):
        return f"{dataset_id}:episode:{episode_index}"
    episode_id = row.get("episode_id")
    if isinstance(episode_id, str) and episode_id.strip():
        return episode_id
    raise ValueError("Episode requires dataset_id/index or a non-empty episode_id")


def resolve_event_interval(
    event: Mapping[str, Any],
    *,
    prefer_core: bool = True,
) -> tuple[int, int]:
    """Return a validated half-open event interval.

    ``core_*`` excludes context padding introduced by state-line extraction.
    It is preferred when available; otherwise the full candidate window is
    used.
    """

    has_core = event.get("core_start_frame") is not None
    has_core_end = event.get("core_end_frame") is not None
    if has_core != has_core_end:
        raise ValueError(
            f"Event {event.get('event_id')} must provide both core interval bounds"
        )
    if prefer_core and has_core:
        start = _nonnegative_int(event["core_start_frame"], "core_start_frame")
        end = _nonnegative_int(event["core_end_frame"], "core_end_frame")
    else:
        start = _nonnegative_int(event.get("start_frame"), "start_frame")
        end = _nonnegative_int(event.get("end_frame"), "end_frame")
    if start >= end:
        raise ValueError(
            f"Event {event.get('event_id')} has empty or reversed interval "
            f"[{start}, {end})"
        )
    return start, end


def _index_unique(
    rows: Sequence[dict[str, Any]],
    field: str,
    *,
    label: str,
) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for row in rows:
        identity = _nonempty_string(row.get(field), f"{label}.{field}")
        if identity in index:
            raise ValueError(f"Duplicate {field} in {label}: {identity}")
        index[identity] = row
    return index


def build_pair_records(
    episode_rows: Sequence[dict[str, Any]],
    event_rows: Sequence[dict[str, Any]],
    pair_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Join and validate the three EveRobot ledgers without loading actions."""

    episodes_by_id = _index_unique(episode_rows, "episode_id", label="episode ledger")
    events_by_id = _index_unique(event_rows, "event_id", label="event ledger")
    records: list[dict[str, Any]] = []
    seen_pair_ids: set[str] = set()

    for pair in pair_rows:
        pair_id = _nonempty_string(pair.get("pair_id"), "pair_id")
        if pair_id in seen_pair_ids:
            raise ValueError(f"Duplicate pair_id: {pair_id}")
        seen_pair_ids.add(pair_id)

        success_event_id = _nonempty_string(
            pair.get("success_event_id"), f"{pair_id}.success_event_id"
        )
        failure_event_id = _nonempty_string(
            pair.get("failure_event_id"), f"{pair_id}.failure_event_id"
        )
        try:
            success_event = events_by_id[success_event_id]
            failure_event = events_by_id[failure_event_id]
        except KeyError as error:
            raise ValueError(
                f"Pair {pair_id} references missing event {error.args[0]}"
            ) from error

        success_episode_id = _nonempty_string(
            success_event.get("episode_id"), f"{success_event_id}.episode_id"
        )
        failure_episode_id = _nonempty_string(
            failure_event.get("episode_id"), f"{failure_event_id}.episode_id"
        )
        try:
            success_episode = episodes_by_id[success_episode_id]
            failure_episode = episodes_by_id[failure_episode_id]
        except KeyError as error:
            raise ValueError(
                f"Pair {pair_id} references missing episode {error.args[0]}"
            ) from error

        if success_episode.get("episode_outcome") != "success":
            raise ValueError(f"Pair {pair_id} success episode is not marked success")
        if failure_episode.get("episode_outcome") != "failure":
            raise ValueError(f"Pair {pair_id} failure episode is not marked failure")
        for event, episode, expected in (
            (success_event, success_episode, "success"),
            (failure_event, failure_episode, "failure"),
        ):
            if event.get("dataset_id") != episode.get("dataset_id"):
                raise ValueError(f"Pair {pair_id} event/episode dataset mismatch")
            if event.get("episode_index") != episode.get("episode_index"):
                raise ValueError(f"Pair {pair_id} event/episode index mismatch")
            event_outcome = event.get("event_outcome")
            if event_outcome not in {None, "unknown", expected}:
                raise ValueError(
                    f"Pair {pair_id} has incompatible event outcome {event_outcome!r}"
                )

        records.append(
            {
                "pair_id": pair_id,
                "split": pair.get("split"),
                "pair_weight": _finite_weight(
                    pair.get("pair_weight", 1.0), f"{pair_id}.pair_weight"
                ),
                "success_event_id": success_event_id,
                "failure_event_id": failure_event_id,
                "success_event": success_event,
                "failure_event": failure_event,
                "success_episode": success_episode,
                "failure_episode": failure_episode,
                "success_episode_key": episode_key(success_episode),
                "failure_episode_key": episode_key(failure_episode),
            }
        )
    if not records:
        raise ValueError("Pair ledger contains no trainable pairs")
    return records


def split_by_declared_ledger(
    records: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Use EveRobot event/pair splits as the sole formal split authority."""

    train: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []
    for record in records:
        pair_id = str(record["pair_id"])
        pair_split = record.get("split")
        success_split = str(
            record["success_event"].get(
                "split",
                record["success_episode"].get("split", "train"),
            )
        )
        failure_split = str(
            record["failure_event"].get(
                "split",
                record["failure_episode"].get("split", "train"),
            )
        )
        if success_split != failure_split:
            raise ValueError(
                f"Pair {pair_id} crosses event splits: "
                f"{success_split!r} vs {failure_split!r}"
            )
        if pair_split is None:
            raise ValueError(
                f"Pair {pair_id} has no declared split. Rebuild the pair ledger "
                "from split-aware EveRobot events, or explicitly use "
                "`--allow-generated-split` for a legacy smoke test."
            )
        if str(pair_split) != success_split:
            raise ValueError(
                f"Pair {pair_id}.split={pair_split!r} conflicts with event "
                f"split={success_split!r}"
            )
        if pair_split == "train":
            train.append(record)
        elif pair_split == "val":
            validation.append(record)
        elif pair_split == "test":
            raise ValueError(
                f"Pair {pair_id} belongs to test and cannot train the Teacher."
            )
        else:
            raise ValueError(f"Pair {pair_id} has invalid split {pair_split!r}")

    if not train or not validation:
        raise ValueError(
            "Declared pair ledger must contain non-empty train and val pairs."
        )
    train_episodes = {
        str(record[key])
        for record in train
        for key in ("success_episode_key", "failure_episode_key")
    }
    val_episodes = {
        str(record[key])
        for record in validation
        for key in ("success_episode_key", "failure_episode_key")
    }
    overlap = train_episodes & val_episodes
    if overlap:
        raise ValueError(
            f"Declared EveRobot split leaks episodes: {sorted(overlap)}"
        )
    return train, validation


class _DisjointSet:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, item: str) -> str:
        self.parent.setdefault(item, item)
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[max(left_root, right_root)] = min(left_root, right_root)


def split_episode_disjoint(
    records: Sequence[dict[str, Any]],
    *,
    val_fraction: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split whole connected episode components into train and validation.

    A connected component is necessary because bounded matching may reuse an
    episode in several pairs.  Assigning individual pairs would leak that
    episode's trajectory into both splits.
    """

    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be in (0, 1)")
    if len(records) < 2:
        raise ValueError("At least two pairs are required for train/validation")

    disjoint = _DisjointSet()
    for record in records:
        disjoint.union(
            str(record["success_episode_key"]),
            str(record["failure_episode_key"]),
        )

    components: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        root = disjoint.find(str(record["success_episode_key"]))
        components.setdefault(root, []).append(record)
    if len(components) < 2:
        raise ValueError(
            "All pairs share one connected episode component; an "
            "episode-disjoint validation split is impossible"
        )

    ordered_components = sorted(
        components.values(),
        key=lambda component: hashlib.sha256(
            (
                f"{seed}|"
                + "|".join(sorted(str(row["pair_id"]) for row in component))
            ).encode("utf-8")
        ).hexdigest(),
    )
    target = max(1, min(len(records) - 1, round(len(records) * val_fraction)))
    validation: list[dict[str, Any]] = []
    remaining = len(records)
    for component in ordered_components:
        component_size = len(component)
        if remaining - component_size <= 0:
            continue
        current_error = abs(target - len(validation))
        proposed_error = abs(target - (len(validation) + component_size))
        if not validation or proposed_error <= current_error:
            validation.extend(component)
            remaining -= component_size
    if not validation:
        smallest = min(ordered_components, key=len)
        if len(smallest) == len(records):
            raise ValueError("Could not construct a non-empty disjoint validation split")
        validation.extend(smallest)

    validation_ids = {str(record["pair_id"]) for record in validation}
    train = [
        record for record in records if str(record["pair_id"]) not in validation_ids
    ]
    train_episodes = {
        str(record[key])
        for record in train
        for key in ("success_episode_key", "failure_episode_key")
    }
    val_episodes = {
        str(record[key])
        for record in validation
        for key in ("success_episode_key", "failure_episode_key")
    }
    overlap = train_episodes & val_episodes
    if overlap:
        raise AssertionError(f"Episode split leakage: {sorted(overlap)}")
    if not train or not validation:
        raise ValueError("Both train and validation splits must be non-empty")
    return train, validation


def locate_episode_parquet(episode: Mapping[str, Any]) -> Path:
    dataset_root = Path(
        _nonempty_string(episode.get("dataset_root"), "episode.dataset_root")
    ).expanduser()
    episode_index = _nonnegative_int(
        episode.get("episode_index"), "episode.episode_index"
    )
    info_path = dataset_root / "meta" / "info.json"
    info: dict[str, Any] = {}
    if info_path.exists():
        info = json.loads(info_path.read_text(encoding="utf-8"))
    chunks_size = int(info.get("chunks_size", 1000))
    if chunks_size <= 0:
        raise ValueError(f"{info_path} has invalid chunks_size={chunks_size}")
    episode_chunk = episode_index // chunks_size
    template = str(
        info.get(
            "data_path",
            "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        )
    )
    try:
        relative_path = template.format(
            episode_chunk=episode_chunk,
            chunk_index=episode_chunk,
            episode_index=episode_index,
        )
    except (KeyError, ValueError) as error:
        raise ValueError(f"Unsupported LeRobot data_path template: {template}") from error
    candidate = dataset_root / relative_path
    if candidate.is_file():
        return candidate.resolve()
    matches = sorted(
        (dataset_root / "data").glob(
            f"chunk-*/episode_{episode_index:06d}.parquet"
        )
    )
    if len(matches) == 1:
        return matches[0].resolve()
    if not matches:
        raise FileNotFoundError(
            f"No Parquet file for episode {episode_index} under {dataset_root}"
        )
    raise ValueError(
        f"Multiple Parquet files match episode {episode_index}: {matches}"
    )


def read_event_actions(
    event: Mapping[str, Any],
    episode: Mapping[str, Any],
    *,
    action_column: str = "action",
    action_dim: int = DEFAULT_ACTION_DIM,
    prefer_core: bool = True,
) -> list[list[float]]:
    """Read one variable-length action interval from a LeRobot Parquet file."""

    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise RuntimeError(
            "Reading LeRobot actions requires pyarrow in the training environment"
        ) from error

    path = locate_episode_parquet(episode)
    table = parquet.read_table(path, columns=[action_column])
    if action_column not in table.column_names:
        raise ValueError(f"{path} has no action column {action_column!r}")
    values = table[action_column].combine_chunks().to_pylist()
    start, end = resolve_event_interval(event, prefer_core=prefer_core)
    if end > len(values):
        raise ValueError(
            f"Event {event.get('event_id')} interval [{start}, {end}) exceeds "
            f"{path} length {len(values)}"
        )
    trajectory: list[list[float]] = []
    for frame_index, raw_action in enumerate(values[start:end], start=start):
        if not isinstance(raw_action, (list, tuple)) or len(raw_action) != action_dim:
            raise ValueError(
                f"{path}:{frame_index} action must have dimension {action_dim}"
            )
        action = [float(value) for value in raw_action]
        if not all(math.isfinite(value) for value in action):
            raise ValueError(f"{path}:{frame_index} action contains non-finite values")
        trajectory.append(action)
    if not trajectory:
        raise ValueError(f"Event {event.get('event_id')} has no action frames")
    return trajectory


def load_action_pairs(
    records: Sequence[dict[str, Any]],
    *,
    action_loader: Callable[
        [Mapping[str, Any], Mapping[str, Any]], list[list[float]]
    ],
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for record in records:
        sample = dict(record)
        sample["success_actions"] = action_loader(
            record["success_event"], record["success_episode"]
        )
        sample["failure_actions"] = action_loader(
            record["failure_event"], record["failure_episode"]
        )
        samples.append(sample)
    return samples


def compute_action_statistics(
    samples: Sequence[dict[str, Any]],
    *,
    action_dim: int,
    min_std: float = 1e-6,
) -> tuple[list[float], list[float]]:
    """Compute per-dimension statistics from train episodes only."""

    count = 0
    mean = [0.0] * action_dim
    squared_delta = [0.0] * action_dim
    for sample in samples:
        for trajectory_key in ("success_actions", "failure_actions"):
            for action in sample[trajectory_key]:
                if len(action) != action_dim:
                    raise ValueError(
                        f"{trajectory_key} action dimension must be {action_dim}"
                    )
                count += 1
                for index, value in enumerate(action):
                    value = float(value)
                    if not math.isfinite(value):
                        raise ValueError("Action statistics received non-finite value")
                    delta = value - mean[index]
                    mean[index] += delta / count
                    squared_delta[index] += delta * (value - mean[index])
    if count < 2:
        raise ValueError("At least two action frames are required for normalization")
    std = [
        max(math.sqrt(squared_delta[index] / count), min_std)
        for index in range(action_dim)
    ]
    return mean, std


def _require_torch():
    try:
        import torch
    except ImportError as error:
        raise RuntimeError(
            "Offline Steer teacher training requires PyTorch"
        ) from error
    return torch


def collate_action_pairs(
    batch: Sequence[dict[str, Any]],
    *,
    action_mean: Sequence[float],
    action_std: Sequence[float],
    max_steps: int | None,
) -> dict[str, Any]:
    torch = _require_torch()
    action_dim = len(action_mean)
    if len(action_std) != action_dim:
        raise ValueError("action_mean and action_std dimensions differ")

    def prepare(key: str) -> tuple[Any, Any]:
        trajectories: list[list[list[float]]] = []
        for sample in batch:
            trajectory = list(sample[key])
            if max_steps is not None and len(trajectory) > max_steps:
                offset = (len(trajectory) - max_steps) // 2
                trajectory = trajectory[offset : offset + max_steps]
            trajectories.append(trajectory)
        maximum = max(len(trajectory) for trajectory in trajectories)
        actions = torch.zeros(
            len(batch), maximum, action_dim, dtype=torch.float32
        )
        mask = torch.zeros(len(batch), maximum, dtype=torch.bool)
        mean = torch.tensor(action_mean, dtype=torch.float32)
        std = torch.tensor(action_std, dtype=torch.float32)
        for row_index, trajectory in enumerate(trajectories):
            tensor = torch.tensor(trajectory, dtype=torch.float32)
            if tensor.ndim != 2 or tensor.shape[1] != action_dim:
                raise ValueError(
                    f"{key} must have shape [steps, {action_dim}], "
                    f"got {tuple(tensor.shape)}"
                )
            length = tensor.shape[0]
            actions[row_index, :length] = (tensor - mean) / std
            mask[row_index, :length] = True
        return actions, mask

    success_actions, success_mask = prepare("success_actions")
    failure_actions, failure_mask = prepare("failure_actions")
    return {
        "pair_id": [str(sample["pair_id"]) for sample in batch],
        "success_event_id": [str(sample["success_event_id"]) for sample in batch],
        "failure_event_id": [str(sample["failure_event_id"]) for sample in batch],
        "success_actions": success_actions,
        "success_mask": success_mask,
        "failure_actions": failure_actions,
        "failure_mask": failure_mask,
        "pair_weight": torch.tensor(
            [float(sample["pair_weight"]) for sample in batch],
            dtype=torch.float32,
        ),
    }


def augment_trajectory(
    actions: Any,
    valid_mask: Any,
    *,
    mask_probability: float,
    jitter_std: float,
    generator: Any | None = None,
) -> tuple[Any, Any]:
    """Create one masked and jittered trajectory view."""

    torch = _require_torch()
    if not 0.0 <= mask_probability < 1.0:
        raise ValueError("mask_probability must be in [0, 1)")
    if jitter_std < 0.0 or not math.isfinite(jitter_std):
        raise ValueError("jitter_std must be finite and non-negative")
    if actions.ndim != 3 or valid_mask.shape != actions.shape[:2]:
        raise ValueError("actions/mask shapes must be [batch, steps, dim]/[batch, steps]")

    drop = (
        torch.rand(
            valid_mask.shape,
            device=actions.device,
            generator=generator,
        )
        < mask_probability
    ) & valid_mask
    augmented_mask = valid_mask & ~drop
    empty_rows = ~augmented_mask.any(dim=1)
    if empty_rows.any():
        first_valid = valid_mask.to(torch.int64).argmax(dim=1)
        rows = torch.arange(valid_mask.shape[0], device=actions.device)
        augmented_mask[rows[empty_rows], first_valid[empty_rows]] = True

    if jitter_std:
        noise = torch.randn(
            actions.shape,
            device=actions.device,
            dtype=actions.dtype,
            generator=generator,
        ) * jitter_std
    else:
        noise = torch.zeros_like(actions)
    augmented = torch.where(
        augmented_mask.unsqueeze(-1),
        actions + noise,
        torch.zeros_like(actions),
    )
    return augmented, augmented_mask


def paired_augmented_infonce(
    success_view_one: Any,
    success_view_two: Any,
    failure_view_one: Any,
    failure_view_two: Any,
    pair_weight: Any,
    *,
    temperature: float,
    hard_negative_bias: float,
) -> Any:
    """NT-Xent with paired opposite outcomes emphasized as hard negatives."""

    torch = _require_torch()
    import torch.nn.functional as functional

    embeddings = (
        success_view_one,
        success_view_two,
        failure_view_one,
        failure_view_two,
    )
    if any(value.ndim != 2 for value in embeddings):
        raise ValueError("All embeddings must have shape [batch, embedding_dim]")
    if any(value.shape != embeddings[0].shape for value in embeddings[1:]):
        raise ValueError("All embeddings must have the same shape")
    batch_size = embeddings[0].shape[0]
    if batch_size < 1:
        raise ValueError("InfoNCE requires a non-empty batch")
    if pair_weight.shape != (batch_size,):
        raise ValueError("pair_weight must have shape [batch]")
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")
    if not math.isfinite(hard_negative_bias) or hard_negative_bias < 0.0:
        raise ValueError("hard_negative_bias must be finite and non-negative")
    if not torch.isfinite(pair_weight).all() or (pair_weight <= 0).any():
        raise ValueError("pair_weight must contain finite positive values")

    normalized = [
        functional.normalize(value, dim=-1) for value in embeddings
    ]
    features = torch.cat(normalized, dim=0)
    logits = features @ features.transpose(0, 1)
    logits = logits / temperature
    total = 4 * batch_size
    row_ids = torch.arange(total, device=logits.device)

    positive = torch.empty(total, dtype=torch.long, device=logits.device)
    positive[0:batch_size] = row_ids[batch_size : 2 * batch_size]
    positive[batch_size : 2 * batch_size] = row_ids[0:batch_size]
    positive[2 * batch_size : 3 * batch_size] = row_ids[3 * batch_size : total]
    positive[3 * batch_size : total] = row_ids[2 * batch_size : 3 * batch_size]

    if hard_negative_bias:
        pair_offsets = torch.arange(batch_size, device=logits.device)
        bias = torch.zeros_like(logits)
        for success_shift in (0, batch_size):
            rows = success_shift + pair_offsets
            bias[rows, 2 * batch_size + pair_offsets] = hard_negative_bias
            bias[rows, 3 * batch_size + pair_offsets] = hard_negative_bias
        for failure_shift in (2 * batch_size, 3 * batch_size):
            rows = failure_shift + pair_offsets
            bias[rows, pair_offsets] = hard_negative_bias
            bias[rows, batch_size + pair_offsets] = hard_negative_bias
        logits = logits + bias
    logits = logits.masked_fill(
        torch.eye(total, dtype=torch.bool, device=logits.device),
        torch.finfo(logits.dtype).min,
    )

    losses = -functional.log_softmax(logits, dim=1)[row_ids, positive]
    weights = pair_weight.to(device=losses.device, dtype=losses.dtype).repeat(4)
    return (losses * weights).sum() / weights.sum()


def representation_metrics(
    success_view_one: Any,
    success_view_two: Any,
    failure_view_one: Any,
    failure_view_two: Any,
    pair_weight: Any,
) -> dict[str, float]:
    """Measure validation representation quality without building a grad graph.

    Positive pairs are two views of the same action trajectory.  Paired
    success/failure trajectories are the explicit negatives.  Retrieval uses
    every success and failure trajectory in the supplied validation collection
    as a candidate, rather than computing an easier batch-local score.
    """

    torch = _require_torch()
    import torch.nn.functional as functional

    embeddings = (
        success_view_one,
        success_view_two,
        failure_view_one,
        failure_view_two,
    )
    if any(value.ndim != 2 for value in embeddings):
        raise ValueError("All embeddings must have shape [pairs, embedding_dim]")
    if any(value.shape != embeddings[0].shape for value in embeddings[1:]):
        raise ValueError("All embeddings must have the same shape")
    pair_count, embedding_dim = embeddings[0].shape
    if pair_weight.shape != (pair_count,):
        raise ValueError("pair_weight must have shape [pairs]")
    if embedding_dim <= 0:
        raise ValueError("embedding_dim must be positive")

    metric_names = (
        "positive_cosine",
        "negative_cosine",
        "cosine_gap",
        "embedding_variance",
        "top1_paired_retrieval_accuracy",
    )
    if pair_count == 0:
        return {name: 0.0 for name in metric_names}

    weights = pair_weight.detach().to(dtype=torch.float32, device="cpu")
    if not torch.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("pair_weight must contain finite non-negative values")
    total_pair_weight = weights.sum()
    if not torch.isfinite(total_pair_weight) or total_pair_weight <= 0:
        return {name: 0.0 for name in metric_names}

    # Validation metrics and exported Teacher targets are diagnostics/targets;
    # neither path is allowed to backpropagate into the Teacher.
    normalized = [
        functional.normalize(value.detach().to(device="cpu", dtype=torch.float32), dim=-1)
        for value in embeddings
    ]
    success_one, success_two, failure_one, failure_two = normalized
    doubled_weights = weights.repeat(2)
    doubled_weight_sum = doubled_weights.sum()

    positive_values = torch.cat(
        (
            (success_one * success_two).sum(dim=-1),
            (failure_one * failure_two).sum(dim=-1),
        )
    )
    negative_values = torch.cat(
        (
            (success_one * failure_two).sum(dim=-1),
            (failure_one * success_two).sum(dim=-1),
        )
    )
    positive_cosine = (positive_values * doubled_weights).sum() / doubled_weight_sum
    negative_cosine = (negative_values * doubled_weights).sum() / doubled_weight_sum

    anchors = torch.cat((success_one, failure_one), dim=0)
    candidates = torch.cat((success_two, failure_two), dim=0)
    similarities = anchors @ candidates.transpose(0, 1)
    expected = torch.arange(2 * pair_count, device=similarities.device)
    retrieved = similarities.argmax(dim=1)
    retrieval = (
        (retrieved == expected).to(doubled_weights.dtype) * doubled_weights
    ).sum() / doubled_weight_sum

    all_embeddings = torch.cat(normalized, dim=0)
    all_weights = weights.repeat(4)
    all_weight_sum = all_weights.sum()
    weighted_mean = (
        all_embeddings * all_weights.unsqueeze(-1)
    ).sum(dim=0) / all_weight_sum
    centered = all_embeddings - weighted_mean
    embedding_variance = (
        centered.square() * all_weights.unsqueeze(-1)
    ).sum() / (all_weight_sum * embedding_dim)

    result = {
        "positive_cosine": float(positive_cosine.item()),
        "negative_cosine": float(negative_cosine.item()),
        "cosine_gap": float((positive_cosine - negative_cosine).item()),
        "embedding_variance": float(embedding_variance.item()),
        "top1_paired_retrieval_accuracy": float(retrieval.item()),
    }
    if not all(math.isfinite(value) for value in result.values()):
        raise FloatingPointError(f"Non-finite representation metrics: {result}")
    return result


@dataclass(frozen=True)
class TeacherConfig:
    action_dim: int = DEFAULT_ACTION_DIM
    hidden_dim: int = DEFAULT_HIDDEN_DIM
    embedding_dim: int = DEFAULT_EMBEDDING_DIM
    num_layers: int = DEFAULT_NUM_LAYERS
    num_heads: int = DEFAULT_NUM_HEADS
    dropout: float = DEFAULT_DROPOUT


def _move_batch(batch: Mapping[str, Any], device: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in batch.items():
        result[key] = value.to(device) if hasattr(value, "to") else value
    return result


def _teacher_loss(
    model: Any,
    batch: Mapping[str, Any],
    *,
    mask_probability: float,
    jitter_std: float,
    temperature: float,
    hard_negative_bias: float,
    generator: Any | None,
) -> Any:
    success_one = augment_trajectory(
        batch["success_actions"],
        batch["success_mask"],
        mask_probability=mask_probability,
        jitter_std=jitter_std,
        generator=generator,
    )
    success_two = augment_trajectory(
        batch["success_actions"],
        batch["success_mask"],
        mask_probability=mask_probability,
        jitter_std=jitter_std,
        generator=generator,
    )
    failure_one = augment_trajectory(
        batch["failure_actions"],
        batch["failure_mask"],
        mask_probability=mask_probability,
        jitter_std=jitter_std,
        generator=generator,
    )
    failure_two = augment_trajectory(
        batch["failure_actions"],
        batch["failure_mask"],
        mask_probability=mask_probability,
        jitter_std=jitter_std,
        generator=generator,
    )
    embeddings = [
        model(actions, mask)
        for actions, mask in (
            success_one,
            success_two,
            failure_one,
            failure_two,
        )
    ]
    return paired_augmented_infonce(
        *embeddings,
        batch["pair_weight"],
        temperature=temperature,
        hard_negative_bias=hard_negative_bias,
    )


def _run_epoch(
    *,
    model: Any,
    loader: Any,
    device: Any,
    optimizer: Any | None,
    mask_probability: float,
    jitter_std: float,
    temperature: float,
    hard_negative_bias: float,
    grad_clip: float,
    generator: Any | None,
) -> float:
    torch = _require_torch()
    training = optimizer is not None
    model.train(training)
    weighted_loss = 0.0
    total_weight = 0.0
    context = torch.enable_grad if training else torch.no_grad
    with context():
        for raw_batch in loader:
            batch = _move_batch(raw_batch, device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            loss = _teacher_loss(
                model,
                batch,
                mask_probability=mask_probability,
                jitter_std=jitter_std,
                temperature=temperature,
                hard_negative_bias=hard_negative_bias,
                generator=generator,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite teacher loss: {loss.item()}")
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            batch_weight = float(batch["pair_weight"].sum().item())
            weighted_loss += float(loss.item()) * batch_weight
            total_weight += batch_weight
    if total_weight <= 0.0:
        raise ValueError("Epoch contains no positive pair weight")
    return weighted_loss / total_weight


def _run_representation_validation(
    *,
    model: Any,
    loader: Any,
    device: Any,
    mask_probability: float,
    jitter_std: float,
    generator: Any | None,
) -> dict[str, float]:
    """Encode deterministic validation views and compute global metrics."""

    torch = _require_torch()
    model.eval()
    success_one: list[Any] = []
    success_two: list[Any] = []
    failure_one: list[Any] = []
    failure_two: list[Any] = []
    weights: list[Any] = []
    with torch.no_grad():
        for raw_batch in loader:
            batch = _move_batch(raw_batch, device)
            augmented = [
                augment_trajectory(
                    batch[action_key],
                    batch[mask_key],
                    mask_probability=mask_probability,
                    jitter_std=jitter_std,
                    generator=generator,
                )
                for action_key, mask_key in (
                    ("success_actions", "success_mask"),
                    ("success_actions", "success_mask"),
                    ("failure_actions", "failure_mask"),
                    ("failure_actions", "failure_mask"),
                )
            ]
            encoded = [
                model(actions, valid_mask).detach().cpu()
                for actions, valid_mask in augmented
            ]
            success_one.append(encoded[0])
            success_two.append(encoded[1])
            failure_one.append(encoded[2])
            failure_two.append(encoded[3])
            weights.append(batch["pair_weight"].detach().cpu())

    if not weights:
        embedding_dim = int(getattr(model, "embedding_dim", 1))
        empty_embedding = torch.empty(0, embedding_dim, dtype=torch.float32)
        empty_weight = torch.empty(0, dtype=torch.float32)
        return representation_metrics(
            empty_embedding,
            empty_embedding,
            empty_embedding,
            empty_embedding,
            empty_weight,
        )
    return representation_metrics(
        torch.cat(success_one, dim=0),
        torch.cat(success_two, dim=0),
        torch.cat(failure_one, dim=0),
        torch.cat(failure_two, dim=0),
        torch.cat(weights, dim=0),
    )


def _torch_save_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    torch = _require_torch()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _sha256_json(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _protocol_payload(
    *,
    args: argparse.Namespace,
    teacher_config: TeacherConfig,
    episode_ledger_sha256: str,
    event_ledger_sha256: str,
    pair_ledger_sha256: str,
    split_sha256: str,
    action_mean: Sequence[float],
    action_std: Sequence[float],
    resolved_device: str,
) -> dict[str, Any]:
    """Return every data/model/training input that must match on resume."""

    return {
        "schema_version": 1,
        "ledgers": {
            "episode_sha256": episode_ledger_sha256,
            "event_sha256": event_ledger_sha256,
            "pair_sha256": pair_ledger_sha256,
        },
        "split_sha256": split_sha256,
        "teacher": asdict(teacher_config),
        "data": {
            "action_column": args.action_column,
            "prefer_core_event_window": not args.full_event_window,
            "max_steps": args.max_steps,
            "allow_generated_split": args.allow_generated_split,
            "val_fraction": args.val_fraction,
            "split_seed": args.split_seed,
            "action_mean": list(action_mean),
            "action_std": list(action_std),
        },
        "optimization": {
            "seed": args.seed,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "grad_clip": args.grad_clip,
            "temperature": args.temperature,
            "hard_negative_bias": args.hard_negative_bias,
            "mask_probability": args.mask_probability,
            "jitter_std": args.jitter_std,
        },
        "runtime": {
            "device": resolved_device,
            "export_format": args.export_format,
            "wandb_mode": args.wandb_mode,
            "wandb_project": args.wandb_project,
            "wandb_entity": args.wandb_entity,
            "wandb_name": args.wandb_name,
        },
    }


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"Required resume artifact is missing: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _require_protocol_match(
    *,
    stored_config: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    current_protocol: Mapping[str, Any],
    current_config_sha256: str,
) -> str:
    """Reject any resume whose data split or training protocol has drifted."""

    current_hash = _sha256_json(current_protocol)
    stored_protocol = stored_config.get("protocol")
    stored_hash = stored_config.get("protocol_sha256")
    if not isinstance(stored_protocol, dict) or not isinstance(stored_hash, str):
        raise ValueError(
            "Resume config predates the strict protocol schema; start a new run"
        )
    if _sha256_json(stored_protocol) != stored_hash:
        raise ValueError("Stored config protocol_sha256 is invalid")
    if stored_hash != current_hash or stored_protocol != dict(current_protocol):
        raise ValueError(
            "Resume protocol mismatch: ledger/split/config hashes or current "
            "training parameters differ"
        )
    checkpoint_hash = checkpoint.get("protocol_sha256")
    if checkpoint_hash != current_hash:
        raise ValueError("Checkpoint protocol_sha256 does not match current config")
    stored_config_hash = stored_config.get("config_sha256")
    stored_config_payload = dict(stored_config)
    stored_config_payload.pop("config_sha256", None)
    if (
        not isinstance(stored_config_hash, str)
        or _sha256_json(stored_config_payload) != stored_config_hash
    ):
        raise ValueError("Stored config.json SHA-256 is invalid")
    if stored_config_hash != current_config_sha256:
        raise ValueError("Current arguments do not match the stored config.json")
    if checkpoint.get("config_sha256") != current_config_sha256:
        raise ValueError("Checkpoint config SHA-256 does not match config.json")
    checkpoint_hashes = checkpoint.get("artifact_hashes")
    expected_hashes = {
        "episode_ledger_sha256": current_protocol["ledgers"]["episode_sha256"],
        "event_ledger_sha256": current_protocol["ledgers"]["event_sha256"],
        "pair_ledger_sha256": current_protocol["ledgers"]["pair_sha256"],
        "split_sha256": current_protocol["split_sha256"],
    }
    if checkpoint_hashes != expected_hashes:
        raise ValueError("Checkpoint ledger/split hashes do not match current inputs")
    return current_hash


def _capture_rng_state(loader_generator: Any) -> dict[str, Any]:
    torch = _require_torch()
    return {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "loader_generator": loader_generator.get_state(),
    }


def _restore_rng_state(
    state: Mapping[str, Any],
    *,
    loader_generator: Any,
) -> None:
    torch = _require_torch()
    required = {"python", "torch", "cuda", "loader_generator"}
    if set(state) != required:
        raise ValueError(
            f"Checkpoint RNG state must contain exactly {sorted(required)}"
        )
    cuda_states = state["cuda"]
    if not isinstance(cuda_states, (list, tuple)):
        raise ValueError("Checkpoint CUDA RNG state must be a list")
    current_cuda_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if len(cuda_states) != current_cuda_count:
        raise ValueError(
            "Checkpoint CUDA RNG device count does not match the current runtime"
        )
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"].cpu())
    if current_cuda_count:
        torch.cuda.set_rng_state_all([value.cpu() for value in cuda_states])
    loader_generator.set_state(state["loader_generator"].cpu())


def _normalize_metric_row(row: Mapping[str, Any]) -> dict[str, Any]:
    if set(row) != set(METRIC_FIELDS):
        raise ValueError(
            f"Metric row fields must be exactly {sorted(METRIC_FIELDS)}"
        )
    normalized = {"epoch": int(row["epoch"])}
    for field in METRIC_FIELDS[1:]:
        value = float(row[field])
        if not math.isfinite(value):
            raise ValueError(f"Metric {field} must be finite")
        normalized[field] = value
    return normalized


def _read_metrics_csv(path: Path) -> list[dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != METRIC_FIELDS:
                raise ValueError(f"{path} has an incompatible metric schema")
            return [_normalize_metric_row(row) for row in reader]
    except FileNotFoundError as error:
        raise ValueError(f"Required resume artifact is missing: {path}") from error


def _validate_metric_sequence(rows: Sequence[Mapping[str, Any]]) -> None:
    epochs = [int(row["epoch"]) for row in rows]
    if epochs != list(range(1, len(rows) + 1)):
        raise ValueError(
            "Metrics must contain exactly one contiguous row for every epoch"
        )


def _metric_rows_equal(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> bool:
    left = _normalize_metric_row(left)
    right = _normalize_metric_row(right)
    return all(left[field] == right[field] for field in METRIC_FIELDS)


def _reconcile_metrics_for_resume(
    *,
    metrics_jsonl: Path,
    metrics_csv: Path,
    checkpoint_epoch: int,
    checkpoint_metric_row: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Repair only an interrupted final write and return the durable history."""

    json_rows = [_normalize_metric_row(row) for row in read_jsonl(metrics_jsonl)]
    csv_rows = _read_metrics_csv(metrics_csv)
    _validate_metric_sequence(json_rows)
    _validate_metric_sequence(csv_rows)
    common = min(len(json_rows), len(csv_rows), checkpoint_epoch)
    for index in range(common):
        if not _metric_rows_equal(json_rows[index], csv_rows[index]):
            raise ValueError(
                f"metrics.jsonl and metrics.csv disagree at epoch {index + 1}"
            )

    checkpoint_row = _normalize_metric_row(checkpoint_metric_row)
    if checkpoint_row["epoch"] != checkpoint_epoch:
        raise ValueError("Checkpoint metric row epoch does not match checkpoint epoch")
    durable_rows = json_rows[:checkpoint_epoch]
    if len(durable_rows) < checkpoint_epoch:
        if len(durable_rows) != checkpoint_epoch - 1:
            raise ValueError("metrics.jsonl is missing more than the final epoch")
        durable_rows.append(checkpoint_row)
    elif not _metric_rows_equal(durable_rows[-1], checkpoint_row):
        raise ValueError("Checkpoint metric row disagrees with metrics.jsonl")

    csv_prefix = csv_rows[:checkpoint_epoch]
    if len(csv_prefix) < checkpoint_epoch:
        if len(csv_prefix) != checkpoint_epoch - 1:
            raise ValueError("metrics.csv is missing more than the final epoch")
        csv_prefix.append(checkpoint_row)
    elif not _metric_rows_equal(csv_prefix[-1], checkpoint_row):
        raise ValueError("Checkpoint metric row disagrees with metrics.csv")
    if len(durable_rows) != len(csv_prefix):
        raise ValueError("Metric files have incompatible epoch counts")
    for index, (json_row, csv_row) in enumerate(
        zip(durable_rows, csv_prefix),
        start=1,
    ):
        if not _metric_rows_equal(json_row, csv_row):
            raise ValueError(f"Metric files disagree at epoch {index}")

    _write_jsonl_atomic(metrics_jsonl, durable_rows)
    _write_metrics_csv_atomic(metrics_csv, durable_rows)
    return durable_rows


def _validate_checkpoint_structure(checkpoint: Mapping[str, Any]) -> None:
    required = {
        "checkpoint_schema_version",
        "model",
        "optimizer",
        "scheduler",
        "epoch",
        "val_loss",
        "val_representation_metrics",
        "best_val_loss",
        "best_epoch",
        "teacher_config",
        "action_mean",
        "action_std",
        "protocol_sha256",
        "config_sha256",
        "artifact_hashes",
        "metric_row",
        "rng_state",
        "started_at",
        "wandb_run_id",
    }
    missing = required - set(checkpoint)
    if missing:
        raise ValueError(f"Checkpoint is missing resume state: {sorted(missing)}")
    if checkpoint["checkpoint_schema_version"] != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("Checkpoint schema version is not resumable")
    epoch = checkpoint["epoch"]
    best_epoch = checkpoint["best_epoch"]
    if (
        isinstance(epoch, bool)
        or not isinstance(epoch, int)
        or epoch < 1
        or isinstance(best_epoch, bool)
        or not isinstance(best_epoch, int)
        or not 1 <= best_epoch <= epoch
    ):
        raise ValueError("Checkpoint epoch/best_epoch is invalid")
    if not math.isfinite(float(checkpoint["best_val_loss"])):
        raise ValueError("Checkpoint best_val_loss must be finite")


def _restore_training_checkpoint(
    *,
    checkpoint: Mapping[str, Any],
    model: Any,
    optimizer: Any,
    scheduler: Any,
    loader_generator: Any,
) -> tuple[int, float, int, str, str | None]:
    _validate_checkpoint_structure(checkpoint)
    model.load_state_dict(checkpoint["model"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    _restore_rng_state(checkpoint["rng_state"], loader_generator=loader_generator)
    return (
        int(checkpoint["epoch"]),
        float(checkpoint["best_val_loss"]),
        int(checkpoint["best_epoch"]),
        str(checkpoint["started_at"]),
        checkpoint["wandb_run_id"],
    )


def _ensure_best_checkpoint(
    *,
    best_path: Path,
    last_checkpoint: Mapping[str, Any],
) -> None:
    """Retain the recorded best checkpoint, repairing one interrupted best write."""

    torch = _require_torch()
    best_epoch = int(last_checkpoint["best_epoch"])
    best_val = float(last_checkpoint["best_val_loss"])
    if best_path.exists():
        best_checkpoint = torch.load(
            best_path, map_location="cpu", weights_only=False
        )
        _validate_checkpoint_structure(best_checkpoint)
        if (
            int(best_checkpoint["epoch"]) == best_epoch
            and float(best_checkpoint["val_loss"]) == best_val
            and best_checkpoint["protocol_sha256"]
            == last_checkpoint["protocol_sha256"]
        ):
            return
    if int(last_checkpoint["epoch"]) != best_epoch:
        raise ValueError(
            "best_teacher.pt is missing or inconsistent with the recorded best epoch"
        )
    _torch_save_atomic(best_path, last_checkpoint)


def export_pair_targets(
    *,
    model: Any,
    samples_by_split: Mapping[str, Sequence[dict[str, Any]]],
    action_mean: Sequence[float],
    action_std: Sequence[float],
    max_steps: int | None,
    batch_size: int,
    device: Any,
    teacher_hash: str,
    output_dir: Path,
    export_format: str,
) -> list[dict[str, Any]]:
    torch = _require_torch()
    from torch.utils.data import DataLoader

    model.eval()
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for split, samples in samples_by_split.items():
            loader = DataLoader(
                list(samples),
                batch_size=batch_size,
                shuffle=False,
                collate_fn=partial(
                    collate_action_pairs,
                    action_mean=action_mean,
                    action_std=action_std,
                    max_steps=max_steps,
                ),
            )
            for raw_batch in loader:
                batch = _move_batch(raw_batch, device)
                success = model(batch["success_actions"], batch["success_mask"])
                failure = model(batch["failure_actions"], batch["failure_mask"])
                for index, pair_id in enumerate(batch["pair_id"]):
                    rows.append(
                        {
                            "pair_id": pair_id,
                            "success_event_id": batch["success_event_id"][index],
                            "failure_event_id": batch["failure_event_id"][index],
                            "split": split,
                            "pair_weight": float(
                                batch["pair_weight"][index].detach().cpu().item()
                            ),
                            "z_plus": success[index].detach().cpu().tolist(),
                            "z_minus": failure[index].detach().cpu().tolist(),
                            "teacher_sha256": teacher_hash,
                        }
                    )
    rows.sort(key=lambda row: row["pair_id"])

    if export_format in {"parquet", "both"}:
        try:
            import pyarrow as pa
            import pyarrow.parquet as parquet
        except ImportError as error:
            raise RuntimeError("Parquet target export requires pyarrow") from error
        path = output_dir / "pair_targets.parquet"
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        try:
            parquet.write_table(pa.Table.from_pylist(rows), temporary)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    if export_format in {"npz", "both"}:
        try:
            import numpy as np
        except ImportError as error:
            raise RuntimeError("NPZ target export requires numpy") from error
        path = output_dir / "pair_targets.npz"
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
        try:
            np.savez_compressed(
                temporary,
                pair_id=np.asarray([row["pair_id"] for row in rows]),
                success_event_id=np.asarray(
                    [row["success_event_id"] for row in rows]
                ),
                failure_event_id=np.asarray(
                    [row["failure_event_id"] for row in rows]
                ),
                split=np.asarray([row["split"] for row in rows]),
                pair_weight=np.asarray(
                    [row["pair_weight"] for row in rows], dtype=np.float32
                ),
                z_plus=np.asarray([row["z_plus"] for row in rows], dtype=np.float32),
                z_minus=np.asarray([row["z_minus"] for row in rows], dtype=np.float32),
                teacher_sha256=np.asarray(
                    [row["teacher_sha256"] for row in rows]
                ),
            )
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    return rows


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eve-root", type=Path, required=True)
    parser.add_argument("--episode-ledger", type=Path)
    parser.add_argument("--event-ledger", type=Path)
    parser.add_argument("--pair-ledger", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume this output directory strictly from last_teacher.pt. "
            "All data, split, model, optimization, and runtime parameters must match."
        ),
    )
    parser.add_argument("--action-column", default="action")
    parser.add_argument("--full-event-window", action="store_true")
    parser.add_argument("--max-steps", type=int, default=192)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=20260717)
    parser.add_argument(
        "--allow-generated-split",
        action="store_true",
        help=(
            "Legacy smoke-only fallback: generate an episode-disjoint pair split "
            "when pair rows lack EveRobot split labels. Formal runs must omit it."
        ),
    )
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--hard-negative-bias", type=float, default=0.5)
    parser.add_argument("--mask-probability", type=float, default=0.1)
    parser.add_argument("--jitter-std", type=float, default=0.01)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--export-format", choices=("parquet", "npz", "both"), default="both"
    )
    parser.add_argument(
        "--wandb-mode",
        choices=("disabled", "offline", "online"),
        default="disabled",
    )
    parser.add_argument("--wandb-project", default="fitwam-offline-steer-teacher")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-name")
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    if args.epochs <= 0 or args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("epochs/batch_size must be positive; num_workers non-negative")
    if args.max_steps is not None and args.max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if args.learning_rate <= 0.0 or args.weight_decay < 0.0:
        raise ValueError("learning_rate must be positive and weight_decay non-negative")
    if args.grad_clip <= 0.0:
        raise ValueError("grad_clip must be positive")
    if not math.isfinite(args.temperature) or args.temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")
    if not 0.0 <= args.mask_probability < 1.0:
        raise ValueError("mask_probability must be in [0, 1)")
    if args.jitter_std < 0.0:
        raise ValueError("jitter_std must be non-negative")


def run(args: argparse.Namespace) -> dict[str, Any]:
    _validate_args(args)
    torch = _require_torch()
    from torch.utils.data import DataLoader
    from fastwam.models.wan22.offline_steer import TrajectoryTeacher

    eve_root = args.eve_root.expanduser().resolve()
    episode_path = (
        args.episode_ledger or eve_root / "episode_meta.jsonl"
    ).expanduser().resolve()
    event_path = (
        args.event_ledger or eve_root / "event_meta.jsonl"
    ).expanduser().resolve()
    pair_path = (
        args.pair_ledger or eve_root / "pairs" / "event_pair_v1.jsonl"
    ).expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    last_path = output_dir / "last_teacher.pt"
    if args.resume:
        if not output_dir.is_dir() or not last_path.is_file():
            raise FileNotFoundError(
                f"--resume requires an existing {last_path}"
            )
    else:
        if output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError(
                f"Output directory must be new or empty to avoid mixing runs: "
                f"{output_dir}"
            )
        output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    episode_rows = read_jsonl(episode_path)
    event_rows = read_jsonl(event_path)
    pair_rows = read_jsonl(pair_path)
    records = build_pair_records(episode_rows, event_rows, pair_rows)
    if args.allow_generated_split:
        train_records, val_records = split_episode_disjoint(
            records,
            val_fraction=args.val_fraction,
            seed=args.split_seed,
        )
    else:
        train_records, val_records = split_by_declared_ledger(records)
    action_loader = partial(
        read_event_actions,
        action_column=args.action_column,
        action_dim=DEFAULT_ACTION_DIM,
        prefer_core=not args.full_event_window,
    )
    train_samples = load_action_pairs(train_records, action_loader=action_loader)
    val_samples = load_action_pairs(val_records, action_loader=action_loader)
    action_mean, action_std = compute_action_statistics(
        train_samples,
        action_dim=DEFAULT_ACTION_DIM,
    )

    split_payload = {
        "train_pair_ids": sorted(row["pair_id"] for row in train_records),
        "val_pair_ids": sorted(row["pair_id"] for row in val_records),
        "split_seed": args.split_seed,
        "val_fraction": args.val_fraction,
    }
    split_hash = hashlib.sha256(
        canonical_json(split_payload).encode("utf-8")
    ).hexdigest()
    teacher_config = TeacherConfig()
    episode_ledger_sha256 = sha256_file(episode_path)
    event_ledger_sha256 = sha256_file(event_path)
    pair_ledger_sha256 = sha256_file(pair_path)
    protocol = _protocol_payload(
        args=args,
        teacher_config=teacher_config,
        episode_ledger_sha256=episode_ledger_sha256,
        event_ledger_sha256=event_ledger_sha256,
        pair_ledger_sha256=pair_ledger_sha256,
        split_sha256=split_hash,
        action_mean=action_mean,
        action_std=action_std,
        resolved_device=str(device),
    )
    protocol_hash = _sha256_json(protocol)
    run_config = {
        **{key: value for key, value in vars(args).items() if key != "resume"},
        "eve_root": str(eve_root),
        "episode_ledger": str(episode_path),
        "event_ledger": str(event_path),
        "pair_ledger": str(pair_path),
        "output_dir": str(output_dir),
        "teacher": asdict(teacher_config),
        "episode_ledger_sha256": episode_ledger_sha256,
        "event_ledger_sha256": event_ledger_sha256,
        "pair_ledger_sha256": pair_ledger_sha256,
        "split_sha256": split_hash,
        "train_pairs": len(train_samples),
        "val_pairs": len(val_samples),
        "action_mean": action_mean,
        "action_std": action_std,
        "protocol": protocol,
        "protocol_sha256": protocol_hash,
    }
    config_hash = _sha256_json(run_config)
    run_config["config_sha256"] = config_hash
    resume_checkpoint: dict[str, Any] | None = None
    if args.resume:
        resume_checkpoint = torch.load(
            last_path, map_location="cpu", weights_only=False
        )
        if not isinstance(resume_checkpoint, dict):
            raise ValueError("last_teacher.pt must contain a checkpoint dictionary")
        _validate_checkpoint_structure(resume_checkpoint)
        stored_config = _load_json_object(output_dir / "config.json")
        stored_split = _load_json_object(output_dir / "split.json")
        if _sha256_json(stored_split) != split_hash or stored_split != split_payload:
            raise ValueError("Stored split.json does not match the current split")
        _require_protocol_match(
            stored_config=stored_config,
            checkpoint=resume_checkpoint,
            current_protocol=protocol,
            current_config_sha256=config_hash,
        )
    else:
        write_json_atomic(output_dir / "config.json", run_config)
        write_json_atomic(output_dir / "split.json", split_payload)

    model = TrajectoryTeacher(
        action_dim=teacher_config.action_dim,
        hidden_dim=teacher_config.hidden_dim,
        embedding_dim=teacher_config.embedding_dim,
        num_heads=teacher_config.num_heads,
        num_layers=teacher_config.num_layers,
        dropout=teacher_config.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    collate = partial(
        collate_action_pairs,
        action_mean=action_mean,
        action_std=action_std,
        max_steps=args.max_steps,
    )
    loader_generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_samples,
        batch_size=args.batch_size,
        shuffle=True,
        generator=loader_generator,
        num_workers=args.num_workers,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        val_samples,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
    )

    metrics_jsonl = output_dir / "metrics.jsonl"
    metrics_csv = output_dir / "metrics.csv"
    best_path = output_dir / "best_teacher.pt"
    if resume_checkpoint is not None and int(resume_checkpoint["epoch"]) >= args.epochs:
        raise ValueError(
            "last_teacher.pt already reached --epochs; start a new run to use "
            "a different total training budget"
        )

    wandb_run = None
    if args.wandb_mode != "disabled":
        try:
            import wandb
        except ImportError as error:
            raise RuntimeError(
                "W&B was requested but wandb is not installed"
            ) from error
        wandb_resume_id = (
            resume_checkpoint.get("wandb_run_id")
            if resume_checkpoint is not None
            else None
        )
        if args.resume and not isinstance(wandb_resume_id, str):
            raise ValueError("Resumed W&B run is missing its original run id")
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name,
            mode=args.wandb_mode,
            config=run_config,
            dir=str(output_dir),
            id=wandb_resume_id,
            resume="must" if args.resume else None,
        )
    wandb_run_id = None if wandb_run is None else str(wandb_run.id)

    if resume_checkpoint is None:
        start_epoch = 1
        best_val = math.inf
        best_epoch = -1
        started_at = datetime.now(timezone.utc).isoformat()
        _write_jsonl_atomic(metrics_jsonl, [])
        _write_metrics_csv_atomic(metrics_csv, [])
    else:
        (
            completed_epoch,
            best_val,
            best_epoch,
            started_at,
            checkpoint_wandb_id,
        ) = _restore_training_checkpoint(
            checkpoint=resume_checkpoint,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            loader_generator=loader_generator,
        )
        if checkpoint_wandb_id != wandb_run_id:
            raise ValueError("Checkpoint W&B run id does not match the resumed run")
        _reconcile_metrics_for_resume(
            metrics_jsonl=metrics_jsonl,
            metrics_csv=metrics_csv,
            checkpoint_epoch=completed_epoch,
            checkpoint_metric_row=resume_checkpoint["metric_row"],
        )
        _ensure_best_checkpoint(
            best_path=best_path,
            last_checkpoint=resume_checkpoint,
        )
        start_epoch = completed_epoch + 1

    stop_payload: dict[str, Any]
    try:
        with metrics_csv.open("a", encoding="utf-8", newline="") as csv_stream:
            writer = csv.DictWriter(
                csv_stream,
                fieldnames=METRIC_FIELDS,
            )
            for epoch in range(start_epoch, args.epochs + 1):
                train_generator = torch.Generator(device=device).manual_seed(
                    args.seed + epoch
                )
                train_loss = _run_epoch(
                    model=model,
                    loader=train_loader,
                    device=device,
                    optimizer=optimizer,
                    mask_probability=args.mask_probability,
                    jitter_std=args.jitter_std,
                    temperature=args.temperature,
                    hard_negative_bias=args.hard_negative_bias,
                    grad_clip=args.grad_clip,
                    generator=train_generator,
                )
                # Validation uses exact trajectories.  This makes the best
                # checkpoint criterion deterministic across epochs.
                val_loss = _run_epoch(
                    model=model,
                    loader=val_loader,
                    device=device,
                    optimizer=None,
                    mask_probability=0.0,
                    jitter_std=0.0,
                    temperature=args.temperature,
                    hard_negative_bias=args.hard_negative_bias,
                    grad_clip=args.grad_clip,
                    generator=None,
                )
                # Reuse one fixed augmentation stream every epoch.  The model
                # changes, while validation views stay comparable.
                representation_generator = torch.Generator(device=device).manual_seed(
                    args.seed + 1_000_003
                )
                representation = _run_representation_validation(
                    model=model,
                    loader=val_loader,
                    device=device,
                    mask_probability=args.mask_probability,
                    jitter_std=args.jitter_std,
                    generator=representation_generator,
                )
                learning_rate = float(optimizer.param_groups[0]["lr"])
                metric_row = {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    **{
                        f"val_{name}": value
                        for name, value in representation.items()
                    },
                    "learning_rate": learning_rate,
                }
                improved = val_loss < best_val
                if improved:
                    best_val = val_loss
                    best_epoch = epoch
                scheduler.step()
                checkpoint = {
                    "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "val_representation_metrics": representation,
                    "best_val_loss": best_val,
                    "best_epoch": best_epoch,
                    "teacher_config": asdict(teacher_config),
                    "action_mean": action_mean,
                    "action_std": action_std,
                    "protocol_sha256": protocol_hash,
                    "config_sha256": config_hash,
                    "artifact_hashes": {
                        "episode_ledger_sha256": episode_ledger_sha256,
                        "event_ledger_sha256": event_ledger_sha256,
                        "pair_ledger_sha256": pair_ledger_sha256,
                        "split_sha256": split_hash,
                    },
                    "metric_row": metric_row,
                    "rng_state": _capture_rng_state(loader_generator),
                    "started_at": started_at,
                    "wandb_run_id": wandb_run_id,
                }
                _torch_save_atomic(last_path, checkpoint)
                if improved:
                    _torch_save_atomic(best_path, checkpoint)
                    write_json_atomic(
                        output_dir / "best_checkpoint.json",
                        {
                            "path": str(best_path),
                            "epoch": best_epoch,
                            "val_loss": best_val,
                            "val_representation_metrics": representation,
                        },
                    )
                append_jsonl(metrics_jsonl, metric_row)
                writer.writerow(metric_row)
                csv_stream.flush()
                os.fsync(csv_stream.fileno())
                if wandb_run is not None:
                    wandb_run.log(metric_row, step=epoch)

        best_checkpoint = torch.load(
            best_path, map_location=device, weights_only=False
        )
        model.load_state_dict(best_checkpoint["model"])
        teacher_hash = sha256_file(best_path)
        target_rows = export_pair_targets(
            model=model,
            samples_by_split={"train": train_samples, "val": val_samples},
            action_mean=action_mean,
            action_std=action_std,
            max_steps=args.max_steps,
            batch_size=args.batch_size,
            device=device,
            teacher_hash=teacher_hash,
            output_dir=output_dir,
            export_format=args.export_format,
        )
        stop_payload = {
            "reason": "completed_max_epochs",
            "started_at": started_at,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "epochs_completed": args.epochs,
            "best_epoch": best_epoch,
            "best_val_loss": best_val,
            "teacher_sha256": teacher_hash,
            "exported_pairs": len(target_rows),
        }
    except KeyboardInterrupt:
        stop_payload = {
            "reason": "interrupted",
            "started_at": started_at,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "best_epoch": best_epoch,
            "best_val_loss": None if not math.isfinite(best_val) else best_val,
        }
        write_json_atomic(output_dir / "stop_reason.json", stop_payload)
        raise
    except Exception as error:
        stop_payload = {
            "reason": "exception",
            "error_type": type(error).__name__,
            "error": str(error),
            "started_at": started_at,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "best_epoch": best_epoch,
            "best_val_loss": None if not math.isfinite(best_val) else best_val,
        }
        write_json_atomic(output_dir / "stop_reason.json", stop_payload)
        raise
    finally:
        if wandb_run is not None:
            wandb_run.finish()

    write_json_atomic(output_dir / "stop_reason.json", stop_payload)
    return stop_payload


def main(argv: Sequence[str] | None = None) -> int:
    result = run(parse_args(argv))
    print(canonical_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
