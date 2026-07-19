#!/usr/bin/env python3
"""Extract EveRobot interaction candidates from proprioceptive state-line scores.

The calibration is fitted only from EveRobot episodes in the requested
calibration split (``train`` by default). The resulting non-semantic candidate
windows are appended to ``event_meta.jsonl`` without rewriting existing rows.
Frame-level scores are stored separately as Parquet.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fastwam.everobot_events import extract_candidate_windows  # noqa: E402
from fastwam.everobot_schema import SCHEMA_VERSION  # noqa: E402
from scripts.everobot import build_eve_sidecar  # noqa: E402


STATE_COLUMN = "observation.state"
ALGORITHM_VERSION = "state_line_candidate_v2"
CALIBRATION_FORMAT = "EveRobotStateLineCalibration"


@dataclass(frozen=True)
class ExtractionParameters:
    """Post-processing parameters for one calibrated score sequence."""

    median_window: int = 5
    ema_alpha: float = 0.25
    high_threshold: float = 0.55
    low_threshold: float = 0.35
    max_gap: int = 8
    min_run: int = 5
    pre_padding: int = 12
    post_padding: int = 12
    min_window: int = 33
    max_candidate: int = 96
    max_candidates_per_episode: int | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_candidate, bool)
            or int(self.max_candidate) != self.max_candidate
            or self.max_candidate < 1
        ):
            raise ValueError("max_candidate must be a positive integer")
        if self.max_candidates_per_episode is not None and (
            isinstance(self.max_candidates_per_episode, bool)
            or int(self.max_candidates_per_episode)
            != self.max_candidates_per_episode
            or self.max_candidates_per_episode < 1
        ):
            raise ValueError("max_candidates_per_episode must be a positive integer")

    def as_dict(self) -> dict[str, int | float | None]:
        return {
            "median_window": self.median_window,
            "ema_alpha": self.ema_alpha,
            "high_threshold": self.high_threshold,
            "low_threshold": self.low_threshold,
            "max_gap": self.max_gap,
            "min_run": self.min_run,
            "pre_padding": self.pre_padding,
            "post_padding": self.post_padding,
            "min_window": self.min_window,
            "max_candidate": self.max_candidate,
            "max_candidates_per_episode": self.max_candidates_per_episode,
        }

    def extractor_kwargs(self) -> dict[str, int | float | None]:
        parameters = self.as_dict()
        parameters.pop("max_candidate")
        return parameters


def canonical_json(payload: Any) -> str:
    """Serialize JSON deterministically for stable identities."""

    return json.dumps(
        payload,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_json(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def calibration_input_hash(
    states_by_episode: Sequence[tuple[str, np.ndarray]],
) -> str:
    """Hash calibration episode identities, shapes, and complete state arrays."""

    digest = hashlib.sha256(b"EveRobotStateLineCalibrationInputV1\0")
    normalized: list[tuple[str, np.ndarray]] = []
    for episode_id, states in states_by_episode:
        array = np.ascontiguousarray(np.asarray(states, dtype="<f8"))
        normalized.append((str(episode_id), array))
    for episode_id, array in sorted(normalized, key=lambda item: item[0]):
        digest.update(episode_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(canonical_json(list(array.shape)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def make_method_id(
    *,
    algorithm_version: str,
    calibration_id: str,
    parameters: ExtractionParameters,
) -> str:
    """Identify the complete extraction method, including every threshold."""

    identity = {
        "algorithm_version": algorithm_version,
        "calibration_id": calibration_id,
        "extraction_parameters": parameters.as_dict(),
    }
    return f"state-line-method-{sha256_json(identity)[:16]}"


def score_rows_hash(rows: Sequence[Mapping[str, Any]]) -> str:
    return sha256_json([dict(row) for row in rows])


def _finite_mean(values: np.ndarray, axis: int) -> np.ndarray:
    finite = np.isfinite(values)
    count = finite.sum(axis=axis)
    total = np.where(finite, values, 0.0).sum(axis=axis)
    output = np.full(count.shape, np.nan, dtype=np.float64)
    np.divide(total, count, out=output, where=count > 0)
    return output


def state_line_distances(states: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Return the mean a3-to-line(a1,a2) distance for every frame."""

    states = np.asarray(states, dtype=np.float64)
    scale = np.asarray(scale, dtype=np.float64)
    if states.ndim != 2 or states.shape[1] == 0:
        raise ValueError("states must have shape [frames, state_dim]")
    if scale.shape != (states.shape[1],):
        raise ValueError("scale must have shape [state_dim]")
    if not np.isfinite(scale).all() or np.any(scale <= 0.0):
        raise ValueError("scale must contain finite positive values")

    normalized = states / scale
    per_dimension = np.full(normalized.shape, np.nan, dtype=np.float64)
    if len(normalized) >= 3:
        a1 = normalized[:-2]
        a2 = normalized[1:-1]
        a3 = normalized[2:]
        slope = a2 - a1
        residual = (2.0 * a2) - a1 - a3
        per_dimension[2:] = np.abs(residual) / np.sqrt((slope * slope) + 1.0)
    return _finite_mean(per_dimension, axis=1)


def apply_calibration(distances: np.ndarray, calibration: Mapping[str, Any]) -> np.ndarray:
    """Normalize raw distances with a persisted robust calibration."""

    distances = np.asarray(distances, dtype=np.float64)
    low = float(calibration["distance_low"])
    high = float(calibration["distance_high"])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        raise ValueError("calibration distance_high must exceed distance_low")
    scores = np.clip((distances - low) / (high - low), 0.0, 1.0)
    scores[~np.isfinite(distances)] = np.nan
    return scores


def fit_robust_calibration(
    states_by_episode: Sequence[tuple[str, np.ndarray]],
    *,
    calibration_split: str = "train",
    low_quantile: float = 0.10,
    high_quantile: float = 0.95,
    algorithm_version: str = ALGORITHM_VERSION,
) -> dict[str, Any]:
    """Fit state scale and robust distance bounds from calibration episodes."""

    if not states_by_episode:
        raise ValueError("No episodes were provided for calibration")
    if not 0.0 <= low_quantile < high_quantile <= 1.0:
        raise ValueError("quantiles must satisfy 0 <= low < high <= 1")

    episode_ids: list[str] = []
    arrays: list[np.ndarray] = []
    state_dim: int | None = None
    for episode_id, raw_states in states_by_episode:
        states = np.asarray(raw_states, dtype=np.float64)
        if states.ndim != 2 or states.shape[0] == 0 or states.shape[1] == 0:
            raise ValueError(f"Episode {episode_id} has invalid state shape {states.shape}")
        if state_dim is None:
            state_dim = states.shape[1]
        elif states.shape[1] != state_dim:
            raise ValueError("All calibration episodes must use the same state dimension")
        episode_ids.append(str(episode_id))
        arrays.append(states)

    all_states = np.concatenate(arrays, axis=0)
    scale = np.nanstd(all_states, axis=0)
    scale[~np.isfinite(scale) | (scale < 1e-6)] = 1.0

    all_distances = np.concatenate(
        [state_line_distances(states, scale) for states in arrays]
    )
    finite = all_distances[np.isfinite(all_distances)]
    if finite.size == 0:
        raise ValueError("Calibration episodes contain no finite state-line distances")
    low = float(np.quantile(finite, low_quantile))
    high = float(np.quantile(finite, high_quantile))
    if high <= low:
        high = low + 1e-6

    identity_payload = {
        "format": CALIBRATION_FORMAT,
        "algorithm_version": algorithm_version,
        "state_column": STATE_COLUMN,
        "calibration_split": calibration_split,
        "low_quantile": float(low_quantile),
        "high_quantile": float(high_quantile),
        "episode_ids": sorted(episode_ids),
        "calibration_input_sha256": calibration_input_hash(states_by_episode),
        "state_dim": int(state_dim),
        "state_scale": scale.tolist(),
        "distance_low": low,
        "distance_high": high,
        "distance_median": float(np.median(finite)),
    }
    return {
        **identity_payload,
        "calibration_id": f"state-line-{sha256_json(identity_payload)[:16]}",
        "num_episodes": len(episode_ids),
        "num_frames": int(sum(len(states) for states in arrays)),
        "num_finite_distances": int(finite.size),
    }


def episode_scores(
    states: np.ndarray, calibration: Mapping[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    scale = np.asarray(calibration["state_scale"], dtype=np.float64)
    distances = state_line_distances(states, scale)
    return distances, apply_calibration(distances, calibration)


def stable_event_id(
    episode: Mapping[str, Any],
    *,
    candidate_index: int,
    algorithm_version: str,
    calibration_id: str,
    method_id: str,
) -> str:
    """Build an identity that changes when the algorithm or calibration changes."""

    version = re.sub(r"[^A-Za-z0-9_.-]+", "-", algorithm_version).strip("-")
    calibration = re.sub(r"[^A-Za-z0-9_.-]+", "-", calibration_id).strip("-")
    method = re.sub(r"[^A-Za-z0-9_.-]+", "-", method_id).strip("-")
    return (
        f"{episode['dataset_id']}_ep{int(episode['episode_index']):06d}_"
        f"{version}_{calibration}_{method}_candidate_{candidate_index:03d}"
    )


def extract_episode_rows(
    episode: Mapping[str, Any],
    states: np.ndarray,
    calibration: Mapping[str, Any],
    *,
    parameters: ExtractionParameters,
    scores_artifact: str,
    method_id: str,
    algorithm_version: str = ALGORITHM_VERSION,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Extract candidate ledger rows and frame-level score rows for one episode."""

    outcome = str(episode.get("episode_outcome", ""))
    if outcome not in {"success", "failure"}:
        raise ValueError(
            f"Episode {episode.get('episode_id')} has unsupported outcome {outcome!r}"
        )
    states = np.asarray(states, dtype=np.float64)
    expected_length = int(episode["length"])
    if states.ndim != 2 or len(states) != expected_length:
        raise ValueError(
            f"Episode {episode['episode_id']} state length {len(states)} "
            f"does not match ledger length {expected_length}"
        )

    distances, scores = episode_scores(states, calibration)
    extraction = extract_candidate_windows(scores, **parameters.extractor_kwargs())
    calibration_id = str(calibration["calibration_id"])
    action_loss = "enabled" if outcome == "success" else "disabled"
    sample_role = "success_event" if outcome == "success" else "failure_context"
    annotation_parameters = parameters.as_dict()

    event_rows: list[dict[str, Any]] = []
    for index, candidate in enumerate(extraction.candidates):
        exceeds_max_candidate = (
            candidate.end_frame - candidate.start_frame > parameters.max_candidate
        )
        event_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "event_id": stable_event_id(
                    episode,
                    candidate_index=index,
                    algorithm_version=algorithm_version,
                    calibration_id=calibration_id,
                    method_id=method_id,
                ),
                "episode_id": str(episode["episode_id"]),
                "round_id": str(episode["round_id"]),
                "dataset_id": str(episode["dataset_id"]),
                "dataset_root": str(episode["dataset_root"]),
                "episode_index": int(episode["episode_index"]),
                "task_name": str(episode.get("task_name", "")),
                "task": str(episode.get("task", "")),
                "event_type": "interaction_candidate",
                "event_level": "candidate",
                "event_label": "state_line_transition",
                "effector": "global",
                "event_outcome": "unknown",
                "episode_outcome": outcome,
                "source_policy": episode.get("source_policy"),
                "collection_round": int(episode.get("collection_round", -1)),
                "start_frame": candidate.start_frame,
                "end_frame": candidate.end_frame,
                "core_start_frame": candidate.core_start_frame,
                "core_end_frame": candidate.core_end_frame,
                "core_interval": [
                    candidate.core_start_frame,
                    candidate.core_end_frame,
                ],
                "peak_score": candidate.peak_score,
                "absolute_confidence": candidate.confidence,
                "episode_sampling_weight": candidate.episode_weight,
                "event_weight": candidate.episode_weight,
                "exceeds_max_candidate": exceeds_max_candidate,
                "action_loss": action_loss,
                "sample_role": sample_role,
                "annotation": {
                    "source": "auto",
                    "method": "state_line_distance",
                    "version": algorithm_version,
                    "confidence": candidate.confidence,
                    "calibration_id": calibration_id,
                    "method_id": method_id,
                    "calibration_split": calibration["calibration_split"],
                    "parameters": annotation_parameters,
                    "long_candidate_policy": (
                        "preserve_coarse_event_and_defer_sliding_window_to_loader"
                    ),
                    "scores_artifact": scores_artifact,
                },
                "split": str(episode.get("split", "train")),
            }
        )

    frame_rows = [
        {
            "episode_id": str(episode["episode_id"]),
            "dataset_id": str(episode["dataset_id"]),
            "episode_index": int(episode["episode_index"]),
            "frame_index": frame_index,
            "state_line_distance": (
                float(distances[frame_index])
                if np.isfinite(distances[frame_index])
                else None
            ),
            "event_transition_score": (
                float(scores[frame_index]) if np.isfinite(scores[frame_index]) else None
            ),
            "smoothed_event_score": (
                float(extraction.smoothed_scores[frame_index])
                if np.isfinite(extraction.smoothed_scores[frame_index])
                else None
            ),
            "active_candidate": bool(extraction.active_mask[frame_index]),
            "algorithm_version": algorithm_version,
            "calibration_id": calibration_id,
            "method_id": method_id,
        }
        for frame_index in range(expected_length)
    ]
    return event_rows, frame_rows


def load_episode_states(path: Path) -> np.ndarray:
    """Read ``observation.state`` from one LeRobot episode Parquet."""

    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError(
            "pyarrow is required to read LeRobot episode Parquet files; "
            "install pyarrow before running state-line extraction"
        ) from error
    table = pq.ParquetFile(path).read(columns=[STATE_COLUMN])
    return np.asarray(
        table[STATE_COLUMN].combine_chunks().to_pylist(), dtype=np.float64
    )


def locate_episode_parquet(episode: Mapping[str, Any]) -> Path:
    dataset_root = Path(str(episode["dataset_root"])).expanduser().resolve()
    episode_index = int(episode["episode_index"])
    info_path = dataset_root / "meta" / "info.json"
    info = (
        json.loads(info_path.read_text(encoding="utf-8"))
        if info_path.exists()
        else {}
    )
    chunks_size = int(info.get("chunks_size", 1000))
    pattern = str(
        info.get(
            "data_path",
            "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        )
    )
    formatted = pattern.format(
        episode_chunk=episode_index // chunks_size,
        episode_index=episode_index,
    )
    path = dataset_root / formatted
    if path.exists():
        return path
    matches = sorted(
        (dataset_root / "data").glob(
            f"chunk-*/episode_{episode_index:06d}.parquet"
        )
    )
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(
            f"Missing Parquet for {episode.get('episode_id')}: expected {path}"
        )
    raise ValueError(
        f"Multiple Parquet files found for {episode.get('episode_id')}: {matches}"
    )


def write_scores_parquet(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write one immutable frame-level score artifact as Parquet."""

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError(
            "pyarrow is required to write event_scores.parquet; "
            "install pyarrow before running state-line extraction"
        ) from error
    content_hash = score_rows_hash(rows)
    if path.exists():
        parquet = pq.ParquetFile(path)
        metadata = parquet.schema_arrow.metadata or {}
        existing_hash = metadata.get(b"eve_content_sha256", b"").decode("ascii")
        if existing_hash != content_hash:
            raise ValueError(
                f"Immutable score artifact collision at {path}; "
                "use the method-versioned default path or a new --scores-path"
            )
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp.parquet", dir=path.parent
    )
    os.close(fd)
    temporary_path = Path(temporary_name)
    try:
        table = pa.Table.from_pylist([dict(row) for row in rows])
        metadata = dict(table.schema.metadata or {})
        metadata[b"eve_content_sha256"] = content_hash.encode("ascii")
        method_ids = sorted({str(row["method_id"]) for row in rows})
        if len(method_ids) != 1:
            raise ValueError("Score rows must belong to exactly one method_id")
        metadata[b"eve_method_id"] = method_ids[0].encode("ascii")
        table = table.replace_schema_metadata(metadata)
        pq.write_table(table, temporary_path)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def persist_calibration(path: Path, calibration: Mapping[str, Any]) -> None:
    """Persist one immutable calibration record."""

    payload = json.dumps(dict(calibration), indent=2, ensure_ascii=True) + "\n"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if canonical_json(existing) != canonical_json(calibration):
            raise ValueError(f"Calibration path already contains different data: {path}")
        return
    build_eve_sidecar.write_text_atomic(path, payload)


def append_event_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> int:
    """Append through EveRobot's immutable preflight and atomic replace helper."""

    return build_eve_sidecar.append_immutable_jsonl_group(
        [
            (
                path,
                [dict(row) for row in rows],
                ("event_id",),
                ("dataset_root",),
            )
        ]
    )[0]


def preflight_event_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Validate immutable identities before writing the score artifact."""

    build_eve_sidecar.prepare_immutable_jsonl(
        path,
        [dict(row) for row in rows],
        key_fields=("event_id",),
        compare_ignore_fields=("dataset_root",),
    )


def select_episode_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    dataset_ids: set[str] | None = None,
    extraction_splits: set[str] | None = None,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw_row in rows:
        row = dict(raw_row)
        episode_id = str(row.get("episode_id", ""))
        if not episode_id:
            raise ValueError("Every episode ledger row must have episode_id")
        if episode_id in seen:
            raise ValueError(f"Duplicate episode_id in ledger: {episode_id}")
        seen.add(episode_id)
        if dataset_ids is not None and str(row.get("dataset_id")) not in dataset_ids:
            continue
        if (
            extraction_splits is not None
            and str(row.get("split", "train")) not in extraction_splits
        ):
            continue
        selected.append(row)
    return selected


def run_extraction(
    *,
    eve_root: Path,
    episode_rows: Sequence[Mapping[str, Any]],
    state_loader: Callable[[Mapping[str, Any]], np.ndarray],
    parameters: ExtractionParameters,
    calibration_split: str,
    low_quantile: float,
    high_quantile: float,
    algorithm_version: str,
    scores_path: Path | None,
    append_ledger: bool = True,
    scores_writer: Callable[[Path, Sequence[Mapping[str, Any]]], None] = write_scores_parquet,
) -> dict[str, Any]:
    """Run extraction with injectable I/O for dependency-light tests."""

    states_by_id: dict[str, np.ndarray] = {}
    for episode in episode_rows:
        states_by_id[str(episode["episode_id"])] = np.asarray(
            state_loader(episode), dtype=np.float64
        )

    calibration_inputs = [
        (str(episode["episode_id"]), states_by_id[str(episode["episode_id"])])
        for episode in episode_rows
        if str(episode.get("split", "train")) == calibration_split
    ]
    calibration = fit_robust_calibration(
        calibration_inputs,
        calibration_split=calibration_split,
        low_quantile=low_quantile,
        high_quantile=high_quantile,
        algorithm_version=algorithm_version,
    )
    calibration_path = (
        eve_root
        / "annotations"
        / "calibrations"
        / f"{calibration['calibration_id']}.json"
    )
    persist_calibration(calibration_path, calibration)
    method_id = make_method_id(
        algorithm_version=algorithm_version,
        calibration_id=str(calibration["calibration_id"]),
        parameters=parameters,
    )
    resolved_scores_path = (
        scores_path
        if scores_path is not None
        else eve_root / "annotations" / f"event_scores_{method_id}.parquet"
    )

    try:
        scores_artifact = resolved_scores_path.relative_to(eve_root).as_posix()
    except ValueError:
        scores_artifact = str(resolved_scores_path)

    all_events: list[dict[str, Any]] = []
    all_scores: list[dict[str, Any]] = []
    for episode in episode_rows:
        events, scores = extract_episode_rows(
            episode,
            states_by_id[str(episode["episode_id"])],
            calibration,
            parameters=parameters,
            scores_artifact=scores_artifact,
            method_id=method_id,
            algorithm_version=algorithm_version,
        )
        all_events.extend(events)
        all_scores.extend(scores)

    scores_sha256 = score_rows_hash(all_scores)
    for event in all_events:
        event["annotation"]["scores_sha256"] = scores_sha256

    appended = 0
    if append_ledger:
        with build_eve_sidecar.sidecar_write_lock(eve_root):
            ledger_path = eve_root / "event_meta.jsonl"
            preflight_event_rows(ledger_path, all_events)
            scores_writer(resolved_scores_path, all_scores)
            appended = append_event_rows(ledger_path, all_events)
    else:
        scores_writer(resolved_scores_path, all_scores)
    return {
        "calibration": calibration,
        "calibration_path": str(calibration_path),
        "method_id": method_id,
        "scores_sha256": scores_sha256,
        "scores_path": str(resolved_scores_path),
        "num_episodes": len(episode_rows),
        "num_score_rows": len(all_scores),
        "num_candidates": len(all_events),
        "num_appended_candidates": appended,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eve-root", type=Path, required=True)
    parser.add_argument("--dataset-ids", nargs="+", default=None)
    parser.add_argument(
        "--extract-splits",
        nargs="+",
        default=None,
        help="Episode splits to extract. Defaults to all ledger rows.",
    )
    parser.add_argument("--calibration-split", default="train")
    parser.add_argument("--low-quantile", type=float, default=0.10)
    parser.add_argument("--high-quantile", type=float, default=0.95)
    parser.add_argument("--algorithm-version", default=ALGORITHM_VERSION)
    parser.add_argument("--scores-path", type=Path, default=None)
    parser.add_argument("--median-window", type=int, default=5)
    parser.add_argument("--ema-alpha", type=float, default=0.25)
    parser.add_argument("--high-threshold", type=float, default=0.55)
    parser.add_argument("--low-threshold", type=float, default=0.35)
    parser.add_argument("--max-gap", type=int, default=8)
    parser.add_argument("--min-run", type=int, default=5)
    parser.add_argument("--pre-padding", type=int, default=12)
    parser.add_argument("--post-padding", type=int, default=12)
    parser.add_argument("--min-window", type=int, default=33)
    parser.add_argument(
        "--max-candidate",
        type=int,
        default=96,
        help=(
            "Mark longer padded candidates for loader-side sliding windows; "
            "the extractor does not split them without a reliable score valley."
        ),
    )
    parser.add_argument(
        "--max-candidates-per-episode",
        type=int,
        default=None,
        help=(
            "Retain at most this many candidates per episode, ranked by "
            "confidence and peak score before restoring temporal order."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write calibration and scores but do not append event_meta.jsonl.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    eve_root = args.eve_root.expanduser().resolve()
    episode_meta = eve_root / "episode_meta.jsonl"
    if not episode_meta.exists():
        raise FileNotFoundError(f"Missing EveRobot episode ledger: {episode_meta}")
    rows = select_episode_rows(
        build_eve_sidecar.load_jsonl(episode_meta),
        dataset_ids=set(args.dataset_ids) if args.dataset_ids else None,
        extraction_splits=set(args.extract_splits) if args.extract_splits else None,
    )
    if not rows:
        raise ValueError("No episode rows matched the requested extraction filters")
    if not any(
        str(row.get("split", "train")) == args.calibration_split for row in rows
    ):
        raise ValueError(
            f"No selected episodes belong to calibration split "
            f"{args.calibration_split!r}"
        )
    parameters = ExtractionParameters(
        median_window=args.median_window,
        ema_alpha=args.ema_alpha,
        high_threshold=args.high_threshold,
        low_threshold=args.low_threshold,
        max_gap=args.max_gap,
        min_run=args.min_run,
        pre_padding=args.pre_padding,
        post_padding=args.post_padding,
        min_window=args.min_window,
        max_candidate=args.max_candidate,
        max_candidates_per_episode=args.max_candidates_per_episode,
    )
    scores_path = (
        args.scores_path.expanduser().resolve()
        if args.scores_path is not None
        else None
    )
    result = run_extraction(
        eve_root=eve_root,
        episode_rows=rows,
        state_loader=lambda episode: load_episode_states(
            locate_episode_parquet(episode)
        ),
        parameters=parameters,
        calibration_split=args.calibration_split,
        low_quantile=args.low_quantile,
        high_quantile=args.high_quantile,
        algorithm_version=args.algorithm_version,
        scores_path=scores_path,
        append_ledger=not args.dry_run,
    )
    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
