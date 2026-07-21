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
import hmac
import json
import math
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
from fastwam.everobot_schema import SCHEMA_VERSION, validate_manifest  # noqa: E402
from scripts.everobot import build_eve_sidecar  # noqa: E402


STATE_COLUMN = "observation.state"
ALGORITHM_VERSION = "state_line_candidate_v2"
CALIBRATION_FORMAT = "EveRobotStateLineCalibration"
TAIL_CONSENSUS_FORMAT = "FITWAMTailConsensusReport"
TAIL_CONSENSUS_SCHEMA_VERSION = "1.0"
TAIL_CONSENSUS_RULE = {
    "trim_condition": "all_inputs_should_trim",
    "trim_cutoff": "max_input_cutoff_frame",
    "no_trim_cutoff": "num_frames",
}


@dataclass(frozen=True)
class TailConsensusDecision:
    """Validated candidate-visibility decision for one failure episode."""

    num_frames: int
    should_trim: bool
    effective_end_frame: int


@dataclass(frozen=True)
class TailConsensusBundle:
    """Validated immutable inputs used to suppress failure-tail candidates."""

    report_sha256: str
    cutoffs_sha256: str
    source_manifest_sha256: str
    cutoff_records_count: int
    decisions: Mapping[tuple[str, int], TailConsensusDecision]

    def method_provenance(self) -> dict[str, Any]:
        return {
            "format": TAIL_CONSENSUS_FORMAT,
            "schema_version": TAIL_CONSENSUS_SCHEMA_VERSION,
            "rule": dict(TAIL_CONSENSUS_RULE),
            "tail_consensus_report_sha256": self.report_sha256,
            "tail_consensus_cutoffs_sha256": self.cutoffs_sha256,
            "source_manifest_sha256": self.source_manifest_sha256,
            "cutoff_records_count": self.cutoff_records_count,
            "effective_end_frame_policy": (
                "cutoff_frame_if_should_trim_else_num_frames"
            ),
        }


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
            or int(self.max_candidates_per_episode) != self.max_candidates_per_episode
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(value: Any, *, field: str) -> str:
    digest = str(value).strip().lower()
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError(f"{field} must be a 64-character SHA256 digest")
    return digest


def _strict_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return int(value)


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"{label} does not exist: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"{label} is not valid JSON: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return payload


def validate_calibration(
    payload: Mapping[str, Any], *, expected_algorithm_version: str
) -> dict[str, Any]:
    """Validate a persisted calibration and its content-derived identity."""

    calibration = dict(payload)
    if calibration.get("format") != CALIBRATION_FORMAT:
        raise ValueError(f"calibration format must be {CALIBRATION_FORMAT!r}")
    if calibration.get("algorithm_version") != expected_algorithm_version:
        raise ValueError(
            "calibration algorithm_version does not match requested extraction "
            f"version {expected_algorithm_version!r}"
        )
    if calibration.get("state_column") != STATE_COLUMN:
        raise ValueError(f"calibration state_column must be {STATE_COLUMN!r}")
    split = calibration.get("calibration_split")
    if not isinstance(split, str) or not split:
        raise ValueError("calibration_split must be a non-empty string")

    low_quantile = calibration.get("low_quantile")
    high_quantile = calibration.get("high_quantile")
    if (
        isinstance(low_quantile, bool)
        or isinstance(high_quantile, bool)
        or not isinstance(low_quantile, (int, float))
        or not isinstance(high_quantile, (int, float))
        or not 0.0 <= float(low_quantile) < float(high_quantile) <= 1.0
    ):
        raise ValueError("calibration quantiles must satisfy 0 <= low < high <= 1")

    episode_ids = calibration.get("episode_ids")
    if (
        not isinstance(episode_ids, list)
        or not episode_ids
        or not all(isinstance(item, str) and item for item in episode_ids)
        or len(set(episode_ids)) != len(episode_ids)
        or episode_ids != sorted(episode_ids)
    ):
        raise ValueError("calibration episode_ids must be sorted unique strings")
    _require_sha256(
        calibration.get("calibration_input_sha256"),
        field="calibration_input_sha256",
    )
    state_dim = _strict_int(calibration.get("state_dim"), field="calibration state_dim")
    if state_dim <= 0:
        raise ValueError("calibration state_dim must be positive")
    try:
        state_scale = np.asarray(calibration.get("state_scale"), dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "calibration state_scale must contain finite values"
        ) from error
    if (
        state_scale.shape != (state_dim,)
        or not np.isfinite(state_scale).all()
        or np.any(state_scale <= 0.0)
    ):
        raise ValueError(
            "calibration state_scale must contain state_dim positive values"
        )
    for field in ("distance_low", "distance_high", "distance_median"):
        value = calibration.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"calibration {field} must be finite")
        if not math.isfinite(float(value)):
            raise ValueError(f"calibration {field} must be finite")
    if float(calibration["distance_high"]) <= float(calibration["distance_low"]):
        raise ValueError("calibration distance_high must exceed distance_low")

    identity_fields = (
        "format",
        "algorithm_version",
        "state_column",
        "calibration_split",
        "low_quantile",
        "high_quantile",
        "episode_ids",
        "calibration_input_sha256",
        "state_dim",
        "state_scale",
        "distance_low",
        "distance_high",
        "distance_median",
    )
    identity = {field: calibration[field] for field in identity_fields}
    expected_id = f"state-line-{sha256_json(identity)[:16]}"
    if calibration.get("calibration_id") != expected_id:
        raise ValueError(
            "calibration_id does not match the persisted calibration contents"
        )
    return calibration


def load_calibration(path: Path, *, expected_algorithm_version: str) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    return validate_calibration(
        _load_json_object(resolved, label="calibration"),
        expected_algorithm_version=expected_algorithm_version,
    )


def load_tail_consensus_bundle(
    *,
    report_path: Path,
    cutoffs_path: Path,
    source_manifest_path: Path,
    episode_rows: Sequence[Mapping[str, Any]],
) -> TailConsensusBundle:
    """Load consensus inputs and fail closed before any extraction or write."""

    report_path = report_path.expanduser().resolve()
    cutoffs_path = cutoffs_path.expanduser().resolve()
    source_manifest_path = source_manifest_path.expanduser().resolve()
    report = _load_json_object(report_path, label="tail consensus report")
    if report.get("format") != TAIL_CONSENSUS_FORMAT:
        raise ValueError(
            f"tail consensus report format must be {TAIL_CONSENSUS_FORMAT!r}"
        )
    if report.get("schema_version") != TAIL_CONSENSUS_SCHEMA_VERSION:
        raise ValueError(
            "tail consensus report schema_version must be "
            f"{TAIL_CONSENSUS_SCHEMA_VERSION!r}"
        )
    if report.get("status") != "ok":
        raise ValueError("tail consensus report status must be 'ok'")
    if report.get("rule") != TAIL_CONSENSUS_RULE:
        raise ValueError("tail consensus report rule is not the supported rule")

    outputs = report.get("outputs")
    if not isinstance(outputs, Mapping):
        raise ValueError("tail consensus report outputs must be an object")
    expected_cutoffs_sha256 = _require_sha256(
        outputs.get("cutoff_records_sha256"),
        field="outputs.cutoff_records_sha256",
    )
    expected_cutoff_count = _strict_int(
        outputs.get("cutoff_records_count"),
        field="outputs.cutoff_records_count",
    )
    if expected_cutoff_count < 0:
        raise ValueError("outputs.cutoff_records_count must be non-negative")
    observed_cutoffs_sha256 = sha256_file(cutoffs_path)
    if not hmac.compare_digest(expected_cutoffs_sha256, observed_cutoffs_sha256):
        raise ValueError("tail consensus cutoff hash does not match report")

    expected_source_sha256 = _require_sha256(
        report.get("manifest_sha256"), field="manifest_sha256"
    )
    observed_source_sha256 = sha256_file(source_manifest_path)
    if not hmac.compare_digest(expected_source_sha256, observed_source_sha256):
        raise ValueError("source manifest SHA256 does not match tail consensus report")
    source_manifest = _load_json_object(source_manifest_path, label="source manifest")
    if source_manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"source manifest schema_version must be {SCHEMA_VERSION!r}")
    validate_manifest(source_manifest, strict=True)

    selected_episodes: dict[tuple[str, int], Mapping[str, Any]] = {}
    selected_failures: set[tuple[str, int]] = set()
    for row in episode_rows:
        dataset_id = str(row.get("dataset_id", ""))
        episode_index = row.get("episode_index")
        if not dataset_id:
            raise ValueError("episode ledger row has an empty dataset_id")
        episode_index = _strict_int(
            episode_index, field=f"episode_index for dataset {dataset_id!r}"
        )
        key = (dataset_id, episode_index)
        if key in selected_episodes:
            raise ValueError(f"duplicate selected episode identity: {key!r}")
        selected_episodes[key] = row
        outcome = str(row.get("episode_outcome", ""))
        if outcome not in {"success", "failure"}:
            raise ValueError(
                f"selected episode {key!r} has invalid outcome {outcome!r}"
            )
        if outcome == "failure":
            selected_failures.add(key)

    manifest_failures = {
        (str(sample.get("dataset_id", "")), int(sample["episode_index"]))
        for sample in source_manifest["samples"]
        if sample.get("episode_outcome") == "failure"
    }
    if manifest_failures != selected_failures:
        missing = sorted(selected_failures - manifest_failures)
        extra = sorted(manifest_failures - selected_failures)
        raise ValueError(
            "source manifest failure episode coverage mismatch: "
            f"missing={missing[:10]!r}, extra={extra[:10]!r}"
        )

    decisions: dict[tuple[str, int], TailConsensusDecision] = {}
    try:
        stream = cutoffs_path.open("r", encoding="utf-8")
    except FileNotFoundError as error:
        raise ValueError(
            f"tail consensus cutoffs do not exist: {cutoffs_path}"
        ) from error
    with stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"tail consensus cutoff line {line_number} is invalid JSON: {error}"
                ) from error
            if not isinstance(row, dict):
                raise ValueError(
                    f"tail consensus cutoff line {line_number} must be an object"
                )
            dataset_id = str(row.get("dataset_id", ""))
            if not dataset_id:
                raise ValueError(
                    f"tail consensus cutoff line {line_number} has empty dataset_id"
                )
            episode_index = _strict_int(
                row.get("episode_index"),
                field=f"cutoff line {line_number} episode_index",
            )
            num_frames = _strict_int(
                row.get("num_frames"), field=f"cutoff line {line_number} num_frames"
            )
            cutoff_frame = _strict_int(
                row.get("cutoff_frame"),
                field=f"cutoff line {line_number} cutoff_frame",
            )
            should_trim = row.get("should_trim")
            if not isinstance(should_trim, bool):
                raise ValueError(
                    f"cutoff line {line_number} should_trim must be boolean"
                )
            key = (dataset_id, episode_index)
            if key in decisions:
                raise ValueError(f"duplicate tail consensus cutoff episode: {key!r}")
            if num_frames <= 0:
                raise ValueError(
                    f"tail consensus cutoff {key!r} num_frames must be positive"
                )
            if not 0 < cutoff_frame <= num_frames:
                raise ValueError(
                    f"tail consensus cutoff {key!r} must satisfy 0 < cutoff <= num_frames"
                )
            if should_trim and cutoff_frame >= num_frames:
                raise ValueError(
                    f"tail consensus cutoff {key!r} with should_trim=true must trim"
                )
            if not should_trim and cutoff_frame != num_frames:
                raise ValueError(
                    f"tail consensus cutoff {key!r} with should_trim=false must equal num_frames"
                )
            decisions[key] = TailConsensusDecision(
                num_frames=num_frames,
                should_trim=should_trim,
                effective_end_frame=cutoff_frame,
            )

    if len(decisions) != expected_cutoff_count:
        raise ValueError(
            "tail consensus cutoff count does not match report: "
            f"expected={expected_cutoff_count}, observed={len(decisions)}"
        )
    observed_failures = set(decisions)
    if observed_failures != selected_failures:
        missing = sorted(selected_failures - observed_failures)
        extra = sorted(observed_failures - selected_failures)
        raise ValueError(
            "tail consensus failure episode coverage mismatch: "
            f"missing={missing[:10]!r}, extra={extra[:10]!r}"
        )
    for key, decision in decisions.items():
        ledger_length = _strict_int(
            selected_episodes[key].get("length"),
            field=f"episode ledger length for {key!r}",
        )
        if decision.num_frames != ledger_length:
            raise ValueError(
                f"tail consensus frame count mismatch for {key!r}: "
                f"cutoff={decision.num_frames}, ledger={ledger_length}"
            )

    return TailConsensusBundle(
        report_sha256=sha256_file(report_path),
        cutoffs_sha256=observed_cutoffs_sha256,
        source_manifest_sha256=observed_source_sha256,
        cutoff_records_count=len(decisions),
        decisions=decisions,
    )


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
    tail_consensus_provenance: Mapping[str, Any] | None = None,
) -> str:
    """Identify the complete extraction method, including every threshold."""

    identity = {
        "algorithm_version": algorithm_version,
        "calibration_id": calibration_id,
        "extraction_parameters": parameters.as_dict(),
    }
    if tail_consensus_provenance is not None:
        identity["tail_consensus"] = dict(tail_consensus_provenance)
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


def apply_calibration(
    distances: np.ndarray, calibration: Mapping[str, Any]
) -> np.ndarray:
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
            raise ValueError(
                f"Episode {episode_id} has invalid state shape {states.shape}"
            )
        if state_dim is None:
            state_dim = states.shape[1]
        elif states.shape[1] != state_dim:
            raise ValueError(
                "All calibration episodes must use the same state dimension"
            )
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
    effective_end_frame: int | None = None,
    tail_should_trim: bool | None = None,
    tail_consensus_provenance: Mapping[str, Any] | None = None,
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
    tail_enabled = tail_consensus_provenance is not None
    if tail_enabled:
        if effective_end_frame is None or tail_should_trim is None:
            raise ValueError(
                "tail consensus extraction requires effective_end_frame and "
                "tail_should_trim"
            )
    elif effective_end_frame is not None or tail_should_trim is not None:
        raise ValueError(
            "effective_end_frame and tail_should_trim require tail consensus provenance"
        )
    resolved_end_frame = (
        expected_length if effective_end_frame is None else int(effective_end_frame)
    )
    if not 0 < resolved_end_frame <= expected_length:
        raise ValueError(
            f"Episode {episode['episode_id']} effective_end_frame must satisfy "
            "0 < effective_end_frame <= length"
        )
    if outcome == "success" and resolved_end_frame != expected_length:
        raise ValueError("Tail consensus must not shorten success episodes")
    if tail_enabled:
        assert tail_should_trim is not None
        if tail_should_trim != (resolved_end_frame < expected_length):
            raise ValueError(
                f"Episode {episode['episode_id']} tail_should_trim conflicts with "
                "effective_end_frame"
            )

    distances, scores = episode_scores(states, calibration)
    full_extraction = extract_candidate_windows(scores, **parameters.extractor_kwargs())
    extraction = (
        full_extraction
        if resolved_end_frame == expected_length
        else extract_candidate_windows(
            scores[:resolved_end_frame], **parameters.extractor_kwargs()
        )
    )
    calibration_id = str(calibration["calibration_id"])
    action_loss = "enabled" if outcome == "success" else "disabled"
    sample_role = "success_event" if outcome == "success" else "failure_context"
    annotation_parameters = parameters.as_dict()

    event_rows: list[dict[str, Any]] = []
    for index, candidate in enumerate(extraction.candidates):
        exceeds_max_candidate = (
            candidate.end_frame - candidate.start_frame > parameters.max_candidate
        )
        annotation = {
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
        }
        if tail_enabled:
            annotation["tail_consensus"] = {
                **dict(tail_consensus_provenance or {}),
                "should_trim": bool(tail_should_trim),
                "effective_end_frame": resolved_end_frame,
            }
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
                "annotation": annotation,
                "split": str(episode.get("split", "train")),
            }
        )

    frame_rows: list[dict[str, Any]] = []
    for frame_index in range(expected_length):
        row = {
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
                float(full_extraction.smoothed_scores[frame_index])
                if np.isfinite(full_extraction.smoothed_scores[frame_index])
                else None
            ),
            "active_candidate": bool(
                frame_index < resolved_end_frame and extraction.active_mask[frame_index]
            ),
            "algorithm_version": algorithm_version,
            "calibration_id": calibration_id,
            "method_id": method_id,
        }
        if tail_enabled:
            provenance = dict(tail_consensus_provenance or {})
            row.update(
                {
                    "effective_end_frame": resolved_end_frame,
                    "visible_for_candidate_extraction": (
                        frame_index < resolved_end_frame
                    ),
                    "tail_should_trim": bool(tail_should_trim),
                    "tail_consensus_report_sha256": provenance[
                        "tail_consensus_report_sha256"
                    ],
                    "tail_consensus_cutoffs_sha256": provenance[
                        "tail_consensus_cutoffs_sha256"
                    ],
                    "tail_source_manifest_sha256": provenance["source_manifest_sha256"],
                }
            )
        frame_rows.append(row)
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
        json.loads(info_path.read_text(encoding="utf-8")) if info_path.exists() else {}
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
        (dataset_root / "data").glob(f"chunk-*/episode_{episode_index:06d}.parquet")
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
            raise ValueError(
                f"Calibration path already contains different data: {path}"
            )
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
    scores_writer: Callable[
        [Path, Sequence[Mapping[str, Any]]], None
    ] = write_scores_parquet,
    calibration: Mapping[str, Any] | None = None,
    calibration_source_path: Path | None = None,
    tail_consensus: TailConsensusBundle | None = None,
) -> dict[str, Any]:
    """Run extraction with injectable I/O for dependency-light tests."""

    states_by_id: dict[str, np.ndarray] = {}
    for episode in episode_rows:
        states_by_id[str(episode["episode_id"])] = np.asarray(
            state_loader(episode), dtype=np.float64
        )

    if calibration is None and calibration_source_path is not None:
        raise ValueError("calibration_source_path requires a frozen calibration")

    if calibration is None:
        calibration_inputs = [
            (str(episode["episode_id"]), states_by_id[str(episode["episode_id"])])
            for episode in episode_rows
            if str(episode.get("split", "train")) == calibration_split
        ]
        resolved_calibration = fit_robust_calibration(
            calibration_inputs,
            calibration_split=calibration_split,
            low_quantile=low_quantile,
            high_quantile=high_quantile,
            algorithm_version=algorithm_version,
        )
        resolved_calibration_path = (
            eve_root
            / "annotations"
            / "calibrations"
            / f"{resolved_calibration['calibration_id']}.json"
        )
        persist_calibration(resolved_calibration_path, resolved_calibration)
    else:
        resolved_calibration = validate_calibration(
            calibration, expected_algorithm_version=algorithm_version
        )
        resolved_calibration_path = (
            calibration_source_path.expanduser().resolve()
            if calibration_source_path is not None
            else eve_root
            / "annotations"
            / "calibrations"
            / f"{resolved_calibration['calibration_id']}.json"
        )
        if calibration_source_path is None:
            persist_calibration(resolved_calibration_path, resolved_calibration)
    tail_provenance = (
        tail_consensus.method_provenance() if tail_consensus is not None else None
    )
    method_id = make_method_id(
        algorithm_version=algorithm_version,
        calibration_id=str(resolved_calibration["calibration_id"]),
        parameters=parameters,
        tail_consensus_provenance=tail_provenance,
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
        outcome = str(episode.get("episode_outcome", ""))
        effective_end_frame: int | None = None
        tail_should_trim: bool | None = None
        if tail_consensus is not None:
            expected_length = int(episode["length"])
            if outcome == "failure":
                key = (str(episode["dataset_id"]), int(episode["episode_index"]))
                decision = tail_consensus.decisions.get(key)
                if decision is None:
                    raise ValueError(f"Missing tail consensus decision for {key!r}")
                if decision.num_frames != expected_length:
                    raise ValueError(
                        f"Tail consensus decision for {key!r} has num_frames "
                        f"{decision.num_frames}, expected {expected_length}"
                    )
                effective_end_frame = decision.effective_end_frame
                tail_should_trim = decision.should_trim
            else:
                effective_end_frame = expected_length
                tail_should_trim = False
        events, scores = extract_episode_rows(
            episode,
            states_by_id[str(episode["episode_id"])],
            resolved_calibration,
            parameters=parameters,
            scores_artifact=scores_artifact,
            method_id=method_id,
            algorithm_version=algorithm_version,
            effective_end_frame=effective_end_frame,
            tail_should_trim=tail_should_trim,
            tail_consensus_provenance=tail_provenance,
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
    result = {
        "calibration": resolved_calibration,
        "calibration_path": str(resolved_calibration_path),
        "method_id": method_id,
        "scores_sha256": scores_sha256,
        "scores_path": str(resolved_scores_path),
        "num_episodes": len(episode_rows),
        "num_score_rows": len(all_scores),
        "num_candidates": len(all_events),
        "num_appended_candidates": appended,
    }
    if tail_provenance is not None:
        result["tail_consensus"] = tail_provenance
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
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
    parser.add_argument(
        "--calibration",
        type=Path,
        default=None,
        help=(
            "Load an existing train-only calibration JSON instead of fitting a "
            "new calibration from selected episodes."
        ),
    )
    parser.add_argument("--algorithm-version", default=ALGORITHM_VERSION)
    parser.add_argument("--scores-path", type=Path, default=None)
    parser.add_argument(
        "--tail-consensus-report",
        type=Path,
        default=None,
        help="Validated tail_consensus_report.json for failure-only visibility.",
    )
    parser.add_argument(
        "--tail-consensus-cutoffs",
        type=Path,
        default=None,
        help="Validated tail_consensus_cutoffs.jsonl paired with the report.",
    )
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=None,
        help="Exact source manifest whose file SHA256 is bound by the report.",
    )
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
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
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
    if args.calibration is None and not any(
        str(row.get("split", "train")) == args.calibration_split for row in rows
    ):
        raise ValueError(
            f"No selected episodes belong to calibration split "
            f"{args.calibration_split!r}"
        )
    tail_paths = (
        args.tail_consensus_report,
        args.tail_consensus_cutoffs,
        args.source_manifest,
    )
    if any(path is not None for path in tail_paths) and not all(
        path is not None for path in tail_paths
    ):
        raise ValueError(
            "--tail-consensus-report, --tail-consensus-cutoffs, and "
            "--source-manifest must be provided together"
        )
    tail_consensus = None
    if all(path is not None for path in tail_paths):
        assert args.tail_consensus_report is not None
        assert args.tail_consensus_cutoffs is not None
        assert args.source_manifest is not None
        tail_consensus = load_tail_consensus_bundle(
            report_path=args.tail_consensus_report,
            cutoffs_path=args.tail_consensus_cutoffs,
            source_manifest_path=args.source_manifest,
            episode_rows=rows,
        )
    frozen_calibration = (
        load_calibration(
            args.calibration, expected_algorithm_version=args.algorithm_version
        )
        if args.calibration is not None
        else None
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
        calibration=frozen_calibration,
        calibration_source_path=(
            args.calibration.expanduser().resolve()
            if args.calibration is not None
            else None
        ),
        tail_consensus=tail_consensus,
    )
    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
