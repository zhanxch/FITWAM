#!/usr/bin/env python3
"""Validate EveRobot state-line candidates and offline event-pair quality.

The validator is read-only with respect to all input ledgers. It recomputes
split-level metrics from episode, event, and pair rows, checks that the pair
diagnostics describe the same frozen pair selection, and atomically writes one
JSON report. A quality-gate failure returns exit code 1; malformed or
inconsistent inputs return exit code 2.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import tempfile
from typing import Any, Mapping, Sequence


REPORT_FORMAT = "EveRobotOfflineEventPairQuality"
REPORT_SCHEMA_VERSION = "0.1"
DEFAULT_SPLITS = ("train", "val")
OUTCOMES = ("success", "failure")


def canonical_json(payload: Any) -> str:
    return json.dumps(
        payload,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


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
                value = json.loads(stripped)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from error
            if not isinstance(value, dict):
                raise ValueError(
                    f"{path}:{line_number}: expected one JSON object per line"
                )
            rows.append(value)
    return rows


def read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{path}: invalid JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = (canonical_json(dict(payload)) + "\n").encode("utf-8")
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary_path.unlink(missing_ok=True)


def _nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _finite_float(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def _index_unique(
    rows: Sequence[dict[str, Any]], key: str, label: str
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        identity = _nonempty_string(row.get(key), f"{label}[{index}].{key}")
        if identity in indexed:
            raise ValueError(f"Duplicate {key} in {label}: {identity}")
        indexed[identity] = row
    return indexed


def _linear_percentile(values: Sequence[int], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _is_long_candidate(event: Mapping[str, Any], label: str) -> bool:
    marker = event.get("exceeds_max_candidate")
    if isinstance(marker, bool):
        return marker
    annotation = event.get("annotation")
    parameters = (
        annotation.get("parameters")
        if isinstance(annotation, Mapping)
        else None
    )
    max_candidate = (
        parameters.get("max_candidate")
        if isinstance(parameters, Mapping)
        else None
    )
    if (
        isinstance(max_candidate, bool)
        or not isinstance(max_candidate, int)
        or max_candidate <= 0
    ):
        raise ValueError(
            f"{label} requires boolean exceeds_max_candidate or a positive "
            "annotation.parameters.max_candidate"
        )
    start = event.get("start_frame")
    end = event.get("end_frame")
    if (
        isinstance(start, bool)
        or isinstance(end, bool)
        or not isinstance(start, int)
        or not isinstance(end, int)
        or start < 0
        or end <= start
    ):
        raise ValueError(f"{label} has an invalid frame interval")
    return end - start > max_candidate


def _validate_episode(
    row: Mapping[str, Any], index: int, splits: set[str]
) -> tuple[str, str]:
    split = _nonempty_string(row.get("split"), f"episode_meta[{index}].split")
    outcome = _nonempty_string(
        row.get("episode_outcome"), f"episode_meta[{index}].episode_outcome"
    )
    if split not in splits:
        return split, outcome
    if outcome not in OUTCOMES:
        raise ValueError(
            f"episode_meta[{index}].episode_outcome must be success or failure"
        )
    return split, outcome


def _candidate_metrics(
    episode_rows: Sequence[dict[str, Any]],
    event_rows: Sequence[dict[str, Any]],
    *,
    splits: Sequence[str],
) -> tuple[
    dict[str, dict[str, dict[str, Any]]],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    split_set = set(splits)
    episodes_by_id = _index_unique(episode_rows, "episode_id", "episode_meta")
    _index_unique(event_rows, "event_id", "event_meta")
    eligible_episodes: dict[tuple[str, str], list[str]] = {
        (split, outcome): [] for split in splits for outcome in OUTCOMES
    }
    episode_class: dict[str, tuple[str, str]] = {}
    for index, row in enumerate(episode_rows):
        episode_id = _nonempty_string(
            row.get("episode_id"), f"episode_meta[{index}].episode_id"
        )
        split, outcome = _validate_episode(row, index, split_set)
        if split in split_set:
            eligible_episodes[(split, outcome)].append(episode_id)
            episode_class[episode_id] = (split, outcome)

    candidate_events: dict[str, dict[str, Any]] = {}
    candidate_counts: dict[str, int] = {
        episode_id: 0 for episode_id in episode_class
    }
    candidate_long: dict[tuple[str, str], int] = {
        key: 0 for key in eligible_episodes
    }
    for index, event in enumerate(event_rows):
        if event.get("event_type") != "interaction_candidate":
            continue
        event_id = _nonempty_string(
            event.get("event_id"), f"event_meta[{index}].event_id"
        )
        episode_id = _nonempty_string(
            event.get("episode_id"), f"event_meta[{index}].episode_id"
        )
        episode = episodes_by_id.get(episode_id)
        if episode is None:
            raise ValueError(
                f"event_meta[{index}] references unknown episode_id={episode_id}"
            )
        episode_split = _nonempty_string(
            episode.get("split"), f"episode[{episode_id}].split"
        )
        episode_outcome = _nonempty_string(
            episode.get("episode_outcome"),
            f"episode[{episode_id}].episode_outcome",
        )
        event_split = _nonempty_string(
            event.get("split"), f"event_meta[{index}].split"
        )
        event_outcome = _nonempty_string(
            event.get("episode_outcome"),
            f"event_meta[{index}].episode_outcome",
        )
        if event_split != episode_split or event_outcome != episode_outcome:
            raise ValueError(
                f"event_meta[{index}] split/outcome disagrees with episode "
                f"{episode_id}"
            )
        if episode_split not in split_set:
            continue
        if episode_outcome not in OUTCOMES:
            raise ValueError(
                f"event_meta[{index}] has unsupported outcome {episode_outcome!r}"
            )
        start = event.get("start_frame")
        end = event.get("end_frame")
        if (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, int)
            or not isinstance(end, int)
            or start < 0
            or end <= start
        ):
            raise ValueError(f"event_meta[{index}] has an invalid frame interval")
        episode_length = episode.get("length")
        if (
            isinstance(episode_length, bool)
            or not isinstance(episode_length, int)
            or episode_length <= 0
            or end > episode_length
        ):
            raise ValueError(
                f"event_meta[{index}] interval exceeds episode length"
            )
        key = (episode_split, episode_outcome)
        candidate_counts[episode_id] += 1
        if _is_long_candidate(event, f"event_meta[{index}]"):
            candidate_long[key] += 1
        candidate_events[event_id] = event

    metrics: dict[str, dict[str, dict[str, Any]]] = {}
    for split in splits:
        metrics[split] = {}
        for outcome in OUTCOMES:
            episode_ids = eligible_episodes[(split, outcome)]
            counts = [candidate_counts[episode_id] for episode_id in episode_ids]
            candidate_count = sum(counts)
            episodes_with_candidates = sum(count > 0 for count in counts)
            long_count = candidate_long[(split, outcome)]
            metrics[split][outcome] = {
                "episode_count": len(episode_ids),
                "episodes_with_candidates": episodes_with_candidates,
                "candidate_episode_coverage": (
                    episodes_with_candidates / len(episode_ids)
                    if episode_ids
                    else 0.0
                ),
                "candidate_count": candidate_count,
                "events_per_episode": {
                    "median": (
                        float(statistics.median(counts)) if counts else 0.0
                    ),
                    "p95": _linear_percentile(counts, 0.95),
                    "percentile_method": "linear",
                },
                "long_candidate_count": long_count,
                "long_candidate_ratio": (
                    long_count / candidate_count if candidate_count else 0.0
                ),
            }
    return metrics, episodes_by_id, candidate_events


def _validate_diagnostics(
    diagnostics: Mapping[str, Any],
    pair_rows: Sequence[dict[str, Any]],
    candidate_events: Mapping[str, Mapping[str, Any]],
    *,
    splits: Sequence[str],
) -> dict[str, Any]:
    failure_rows = diagnostics.get("failure_events")
    if not isinstance(failure_rows, list) or not all(
        isinstance(row, dict) for row in failure_rows
    ):
        raise ValueError("pair diagnostics requires failure_events as a list")
    coverage = diagnostics.get("coverage")
    if not isinstance(coverage, Mapping):
        raise ValueError("pair diagnostics requires coverage")
    total_failure_events = coverage.get("total_failure_events")
    if (
        isinstance(total_failure_events, bool)
        or not isinstance(total_failure_events, int)
        or total_failure_events != len(failure_rows)
    ):
        raise ValueError(
            "pair diagnostics total_failure_events does not match "
            "failure_events"
        )
    selected_pairs = coverage.get("selected_pairs")
    if (
        isinstance(selected_pairs, bool)
        or not isinstance(selected_pairs, int)
        or selected_pairs != len(pair_rows)
    ):
        raise ValueError(
            "pair diagnostics selected_pairs does not match pair ledger"
        )
    weight_distribution = diagnostics.get("pair_weight_distribution")
    if not isinstance(weight_distribution, Mapping):
        raise ValueError("pair diagnostics requires pair_weight_distribution")
    weight_count = weight_distribution.get("count")
    if (
        isinstance(weight_count, bool)
        or not isinstance(weight_count, int)
        or weight_count != len(pair_rows)
    ):
        raise ValueError(
            "pair diagnostics weight count does not match pair ledger"
        )

    pairs_by_failure: dict[str, list[str]] = {}
    for index, pair in enumerate(pair_rows):
        failure_event_id = _nonempty_string(
            pair.get("failure_event_id"),
            f"pair_ledger[{index}].failure_event_id",
        )
        pair_id = _nonempty_string(
            pair.get("pair_id"), f"pair_ledger[{index}].pair_id"
        )
        pairs_by_failure.setdefault(failure_event_id, []).append(pair_id)

    diagnostic_ids: set[str] = set()
    selected_diagnostic_ids: set[str] = set()
    for index, row in enumerate(failure_rows):
        event_id = _nonempty_string(
            row.get("failure_event_id"),
            f"pair_diagnostics.failure_events[{index}].failure_event_id",
        )
        if event_id in diagnostic_ids:
            raise ValueError(
                f"Duplicate failure_event_id in pair diagnostics: {event_id}"
            )
        diagnostic_ids.add(event_id)
        event = candidate_events.get(event_id)
        if event is None:
            raise ValueError(
                f"pair diagnostics references unknown candidate event {event_id}"
            )
        event_split = _nonempty_string(
            event.get("split"), f"event[{event_id}].split"
        )
        diagnostic_split = _nonempty_string(
            row.get("split"), f"pair_diagnostics.failure_events[{index}].split"
        )
        if diagnostic_split != event_split:
            raise ValueError(
                f"pair diagnostics split disagrees for failure event {event_id}"
            )
        selected = row.get("selected")
        if not isinstance(selected, bool):
            raise ValueError(
                f"pair diagnostics selected flag must be boolean for {event_id}"
            )
        selected_pair_ids = row.get("selected_pair_ids")
        if not isinstance(selected_pair_ids, list) or not all(
            isinstance(pair_id, str) and pair_id for pair_id in selected_pair_ids
        ):
            raise ValueError(
                f"pair diagnostics selected_pair_ids is invalid for {event_id}"
            )
        expected_pair_ids = sorted(pairs_by_failure.get(event_id, []))
        if sorted(selected_pair_ids) != expected_pair_ids:
            raise ValueError(
                f"pair diagnostics selected_pair_ids disagrees for {event_id}"
            )
        if selected != bool(expected_pair_ids):
            raise ValueError(
                f"pair diagnostics selected flag disagrees for {event_id}"
            )
        if selected:
            selected_diagnostic_ids.add(event_id)

    required_failure_ids = {
        event_id
        for event_id, event in candidate_events.items()
        if event.get("episode_outcome") == "failure"
        and event.get("split") in set(splits)
    }
    if diagnostic_ids != required_failure_ids:
        missing = sorted(required_failure_ids - diagnostic_ids)
        extra = sorted(diagnostic_ids - required_failure_ids)
        raise ValueError(
            "pair diagnostics failure-event set does not match event ledger: "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )
    if selected_diagnostic_ids != set(pairs_by_failure):
        raise ValueError(
            "pair diagnostics selected failure events do not match pair ledger"
        )
    selected_failure_events = coverage.get("selected_failure_events")
    if (
        isinstance(selected_failure_events, bool)
        or not isinstance(selected_failure_events, int)
        or selected_failure_events != len(selected_diagnostic_ids)
    ):
        raise ValueError(
            "pair diagnostics selected_failure_events does not match "
            "failure_events"
        )
    reported_coverage = _finite_float(
        coverage.get("failure_event_coverage"),
        "pair_diagnostics.coverage.failure_event_coverage",
    )
    actual_coverage = (
        len(selected_diagnostic_ids) / len(diagnostic_ids)
        if diagnostic_ids
        else 0.0
    )
    if not math.isclose(
        reported_coverage, actual_coverage, rel_tol=1e-9, abs_tol=1e-12
    ):
        raise ValueError(
            "pair diagnostics failure_event_coverage does not match "
            "failure_events"
        )
    return {
        "status": "consistent",
        "failure_event_count": len(diagnostic_ids),
        "selected_failure_event_count": len(selected_diagnostic_ids),
        "selected_pair_count": len(pair_rows),
    }


def _pair_metrics(
    pair_rows: Sequence[dict[str, Any]],
    episodes_by_id: Mapping[str, Mapping[str, Any]],
    candidate_events: Mapping[str, Mapping[str, Any]],
    *,
    splits: Sequence[str],
    low_weight_cutoff: float,
) -> dict[str, dict[str, Any]]:
    split_set = set(splits)
    rows_by_split: dict[str, list[dict[str, Any]]] = {
        split: [] for split in splits
    }
    failure_episode_by_event: dict[str, str] = {}
    for index, pair in enumerate(pair_rows):
        split = _nonempty_string(
            pair.get("split"), f"pair_ledger[{index}].split"
        )
        if split not in split_set:
            continue
        success_event_id = _nonempty_string(
            pair.get("success_event_id"),
            f"pair_ledger[{index}].success_event_id",
        )
        failure_event_id = _nonempty_string(
            pair.get("failure_event_id"),
            f"pair_ledger[{index}].failure_event_id",
        )
        success_event = candidate_events.get(success_event_id)
        failure_event = candidate_events.get(failure_event_id)
        if success_event is None or failure_event is None:
            raise ValueError(
                f"pair_ledger[{index}] references an unknown candidate event"
            )
        if (
            success_event.get("episode_outcome") != "success"
            or failure_event.get("episode_outcome") != "failure"
        ):
            raise ValueError(
                f"pair_ledger[{index}] does not link success to failure"
            )
        if success_event.get("split") != split or failure_event.get("split") != split:
            raise ValueError(
                f"pair_ledger[{index}] crosses split boundaries"
            )
        failure_episode_id = _nonempty_string(
            failure_event.get("episode_id"),
            f"event[{failure_event_id}].episode_id",
        )
        if failure_episode_id not in episodes_by_id:
            raise ValueError(
                f"failure event {failure_event_id} references an unknown episode"
            )
        failure_episode_by_event[failure_event_id] = failure_episode_id
        weight = _finite_float(
            pair.get("pair_weight"), f"pair_ledger[{index}].pair_weight"
        )
        if not 0.0 < weight <= 1.0:
            raise ValueError(
                f"pair_ledger[{index}].pair_weight must be in (0, 1]"
            )
        rows_by_split[split].append(pair)

    metrics: dict[str, dict[str, Any]] = {}
    for split in splits:
        rows = rows_by_split[split]
        failure_events = {
            event_id
            for event_id, event in candidate_events.items()
            if event.get("split") == split
            and event.get("episode_outcome") == "failure"
        }
        paired_failure_events = {
            str(row["failure_event_id"]) for row in rows
        }
        episode_pair_counts: dict[str, int] = {}
        for row in rows:
            episode_id = failure_episode_by_event[str(row["failure_event_id"])]
            episode_pair_counts[episode_id] = (
                episode_pair_counts.get(episode_id, 0) + 1
            )
        weights = [float(row["pair_weight"]) for row in rows]
        low_count = sum(weight <= low_weight_cutoff for weight in weights)
        metrics[split] = {
            "pair_count": len(rows),
            "failure_event_count": len(failure_events),
            "paired_failure_event_count": len(paired_failure_events),
            "failure_event_coverage": (
                len(paired_failure_events) / len(failure_events)
                if failure_events
                else 0.0
            ),
            "unique_failure_episode_count": len(episode_pair_counts),
            "max_single_failure_episode_pair_share": (
                max(episode_pair_counts.values()) / len(rows)
                if rows
                else 0.0
            ),
            "pair_weight_median": (
                float(statistics.median(weights)) if weights else 0.0
            ),
            "low_pair_weight_cutoff": low_weight_cutoff,
            "low_pair_weight_count": low_count,
            "low_pair_weight_ratio": (
                low_count / len(weights) if weights else 0.0
            ),
        }
    return metrics


def _check(
    checks: list[dict[str, Any]],
    *,
    name: str,
    scope: str,
    actual: float | int,
    operator: str,
    threshold: float | int,
) -> None:
    if operator == ">=":
        passed = actual >= threshold
    elif operator == "<=":
        passed = actual <= threshold
    else:
        raise ValueError(f"Unsupported check operator: {operator}")
    checks.append(
        {
            "name": name,
            "scope": scope,
            "actual": actual,
            "operator": operator,
            "threshold": threshold,
            "passed": passed,
        }
    )


def _threshold_checks(
    candidate_metrics: Mapping[str, Mapping[str, Mapping[str, Any]]],
    pair_metrics: Mapping[str, Mapping[str, Any]],
    thresholds: Mapping[str, Any],
    *,
    splits: Sequence[str],
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for split in splits:
        minimum_candidates = int(
            thresholds[
                "min_train_outcome_candidates"
                if split == "train"
                else "min_val_outcome_candidates"
            ]
        )
        for outcome in OUTCOMES:
            scope = f"{split}/{outcome}"
            metrics = candidate_metrics[split][outcome]
            _check(
                checks,
                name="candidate_episode_coverage",
                scope=scope,
                actual=float(metrics["candidate_episode_coverage"]),
                operator=">=",
                threshold=float(thresholds["min_candidate_episode_coverage"]),
            )
            _check(
                checks,
                name="candidate_count",
                scope=scope,
                actual=int(metrics["candidate_count"]),
                operator=">=",
                threshold=minimum_candidates,
            )
            median = float(metrics["events_per_episode"]["median"])
            _check(
                checks,
                name="events_per_episode_median_min",
                scope=scope,
                actual=median,
                operator=">=",
                threshold=float(thresholds["min_events_per_episode_median"]),
            )
            _check(
                checks,
                name="events_per_episode_median_max",
                scope=scope,
                actual=median,
                operator="<=",
                threshold=float(thresholds["max_events_per_episode_median"]),
            )
            _check(
                checks,
                name="events_per_episode_p95",
                scope=scope,
                actual=float(metrics["events_per_episode"]["p95"]),
                operator="<=",
                threshold=float(thresholds["max_events_per_episode_p95"]),
            )
            _check(
                checks,
                name="long_candidate_ratio",
                scope=scope,
                actual=float(metrics["long_candidate_ratio"]),
                operator="<=",
                threshold=float(thresholds["max_long_candidate_ratio"]),
            )

        metrics = pair_metrics[split]
        _check(
            checks,
            name="pair_count",
            scope=split,
            actual=int(metrics["pair_count"]),
            operator=">=",
            threshold=int(
                thresholds["min_train_pairs" if split == "train" else "min_val_pairs"]
            ),
        )
        _check(
            checks,
            name="failure_event_coverage",
            scope=split,
            actual=float(metrics["failure_event_coverage"]),
            operator=">=",
            threshold=float(
                thresholds[
                    "min_train_failure_coverage"
                    if split == "train"
                    else "min_val_failure_coverage"
                ]
            ),
        )
        _check(
            checks,
            name="unique_failure_episode_count",
            scope=split,
            actual=int(metrics["unique_failure_episode_count"]),
            operator=">=",
            threshold=int(
                thresholds[
                    "min_train_failure_episodes"
                    if split == "train"
                    else "min_val_failure_episodes"
                ]
            ),
        )
        _check(
            checks,
            name="max_single_failure_episode_pair_share",
            scope=split,
            actual=float(metrics["max_single_failure_episode_pair_share"]),
            operator="<=",
            threshold=float(
                thresholds[
                    "max_train_single_episode_pair_share"
                    if split == "train"
                    else "max_val_single_episode_pair_share"
                ]
            ),
        )
        _check(
            checks,
            name="pair_weight_median",
            scope=split,
            actual=float(metrics["pair_weight_median"]),
            operator=">=",
            threshold=float(thresholds["min_pair_weight_median"]),
        )
        _check(
            checks,
            name="low_pair_weight_ratio",
            scope=split,
            actual=float(metrics["low_pair_weight_ratio"]),
            operator="<=",
            threshold=float(thresholds["max_low_pair_weight_ratio"]),
        )
    return checks


def validate(
    *,
    episode_meta: Path,
    event_meta: Path,
    pair_ledger: Path,
    pair_diagnostics: Path,
    splits: Sequence[str],
    thresholds: Mapping[str, Any],
) -> dict[str, Any]:
    if not splits or len(set(splits)) != len(splits):
        raise ValueError("splits must be non-empty and unique")
    unsupported = set(splits) - set(DEFAULT_SPLITS)
    if unsupported:
        raise ValueError(
            f"Only train and val splits have defined thresholds: {sorted(unsupported)}"
        )
    episode_rows = read_jsonl(episode_meta)
    event_rows = read_jsonl(event_meta)
    pair_rows = read_jsonl(pair_ledger)
    diagnostics = read_json_object(pair_diagnostics)
    _index_unique(pair_rows, "pair_id", "pair_ledger")

    candidate_metrics, episodes_by_id, candidate_events = _candidate_metrics(
        episode_rows,
        event_rows,
        splits=splits,
    )
    diagnostics_consistency = _validate_diagnostics(
        diagnostics,
        pair_rows,
        candidate_events,
        splits=splits,
    )
    pair_metrics = _pair_metrics(
        pair_rows,
        episodes_by_id,
        candidate_events,
        splits=splits,
        low_weight_cutoff=float(thresholds["low_pair_weight_cutoff"]),
    )
    checks = _threshold_checks(
        candidate_metrics,
        pair_metrics,
        thresholds,
        splits=splits,
    )
    failed_checks = [check for check in checks if not check["passed"]]
    inputs = {
        "episode_meta": {
            "path": str(episode_meta.resolve()),
            "sha256": file_sha256(episode_meta),
            "rows": len(episode_rows),
        },
        "event_meta": {
            "path": str(event_meta.resolve()),
            "sha256": file_sha256(event_meta),
            "rows": len(event_rows),
        },
        "pair_ledger": {
            "path": str(pair_ledger.resolve()),
            "sha256": file_sha256(pair_ledger),
            "rows": len(pair_rows),
        },
        "pair_diagnostics": {
            "path": str(pair_diagnostics.resolve()),
            "sha256": file_sha256(pair_diagnostics),
        },
    }
    return {
        "format": REPORT_FORMAT,
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": "passed" if not failed_checks else "failed",
        "splits": list(splits),
        "inputs": inputs,
        "thresholds": dict(thresholds),
        "candidate_metrics": candidate_metrics,
        "pair_metrics": pair_metrics,
        "diagnostics_consistency": diagnostics_consistency,
        "checks": checks,
        "failed_checks": failed_checks,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-meta", type=Path, required=True)
    parser.add_argument("--event-meta", type=Path, required=True)
    parser.add_argument("--pair-ledger", type=Path, required=True)
    parser.add_argument("--pair-diagnostics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=list(DEFAULT_SPLITS))

    parser.add_argument("--min-candidate-episode-coverage", type=float, default=0.8)
    parser.add_argument("--min-train-outcome-candidates", type=int, default=32)
    parser.add_argument("--min-val-outcome-candidates", type=int, default=8)
    parser.add_argument("--min-events-per-episode-median", type=float, default=1.0)
    parser.add_argument("--max-events-per-episode-median", type=float, default=6.0)
    parser.add_argument("--max-events-per-episode-p95", type=float, default=10.0)
    parser.add_argument("--max-long-candidate-ratio", type=float, default=0.1)

    parser.add_argument("--min-train-pairs", type=int, default=32)
    parser.add_argument("--min-val-pairs", type=int, default=8)
    parser.add_argument("--min-train-failure-coverage", type=float, default=0.30)
    parser.add_argument("--min-val-failure-coverage", type=float, default=0.25)
    parser.add_argument("--min-train-failure-episodes", type=int, default=16)
    parser.add_argument("--min-val-failure-episodes", type=int, default=4)
    parser.add_argument(
        "--max-train-single-episode-pair-share",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--max-val-single-episode-pair-share",
        type=float,
        default=0.25,
    )
    parser.add_argument("--min-pair-weight-median", type=float, default=0.10)
    parser.add_argument("--low-pair-weight-cutoff", type=float, default=0.05)
    parser.add_argument("--max-low-pair-weight-ratio", type=float, default=0.25)
    return parser.parse_args(argv)


def _thresholds_from_args(args: argparse.Namespace) -> dict[str, Any]:
    names = (
        "min_candidate_episode_coverage",
        "min_train_outcome_candidates",
        "min_val_outcome_candidates",
        "min_events_per_episode_median",
        "max_events_per_episode_median",
        "max_events_per_episode_p95",
        "max_long_candidate_ratio",
        "min_train_pairs",
        "min_val_pairs",
        "min_train_failure_coverage",
        "min_val_failure_coverage",
        "min_train_failure_episodes",
        "min_val_failure_episodes",
        "max_train_single_episode_pair_share",
        "max_val_single_episode_pair_share",
        "min_pair_weight_median",
        "low_pair_weight_cutoff",
        "max_low_pair_weight_ratio",
    )
    thresholds = {name: getattr(args, name) for name in names}
    nonnegative = (
        "min_train_outcome_candidates",
        "min_val_outcome_candidates",
        "min_train_pairs",
        "min_val_pairs",
        "min_train_failure_episodes",
        "min_val_failure_episodes",
    )
    for name in nonnegative:
        if thresholds[name] < 0:
            raise ValueError(f"{name} must be non-negative")
    bounded = (
        "min_candidate_episode_coverage",
        "max_long_candidate_ratio",
        "min_train_failure_coverage",
        "min_val_failure_coverage",
        "max_train_single_episode_pair_share",
        "max_val_single_episode_pair_share",
        "min_pair_weight_median",
        "low_pair_weight_cutoff",
        "max_low_pair_weight_ratio",
    )
    for name in bounded:
        value = _finite_float(thresholds[name], name)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")
    median_minimum = _finite_float(
        thresholds["min_events_per_episode_median"],
        "min_events_per_episode_median",
    )
    median_maximum = _finite_float(
        thresholds["max_events_per_episode_median"],
        "max_events_per_episode_median",
    )
    p95_maximum = _finite_float(
        thresholds["max_events_per_episode_p95"],
        "max_events_per_episode_p95",
    )
    if median_minimum < 0:
        raise ValueError("min_events_per_episode_median must be non-negative")
    if median_maximum < median_minimum:
        raise ValueError(
            "max_events_per_episode_median must be at least the minimum"
        )
    if p95_maximum < 0:
        raise ValueError("max_events_per_episode_p95 must be non-negative")
    return thresholds


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_path = args.output.resolve()
    input_paths = {
        args.episode_meta.resolve(),
        args.event_meta.resolve(),
        args.pair_ledger.resolve(),
        args.pair_diagnostics.resolve(),
    }
    if output_path in input_paths:
        print(
            "Output path must differ from every read-only input path",
            file=sys.stderr,
        )
        return 2
    try:
        thresholds = _thresholds_from_args(args)
        report = validate(
            episode_meta=args.episode_meta.resolve(),
            event_meta=args.event_meta.resolve(),
            pair_ledger=args.pair_ledger.resolve(),
            pair_diagnostics=args.pair_diagnostics.resolve(),
            splits=args.splits,
            thresholds=thresholds,
        )
        exit_code = 0 if report["status"] == "passed" else 1
    except (OSError, ValueError, TypeError, KeyError) as error:
        report = {
            "format": REPORT_FORMAT,
            "schema_version": REPORT_SCHEMA_VERSION,
            "status": "error",
            "error": str(error),
        }
        exit_code = 2
    write_json_atomic(output_path, report)
    print(canonical_json(report))
    if exit_code != 0:
        print(
            f"Offline event/pair quality validation {report['status']}",
            file=sys.stderr,
        )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
