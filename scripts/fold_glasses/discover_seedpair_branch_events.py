#!/usr/bin/env python3
"""Discover pre-divergence success/failure action branches in mixed seeds.

This scanner does not start from failure action-width jumps.  It aligns rollout
phase with action-free visual/proprioceptive features, then asks whether the
executed failure block is measurably separated from every observed successful
action block while the current observations are still supported by successful
rollouts.  This is screening, not an estimate of mode support.  Future
visual/proprioceptive divergence is evidence of a consequence, never a training
label by itself.

The output is observational.  ``training_eligible`` remains false until an
exact-prefix intervention produces a successful closed-loop counterfactual.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.fold_glasses import build_seedpair_width_candidates as context


DEFAULT_ACTION_STATS = Path(
    "/data_all/xiangchengzhan/FastWAM-infer-in-DexJoco/"
    "artifacts/fold_glasses/dataset_stats.json"
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(dict(row), sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_global_zscore(path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    stats = payload["action"]["default"]
    mean = np.asarray(stats["global_mean"], dtype=np.float32)
    std = np.asarray(stats["global_std"], dtype=np.float32)
    if mean.shape != (22,) or std.shape != (22,):
        raise ValueError(
            f"Expected 22-D global action statistics, got {mean.shape}, {std.shape}"
        )
    if not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise ValueError("Action statistics contain NaN or infinity")
    if np.any(std <= 0):
        raise ValueError("Action global_std must be positive")
    return mean, std


def normalize_actions(
    actions: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> np.ndarray:
    """Match the checkpoint's global z-score normalizer, including clipping."""

    values = np.asarray(actions, dtype=np.float32)
    if values.shape[-1] != 22:
        raise ValueError(f"Expected final action dimension 22, got {values.shape}")
    return np.clip((values - mean) / (std + 1e-8), -5.0, 5.0).astype(
        np.float32
    )


def action_block_rms(
    left: np.ndarray,
    right: np.ndarray,
    left_valid: np.ndarray,
    right_valid: np.ndarray,
) -> float:
    valid = np.asarray(left_valid, dtype=bool) & np.asarray(right_valid, dtype=bool)
    if not np.any(valid):
        return float("nan")
    delta = np.asarray(left, dtype=np.float32)[valid] - np.asarray(
        right, dtype=np.float32
    )[valid]
    return float(np.sqrt(np.mean(delta.astype(np.float64) ** 2)))


def action_step_rms(
    left: np.ndarray,
    right: np.ndarray,
    left_valid: np.ndarray,
    right_valid: np.ndarray,
) -> np.ndarray:
    valid = np.asarray(left_valid, dtype=bool) & np.asarray(right_valid, dtype=bool)
    result = np.full(len(valid), np.nan, dtype=np.float32)
    if np.any(valid):
        delta = np.asarray(left, dtype=np.float32)[valid] - np.asarray(
            right, dtype=np.float32
        )[valid]
        result[valid] = np.sqrt(np.mean(delta.astype(np.float64) ** 2, axis=1))
    return result


def max_true_run(mask: np.ndarray) -> int:
    best = 0
    current_run = 0
    for value in np.asarray(mask, dtype=bool).tolist():
        if value:
            current_run += 1
            best = max(best, current_run)
        else:
            current_run = 0
    return best


def monotone_endpoint_map(
    cost: np.ndarray,
    *,
    warp_fraction: float = 0.45,
    min_warp_blocks: int = 4,
    skip_penalty: float = 0.10,
    local_radius: int = 2,
) -> np.ndarray:
    """Build an action-free, monotone online alignment for every left prefix."""

    cost = np.asarray(cost, dtype=np.float64)
    if cost.ndim != 2 or min(cost.shape) == 0:
        raise ValueError(f"Expected non-empty 2-D cost matrix, got {cost.shape}")
    rows, cols = cost.shape
    accumulated = np.full((rows, cols), np.inf, dtype=np.float64)
    lengths = np.zeros((rows, cols), dtype=np.int32)
    accumulated[0, 0] = cost[0, 0]
    lengths[0, 0] = 1
    for row in range(rows):
        for col in range(cols):
            if row == 0 and col == 0:
                continue
            choices: list[tuple[float, int, int]] = []
            if row > 0 and col > 0 and np.isfinite(accumulated[row - 1, col - 1]):
                choices.append((accumulated[row - 1, col - 1], row - 1, col - 1))
            if row > 0 and np.isfinite(accumulated[row - 1, col]):
                choices.append(
                    (accumulated[row - 1, col] + skip_penalty, row - 1, col)
                )
            if col > 0 and np.isfinite(accumulated[row, col - 1]):
                choices.append(
                    (accumulated[row, col - 1] + skip_penalty, row, col - 1)
                )
            if not choices:
                continue
            _, previous_row, previous_col = min(
                choices, key=lambda item: (item[0], item[1], item[2])
            )
            accumulated[row, col] = accumulated[previous_row, previous_col] + cost[
                row, col
            ]
            lengths[row, col] = lengths[previous_row, previous_col] + 1

    mapping = np.zeros(rows, dtype=np.int32)
    previous_col = 0
    for row in range(rows):
        warp = max(min_warp_blocks, int(math.ceil(warp_fraction * (row + 1))))
        start = max(previous_col, row - warp, 0)
        end = min(cols - 1, row + warp)
        if start > end:
            start = min(previous_col, cols - 1)
            end = cols - 1
        candidates: list[tuple[float, int]] = []
        for col in range(start, end + 1):
            if not np.isfinite(accumulated[row, col]):
                continue
            radius = min(local_radius, row, col)
            local = float(
                np.mean(
                    [cost[row - offset, col - offset] for offset in range(radius + 1)]
                )
            )
            average = float(accumulated[row, col] / max(int(lengths[row, col]), 1))
            time_penalty = 0.05 * abs(col - row) / max(row + 1, 1)
            candidates.append((average + 0.5 * local + time_penalty, col))
        if not candidates:
            mapping[row] = previous_col
            continue
        _, selected = min(candidates, key=lambda item: (item[0], item[1]))
        mapping[row] = selected
        previous_col = selected
    if np.any(np.diff(mapping) < 0):
        raise RuntimeError("Internal error: endpoint map is not monotone")
    return mapping


def distribution(values: Sequence[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        raise ValueError("Cannot summarize an empty distribution")
    return {
        "p50": float(np.quantile(array, 0.50)),
        "p75": float(np.quantile(array, 0.75)),
        "p90": float(np.quantile(array, 0.90)),
        "p95": float(np.quantile(array, 0.95)),
        "count": int(array.size),
    }


def group_outcomes(
    outcomes: Sequence[Mapping[str, Any]],
) -> dict[int, dict[str, list[int]]]:
    grouped: dict[int, dict[str, list[int]]] = defaultdict(
        lambda: {"success": [], "failure": []}
    )
    for row in outcomes:
        grouped[int(row["seed"])][context.outcome_value(row)].append(
            int(row["episode_index"])
        )
    return grouped


def parse_int_set(raw: str) -> set[int] | None:
    values = {int(value.strip()) for value in raw.split(",") if value.strip()}
    return values or None


def prepare_actions(
    features: Mapping[int, Mapping[str, np.ndarray]],
    mean: np.ndarray,
    std: np.ndarray,
) -> dict[int, np.ndarray]:
    return {
        episode: normalize_actions(feature["action_blocks"], mean, std)
        for episode, feature in features.items()
    }


def calibrate_success_action_variation(
    features: Mapping[int, Mapping[str, np.ndarray]],
    normalized: Mapping[int, np.ndarray],
    outcomes: Sequence[Mapping[str, Any]],
    scales: Mapping[str, float],
    *,
    context_support_threshold: float,
) -> dict[str, Any]:
    """Calibrate harmless action variation using mixed-seed successes only."""

    grouped = group_outcomes(outcomes)
    block_values: list[float] = []
    step_values: list[float] = []
    pair_count = 0
    supported_points = 0
    for groups in grouped.values():
        if not groups["success"] or not groups["failure"]:
            continue
        successes = sorted(groups["success"])
        for left_position, left_episode in enumerate(successes):
            for right_episode in successes[left_position + 1 :]:
                left = features[left_episode]
                right = features[right_episode]
                cost = context.context_cost_matrix(left, right, scales)
                mapping = monotone_endpoint_map(cost)
                pair_count += 1
                for left_index, right_index in enumerate(mapping.tolist()):
                    if float(cost[left_index, right_index]) > context_support_threshold:
                        continue
                    block_distance = action_block_rms(
                        normalized[left_episode][left_index],
                        normalized[right_episode][right_index],
                        left["action_valid"][left_index],
                        right["action_valid"][right_index],
                    )
                    if np.isfinite(block_distance):
                        block_values.append(block_distance)
                    steps = action_step_rms(
                        normalized[left_episode][left_index],
                        normalized[right_episode][right_index],
                        left["action_valid"][left_index],
                        right["action_valid"][right_index],
                    )
                    step_values.extend(steps[np.isfinite(steps)].astype(float).tolist())
                    supported_points += 1
    if not block_values or not step_values:
        raise ValueError("No supported mixed-seed success action pairs for calibration")
    return {
        "seed_population": "mixed_seed_successes_only",
        "same_seed_success_pair_count": pair_count,
        "supported_aligned_block_count": supported_points,
        "block_rms": distribution(block_values),
        "step_rms": distribution(step_values),
    }


def future_context_evidence(
    cost: np.ndarray, left_index: int, right_index: int, horizon: int
) -> dict[str, Any] | None:
    """Return fixed-lag evidence, or an explicit successful-terminal proxy."""

    left_future = left_index + horizon
    requested_right = right_index + horizon
    if left_future >= cost.shape[0]:
        return None
    right_future = min(requested_right, cost.shape[1] - 1)
    terminal_proxy = requested_right >= cost.shape[1]
    return {
        "cost": float(cost[left_future, right_future]),
        "failure_future_replan_index": int(left_future),
        "success_future_replan_index": int(right_future),
        "requested_success_future_replan_index": int(requested_right),
        "success_terminal_proxy": bool(terminal_proxy),
        "terminal_proxy_definition": (
            "last_available_replan_from_successful_episode"
            if terminal_proxy
            else None
        ),
    }


def score_failure_blocks(
    *,
    seed: int,
    failure_episode: int,
    success_episodes: Sequence[int],
    features: Mapping[int, Mapping[str, np.ndarray]],
    normalized: Mapping[int, np.ndarray],
    scales: Mapping[str, float],
    context_support_threshold: float,
    path_support_threshold: float,
    min_action_block_rms: float,
    min_action_step_rms: float,
    head_guard_blocks: int,
    min_action_run_steps: int,
    future_horizons: Sequence[int],
    min_future_gain: float,
    min_future_consecutive: int,
) -> list[dict[str, Any]]:
    failure = features[failure_episode]
    aligned: dict[int, dict[str, Any]] = {}
    for success_episode in success_episodes:
        success = features[success_episode]
        cost = context.context_cost_matrix(failure, success, scales)
        aligned[success_episode] = {
            "feature": success,
            "cost": cost,
            "mapping": monotone_endpoint_map(cost),
        }

    rows: list[dict[str, Any]] = []
    for failure_index in range(len(failure["frame_indices"])):
        reasons: list[str] = []
        if failure_index < head_guard_blocks:
            reasons.append("head_guard")
        if int(np.sum(failure["action_valid"][failure_index])) < min_action_run_steps:
            reasons.append("insufficient_failure_action_steps")

        supported: list[dict[str, Any]] = []
        all_alignments: list[dict[str, Any]] = []
        for success_episode in success_episodes:
            item = aligned[success_episode]
            success_index = int(item["mapping"][failure_index])
            current_cost = float(item["cost"][failure_index, success_index])
            prefix_start = max(0, failure_index - 2)
            prefix_costs = [
                float(item["cost"][index, int(item["mapping"][index])])
                for index in range(prefix_start, failure_index + 1)
            ]
            prefix_cost = float(np.mean(prefix_costs))
            record = {
                "success_episode_index": int(success_episode),
                "success_replan_index": success_index,
                "success_frame": int(
                    item["feature"]["frame_indices"][success_index]
                ),
                "current_context_cost": current_cost,
                "recent_prefix_context_cost": prefix_cost,
            }
            all_alignments.append(record)
            if (
                current_cost <= context_support_threshold
                and prefix_cost <= path_support_threshold
                and int(np.sum(item["feature"]["action_valid"][success_index]))
                >= min_action_run_steps
            ):
                supported.append(record)
        if not supported:
            reasons.append("no_success_context_support")

        block_distances: list[float] = []
        step_distance_modes: list[np.ndarray] = []
        for record in supported:
            success_episode = int(record["success_episode_index"])
            success_index = int(record["success_replan_index"])
            success = features[success_episode]
            block_distance = action_block_rms(
                normalized[failure_episode][failure_index],
                normalized[success_episode][success_index],
                failure["action_valid"][failure_index],
                success["action_valid"][success_index],
            )
            record["executed_action_block_rms"] = block_distance
            block_distances.append(block_distance)
            step_distance_modes.append(
                action_step_rms(
                    normalized[failure_episode][failure_index],
                    normalized[success_episode][success_index],
                    failure["action_valid"][failure_index],
                    success["action_valid"][success_index],
                )
            )

        nearest_block_distance = (
            float(np.nanmin(np.asarray(block_distances, dtype=np.float64)))
            if block_distances
            else None
        )
        if step_distance_modes:
            mode_steps = np.stack(step_distance_modes, axis=0)
            finite = np.isfinite(mode_steps)
            step_min = np.min(np.where(finite, mode_steps, np.inf), axis=0)
            step_min[~np.any(finite, axis=0)] = np.nan
            step_above = np.isfinite(step_min) & (step_min > min_action_step_rms)
            action_run_steps = max_true_run(step_above)
            step_above_fraction = float(np.mean(step_above[np.isfinite(step_min)]))
        else:
            step_min = np.zeros(0, dtype=np.float32)
            action_run_steps = 0
            step_above_fraction = 0.0

        if nearest_block_distance is None or nearest_block_distance <= min_action_block_rms:
            reasons.append("failure_action_not_separated_from_observed_success_blocks")
        if action_run_steps < min_action_run_steps:
            reasons.append("action_separation_not_persistent_within_block")

        current_costs = [float(row["current_context_cost"]) for row in supported]
        current_context_median = (
            float(np.median(np.asarray(current_costs, dtype=np.float64)))
            if current_costs
            else None
        )
        future_medians: dict[str, float | None] = {}
        future_source_counts: dict[str, int] = {}
        future_terminal_source_counts: dict[str, int] = {}
        future_sources: dict[str, list[dict[str, Any]]] = {}
        for horizon in future_horizons:
            values: list[float] = []
            sources: list[dict[str, Any]] = []
            for record in supported:
                success_episode = int(record["success_episode_index"])
                evidence = future_context_evidence(
                    aligned[success_episode]["cost"],
                    failure_index,
                    int(record["success_replan_index"]),
                    int(horizon),
                )
                if evidence is not None:
                    values.append(float(evidence["cost"]))
                    sources.append(
                        {
                            "success_episode_index": success_episode,
                            **evidence,
                        }
                    )
            horizon_key = str(int(horizon))
            future_medians[horizon_key] = (
                float(np.median(np.asarray(values, dtype=np.float64)))
                if values
                else None
            )
            future_source_counts[horizon_key] = len(sources)
            future_terminal_source_counts[horizon_key] = sum(
                bool(source["success_terminal_proxy"]) for source in sources
            )
            future_sources[horizon_key] = sources
        if current_context_median is None:
            future_persistence = {
                "passed": False,
                "min_consecutive": int(min_future_consecutive),
                "run_length": 0,
                "onset_horizon": None,
                "run_horizons": [],
                "run_costs": [],
            }
        else:
            future_persistence = context.persistent_future_divergence(
                future_medians,
                current_cost=current_context_median,
                support_threshold=context_support_threshold,
                min_gain=min_future_gain,
                min_consecutive=min_future_consecutive,
            )
        if not bool(future_persistence["passed"]):
            reasons.append("future_context_not_persistently_diverged")

        observational = not reasons
        future_gain = None
        finite_future = [
            float(value) for value in future_medians.values() if value is not None
        ]
        if current_context_median is not None and finite_future:
            future_gain = float(max(finite_future) - current_context_median)
        action_margin = (
            float(nearest_block_distance / max(min_action_block_rms, 1e-8))
            if nearest_block_distance is not None
            else None
        )
        score = None
        if observational:
            onset = int(future_persistence["onset_horizon"])
            action_evidence = min(float(action_margin), 4.0) / 4.0
            context_evidence = max(
                0.0,
                1.0
                - float(current_context_median)
                / max(context_support_threshold, 1e-8),
            )
            future_evidence = min(
                float(future_gain or 0.0) / max(min_future_gain, 1e-8), 4.0
            ) / 4.0
            support_evidence = len(supported) / max(len(success_episodes), 1)
            causal_lead_evidence = min(onset / 3.0, 1.0)
            score = float(
                action_evidence
                + context_evidence
                + future_evidence
                + support_evidence
                + causal_lead_evidence
            )
        failure_frame = int(failure["frame_indices"][failure_index])
        rows.append(
            {
                "candidate_id": (
                    f"seed{seed}_fail{failure_episode:06d}_f{failure_frame:04d}"
                ),
                "seed": int(seed),
                "failure_episode_index": int(failure_episode),
                "failure_replan_index": int(failure_index),
                "failure_frame": failure_frame,
                "stride": int(np.asarray(failure["stride"]).item()),
                "success_count": len(success_episodes),
                "confidence_tier": context.confidence_tier(len(success_episodes)),
                "all_success_alignments": all_alignments,
                "success_alignments": supported,
                "shared_context": {
                    "supported_success_count": len(supported),
                    "current_context_cost_median": current_context_median,
                    "support_threshold": float(context_support_threshold),
                },
                "executed_action_branch": {
                    "nearest_success_mode_block_rms": nearest_block_distance,
                    "block_screening_floor": float(min_action_block_rms),
                    "block_margin_ratio": action_margin,
                    "screening_evidence_saturates_at_margin_ratio": 4.0,
                    "nearest_success_mode_step_rms": step_min.tolist(),
                    "step_screening_floor": float(min_action_step_rms),
                    "step_fraction_above_threshold": step_above_fraction,
                    "max_consecutive_steps_above_threshold": action_run_steps,
                    "required_consecutive_steps": int(min_action_run_steps),
                },
                "future_context": {
                    "fixed_lag_cost_medians": future_medians,
                    "evidence_source_counts": future_source_counts,
                    "terminal_proxy_source_counts": future_terminal_source_counts,
                    "sources": future_sources,
                    "maximum_gain_from_anchor": future_gain,
                    "persistence": future_persistence,
                },
                "observational_event_supported": observational,
                "probe_eligible": observational,
                "training_eligible": False,
                "causal_intervention_status": "not_run",
                "selection_score": score,
                "selection_score_is_bounded_observational_priority_only": True,
                "status": (
                    "awaiting_exact_context_policy_probe"
                    if observational
                    else "rejected_observationally"
                ),
                "rejection_reasons": reasons,
            }
        )
    return rows


def select_nonredundant_events(
    rows: Sequence[Mapping[str, Any]], *, max_events_per_failure: int
) -> list[dict[str, Any]]:
    """Keep the earliest block in each contiguous qualifying interval."""

    eligible = [dict(row) for row in rows if row["observational_event_supported"]]
    eligible.sort(key=lambda row: int(row["failure_replan_index"]))
    runs: list[list[dict[str, Any]]] = []
    for row in eligible:
        if (
            not runs
            or int(row["failure_replan_index"])
            > int(runs[-1][-1]["failure_replan_index"]) + 1
        ):
            runs.append([row])
        else:
            runs[-1].append(row)
    selected: list[dict[str, Any]] = []
    for event_index, run in enumerate(runs[:max_events_per_failure], start=1):
        earliest = dict(run[0])
        earliest["event_index_within_failure"] = event_index
        earliest["qualifying_interval_replan_indices"] = [
            int(row["failure_replan_index"]) for row in run
        ]
        earliest["selection_policy"] = "earliest_block_of_contiguous_interval"
        selected.append(earliest)
    return selected


def discover_events(
    outcomes: Sequence[Mapping[str, Any]],
    features: Mapping[int, Mapping[str, np.ndarray]],
    *,
    action_mean: np.ndarray,
    action_std: np.ndarray,
    context_support_quantile: str = "p90",
    path_support_quantile: str = "p95",
    min_action_block_rms: float = 0.05,
    min_action_step_rms: float = 0.05,
    head_guard_blocks: int = 2,
    min_action_run_steps: int = 8,
    future_horizons: Sequence[int] = (1, 2, 3, 4, 5, 6),
    min_future_gain: float = 0.25,
    min_future_consecutive: int = 2,
    max_events_per_failure: int = 2,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    calibration = context.calibrate_modalities(features, outcomes)
    scales = calibration["scales"]
    context_distribution = context.composite_support_distribution(
        features, outcomes, scales
    )
    context_threshold = float(context_distribution[context_support_quantile])
    path_threshold = float(context_distribution[path_support_quantile])
    normalized = prepare_actions(features, action_mean, action_std)
    action_calibration = calibrate_success_action_variation(
        features,
        normalized,
        outcomes,
        scales,
        context_support_threshold=context_threshold,
    )
    if min_action_block_rms <= 0 or min_action_step_rms <= 0:
        raise ValueError("Action screening floors must be positive")

    grouped = group_outcomes(outcomes)
    scored: list[dict[str, Any]] = []
    selected: list[dict[str, Any]] = []
    seed_audit: list[dict[str, Any]] = []
    for seed, groups in sorted(grouped.items()):
        successes = sorted(groups["success"])
        failures = sorted(groups["failure"])
        if not failures:
            classification = "all_success"
        elif not successes:
            classification = "all_failure"
        else:
            classification = "mixed"
        seed_audit.append(
            {
                "seed": int(seed),
                "classification": classification,
                "success_episodes": successes,
                "failure_episodes": failures,
                "use_for_discovery": classification == "mixed",
                "use_for_training": classification == "mixed",
                "use_for_evaluation": classification in {"mixed", "all_failure"},
            }
        )
        if classification != "mixed":
            continue
        for failure_episode in failures:
            failure_rows = score_failure_blocks(
                seed=seed,
                failure_episode=failure_episode,
                success_episodes=successes,
                features=features,
                normalized=normalized,
                scales=scales,
                context_support_threshold=context_threshold,
                path_support_threshold=path_threshold,
                min_action_block_rms=min_action_block_rms,
                min_action_step_rms=min_action_step_rms,
                head_guard_blocks=head_guard_blocks,
                min_action_run_steps=min_action_run_steps,
                future_horizons=future_horizons,
                min_future_gain=min_future_gain,
                min_future_consecutive=min_future_consecutive,
            )
            scored.extend(failure_rows)
            selected.extend(
                select_nonredundant_events(
                    failure_rows, max_events_per_failure=max_events_per_failure
                )
            )
    selected.sort(
        key=lambda row: (
            -float(row["selection_score"]),
            int(row["seed"]),
            int(row["failure_episode_index"]),
            int(row["failure_frame"]),
        )
    )
    for rank, row in enumerate(selected, start=1):
        row["global_observational_rank"] = rank

    diagnostics = {
        "format": "FoldGlassesSeedPairBranchEvents",
        "version": "1.0",
        "phase_alignment_uses_actions": False,
        "failure_width_jump_used_for_discovery": False,
        "policy_width_is_downstream_diagnostic": True,
        "comparison_action_space": "checkpoint_global_zscore_clipped_-5_5",
        "context_calibration": calibration,
        "context_support_distribution": context_distribution,
        "context_support_quantile": context_support_quantile,
        "path_support_quantile": path_support_quantile,
        "action_calibration": action_calibration,
        "action_threshold_source": (
            "fixed_low_screening_floor; per-mode support requires policy probe"
        ),
        "min_action_block_rms": float(min_action_block_rms),
        "min_action_step_rms": float(min_action_step_rms),
        "head_guard_blocks": int(head_guard_blocks),
        "min_action_run_steps": int(min_action_run_steps),
        "future_horizons_replans": [int(value) for value in future_horizons],
        "min_future_gain": float(min_future_gain),
        "min_future_consecutive": int(min_future_consecutive),
        "max_events_per_failure": int(max_events_per_failure),
        "num_seeds": len(grouped),
        "num_mixed_seeds": sum(
            bool(row["classification"] == "mixed") for row in seed_audit
        ),
        "num_scored_blocks": len(scored),
        "num_observational_events": len(selected),
        "training_eligible_events": 0,
    }
    return selected, scored, seed_audit, diagnostics


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outcomes", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--action-stats", type=Path, default=DEFAULT_ACTION_STATS)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--seeds",
        default="",
        help="Optional comma-separated seed subset for bounded audits.",
    )
    parser.add_argument("--head-guard-blocks", type=int, default=2)
    parser.add_argument("--min-action-run-steps", type=int, default=8)
    parser.add_argument("--min-action-block-rms", type=float, default=0.05)
    parser.add_argument("--min-action-step-rms", type=float, default=0.05)
    parser.add_argument("--max-events-per-failure", type=int, default=2)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    outcomes_path = args.outcomes.expanduser().resolve()
    feature_root = args.features.expanduser().resolve()
    stats_path = args.action_stats.expanduser().resolve()
    output = args.output.expanduser().resolve()
    outcomes = read_jsonl(outcomes_path)
    seed_filter = parse_int_set(str(args.seeds))
    if seed_filter is not None:
        all_present = {int(row["seed"]) for row in outcomes}
        missing = seed_filter - all_present
        if missing:
            raise ValueError(f"Unknown seeds: {sorted(missing)}")
        outcomes = [
            row for row in outcomes if int(row["seed"]) in seed_filter
        ]
    features: dict[int, dict[str, np.ndarray]] = {}
    for row in outcomes:
        episode = int(row["episode_index"])
        path = feature_root / "episodes" / f"ep{episode:06d}.npz"
        if not path.exists():
            raise FileNotFoundError(path)
        features[episode] = context.load_feature(path)
    mean, std = load_global_zscore(stats_path)
    selected, scored, seed_audit, diagnostics = discover_events(
        outcomes,
        features,
        action_mean=mean,
        action_std=std,
        head_guard_blocks=int(args.head_guard_blocks),
        min_action_run_steps=int(args.min_action_run_steps),
        min_action_block_rms=float(args.min_action_block_rms),
        min_action_step_rms=float(args.min_action_step_rms),
        max_events_per_failure=int(args.max_events_per_failure),
    )
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "candidate_events.jsonl", selected)
    write_jsonl(output / "all_scored_blocks.jsonl", scored)
    write_jsonl(output / "seed_audit.jsonl", seed_audit)
    diagnostics.update(
        {
            "outcomes": str(outcomes_path),
            "features": str(feature_root),
            "action_stats": str(stats_path),
            "action_stats_sha256": sha256_file(stats_path),
            "output": str(output),
        }
    )
    (output / "diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(diagnostics, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
