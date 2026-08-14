#!/usr/bin/env python3
"""Align failure width jumps to same-seed successful rollout phases.

This stage proposes contexts for an expensive action-distribution probe. It does
not select training data by itself. Alignment uses only visual and proprioceptive
context; policy actions are intentionally excluded to avoid defining task phase
with the same signal later used to claim an action-distribution difference.

The event observation must still be supported by successful rollouts. Visual
divergence is evaluated only at later replans and must persist; a one-frame peak
is not evidence that the action block at the anchor changed task progress.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


MODALITIES = ("front", "wrist", "state")


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


def load_feature(path: Path) -> dict[str, Any]:
    with np.load(path) as payload:
        return {key: np.asarray(payload[key]) for key in payload.files}


def outcome_value(row: Mapping[str, Any]) -> str:
    return (
        "success"
        if bool(row.get("success")) or row.get("outcome") == "success"
        else "failure"
    )


def relative_pairs(length_a: int, length_b: int) -> tuple[np.ndarray, np.ndarray]:
    count = min(length_a, length_b)
    if count <= 0:
        return np.zeros(0, dtype=int), np.zeros(0, dtype=int)
    progress = np.linspace(0.0, 1.0, count)
    a = np.rint(progress * max(length_a - 1, 0)).astype(int)
    b = np.rint(progress * max(length_b - 1, 0)).astype(int)
    return a, b


def modality_distances(
    feature_a: Mapping[str, np.ndarray],
    feature_b: Mapping[str, np.ndarray],
    indices_a: np.ndarray,
    indices_b: np.ndarray,
) -> dict[str, np.ndarray]:
    front = 1.0 - np.sum(
        feature_a["front_visual"][indices_a].astype(np.float32)
        * feature_b["front_visual"][indices_b].astype(np.float32),
        axis=1,
    )
    wrist = 1.0 - np.sum(
        feature_a["wrist_visual"][indices_a].astype(np.float32)
        * feature_b["wrist_visual"][indices_b].astype(np.float32),
        axis=1,
    )
    state = np.sqrt(
        np.mean(
            (
                feature_a["states"][indices_a].astype(np.float32)
                - feature_b["states"][indices_b].astype(np.float32)
            )
            ** 2,
            axis=1,
        )
    )
    return {
        "front": np.maximum(front, 0.0),
        "wrist": np.maximum(wrist, 0.0),
        "state": state,
    }


def calibrate_modalities(
    features: Mapping[int, Mapping[str, np.ndarray]],
    outcomes: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Estimate success variation using only mixed-seed successful rollouts."""

    grouped: dict[int, dict[str, list[int]]] = defaultdict(
        lambda: {"success": [], "failure": []}
    )
    for row in outcomes:
        grouped[int(row["seed"])][outcome_value(row)].append(
            int(row["episode_index"])
        )
    by_seed = {
        seed: groups["success"]
        for seed, groups in grouped.items()
        if groups["success"] and groups["failure"]
    }

    pooled: dict[str, list[np.ndarray]] = {key: [] for key in MODALITIES}
    pair_count = 0
    for episode_indices in by_seed.values():
        ordered = sorted(episode_indices)
        for left_pos, left in enumerate(ordered):
            for right in ordered[left_pos + 1 :]:
                feature_left = features[left]
                feature_right = features[right]
                indices_left, indices_right = relative_pairs(
                    len(feature_left["frame_indices"]),
                    len(feature_right["frame_indices"]),
                )
                distances = modality_distances(
                    feature_left,
                    feature_right,
                    indices_left,
                    indices_right,
                )
                for key, values in distances.items():
                    pooled[key].append(values)
                pair_count += 1
    if pair_count == 0:
        raise ValueError("Need at least one same-seed pair of successful rollouts")

    scales: dict[str, float] = {}
    distributions: dict[str, dict[str, float]] = {}
    for key, chunks in pooled.items():
        values = np.concatenate(chunks).astype(np.float64)
        values = values[np.isfinite(values)]
        if values.size == 0:
            raise ValueError(f"No finite calibration values for {key}")
        scale = max(float(np.quantile(values, 0.75)), 1e-6)
        scales[key] = scale
        distributions[key] = {
            "p50": float(np.quantile(values, 0.50)),
            "p75": float(np.quantile(values, 0.75)),
            "p90": float(np.quantile(values, 0.90)),
            "p95": float(np.quantile(values, 0.95)),
            "count": int(values.size),
        }
    return {
        "seed_population": "mixed_only",
        "same_seed_success_pair_count": pair_count,
        "scales": scales,
        "raw_distance_distributions": distributions,
    }


def context_cost_matrix(
    failure: Mapping[str, np.ndarray],
    success: Mapping[str, np.ndarray],
    scales: Mapping[str, float],
    *,
    front_weight: float = 0.45,
    wrist_weight: float = 0.45,
    state_weight: float = 0.10,
) -> np.ndarray:
    failure_front = failure["front_visual"].astype(np.float32)
    success_front = success["front_visual"].astype(np.float32)
    failure_wrist = failure["wrist_visual"].astype(np.float32)
    success_wrist = success["wrist_visual"].astype(np.float32)
    front = np.maximum(1.0 - failure_front @ success_front.T, 0.0)
    wrist = np.maximum(1.0 - failure_wrist @ success_wrist.T, 0.0)
    state_delta = (
        failure["states"].astype(np.float32)[:, None, :]
        - success["states"].astype(np.float32)[None, :, :]
    )
    state = np.sqrt(np.mean(state_delta**2, axis=2))
    return (
        front_weight * np.minimum(front / float(scales["front"]), 8.0)
        + wrist_weight * np.minimum(wrist / float(scales["wrist"]), 8.0)
        + state_weight * np.minimum(state / float(scales["state"]), 8.0)
    ).astype(np.float32)


def dtw_prefix_endpoint(
    cost: np.ndarray,
    event_index: int,
    *,
    warp_fraction: float = 0.45,
    min_warp_blocks: int = 4,
    skip_penalty: float = 0.10,
    local_radius: int = 2,
) -> dict[str, Any]:
    """Map a failure event to one success prefix with anchored monotone DTW."""

    if cost.ndim != 2 or min(cost.shape) == 0:
        raise ValueError(f"Expected non-empty 2-D cost matrix, got {cost.shape}")
    if not 0 <= event_index < cost.shape[0]:
        raise IndexError(event_index)
    success_length = cost.shape[1]
    warp = max(min_warp_blocks, int(math.ceil(warp_fraction * (event_index + 1))))
    candidate_start = max(0, event_index - warp)
    candidate_end = min(success_length - 1, event_index + warp)
    if candidate_start > candidate_end:
        raise RuntimeError("Empty DTW endpoint range")

    prefix = cost[: event_index + 1]
    rows, cols = prefix.shape
    accumulated = np.full((rows, cols), np.inf, dtype=np.float64)
    path_length = np.zeros((rows, cols), dtype=np.int32)
    predecessor = np.full((rows, cols, 2), -1, dtype=np.int32)
    accumulated[0, 0] = float(prefix[0, 0])
    path_length[0, 0] = 1
    for row in range(rows):
        for col in range(cols):
            if row == 0 and col == 0:
                continue
            choices: list[tuple[float, int, int, int]] = []
            if row > 0 and col > 0 and np.isfinite(accumulated[row - 1, col - 1]):
                choices.append(
                    (accumulated[row - 1, col - 1], 0, row - 1, col - 1)
                )
            if row > 0 and np.isfinite(accumulated[row - 1, col]):
                choices.append(
                    (
                        accumulated[row - 1, col] + skip_penalty,
                        1,
                        row - 1,
                        col,
                    )
                )
            if col > 0 and np.isfinite(accumulated[row, col - 1]):
                choices.append(
                    (
                        accumulated[row, col - 1] + skip_penalty,
                        2,
                        row,
                        col - 1,
                    )
                )
            if not choices:
                continue
            _, _, previous_row, previous_col = min(
                choices, key=lambda item: (item[0], item[1])
            )
            accumulated[row, col] = (
                accumulated[previous_row, previous_col] + float(prefix[row, col])
            )
            path_length[row, col] = path_length[previous_row, previous_col] + 1
            predecessor[row, col] = (previous_row, previous_col)

    endpoints: list[tuple[float, int, float, float]] = []
    for success_index in range(candidate_start, candidate_end + 1):
        if not np.isfinite(accumulated[event_index, success_index]):
            continue
        radius = min(local_radius, event_index, success_index)
        local = float(
            np.mean(
                [
                    cost[event_index - offset, success_index - offset]
                    for offset in range(radius + 1)
                ]
            )
        )
        average = float(
            accumulated[event_index, success_index]
            / max(int(path_length[event_index, success_index]), 1)
        )
        time_penalty = 0.05 * abs(success_index - event_index) / max(
            event_index + 1, 1
        )
        objective = average + 0.50 * local + time_penalty
        endpoints.append((objective, success_index, average, local))
    if not endpoints:
        raise RuntimeError("No finite DTW endpoint")
    objective, success_index, average, local = min(
        endpoints, key=lambda item: (item[0], item[1])
    )

    path: list[list[int]] = []
    row, col = event_index, success_index
    while row >= 0 and col >= 0:
        path.append([int(row), int(col)])
        if row == 0 and col == 0:
            break
        next_row, next_col = predecessor[row, col]
        if next_row < 0 or next_col < 0:
            raise RuntimeError("Broken DTW predecessor chain")
        row, col = int(next_row), int(next_col)
    path.reverse()
    return {
        "success_index": int(success_index),
        "path_average_cost": average,
        "local_context_cost": local,
        "objective": float(objective),
        "path": path,
        "candidate_success_index_range": [candidate_start, candidate_end],
    }


def composite_support_distribution(
    features: Mapping[int, Mapping[str, np.ndarray]],
    outcomes: Sequence[Mapping[str, Any]],
    scales: Mapping[str, float],
) -> dict[str, float]:
    grouped: dict[int, dict[str, list[int]]] = defaultdict(
        lambda: {"success": [], "failure": []}
    )
    for row in outcomes:
        grouped[int(row["seed"])][outcome_value(row)].append(
            int(row["episode_index"])
        )
    by_seed = {
        seed: groups["success"]
        for seed, groups in grouped.items()
        if groups["success"] and groups["failure"]
    }
    values: list[np.ndarray] = []
    for episode_indices in by_seed.values():
        ordered = sorted(episode_indices)
        for left_pos, left in enumerate(ordered):
            for right in ordered[left_pos + 1 :]:
                left_feature = features[left]
                right_feature = features[right]
                left_indices, right_indices = relative_pairs(
                    len(left_feature["frame_indices"]),
                    len(right_feature["frame_indices"]),
                )
                matrix = context_cost_matrix(
                    left_feature, right_feature, scales
                )
                values.append(matrix[left_indices, right_indices])
    pooled = np.concatenate(values).astype(np.float64)
    return {
        "p50": float(np.quantile(pooled, 0.50)),
        "p75": float(np.quantile(pooled, 0.75)),
        "p90": float(np.quantile(pooled, 0.90)),
        "p95": float(np.quantile(pooled, 0.95)),
        "count": int(pooled.size),
    }


def persistent_future_divergence(
    future_costs: Mapping[str, float | None],
    *,
    current_cost: float,
    support_threshold: float,
    min_gain: float = 0.25,
    min_consecutive: int = 2,
) -> dict[str, Any]:
    """Require a run of later out-of-support contexts, not a transient peak."""

    if min_consecutive <= 0:
        raise ValueError("min_consecutive must be positive")
    ordered = sorted(
        ((int(horizon), value) for horizon, value in future_costs.items()),
        key=lambda item: item[0],
    )
    best_run: list[tuple[int, float]] = []
    current_run: list[tuple[int, float]] = []
    previous_horizon: int | None = None
    for horizon, value in ordered:
        qualifies = bool(
            value is not None
            and np.isfinite(float(value))
            and float(value) > support_threshold
            and float(value) - current_cost > min_gain
        )
        if not qualifies:
            current_run = []
            previous_horizon = None
            continue
        if previous_horizon is None or horizon == previous_horizon + 1:
            current_run.append((horizon, float(value)))
        else:
            current_run = [(horizon, float(value))]
        previous_horizon = horizon
        if len(current_run) > len(best_run):
            best_run = list(current_run)
    passed = len(best_run) >= min_consecutive
    return {
        "passed": passed,
        "min_consecutive": int(min_consecutive),
        "run_length": len(best_run),
        "onset_horizon": int(best_run[0][0]) if passed else None,
        "run_horizons": [int(item[0]) for item in best_run],
        "run_costs": [float(item[1]) for item in best_run],
    }


def event_frame(row: Mapping[str, Any]) -> int:
    value = row.get("event_center_frame")
    if value is None:
        value = (
            int(row["core_start_frame"]) + int(row["core_end_frame"])
        ) // 2
    return int(value)


def confidence_tier(success_count: int) -> str:
    if success_count >= 3:
        return "high_3s1f"
    if success_count == 2:
        return "medium_2s2f"
    if success_count == 1:
        return "low_1s3f"
    return "unusable_no_success"


def parse_int_set(raw: str) -> set[int] | None:
    values = {int(value.strip()) for value in raw.split(",") if value.strip()}
    return values or None


def build_candidates(
    outcomes: Sequence[Mapping[str, Any]],
    failure_events: Sequence[Mapping[str, Any]],
    features: Mapping[int, Mapping[str, np.ndarray]],
    *,
    min_event_block: int = 2,
    future_horizons: Sequence[int] = (1, 2, 3, 4),
    support_quantile: str = "p90",
    path_quantile: str = "p95",
    min_future_gain: float = 0.25,
    min_future_consecutive: int = 2,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    by_seed: dict[int, dict[str, list[int]]] = defaultdict(
        lambda: {"success": [], "failure": []}
    )
    for row in outcomes:
        episode_index = int(row["episode_index"])
        by_seed[int(row["seed"])][outcome_value(row)].append(episode_index)
    event_by_episode = {
        int(row["episode_index"]): row for row in failure_events
    }

    calibration = calibrate_modalities(features, outcomes)
    scales = calibration["scales"]
    support = composite_support_distribution(features, outcomes, scales)
    support_threshold = float(support[support_quantile])
    path_threshold = float(support[path_quantile])

    seed_rows: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    for seed, groups in sorted(by_seed.items()):
        successes = sorted(groups["success"])
        failures = sorted(groups["failure"])
        if not failures:
            seed_rows.append(
                {
                    "seed": seed,
                    "success_episodes": successes,
                    "failure_episodes": failures,
                    "classification": "all_success",
                    "use_for_training": False,
                    "reason": "no same-seed failure contrast",
                }
            )
            continue
        if not successes:
            seed_rows.append(
                {
                    "seed": seed,
                    "success_episodes": successes,
                    "failure_episodes": failures,
                    "classification": "all_failure",
                    "use_for_training": False,
                    "use_for_evaluation": True,
                    "reason": "no same-seed successful action target",
                }
            )
            continue

        seed_rows.append(
            {
                "seed": seed,
                "success_episodes": successes,
                "failure_episodes": failures,
                "classification": "mixed",
                "confidence_tier": confidence_tier(len(successes)),
                "use_for_candidate_discovery": True,
            }
        )
        for failure_episode in failures:
            failure_event = event_by_episode.get(failure_episode)
            rejection_reasons: list[str] = []
            if failure_event is None:
                rejection_reasons.append("missing_failure_width_jump")
                candidates.append(
                    {
                        "candidate_id": f"seed{seed}_fail{failure_episode:06d}_noevent",
                        "seed": seed,
                        "failure_episode_index": failure_episode,
                        "success_episode_indices": successes,
                        "status": "rejected_before_probe",
                        "rejection_reasons": rejection_reasons,
                    }
                )
                continue
            failure_feature = features[failure_episode]
            stride = int(np.asarray(failure_feature["stride"]).item())
            center_frame = event_frame(failure_event)
            failure_index = int(np.argmin(
                np.abs(
                    failure_feature["frame_indices"].astype(int) - center_frame
                )
            ))
            if failure_index < min_event_block:
                rejection_reasons.append("event_in_head_guard")

            alignments: list[dict[str, Any]] = []
            for success_episode in successes:
                success_feature = features[success_episode]
                cost = context_cost_matrix(
                    failure_feature, success_feature, scales
                )
                alignment = dtw_prefix_endpoint(cost, failure_index)
                success_index = int(alignment["success_index"])
                anchor_context_cost = float(cost[failure_index, success_index])
                future_costs: dict[str, float | None] = {}
                for horizon in future_horizons:
                    failure_future = failure_index + int(horizon)
                    success_future = success_index + int(horizon)
                    if (
                        failure_future < cost.shape[0]
                        and success_future < cost.shape[1]
                    ):
                        future_costs[str(horizon)] = float(
                            cost[failure_future, success_future]
                        )
                    else:
                        future_costs[str(horizon)] = None
                alignments.append(
                    {
                        "success_episode_index": success_episode,
                        "success_replan_index": success_index,
                        "success_frame": int(
                            success_feature["frame_indices"][success_index]
                        ),
                        "failure_replan_index": failure_index,
                        "failure_frame": int(
                            failure_feature["frame_indices"][failure_index]
                        ),
                        "anchor_context_cost": anchor_context_cost,
                        "local_context_cost": float(
                            alignment["local_context_cost"]
                        ),
                        "path_average_cost": float(
                            alignment["path_average_cost"]
                        ),
                        "future_context_costs": future_costs,
                        "dtw_endpoint_objective": float(alignment["objective"]),
                        "dtw_path_length": len(alignment["path"]),
                    }
                )

            current_costs = np.asarray(
                [row["anchor_context_cost"] for row in alignments],
                dtype=np.float64,
            )
            path_costs = np.asarray(
                [row["path_average_cost"] for row in alignments],
                dtype=np.float64,
            )
            current_median = float(np.median(current_costs))
            path_median = float(np.median(path_costs))
            if current_median > support_threshold:
                rejection_reasons.append("failure_context_outside_success_support")
            if path_median > path_threshold:
                rejection_reasons.append("poor_monotone_prefix_alignment")

            future_medians: dict[str, float | None] = {}
            for horizon in future_horizons:
                values = [
                    row["future_context_costs"][str(horizon)]
                    for row in alignments
                    if row["future_context_costs"][str(horizon)] is not None
                ]
                future_medians[str(horizon)] = (
                    float(np.median(np.asarray(values, dtype=np.float64)))
                    if values
                    else None
                )
            finite_future = [value for value in future_medians.values() if value is not None]
            if len(finite_future) < min_future_consecutive:
                rejection_reasons.append("insufficient_future_for_causal_order")
                maximum_future = None
                future_gain = None
                persistence = {
                    "passed": False,
                    "min_consecutive": int(min_future_consecutive),
                    "run_length": 0,
                    "onset_horizon": None,
                    "run_horizons": [],
                    "run_costs": [],
                }
            else:
                maximum_future = float(max(finite_future))
                future_gain = maximum_future - current_median
                persistence = persistent_future_divergence(
                    future_medians,
                    current_cost=current_median,
                    support_threshold=support_threshold,
                    min_gain=min_future_gain,
                    min_consecutive=min_future_consecutive,
                )
            future_diverged = bool(persistence["passed"])
            if not future_diverged and "insufficient_future_for_causal_order" not in rejection_reasons:
                rejection_reasons.append("future_not_persistently_diverged")

            context_probe_rejections = {
                "event_in_head_guard",
                "failure_context_outside_success_support",
                "poor_monotone_prefix_alignment",
                "insufficient_future_for_causal_order",
            }
            context_probe_eligible = not any(
                reason in context_probe_rejections for reason in rejection_reasons
            )
            observational_event_supported = bool(
                context_probe_eligible and future_diverged
            )
            candidate_id = (
                f"seed{seed}_fail{failure_episode:06d}_f"
                f"{int(failure_feature['frame_indices'][failure_index]):04d}"
            )
            candidates.append(
                {
                    "candidate_id": candidate_id,
                    "seed": seed,
                    "confidence_tier": confidence_tier(len(successes)),
                    "success_count": len(successes),
                    "failure_count": len(failures),
                    "failure_episode_index": failure_episode,
                    "failure_event_center_frame": center_frame,
                    "failure_replan_index": failure_index,
                    "failure_frame": int(
                        failure_feature["frame_indices"][failure_index]
                    ),
                    "stride": stride,
                    "width_jump": {
                        key: failure_event.get(key)
                        for key in (
                            "width",
                            "baseline_median",
                            "jump_ratio",
                            "jump_ratio_obs",
                        )
                        if failure_event.get(key) is not None
                    },
                    "success_alignments": alignments,
                    "alignment_summary": {
                        "current_context_cost_median": current_median,
                        "path_average_cost_median": path_median,
                        "support_threshold": support_threshold,
                        "path_threshold": path_threshold,
                        "future_context_cost_medians": future_medians,
                        "maximum_future_context_cost": maximum_future,
                        "future_context_cost_gain": future_gain,
                        "future_diverged_after_action": future_diverged,
                        "future_divergence_persistence": persistence,
                    },
                    "probe_eligible": context_probe_eligible,
                    "context_probe_eligible": context_probe_eligible,
                    "observational_event_supported": observational_event_supported,
                    "causal_intervention_status": "not_run",
                    "status": (
                        "awaiting_action_distribution_probe"
                        if context_probe_eligible
                        else "rejected_before_probe"
                    ),
                    "rejection_reasons": rejection_reasons,
                }
            )

    diagnostics = {
        "format": "FoldGlassesSeedPairWidthCandidates",
        "version": "1.0",
        "phase_alignment_uses_actions": False,
        "failure_width_jump_is_candidate_anchor": True,
        "calibration": calibration,
        "composite_same_seed_success_support": support,
        "support_quantile": support_quantile,
        "path_quantile": path_quantile,
        "future_horizons_replans": [int(value) for value in future_horizons],
        "min_future_gain": float(min_future_gain),
        "min_future_consecutive": int(min_future_consecutive),
        "num_seeds": len(by_seed),
        "num_candidates": len(candidates),
        "num_probe_eligible": sum(
            bool(row.get("probe_eligible")) for row in candidates
        ),
    }
    return candidates, seed_rows, diagnostics


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outcomes", type=Path, required=True)
    parser.add_argument("--failure-events", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-event-block", type=int, default=2)
    parser.add_argument(
        "--seeds",
        default="",
        help="Optional comma-separated seed subset for bounded audits.",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    output = args.output.expanduser().resolve()
    feature_root = args.features.expanduser().resolve()
    outcomes = read_jsonl(args.outcomes.expanduser().resolve())
    failure_events = read_jsonl(args.failure_events.expanduser().resolve())
    seed_filter = parse_int_set(str(args.seeds))
    if seed_filter is not None:
        outcomes = [
            row for row in outcomes if int(row["seed"]) in seed_filter
        ]
        failure_events = [
            row for row in failure_events if int(row["seed"]) in seed_filter
        ]
        present = {int(row["seed"]) for row in outcomes}
        missing = seed_filter - present
        if missing:
            raise ValueError(f"Unknown seeds: {sorted(missing)}")
    features: dict[int, dict[str, Any]] = {}
    for row in outcomes:
        episode_index = int(row["episode_index"])
        path = feature_root / "episodes" / f"ep{episode_index:06d}.npz"
        if not path.exists():
            raise FileNotFoundError(path)
        features[episode_index] = load_feature(path)
    candidates, seeds, diagnostics = build_candidates(
        outcomes,
        failure_events,
        features,
        min_event_block=int(args.min_event_block),
    )
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "candidate_events.jsonl", candidates)
    write_jsonl(output / "seed_audit.jsonl", seeds)
    (output / "diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output), **diagnostics}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
