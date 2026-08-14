#!/usr/bin/env python3
"""Score successful action support at aligned failure contexts.

The score answers a narrower question than scalar action width: does the policy
at a shared failure observation put measurable probability near any actually
successful action block? Successful blocks may form several valid modes, so the
script keeps one support ball per successful rollout instead of forcing them to
have a low aggregate variance.

All geometry is computed in the checkpoint's training-normalized action space.
These observational scores do not declare an action block causally useful;
`training_eligible` remains false until an executable intervention passes.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


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


def load_probe(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as payload:
        return {key: np.asarray(payload[key]) for key in payload.files}


def block_rms(
    left: np.ndarray,
    right: np.ndarray,
    *,
    valid: np.ndarray | None = None,
    prefix_steps: int | None = None,
) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError(f"Expected equal [T,D] blocks, got {left.shape}, {right.shape}")
    limit = left.shape[0] if prefix_steps is None else min(int(prefix_steps), left.shape[0])
    mask = np.ones(limit, dtype=bool) if valid is None else np.asarray(valid, dtype=bool)[:limit]
    if not mask.any():
        raise ValueError("Action block has no valid steps")
    delta = left[:limit][mask] - right[:limit][mask]
    return float(np.sqrt(np.mean(delta**2)))


def wilson_interval(successes: int, trials: int, *, z: float = 1.959963984540054) -> list[float]:
    if trials <= 0 or not 0 <= successes <= trials:
        raise ValueError(f"Invalid binomial counts {successes}/{trials}")
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    center = (proportion + z * z / (2.0 * trials)) / denominator
    half = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / trials
            + z * z / (4.0 * trials * trials)
        )
        / denominator
    )
    return [float(max(0.0, center - half)), float(min(1.0, center + half))]


def empirical_radius(
    distances: Sequence[float],
    *,
    quantile: float = 0.90,
    multiplier: float = 1.5,
    minimum: float = 1e-6,
) -> float:
    values = np.asarray(distances, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("Cannot estimate support radius without distances")
    if not 0.0 < quantile <= 1.0 or multiplier <= 0.0:
        raise ValueError("Invalid support radius parameters")
    return max(float(np.quantile(values, quantile)) * multiplier, minimum)


def distance_summary(values: Sequence[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"count": 0}
    return {
        "count": int(array.size),
        "min": float(array.min()),
        "p50": float(np.quantile(array, 0.50)),
        "p90": float(np.quantile(array, 0.90)),
        "max": float(array.max()),
    }


def probe_path(probe_root: Path, episode_index: int, frame: int) -> Path:
    return probe_root / "contexts" / f"ep{episode_index:06d}_f{frame:04d}.npz"


def score_candidate(
    candidate: Mapping[str, Any],
    probe_root: Path,
    *,
    support_quantile: float = 0.90,
    support_radius_multiplier: float = 1.5,
) -> dict[str, Any]:
    candidate_id = str(candidate["candidate_id"])
    failure_episode = int(candidate["failure_episode_index"])
    failure_frame = int(candidate["failure_frame"])
    failure = load_probe(probe_path(probe_root, failure_episode, failure_frame))
    failure_samples = np.asarray(
        failure["action_samples_normalized"], dtype=np.float32
    )
    failure_actual = np.asarray(
        failure["actual_action_block_normalized"], dtype=np.float32
    )
    failure_valid = np.asarray(failure["actual_action_valid"], dtype=bool)

    modes: list[dict[str, Any]] = []
    for alignment in candidate.get("success_alignments", []):
        episode = int(alignment["success_episode_index"])
        frame = int(alignment["success_frame"])
        success = load_probe(probe_path(probe_root, episode, frame))
        actual = np.asarray(
            success["actual_action_block_normalized"], dtype=np.float32
        )
        valid = np.asarray(success["actual_action_valid"], dtype=bool)
        joint_valid = failure_valid & valid
        own_distances = [
            block_rms(sample, actual, valid=valid)
            for sample in np.asarray(
                success["action_samples_normalized"], dtype=np.float32
            )
        ]
        radius = empirical_radius(
            own_distances,
            quantile=support_quantile,
            multiplier=support_radius_multiplier,
        )
        failure_sample_distances = [
            block_rms(sample, actual, valid=joint_valid)
            for sample in failure_samples
        ]
        hits = [distance <= radius for distance in failure_sample_distances]
        modes.append(
            {
                "success_episode_index": episode,
                "success_frame": frame,
                "support_radius": radius,
                "support_radius_source": "own_context_policy_to_executed_success",
                "own_context_policy_distance": distance_summary(own_distances),
                "failure_actual_distance": block_rms(
                    failure_actual, actual, valid=joint_valid
                ),
                "failure_policy_distance": distance_summary(
                    failure_sample_distances
                ),
                "failure_policy_support_hits": int(sum(hits)),
                "failure_policy_support_mass": float(np.mean(hits)),
                "failure_policy_support_mass_wilson95": wilson_interval(
                    int(sum(hits)), len(hits)
                ),
            }
        )

    if not modes:
        raise ValueError(f"Candidate {candidate_id} has no successful alignment")

    success_actuals = []
    for mode in modes:
        payload = load_probe(
            probe_path(
                probe_root,
                int(mode["success_episode_index"]),
                int(mode["success_frame"]),
            )
        )
        success_actuals.append(
            np.asarray(payload["actual_action_block_normalized"], dtype=np.float32)
        )
    pairwise = [
        block_rms(success_actuals[left], success_actuals[right])
        for left in range(len(success_actuals))
        for right in range(left + 1, len(success_actuals))
    ]

    sample_any_hits: list[bool] = []
    sample_nearest_distances: list[float] = []
    sample_nearest_radius_ratios: list[float] = []
    for sample in failure_samples:
        distances = [
            block_rms(sample, actual, valid=failure_valid)
            for actual in success_actuals
        ]
        ratios = [
            distance / float(mode["support_radius"])
            for distance, mode in zip(distances, modes)
        ]
        sample_any_hits.append(any(ratio <= 1.0 for ratio in ratios))
        sample_nearest_distances.append(min(distances))
        sample_nearest_radius_ratios.append(min(ratios))

    failure_self_distances = [
        block_rms(sample, failure_actual, valid=failure_valid)
        for sample in failure_samples
    ]
    any_hits = int(sum(sample_any_hits))
    trials = len(sample_any_hits)
    empirical_mass = any_hits / trials
    branching_observed = bool(any_hits > 0 and any(not hit for hit in sample_any_hits))
    return {
        "candidate_id": candidate_id,
        "seed": int(candidate["seed"]),
        "failure_episode_index": failure_episode,
        "failure_frame": failure_frame,
        "evidence_level": "observational_action_support",
        "comparison_action_space": "checkpoint_training_normalized",
        "num_policy_samples": trials,
        "success_modes": modes,
        "success_mode_pairwise_actual_distance": distance_summary(pairwise),
        "failure_policy_to_factual_distance": distance_summary(
            failure_self_distances
        ),
        "failure_policy_to_nearest_success_distance": distance_summary(
            sample_nearest_distances
        ),
        "failure_policy_nearest_success_radius_ratio": distance_summary(
            sample_nearest_radius_ratios
        ),
        "failure_policy_any_success_support_hits": any_hits,
        "failure_policy_any_success_support_mass": float(empirical_mass),
        "failure_policy_any_success_support_mass_wilson95": wilson_interval(
            any_hits, trials
        ),
        "sampled_distribution_contains_success_and_failure_branches": branching_observed,
        "passive_future_divergence_supported": bool(
            candidate.get("observational_event_supported", False)
        ),
        "causal_intervention_status": "not_run",
        "training_eligible": False,
        "training_blocker": "requires_successful_exact_prefix_intervention",
    }


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--probe-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--support-quantile", type=float, default=0.90)
    parser.add_argument("--support-radius-multiplier", type=float, default=1.5)
    parser.add_argument("--include-rejected", action="store_true")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    candidates = read_jsonl(args.candidates.expanduser().resolve())
    selected = [
        row
        for row in candidates
        if args.include_rejected or bool(row.get("probe_eligible"))
    ]
    if not selected:
        raise ValueError("No candidates are eligible for action-support scoring")
    probe_root = args.probe_root.expanduser().resolve()
    rows = [
        score_candidate(
            candidate,
            probe_root,
            support_quantile=float(args.support_quantile),
            support_radius_multiplier=float(args.support_radius_multiplier),
        )
        for candidate in selected
    ]
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "action_support_scores.jsonl", rows)
    summary = {
        "format": "FoldGlassesSeedPairActionSupportScores",
        "version": "1.0",
        "comparison_action_space": "checkpoint_training_normalized",
        "support_quantile": float(args.support_quantile),
        "support_radius_multiplier": float(args.support_radius_multiplier),
        "num_candidates": len(rows),
        "num_sampled_branching_observed": sum(
            bool(row["sampled_distribution_contains_success_and_failure_branches"])
            for row in rows
        ),
        "num_training_eligible_before_intervention": 0,
        "output": str(output),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
