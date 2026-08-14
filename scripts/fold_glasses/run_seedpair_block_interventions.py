#!/usr/bin/env python3
"""Run exact-prefix action-block interventions for Fold Glasses candidates.

The factual failure prefix is replayed once and snapshotted immediately before
the candidate block. Every branch restores that exact MuJoCo integration state,
executes one 24-step action block, then optionally receives the same recorded
factual continuation. This separates immediate task progress from delayed
consequences while holding the pre-action state and downstream controls fixed.

Branches include the factual block, aligned successful blocks, policy samples at
the failure context, phase-shifted successful controls, and normalized-RMS
matched perturbations orthogonal to the span of successful action directions.
No branch becomes training data unless a later closed-loop continuation succeeds.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import mujoco
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.fold_glasses.score_seedpair_action_support import block_rms
from scripts.fold_glasses.validate_factual_replay import (
    attempt_for_episode,
    create_environment,
    load_episode,
    progress_metrics,
    read_json,
    read_jsonl,
    render_current_observation,
    reset_to_repeat,
    setup_paths,
)


DEFAULT_STATS = Path(
    "/data_all/xiangchengzhan/FastWAM-infer-in-DexJoco/"
    "artifacts/fold_glasses/dataset_stats.json"
)


def candidate_by_id(path: Path, candidate_id: str) -> dict[str, Any]:
    rows = read_jsonl(path)
    if candidate_id:
        matches = [row for row in rows if row.get("candidate_id") == candidate_id]
    else:
        matches = [row for row in rows if bool(row.get("probe_eligible"))]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one candidate, found {len(matches)}; pass --candidate-id"
        )
    return dict(matches[0])


def load_action_stats(path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = read_json(path)
    fields = payload["action"]["default"]
    mean = np.asarray(fields["global_mean"], dtype=np.float32)
    std = np.asarray(fields["global_std"], dtype=np.float32)
    if mean.shape != (22,) or std.shape != (22,) or np.any(std <= 0.0):
        raise ValueError(f"Unexpected action stats mean={mean.shape}, std={std.shape}")
    return mean, std


def normalize_action(block: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (np.asarray(block, dtype=np.float32) - mean) / std


def denormalize_action(block: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return np.asarray(block, dtype=np.float32) * std + mean


def orthogonal_matched_control(
    factual_normalized: np.ndarray,
    target_normalized: np.ndarray,
    basis_normalized: np.ndarray,
    success_directions: Sequence[np.ndarray],
) -> np.ndarray:
    """Match target RMS while removing all successful-direction components."""

    factual = np.asarray(factual_normalized, dtype=np.float64)
    target_delta = np.asarray(target_normalized, dtype=np.float64) - factual
    direction = np.asarray(basis_normalized, dtype=np.float64) - factual
    flat = direction.reshape(-1)
    columns = []
    for success in success_directions:
        delta = (np.asarray(success, dtype=np.float64) - factual).reshape(-1)
        norm = float(np.linalg.norm(delta))
        if norm > 1e-12:
            columns.append(delta / norm)
    if columns:
        q, _ = np.linalg.qr(np.stack(columns, axis=1))
        flat = flat - q @ (q.T @ flat)
    flat_norm = float(np.linalg.norm(flat))
    target_norm = float(np.linalg.norm(target_delta.reshape(-1)))
    if flat_norm <= 1e-12 or target_norm <= 1e-12:
        raise ValueError("Cannot construct non-degenerate matched control")
    matched = factual.reshape(-1) + flat * (target_norm / flat_norm)
    return matched.reshape(factual.shape).astype(np.float32)


def progress_delta(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    before_xyz = np.asarray(before["glass_minus_box_xyz"], dtype=np.float64)
    after_xyz = np.asarray(after["glass_minus_box_xyz"], dtype=np.float64)
    return {
        "hinge_0": float(after["hinge_0"] - before["hinge_0"]),
        "hinge_1": float(after["hinge_1"] - before["hinge_1"]),
        "hinge_min": float(after["hinge_min"] - before["hinge_min"]),
        "glass_minus_box_xyz": (after_xyz - before_xyz).tolist(),
        "glass_to_box_xy_distance": float(
            np.linalg.norm(after_xyz[:2]) - np.linalg.norm(before_xyz[:2])
        ),
        "trigger_active_changed": bool(
            after["trigger_active"] and not before["trigger_active"]
        ),
    }


def snapshot_integration_state(env: Any) -> tuple[np.ndarray, dict[str, Any]]:
    raw = env.unwrapped
    specification = mujoco.mjtState.mjSTATE_INTEGRATION
    size = mujoco.mj_stateSize(raw._model, specification)
    state = np.empty(size, dtype=np.float64)
    mujoco.mj_getState(raw._model, raw._data, state, specification)
    attrs = {
        "env_step": int(raw.env_step),
        "success_trigger_count": int(raw._success_trigger_count),
        "reset_trigger": bool(raw.reset_trigger),
    }
    return state, attrs


def restore_integration_state(
    env: Any, state: np.ndarray, attrs: Mapping[str, Any]
) -> None:
    raw = env.unwrapped
    specification = mujoco.mjtState.mjSTATE_INTEGRATION
    mujoco.mj_setState(raw._model, raw._data, state, specification)
    raw.env_step = int(attrs["env_step"])
    raw._success_trigger_count = int(attrs["success_trigger_count"])
    raw.reset_trigger = bool(attrs["reset_trigger"])
    mujoco.mj_forward(raw._model, raw._data)


def save_observation_images(
    observation: Mapping[str, np.ndarray], output: Path, branch_id: str, stage: str
) -> None:
    for camera in ("front", "wrist"):
        Image.fromarray(np.asarray(observation[camera], dtype=np.uint8)).save(
            output / f"{branch_id}_{stage}_{camera}.png"
        )


def make_branches(
    dataset: Path,
    candidate: Mapping[str, Any],
    probe_root: Path,
    *,
    mean: np.ndarray,
    std: np.ndarray,
    block_size: int,
    include_policy_samples: bool,
    include_phase_controls: bool,
    include_orthogonal_controls: bool,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    failure_episode = int(candidate["failure_episode_index"])
    failure_frame = int(candidate["failure_frame"])
    failure_actions, _ = load_episode(dataset, failure_episode)
    factual = failure_actions[failure_frame : failure_frame + block_size]
    if factual.shape != (block_size, 22):
        raise ValueError(f"Incomplete factual block {factual.shape}")
    factual_norm = normalize_action(factual, mean, std)
    branches: list[dict[str, Any]] = [
        {
            "branch_id": "factual",
            "kind": "factual_control",
            "actions_robot": factual,
            "source_episode": failure_episode,
            "source_frame": failure_frame,
        }
    ]
    success_blocks: list[np.ndarray] = []
    success_meta: list[tuple[int, int]] = []
    for alignment in candidate.get("success_alignments", []):
        episode = int(alignment["success_episode_index"])
        frame = int(alignment["success_frame"])
        actions, _ = load_episode(dataset, episode)
        block = actions[frame : frame + block_size]
        if block.shape != (block_size, 22):
            continue
        success_blocks.append(normalize_action(block, mean, std))
        success_meta.append((episode, frame))
        branches.append(
            {
                "branch_id": f"success_ep{episode:06d}_f{frame:04d}",
                "kind": "aligned_success_actual",
                "actions_robot": block,
                "source_episode": episode,
                "source_frame": frame,
            }
        )
        if include_phase_controls:
            for shift_name, shifted_frame in (
                ("early", frame - block_size),
                ("late", frame + block_size),
            ):
                shifted = actions[shifted_frame : shifted_frame + block_size]
                if shifted_frame < 0 or shifted.shape != (block_size, 22):
                    continue
                branches.append(
                    {
                        "branch_id": (
                            f"phase_{shift_name}_ep{episode:06d}_f{shifted_frame:04d}"
                        ),
                        "kind": "phase_shift_control",
                        "actions_robot": shifted,
                        "source_episode": episode,
                        "source_frame": shifted_frame,
                        "aligned_success_frame": frame,
                    }
                )

    probe_path = (
        probe_root
        / "contexts"
        / f"ep{failure_episode:06d}_f{failure_frame:04d}.npz"
    )
    with np.load(probe_path) as payload:
        if include_policy_samples:
            samples = np.asarray(payload["action_samples_robot"], dtype=np.float32)
            noise_seeds = np.asarray(payload["noise_seeds"], dtype=np.int64)
            for index, (sample, noise_seed) in enumerate(zip(samples, noise_seeds)):
                branches.append(
                    {
                        "branch_id": f"failure_policy_sample_{index:02d}",
                        "kind": "failure_context_policy_sample",
                        "actions_robot": sample,
                        "noise_seed": int(noise_seed),
                    }
                )

    if include_orthogonal_controls and success_blocks:
        basis_candidates = [
            normalize_action(branch["actions_robot"], mean, std)
            for branch in branches
            if branch["kind"] == "phase_shift_control"
        ]
        if not basis_candidates:
            basis_candidates = [np.flip(factual_norm, axis=0).copy()]
        for index, (target, meta) in enumerate(zip(success_blocks, success_meta)):
            control_norm = orthogonal_matched_control(
                factual_norm,
                target,
                basis_candidates[index % len(basis_candidates)],
                success_blocks,
            )
            control = denormalize_action(control_norm, mean, std)
            branches.append(
                {
                    "branch_id": f"orthogonal_rms_matched_{index:02d}",
                    "kind": "orthogonal_rms_matched_control",
                    "actions_robot": control,
                    "matched_success_episode": int(meta[0]),
                    "matched_success_frame": int(meta[1]),
                }
            )
    return branches, factual


def run_interventions(
    *,
    dataset: Path,
    candidate: Mapping[str, Any],
    probe_root: Path,
    factual_validation: Path,
    dataset_stats: Path,
    output: Path,
    block_size: int = 24,
    continuation_steps: int = 96,
    max_abs_normalized: float = 5.0,
    include_policy_samples: bool = True,
    include_phase_controls: bool = True,
    include_orthogonal_controls: bool = True,
) -> dict[str, Any]:
    validation = read_json(factual_validation)
    if not bool(validation.get("factual_replay_passed")):
        raise ValueError("Factual replay validation did not pass")
    failure_episode = int(candidate["failure_episode_index"])
    failure_frame = int(candidate["failure_frame"])
    if int(validation["episode_index"]) != failure_episode:
        raise ValueError("Factual validation episode does not match candidate")
    if not bool(validation.get("full_replay")):
        raise ValueError("Intervention requires a successful full factual replay gate")

    setup_paths()
    from fastwam_dexjoco.policy import fastwam_action_to_dexjoco

    mean, std = load_action_stats(dataset_stats)
    branches, factual_block = make_branches(
        dataset,
        candidate,
        probe_root,
        mean=mean,
        std=std,
        block_size=block_size,
        include_policy_samples=include_policy_samples,
        include_phase_controls=include_phase_controls,
        include_orthogonal_controls=include_orthogonal_controls,
    )
    failure_actions, recorded_states = load_episode(dataset, failure_episode)
    continuation_start = failure_frame + block_size
    continuation = failure_actions[
        continuation_start : continuation_start + continuation_steps
    ]
    attempt = attempt_for_episode(dataset, failure_episode)
    _, env = create_environment(int(attempt["seed"]))
    output.mkdir(parents=True, exist_ok=True)
    try:
        observation, _ = reset_to_repeat(env, int(attempt["repeat"]))
        env.unwrapped.image_obs = False
        for frame in range(failure_frame):
            observation, _, terminated, truncated, _ = env.step(
                fastwam_action_to_dexjoco(failure_actions[frame])
            )
            if terminated or truncated:
                raise RuntimeError(f"Factual prefix terminated at frame {frame}")
        prefix_observation = render_current_observation(env)
        prefix_state = np.asarray(prefix_observation["state"], dtype=np.float32)[:23]
        prefix_error = float(
            np.max(np.abs(prefix_state - recorded_states[failure_frame]))
        )
        if prefix_error > float(validation["thresholds"]["state_max_abs"]):
            raise RuntimeError(
                f"Intervention prefix state error {prefix_error} exceeds factual gate"
            )
        save_observation_images(prefix_observation, output, "shared_prefix", "before")
        snapshot, attrs = snapshot_integration_state(env)
        before = progress_metrics(env)

        rows: list[dict[str, Any]] = []
        factual_norm = normalize_action(factual_block, mean, std)
        for branch in branches:
            actions_robot = np.asarray(branch.pop("actions_robot"), dtype=np.float32)
            normalized = normalize_action(actions_robot, mean, std)
            max_abs = float(np.max(np.abs(normalized)))
            record: dict[str, Any] = {
                **branch,
                "num_block_steps": len(actions_robot),
                "normalized_rms_from_factual": block_rms(normalized, factual_norm),
                "normalized_max_abs": max_abs,
                "executed": False,
            }
            if max_abs > max_abs_normalized:
                record["skip_reason"] = "normalized_action_exceeds_safety_bound"
                rows.append(record)
                continue
            restore_integration_state(env, snapshot, attrs)
            terminated = False
            truncated = False
            info: Mapping[str, Any] = {"succeed": False}
            for action in actions_robot:
                _, _, terminated, truncated, info = env.step(
                    fastwam_action_to_dexjoco(action)
                )
                if terminated or truncated:
                    break
            after_block = progress_metrics(env)
            after_block_observation = render_current_observation(env)
            save_observation_images(
                after_block_observation, output, str(branch["branch_id"]), "after_block"
            )
            continuation_executed = 0
            if not (terminated or truncated):
                for action in continuation:
                    _, _, terminated, truncated, info = env.step(
                        fastwam_action_to_dexjoco(action)
                    )
                    continuation_executed += 1
                    if terminated or truncated:
                        break
            after_continuation = progress_metrics(env)
            final_observation = render_current_observation(env)
            save_observation_images(
                final_observation,
                output,
                str(branch["branch_id"]),
                "after_continuation",
            )
            record.update(
                {
                    "executed": True,
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "succeeded": bool(info.get("succeed", False)),
                    "continuation_steps_executed": continuation_executed,
                    "progress_before": before,
                    "progress_after_block": after_block,
                    "progress_delta_after_block": progress_delta(before, after_block),
                    "progress_after_continuation": after_continuation,
                    "progress_delta_after_continuation": progress_delta(
                        before, after_continuation
                    ),
                }
            )
            rows.append(record)
    finally:
        env.close()

    factual_rows = [row for row in rows if row["kind"] == "factual_control"]
    if len(factual_rows) != 1 or not factual_rows[0].get("executed"):
        raise RuntimeError("Factual intervention control did not execute")
    factual_row = factual_rows[0]
    for row in rows:
        if not row.get("executed"):
            continue
        row["effect_vs_factual"] = {
            "hinge_min_after_block": float(
                row["progress_after_block"]["hinge_min"]
                - factual_row["progress_after_block"]["hinge_min"]
            ),
            "hinge_min_after_continuation": float(
                row["progress_after_continuation"]["hinge_min"]
                - factual_row["progress_after_continuation"]["hinge_min"]
            ),
            "glass_to_box_xy_after_continuation": float(
                np.linalg.norm(row["progress_after_continuation"]["glass_minus_box_xyz"][:2])
                - np.linalg.norm(
                    factual_row["progress_after_continuation"]["glass_minus_box_xyz"][:2]
                )
            ),
        }

    ledger = output / "branch_results.jsonl"
    ledger.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    summary = {
        "format": "FoldGlassesExactPrefixBlockIntervention",
        "version": "1.0",
        "candidate_id": str(candidate["candidate_id"]),
        "failure_episode_index": failure_episode,
        "failure_frame": failure_frame,
        "seed": int(attempt["seed"]),
        "repeat": int(attempt["repeat"]),
        "factual_validation": str(factual_validation),
        "prefix_state_max_abs_error": prefix_error,
        "block_size": block_size,
        "factual_continuation_steps": len(continuation),
        "num_branches": len(rows),
        "num_executed": sum(bool(row.get("executed")) for row in rows),
        "num_succeeded_within_intervention": sum(bool(row.get("succeeded")) for row in rows),
        "training_data_generated": False,
        "requires_closed_loop_continuation_for_training": True,
        "branch_results": str(ledger),
        "output": str(output),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--candidate-id", default="")
    parser.add_argument("--probe-root", type=Path, required=True)
    parser.add_argument("--factual-validation", type=Path, required=True)
    parser.add_argument("--dataset-stats", type=Path, default=DEFAULT_STATS)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--block-size", type=int, default=24)
    parser.add_argument("--continuation-steps", type=int, default=96)
    parser.add_argument("--max-abs-normalized", type=float, default=5.0)
    parser.add_argument("--no-policy-samples", action="store_true")
    parser.add_argument("--no-phase-controls", action="store_true")
    parser.add_argument("--no-orthogonal-controls", action="store_true")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if args.block_size <= 0 or args.continuation_steps < 0:
        raise ValueError("Invalid intervention lengths")
    candidate = candidate_by_id(
        args.candidates.expanduser().resolve(), str(args.candidate_id)
    )
    result = run_interventions(
        dataset=args.dataset.expanduser().resolve(),
        candidate=candidate,
        probe_root=args.probe_root.expanduser().resolve(),
        factual_validation=args.factual_validation.expanduser().resolve(),
        dataset_stats=args.dataset_stats.expanduser().resolve(),
        output=args.output.expanduser().resolve(),
        block_size=int(args.block_size),
        continuation_steps=int(args.continuation_steps),
        max_abs_normalized=float(args.max_abs_normalized),
        include_policy_samples=not bool(args.no_policy_samples),
        include_phase_controls=not bool(args.no_phase_controls),
        include_orthogonal_controls=not bool(args.no_orthogonal_controls),
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
