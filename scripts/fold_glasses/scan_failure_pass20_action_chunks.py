#!/usr/bin/env python3
"""Pass@20 action-chunk analysis on all S0 rollout failures.

This is the event-screening control protocol (exact GT prefix, then S0
closed-loop continuation on the replan grid) with three analysis changes:

* every recorded failure episode is scanned, not one failure per seed
* each node records a full Pass@20 (all 20 trials, no first-success skip)
* scanning a trajectory stops after three consecutive Pass@20 = 0 nodes

Every trial stores the first predicted action chunk and whether that
continuation eventually succeeded. Action loss is the z-score MSE between
that chunk and the factual GT actions at the same prefix.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.fold_glasses.discover_seedpair_branch_events import (
    load_global_zscore,
    normalize_actions,
)
from scripts.fold_glasses.scan_failure_recoverability_frontier import (
    DEFAULT_CFG,
    DEFAULT_CKPT,
    DEFAULT_STATS,
    DEFAULT_TEXT,
    FORMAT_VERSION as FRONTIER_FORMAT_VERSION,
    _compatible_cached_trajectory,
    _trajectory_public_row,
    assert_pin,
    atomic_write_json,
    atomic_write_jsonl,
    build_scan_frames,
    parse_ints,
    prepare_factual_snapshots,
    run_closed_loop_continuation,
    utc_now,
)
from scripts.fold_glasses.validate_factual_replay import (
    attempt_for_episode,
    create_environment,
    load_episode,
    read_json,
    setup_paths,
)


FORMAT_VERSION = "1.0"
DEFAULT_PASS_M = 20
DEFAULT_CONSECUTIVE_ZEROS = 3
DEFAULT_SCAN_START = 48
REPLAN_STEPS = 24
ACTION_HORIZON = 32


def _complete_attempt(row: Mapping[str, Any]) -> bool:
    return row.get("saved_episode_index") is not None and row.get("success") is not None


def select_all_failures(
    attempts: Sequence[Mapping[str, Any]],
    *,
    preferred_episode_indices: set[int] | None = None,
    seed_filter: set[int] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Select every completed failure attempt, preserving collection order."""

    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for raw in attempts:
        if not _complete_attempt(raw):
            continue
        grouped[int(raw["seed"])].append(dict(raw))

    selected: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    for seed in sorted(grouped):
        rows = sorted(
            grouped[seed],
            key=lambda row: (
                int(row.get("repeat", 1 << 30)),
                int(row["saved_episode_index"]),
            ),
        )
        successes = [row for row in rows if bool(row["success"])]
        failures = [row for row in rows if not bool(row["success"])]
        if successes and failures:
            classification = "mixed"
        elif successes:
            classification = "all_success"
        else:
            classification = "all_failure"

        kept: list[dict[str, Any]] = []
        skip_reason = "selected"
        if seed_filter is not None and seed not in seed_filter:
            skip_reason = "seed_not_requested"
        elif classification == "all_success":
            skip_reason = "all_success_excluded"
        else:
            candidates = failures
            if preferred_episode_indices is not None:
                candidates = [
                    row
                    for row in candidates
                    if int(row["saved_episode_index"]) in preferred_episode_indices
                ]
                if not candidates:
                    skip_reason = "no_requested_failure_episode"
            kept = candidates

        for row in kept:
            selected.append(
                {
                    **row,
                    "seed_classification": classification,
                    "training_eligible": False,
                    "evaluation_only": True,
                }
            )
        audit.append(
            {
                "seed": seed,
                "classification": classification,
                "success_episode_indices": [
                    int(row["saved_episode_index"]) for row in successes
                ],
                "failure_episode_indices": [
                    int(row["saved_episode_index"]) for row in failures
                ],
                "selected_failure_episode_indices": [
                    int(row["saved_episode_index"]) for row in kept
                ],
                "selection_reason": skip_reason if not kept else "selected_all_failures",
            }
        )
    selected.sort(
        key=lambda row: (
            int(row["seed"]),
            int(row.get("repeat", 1 << 30)),
            int(row["saved_episode_index"]),
        )
    )
    return selected, audit


def consecutive_zero_nodes_should_stop(
    prefix_rows: Sequence[Mapping[str, Any]], *, consecutive_zeros: int
) -> bool:
    """Stop after ``consecutive_zeros`` adjacent Pass@M = 0 prefixes."""

    if consecutive_zeros <= 0:
        return False
    if len(prefix_rows) < consecutive_zeros:
        return False
    tail = prefix_rows[-consecutive_zeros:]
    frames = [int(row["prefix_frame"]) for row in tail]
    if any(int(row.get("success_count", 0)) != 0 for row in tail):
        return False
    stride = frames[1] - frames[0]
    if stride <= 0:
        return False
    return all(
        frames[index] - frames[index - 1] == stride for index in range(1, len(frames))
    )


def action_chunk_mse(
    predicted: np.ndarray,
    gt: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    *,
    horizon: int,
) -> tuple[float, int]:
    """Z-score MSE over the overlapping prefix of predicted vs factual GT."""

    pred = np.asarray(predicted, dtype=np.float32)
    target = np.asarray(gt, dtype=np.float32)
    if pred.ndim != 2 or pred.shape[-1] != 22:
        raise ValueError(f"Predicted chunk must be [T, 22], got {pred.shape}")
    if target.ndim != 2 or target.shape[-1] != 22:
        raise ValueError(f"GT actions must be [T, 22], got {target.shape}")
    count = min(int(horizon), len(pred), len(target))
    if count <= 0:
        return float("nan"), 0
    left = normalize_actions(pred[:count], mean, std)
    right = normalize_actions(target[:count], mean, std)
    token = np.mean((left - right).astype(np.float64) ** 2, axis=1)
    return float(np.mean(token)), int(count)


def chunk_paths(output: Path, episode_index: int, prefix_frame: int, replicate: int) -> dict[str, Path]:
    prefix_dir = (
        output
        / "prefixes"
        / f"ep{episode_index:06d}_f{prefix_frame:04d}"
        / f"replicate_{replicate:02d}"
    )
    return {
        "dir": prefix_dir,
        "row": prefix_dir / "trajectory.json",
        "arrays": prefix_dir / "action_chunk.npz",
    }


def save_action_chunk_arrays(
    path: Path,
    *,
    predicted_chunk: np.ndarray,
    gt_chunk: np.ndarray,
    predicted_chunks: np.ndarray | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp.npz")
    payload: dict[str, np.ndarray] = {
        "first_action_chunk": np.asarray(predicted_chunk, dtype=np.float32),
        "gt_action_chunk": np.asarray(gt_chunk, dtype=np.float32),
    }
    if predicted_chunks is not None:
        payload["predicted_chunks"] = np.asarray(predicted_chunks, dtype=np.float32)
    np.savez_compressed(temporary, **payload)
    temporary.replace(path)


def gt_window(actions: np.ndarray, prefix_frame: int, horizon: int) -> np.ndarray:
    start = int(prefix_frame)
    end = min(len(actions), start + int(horizon))
    if start >= len(actions):
        return np.zeros((0, 22), dtype=np.float32)
    return np.asarray(actions[start:end], dtype=np.float32)


def scan_prefix(
    env: Any,
    policy: Any,
    *,
    snapshot: tuple[np.ndarray, dict[str, Any]],
    attempt: Mapping[str, Any],
    prefix_frame: int,
    pass_m: int,
    base_noise_seed: int,
    max_steps: int,
    output: Path,
    run_signature: Mapping[str, Any],
    overwrite: bool,
    gt_actions: np.ndarray,
    action_mean: np.ndarray,
    action_std: np.ndarray,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    episode_index = int(attempt["saved_episode_index"])
    target = gt_window(gt_actions, prefix_frame, ACTION_HORIZON)
    ledger_rows: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []
    for replicate_index in range(pass_m):
        paths = chunk_paths(output, episode_index, prefix_frame, replicate_index)
        row_path = paths["row"]
        if row_path.exists() and not overwrite:
            cached = read_json(row_path)
            if not _compatible_cached_trajectory(
                cached,
                run_signature=run_signature,
                replicate_index=replicate_index,
            ):
                raise RuntimeError(
                    f"Cached trajectory is incompatible with this run: {row_path}"
                )
            if not paths["arrays"].is_file():
                raise RuntimeError(f"Cached action-chunk artifact is missing: {row_path}")
            trajectory_rows.append(dict(cached))
            ledger_rows.append(dict(cached["ledger"]))
            continue

        result = run_closed_loop_continuation(
            env,
            policy,
            snapshot=snapshot,
            episode_index=episode_index,
            prefix_frame=prefix_frame,
            replicate_index=replicate_index,
            base_noise_seed=base_noise_seed,
            max_steps=max_steps,
        )
        predicted = np.asarray(result["first_action_chunk"], dtype=np.float32)
        loss_horizon, n_horizon = action_chunk_mse(
            predicted, target, action_mean, action_std, horizon=ACTION_HORIZON
        )
        loss_replan, n_replan = action_chunk_mse(
            predicted, target, action_mean, action_std, horizon=REPLAN_STEPS
        )
        save_action_chunk_arrays(
            paths["arrays"],
            predicted_chunk=predicted,
            gt_chunk=target,
            predicted_chunks=result.get("predicted_chunks"),
        )
        ledger = {
            "episode_index": episode_index,
            "seed": int(attempt["seed"]),
            "repeat": int(attempt["repeat"]),
            "prefix_frame": int(prefix_frame),
            "replicate_index": int(replicate_index),
            "success": bool(result["success"]),
            "pass_m": int(pass_m),
            "action_loss": float(loss_horizon),
            "action_loss_horizon": int(ACTION_HORIZON),
            "action_loss_n_valid": int(n_horizon),
            "action_loss_replan": float(loss_replan),
            "action_loss_replan_n_valid": int(n_replan),
            "steps_executed": int(result["steps_executed"]),
            "final_global_frame_exclusive": int(result["final_global_frame_exclusive"]),
            "action_chunk_arrays": str(paths["arrays"].resolve()),
        }
        row = {
            **_trajectory_public_row(result),
            "format": "FoldGlassesPassAtKActionChunkContinuation",
            "version": FORMAT_VERSION,
            "status": "complete",
            "seed": int(attempt["seed"]),
            "source_repeat": int(attempt["repeat"]),
            "source_failure_episode_index": episode_index,
            "ledger": ledger,
            "run_signature": dict(run_signature),
            "completed_at": utc_now(),
        }
        atomic_write_json(row_path, row)
        trajectory_rows.append(row)
        ledger_rows.append(ledger)

    success_count = sum(bool(row["success"]) for row in trajectory_rows)
    summary = {
        "format": "FoldGlassesPassAtKPrefixResult",
        "version": FORMAT_VERSION,
        "seed": int(attempt["seed"]),
        "seed_classification": str(attempt.get("seed_classification", "")),
        "source_failure_episode_index": episode_index,
        "source_repeat": int(attempt["repeat"]),
        "prefix_frame": int(prefix_frame),
        "pass_m": int(pass_m),
        "replicates_evaluated": len(trajectory_rows),
        "full_pass_at_m": len(trajectory_rows) >= int(pass_m),
        "success_count": int(success_count),
        "success_rate": float(success_count / pass_m),
        "pass_at_m_hit": bool(success_count > 0),
        "successful_replicate_indices": [
            int(row["replicate_index"])
            for row in trajectory_rows
            if bool(row["success"])
        ],
        "mean_action_loss_success": _json_float(_mean_loss(ledger_rows, success=True)),
        "mean_action_loss_failure": _json_float(_mean_loss(ledger_rows, success=False)),
        "run_signature": dict(run_signature),
        "completed_at": utc_now(),
    }
    prefix_dir = output / "prefixes" / f"ep{episode_index:06d}_f{prefix_frame:04d}"
    atomic_write_json(prefix_dir / "summary.json", summary)
    return summary, ledger_rows


def _json_float(value: float) -> float | None:
    if value is None or not np.isfinite(value):
        return None
    return float(value)


def _mean_loss(rows: Sequence[Mapping[str, Any]], *, success: bool) -> float:
    values = [
        float(row["action_loss"])
        for row in rows
        if bool(row["success"]) is success and np.isfinite(row.get("action_loss", np.nan))
    ]
    if not values:
        return float("nan")
    return float(np.mean(values))


def run_episode_scan(
    *,
    dataset: Path,
    output: Path,
    attempt: Mapping[str, Any],
    policy: Any,
    scan_frames: Sequence[int],
    pass_m: int,
    consecutive_zeros: int,
    base_noise_seed: int,
    max_steps: int,
    state_atol: float,
    run_signature: Mapping[str, Any],
    overwrite: bool,
    action_mean: np.ndarray,
    action_std: np.ndarray,
    task_name: str = "fold_glasses",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    episode_index = int(attempt["saved_episode_index"])
    canonical_attempt = attempt_for_episode(dataset, episode_index)
    for key in ("seed", "repeat", "success", "saved_episode_index"):
        if canonical_attempt[key] != attempt[key]:
            raise ValueError(
                f"Collection-summary mismatch for episode {episode_index}: {key}"
            )
    if bool(canonical_attempt["success"]):
        raise ValueError(f"Episode {episode_index} is not a failure")
    actions, recorded_states = load_episode(dataset, episode_index)
    _, env = create_environment(int(attempt["seed"]), task_name=task_name)
    episode_dir = output / "episodes" / f"ep{episode_index:06d}"
    try:
        snapshots, factual_gate = prepare_factual_snapshots(
            env,
            actions=actions,
            recorded_states=recorded_states,
            attempt=canonical_attempt,
            scan_frames=scan_frames,
            max_steps=max_steps,
            state_atol=state_atol,
        )
        atomic_write_json(episode_dir / "factual_replay_gate.json", factual_gate)

        prefix_rows: list[dict[str, Any]] = []
        ledger_rows: list[dict[str, Any]] = []
        stopped_reason = "completed_scan_grid"
        for position, prefix_frame in enumerate(scan_frames, start=1):
            print(
                f"[episode {episode_index} {position}/{len(scan_frames)}] "
                f"prefix={prefix_frame} M={pass_m}",
                flush=True,
            )
            prefix, ledgers = scan_prefix(
                env,
                policy,
                snapshot=snapshots[prefix_frame],
                attempt=attempt,
                prefix_frame=prefix_frame,
                pass_m=pass_m,
                base_noise_seed=base_noise_seed,
                max_steps=max_steps,
                output=output,
                run_signature=run_signature,
                overwrite=overwrite,
                gt_actions=actions,
                action_mean=action_mean,
                action_std=action_std,
            )
            prefix_rows.append(prefix)
            ledger_rows.extend(ledgers)
            if int(prefix["success_count"]) == 0:
                print(
                    f"[episode {episode_index}] Pass@{pass_m}=0 at prefix={prefix_frame}",
                    flush=True,
                )
            if consecutive_zero_nodes_should_stop(
                prefix_rows, consecutive_zeros=consecutive_zeros
            ):
                stopped_reason = (
                    f"consecutive_{consecutive_zeros}_pass_at_{pass_m}_zero"
                )
                print(
                    f"[episode {episode_index}] stop: {stopped_reason} "
                    f"ending at prefix={prefix_frame}",
                    flush=True,
                )
                break

        episode_summary = {
            "format": "FoldGlassesPassAtKEpisodeScan",
            "version": FORMAT_VERSION,
            "seed": int(attempt["seed"]),
            "seed_classification": str(attempt.get("seed_classification", "")),
            "source_failure_episode_index": episode_index,
            "source_repeat": int(attempt["repeat"]),
            "num_scan_points": len(prefix_rows),
            "pass_m": int(pass_m),
            "consecutive_zeros_stop": int(consecutive_zeros),
            "stopped_reason": stopped_reason,
            "prefix_success_counts": [
                {
                    "prefix_frame": int(row["prefix_frame"]),
                    "success_count": int(row["success_count"]),
                    "pass_at_m_hit": bool(row["pass_at_m_hit"]),
                }
                for row in prefix_rows
            ],
            "factual_replay_gate": str(
                (episode_dir / "factual_replay_gate.json").resolve()
            ),
            "run_signature": dict(run_signature),
            "completed_at": utc_now(),
        }
        atomic_write_json(episode_dir / "summary.json", episode_summary)
        return prefix_rows, ledger_rows, episode_summary
    finally:
        env.close()


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--collection-summary", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episode-indices", default="")
    parser.add_argument("--seeds", default="")
    parser.add_argument("--scan-frames", default="")
    parser.add_argument("--scan-start", type=int, default=DEFAULT_SCAN_START)
    parser.add_argument("--scan-end", type=int, default=0)
    parser.add_argument("--scan-stride", type=int, default=REPLAN_STEPS)
    parser.add_argument("--pass-m", type=int, default=DEFAULT_PASS_M)
    parser.add_argument(
        "--stop-after-consecutive-zeros",
        type=int,
        default=DEFAULT_CONSECUTIVE_ZEROS,
    )
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--action-horizon", type=int, default=ACTION_HORIZON)
    parser.add_argument("--replan-steps", type=int, default=REPLAN_STEPS)
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--base-noise-seed", type=int, default=20260813)
    parser.add_argument("--state-atol", type=float, default=2e-4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--model-config", type=Path, default=DEFAULT_CFG)
    parser.add_argument("--dataset-stats", type=Path, default=DEFAULT_STATS)
    parser.add_argument("--text-embedding", type=Path, default=DEFAULT_TEXT)
    parser.add_argument("--task-name", default="fold_glasses")
    parser.add_argument("--skip-pin-check", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--shard-rank", type=int, default=0)
    parser.add_argument("--shard-world", type=int, default=1)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if args.pass_m <= 0:
        raise ValueError("--pass-m must be positive")
    if args.stop_after_consecutive_zeros <= 0:
        raise ValueError("--stop-after-consecutive-zeros must be positive")
    if args.action_horizon != ACTION_HORIZON:
        raise ValueError(f"--action-horizon is fixed to {ACTION_HORIZON} for this analysis")
    if not 1 <= args.replan_steps <= args.action_horizon:
        raise ValueError("--replan-steps must lie within --action-horizon")
    scan_frames = build_scan_frames(
        requested_frames=parse_ints(args.scan_frames) or None,
        scan_start=int(args.scan_start),
        scan_end=int(args.scan_end),
        scan_stride=int(args.scan_stride),
        replan_steps=int(args.replan_steps),
        max_steps=int(args.max_steps),
    )

    dataset = args.dataset.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    collection_summary = (
        dataset / "collection_summary.json"
        if args.collection_summary is None
        else args.collection_summary.expanduser().resolve()
    )
    collection = read_json(collection_summary)
    preferred = set(parse_ints(args.episode_indices)) or None
    seed_filter = set(parse_ints(args.seeds)) or None
    selected, selection_audit = select_all_failures(
        collection.get("attempt_log", []),
        preferred_episode_indices=preferred,
        seed_filter=seed_filter,
    )
    atomic_write_jsonl(output / "seed_selection.jsonl", selection_audit)
    if not selected:
        raise ValueError("No failure episodes were selected")
    shard_world = int(args.shard_world)
    shard_rank = int(args.shard_rank)
    if shard_world < 1 or not 0 <= shard_rank < shard_world:
        raise ValueError(f"Invalid shard: rank={shard_rank} world={shard_world}")
    if shard_world > 1:
        selected = selected[shard_rank::shard_world]
        print(
            f"shard {shard_rank}/{shard_world}: {len(selected)} failure episodes",
            flush=True,
        )
        if not selected:
            atomic_write_json(
                output / "summary.json",
                {
                    "format": "FoldGlassesPassAtKActionChunkScan",
                    "version": FORMAT_VERSION,
                    "status": "complete",
                    "shard_rank": shard_rank,
                    "shard_world": shard_world,
                    "num_selected_failure_episodes": 0,
                },
            )
            return 0

    run_signature = {
        "format_version": FORMAT_VERSION,
        "frontier_format_version": FRONTIER_FORMAT_VERSION,
        "dataset": str(dataset),
        "collection_summary": str(collection_summary),
        "checkpoint": str(args.checkpoint.expanduser().resolve()),
        "model_config": str(args.model_config.expanduser().resolve()),
        "dataset_stats": str(args.dataset_stats.expanduser().resolve()),
        "text_embedding": str(args.text_embedding.expanduser().resolve()),
        "task_name": str(args.task_name),
        "action_horizon": int(args.action_horizon),
        "replan_steps": int(args.replan_steps),
        "num_inference_steps": int(args.num_inference_steps),
        "max_steps": int(args.max_steps),
        "pass_m": int(args.pass_m),
        "stop_after_consecutive_zeros": int(args.stop_after_consecutive_zeros),
        "scan_frames": scan_frames,
        "base_noise_seed": int(args.base_noise_seed),
        "selection": "all_failures",
    }
    atomic_write_json(
        output / "config.json",
        {
            **run_signature,
            "selected_failure_episodes": [
                int(row["saved_episode_index"]) for row in selected
            ],
            "created_at": utc_now(),
        },
    )

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("DIFFSYNTH_SKIP_DOWNLOAD", "true")
    os.environ.setdefault("DIFFSYNTH_MODEL_BASE_PATH", str(ROOT / "checkpoints"))
    if not args.skip_pin_check:
        assert_pin()
    setup_paths()
    from fastwam_dexjoco.policy import FastWAMDexJocoPolicy

    action_mean, action_std = load_global_zscore(
        args.dataset_stats.expanduser().resolve()
    )
    print(
        f"Loading policy once on {args.device}; episodes="
        f"{[int(row['saved_episode_index']) for row in selected]}",
        flush=True,
    )
    policy = FastWAMDexJocoPolicy(
        model_config=args.model_config.expanduser().resolve(),
        checkpoint=args.checkpoint.expanduser().resolve(),
        dataset_stats=args.dataset_stats.expanduser().resolve(),
        text_embedding=args.text_embedding.expanduser().resolve(),
        device=str(args.device),
        action_horizon=int(args.action_horizon),
        replan_steps=int(args.replan_steps),
        num_inference_steps=int(args.num_inference_steps),
    )

    all_prefix_rows: list[dict[str, Any]] = []
    all_ledger_rows: list[dict[str, Any]] = []
    episode_summaries: list[dict[str, Any]] = []
    started = time.perf_counter()
    for position, attempt in enumerate(selected, start=1):
        print(
            f"[selection {position}/{len(selected)}] seed={attempt['seed']} "
            f"episode={attempt['saved_episode_index']} "
            f"class={attempt['seed_classification']}",
            flush=True,
        )
        prefix_rows, ledger_rows, episode_summary = run_episode_scan(
            dataset=dataset,
            output=output,
            attempt=attempt,
            policy=policy,
            scan_frames=scan_frames,
            pass_m=int(args.pass_m),
            consecutive_zeros=int(args.stop_after_consecutive_zeros),
            base_noise_seed=int(args.base_noise_seed),
            max_steps=int(args.max_steps),
            state_atol=float(args.state_atol),
            run_signature=run_signature,
            overwrite=bool(args.overwrite),
            action_mean=action_mean,
            action_std=action_std,
            task_name=str(args.task_name),
        )
        all_prefix_rows.extend(prefix_rows)
        all_ledger_rows.extend(ledger_rows)
        episode_summaries.append(episode_summary)
        atomic_write_jsonl(output / "prefix_results.jsonl", all_prefix_rows)
        atomic_write_jsonl(output / "action_chunk_ledger.jsonl", all_ledger_rows)

    summary = {
        "format": "FoldGlassesPassAtKActionChunkScan",
        "version": FORMAT_VERSION,
        "status": "complete",
        "dataset": str(dataset),
        "num_selected_failure_episodes": len(selected),
        "num_prefix_results": len(all_prefix_rows),
        "num_action_chunks": len(all_ledger_rows),
        "pass_m": int(args.pass_m),
        "stop_after_consecutive_zeros": int(args.stop_after_consecutive_zeros),
        "prefix_results": str((output / "prefix_results.jsonl").resolve()),
        "action_chunk_ledger": str((output / "action_chunk_ledger.jsonl").resolve()),
        "run_signature": run_signature,
        "elapsed_s": float(time.perf_counter() - started),
        "completed_at": utc_now(),
    }
    atomic_write_json(output / "summary.json", summary)
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
