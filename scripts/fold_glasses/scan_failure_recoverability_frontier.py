#!/usr/bin/env python3
"""Scan recoverability pair events along recorded Fold Glasses failures.

For each seed, select one failed rollout if any exist. Starting at frame 48
and stepping by the 24-frame replan grid, replay the factual GT prefix to M,
then run S0 closed-loop continuations to the original horizon. Every prefix
records a full Pass@4 (all M trials). The first 0/4 prefix is the
unrecoverable failure frame M. The pair is:

* failure event: the factual GT window ``[M-33, M)``
* success event: crop of that same window from a **saved closed-loop success
  rollout** started at ``t = M-24``

Pass@4 therefore captures RGB for every continuation frame and writes the
full success rollout before any event is materialized. The event is sliced
from that saved rollout. It is not a same-noise policy rerun, and it is not
an open-loop replay of recorded actions.

Later recovery islands are ignored. Original S0 success rollouts are not
used as pair training data.
"""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

# These must be selected before importing MuJoCo through the snapshot helper.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import av
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OPEN = Path(
    os.environ.get(
        "FASTWAM_OPEN_REPO",
        str(ROOT.parent / "FastWAM-infer-in-DexJoco"),
    )
)
FASTWAM_PIN = Path(
    os.environ.get("FASTWAM_PIN", str(ROOT / "third_party/FastWAM_pin_45d8e14"))
)
EXPECTED_PIN = "45d8e1458921d83f8ad6cf9ce993d371208dabd0"
DEFAULT_CFG = OPEN / "configs/fastwam_dexjoco.yaml"
DEFAULT_CKPT = (
    ROOT / "checkpoints/dexjoco/fold_glasses_fastwam/weights/step_010000.pt"
)
DEFAULT_STATS = OPEN / "artifacts/fold_glasses/dataset_stats.json"
DEFAULT_TEXT = (
    OPEN
    / "artifacts/fold_glasses"
    / "0c3367ce1d74848cc46b93c6d2eee5e2097dca410a2c95f3da48bd8c8673fa20.t5_len128.wan22ti2v5b.pt"
)


def assert_pin() -> None:
    import subprocess

    head = subprocess.check_output(
        ["git", "-C", str(FASTWAM_PIN), "rev-parse", "HEAD"], text=True
    ).strip()
    if head != EXPECTED_PIN:
        raise RuntimeError(f"FastWAM pin mismatch: {head}")


from scripts.fold_glasses.run_seedpair_block_interventions import (
    restore_integration_state,
    snapshot_integration_state,
)
from scripts.fold_glasses.validate_factual_replay import (
    attempt_for_episode,
    create_environment,
    load_episode,
    progress_metrics,
    read_json,
    read_video_frames,
    render_current_observation,
    reset_to_repeat,
    setup_paths,
    video_paths,
)


FORMAT_VERSION = "2.0"
DEFAULT_HARD_TRANSFER_SEEDS: tuple[int, ...] = ()
SUCCESS_EVENT_SOURCE = "cropped_from_saved_success_rollout"
NOISE_SCHEME = (
    "blake2b64(fastwam-recoverability-v2-common-prefix fields), masked to int63"
)
EVENT_NUM_FRAMES = 33
REPLAN_STEPS = 24
EVENT_POST_FRAMES = REPLAN_STEPS
EVENT_PRE_FRAMES = EVENT_NUM_FRAMES - EVENT_POST_FRAMES
DEFAULT_SCAN_START = 48
SUCCESS_TRIGGER_TARGET = 50


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_ints(raw: str) -> list[int]:
    return sorted({int(value.strip()) for value in raw.split(",") if value.strip()})


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def atomic_write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(path)


def _complete_attempt(row: Mapping[str, Any]) -> bool:
    return row.get("saved_episode_index") is not None and row.get("success") is not None


def select_one_failure_per_seed(
    attempts: Sequence[Mapping[str, Any]],
    *,
    preferred_episode_indices: set[int] | None = None,
    seed_filter: set[int] | None = None,
    hard_transfer_seeds: set[int] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Select at most one failure per seed.

    Seeds with no failures are skipped. Mixed and all-failure seeds each
    contribute their earliest recorded failure.
    """

    hard_transfer = (
        set(DEFAULT_HARD_TRANSFER_SEEDS)
        if hard_transfer_seeds is None
        else {int(value) for value in hard_transfer_seeds}
    )
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for raw in attempts:
        if not _complete_attempt(raw):
            continue
        row = dict(raw)
        grouped[int(row["seed"])].append(row)

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

        reason = "selected"
        candidates = failures
        if seed_filter is not None and seed not in seed_filter:
            reason = "seed_not_requested"
            candidates = []
        elif classification == "all_success":
            reason = "all_success_excluded"
            candidates = []
        elif classification == "all_failure":
            reason = "selected"
        if preferred_episode_indices is not None and candidates:
            candidates = [
                row
                for row in candidates
                if int(row["saved_episode_index"]) in preferred_episode_indices
            ]
            if not candidates:
                reason = "no_requested_failure_episode"

        chosen: dict[str, Any] | None = None
        if candidates:
            # The earliest failure is the least selected-by-outcome-biased
            # representative.  Repeat is the collector's chronological order;
            # episode index is a stable tie-breaker for older ledgers.
            chosen = min(
                candidates,
                key=lambda row: (
                    int(row.get("repeat", 1 << 30)),
                    int(row["saved_episode_index"]),
                ),
            )
            chosen = {
                **chosen,
                "seed_classification": classification,
                "training_eligible": True,
                "evaluation_only": False,
            }
            selected.append(chosen)

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
                "selected_failure_episode_index": (
                    None if chosen is None else int(chosen["saved_episode_index"])
                ),
                "training_eligible": bool(
                    chosen is not None and chosen["training_eligible"]
                ),
                "evaluation_only": bool(
                    chosen is not None and chosen["evaluation_only"]
                ),
                "selection_reason": reason,
            }
        )
    return selected, audit


def recorded_horizon(
    actions: Sequence[Any],
    recorded_states: Sequence[Any],
    *,
    max_steps: int,
) -> int:
    """DexJoCo episodes stop at env horizon (~1000) or earlier on success/fail."""

    n = min(len(actions), len(recorded_states))
    if n <= 0:
        raise ValueError("Episode has no frames")
    return min(int(max_steps), n)


def clip_scan_frames(frames: Sequence[int], *, horizon: int) -> list[int]:
    return [int(frame) for frame in frames if 0 <= int(frame) < int(horizon)]


def validate_scan_frames(
    frames: Sequence[int], *, replan_steps: int, max_steps: int
) -> list[int]:
    if replan_steps <= 0 or max_steps <= 0:
        raise ValueError("replan_steps and max_steps must be positive")
    result = sorted({int(frame) for frame in frames})
    if not result:
        raise ValueError("No scan frames were selected")
    invalid_range = [frame for frame in result if not 0 <= frame < max_steps]
    if invalid_range:
        raise ValueError(
            f"Scan frames must lie in [0, {max_steps}); got {invalid_range}"
        )
    misaligned = [frame for frame in result if frame % replan_steps != 0]
    if misaligned:
        raise ValueError(
            f"Exact-prefix frames must be aligned to replan_steps={replan_steps}; "
            f"got {misaligned}"
        )
    return result


def build_scan_frames(
    *,
    requested_frames: Sequence[int] | None,
    scan_start: int,
    scan_end: int,
    scan_stride: int,
    replan_steps: int,
    max_steps: int,
) -> list[int]:
    if scan_stride <= 0 or scan_stride % replan_steps != 0:
        raise ValueError(
            "scan_stride must be a positive multiple of replan_steps "
            f"({replan_steps})"
        )
    if requested_frames:
        return validate_scan_frames(
            requested_frames, replan_steps=replan_steps, max_steps=max_steps
        )
    end = max_steps if scan_end <= 0 else min(scan_end, max_steps)
    if scan_start < 0 or scan_start >= end:
        raise ValueError(f"Invalid scan interval [{scan_start}, {end})")
    if scan_start % replan_steps != 0:
        raise ValueError("scan_start must be replan aligned")
    return validate_scan_frames(
        range(scan_start, end, scan_stride),
        replan_steps=replan_steps,
        max_steps=max_steps,
    )


def find_recoverability_frontiers(
    prefix_rows: Sequence[Mapping[str, Any]],
    *,
    block_size: int = 24,
    expansion_blocks: int = 1,
    event_pre_frames: int = EVENT_PRE_FRAMES,
    event_post_frames: int = EVENT_POST_FRAMES,
    pass_m: int | None = 4,
    max_steps: int | None = None,
) -> list[dict[str, Any]]:
    """Return the first recoverable prefix followed by an adjacent 0/M point.

    ``t`` is the last prefix with at least one success. ``M = t+24`` is the
    first 0/4 unrecoverable scan point. The event is ``[M-33, M)``. A 4/4
    prefix followed by 0/4 is a valid pair. Isolated later recovery islands
    are ignored because the scanner stops at the first cliff.
    """

    del expansion_blocks
    if block_size <= 0:
        raise ValueError("Invalid event expansion")
    if block_size != REPLAN_STEPS:
        raise ValueError(
            f"Recoverability scan block is fixed to {REPLAN_STEPS} frames; "
            f"got {block_size}"
        )
    if event_pre_frames != EVENT_PRE_FRAMES or event_post_frames != EVENT_POST_FRAMES:
        raise ValueError(
            f"Recoverability event window is fixed to {EVENT_NUM_FRAMES} frames "
            f"ending at the first 0/M point"
        )
    rows = sorted(prefix_rows, key=lambda row: int(row["prefix_frame"]))
    frames = [int(row["prefix_frame"]) for row in rows]
    if len(frames) != len(set(frames)):
        raise ValueError("Duplicate prefix frames")

    last_hit: Mapping[str, Any] | None = None
    for row in rows:
        success_count = int(row.get("success_count", 0))
        frame = int(row["prefix_frame"])
        if success_count > 0:
            last_hit = row
            continue
        if last_hit is None:
            continue
        last_recoverable = int(last_hit["prefix_frame"])
        if frame != last_recoverable + block_size:
            continue
        failure_frame = frame
        event_end = failure_frame
        event_start = failure_frame - EVENT_NUM_FRAMES
        short_event_window = event_start < 0
        if short_event_window:
            event_start = 0
        if max_steps is not None:
            event_end = min(int(max_steps), event_end)
        requested_num_frames = event_end - event_start
        row_pass_m = int(last_hit.get("pass_m", pass_m or 0))
        time_budget_censored = bool(
            max_steps is not None
            and failure_frame > int(max_steps) - SUCCESS_TRIGGER_TARGET
        )
        return [
            {
                "frontier_id": f"frontier_00_f{failure_frame:04d}",
                "recovery_island_index": 0,
                "t_frame": last_recoverable,
                "t_plus_24_frame": failure_frame,
                "last_recoverable_frame": last_recoverable,
                "last_recoverable_success_count": int(last_hit["success_count"]),
                "first_zero_frame": failure_frame,
                "failure_frame": failure_frame,
                "core_event_start": last_recoverable,
                "core_event_end": failure_frame,
                "snapshot_frame": last_recoverable,
                "event_start": event_start,
                "event_end_exclusive": event_end,
                "event_window": [event_start, event_end],
                "event_pre_frames": int(event_pre_frames),
                "event_post_frames": int(event_post_frames),
                "num_event_frames": int(requested_num_frames),
                "short_event_window": bool(
                    short_event_window or requested_num_frames != EVENT_NUM_FRAMES
                ),
                "pass_m": row_pass_m if row_pass_m > 0 else pass_m,
                "prefix_is_mixed": bool(
                    1 <= int(last_hit["success_count"]) < (row_pass_m or pass_m or 4)
                ),
                "time_budget_censored": time_budget_censored,
                "empirical_frontier_only": True,
                "absolute_irreversibility_claimed": False,
            }
        ]
    return []


def training_pair_eligible(
    attempt: Mapping[str, Any], frontier: Mapping[str, Any], *, pass_m: int
) -> bool:
    """Return whether the first 0/4 cliff can be materialized as a pair."""

    del attempt
    count = int(frontier.get("last_recoverable_success_count", 0))
    return bool(
        count >= 1
        and not bool(frontier.get("time_budget_censored", False))
        and not bool(frontier.get("short_event_window", False))
        and int(frontier.get("num_event_frames", EVENT_NUM_FRAMES))
        == EVENT_NUM_FRAMES
    )


def estimate_scan_cost(
    *,
    num_episodes: int,
    scan_frames: Sequence[int],
    pass_m: int,
    max_steps: int,
    replan_steps: int,
) -> dict[str, Any]:
    """Estimate simulation/inference work before loading MuJoCo or the policy."""

    if num_episodes < 0 or pass_m <= 0 or max_steps <= 0 or replan_steps <= 0:
        raise ValueError("Invalid cost-estimate inputs")
    continuation_steps = sum(
        max(0, max_steps - int(frame)) * int(pass_m) for frame in scan_frames
    )
    continuation_replans = sum(
        ((max(0, max_steps - int(frame)) + replan_steps - 1) // replan_steps)
        * int(pass_m)
        for frame in scan_frames
    )
    factual_steps = int(num_episodes) * int(max_steps)
    return {
        "num_episodes": int(num_episodes),
        "num_scan_points_per_episode": len(scan_frames),
        "pass_m": int(pass_m),
        "max_steps": int(max_steps),
        "replan_steps": int(replan_steps),
        "factual_replay_steps": factual_steps,
        "continuation_steps": int(continuation_steps * num_episodes),
        "total_simulation_steps": int(
            factual_steps + continuation_steps * num_episodes
        ),
        "continuation_policy_replans": int(continuation_replans * num_episodes),
        "policy_load_count": 1,
        "note": (
            "Upper bound assumes every prefix runs all Pass@4 trials to "
            "max_steps. Successful continuations may still terminate before "
            "max_steps; later prefixes after the first 0/M cliff are not scanned."
        ),
    }


def continuation_noise_seed(
    base_seed: int,
    episode_index: int,
    prefix_frame: int,
    replicate_index: int,
    replan_index: int,
) -> int:
    # Common-random-number schedule: replicate i uses the same stream at every
    # prefix K, making changes in pass@M attributable to recoverability rather
    # than a different diffusion draw.  Prefix is retained in the signature for
    # provenance/API compatibility but intentionally excluded from the hash.
    fields = (
        "fastwam-recoverability-v1",
        int(base_seed),
        int(episode_index),
        int(replicate_index),
        int(replan_index),
    )
    digest = hashlib.blake2b(
        ":".join(str(value) for value in fields).encode("ascii"), digest_size=8
    ).digest()
    return int.from_bytes(digest, "little") & ((1 << 63) - 1)


def _state23(observation: Mapping[str, Any]) -> np.ndarray:
    state = np.asarray(observation["state"], dtype=np.float32).reshape(-1)
    if state.shape[0] < 23:
        raise ValueError(f"Observation state is too short: {state.shape}")
    return state[:23].copy()


def prepare_factual_snapshots(
    env: Any,
    *,
    actions: np.ndarray,
    recorded_states: np.ndarray,
    attempt: Mapping[str, Any],
    scan_frames: Sequence[int],
    max_steps: int,
    state_atol: float,
) -> tuple[dict[int, tuple[np.ndarray, dict[str, Any]]], dict[str, Any]]:
    """Run the full factual gate once and retain exact integration snapshots."""

    from fastwam_dexjoco.policy import fastwam_action_to_dexjoco

    horizon = recorded_horizon(actions, recorded_states, max_steps=max_steps)
    wanted = set(clip_scan_frames(scan_frames, horizon=horizon))
    if not wanted:
        raise ValueError(
            f"No scan frames within recorded horizon={horizon} "
            f"(max_steps={max_steps})"
        )
    snapshots: dict[int, tuple[np.ndarray, dict[str, Any]]] = {}
    checks: list[dict[str, Any]] = []
    observation, _ = reset_to_repeat(env, int(attempt["repeat"]))
    env.unwrapped.image_obs = False
    final_info: Mapping[str, Any] = {"succeed": False}
    terminated = False
    truncated = False
    executed = 0
    for frame in range(horizon):
        if frame in wanted:
            error = np.abs(_state23(observation) - recorded_states[frame, :23])
            checks.append(
                {
                    "frame": frame,
                    "state_max_abs": float(error.max()),
                    "state_rmse": float(np.sqrt(np.mean(error**2))),
                }
            )
            snapshots[frame] = snapshot_integration_state(env)
        observation, _, terminated, truncated, final_info = env.step(
            fastwam_action_to_dexjoco(actions[frame])
        )
        executed += 1
        if terminated or truncated:
            if frame + 1 < horizon:
                raise RuntimeError(
                    f"Factual replay terminated at frame {frame}, before {horizon}"
                )
            break

    missing = sorted(wanted - set(snapshots))
    if missing:
        raise RuntimeError(f"Factual replay did not reach scan frames {missing}")
    max_error = max((row["state_max_abs"] for row in checks), default=0.0)
    replayed_success = bool(final_info.get("succeed", False))
    recorded_success = bool(attempt["success"])
    passed = bool(
        executed == horizon
        and max_error <= state_atol
        and replayed_success == recorded_success
    )
    gate = {
        "format": "FoldGlassesIntegratedFactualReplayGate",
        "version": FORMAT_VERSION,
        "episode_index": int(attempt["saved_episode_index"]),
        "seed": int(attempt["seed"]),
        "repeat": int(attempt["repeat"]),
        "recorded_success": recorded_success,
        "replayed_success": replayed_success,
        "replayed_steps": executed,
        "state_atol": float(state_atol),
        "state_max_abs": float(max_error),
        "checks": checks,
        "factual_replay_passed": passed,
        "final_progress": progress_metrics(env),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
    }
    if not passed:
        raise RuntimeError(
            "Integrated factual replay gate failed: "
            f"episode={attempt['saved_episode_index']} max_error={max_error:.6g} "
            f"outcome={replayed_success}/{recorded_success} steps={executed}/{horizon}"
        )
    return snapshots, gate


def run_closed_loop_continuation(
    env: Any,
    policy: Any,
    *,
    snapshot: tuple[np.ndarray, Mapping[str, Any]],
    episode_index: int,
    prefix_frame: int,
    replicate_index: int,
    base_noise_seed: int,
    max_steps: int,
    capture_window: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """Restore one exact prefix and start a fresh receding-horizon rollout."""

    from fastwam_dexjoco.policy import fastwam_action_to_dexjoco

    restore_integration_state(env, snapshot[0], snapshot[1])
    observation = render_current_observation(env)
    env.unwrapped.image_obs = False
    pending: deque[np.ndarray] = deque()
    actions: list[np.ndarray] = []
    states: list[np.ndarray] = []
    captured_frames: list[int] = []
    fronts: list[np.ndarray] = []
    wrists: list[np.ndarray] = []
    predicted_chunks: list[np.ndarray] = []
    noise_schedule: list[dict[str, int]] = []
    replan_index = 0
    terminated = False
    truncated = False
    succeeded = False
    final_info: Mapping[str, Any] = {"succeed": False}
    started = time.perf_counter()

    for frame in range(prefix_frame, max_steps):
        capture = bool(
            capture_window is not None
            and capture_window[0] <= frame < capture_window[1]
        )
        rendered: Mapping[str, Any] | None = None
        if not pending or capture:
            rendered = render_current_observation(env)
        if not pending:
            assert rendered is not None
            noise_seed = continuation_noise_seed(
                base_noise_seed,
                episode_index,
                prefix_frame,
                replicate_index,
                replan_index,
            )
            chunk = np.asarray(
                policy.infer(dict(rendered), noise_seed=noise_seed), dtype=np.float32
            )
            if chunk.shape != (policy.action_horizon, 22):
                raise ValueError(f"Unexpected policy action chunk {chunk.shape}")
            predicted_chunks.append(np.asarray(chunk, dtype=np.float32).copy())
            pending.extend(chunk[: policy.replan_steps])
            noise_schedule.append(
                {
                    "replan_index": replan_index,
                    "global_frame": frame,
                    "noise_seed": noise_seed,
                }
            )
            replan_index += 1

        state_observation = rendered if rendered is not None else observation
        states.append(_state23(state_observation))
        if capture:
            assert rendered is not None
            captured_frames.append(frame)
            fronts.append(np.asarray(rendered["front"], dtype=np.uint8).copy())
            wrists.append(np.asarray(rendered["wrist"], dtype=np.uint8).copy())

        action = np.asarray(pending.popleft(), dtype=np.float32)
        actions.append(action)
        observation, _, terminated, truncated, final_info = env.step(
            fastwam_action_to_dexjoco(action)
        )
        succeeded = succeeded or bool(final_info.get("succeed", False))
        if terminated or truncated:
            break

    action_array = np.asarray(actions, dtype=np.float32).reshape(-1, 22)
    state_array = np.asarray(states, dtype=np.float32).reshape(-1, 23)
    chunk_array = (
        np.stack(predicted_chunks).astype(np.float32)
        if predicted_chunks
        else np.zeros((0, policy.action_horizon, 22), dtype=np.float32)
    )
    first_chunk = (
        np.asarray(predicted_chunks[0], dtype=np.float32)
        if predicted_chunks
        else np.zeros((policy.action_horizon, 22), dtype=np.float32)
    )
    return {
        "episode_index": int(episode_index),
        "prefix_frame": int(prefix_frame),
        "replicate_index": int(replicate_index),
        "success": bool(succeeded),
        "steps_executed": len(action_array),
        "final_global_frame_exclusive": int(prefix_frame + len(action_array)),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "hit_max_steps": bool(
            prefix_frame + len(action_array) == max_steps
            and not (terminated or truncated)
        ),
        "noise_scheme": NOISE_SCHEME,
        "noise_schedule": noise_schedule,
        "actions": action_array,
        "states": state_array,
        "predicted_chunks": chunk_array,
        "first_action_chunk": first_chunk,
        "capture_frames": np.asarray(captured_frames, dtype=np.int32),
        "fronts": np.asarray(fronts, dtype=np.uint8),
        "wrists": np.asarray(wrists, dtype=np.uint8),
        "final_progress": progress_metrics(env),
        "elapsed_s": float(time.perf_counter() - started),
    }


def _trajectory_public_row(result: Mapping[str, Any]) -> dict[str, Any]:
    excluded = {
        "actions",
        "states",
        "capture_frames",
        "fronts",
        "wrists",
        "predicted_chunks",
        "first_action_chunk",
    }
    return {key: value for key, value in result.items() if key not in excluded}


def save_trajectory_arrays(path: Path, result: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp.npz")
    schedule = list(result["noise_schedule"])
    np.savez_compressed(
        temporary,
        global_frames=np.arange(
            int(result["prefix_frame"]),
            int(result["final_global_frame_exclusive"]),
            dtype=np.int32,
        ),
        actions=np.asarray(result["actions"], dtype=np.float32),
        states=np.asarray(result["states"], dtype=np.float32),
        noise_replan_frames=np.asarray(
            [row["global_frame"] for row in schedule], dtype=np.int32
        ),
        noise_seeds=np.asarray(
            [row["noise_seed"] for row in schedule], dtype=np.int64
        ),
        succeeded=np.asarray(bool(result["success"])),
        terminated=np.asarray(bool(result["terminated"])),
        truncated=np.asarray(bool(result["truncated"])),
    )
    temporary.replace(path)


# Compatibility alias: every Pass@M replicate is persisted, success or failure.
save_success_trajectory = save_trajectory_arrays


def prefix_scan_should_stop(
    trajectory_rows: Sequence[Mapping[str, Any]], *, pass_m: int
) -> bool:
    """Stop only after all Pass@M trials have been recorded."""

    return len(trajectory_rows) >= int(pass_m)


def ordered_success_candidates(
    trajectory_rows: Sequence[Mapping[str, Any]], *, prefix_frame: int
) -> list[dict[str, Any]]:
    """Success replicates at ``prefix_frame``, lowest index first."""

    selected = [
        dict(row)
        for row in trajectory_rows
        if int(row["prefix_frame"]) == int(prefix_frame) and bool(row.get("success"))
    ]
    selected.sort(key=lambda row: int(row["replicate_index"]))
    return selected


def _compatible_cached_trajectory(
    row: Mapping[str, Any], *, run_signature: Mapping[str, Any], replicate_index: int
) -> bool:
    return bool(
        row.get("status") == "complete"
        and int(row.get("replicate_index", -1)) == replicate_index
        and row.get("run_signature") == run_signature
    )


def scan_prefix(
    env: Any,
    policy: Any,
    *,
    snapshot: tuple[np.ndarray, Mapping[str, Any]],
    attempt: Mapping[str, Any],
    prefix_frame: int,
    pass_m: int,
    base_noise_seed: int,
    max_steps: int,
    fps: int,
    output: Path,
    run_signature: Mapping[str, Any],
    overwrite: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    episode_index = int(attempt["saved_episode_index"])
    prefix_dir = output / "prefixes" / f"ep{episode_index:06d}_f{prefix_frame:04d}"
    trajectory_rows: list[dict[str, Any]] = []
    for replicate_index in range(pass_m):
        replicate_dir = prefix_dir / f"replicate_{replicate_index:02d}"
        row_path = replicate_dir / "trajectory.json"
        used_cache = False
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
            artifact = cached.get("trajectory_arrays")
            if not artifact or not Path(artifact).is_file():
                raise RuntimeError(f"Cached trajectory artifact is missing: {row_path}")
            if saved_success_rollout_videos_complete(cached):
                trajectory_rows.append(dict(cached))
                used_cache = True
            else:
                print(
                    f"[episode {episode_index}] prefix={prefix_frame} "
                    f"replicate={replicate_index} cache missing full success "
                    f"rollout RGB; re-running closed-loop",
                    flush=True,
                )
        if not used_cache:
            result = run_closed_loop_continuation(
                env,
                policy,
                snapshot=snapshot,
                episode_index=episode_index,
                prefix_frame=prefix_frame,
                replicate_index=replicate_index,
                base_noise_seed=base_noise_seed,
                max_steps=max_steps,
                capture_window=(prefix_frame, max_steps),
            )
            artifact = replicate_dir / "trajectory.npz"
            save_trajectory_arrays(artifact, result)
            video_meta = save_success_rollout_videos(
                replicate_dir, result, fps=fps
            )
            row = {
                **_trajectory_public_row(result),
                **video_meta,
                "format": "FoldGlassesRecoverabilityContinuation",
                "version": FORMAT_VERSION,
                "status": "complete",
                "seed": int(attempt["seed"]),
                "source_repeat": int(attempt["repeat"]),
                "source_failure_episode_index": episode_index,
                "trajectory_arrays": str(artifact.resolve()),
                "run_signature": dict(run_signature),
                "completed_at": utc_now(),
            }
            atomic_write_json(row_path, row)
            trajectory_rows.append(row)
        if prefix_scan_should_stop(trajectory_rows, pass_m=pass_m):
            break

    success_count = sum(bool(row["success"]) for row in trajectory_rows)
    replicates_evaluated = len(trajectory_rows)
    summary = {
        "format": "FoldGlassesRecoverabilityPrefixResult",
        "version": FORMAT_VERSION,
        "seed": int(attempt["seed"]),
        "seed_classification": str(attempt["seed_classification"]),
        "training_eligible": bool(attempt["training_eligible"]),
        "source_failure_episode_index": episode_index,
        "source_repeat": int(attempt["repeat"]),
        "prefix_frame": int(prefix_frame),
        "pass_m": int(pass_m),
        "replicates_evaluated": int(replicates_evaluated),
        "early_stop_on_first_success": False,
        "truncated_after_first_success": False,
        "full_pass_at_m": bool(replicates_evaluated >= int(pass_m)),
        "success_count": int(success_count),
        "success_rate": float(success_count / pass_m),
        "pass_at_m_hit": bool(success_count > 0),
        "successful_replicate_indices": [
            int(row["replicate_index"])
            for row in trajectory_rows
            if bool(row["success"])
        ],
        "trajectory_ledgers": [
            str(
                (
                    prefix_dir
                    / f"replicate_{int(row['replicate_index']):02d}"
                    / "trajectory.json"
                ).resolve()
            )
            for row in trajectory_rows
        ],
        "empirical_estimate_only": True,
        "run_signature": dict(run_signature),
        "completed_at": utc_now(),
    }
    atomic_write_json(prefix_dir / "summary.json", summary)
    return summary, trajectory_rows


def write_rgb_video(path: Path, frames: np.ndarray, fps: int) -> None:
    images = np.asarray(frames, dtype=np.uint8)
    if images.ndim != 4 or images.shape[-1] != 3 or len(images) == 0:
        raise ValueError(f"Expected non-empty [T,H,W,3] frames, got {images.shape}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp" + path.suffix)
    with av.open(str(temporary), mode="w") as container:
        stream = container.add_stream("libx264", rate=int(fps))
        stream.width = int(images.shape[2])
        stream.height = int(images.shape[1])
        stream.pix_fmt = "yuv420p"
        stream.options = {"preset": "veryfast", "crf": "23"}
        for image in images:
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    temporary.replace(path)


def save_event_arrays(
    path: Path,
    *,
    frame_indices: np.ndarray,
    actions: np.ndarray,
    states: np.ndarray,
) -> None:
    temporary = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(
        temporary,
        frame_indices=np.asarray(frame_indices, dtype=np.int32),
        actions=np.asarray(actions, dtype=np.float32),
        states=np.asarray(states, dtype=np.float32),
    )
    temporary.replace(path)


def continuation_video_paths(replicate_dir: Path) -> dict[str, Path]:
    return {
        "front": replicate_dir / "continuation_front.mp4",
        "wrist": replicate_dir / "continuation_wrist.mp4",
    }


def saved_success_rollout_videos_complete(row: Mapping[str, Any]) -> bool:
    """Failure caches need no RGB; success caches must have the full rollout videos."""

    if not bool(row.get("success")):
        return True
    front = row.get("continuation_front_video")
    wrist = row.get("continuation_wrist_video")
    return bool(
        front
        and wrist
        and Path(str(front)).is_file()
        and Path(str(wrist)).is_file()
    )


def save_success_rollout_videos(
    replicate_dir: Path,
    result: Mapping[str, Any],
    *,
    fps: int,
) -> dict[str, Any]:
    """Write the closed-loop success RGB. The pair event is cropped from this."""

    if not bool(result["success"]):
        return {}
    prefix_frame = int(result["prefix_frame"])
    actions = np.asarray(result["actions"])
    capture_frames = np.asarray(result["capture_frames"], dtype=np.int32)
    expected = np.arange(prefix_frame, prefix_frame + len(actions), dtype=np.int32)
    if capture_frames.shape != expected.shape or not np.array_equal(
        capture_frames, expected
    ):
        raise RuntimeError(
            "Closed-loop success RGB must cover every continuation frame "
            f"[{prefix_frame}, {prefix_frame + len(actions)}); "
            f"captured {len(capture_frames)}"
        )
    fronts = np.asarray(result["fronts"], dtype=np.uint8)
    wrists = np.asarray(result["wrists"], dtype=np.uint8)
    if len(fronts) != len(actions) or len(wrists) != len(actions):
        raise RuntimeError("Success rollout RGB length does not match actions")
    paths = continuation_video_paths(replicate_dir)
    write_rgb_video(paths["front"], fronts, fps)
    write_rgb_video(paths["wrist"], wrists, fps)
    return {
        "continuation_front_video": str(paths["front"].resolve()),
        "continuation_wrist_video": str(paths["wrist"].resolve()),
        "captured_frame_start": prefix_frame,
        "captured_frame_end_exclusive": int(prefix_frame + len(actions)),
        "success_event_source": SUCCESS_EVENT_SOURCE,
    }


def crop_counterfactual_success_event(
    *,
    event_start: int,
    event_end: int,
    prefix_frame: int,
    factual_actions: np.ndarray,
    factual_states: np.ndarray,
    factual_front: Mapping[int, np.ndarray],
    factual_wrist: Mapping[int, np.ndarray],
    continuation_actions: np.ndarray,
    continuation_states: np.ndarray,
    continuation_fronts: np.ndarray,
    continuation_wrists: np.ndarray,
) -> dict[str, np.ndarray]:
    """Crop ``[event_start, event_end)`` from a saved closed-loop success rollout.

    Frames before ``prefix_frame`` stay on the factual failure prefix. Frames at
    or after it come from the saved success continuation. This does not replay
    actions and does not re-infer the policy.
    """

    continuation_actions = np.asarray(continuation_actions, dtype=np.float32)
    continuation_states = np.asarray(continuation_states, dtype=np.float32)
    continuation_fronts = np.asarray(continuation_fronts, dtype=np.uint8)
    continuation_wrists = np.asarray(continuation_wrists, dtype=np.uint8)
    continuation_end = int(prefix_frame) + len(continuation_actions)
    materialized_end = min(int(event_end), continuation_end)
    if materialized_end <= int(prefix_frame):
        raise RuntimeError("Saved success rollout ended before the event branch")
    if materialized_end <= int(event_start):
        raise RuntimeError("Saved success rollout does not overlap the event window")
    frames = np.arange(int(event_start), materialized_end, dtype=np.int32)
    actions: list[np.ndarray] = []
    states: list[np.ndarray] = []
    fronts: list[np.ndarray] = []
    wrists: list[np.ndarray] = []
    for frame in frames:
        frame_i = int(frame)
        if frame_i < int(prefix_frame):
            actions.append(np.asarray(factual_actions[frame_i], dtype=np.float32))
            states.append(np.asarray(factual_states[frame_i, :23], dtype=np.float32))
            fronts.append(np.asarray(factual_front[frame_i], dtype=np.uint8))
            wrists.append(np.asarray(factual_wrist[frame_i], dtype=np.uint8))
            continue
        index = frame_i - int(prefix_frame)
        actions.append(np.asarray(continuation_actions[index], dtype=np.float32))
        states.append(np.asarray(continuation_states[index, :23], dtype=np.float32))
        fronts.append(np.asarray(continuation_fronts[index], dtype=np.uint8))
        wrists.append(np.asarray(continuation_wrists[index], dtype=np.uint8))
    return {
        "frame_indices": frames,
        "actions": np.stack(actions).astype(np.float32),
        "states": np.stack(states).astype(np.float32),
        "front": np.stack(fronts).astype(np.uint8),
        "wrist": np.stack(wrists).astype(np.uint8),
        "materialized_end": np.asarray(materialized_end, dtype=np.int32),
    }


def load_saved_success_rollout(
    row: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    payload = np.load(Path(str(row["trajectory_arrays"])))
    actions = np.asarray(payload["actions"], dtype=np.float32)
    states = np.asarray(payload["states"], dtype=np.float32)
    count = len(actions)
    front_map = read_video_frames(
        Path(str(row["continuation_front_video"])), range(count)
    )
    wrist_map = read_video_frames(
        Path(str(row["continuation_wrist_video"])), range(count)
    )
    fronts = np.stack([front_map[index] for index in range(count)])
    wrists = np.stack([wrist_map[index] for index in range(count)])
    return actions, states, fronts, wrists


def stitch_full_success_video(
    *,
    prefix_frame: int,
    source_path: Path,
    continuation: np.ndarray,
) -> np.ndarray:
    if prefix_frame <= 0:
        return np.asarray(continuation, dtype=np.uint8)
    pre = read_video_frames(source_path, range(prefix_frame))
    prefix_images = np.stack([pre[index] for index in range(prefix_frame)])
    return np.concatenate(
        [prefix_images, np.asarray(continuation, dtype=np.uint8)], axis=0
    )


def materialize_frontier_pair(
    env: Any,
    policy: Any,
    *,
    dataset: Path,
    actions: np.ndarray,
    recorded_states: np.ndarray,
    snapshot: tuple[np.ndarray, Mapping[str, Any]],
    attempt: Mapping[str, Any],
    frontier: Mapping[str, Any],
    successful_trajectories: Sequence[Mapping[str, Any]],
    base_noise_seed: int,
    max_steps: int,
    fps: int,
    output: Path,
    run_signature: Mapping[str, Any],
    overwrite: bool,
) -> dict[str, Any]:
    del env, policy, snapshot, base_noise_seed, max_steps
    episode_index = int(attempt["saved_episode_index"])
    frontier_id = str(frontier["frontier_id"])
    pair_dir = output / "event_pairs" / f"ep{episode_index:06d}_{frontier_id}"
    pair_path = pair_dir / "pair.json"
    candidates = [
        dict(row)
        for row in successful_trajectories
        if bool(row.get("success"))
    ]
    candidates.sort(key=lambda row: int(row["replicate_index"]))
    if not candidates:
        raise RuntimeError(
            f"Frontier {frontier_id} has no success candidates to materialize"
        )
    if pair_path.exists() and not overwrite:
        cached = read_json(pair_path)
        if cached.get("run_signature") != run_signature:
            raise RuntimeError(f"Cached pair is incompatible with this run: {pair_path}")
        if (
            cached.get("status") == "complete"
            and cached.get("success_event_source") == SUCCESS_EVENT_SOURCE
        ):
            return dict(cached)

    prefix_frame = int(frontier["last_recoverable_frame"])
    requested_start = int(frontier["event_start"])
    requested_end = int(frontier["event_end_exclusive"])
    skipped_missing_rollouts: list[int] = []
    successful_trajectory: Mapping[str, Any] | None = None
    for candidate in candidates:
        replicate_index = int(candidate["replicate_index"])
        if not saved_success_rollout_videos_complete(candidate):
            skipped_missing_rollouts.append(replicate_index)
            print(
                f"[episode {episode_index}] prefix={prefix_frame} "
                f"success replicate={replicate_index} has no saved full "
                f"success rollout RGB; trying next Pass@4 candidate",
                flush=True,
            )
            continue
        successful_trajectory = candidate
        break
    if successful_trajectory is None:
        pair = {
            "format": "FoldGlassesRecoverabilityEventPair",
            "version": FORMAT_VERSION,
            "status": "saved_success_rollout_missing",
            "success_event_source": SUCCESS_EVENT_SOURCE,
            "seed": int(attempt["seed"]),
            "source_failure_episode_index": episode_index,
            "frontier": dict(frontier),
            "successful_replicate_index": None,
            "tried_success_replicate_indices": [
                int(row["replicate_index"]) for row in candidates
            ],
            "skipped_missing_rollout_indices": skipped_missing_rollouts,
            "training_eligible": False,
            "run_signature": dict(run_signature),
            "completed_at": utc_now(),
        }
        atomic_write_json(pair_path, pair)
        return pair
    replicate_index = int(successful_trajectory["replicate_index"])
    (
        continuation_actions,
        continuation_states,
        continuation_fronts,
        continuation_wrists,
    ) = load_saved_success_rollout(successful_trajectory)

    source_paths = video_paths(dataset, episode_index)
    continuation_end = prefix_frame + len(continuation_actions)
    source_needed = list(range(requested_start, min(requested_end, continuation_end)))
    source_images = {
        camera: read_video_frames(path, source_needed)
        for camera, path in source_paths.items()
    }
    cropped = crop_counterfactual_success_event(
        event_start=requested_start,
        event_end=requested_end,
        prefix_frame=prefix_frame,
        factual_actions=actions,
        factual_states=recorded_states,
        factual_front=source_images["front"],
        factual_wrist=source_images["wrist"],
        continuation_actions=continuation_actions,
        continuation_states=continuation_states,
        continuation_fronts=continuation_fronts,
        continuation_wrists=continuation_wrists,
    )
    frames = np.asarray(cropped["frame_indices"], dtype=np.int32)
    materialized_end = int(cropped["materialized_end"])
    success_actions = np.asarray(cropped["actions"], dtype=np.float32)
    success_states = np.asarray(cropped["states"], dtype=np.float32)
    success_front = np.asarray(cropped["front"], dtype=np.uint8)
    success_wrist = np.asarray(cropped["wrist"], dtype=np.uint8)
    failure_front = np.stack(
        [source_images["front"][int(frame)] for frame in frames]
    )
    failure_wrist = np.stack(
        [source_images["wrist"][int(frame)] for frame in frames]
    )
    failure_actions = actions[requested_start:materialized_end]
    failure_states = recorded_states[requested_start:materialized_end, :23]
    expected_shape = (len(frames), 22)
    if success_actions.shape != expected_shape or failure_actions.shape != expected_shape:
        raise RuntimeError(
            f"Paired event action shape mismatch: success={success_actions.shape} "
            f"failure={failure_actions.shape} expected={expected_shape}"
        )

    pair_dir.mkdir(parents=True, exist_ok=True)
    failure_arrays = pair_dir / "failure_event.npz"
    success_arrays = pair_dir / "success_event.npz"
    save_event_arrays(
        failure_arrays,
        frame_indices=frames,
        actions=failure_actions,
        states=failure_states,
    )
    save_event_arrays(
        success_arrays,
        frame_indices=frames,
        actions=success_actions,
        states=success_states,
    )
    failure_front_video = pair_dir / "failure_front.mp4"
    failure_wrist_video = pair_dir / "failure_wrist.mp4"
    success_front_video = pair_dir / "success_front.mp4"
    success_wrist_video = pair_dir / "success_wrist.mp4"
    success_full_front_video = pair_dir / "success_full_front.mp4"
    success_full_wrist_video = pair_dir / "success_full_wrist.mp4"
    write_rgb_video(failure_front_video, failure_front, fps)
    write_rgb_video(failure_wrist_video, failure_wrist, fps)
    write_rgb_video(success_front_video, success_front, fps)
    write_rgb_video(success_wrist_video, success_wrist, fps)
    write_rgb_video(
        success_full_front_video,
        stitch_full_success_video(
            prefix_frame=prefix_frame,
            source_path=source_paths["front"],
            continuation=continuation_fronts,
        ),
        fps,
    )
    write_rgb_video(
        success_full_wrist_video,
        stitch_full_success_video(
            prefix_frame=prefix_frame,
            source_path=source_paths["wrist"],
            continuation=continuation_wrists,
        ),
        fps,
    )

    common = {
        "seed": int(attempt["seed"]),
        "source_failure_episode_index": episode_index,
        "frame_start": requested_start,
        "frame_end_exclusive": materialized_end,
        "requested_frame_end_exclusive": requested_end,
        "num_frames": len(frames),
        "exact_counterfactual_prefix_frame": prefix_frame,
        "frontier_first_zero_frame": int(frontier["first_zero_frame"]),
    }
    failure_descriptor = {
        "format": "FoldGlassesFactualFailureEvent",
        "version": FORMAT_VERSION,
        **common,
        "outcome": "failure",
        "action_loss": "disabled",
        "batch_role": "auxiliary",
        "arrays": str(failure_arrays.resolve()),
        "front_video": str(failure_front_video.resolve()),
        "wrist_video": str(failure_wrist_video.resolve()),
        "source_front_video": str(source_paths["front"].resolve()),
        "source_wrist_video": str(source_paths["wrist"].resolve()),
        "source_frame_interval": [requested_start, materialized_end],
        "source": "factual_gt_failure_rollout",
    }
    failure_descriptor_path = pair_dir / "failure_event.json"
    atomic_write_json(failure_descriptor_path, failure_descriptor)

    success_descriptor = {
        "format": "FoldGlassesCounterfactualSuccessEvent",
        "version": FORMAT_VERSION,
        **common,
        "outcome": "success",
        "action_loss": "enabled",
        "batch_role": "primary",
        "action_loss_window": [prefix_frame, prefix_frame + REPLAN_STEPS],
        "arrays": str(success_arrays.resolve()),
        "front_video": str(success_front_video.resolve()),
        "wrist_video": str(success_wrist_video.resolve()),
        "full_front_video": str(success_full_front_video.resolve()),
        "full_wrist_video": str(success_full_wrist_video.resolve()),
        "successful_continuation_arrays": successful_trajectory[
            "trajectory_arrays"
        ],
        "successful_continuation_ledger": str(
            (
                output
                / "prefixes"
                / f"ep{episode_index:06d}_f{prefix_frame:04d}"
                / f"replicate_{replicate_index:02d}"
                / "trajectory.json"
            ).resolve()
        ),
        "successful_replicate_index": replicate_index,
        "tried_success_replicate_indices": [
            int(row["replicate_index"]) for row in candidates
        ],
        "skipped_missing_rollout_indices": skipped_missing_rollouts,
        "t_frame": prefix_frame,
        "t_plus_24_frame": int(frontier["t_plus_24_frame"]),
        "noise_scheme": NOISE_SCHEME,
        "noise_schedule": list(successful_trajectory.get("noise_schedule") or []),
        "success_event_source": SUCCESS_EVENT_SOURCE,
        "deterministic_rerun_succeeded": True,
    }
    success_descriptor_path = pair_dir / "success_event.json"
    atomic_write_json(success_descriptor_path, success_descriptor)

    pair = {
        "format": "FoldGlassesRecoverabilityEventPair",
        "version": FORMAT_VERSION,
        "status": "complete",
        "pair_id": f"seed{int(attempt['seed'])}_ep{episode_index}_{frontier_id}",
        "seed": int(attempt["seed"]),
        "seed_classification": str(attempt["seed_classification"]),
        "source_failure_episode_index": episode_index,
        "source_repeat": int(attempt["repeat"]),
        "frontier": dict(frontier),
        "factual_failure_event": str(failure_descriptor_path.resolve()),
        "counterfactual_success_event": str(success_descriptor_path.resolve()),
        "successful_replicate_index": replicate_index,
        "tried_success_replicate_indices": [
            int(row["replicate_index"]) for row in candidates
        ],
        "skipped_missing_rollout_indices": skipped_missing_rollouts,
        "success_event_source": SUCCESS_EVENT_SOURCE,
        "training_eligible": bool(attempt["training_eligible"]),
        "evaluation_only": bool(attempt["evaluation_only"]),
        "selector_interpretation": (
            "empirical Pass@M recovery frontier; not absolute irreversibility"
        ),
        "run_signature": dict(run_signature),
        "completed_at": utc_now(),
    }
    atomic_write_json(pair_path, pair)
    return pair


def run_episode_scan(
    *,
    dataset: Path,
    output: Path,
    attempt: Mapping[str, Any],
    policy: Any,
    scan_frames: Sequence[int],
    pass_m: int,
    base_noise_seed: int,
    max_steps: int,
    state_atol: float,
    fps: int,
    event_expansion_blocks: int,
    run_signature: Mapping[str, Any],
    overwrite: bool,
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
    horizon = recorded_horizon(actions, recorded_states, max_steps=max_steps)
    episode_frames = clip_scan_frames(scan_frames, horizon=horizon)
    episode_dir = output / "episodes" / f"ep{episode_index:06d}"
    if not episode_frames:
        episode_summary = {
            "format": "FoldGlassesRecoverabilityEpisodeScan",
            "version": FORMAT_VERSION,
            "status": "skipped_short_episode",
            "seed": int(attempt["seed"]),
            "seed_classification": str(attempt["seed_classification"]),
            "source_failure_episode_index": episode_index,
            "recorded_steps": horizon,
            "max_steps": int(max_steps),
            "num_scan_points": 0,
            "num_complete_event_pairs": 0,
            "run_signature": dict(run_signature),
            "completed_at": utc_now(),
        }
        atomic_write_json(episode_dir / "summary.json", episode_summary)
        return [], [], episode_summary
    _, env = create_environment(int(attempt["seed"]), task_name=task_name)
    try:
        snapshots, factual_gate = prepare_factual_snapshots(
            env,
            actions=actions,
            recorded_states=recorded_states,
            attempt=canonical_attempt,
            scan_frames=episode_frames,
            max_steps=max_steps,
            state_atol=state_atol,
        )
        atomic_write_json(episode_dir / "factual_replay_gate.json", factual_gate)

        prefix_rows: list[dict[str, Any]] = []
        trajectory_rows: list[dict[str, Any]] = []
        for position, prefix_frame in enumerate(episode_frames, start=1):
            print(
                f"[episode {episode_index} {position}/{len(episode_frames)}] "
                f"prefix={prefix_frame} M={pass_m}",
                flush=True,
            )
            prefix, trajectories = scan_prefix(
                env,
                policy,
                snapshot=snapshots[prefix_frame],
                attempt=attempt,
                prefix_frame=prefix_frame,
                pass_m=pass_m,
                base_noise_seed=base_noise_seed,
                max_steps=max_steps,
                fps=fps,
                output=output,
                run_signature=run_signature,
                overwrite=overwrite,
            )
            prefix_rows.append(prefix)
            trajectory_rows.extend(trajectories)
            if int(prefix["success_count"]) == 0:
                print(
                    f"[episode {episode_index}] first 0/{pass_m} at prefix={prefix_frame}",
                    flush=True,
                )
                break

        frontiers = find_recoverability_frontiers(
            prefix_rows,
            block_size=int(policy.replan_steps),
            expansion_blocks=event_expansion_blocks,
            max_steps=max_steps,
        )
        pair_rows: list[dict[str, Any]] = []
        for frontier in frontiers:
            recoverable = int(frontier["last_recoverable_frame"])
            if not training_pair_eligible(attempt, frontier, pass_m=pass_m):
                pair_rows.append(
                    {
                        "format": "FoldGlassesRecoverabilityEventPair",
                        "version": FORMAT_VERSION,
                        "status": "training_ineligible_prefix",
                        "seed": int(attempt["seed"]),
                        "source_failure_episode_index": episode_index,
                        "frontier": dict(frontier),
                        "training_eligible": False,
                        "evaluation_only": True,
                        "reason": "short_or_censored_event_window",
                        "run_signature": dict(run_signature),
                    }
                )
                continue
            candidates = ordered_success_candidates(
                trajectory_rows, prefix_frame=recoverable
            )
            if not candidates:
                raise RuntimeError(
                    f"Frontier {frontier['frontier_id']} has no saved success replicate"
                )
            pair_rows.append(
                materialize_frontier_pair(
                    env,
                    policy,
                    dataset=dataset,
                    actions=actions,
                    recorded_states=recorded_states,
                    snapshot=snapshots[recoverable],
                    attempt=attempt,
                    frontier=frontier,
                    successful_trajectories=candidates,
                    base_noise_seed=base_noise_seed,
                    max_steps=max_steps,
                    fps=fps,
                    output=output,
                    run_signature=run_signature,
                    overwrite=overwrite,
                )
            )

        episode_summary = {
            "format": "FoldGlassesRecoverabilityEpisodeScan",
            "version": FORMAT_VERSION,
            "seed": int(attempt["seed"]),
            "seed_classification": str(attempt["seed_classification"]),
            "training_eligible": bool(attempt["training_eligible"]),
            "source_failure_episode_index": episode_index,
            "source_repeat": int(attempt["repeat"]),
            "num_scan_points": len(prefix_rows),
            "num_empirical_frontiers": len(frontiers),
            "frontiers": frontiers,
            "num_complete_event_pairs": sum(
                row.get("status") == "complete" for row in pair_rows
            ),
            "factual_replay_gate": str(
                (episode_dir / "factual_replay_gate.json").resolve()
            ),
            "empirical_frontier_only": True,
            "absolute_irreversibility_claimed": False,
            "run_signature": dict(run_signature),
            "completed_at": utc_now(),
        }
        atomic_write_json(episode_dir / "summary.json", episode_summary)
        return prefix_rows, pair_rows, episode_summary
    finally:
        env.close()


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--collection-summary", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--episode-indices",
        default="",
        help="Optional comma-separated failure episodes; still deduplicated by seed.",
    )
    parser.add_argument("--seeds", default="", help="Optional comma-separated seeds.")
    parser.add_argument(
        "--hard-transfer-seeds",
        default="",
        help="Unused compatibility flag; all-failure seeds are now training-eligible.",
    )
    parser.add_argument(
        "--scan-frames",
        default="",
        help="Optional comma-separated exact-prefix frames; otherwise use the grid.",
    )
    parser.add_argument("--scan-start", type=int, default=DEFAULT_SCAN_START)
    parser.add_argument("--scan-end", type=int, default=0)
    parser.add_argument("--scan-stride", type=int, default=24)
    parser.add_argument("--pass-m", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--action-horizon", type=int, default=32)
    parser.add_argument("--replan-steps", type=int, default=24)
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--event-expansion-blocks", type=int, default=1)
    parser.add_argument("--base-noise-seed", type=int, default=20260813)
    parser.add_argument("--state-atol", type=float, default=2e-4)
    parser.add_argument("--fps", type=int, default=30)
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
    if not 1 <= args.replan_steps <= args.action_horizon:
        raise ValueError("--replan-steps must lie within --action-horizon")
    if args.event_expansion_blocks < 0 or args.state_atol < 0.0 or args.fps <= 0:
        raise ValueError("Invalid event expansion, factual threshold, or FPS")
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
    hard_transfer = set(parse_ints(args.hard_transfer_seeds))
    selected, selection_audit = select_one_failure_per_seed(
        collection.get("attempt_log", []),
        preferred_episode_indices=preferred,
        seed_filter=seed_filter,
        hard_transfer_seeds=hard_transfer,
    )
    atomic_write_jsonl(output / "seed_selection.jsonl", selection_audit)
    if not selected:
        raise ValueError("No eligible failure episodes were selected")
    shard_world = int(args.shard_world)
    shard_rank = int(args.shard_rank)
    if shard_world < 1 or not 0 <= shard_rank < shard_world:
        raise ValueError(
            f"Invalid shard: rank={shard_rank} world={shard_world}"
        )
    if shard_world > 1:
        selected = selected[shard_rank::shard_world]
        print(
            f"shard {shard_rank}/{shard_world}: "
            f"{len(selected)} failure episodes",
            flush=True,
        )
        if not selected:
            atomic_write_json(
                output / "summary.json",
                {
                    "format": "FoldGlassesFailureRecoverabilityFrontierScan",
                    "version": FORMAT_VERSION,
                    "status": "complete",
                    "shard_rank": shard_rank,
                    "shard_world": shard_world,
                    "num_selected_failure_episodes": 0,
                    "num_complete_event_pairs": 0,
                },
            )
            return 0

    run_signature = {
        "format_version": FORMAT_VERSION,
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
        "scan_frames": scan_frames,
        "base_noise_seed": int(args.base_noise_seed),
        "noise_scheme": NOISE_SCHEME,
        "event_expansion_blocks": int(args.event_expansion_blocks),
    }
    atomic_write_json(
        output / "config.json",
        {
            **run_signature,
            "selected_failure_episodes": [
                int(row["saved_episode_index"]) for row in selected
            ],
            "hard_transfer_seeds": sorted(hard_transfer),
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
    all_pair_rows: list[dict[str, Any]] = []
    episode_summaries: list[dict[str, Any]] = []
    started = time.perf_counter()
    for position, attempt in enumerate(selected, start=1):
        print(
            f"[selection {position}/{len(selected)}] seed={attempt['seed']} "
            f"episode={attempt['saved_episode_index']} "
            f"class={attempt['seed_classification']}",
            flush=True,
        )
        prefix_rows, pair_rows, episode_summary = run_episode_scan(
            dataset=dataset,
            output=output,
            attempt=attempt,
            policy=policy,
            scan_frames=scan_frames,
            pass_m=int(args.pass_m),
            base_noise_seed=int(args.base_noise_seed),
            max_steps=int(args.max_steps),
            state_atol=float(args.state_atol),
            fps=int(args.fps),
            event_expansion_blocks=int(args.event_expansion_blocks),
            run_signature=run_signature,
            overwrite=bool(args.overwrite),
            task_name=str(args.task_name),
        )
        all_prefix_rows.extend(prefix_rows)
        all_pair_rows.extend(pair_rows)
        episode_summaries.append(episode_summary)
        atomic_write_jsonl(output / "prefix_results.jsonl", all_prefix_rows)
        atomic_write_jsonl(output / "event_pair_manifest.jsonl", all_pair_rows)

    summary = {
        "format": "FoldGlassesFailureRecoverabilityFrontierScan",
        "version": FORMAT_VERSION,
        "status": "complete",
        "dataset": str(dataset),
        "num_selected_failure_episodes": len(selected),
        "num_mixed_seed_episodes": sum(
            row["seed_classification"] == "mixed" for row in selected
        ),
        "num_hard_transfer_episodes": sum(
            row["seed_classification"] == "all_failure" for row in selected
        ),
        "num_prefix_results": len(all_prefix_rows),
        "num_empirical_frontiers": sum(
            int(row["num_empirical_frontiers"]) for row in episode_summaries
        ),
        "num_complete_event_pairs": sum(
            row.get("status") == "complete" for row in all_pair_rows
        ),
        "num_training_eligible_pairs": sum(
            row.get("status") == "complete" and bool(row.get("training_eligible"))
            for row in all_pair_rows
        ),
        "seed_selection": str((output / "seed_selection.jsonl").resolve()),
        "prefix_results": str((output / "prefix_results.jsonl").resolve()),
        "event_pair_manifest": str(
            (output / "event_pair_manifest.jsonl").resolve()
        ),
        "policy_load_count": 1,
        "empirical_frontier_only": True,
        "absolute_irreversibility_claimed": False,
        "run_signature": run_signature,
        "elapsed_s": float(time.perf_counter() - started),
        "completed_at": utc_now(),
    }
    atomic_write_json(output / "summary.json", summary)
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
