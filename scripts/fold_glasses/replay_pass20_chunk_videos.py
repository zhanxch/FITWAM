#!/usr/bin/env python3
"""Replay Pass@20 first-action chunks in DexJoco and measure video S/U/AUROC.

No policy inference. For each completed prefix with 20 saved chunks:
restore the GT snapshot, execute chunk[:24], render front+wrist, resize to
224, then compute the same prefix-conditional metrics used for actions.

Writes outside the scan shards so a running Pass@20 job is not touched.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analysis.pass20_future_metrics import branch_metrics
from scripts.fold_glasses.scan_failure_recoverability_frontier import (
    FORMAT_VERSION as FRONTIER_FORMAT_VERSION,
    atomic_write_json,
    prepare_factual_snapshots,
    utc_now,
)
from scripts.fold_glasses.validate_factual_replay import (
    attempt_for_episode,
    create_environment,
    load_episode,
    read_json,
    render_current_observation,
    setup_paths,
)


FORMAT_VERSION = "1.0"
PASS_M = 20
REPLAN_STEPS = 24
IMAGE_SIZE = 224
ARCHIVE_SIZE = 64
PREFIX_DIR_RE = re.compile(r"ep(\d+)_f(\d+)$")


def resize_rgb(image: np.ndarray, size: int) -> np.ndarray:
    image = np.asarray(image, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected HWC RGB, got {image.shape}")
    if image.shape[0] == size and image.shape[1] == size:
        return image
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)


def cameras_224(observation: Mapping[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    return resize_rgb(observation["front"], IMAGE_SIZE), resize_rgb(
        observation["wrist"], IMAGE_SIZE
    )


def parse_prefix_dir(path: Path) -> tuple[int, int] | None:
    match = PREFIX_DIR_RE.search(path.name)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def prefix_is_mixed(prefix_dir: Path, *, pass_m: int = PASS_M) -> bool:
    summary_path = prefix_dir / "summary.json"
    if summary_path.is_file():
        summary = read_json(summary_path)
        n_ok = int(summary.get("success_count", -1))
        n_eval = int(summary.get("replicates_evaluated", pass_m))
        return 0 < n_ok < n_eval
    n_ok = 0
    for replicate in range(pass_m):
        row = read_json(prefix_dir / f"replicate_{replicate:02d}" / "trajectory.json")
        n_ok += int(bool(row["ledger"]["success"]))
    return 0 < n_ok < pass_m


def prefix_complete(prefix_dir: Path, *, pass_m: int = PASS_M) -> bool:
    for replicate in range(pass_m):
        folder = prefix_dir / f"replicate_{replicate:02d}"
        if not (folder / "trajectory.json").is_file():
            return False
        if not (folder / "action_chunk.npz").is_file():
            return False
    return True


def discover_ready_prefixes(scan_root: Path, *, pass_m: int = PASS_M) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    for prefix_dir in sorted(scan_root.glob("shard*/prefixes/ep*_f*")):
        parsed = parse_prefix_dir(prefix_dir)
        if parsed is None or not prefix_dir.is_dir():
            continue
        if not prefix_complete(prefix_dir, pass_m=pass_m):
            continue
        episode_index, prefix_frame = parsed
        first = read_json(prefix_dir / "replicate_00" / "trajectory.json")
        found.append(
            {
                "prefix_dir": prefix_dir,
                "episode_index": episode_index,
                "prefix_frame": prefix_frame,
                "seed": int(first["seed"]),
                "repeat": int(first["source_repeat"]),
                "shard": prefix_dir.parts[-3],
            }
        )
    found.sort(key=lambda row: (int(row["episode_index"]), int(row["prefix_frame"])))
    return found


def dataset_from_scan(scan_root: Path) -> Path:
    for config_path in sorted(scan_root.glob("shard*/config.json")):
        payload = read_json(config_path)
        dataset = Path(payload["dataset"])
        if dataset.is_dir():
            return dataset
    raise FileNotFoundError(f"No shard config with a dataset under {scan_root}")


def output_prefix_dir(output: Path, episode_index: int, prefix_frame: int) -> Path:
    return output / f"ep{episode_index:06d}_f{prefix_frame:04d}"


def metrics_done(output: Path, episode_index: int, prefix_frame: int) -> bool:
    path = output_prefix_dir(output, episode_index, prefix_frame) / "metrics.json"
    if not path.is_file():
        return False
    try:
        payload = read_json(path)
    except json.JSONDecodeError:
        return False
    return payload.get("format") == "FoldGlassesPassAtKChunkVideoReplay" and payload.get(
        "status"
    ) == "complete"


def load_prefix_chunks(prefix_dir: Path, *, pass_m: int = PASS_M) -> dict[str, Any]:
    chunks: list[np.ndarray] = []
    success: list[bool] = []
    for replicate in range(pass_m):
        folder = prefix_dir / f"replicate_{replicate:02d}"
        arrays = np.load(folder / "action_chunk.npz")
        chunk = np.asarray(arrays["first_action_chunk"], dtype=np.float32)
        if chunk.shape != (32, 22):
            raise ValueError(f"Unexpected chunk shape {chunk.shape} in {folder}")
        ledger = read_json(folder / "trajectory.json")["ledger"]
        chunks.append(chunk)
        success.append(bool(ledger["success"]))
    return {
        "chunk": np.stack(chunks, axis=0),
        "success": np.asarray(success, dtype=bool),
    }


def replay_chunk_video(
    env: Any,
    *,
    snapshot: tuple[np.ndarray, dict[str, Any]],
    chunk: np.ndarray,
    convert,
    steps: int = REPLAN_STEPS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from scripts.fold_glasses.run_seedpair_block_interventions import (
        restore_integration_state,
    )

    restore_integration_state(env, snapshot[0], snapshot[1])
    start = render_current_observation(env)
    start_front, start_wrist = cameras_224(start)
    fronts: list[np.ndarray] = []
    wrists: list[np.ndarray] = []
    env.unwrapped.image_obs = False
    for frame in range(steps):
        env.step(convert(chunk[frame]))
        rendered = render_current_observation(env)
        front, wrist = cameras_224(rendered)
        fronts.append(front)
        wrists.append(wrist)
    return (
        start_front,
        start_wrist,
        np.stack(fronts, axis=0),
        np.stack(wrists, axis=0),
    )


def stack_videos(fronts: np.ndarray, wrists: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [
            fronts.reshape(len(fronts), -1).astype(np.float32),
            wrists.reshape(len(wrists), -1).astype(np.float32),
        ],
        axis=1,
    ) / 255.0


def downsample_stack(frames: np.ndarray, size: int) -> np.ndarray:
    out = np.empty((len(frames), size, size, 3), dtype=np.uint8)
    for index, frame in enumerate(frames):
        out[index] = resize_rgb(frame, size)
    return out


def scan_shards_complete(scan_root: Path, world: int = 4) -> bool:
    return all((scan_root / f"shard{rank}" / "summary.json").is_file() for rank in range(world))


def replay_episode(
    *,
    env: Any,
    dataset: Path,
    attempt: Mapping[str, Any],
    prefixes: list[dict[str, Any]],
    output: Path,
    convert,
    max_steps: int,
    state_atol: float,
    save_archive: bool,
) -> int:
    episode_index = int(attempt["saved_episode_index"])
    actions, recorded_states = load_episode(dataset, episode_index)
    frames = [int(row["prefix_frame"]) for row in prefixes]
    snapshots, gate = prepare_factual_snapshots(
        env,
        actions=actions,
        recorded_states=recorded_states,
        attempt=attempt,
        scan_frames=frames,
        max_steps=max_steps,
        state_atol=state_atol,
    )
    written = 0
    for row in prefixes:
        prefix_frame = int(row["prefix_frame"])
        if metrics_done(output, episode_index, prefix_frame):
            continue
        payload = load_prefix_chunks(row["prefix_dir"])
        start_fronts: list[np.ndarray] = []
        start_wrists: list[np.ndarray] = []
        post_fronts: list[np.ndarray] = []
        post_wrists: list[np.ndarray] = []
        started = time.perf_counter()
        for replicate, chunk in enumerate(payload["chunk"]):
            start_front, start_wrist, fronts, wrists = replay_chunk_video(
                env,
                snapshot=snapshots[prefix_frame],
                chunk=chunk,
                convert=convert,
            )
            start_fronts.append(start_front)
            start_wrists.append(start_wrist)
            post_fronts.append(fronts)
            post_wrists.append(wrists)
        front = np.stack(post_fronts, axis=0)
        wrist = np.stack(post_wrists, axis=0)
        start_front_arr = np.stack(start_fronts, axis=0)
        start_wrist_arr = np.stack(start_wrists, axis=0)
        start_max_abs = float(
            max(
                np.max(np.abs(start_front_arr.astype(np.int16) - start_front_arr[:1])),
                np.max(np.abs(start_wrist_arr.astype(np.int16) - start_wrist_arr[:1])),
            )
        )
        video_metrics = branch_metrics(stack_videos(front, wrist), payload["success"])
        dest = output_prefix_dir(output, episode_index, prefix_frame)
        dest.mkdir(parents=True, exist_ok=True)
        if save_archive:
            np.savez_compressed(
                dest / "archive_64.npz",
                front=np.stack([downsample_stack(item, ARCHIVE_SIZE) for item in front]),
                wrist=np.stack([downsample_stack(item, ARCHIVE_SIZE) for item in wrist]),
                success=payload["success"].astype(np.uint8),
            )
        atomic_write_json(
            dest / "metrics.json",
            {
                "format": "FoldGlassesPassAtKChunkVideoReplay",
                "version": FORMAT_VERSION,
                "status": "complete",
                "episode_index": episode_index,
                "prefix_frame": prefix_frame,
                "seed": int(row["seed"]),
                "repeat": int(row["repeat"]),
                "source_prefix_dir": str(row["prefix_dir"].resolve()),
                "factual_replay_passed": bool(gate["factual_replay_passed"]),
                "image_size": IMAGE_SIZE,
                "post_frames": REPLAN_STEPS,
                "frontier_format_version": FRONTIER_FORMAT_VERSION,
                "start_obs_max_abs": start_max_abs,
                "success": [bool(flag) for flag in payload["success"].tolist()],
                "elapsed_s": float(time.perf_counter() - started),
                "completed_at": utc_now(),
                **video_metrics,
            },
        )
        written += 1
        print(
            f"[video-replay] ep={episode_index} prefix={prefix_frame} "
            f"S={video_metrics['s']:.4f} U={video_metrics['u']:.4f} "
            f"AUROC={video_metrics['auroc']:.3f} start_abs={start_max_abs:.3g}",
            flush=True,
        )
    return written


def group_by_episode(rows: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["episode_index"])].append(row)
    return grouped


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan-root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard-rank", type=int, default=0)
    parser.add_argument("--shard-world", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--state-atol", type=float, default=2e-4)
    parser.add_argument("--task-name", default="fold_glasses")
    parser.add_argument("--pass-m", type=int, default=PASS_M)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--watch-seconds", type=int, default=90)
    parser.add_argument("--scan-shards", type=int, default=4)
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--episode-indices", default="")
    parser.add_argument("--prefix-frames", default="")
    parser.add_argument("--mixed-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--no-archive", action="store_true")
    return parser.parse_args(argv)


def selected_episodes(raw: str) -> set[int] | None:
    values = {int(item.strip()) for item in raw.split(",") if item.strip()}
    return values or None


def pending_for_rank(
    rows: list[dict[str, Any]],
    *,
    output: Path,
    rank: int,
    world: int,
    episode_filter: set[int] | None,
    prefix_filter: set[int] | None,
    mixed_only: bool,
) -> dict[int, list[dict[str, Any]]]:
    grouped = group_by_episode(rows)
    pending: dict[int, list[dict[str, Any]]] = {}
    for episode_index, prefixes in grouped.items():
        if episode_index % world != rank:
            continue
        if episode_filter is not None and episode_index not in episode_filter:
            continue
        remain = [
            row
            for row in prefixes
            if not metrics_done(output, episode_index, int(row["prefix_frame"]))
            and (prefix_filter is None or int(row["prefix_frame"]) in prefix_filter)
            and (not mixed_only or prefix_is_mixed(row["prefix_dir"]))
        ]
        if remain:
            pending[episode_index] = remain
    return pending


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    scan_root = args.scan_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    dataset = (
        args.dataset.expanduser().resolve()
        if args.dataset is not None
        else dataset_from_scan(scan_root)
    )
    episode_filter = selected_episodes(args.episode_indices)
    prefix_filter = selected_episodes(args.prefix_frames)
    rank = int(args.shard_rank)
    world = int(args.shard_world)
    if world < 1 or not 0 <= rank < world:
        raise ValueError(f"Invalid shard rank={rank} world={world}")

    setup_paths()
    from fastwam_dexjoco.policy import fastwam_action_to_dexjoco

    idle_rounds = 0
    while True:
        ready = discover_ready_prefixes(scan_root, pass_m=int(args.pass_m))
        pending = pending_for_rank(
            ready,
            output=output,
            rank=rank,
            world=world,
            episode_filter=episode_filter,
            prefix_filter=prefix_filter,
            mixed_only=bool(args.mixed_only),
        )
        episode_ids = sorted(pending)
        if args.max_episodes > 0:
            episode_ids = episode_ids[: int(args.max_episodes)]
        print(
            f"[video-replay] rank={rank}/{world} ready_prefixes={len(ready)} "
            f"pending_episodes={len(episode_ids)}",
            flush=True,
        )
        for episode_index in episode_ids:
            prefixes = pending[episode_index]
            attempt = attempt_for_episode(dataset, episode_index)
            print(
                f"[video-replay] episode={episode_index} prefixes="
                f"{[int(row['prefix_frame']) for row in prefixes]}",
                flush=True,
            )
            _, env = create_environment(int(attempt["seed"]), task_name=str(args.task_name))
            try:
                replay_episode(
                    env=env,
                    dataset=dataset,
                    attempt=attempt,
                    prefixes=prefixes,
                    output=output,
                    convert=fastwam_action_to_dexjoco,
                    max_steps=int(args.max_steps),
                    state_atol=float(args.state_atol),
                    save_archive=not bool(args.no_archive),
                )
            finally:
                env.close()
            if args.max_episodes > 0:
                break

        if not args.watch:
            break
        if not pending and scan_shards_complete(scan_root, int(args.scan_shards)):
            idle_rounds += 1
            if idle_rounds >= 2:
                print("[video-replay] scan complete and no pending prefixes", flush=True)
                break
        else:
            idle_rounds = 0
        time.sleep(max(int(args.watch_seconds), 15))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
