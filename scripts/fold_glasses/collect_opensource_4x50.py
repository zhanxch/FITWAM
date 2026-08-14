#!/usr/bin/env python3
"""Opensource-aligned fold_glasses 4×50 rollout collection → LeRobot shards.

Uses FastWAM-infer-in-DexJoco FastWAMDexJocoPolicy (224 / z-score / replan=24),
not the local async server / s0_bundle path.

Protocol: seeds [seed_start, seed_end] × repeats (default 10086..10135 × 4 = 200).
Shards seeds across GPUs; each GPU writes its own LeRobot shard; parent merges.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import deque
import multiprocessing as mp
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OPEN = Path(os.environ.get("FASTWAM_OPEN_REPO", "/data_all/xiangchengzhan/FastWAM-infer-in-DexJoco"))
FASTWAM_PIN = Path(os.environ.get("FASTWAM_PIN", str(ROOT / "third_party/FastWAM_pin_45d8e14")))
DEXJOCO = ROOT / "third_party" / "dexjoco" / "dexjoco"
EXPECTED_PIN = "45d8e1458921d83f8ad6cf9ce993d371208dabd0"

DEFAULT_CKPT = OPEN / "checkpoints/fold_glasses/step_010000.pt"
DEFAULT_CFG = OPEN / "configs/fastwam_dexjoco.yaml"
DEFAULT_STATS = OPEN / "artifacts/fold_glasses/dataset_stats.json"
DEFAULT_TEXT = (
    OPEN
    / "artifacts/fold_glasses"
    / "0c3367ce1d74848cc46b93c6d2eee5e2097dca410a2c95f3da48bd8c8673fa20.t5_len128.wan22ti2v5b.pt"
)
DEFAULT_SOURCE = ROOT / "data/dexjoco/dexjoco_lerobot_datasets/fold_glasses"
SUCCESS_PROMPT = "Fold the glasses and place them into the case."


def _setup_paths() -> None:
    paths = [
        str(OPEN / "src"),
        str(FASTWAM_PIN / "src"),
        str(DEXJOCO),
        str(ROOT / "scripts"),
        str(ROOT / "src"),
    ]
    for p in reversed(paths):
        if p in sys.path:
            sys.path.remove(p)
        sys.path.insert(0, p)


def _parse_gpus(raw: str) -> list[int]:
    return [int(x) for x in raw.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gpus", type=str, default="0,1,2,3,4,5,6,7")
    p.add_argument("--seed-start", type=int, default=10086)
    p.add_argument("--seed-end", type=int, default=10135)
    p.add_argument("--repeats", type=int, default=4)
    p.add_argument("--max-steps", type=int, default=1200)
    p.add_argument("--action-horizon", type=int, default=32)
    p.add_argument("--replan-steps", type=int, default=24)
    p.add_argument("--num-inference-steps", type=int, default=10)
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    p.add_argument("--model-config", type=Path, default=DEFAULT_CFG)
    p.add_argument("--dataset-stats", type=Path, default=DEFAULT_STATS)
    p.add_argument("--text-embedding", type=Path, default=DEFAULT_TEXT)
    p.add_argument("--source-dataset", type=Path, default=DEFAULT_SOURCE)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--task-name", default="fold_glasses")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--skip-pin-check", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def assert_pin() -> None:
    import subprocess

    head = subprocess.check_output(
        ["git", "-C", str(FASTWAM_PIN), "rev-parse", "HEAD"], text=True
    ).strip()
    if head != EXPECTED_PIN:
        raise SystemExit(f"FastWAM pin mismatch: {head}")


def _safe_rgb(img: np.ndarray) -> np.ndarray:
    x = np.asarray(img)
    if x.dtype != np.uint8:
        x = np.clip(x, 0, 255).astype(np.uint8)
    if x.ndim == 3 and x.shape[0] in (1, 3) and x.shape[-1] not in (1, 3):
        x = np.transpose(x, (1, 2, 0))
    if x.ndim == 2:
        x = np.stack([x] * 3, axis=-1)
    if x.shape[-1] == 1:
        x = np.concatenate([x] * 3, axis=-1)
    return np.ascontiguousarray(x[..., :3])


def rollout_episode(env: Any, policy: Any, *, seed: int, repeat: int, max_steps: int) -> dict[str, Any]:
    from fastwam_dexjoco.policy import fastwam_action_to_dexjoco

    obs, _ = env.reset()
    pending: deque[np.ndarray] = deque()
    actions: list[np.ndarray] = []
    states: list[np.ndarray] = []
    fronts: list[np.ndarray] = []
    wrists: list[np.ndarray] = []
    replan_index = 0
    success = False
    t0 = time.perf_counter()

    for _step in range(max_steps):
        if not pending:
            noise_seed = seed * 100_000 + repeat * 1_000 + replan_index
            chunk = policy.infer(obs, noise_seed=noise_seed)
            pending.extend(np.asarray(a, dtype=np.float32) for a in chunk[: policy.replan_steps])
            replan_index += 1

        state = np.asarray(obs["state"], dtype=np.float32).reshape(-1)
        if state.shape[0] > 23:
            state = state[:23]
        action = pending.popleft()
        actions.append(np.asarray(action, dtype=np.float32))
        states.append(state.astype(np.float32))
        fronts.append(_safe_rgb(obs["front"]))
        wrists.append(_safe_rgb(obs["wrist"]))

        obs, _, terminated, truncated, info = env.step(fastwam_action_to_dexjoco(action))
        success = bool(info.get("succeed", False))
        if terminated or truncated:
            break

    return {
        "actions": np.stack(actions, axis=0),
        "states": np.stack(states, axis=0),
        "frames": {
            "observation.images.front": fronts,
            "observation.images.wrist": wrists,
        },
        "success": bool(success),
        "seed": int(seed),
        "repeat": int(repeat),
        "steps": len(actions),
        "elapsed_s": float(time.perf_counter() - t0),
    }


def _worker_entry(gpu_id: int, seeds: list[int], args_dict: dict[str, Any]) -> None:
    """Top-level spawn entry (must be picklable)."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    worker_main(gpu_id, seeds, args_dict)


def worker_main(gpu_id: int, seeds: list[int], args_dict: dict[str, Any]) -> None:
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("DIFFSYNTH_SKIP_DOWNLOAD", "true")
    os.environ.setdefault("DIFFSYNTH_MODEL_BASE_PATH", str(ROOT / "checkpoints"))
    _setup_paths()

    import torch
    from dexjoco.tasks import CONFIG_MAPPING
    from fastwam_dexjoco.policy import FastWAMDexJocoPolicy

    from collect_dexjoco_rollouts import (
        OUTCOME_LEDGER_NAME,
        aggregate_stats,
        append_jsonl,
        make_outcome_row,
        prepare_dataset,
        save_lerobot_episode,
        serialize_dict,
        update_info,
        write_json,
    )

    args = argparse.Namespace(**args_dict)
    shard_dir = Path(args.output_dir) / "shards" / f"gpu_{gpu_id}"
    shard_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.output_dir) / "logs" / f"gpu_{gpu_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(msg: str) -> None:
        line = f"[collect gpu{gpu_id} {time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    if not seeds:
        log("no seeds assigned; exit")
        return

    torch.cuda.set_device(0 if os.environ.get("CUDA_VISIBLE_DEVICES") else gpu_id)
    # When parent sets CUDA_VISIBLE_DEVICES=<gpu>, local device is cuda:0.
    device = "cuda:0"
    log(f"loading policy on {device}; seeds={seeds}")
    t_load = time.perf_counter()
    policy = FastWAMDexJocoPolicy(
        model_config=args.model_config,
        checkpoint=args.checkpoint,
        dataset_stats=args.dataset_stats,
        text_embedding=args.text_embedding,
        device=device,
        action_horizon=args.action_horizon,
        replan_steps=args.replan_steps,
        num_inference_steps=args.num_inference_steps,
    )
    log(f"policy ready in {time.perf_counter() - t_load:.1f}s")

    info, n_ep, global_i, stats_list, attempts = prepare_dataset(
        Path(args.source_dataset),
        shard_dir,
        SUCCESS_PROMPT,
        "Failed to finish the whole process.",
        overwrite=bool(args.overwrite),
        resume=not bool(args.overwrite),
        save_all_trajectories=True,
        outcome_task_mode="clean",
    )

    # Resume: skip already-saved (seed, repeat) pairs.
    done_pairs = {
        (int(a["seed"]), int(a.get("repeat", -1)))
        for a in attempts
        if a.get("saved_episode_index") is not None and a.get("success") is not None
    }
    # Older attempts may lack repeat; rebuild from outcomes if present.
    outcomes_path = shard_dir / "meta" / OUTCOME_LEDGER_NAME
    if outcomes_path.exists():
        for line in outcomes_path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            # we store repeat in attempt_log; outcomes only have seed

    attempt_idx = len(attempts)
    for seed in seeds:
        env = CONFIG_MAPPING[args.task_name]().get_environment(
            policy_mode=True,
            render_mode="rgb_array",
            randomize=False,
            seed=int(seed),
            randomize_dynamics=False,
        )
        try:
            for repeat in range(int(args.repeats)):
                if (seed, repeat) in done_pairs:
                    log(f"skip seed={seed} repeat={repeat} (already saved)")
                    continue
                log(f"start seed={seed} repeat={repeat}")
                ep = rollout_episode(
                    env,
                    policy,
                    seed=int(seed),
                    repeat=int(repeat),
                    max_steps=int(args.max_steps),
                )
                length = save_lerobot_episode(
                    shard_dir,
                    info,
                    stats_list,
                    episode_index=n_ep,
                    global_start_index=global_i,
                    episode=ep,
                    task_text=SUCCESS_PROMPT,
                    task_index=0,
                    fps=int(args.fps),
                )
                append_jsonl(
                    shard_dir / "meta" / OUTCOME_LEDGER_NAME,
                    make_outcome_row(
                        episode_index=n_ep,
                        success=bool(ep["success"]),
                        attempt_index=attempt_idx,
                        seed=int(seed),
                    ),
                )
                attempts.append(
                    {
                        "attempt_index": attempt_idx,
                        "seed": int(seed),
                        "repeat": int(repeat),
                        "success": bool(ep["success"]),
                        "done": True,
                        "steps": int(length),
                        "elapsed_s": float(ep["elapsed_s"]),
                        "saved_failure_index": None,
                        "saved_episode_index": int(n_ep),
                    }
                )
                global_i += length
                n_ep += 1
                attempt_idx += 1
                update_info(
                    shard_dir,
                    info,
                    num_episodes=n_ep,
                    total_frames=global_i,
                    total_tasks=1,
                )
                log(
                    f"saved seed={seed} repeat={repeat} success={ep['success']} "
                    f"steps={length} ep={n_ep - 1}"
                )
                # Persist progress after each episode for crash-resume.
                write_json(
                    shard_dir / "collection_summary.json",
                    {
                        "status": "in_progress",
                        "mode": "save_all",
                        "outcome_task_mode": "clean",
                        "target_episodes": len(seeds) * int(args.repeats),
                        "attempts": len(attempts),
                        "episodes": n_ep,
                        "frames": global_i,
                        "failures": sum(1 for a in attempts if not a["success"]),
                        "successes_saved": sum(1 for a in attempts if a["success"]),
                        "attempt_log": attempts,
                        "inference_stack": "opensource_FastWAMDexJocoPolicy",
                        "base_seed": int(args.seed_start),
                        "seed_end": int(args.seed_end),
                        "repeats": int(args.repeats),
                        "gpu_id": int(gpu_id),
                    },
                )
        finally:
            close = getattr(env, "close", None)
            if callable(close):
                close()

    write_json(shard_dir / "meta" / "stats.json", serialize_dict(aggregate_stats(stats_list)))
    write_json(
        shard_dir / "collection_summary.json",
        {
            "status": "complete",
            "mode": "save_all",
            "outcome_task_mode": "clean",
            "target_episodes": len(seeds) * int(args.repeats),
            "attempts": len(attempts),
            "episodes": n_ep,
            "frames": global_i,
            "failures": sum(1 for a in attempts if not a["success"]),
            "successes_saved": sum(1 for a in attempts if a["success"]),
            "attempt_log": attempts,
            "inference_stack": "opensource_FastWAMDexJocoPolicy",
            "base_seed": int(args.seed_start),
            "seed_end": int(args.seed_end),
            "repeats": int(args.repeats),
            "gpu_id": int(gpu_id),
        },
    )
    log(f"DONE episodes={n_ep} frames={global_i}")


def main() -> int:
    args = parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("DIFFSYNTH_SKIP_DOWNLOAD", "true")
    os.environ.setdefault("DIFFSYNTH_MODEL_BASE_PATH", str(ROOT / "checkpoints"))
    _setup_paths()
    if not args.skip_pin_check:
        assert_pin()

    gpus = _parse_gpus(args.gpus)
    seeds = list(range(int(args.seed_start), int(args.seed_end) + 1))
    if not seeds:
        raise SystemExit("empty seed range")
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(
        json.dumps(
            {
                **{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
                "n_seeds": len(seeds),
                "n_episodes": len(seeds) * int(args.repeats),
                "stack": "opensource_FastWAMDexJocoPolicy",
            },
            indent=2,
        )
        + "\n"
    )

    # Shard seeds round-robin across GPUs.
    assignments: dict[int, list[int]] = {g: [] for g in gpus}
    for i, seed in enumerate(seeds):
        assignments[gpus[i % len(gpus)]].append(seed)

    print(f"[collect-orch] gpus={gpus} seeds={seeds[0]}..{seeds[-1]} ×{args.repeats}", flush=True)
    for g, ss in assignments.items():
        print(f"  gpu{g}: {len(ss)} seeds -> {len(ss) * args.repeats} eps", flush=True)

    args_dict = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    # spawn: safe with CUDA (fork + cuda init in parent is unsafe).
    ctx = mp.get_context("spawn")
    procs: list[mp.Process] = []
    for g in gpus:
        seed_list = list(assignments[g])
        proc = ctx.Process(
            target=_worker_entry,
            args=(g, seed_list, args_dict),
            name=f"collect-gpu{g}",
        )
        proc.start()
        procs.append(proc)
        print(f"[collect-orch] launched gpu{g} pid={proc.pid}", flush=True)

    rc = 0
    for proc in procs:
        proc.join()
        if proc.exitcode not in (0, None):
            print(f"[collect-orch] {proc.name} FAILED rc={proc.exitcode}", flush=True)
            rc = proc.exitcode or 1

    if rc != 0:
        return int(rc)

    # Merge shards via CLI (avoid brittle symbol imports).
    shard_dirs = [
        out / "shards" / f"gpu_{g}"
        for g in gpus
        if (out / "shards" / f"gpu_{g}" / "collection_summary.json").exists()
    ]
    raw_out = out / "rollout_raw_200"
    print(f"[collect-orch] merging {len(shard_dirs)} shards -> {raw_out}", flush=True)
    import subprocess

    subprocess.check_call(
        [
            sys.executable,
            str(ROOT / "scripts" / "build_rollout_datasets.py"),
            "merge-shards",
            "--shard-datasets",
            *[str(p) for p in shard_dirs],
            "--output-dataset",
            str(raw_out),
            "--overwrite",
        ],
        cwd=str(ROOT),
    )
    subprocess.check_call(
        [
            sys.executable,
            str(ROOT / "scripts" / "build_rollout_datasets.py"),
            "validate-outcomes",
            "--dataset",
            str(raw_out),
            "--expected-episodes",
            str(len(seeds) * int(args.repeats)),
            "--report",
            str(out / "rollout_outcome_validation.json"),
        ],
        cwd=str(ROOT),
    )

    # Summary rates across attempt_logs.
    pooled_s = sum(
        int(json.loads((out / "shards" / f"gpu_{g}" / "collection_summary.json").read_text())["successes_saved"])
        for g in gpus
    )
    pooled_n = len(seeds) * int(args.repeats)
    agg = {
        "protocol": f"collect_4x50_seeds_{args.seed_start}_{args.seed_end}",
        "inference_stack": "opensource_FastWAMDexJocoPolicy",
        "pooled_successes": pooled_s,
        "pooled_episodes": pooled_n,
        "pooled_success_rate": pooled_s / pooled_n if pooled_n else None,
        "rollout_raw": str(raw_out),
    }
    (out / "aggregate.json").write_text(json.dumps(agg, indent=2) + "\n")
    print(json.dumps(agg, indent=2), flush=True)
    print(f"[collect-orch] DONE raw={raw_out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
