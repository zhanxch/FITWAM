#!/usr/bin/env python3
"""Probe policy action distributions at aligned seed-pair candidate contexts.

Every context is sampled with the same diffusion noise seeds. Full executed
24-step chunks are retained. Comparisons are made in the checkpoint's training
normalization space; robot-unit actions are retained only for replay. Scalar
widths are diagnostics, not event labels.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import av
import numpy as np
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[2]
OPEN = Path(
    os.environ.get(
        "FASTWAM_OPEN_REPO", "/data_all/xiangchengzhan/FastWAM-infer-in-DexJoco"
    )
)
FASTWAM_PIN = Path(
    os.environ.get(
        "FASTWAM_PIN", str(ROOT / "third_party/FastWAM_pin_45d8e14")
    )
)
EXPECTED_PIN = "45d8e1458921d83f8ad6cf9ce993d371208dabd0"
DEFAULT_CKPT = OPEN / "checkpoints/fold_glasses/step_010000.pt"
DEFAULT_CFG = OPEN / "configs/fastwam_dexjoco.yaml"
DEFAULT_STATS = OPEN / "artifacts/fold_glasses/dataset_stats.json"
DEFAULT_TEXT = (
    OPEN
    / "artifacts/fold_glasses"
    / "0c3367ce1d74848cc46b93c6d2eee5e2097dca410a2c95f3da48bd8c8673fa20.t5_len128.wan22ti2v5b.pt"
)


def setup_paths() -> None:
    paths = [
        OPEN / "src",
        FASTWAM_PIN / "src",
        ROOT / "third_party/dexjoco/dexjoco",
        ROOT / "scripts",
        ROOT / "scripts/analysis",
    ]
    for path in paths:
        value = str(path)
        if value in sys.path:
            sys.path.remove(value)
        sys.path.insert(0, value)


def assert_pin() -> None:
    head = subprocess.check_output(
        ["git", "-C", str(FASTWAM_PIN), "rev-parse", "HEAD"], text=True
    ).strip()
    if head != EXPECTED_PIN:
        raise RuntimeError(f"FastWAM pin mismatch: {head}")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def read_video_frame(path: Path, target: int) -> np.ndarray:
    with av.open(str(path)) as container:
        for frame_index, frame in enumerate(container.decode(video=0)):
            if frame_index == target:
                return frame.to_ndarray(format="rgb24")
    raise IndexError(f"{path} has no frame {target}")


def load_context(
    dataset: Path, episode_index: int, frame: int, block_size: int
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    parquet = (
        dataset
        / "data"
        / "chunk-000"
        / f"episode_{episode_index:06d}.parquet"
    )
    table = pq.read_table(parquet, columns=["action", "observation.state"])
    actions = np.asarray(table.column("action").to_pylist(), dtype=np.float32)
    states = np.asarray(
        table.column("observation.state").to_pylist(), dtype=np.float32
    )
    if actions.shape[1] == 23:
        actions = actions[:, 1:]
    if actions.shape[1] != 22 or states.shape[1] < 23:
        raise ValueError(
            f"Unexpected arrays for episode {episode_index}: "
            f"actions={actions.shape}, states={states.shape}"
        )
    if not 0 <= frame < min(len(actions), len(states)):
        raise IndexError(
            f"Frame {frame} outside episode {episode_index} length "
            f"{min(len(actions), len(states))}"
        )
    actual = np.zeros((block_size, 22), dtype=np.float32)
    actual_valid = np.zeros(block_size, dtype=bool)
    count = min(block_size, len(actions) - frame)
    actual[:count] = actions[frame : frame + count]
    actual_valid[:count] = True
    video_root = dataset / "videos" / "chunk-000"
    observation = {
        "front": read_video_frame(
            video_root
            / "observation.images.front"
            / f"episode_{episode_index:06d}.mp4",
            frame,
        ),
        "wrist": read_video_frame(
            video_root
            / "observation.images.wrist"
            / f"episode_{episode_index:06d}.mp4",
            frame,
        ),
        "state": states[frame, :23],
    }
    return observation, actual, actual_valid


def context_id(episode_index: int, frame: int) -> str:
    return f"ep{episode_index:06d}_f{frame:04d}"


def normalize_actions_for_training(policy: Any, actions: np.ndarray) -> np.ndarray:
    """Apply the exact action normalizer used by the loaded policy."""

    import torch

    action_key = policy.processor.shape_meta["action"][0]["key"]
    normalizer = policy.processor.normalizer.normalizers["action"][action_key]
    tensor = torch.as_tensor(actions, dtype=torch.float32)
    normalized = normalizer.forward(tensor)
    return normalized.detach().cpu().numpy().astype(np.float32)


def build_contexts(
    candidates: Sequence[Mapping[str, Any]], *, include_rejected: bool = False
) -> list[dict[str, Any]]:
    indexed: dict[tuple[int, int], dict[str, Any]] = {}
    for candidate in candidates:
        if not include_rejected and not bool(candidate.get("probe_eligible")):
            continue
        candidate_id = str(candidate["candidate_id"])
        failure_episode = int(candidate["failure_episode_index"])
        failure_frame = int(candidate["failure_frame"])
        failure_key = (failure_episode, failure_frame)
        record = indexed.setdefault(
            failure_key,
            {
                "context_id": context_id(*failure_key),
                "episode_index": failure_episode,
                "frame": failure_frame,
                "roles": [],
                "candidate_ids": [],
            },
        )
        record["roles"].append("failure_anchor")
        record["candidate_ids"].append(candidate_id)
        for alignment in candidate.get("success_alignments", []):
            success_key = (
                int(alignment["success_episode_index"]),
                int(alignment["success_frame"]),
            )
            record = indexed.setdefault(
                success_key,
                {
                    "context_id": context_id(*success_key),
                    "episode_index": success_key[0],
                    "frame": success_key[1],
                    "roles": [],
                    "candidate_ids": [],
                },
            )
            record["roles"].append("success_target")
            record["candidate_ids"].append(candidate_id)
    rows: list[dict[str, Any]] = []
    for record in indexed.values():
        record["roles"] = sorted(set(record["roles"]))
        record["candidate_ids"] = sorted(set(record["candidate_ids"]))
        rows.append(record)
    rows.sort(key=lambda row: (int(row["episode_index"]), int(row["frame"])))
    return rows


def probe_context(
    policy: Any,
    dataset: Path,
    context: Mapping[str, Any],
    *,
    noise_seeds: np.ndarray,
    block_size: int,
) -> dict[str, Any]:
    episode_index = int(context["episode_index"])
    frame = int(context["frame"])
    observation, actual, actual_valid = load_context(
        dataset, episode_index, frame, block_size
    )
    started = time.perf_counter()
    samples_robot = np.stack(
        [
            np.asarray(policy.infer(observation, noise_seed=int(seed)), np.float32)[
                :block_size
            ]
            for seed in noise_seeds.tolist()
        ],
        axis=0,
    )
    samples_normalized = normalize_actions_for_training(policy, samples_robot)
    actual_normalized = normalize_actions_for_training(policy, actual)
    first_step_width_l2 = float(
        np.linalg.norm(samples_normalized[:, 0, :].std(axis=0))
    )
    chunk_width_rms = float(
        np.sqrt(
            np.mean(np.var(samples_normalized, axis=0, dtype=np.float64))
        )
    )
    robot_first_step_width_l2 = float(
        np.linalg.norm(samples_robot[:, 0, :].std(axis=0))
    )
    return {
        "samples_robot": samples_robot,
        "samples_normalized": samples_normalized,
        "actual_robot": actual,
        "actual_normalized": actual_normalized,
        "actual_valid": actual_valid,
        "first_step_width_l2": first_step_width_l2,
        "chunk_width_rms": chunk_width_rms,
        "robot_first_step_width_l2": robot_first_step_width_l2,
        "elapsed_s": float(time.perf_counter() - started),
    }


def worker(
    gpu: int,
    contexts: Sequence[Mapping[str, Any]],
    args_dict: Mapping[str, Any],
) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("DIFFSYNTH_SKIP_DOWNLOAD", "true")
    os.environ.setdefault("DIFFSYNTH_MODEL_BASE_PATH", str(ROOT / "checkpoints"))
    setup_paths()
    import torch
    from fastwam_dexjoco.policy import FastWAMDexJocoPolicy

    args = argparse.Namespace(**dict(args_dict))
    torch.cuda.set_device(0)
    policy = FastWAMDexJocoPolicy(
        model_config=args.model_config,
        checkpoint=args.checkpoint,
        dataset_stats=args.dataset_stats,
        text_embedding=args.text_embedding,
        device="cuda:0",
        action_horizon=args.action_horizon,
        replan_steps=args.block_size,
        num_inference_steps=args.num_inference_steps,
    )
    dataset = Path(args.dataset)
    output = Path(args.output)
    probe_root = output / "contexts"
    probe_root.mkdir(parents=True, exist_ok=True)
    log_path = output / "logs" / f"gpu_{gpu}.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    noise_seeds = np.asarray(args.noise_seeds, dtype=np.int64)
    with log_path.open("a", encoding="utf-8") as log:
        for position, context in enumerate(contexts, start=1):
            destination = probe_root / f"{context['context_id']}.npz"
            if destination.exists() and not args.overwrite:
                row = {
                    "context_id": context["context_id"],
                    "status": "cached",
                    "gpu": gpu,
                }
                log.write(json.dumps(row, sort_keys=True) + "\n")
                log.flush()
                print(
                    f"[gpu{gpu} {position}/{len(contexts)}] cached "
                    f"{context['context_id']}",
                    flush=True,
                )
                continue
            result = probe_context(
                policy,
                dataset,
                context,
                noise_seeds=noise_seeds,
                block_size=int(args.block_size),
            )
            np.savez_compressed(
                destination,
                context_id=np.asarray(str(context["context_id"])),
                episode_index=np.asarray(
                    int(context["episode_index"]), dtype=np.int32
                ),
                frame=np.asarray(int(context["frame"]), dtype=np.int32),
                noise_seeds=noise_seeds,
                action_samples_normalized=result["samples_normalized"],
                action_samples_robot=result["samples_robot"],
                actual_action_block_normalized=result["actual_normalized"],
                actual_action_block_robot=result["actual_robot"],
                actual_action_valid=result["actual_valid"],
                first_step_width_l2=np.asarray(
                    result["first_step_width_l2"], dtype=np.float32
                ),
                chunk_width_rms=np.asarray(
                    result["chunk_width_rms"], dtype=np.float32
                ),
                robot_first_step_width_l2=np.asarray(
                    result["robot_first_step_width_l2"], dtype=np.float32
                ),
            )
            row = {
                "context_id": context["context_id"],
                "status": "complete",
                "gpu": gpu,
                "first_step_width_l2": result["first_step_width_l2"],
                "chunk_width_rms": result["chunk_width_rms"],
                "robot_first_step_width_l2": result[
                    "robot_first_step_width_l2"
                ],
                "elapsed_s": result["elapsed_s"],
            }
            log.write(json.dumps(row, sort_keys=True) + "\n")
            log.flush()
            print(
                f"[gpu{gpu} {position}/{len(contexts)}] "
                f"{context['context_id']} elapsed={result['elapsed_s']:.1f}s",
                flush=True,
            )


def parse_gpus(raw: str) -> list[int]:
    values = [int(value.strip()) for value in raw.split(",") if value.strip()]
    if not values or len(values) != len(set(values)):
        raise ValueError(f"Invalid --gpus value {raw!r}")
    return values


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--model-config", type=Path, default=DEFAULT_CFG)
    parser.add_argument("--dataset-stats", type=Path, default=DEFAULT_STATS)
    parser.add_argument("--text-embedding", type=Path, default=DEFAULT_TEXT)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--sample-seed0", type=int, default=20260813)
    parser.add_argument("--action-horizon", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=24)
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--include-rejected", action="store_true")
    parser.add_argument("--skip-pin-check", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.skip_pin_check:
        assert_pin()
    if args.num_samples < 2:
        raise ValueError("--num-samples must be at least 2")
    if not 1 <= args.block_size <= args.action_horizon:
        raise ValueError("--block-size must be within the action horizon")
    gpus = parse_gpus(args.gpus)
    candidates = read_jsonl(args.candidates.expanduser().resolve())
    contexts = build_contexts(
        candidates, include_rejected=bool(args.include_rejected)
    )
    if not contexts:
        raise ValueError("No action-distribution contexts were selected")
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    contexts_path = output / "contexts.jsonl"
    contexts_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in contexts),
        encoding="utf-8",
    )
    noise_seeds = [
        int(args.sample_seed0 + 10007 * index)
        for index in range(int(args.num_samples))
    ]
    config = {
        "format": "FoldGlassesSeedPairActionDistributionProbe",
        "version": "1.0",
        "dataset": str(args.dataset.expanduser().resolve()),
        "candidates": str(args.candidates.expanduser().resolve()),
        "checkpoint": str(args.checkpoint.expanduser().resolve()),
        "model_config": str(args.model_config.expanduser().resolve()),
        "dataset_stats": str(args.dataset_stats.expanduser().resolve()),
        "text_embedding": str(args.text_embedding.expanduser().resolve()),
        "gpus": gpus,
        "num_contexts": len(contexts),
        "num_samples": int(args.num_samples),
        "noise_seeds": noise_seeds,
        "paired_noise_seeds_across_contexts": True,
        "action_horizon": int(args.action_horizon),
        "executed_block_size": int(args.block_size),
        "num_inference_steps": int(args.num_inference_steps),
        "width_is_diagnostic_not_label": True,
        "comparison_action_space": "checkpoint_training_normalized",
        "robot_action_space_retained_for_replay_only": True,
    }
    (output / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args_dict = vars(args).copy()
    for key in (
        "dataset",
        "candidates",
        "output",
        "checkpoint",
        "model_config",
        "dataset_stats",
        "text_embedding",
    ):
        args_dict[key] = str(Path(args_dict[key]).expanduser().resolve())
    args_dict["noise_seeds"] = noise_seeds

    assignments = [contexts[index:: len(gpus)] for index in range(len(gpus))]
    spawn = mp.get_context("spawn")
    processes: list[mp.Process] = []
    for gpu, assigned in zip(gpus, assignments):
        if not assigned:
            continue
        process = spawn.Process(target=worker, args=(gpu, assigned, args_dict))
        process.start()
        processes.append(process)
    failures: list[int] = []
    for process in processes:
        process.join()
        if process.exitcode != 0:
            failures.append(int(process.exitcode or -1))
    if failures:
        raise RuntimeError(f"Action-distribution workers failed: {failures}")

    missing = [
        row["context_id"]
        for row in contexts
        if not (output / "contexts" / f"{row['context_id']}.npz").exists()
    ]
    if missing:
        raise RuntimeError(f"Missing probe outputs: {missing[:10]}")
    summary = {**config, "status": "complete", "output": str(output)}
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
