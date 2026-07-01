#!/usr/bin/env python3
"""Collect failed DexJoCo rollouts as a two-camera LeRobot dataset."""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

os.environ.setdefault("MUJOCO_GL", "egl")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
SRC_ROOT = PROJECT_ROOT / "src"
DEXJOCO_REPO_ROOT = PROJECT_ROOT / "third_party" / "dexjoco"
DEXJOCO_PY_ROOT = DEXJOCO_REPO_ROOT / "dexjoco"
for path in (PROJECT_ROOT, SRC_ROOT, SCRIPTS_ROOT, DEXJOCO_REPO_ROOT, DEXJOCO_PY_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dexjoco_fastwam_adapter import (
    DEFAULT_TASK_CONFIG_DIR,
    ActionConstraintConfig,
    DexJoCoFastWAMAdapter,
    DexJoCoFastWAMEvalEnv,
    DexJoCoTaskConfig,
    _safe_rgb_uint8,
    constrain_rotvec_action,
    load_dexjoco_eval_settings,
    resolve_env_camera_keys,
)
from fastwam.datasets.lerobot.lerobot.datasets.compute_stats import aggregate_stats, compute_episode_stats
from fastwam.datasets.lerobot.lerobot.datasets.utils import cast_stats_to_numpy, serialize_dict
from fastwam.datasets.lerobot.lerobot.datasets.video_utils import get_video_info
from policy_zmq_client import PolicyClient


DEFAULT_SUCCESS_DATASET_ROOT = Path("/data_all/share/datasets/dexjoco/dexjoco_lerobot_datasets")
DEFAULT_FAILURE_DATASET_ROOT = Path("/data_all/share/FastWAM_zhaoyc_failure/artifacts/datasets")
DEFAULT_TASK_NAME = "water_plant"
FAILURE_PHRASE = "Failed to finish the whole process."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect failed FastWAM DexJoCo rollouts in LeRobot v2.1 format."
    )
    parser.add_argument(
        "--task-name",
        type=str,
        default=None,
        help="DexJoCo task name, e.g. hammer_nail. Defaults to water_plant for backward compatibility.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="FastWAM training run directory containing config.yaml.",
    )
    parser.add_argument("--policy-host", type=str, default="127.0.0.1")
    parser.add_argument("--policy-port", type=int, default=5560)
    parser.add_argument("--policy-timeout-ms", type=int, default=300000)
    parser.add_argument(
        "--task-config",
        type=Path,
        default=None,
        help="DexJoCo task yaml. Defaults to rand_obj/<task_name>.yaml.",
    )
    parser.add_argument(
        "--source-dataset",
        type=Path,
        default=None,
        help="Existing successful LeRobot dataset used as schema/template. Defaults to the task dataset.",
    )
    parser.add_argument(
        "--output-dataset",
        type=Path,
        default=None,
        help="Output failure LeRobot dataset. Defaults to artifacts/datasets/<task_name>_failure_fastwam_2cam_text.",
    )
    parser.add_argument("--target-failures", type=int, default=100)
    parser.add_argument("--max-attempts", type=int, default=260)
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--replan-steps", type=int, default=None)
    parser.add_argument("--max-env-steps", type=int, default=600)
    parser.add_argument("--video-fps", type=int, default=30)
    parser.add_argument("--failure-phrase", type=str, default=FAILURE_PHRASE)
    parser.add_argument("--randomize", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--randomize-dynamics", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--action-clip", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--clip-max-xyz-step", type=float, default=0.05)
    parser.add_argument("--clip-max-dz-down", type=float, default=0.03)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Append to an existing output dataset using collection_summary.json to continue seeds.",
    )
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def save_episode_video(frames: list[np.ndarray], path: Path, fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not frames:
        raise ValueError(f"Cannot save empty video: {path}")
    first = _safe_rgb_uint8(frames[0])
    height, width = first.shape[:2]
    container = av.open(str(path), mode="w")
    try:
        stream = container.add_stream("libx264", rate=fps)
        stream.width = int(width)
        stream.height = int(height)
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "21"}
        for frame in frames:
            video_frame = av.VideoFrame.from_ndarray(_safe_rgb_uint8(frame), format="rgb24")
            for packet in stream.encode(video_frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()


def fixed_size_float_array(values: np.ndarray, dim: int) -> pa.FixedSizeListArray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != dim:
        raise ValueError(f"Expected [N,{dim}] float array, got {values.shape}")
    flat = pa.array(values.reshape(-1), type=pa.float32())
    return pa.FixedSizeListArray.from_arrays(flat, dim)


def write_episode_parquet(
    path: Path,
    *,
    actions: np.ndarray,
    states: np.ndarray,
    timestamps: np.ndarray,
    frame_indices: np.ndarray,
    episode_indices: np.ndarray,
    global_indices: np.ndarray,
    task_indices: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "action": fixed_size_float_array(actions, 22),
            "observation.state": fixed_size_float_array(states, 23),
            "timestamp": pa.array(timestamps.astype(np.float32), type=pa.float32()),
            "frame_index": pa.array(frame_indices.astype(np.int64), type=pa.int64()),
            "episode_index": pa.array(episode_indices.astype(np.int64), type=pa.int64()),
            "index": pa.array(global_indices.astype(np.int64), type=pa.int64()),
            "task_index": pa.array(task_indices.astype(np.int64), type=pa.int64()),
        }
    )
    pq.write_table(table, path)


def load_task(task_config: Path) -> DexJoCoTaskConfig:
    with task_config.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return DexJoCoTaskConfig.from_yaml(cfg)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def resolve_task_name(args: argparse.Namespace) -> str:
    if args.task_name:
        return str(args.task_name)
    if args.task_config is not None:
        return Path(args.task_config).stem
    if args.source_dataset is not None:
        return Path(args.source_dataset).name
    return DEFAULT_TASK_NAME


def resolve_input_paths(args: argparse.Namespace) -> tuple[str, Path, Path, Path]:
    task_name = resolve_task_name(args)
    task_config = (
        args.task_config.expanduser()
        if args.task_config is not None
        else DEFAULT_TASK_CONFIG_DIR / f"{task_name}.yaml"
    )
    source_dataset = (
        args.source_dataset.expanduser()
        if args.source_dataset is not None
        else DEFAULT_SUCCESS_DATASET_ROOT / task_name
    )
    output_dataset = (
        args.output_dataset.expanduser()
        if args.output_dataset is not None
        else DEFAULT_FAILURE_DATASET_ROOT / f"{task_name}_failure_fastwam_2cam_text"
    )
    return task_name, task_config, source_dataset, output_dataset


def read_primary_task_text(source_dataset: Path) -> str:
    tasks = load_jsonl(source_dataset / "meta" / "tasks.jsonl")
    if not tasks:
        raise FileNotFoundError(f"Missing or empty task metadata: {source_dataset / 'meta' / 'tasks.jsonl'}")
    task = str(tasks[0].get("task", "")).strip()
    if not task:
        raise ValueError(f"First task row has no non-empty `task`: {source_dataset / 'meta' / 'tasks.jsonl'}")
    return task


def infer_video_keys(source_info: dict[str, Any], task: DexJoCoTaskConfig) -> tuple[tuple[str, str], ...]:
    camera_feature_keys = [
        str(key)
        for key in source_info.get("features", {})
        if str(key).startswith("observation.images.")
    ]
    if not camera_feature_keys:
        raise ValueError("Source dataset has no `observation.images.*` features.")

    dataset_camera_keys = [key.split("observation.images.", 1)[1] for key in camera_feature_keys]
    env_camera_keys = resolve_env_camera_keys(dataset_camera_keys, task.camera_mapping)
    return tuple(zip(camera_feature_keys, env_camera_keys))


def prepare_dataset(
    source_dataset: Path,
    output_dataset: Path,
    failure_task: str,
    video_keys: tuple[tuple[str, str], ...],
    *,
    overwrite: bool,
    resume: bool,
) -> tuple[dict, int, int, list[dict[str, dict]], list[dict[str, Any]]]:
    if overwrite and resume:
        raise ValueError("--overwrite and --resume are mutually exclusive")

    if resume:
        if not output_dataset.exists():
            raise FileNotFoundError(f"Cannot resume missing dataset: {output_dataset}")
        info = read_json(output_dataset / "meta" / "info.json")
        task_lines = load_jsonl(output_dataset / "meta" / "tasks.jsonl")
        if not task_lines or task_lines[0].get("task") != failure_task:
            raise ValueError(
                "Existing dataset task text does not match requested failure task: "
                f"{task_lines[0].get('task') if task_lines else None!r}"
            )
        episode_stats = [
            cast_stats_to_numpy(item["stats"])
            for item in load_jsonl(output_dataset / "meta" / "episodes_stats.jsonl")
        ]
        summary_path = output_dataset / "collection_summary.json"
        attempts = read_json(summary_path).get("attempt_log", []) if summary_path.exists() else []
        return (
            info,
            int(info.get("total_episodes", 0)),
            int(info.get("total_frames", 0)),
            episode_stats,
            attempts,
        )

    if output_dataset.exists():
        if not overwrite:
            raise FileExistsError(f"Output dataset already exists: {output_dataset}")
        shutil.rmtree(output_dataset)

    (output_dataset / "meta").mkdir(parents=True, exist_ok=False)
    (output_dataset / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    for video_key, _ in video_keys:
        (output_dataset / "videos" / "chunk-000" / video_key).mkdir(parents=True, exist_ok=True)

    info = copy.deepcopy(read_json(source_dataset / "meta" / "info.json"))
    info["total_episodes"] = 0
    info["total_frames"] = 0
    info["total_tasks"] = 1
    info["total_videos"] = 0
    info["total_chunks"] = 1
    info["splits"] = {"train": "0:0"}
    info["fps"] = int(info.get("fps", 30))
    info["data_path"] = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
    info["video_path"] = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
    write_json(output_dataset / "meta" / "info.json", info)
    append_jsonl(output_dataset / "meta" / "tasks.jsonl", {"task_index": 0, "task": failure_task})

    modality_path = source_dataset / "meta" / "modality.json"
    if modality_path.exists():
        shutil.copy2(modality_path, output_dataset / "meta" / "modality.json")

    return info, 0, 0, [], []


def update_info(
    output_dataset: Path,
    info: dict,
    *,
    num_episodes: int,
    total_frames: int,
    video_keys: tuple[tuple[str, str], ...],
) -> None:
    info["total_episodes"] = int(num_episodes)
    info["total_frames"] = int(total_frames)
    info["total_tasks"] = 1
    info["total_videos"] = int(num_episodes * len(video_keys))
    info["total_chunks"] = 1 if num_episodes > 0 else 0
    info["splits"] = {"train": f"0:{num_episodes}"}
    if num_episodes > 0:
        for video_key, _ in video_keys:
            video_path = output_dataset / info["video_path"].format(
                episode_chunk=0,
                video_key=video_key,
                episode_index=0,
            )
            try:
                info["features"][video_key]["info"] = get_video_info(video_path)
            except Exception as exc:
                print(f"[warn] get_video_info failed for {video_path}: {exc}", flush=True)
    write_json(output_dataset / "meta" / "info.json", info)


def run_attempt(
    task: DexJoCoTaskConfig,
    *,
    policy: PolicyClient,
    adapter: DexJoCoFastWAMAdapter,
    seed: int,
    replan_steps: int,
    max_env_steps: int,
    video_keys: tuple[tuple[str, str], ...],
    randomize: bool,
    randomize_dynamics: bool,
    action_clip_config: ActionConstraintConfig | None,
) -> dict[str, Any]:
    env = DexJoCoFastWAMEvalEnv(
        task,
        seed=seed,
        randomize=randomize,
        randomize_dynamics=randomize_dynamics,
    )
    action_queue: deque[np.ndarray] = deque()
    frames: dict[str, list[np.ndarray]] = {video_key: [] for video_key, _ in video_keys}
    actions: list[np.ndarray] = []
    states: list[np.ndarray] = []
    policy_query_steps: list[int] = []
    t0 = time.perf_counter()

    try:
        env.reset()
        policy.reset()
        env.click_mouse_warmup()

        steps = 0
        while steps < max_env_steps:
            if not action_queue:
                policy_obs = env.build_policy_obs(adapter)
                response = policy.get_action(policy_obs)
                chunk = adapter.parse_policy_response(response)
                n_exec = max(1, min(replan_steps, chunk.shape[0]))
                policy_query_steps.append(steps)
                for idx in range(n_exec):
                    action_queue.append(chunk[idx])

            rotvec_action = np.asarray(action_queue.popleft(), dtype=np.float32)
            if action_clip_config is not None:
                current_state = np.asarray(env._latest_obs["state"], dtype=np.float32).reshape(-1)
                rotvec_action = constrain_rotvec_action(
                    rotvec_action,
                    current_state,
                    dual_arm=env.task.dual_arm,
                    config=action_clip_config,
                )

            current_obs = env._latest_obs
            states.append(np.asarray(current_obs["state"], dtype=np.float32).reshape(-1)[:23])
            actions.append(rotvec_action.reshape(-1)[:22].astype(np.float32))
            for video_key, env_key in video_keys:
                frames[video_key].append(_safe_rgb_uint8(current_obs[env_key]))

            env.step_rotvec(rotvec_action)
            steps += 1
            if env.is_done:
                break

        return {
            "success": bool(env.is_success),
            "done": bool(env.is_done),
            "steps": int(steps),
            "elapsed_s": time.perf_counter() - t0,
            "actions": np.stack(actions).astype(np.float32) if actions else np.zeros((0, 22), dtype=np.float32),
            "states": np.stack(states).astype(np.float32) if states else np.zeros((0, 23), dtype=np.float32),
            "frames": frames,
            "policy_query_steps": policy_query_steps,
        }
    finally:
        env.close()


def save_failure_episode(
    output_dataset: Path,
    info: dict,
    stats_list: list[dict[str, dict]],
    *,
    episode_index: int,
    global_start_index: int,
    episode: dict[str, Any],
    failure_task: str,
    fps: int,
    video_keys: tuple[tuple[str, str], ...],
) -> int:
    length = int(episode["actions"].shape[0])
    if length <= 0:
        raise ValueError("Cannot save empty episode")

    chunk = episode_index // int(info["chunks_size"])
    actions = np.asarray(episode["actions"], dtype=np.float32)
    states = np.asarray(episode["states"], dtype=np.float32)
    timestamps = np.arange(length, dtype=np.float32) / float(fps)
    frame_indices = np.arange(length, dtype=np.int64)
    episode_indices = np.full((length,), episode_index, dtype=np.int64)
    global_indices = np.arange(global_start_index, global_start_index + length, dtype=np.int64)
    task_indices = np.zeros((length,), dtype=np.int64)

    for video_key, _ in video_keys:
        video_path = output_dataset / info["video_path"].format(
            episode_chunk=chunk,
            video_key=video_key,
            episode_index=episode_index,
        )
        save_episode_video(episode["frames"][video_key], video_path, fps)

    parquet_path = output_dataset / info["data_path"].format(
        episode_chunk=chunk,
        episode_index=episode_index,
    )
    write_episode_parquet(
        parquet_path,
        actions=actions,
        states=states,
        timestamps=timestamps,
        frame_indices=frame_indices,
        episode_indices=episode_indices,
        global_indices=global_indices,
        task_indices=task_indices,
    )

    append_jsonl(
        output_dataset / "meta" / "episodes.jsonl",
        {
            "episode_index": episode_index,
            "tasks": [failure_task],
            "length": length,
        },
    )

    episode_data_for_stats = {
        "action": actions,
        "observation.state": states,
        "timestamp": timestamps,
        "frame_index": frame_indices,
        "episode_index": episode_indices,
        "index": global_indices,
        "task_index": task_indices,
    }
    ep_stats = compute_episode_stats(
        episode_data_for_stats,
        info["features"],
        is_compute_episode_stats_image=False,
    )
    stats_list.append(ep_stats)
    append_jsonl(
        output_dataset / "meta" / "episodes_stats.jsonl",
        {
            "episode_index": episode_index,
            "stats": serialize_dict(ep_stats),
        },
    )
    return length


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    task_name, task_config, source_dataset, output_dataset = resolve_input_paths(args)
    source_dataset = source_dataset.resolve()
    output_dataset = output_dataset.resolve()
    task_config = task_config.resolve()

    if not source_dataset.exists():
        raise FileNotFoundError(source_dataset)
    if not task_config.exists():
        raise FileNotFoundError(task_config)

    task = load_task(task_config)
    source_info = read_json(source_dataset / "meta" / "info.json")
    fps = int(args.video_fps or source_info.get("fps", 30))
    base_task = read_primary_task_text(source_dataset)
    failure_task = f"{base_task} {args.failure_phrase.strip()}".strip()
    video_keys = infer_video_keys(source_info, task)
    info, failures, global_index, stats_list, attempts = prepare_dataset(
        source_dataset,
        output_dataset,
        failure_task,
        video_keys,
        overwrite=args.overwrite,
        resume=args.resume,
    )

    eval_settings = load_dexjoco_eval_settings(run_dir)
    adapter = DexJoCoFastWAMAdapter(eval_settings)
    replan_steps = args.replan_steps
    if replan_steps is None:
        replan_steps = max(1, int(0.8 * adapter.action_horizon))

    action_clip_config = None
    if args.action_clip:
        action_clip_config = ActionConstraintConfig(
            max_xyz_step=args.clip_max_xyz_step,
            max_dz_down=args.clip_max_dz_down,
            clip_to_dataset_bounds=False,
        )

    print(f"[collect] task_name={task_name}", flush=True)
    print(f"[collect] run_dir={run_dir}", flush=True)
    print(f"[collect] task_config={task_config}", flush=True)
    print(f"[collect] source_dataset={source_dataset}", flush=True)
    print(f"[collect] output_dataset={output_dataset}", flush=True)
    print(f"[collect] video_keys={video_keys}", flush=True)
    print(f"[collect] target_failures={args.target_failures} max_attempts={args.max_attempts}", flush=True)
    if args.resume:
        print(
            f"[collect] resume existing failures={failures} attempts={len(attempts)} frames={global_index}",
            flush=True,
        )
    print(f"[collect] policy={args.policy_host}:{args.policy_port} replan_steps={replan_steps}", flush=True)
    print(f"[collect] failure_task={failure_task}", flush=True)

    policy = PolicyClient(
        host=args.policy_host,
        port=args.policy_port,
        timeout_ms=args.policy_timeout_ms,
    )
    if not policy.ping():
        raise RuntimeError(f"Policy server ping failed at {args.policy_host}:{args.policy_port}")

    try:
        for attempt_idx in range(len(attempts), args.max_attempts):
            if failures >= args.target_failures:
                break
            seed = int(args.seed + attempt_idx)
            print(f"[collect] attempt {attempt_idx + 1}/{args.max_attempts} seed={seed}", flush=True)
            episode = run_attempt(
                task,
                policy=policy,
                adapter=adapter,
                seed=seed,
                replan_steps=replan_steps,
                max_env_steps=args.max_env_steps,
                video_keys=video_keys,
                randomize=args.randomize,
                randomize_dynamics=args.randomize_dynamics,
                action_clip_config=action_clip_config,
            )
            attempts.append(
                {
                    "attempt_index": attempt_idx,
                    "seed": seed,
                    "success": bool(episode["success"]),
                    "done": bool(episode["done"]),
                    "steps": int(episode["steps"]),
                    "elapsed_s": float(episode["elapsed_s"]),
                    "saved_failure_index": failures if not episode["success"] else None,
                }
            )
            print(
                f"[collect] result success={episode['success']} done={episode['done']} "
                f"steps={episode['steps']} elapsed={episode['elapsed_s']:.1f}s",
                flush=True,
            )
            if episode["success"]:
                continue

            length = save_failure_episode(
                output_dataset,
                info,
                stats_list,
                episode_index=failures,
                global_start_index=global_index,
                episode=episode,
                failure_task=failure_task,
                fps=fps,
                video_keys=video_keys,
            )
            global_index += length
            failures += 1
            update_info(
                output_dataset,
                info,
                num_episodes=failures,
                total_frames=global_index,
                video_keys=video_keys,
            )
            write_json(
                output_dataset / "collection_summary.json",
                {
                    "status": "running",
                    "task_name": task_name,
                    "task_config": str(task_config),
                    "source_dataset": str(source_dataset),
                    "output_dataset": str(output_dataset),
                    "video_keys": [
                        {"dataset_key": dataset_key, "env_key": env_key}
                        for dataset_key, env_key in video_keys
                    ],
                    "target_failures": args.target_failures,
                    "max_attempts": args.max_attempts,
                    "failures": failures,
                    "attempts": len(attempts),
                    "successes_discarded": sum(1 for item in attempts if item["success"]),
                    "failure_task": failure_task,
                    "attempt_log": attempts,
                },
            )
            print(f"[collect] saved failure {failures}/{args.target_failures}", flush=True)
    finally:
        policy.close()

    if stats_list:
        write_json(output_dataset / "meta" / "stats.json", serialize_dict(aggregate_stats(stats_list)))
    update_info(
        output_dataset,
        info,
        num_episodes=failures,
        total_frames=global_index,
        video_keys=video_keys,
    )
    write_json(
        output_dataset / "collection_summary.json",
        {
            "status": "complete" if failures >= args.target_failures else "incomplete",
            "task_name": task_name,
            "task_config": str(task_config),
            "source_dataset": str(source_dataset),
            "output_dataset": str(output_dataset),
            "video_keys": [
                {"dataset_key": dataset_key, "env_key": env_key}
                for dataset_key, env_key in video_keys
            ],
            "target_failures": args.target_failures,
            "max_attempts": args.max_attempts,
            "failures": failures,
            "attempts": len(attempts),
            "successes_discarded": sum(1 for item in attempts if item["success"]),
            "failure_task": failure_task,
            "attempt_log": attempts,
        },
    )
    print(f"[collect] finished failures={failures} attempts={len(attempts)} frames={global_index}", flush=True)


if __name__ == "__main__":
    main()
