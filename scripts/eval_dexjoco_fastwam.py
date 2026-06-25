#!/usr/bin/env python3
"""Closed-loop DexJoCo evaluation for FastWAM policy server (all 11 tasks)."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
SRC_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_ROOT, SCRIPTS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dexjoco.data.video_writer import Mp4VideoWriter
from dexjoco_fastwam_adapter import (
    DEFAULT_TASK_CONFIG_DIR,
    ActionConstraintConfig,
    DexJoCoFastWAMAdapter,
    DexJoCoFastWAMEvalEnv,
    DexJoCoTaskConfig,
    constrain_rotvec_action,
    load_dexjoco_eval_settings,
    load_task_configs,
)
from policy_zmq_client import PolicyClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate FastWAM on all DexJoCo simulation tasks (closed-loop success rate)."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=PROJECT_ROOT
        / "runs/dexjoco_ego_uncond_1cam_384_1e-4/2026-06-05_17-18-31",
        help="Training run directory (for eval settings / text cache paths).",
    )
    parser.add_argument("--policy-host", type=str, default="127.0.0.1")
    parser.add_argument("--policy-port", type=int, default=5560)
    parser.add_argument("--policy-timeout-ms", type=int, default=300000)
    parser.add_argument(
        "--task-config-dir",
        type=Path,
        default=DEFAULT_TASK_CONFIG_DIR,
        help="Directory with rand_obj/*.yaml task configs.",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="*",
        default=None,
        help="Subset of env_name values to evaluate (default: all 11 tasks).",
    )
    parser.add_argument("--episodes", type=int, default=50, help="Episodes per task.")
    parser.add_argument("--seed", type=int, default=0, help="Base env seed; episode index is added.")
    parser.add_argument(
        "--replan-steps",
        type=int,
        default=None,
        help="Actions to execute per policy query (default: 0.8 * action_horizon).",
    )
    parser.add_argument(
        "--max-env-steps",
        type=int,
        default=1500,
        help="Safety cap on environment steps per episode.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "logs/dexjoco_fastwam_eval",
        help="Root directory for eval logs and videos.",
    )
    parser.add_argument(
        "--save-video",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save ego/front camera mp4 per episode (default: true).",
    )
    parser.add_argument("--video-fps", type=int, default=30)
    parser.add_argument(
        "--randomize",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable rand_full scene randomization (default: false, matches training).",
    )
    parser.add_argument(
        "--randomize-dynamics",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Randomize dynamics at reset (default: false).",
    )
    parser.add_argument(
        "--save-actions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save per-episode executed actions and policy chunks as .npz (default: true).",
    )
    parser.add_argument(
        "--action-clip",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "After policy returns an action, clip it relative to current sim state before "
            "env.step (inference-time only; does not modify dataset or model)."
        ),
    )
    parser.add_argument(
        "--clip-max-xyz-step",
        type=float,
        default=0.05,
        help="Max xyz displacement from current state per step when --action-clip is on.",
    )
    parser.add_argument(
        "--clip-max-dz-down",
        type=float,
        default=0.03,
        help="Max downward z delta from current state per step when --action-clip is on.",
    )
    return parser.parse_args()


def _save_episode_video(frames: list[np.ndarray], path: Path, fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = Mp4VideoWriter.create_h264(
        fps=fps,
        codec="h264",
        input_pix_fmt="rgb24",
        crf=21,
        thread_type="FRAME",
        thread_count=2,
    )
    writer.start(str(path))
    for frame in frames:
        writer.write_frame(frame)
    writer.stop()


def _save_episode_actions(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def run_episode(
    env: DexJoCoFastWAMEvalEnv,
    policy: PolicyClient,
    adapter: DexJoCoFastWAMAdapter,
    *,
    replan_steps: int,
    max_env_steps: int,
    save_video: bool,
    save_actions: bool,
    action_clip: bool = False,
    action_clip_config: ActionConstraintConfig | None = None,
) -> dict[str, Any]:
    obs = env.reset()
    policy.reset()
    env.click_mouse_warmup()

    action_queue: deque[np.ndarray] = deque()
    frames: list[np.ndarray] = []
    executed_actions: list[np.ndarray] = []
    raw_policy_actions: list[np.ndarray] = []
    executed_is_stay: list[bool] = []
    policy_chunks: list[np.ndarray] = []
    policy_query_steps: list[int] = []
    policy_n_exec: list[int] = []
    if save_video:
        frames.append(env.get_camera_frame())

    steps = 0
    success = False
    in_stay = False
    initial_state = np.asarray(obs.get("state", []), dtype=np.float32).reshape(-1)

    while steps < max_env_steps:
        if not action_queue:
            policy_obs = env.build_policy_obs(adapter)
            response = policy.get_action(policy_obs)
            chunk = adapter.parse_policy_response(response)
            n_exec = max(1, min(replan_steps, chunk.shape[0]))
            if save_actions:
                policy_chunks.append(chunk.astype(np.float32))
                policy_query_steps.append(steps)
                policy_n_exec.append(n_exec)
            for idx in range(n_exec):
                action_queue.append(chunk[idx])

        if action_queue:
            rotvec_action = np.asarray(action_queue.popleft(), dtype=np.float32)
            raw_action = rotvec_action.copy()
            if action_clip and action_clip_config is not None:
                current_state = np.asarray(env._latest_obs["state"], dtype=np.float32).reshape(-1)
                rotvec_action = constrain_rotvec_action(
                    rotvec_action,
                    current_state,
                    dual_arm=env.task.dual_arm,
                    config=action_clip_config,
                )
            env.step_rotvec(rotvec_action)
            if save_actions:
                executed_actions.append(np.asarray(rotvec_action, dtype=np.float32))
                raw_policy_actions.append(raw_action)
                executed_is_stay.append(False)
            in_stay = False
        else:
            rotvec_action = env.stay(continue_stay=in_stay)
            if save_actions:
                executed_actions.append(np.asarray(rotvec_action, dtype=np.float32))
                executed_is_stay.append(True)
            in_stay = True

        if save_video:
            frames.append(env.get_camera_frame())
        steps += 1

        if env.is_done:
            success = env.is_success
            break

    result: dict[str, Any] = {"steps": steps, "success": success}
    if save_video:
        result["video_frames"] = frames
    if save_actions:
        result["action_log"] = {
            "initial_state": initial_state,
            "executed_actions": (
                np.stack(executed_actions, axis=0)
                if executed_actions
                else np.zeros((0, adapter.action_output_dim), dtype=np.float32)
            ),
            "raw_policy_actions": (
                np.stack(raw_policy_actions, axis=0)
                if raw_policy_actions
                else np.zeros((0, adapter.action_output_dim), dtype=np.float32)
            ),
            "executed_is_stay": np.asarray(executed_is_stay, dtype=bool),
            "policy_query_steps": np.asarray(policy_query_steps, dtype=np.int32),
            "policy_n_exec": np.asarray(policy_n_exec, dtype=np.int32),
            "policy_chunks": (
                np.stack(policy_chunks, axis=0)
                if policy_chunks
                else np.zeros((0, 0, adapter.action_output_dim), dtype=np.float32)
            ),
            "replan_steps": np.int32(replan_steps),
            "action_horizon": np.int32(adapter.action_horizon),
            "action_dim": np.int32(adapter.action_output_dim),
        }
    return result


def evaluate_task(
    task_cfg: dict[str, Any],
    *,
    policy: PolicyClient,
    adapter: DexJoCoFastWAMAdapter,
    episodes: int,
    seed: int,
    replan_steps: int,
    max_env_steps: int,
    save_video: bool,
    save_actions: bool,
    action_clip: bool,
    action_clip_config: ActionConstraintConfig | None,
    video_fps: int,
    task_output_dir: Path,
    randomize: bool,
    randomize_dynamics: bool,
) -> dict[str, Any]:
    task = DexJoCoTaskConfig.from_yaml(task_cfg)
    episode_results: list[dict[str, Any]] = []
    num_success = 0

    for ep in range(episodes):
        ep_seed = seed + ep
        print(f"  episode {ep + 1}/{episodes} (seed={ep_seed})", flush=True)
        env = DexJoCoFastWAMEvalEnv(
            task,
            seed=ep_seed,
            randomize=randomize,
            randomize_dynamics=randomize_dynamics,
        )
        try:
            t0 = time.perf_counter()
            stats = run_episode(
                env,
                policy,
                adapter,
                replan_steps=replan_steps,
                max_env_steps=max_env_steps,
                save_video=save_video,
                save_actions=save_actions,
                action_clip=action_clip,
                action_clip_config=action_clip_config,
            )
        finally:
            env.close()

        stats["episode"] = ep
        stats["seed"] = ep_seed
        stats["elapsed_s"] = time.perf_counter() - t0

        if save_video and "video_frames" in stats:
            suffix = "success" if stats["success"] else "failure"
            video_path = task_output_dir / f"episode_{ep:02d}_{suffix}.mp4"
            _save_episode_video(stats["video_frames"], video_path, video_fps)
            stats["video_path"] = str(video_path)
            del stats["video_frames"]

        if save_actions and "action_log" in stats:
            suffix = "success" if stats["success"] else "failure"
            action_path = task_output_dir / f"episode_{ep:02d}_{suffix}_actions.npz"
            _save_episode_actions(action_path, stats["action_log"])
            stats["actions_path"] = str(action_path)
            del stats["action_log"]

        if stats["success"]:
            num_success += 1
        episode_results.append(stats)
        print(
            f"    -> success={stats['success']} steps={stats['steps']} "
            f"elapsed={stats['elapsed_s']:.1f}s",
            flush=True,
        )

    success_rate = num_success / episodes if episodes else 0.0
    return {
        "env_name": task.env_name,
        "prompt": task.prompt,
        "dual_arm": task.dual_arm,
        "camera_key": task.camera_key,
        "episodes": episodes,
        "successes": num_success,
        "success_rate": success_rate,
        "episode_results": episode_results,
    }


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"run dir not found: {run_dir}")

    eval_settings = load_dexjoco_eval_settings(run_dir)
    adapter = DexJoCoFastWAMAdapter(eval_settings)
    replan_steps = args.replan_steps
    if replan_steps is None:
        replan_steps = max(1, int(0.8 * adapter.action_horizon))

    action_clip_config: ActionConstraintConfig | None = None
    if args.action_clip:
        action_clip_config = ActionConstraintConfig(
            max_xyz_step=args.clip_max_xyz_step,
            max_dz_down=args.clip_max_dz_down,
            clip_to_dataset_bounds=False,
        )

    all_configs = load_task_configs(args.task_config_dir)
    if args.tasks:
        selected = {name for name in args.tasks}
        task_configs = [cfg for cfg in all_configs if cfg["env_name"] in selected]
        missing = selected - {cfg["env_name"] for cfg in task_configs}
        if missing:
            raise ValueError(f"Unknown task names: {sorted(missing)}")
    else:
        task_configs = all_configs

    session_dir = args.output_dir.expanduser().resolve() / datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir.mkdir(parents=True, exist_ok=True)

    print(f"[eval] run_dir={run_dir}", flush=True)
    print(
        f"[eval] policy={args.policy_host}:{args.policy_port} "
        f"replan_steps={replan_steps} action_horizon={adapter.action_horizon} "
        f"action_clip={args.action_clip}",
        flush=True,
    )
    if args.action_clip:
        print(
            f"[eval] action_clip: max_xyz_step={args.clip_max_xyz_step} "
            f"max_dz_down={args.clip_max_dz_down}",
            flush=True,
        )
    print(f"[eval] tasks={len(task_configs)} episodes_per_task={args.episodes}", flush=True)
    print(f"[eval] output={session_dir}", flush=True)

    print("[eval] connecting to FastWAM policy server...", flush=True)
    policy = PolicyClient(
        host=args.policy_host,
        port=args.policy_port,
        timeout_ms=args.policy_timeout_ms,
    )
    if not policy.ping():
        raise RuntimeError(
            f"FastWAM policy server ping failed at {args.policy_host}:{args.policy_port}. "
            "Start scripts/run_fastwam_server.py first."
        )

    per_task: list[dict[str, Any]] = []
    total_success = 0
    total_episodes = 0

    try:
        for task_idx, task_cfg in enumerate(task_configs):
            env_name = task_cfg["env_name"]
            print(f"\n[eval] task {task_idx + 1}/{len(task_configs)}: {env_name}", flush=True)
            task_output_dir = session_dir / env_name
            task_output_dir.mkdir(parents=True, exist_ok=True)

            task_summary = evaluate_task(
                task_cfg,
                policy=policy,
                adapter=adapter,
                episodes=args.episodes,
                seed=args.seed,
                replan_steps=replan_steps,
                max_env_steps=args.max_env_steps,
                save_video=args.save_video,
                save_actions=args.save_actions,
                action_clip=args.action_clip,
                action_clip_config=action_clip_config,
                video_fps=args.video_fps,
                task_output_dir=task_output_dir,
                randomize=args.randomize,
                randomize_dynamics=args.randomize_dynamics,
            )
            per_task.append(task_summary)
            total_success += int(task_summary["successes"])
            total_episodes += int(task_summary["episodes"])
            print(
                f"[eval] {env_name}: {task_summary['successes']}/{task_summary['episodes']} "
                f"({100 * task_summary['success_rate']:.1f}%)",
                flush=True,
            )
    finally:
        policy.close()

    overall_rate = total_success / total_episodes if total_episodes else 0.0
    summary = {
        "run_dir": str(run_dir),
        "policy_host": args.policy_host,
        "policy_port": args.policy_port,
        "replan_steps": replan_steps,
        "action_horizon": adapter.action_horizon,
        "episodes_per_task": args.episodes,
        "num_tasks": len(task_configs),
        "total_episodes": total_episodes,
        "total_successes": total_success,
        "overall_success_rate": overall_rate,
        "randomize": args.randomize,
        "randomize_dynamics": args.randomize_dynamics,
        "save_actions": args.save_actions,
        "action_clip": args.action_clip,
        "clip_max_xyz_step": args.clip_max_xyz_step,
        "clip_max_dz_down": args.clip_max_dz_down,
        "seed": args.seed,
        "tasks": per_task,
    }

    summary_path = session_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(
        f"\n[eval] overall success rate: {total_success}/{total_episodes} "
        f"({100 * overall_rate:.1f}%)",
        flush=True,
    )
    print(f"[eval] summary saved to {summary_path}", flush=True)


if __name__ == "__main__":
    main()
