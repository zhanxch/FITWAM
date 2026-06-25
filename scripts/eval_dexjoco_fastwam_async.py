#!/usr/bin/env python3
"""Async / batched closed-loop DexJoCo evaluation for FastWAM policy server."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import threading
import time
import uuid
from collections import deque
from datetime import datetime
from pathlib import Path
from queue import Empty
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
    DexJoCoFastWAMAdapter,
    DexJoCoFastWAMEvalEnv,
    DexJoCoTaskConfig,
    KEY_ACTION,
    load_dexjoco_eval_settings,
    load_task_configs,
)
from policy_zmq_client_async import PolicyClientAsync


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Async DexJoCo eval: parallel episode workers + optional batched infer."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=PROJECT_ROOT
        / "runs/dexjoco_ego_uncond_1cam_384_1e-4/2026-06-05_17-18-31",
    )
    parser.add_argument("--policy-host", type=str, default="127.0.0.1")
    parser.add_argument("--policy-port", type=int, default=5561)
    parser.add_argument("--policy-timeout-ms", type=int, default=300000)
    parser.add_argument("--task-config-dir", type=Path, default=DEFAULT_TASK_CONFIG_DIR)
    parser.add_argument("--tasks", type=str, nargs="*", default=None)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of episodes to run in parallel (worker processes).",
    )
    parser.add_argument(
        "--infer-batch-size",
        type=int,
        default=None,
        help="Observations per get_actions_batch call (default: --batch-size).",
    )
    parser.add_argument(
        "--use-batch-infer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Group worker infer requests via get_actions_batch (default: true).",
    )
    parser.add_argument(
        "--infer-gather-ms",
        type=int,
        default=50,
        help="Max wait when collecting a batch of infer requests.",
    )
    parser.add_argument("--replan-steps", type=int, default=None)
    parser.add_argument("--max-env-steps", type=int, default=1500)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "logs/dexjoco_fastwam_eval_async",
    )
    parser.add_argument("--save-video", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--video-fps", type=int, default=30)
    parser.add_argument("--randomize", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--randomize-dynamics",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--save-actions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save per-episode executed actions and policy chunks as .npz (default: true).",
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


class BatchInferenceBridge:
    """Collect worker infer requests and dispatch get_actions_batch on the async server."""

    def __init__(
        self,
        *,
        host: str,
        port: int,
        timeout_ms: int,
        infer_batch_size: int,
        gather_timeout_s: float,
        request_queue: mp.Queue,
        response_queue: mp.Queue,
    ) -> None:
        self.infer_batch_size = max(1, int(infer_batch_size))
        self.gather_timeout_s = float(gather_timeout_s)
        self.request_queue = request_queue
        self.response_queue = response_queue
        self._stop = threading.Event()
        self._client = PolicyClientAsync(host=host, port=port, timeout_ms=timeout_ms)
        self._thread = threading.Thread(target=self._loop, name="batch-infer", daemon=True)

    def start(self) -> None:
        if not self._client.ping():
            raise RuntimeError("Async policy server ping failed.")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
        self._client.close()

    def _collect_batch(self) -> list[tuple[str, dict[str, Any]]]:
        batch: list[tuple[str, dict[str, Any]]] = []
        deadline = time.perf_counter() + self.gather_timeout_s
        while len(batch) < self.infer_batch_size:
            timeout = max(0.0, deadline - time.perf_counter())
            if batch and timeout <= 0.0:
                break
            try:
                item = self.request_queue.get(timeout=timeout if timeout > 0 else 0.001)
            except Empty:
                if batch:
                    break
                if time.perf_counter() >= deadline:
                    break
                continue
            batch.append(item)
        return batch

    def _loop(self) -> None:
        while not self._stop.is_set():
            batch = self._collect_batch()
            if not batch:
                continue
            observations = [obs for _req_id, obs in batch]
            try:
                payload = self._client.get_actions_batch(observations)
                actions = payload["actions"]
            except Exception as exc:
                for req_id, _obs in batch:
                    self.response_queue.put((req_id, {"error": str(exc)}))
                continue
            if len(actions) != len(batch):
                err = (
                    f"get_actions_batch returned {len(actions)} actions for "
                    f"{len(batch)} observations"
                )
                for req_id, _obs in batch:
                    self.response_queue.put((req_id, {"error": err}))
                continue
            for (req_id, _obs), action in zip(batch, actions):
                self.response_queue.put((req_id, {KEY_ACTION: action}))


class _QueueInferClient:
    """Worker-side client that forwards infer requests to the main-process bridge."""

    def __init__(self, request_queue: mp.Queue, response_queue: mp.Queue) -> None:
        self.request_queue = request_queue
        self.response_queue = response_queue

    def ping(self) -> bool:
        return True

    def reset(self) -> None:
        return None

    def get_action(self, observation: dict[str, Any]) -> tuple[dict, dict]:
        req_id = uuid.uuid4().hex
        self.request_queue.put((req_id, observation))
        while True:
            resp_id, payload = self.response_queue.get()
            if resp_id != req_id:
                continue
            if "error" in payload:
                raise RuntimeError(payload["error"])
            return payload, {}

    def close(self) -> None:
        return None


class _PolicyFacade:
    """Unified policy interface for direct async client or batch infer queues."""

    def __init__(
        self,
        *,
        host: str,
        port: int,
        timeout_ms: int,
        worker_id: int,
        request_queue: mp.Queue | None,
        response_queue: mp.Queue | None,
    ) -> None:
        if request_queue is not None and response_queue is not None:
            self._client: _QueueInferClient | PolicyClientAsync = _QueueInferClient(
                request_queue, response_queue
            )
        else:
            self._client = PolicyClientAsync(
                host=host,
                port=port,
                timeout_ms=timeout_ms,
                identity=f"dexjoco-worker-{worker_id}",
            )

    def ping(self) -> bool:
        return self._client.ping()

    def reset(self) -> None:
        self._client.reset()

    def get_action(self, observation: dict[str, Any]) -> tuple[dict, dict]:
        return self._client.get_action(observation)

    def close(self) -> None:
        self._client.close()


def _save_episode_actions(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def run_episode(
    env: DexJoCoFastWAMEvalEnv,
    policy: _PolicyFacade,
    adapter: DexJoCoFastWAMAdapter,
    *,
    replan_steps: int,
    max_env_steps: int,
    save_video: bool,
    save_actions: bool,
) -> dict[str, Any]:
    obs = env.reset()
    policy.reset()
    env.click_mouse_warmup()

    action_queue: deque[np.ndarray] = deque()
    frames: list[np.ndarray] = []
    executed_actions: list[np.ndarray] = []
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
            rotvec_action = action_queue.popleft()
            env.step_rotvec(rotvec_action)
            if save_actions:
                executed_actions.append(np.asarray(rotvec_action, dtype=np.float32))
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


def _episode_worker(job: dict[str, Any]) -> dict[str, Any]:
    task = DexJoCoTaskConfig.from_yaml(job["task_cfg"])
    adapter = DexJoCoFastWAMAdapter(job["eval_settings"])
    policy = _PolicyFacade(
        host=job["policy_host"],
        port=job["policy_port"],
        timeout_ms=job["policy_timeout_ms"],
        worker_id=job["worker_id"],
        request_queue=job.get("request_queue"),
        response_queue=job.get("response_queue"),
    )

    ep = int(job["episode"])
    ep_seed = int(job["seed"]) + ep
    env = DexJoCoFastWAMEvalEnv(
        task,
        seed=ep_seed,
        randomize=bool(job["randomize"]),
        randomize_dynamics=bool(job["randomize_dynamics"]),
    )

    try:
        if job.get("request_queue") is None and not policy.ping():
            raise RuntimeError(
                f"Async policy server ping failed at {job['policy_host']}:{job['policy_port']}"
            )
        t0 = time.perf_counter()
        stats = run_episode(
            env,
            policy,
            adapter,
            replan_steps=int(job["replan_steps"]),
            max_env_steps=int(job["max_env_steps"]),
            save_video=bool(job["save_video"]),
            save_actions=bool(job["save_actions"]),
        )
    finally:
        policy.close()
        env.close()

    stats["episode"] = ep
    stats["seed"] = ep_seed
    stats["elapsed_s"] = time.perf_counter() - t0

    if job["save_video"] and "video_frames" in stats:
        suffix = "success" if stats["success"] else "failure"
        video_path = Path(job["task_output_dir"]) / f"episode_{ep:02d}_{suffix}.mp4"
        _save_episode_video(stats["video_frames"], video_path, int(job["video_fps"]))
        stats["video_path"] = str(video_path)
        del stats["video_frames"]

    if job["save_actions"] and "action_log" in stats:
        suffix = "success" if stats["success"] else "failure"
        action_path = Path(job["task_output_dir"]) / f"episode_{ep:02d}_{suffix}_actions.npz"
        _save_episode_actions(action_path, stats["action_log"])
        stats["actions_path"] = str(action_path)
        del stats["action_log"]

    return stats


def evaluate_task_async(
    task_cfg: dict[str, Any],
    *,
    eval_settings: dict[str, Any],
    episodes: int,
    seed: int,
    batch_size: int,
    infer_batch_size: int,
    use_batch_infer: bool,
    infer_gather_ms: int,
    replan_steps: int,
    max_env_steps: int,
    save_video: bool,
    save_actions: bool,
    video_fps: int,
    task_output_dir: Path,
    randomize: bool,
    randomize_dynamics: bool,
    policy_host: str,
    policy_port: int,
    policy_timeout_ms: int,
) -> dict[str, Any]:
    task = DexJoCoTaskConfig.from_yaml(task_cfg)
    task_output_dir.mkdir(parents=True, exist_ok=True)

    ctx = mp.get_context("spawn")
    request_queue: mp.Queue | None = None
    response_queue: mp.Queue | None = None
    bridge: BatchInferenceBridge | None = None

    if use_batch_infer:
        request_queue = ctx.Queue()
        response_queue = ctx.Queue()
        bridge = BatchInferenceBridge(
            host=policy_host,
            port=policy_port,
            timeout_ms=policy_timeout_ms,
            infer_batch_size=infer_batch_size,
            gather_timeout_s=infer_gather_ms / 1000.0,
            request_queue=request_queue,
            response_queue=response_queue,
        )
        bridge.start()
    else:
        probe = PolicyClientAsync(host=policy_host, port=policy_port, timeout_ms=policy_timeout_ms)
        if not probe.ping():
            probe.close()
            raise RuntimeError(
                f"Async policy server ping failed at {policy_host}:{policy_port}. "
                "Start scripts/run_fastwam_server_async.py first."
            )
        probe.close()

    jobs = [
        {
            "task_cfg": task_cfg,
            "eval_settings": eval_settings,
            "episode": ep,
            "seed": seed,
            "worker_id": ep % max(1, batch_size),
            "replan_steps": replan_steps,
            "max_env_steps": max_env_steps,
            "save_video": save_video,
            "save_actions": save_actions,
            "video_fps": video_fps,
            "task_output_dir": str(task_output_dir),
            "randomize": randomize,
            "randomize_dynamics": randomize_dynamics,
            "policy_host": policy_host,
            "policy_port": policy_port,
            "policy_timeout_ms": policy_timeout_ms,
            "request_queue": request_queue,
            "response_queue": response_queue,
        }
        for ep in range(episodes)
    ]

    episode_results: list[dict[str, Any]] = []
    try:
        with ctx.Pool(processes=max(1, batch_size)) as pool:
            for ep_idx, stats in enumerate(
                pool.imap_unordered(_episode_worker, jobs, chunksize=1),
                start=1,
            ):
                episode_results.append(stats)
                print(
                    f"  [{task.env_name}] finished {ep_idx}/{episodes}: "
                    f"ep={stats['episode']} success={stats['success']} "
                    f"steps={stats['steps']} elapsed={stats['elapsed_s']:.1f}s",
                    flush=True,
                )
    finally:
        if bridge is not None:
            bridge.stop()

    episode_results.sort(key=lambda item: int(item["episode"]))
    num_success = sum(1 for item in episode_results if item["success"])
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
    replan_steps = args.replan_steps
    if replan_steps is None:
        replan_steps = max(1, int(0.8 * (int(eval_settings["action_horizon"]))))

    infer_batch_size = args.infer_batch_size or args.batch_size
    batch_size = max(1, int(args.batch_size))
    infer_batch_size = max(1, int(infer_batch_size))

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

    print(f"[eval-async] run_dir={run_dir}", flush=True)
    print(
        f"[eval-async] policy={args.policy_host}:{args.policy_port} "
        f"batch_size={batch_size} infer_batch_size={infer_batch_size} "
        f"use_batch_infer={args.use_batch_infer}",
        flush=True,
    )
    print(
        f"[eval-async] replan_steps={replan_steps} "
        f"action_horizon={eval_settings['action_horizon']}",
        flush=True,
    )
    print(f"[eval-async] tasks={len(task_configs)} episodes_per_task={args.episodes}", flush=True)
    print(f"[eval-async] output={session_dir}", flush=True)

    per_task: list[dict[str, Any]] = []
    total_success = 0
    total_episodes = 0

    for task_idx, task_cfg in enumerate(task_configs):
        env_name = task_cfg["env_name"]
        print(f"\n[eval-async] task {task_idx + 1}/{len(task_configs)}: {env_name}", flush=True)
        task_summary = evaluate_task_async(
            task_cfg,
            eval_settings=eval_settings,
            episodes=args.episodes,
            seed=args.seed,
            batch_size=batch_size,
            infer_batch_size=infer_batch_size,
            use_batch_infer=args.use_batch_infer,
            infer_gather_ms=args.infer_gather_ms,
            replan_steps=replan_steps,
            max_env_steps=args.max_env_steps,
            save_video=args.save_video,
            save_actions=args.save_actions,
            video_fps=args.video_fps,
            task_output_dir=session_dir / env_name,
            randomize=args.randomize,
            randomize_dynamics=args.randomize_dynamics,
            policy_host=args.policy_host,
            policy_port=args.policy_port,
            policy_timeout_ms=args.policy_timeout_ms,
        )
        per_task.append(task_summary)
        total_success += int(task_summary["successes"])
        total_episodes += int(task_summary["episodes"])
        print(
            f"[eval-async] {env_name}: {task_summary['successes']}/{task_summary['episodes']} "
            f"({100 * task_summary['success_rate']:.1f}%)",
            flush=True,
        )

    overall_rate = total_success / total_episodes if total_episodes else 0.0
    summary = {
        "run_dir": str(run_dir),
        "policy_host": args.policy_host,
        "policy_port": args.policy_port,
        "batch_size": batch_size,
        "infer_batch_size": infer_batch_size,
        "use_batch_infer": args.use_batch_infer,
        "infer_gather_ms": args.infer_gather_ms,
        "replan_steps": replan_steps,
        "action_horizon": eval_settings["action_horizon"],
        "episodes_per_task": args.episodes,
        "num_tasks": len(task_configs),
        "total_episodes": total_episodes,
        "total_successes": total_success,
        "overall_success_rate": overall_rate,
        "randomize": args.randomize,
        "randomize_dynamics": args.randomize_dynamics,
        "seed": args.seed,
        "tasks": per_task,
    }

    summary_path = session_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(
        f"\n[eval-async] overall success rate: {total_success}/{total_episodes} "
        f"({100 * overall_rate:.1f}%)",
        flush=True,
    )
    print(f"[eval-async] summary saved to {summary_path}", flush=True)


if __name__ == "__main__":
    main()
