#!/usr/bin/env python3
"""DexJoCo closed-loop eval with overlap inference, LPF, and action metrics."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
SRC_ROOT = PROJECT_ROOT / "src"
THIS_DIR = Path(__file__).resolve().parent
for path in (PROJECT_ROOT, SRC_ROOT, SCRIPTS_ROOT, THIS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

try:
    from dexjoco.data.video_writer import Mp4VideoWriter
except ModuleNotFoundError:
    import imageio.v2 as imageio

    class Mp4VideoWriter:
        """Small fallback for DexJoCo checkouts without dexjoco.data.video_writer."""

        def __init__(self, fps: int):
            self.fps = int(fps)
            self._writer = None

        @classmethod
        def create_h264(cls, fps: int, **_: Any) -> "Mp4VideoWriter":
            return cls(fps=fps)

        def start(self, path: str) -> None:
            self._writer = imageio.get_writer(
                path,
                fps=self.fps,
                codec="libx264",
                macro_block_size=1,
            )

        def write_frame(self, frame: np.ndarray) -> None:
            if self._writer is None:
                raise RuntimeError("video writer is not started")
            self._writer.append_data(np.asarray(frame, dtype=np.uint8))

        def stop(self) -> None:
            if self._writer is not None:
                self._writer.close()
                self._writer = None
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
from policy_client_async import PolicyClientAsync


def _patch_dexjoco_renderer_compat() -> None:
    """Handle Gymnasium MuJoCo renderer constructor drift across versions."""
    try:
        import inspect

        from gymnasium.envs.mujoco.mujoco_rendering import (
            MujocoRenderer as GymnasiumMujocoRenderer,
        )
        import dexjoco.sim.rendering as dexjoco_rendering
    except Exception:
        return

    signature = inspect.signature(GymnasiumMujocoRenderer.__init__)
    if "width" in signature.parameters and "height" in signature.parameters:
        return
    if getattr(dexjoco_rendering.MujocoRenderer, "_fastwam_compat_patched", False):
        return

    def _compat_init(self, model, data, *args, width=None, height=None, **kwargs):
        default_cam_config = kwargs.pop("default_cam_config", None)
        if args:
            if len(args) > 1:
                raise TypeError(
                    "Gymnasium MujocoRenderer compatibility patch accepts at most "
                    "one positional default_cam_config argument."
                )
            if default_cam_config is None:
                default_cam_config = args[0]
        GymnasiumMujocoRenderer.__init__(
            self,
            model,
            data,
            default_cam_config=default_cam_config,
        )

    dexjoco_rendering.MujocoRenderer.__init__ = _compat_init
    dexjoco_rendering.MujocoRenderer._fastwam_compat_patched = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate FastWAM on DexJoCo with blocking or overlapped inference."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--text-embedding-cache-dir",
        type=Path,
        default=None,
        help="Optional runtime relocation of the cached task context directory.",
    )
    parser.add_argument("--policy-host", type=str, default="127.0.0.1")
    parser.add_argument("--policy-port", type=int, required=True)
    parser.add_argument("--policy-timeout-ms", type=int, default=300000)
    parser.add_argument("--task-config-dir", type=Path, default=DEFAULT_TASK_CONFIG_DIR)
    parser.add_argument("--tasks", type=str, nargs="*", default=None)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--replan-steps", type=int, required=True)
    parser.add_argument(
        "--action-horizon",
        type=int,
        default=None,
        help="Closed-loop action chunk size. Required for EveRobot full-episode runs "
        "(training has no fixed num_frames). For sliding-window runs, defaults to "
        "config num_frames-1. Must match the policy server --action-horizon.",
    )
    parser.add_argument("--max-env-steps", type=int, default=1500)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--save-video", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--video-fps", type=int, default=30)
    parser.add_argument("--randomize", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--randomize-dynamics",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--save-actions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--control-mode",
        choices=["blocking", "overlap"],
        default="blocking",
        help="blocking waits at each replan; overlap submits the next chunk while executing old actions.",
    )
    parser.add_argument(
        "--async-fallback",
        choices=["wait", "hold_last"],
        default="wait",
        help="Behavior if no predicted action is valid at the current control step.",
    )
    parser.add_argument("--low-pass-alpha", type=float, default=None)
    parser.add_argument(
        "--low-pass-continuous-dim",
        type=int,
        default=None,
        help="Number of leading action dims to LPF. Default: all dims for DexJoCo.",
    )
    parser.add_argument("--action-clip", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--clip-max-xyz-step", type=float, default=0.05)
    parser.add_argument("--clip-max-dz-down", type=float, default=0.03)
    return parser.parse_args()


def _copy_for_async(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {k: _copy_for_async(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_copy_for_async(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_copy_for_async(v) for v in value)
    return value


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


def _new_metrics(args: argparse.Namespace, *, action_horizon: int, action_dim: int) -> dict[str, Any]:
    if args.low_pass_alpha is not None and not 0.0 < args.low_pass_alpha <= 1.0:
        raise ValueError(f"--low-pass-alpha must be in (0, 1], got {args.low_pass_alpha}")
    continuous_dim = args.low_pass_continuous_dim
    if continuous_dim is None:
        continuous_dim = action_dim
    if continuous_dim < 0 or continuous_dim > action_dim:
        raise ValueError(
            f"--low-pass-continuous-dim must be in [0, {action_dim}], got {continuous_dim}"
        )
    return {
        "control_mode": args.control_mode,
        "replan_steps": int(args.replan_steps),
        "action_horizon": int(action_horizon),
        "overlap_steps": int(max(0, action_horizon - args.replan_steps)),
        "overlap_ratio": float(max(0, action_horizon - args.replan_steps) / action_horizon),
        "low_pass_enabled": args.low_pass_alpha is not None,
        "low_pass_alpha": args.low_pass_alpha,
        "low_pass_continuous_dim": int(continuous_dim),
        "async_fallback": args.async_fallback,
        "action_clip": bool(args.action_clip),
        "queue_underruns": 0,
        "blocking_wait_events": 0,
        "queue_wait_s": 0.0,
        "async_replan_delays": 0,
        "fresh_chunk_actions": 0,
        "stale_chunk_actions": 0,
        "hold_last_actions": 0,
        "inference_count": 0,
        "_inference_latencies_s": [],
        "_control_periods_s": [],
        "_action_delta_l2_sum": 0.0,
        "_action_delta_l2_count": 0,
        "_action_jerk_l2_sum": 0.0,
        "_action_jerk_l2_count": 0,
        "_sign_flip_count": 0,
        "_sign_flip_total": 0,
        "_prev_action": None,
        "_prev_delta": None,
    }


def _finalize_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    out = {k: v for k, v in metrics.items() if not k.startswith("_")}
    latencies = metrics["_inference_latencies_s"]
    periods = metrics["_control_periods_s"]
    out["inference_latency_mean_s"] = float(np.mean(latencies)) if latencies else None
    out["inference_latency_p95_s"] = float(np.percentile(latencies, 95)) if latencies else None
    out["inference_latency_max_s"] = float(np.max(latencies)) if latencies else None
    out["control_period_mean_s"] = float(np.mean(periods)) if periods else None
    out["control_period_p95_s"] = float(np.percentile(periods, 95)) if periods else None
    out["action_delta_l2_mean"] = (
        metrics["_action_delta_l2_sum"] / metrics["_action_delta_l2_count"]
        if metrics["_action_delta_l2_count"]
        else None
    )
    out["action_jerk_l2_mean"] = (
        metrics["_action_jerk_l2_sum"] / metrics["_action_jerk_l2_count"]
        if metrics["_action_jerk_l2_count"]
        else None
    )
    out["oscillation_sign_flip_rate"] = (
        metrics["_sign_flip_count"] / metrics["_sign_flip_total"]
        if metrics["_sign_flip_total"]
        else None
    )
    return out


def _record_action_metrics(metrics: dict[str, Any], action: np.ndarray) -> None:
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    prev_action = metrics["_prev_action"]
    if prev_action is None:
        metrics["_prev_action"] = action.copy()
        return
    delta = action - prev_action
    metrics["_action_delta_l2_sum"] += float(np.linalg.norm(delta))
    metrics["_action_delta_l2_count"] += 1
    prev_delta = metrics["_prev_delta"]
    if prev_delta is not None:
        jerk = delta - prev_delta
        metrics["_action_jerk_l2_sum"] += float(np.linalg.norm(jerk))
        metrics["_action_jerk_l2_count"] += 1
        active = (np.abs(delta) > 1e-6) & (np.abs(prev_delta) > 1e-6)
        metrics["_sign_flip_count"] += int(np.sum(active & ((delta * prev_delta) < 0.0)))
        metrics["_sign_flip_total"] += int(np.sum(active))
    metrics["_prev_delta"] = delta.copy()
    metrics["_prev_action"] = action.copy()


def _apply_low_pass(
    action: np.ndarray,
    state: np.ndarray | None,
    metrics: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray | None]:
    alpha = metrics["low_pass_alpha"]
    dim = int(metrics["low_pass_continuous_dim"])
    if alpha is None or dim == 0:
        return action, state
    action = np.asarray(action, dtype=np.float32).copy()
    if state is None:
        state = action[:dim].copy()
        return action, state
    filtered = float(alpha) * action[:dim] + (1.0 - float(alpha)) * state
    action[:dim] = filtered
    return action, filtered.copy()


def _predict_chunk(
    client: PolicyClientAsync,
    adapter: DexJoCoFastWAMAdapter,
    policy_obs: dict[str, Any],
    *,
    start_step: int,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    response = client.get_action(_copy_for_async(policy_obs))
    chunk = adapter.parse_policy_response(response)
    return {
        "start_step": int(start_step),
        "chunk": chunk.astype(np.float32),
        "latency_s": float(time.perf_counter() - t0),
    }


def _select_action_from_chunks(
    chunks: list[dict[str, Any]],
    step: int,
    action_horizon: int,
) -> tuple[np.ndarray | None, dict[str, Any] | None, int | None]:
    best: dict[str, Any] | None = None
    best_offset: int | None = None
    for item in chunks:
        offset = step - int(item["start_step"])
        if 0 <= offset < min(action_horizon, item["chunk"].shape[0]):
            if best is None or int(item["start_step"]) > int(best["start_step"]):
                best = item
                best_offset = offset
    if best is None or best_offset is None:
        return None, None, None
    return np.asarray(best["chunk"][best_offset], dtype=np.float32), best, best_offset


def _append_prediction(
    prediction: dict[str, Any],
    chunks: list[dict[str, Any]],
    metrics: dict[str, Any],
    *,
    current_step: int,
    action_horizon: int,
) -> None:
    chunks.append(prediction)
    metrics["inference_count"] += 1
    metrics["_inference_latencies_s"].append(float(prediction["latency_s"]))
    cutoff = current_step - action_horizon
    chunks[:] = [item for item in chunks if int(item["start_step"]) >= cutoff]


def run_episode(
    task_cfg: dict[str, Any],
    adapter: DexJoCoFastWAMAdapter,
    args: argparse.Namespace,
    *,
    episode: int,
    seed: int,
) -> dict[str, Any]:
    _patch_dexjoco_renderer_compat()
    task = DexJoCoTaskConfig.from_yaml(task_cfg)
    env = DexJoCoFastWAMEvalEnv(
        task,
        seed=seed,
        randomize=bool(args.randomize),
        randomize_dynamics=bool(args.randomize_dynamics),
    )
    control_client = PolicyClientAsync(
        host=args.policy_host,
        port=args.policy_port,
        timeout_ms=args.policy_timeout_ms,
        identity=f"dexjoco-control-{episode}",
    )
    infer_client = PolicyClientAsync(
        host=args.policy_host,
        port=args.policy_port,
        timeout_ms=args.policy_timeout_ms,
        identity=f"dexjoco-infer-{episode}",
    )
    action_clip_config = None
    if args.action_clip:
        action_clip_config = ActionConstraintConfig(
            max_xyz_step=float(args.clip_max_xyz_step),
            max_dz_down=float(args.clip_max_dz_down),
            clip_to_dataset_bounds=False,
        )

    try:
        obs = env.reset()
        control_client.reset()
        env.click_mouse_warmup()

        metrics = _new_metrics(
            args,
            action_horizon=adapter.action_horizon,
            action_dim=adapter.action_output_dim,
        )
        chunks: list[dict[str, Any]] = []
        future: Future | None = None
        next_submit_step = 0
        low_pass_state: np.ndarray | None = None
        last_action: np.ndarray | None = None

        frames: list[np.ndarray] = []
        executed_actions: list[np.ndarray] = []
        raw_policy_actions: list[np.ndarray] = []
        low_pass_actions: list[np.ndarray] = []
        executed_is_fallback: list[bool] = []
        policy_chunks: list[np.ndarray] = []
        policy_query_steps: list[int] = []
        policy_arrival_steps: list[int] = []
        policy_latencies: list[float] = []

        if args.save_video:
            frames.append(env.get_camera_frame())

        steps = 0
        success = False
        initial_state = np.asarray(obs.get("state", []), dtype=np.float32).reshape(-1)
        last_loop_t = time.perf_counter()

        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="dexjoco-overlap") as executor:
            while steps < args.max_env_steps:
                loop_t = time.perf_counter()
                metrics["_control_periods_s"].append(float(loop_t - last_loop_t))
                last_loop_t = loop_t

                if future is not None and future.done():
                    pred = future.result()
                    _append_prediction(
                        pred,
                        chunks,
                        metrics,
                        current_step=steps,
                        action_horizon=adapter.action_horizon,
                    )
                    policy_chunks.append(pred["chunk"].astype(np.float32))
                    policy_query_steps.append(int(pred["start_step"]))
                    policy_arrival_steps.append(int(steps))
                    policy_latencies.append(float(pred["latency_s"]))
                    future = None

                if args.control_mode == "blocking":
                    if steps >= next_submit_step:
                        policy_obs = env.build_policy_obs(adapter)
                        pred = _predict_chunk(
                            infer_client,
                            adapter,
                            policy_obs,
                            start_step=steps,
                        )
                        _append_prediction(
                            pred,
                            chunks,
                            metrics,
                            current_step=steps,
                            action_horizon=adapter.action_horizon,
                        )
                        policy_chunks.append(pred["chunk"].astype(np.float32))
                        policy_query_steps.append(int(pred["start_step"]))
                        policy_arrival_steps.append(int(steps))
                        policy_latencies.append(float(pred["latency_s"]))
                        next_submit_step = steps + int(args.replan_steps)
                else:
                    if steps >= next_submit_step:
                        if future is None:
                            policy_obs = env.build_policy_obs(adapter)
                            future = executor.submit(
                                _predict_chunk,
                                infer_client,
                                adapter,
                                policy_obs,
                                start_step=steps,
                            )
                            next_submit_step = steps + int(args.replan_steps)
                        else:
                            metrics["async_replan_delays"] += 1

                raw_action, source_chunk, offset = _select_action_from_chunks(
                    chunks,
                    steps,
                    adapter.action_horizon,
                )

                used_fallback = False
                if raw_action is None:
                    metrics["queue_underruns"] += 1
                    used_fallback = True
                    if args.async_fallback == "wait" and future is not None:
                        wait_start = time.perf_counter()
                        pred = future.result()
                        metrics["blocking_wait_events"] += 1
                        metrics["queue_wait_s"] += float(time.perf_counter() - wait_start)
                        _append_prediction(
                            pred,
                            chunks,
                            metrics,
                            current_step=steps,
                            action_horizon=adapter.action_horizon,
                        )
                        policy_chunks.append(pred["chunk"].astype(np.float32))
                        policy_query_steps.append(int(pred["start_step"]))
                        policy_arrival_steps.append(int(steps))
                        policy_latencies.append(float(pred["latency_s"]))
                        future = None
                        raw_action, source_chunk, offset = _select_action_from_chunks(
                            chunks,
                            steps,
                            adapter.action_horizon,
                        )
                    if raw_action is None and last_action is not None:
                        raw_action = last_action.copy()
                        metrics["hold_last_actions"] += 1
                    if raw_action is None:
                        policy_obs = env.build_policy_obs(adapter)
                        pred = _predict_chunk(
                            infer_client,
                            adapter,
                            policy_obs,
                            start_step=steps,
                        )
                        _append_prediction(
                            pred,
                            chunks,
                            metrics,
                            current_step=steps,
                            action_horizon=adapter.action_horizon,
                        )
                        policy_chunks.append(pred["chunk"].astype(np.float32))
                        policy_query_steps.append(int(pred["start_step"]))
                        policy_arrival_steps.append(int(steps))
                        policy_latencies.append(float(pred["latency_s"]))
                        raw_action, source_chunk, offset = _select_action_from_chunks(
                            chunks,
                            steps,
                            adapter.action_horizon,
                        )
                assert raw_action is not None

                if source_chunk is not None and offset is not None:
                    if int(offset) < int(args.replan_steps):
                        metrics["fresh_chunk_actions"] += 1
                    else:
                        metrics["stale_chunk_actions"] += 1

                filtered_action, low_pass_state = _apply_low_pass(
                    np.asarray(raw_action, dtype=np.float32),
                    low_pass_state,
                    metrics,
                )
                exec_action = filtered_action.copy()
                if args.action_clip and action_clip_config is not None:
                    current_state = np.asarray(env._latest_obs["state"], dtype=np.float32).reshape(-1)
                    exec_action = constrain_rotvec_action(
                        exec_action,
                        current_state,
                        dual_arm=env.task.dual_arm,
                        config=action_clip_config,
                    )

                env.step_rotvec(exec_action)
                last_action = exec_action.copy()
                _record_action_metrics(metrics, exec_action)

                if args.save_actions:
                    raw_policy_actions.append(np.asarray(raw_action, dtype=np.float32))
                    low_pass_actions.append(np.asarray(filtered_action, dtype=np.float32))
                    executed_actions.append(np.asarray(exec_action, dtype=np.float32))
                    executed_is_fallback.append(bool(used_fallback))
                if args.save_video:
                    frames.append(env.get_camera_frame())
                steps += 1
                if env.is_done:
                    success = env.is_success
                    break

        result: dict[str, Any] = {
            "episode": episode,
            "seed": seed,
            "steps": steps,
            "success": bool(success),
            "metrics": _finalize_metrics(metrics),
        }
        if args.save_video:
            result["video_frames"] = frames
        if args.save_actions:
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
                "low_pass_actions": (
                    np.stack(low_pass_actions, axis=0)
                    if low_pass_actions
                    else np.zeros((0, adapter.action_output_dim), dtype=np.float32)
                ),
                "executed_is_fallback": np.asarray(executed_is_fallback, dtype=bool),
                "policy_query_steps": np.asarray(policy_query_steps, dtype=np.int32),
                "policy_arrival_steps": np.asarray(policy_arrival_steps, dtype=np.int32),
                "policy_latencies": np.asarray(policy_latencies, dtype=np.float32),
                "policy_chunks": (
                    np.stack(policy_chunks, axis=0)
                    if policy_chunks
                    else np.zeros((0, 0, adapter.action_output_dim), dtype=np.float32)
                ),
                "replan_steps": np.int32(args.replan_steps),
                "action_horizon": np.int32(adapter.action_horizon),
                "action_dim": np.int32(adapter.action_output_dim),
            }
        return result
    finally:
        infer_client.close()
        control_client.close()
        env.close()


def _aggregate_metric(episode_results: list[dict[str, Any]], key: str) -> float | None:
    vals = [
        item.get("metrics", {}).get(key)
        for item in episode_results
        if item.get("metrics", {}).get(key) is not None
    ]
    return float(np.mean(vals)) if vals else None


def evaluate_task(
    task_cfg: dict[str, Any],
    adapter: DexJoCoFastWAMAdapter,
    args: argparse.Namespace,
    *,
    task_output_dir: Path,
) -> dict[str, Any]:
    task = DexJoCoTaskConfig.from_yaml(task_cfg)
    task_output_dir.mkdir(parents=True, exist_ok=True)
    episode_results: list[dict[str, Any]] = []
    num_success = 0

    for ep in range(int(args.episodes)):
        ep_seed = int(args.seed) + ep
        print(f"  episode {ep + 1}/{args.episodes} seed={ep_seed}", flush=True)
        t0 = time.perf_counter()
        stats = run_episode(task_cfg, adapter, args, episode=ep, seed=ep_seed)
        stats["elapsed_s"] = float(time.perf_counter() - t0)

        suffix = "success" if stats["success"] else "failure"
        if args.save_video and "video_frames" in stats:
            video_path = task_output_dir / f"episode_{ep:02d}_{suffix}.mp4"
            _save_episode_video(stats["video_frames"], video_path, int(args.video_fps))
            stats["video_path"] = str(video_path)
            del stats["video_frames"]
        if args.save_actions and "action_log" in stats:
            action_path = task_output_dir / f"episode_{ep:02d}_{suffix}_actions.npz"
            _save_episode_actions(action_path, stats["action_log"])
            stats["actions_path"] = str(action_path)
            del stats["action_log"]

        if stats["success"]:
            num_success += 1
        episode_results.append(stats)
        print(
            f"    -> success={stats['success']} steps={stats['steps']} "
            f"elapsed={stats['elapsed_s']:.1f}s jerk={stats['metrics'].get('action_jerk_l2_mean')}",
            flush=True,
        )

    return {
        "env_name": task.env_name,
        "prompt": task.prompt,
        "cfg_base_prompt": task.cfg_base_prompt,
        "dual_arm": task.dual_arm,
        "camera_key": task.camera_key,
        "episodes": int(args.episodes),
        "successes": int(num_success),
        "success_rate": float(num_success / args.episodes if args.episodes else 0.0),
        "metric_means": {
            "inference_latency_mean_s": _aggregate_metric(episode_results, "inference_latency_mean_s"),
            "inference_latency_p95_s": _aggregate_metric(episode_results, "inference_latency_p95_s"),
            "action_delta_l2_mean": _aggregate_metric(episode_results, "action_delta_l2_mean"),
            "action_jerk_l2_mean": _aggregate_metric(episode_results, "action_jerk_l2_mean"),
            "oscillation_sign_flip_rate": _aggregate_metric(
                episode_results, "oscillation_sign_flip_rate"
            ),
            "queue_underruns": _aggregate_metric(episode_results, "queue_underruns"),
            "queue_wait_s": _aggregate_metric(episode_results, "queue_wait_s"),
            "async_replan_delays": _aggregate_metric(episode_results, "async_replan_delays"),
        },
        "episode_results": episode_results,
    }


def write_flat_csv(summary: dict[str, Any], path: Path) -> None:
    rows = []
    for task in summary["tasks"]:
        means = task.get("metric_means", {})
        rows.append(
            {
                "label": summary["label"],
                "control_mode": summary["control_mode"],
                "replan_steps": summary["replan_steps"],
                "overlap_steps": summary["overlap_steps"],
                "low_pass_alpha": summary["low_pass_alpha"],
                "env_name": task["env_name"],
                "episodes": task["episodes"],
                "successes": task["successes"],
                "success_rate": task["success_rate"],
                **means,
            }
        )
    if not rows:
        return
    import csv

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_video_manifest(summary: dict[str, Any], path: Path) -> None:
    import csv

    rows = []
    for task in summary["tasks"]:
        for ep in task.get("episode_results", []):
            if ep.get("video_path"):
                rows.append(
                    {
                        "label": summary["label"],
                        "env_name": task["env_name"],
                        "episode": ep["episode"],
                        "seed": ep["seed"],
                        "success": ep["success"],
                        "steps": ep["steps"],
                        "video_path": ep["video_path"],
                        "actions_path": ep.get("actions_path"),
                    }
                )
    with path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "label",
            "env_name",
            "episode",
            "seed",
            "success",
            "steps",
            "video_path",
            "actions_path",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"run dir not found: {run_dir}")

    eval_settings = load_dexjoco_eval_settings(
        run_dir,
        action_horizon_override=args.action_horizon,
        text_embedding_cache_dir_override=args.text_embedding_cache_dir,
    )
    adapter = DexJoCoFastWAMAdapter(eval_settings)
    if args.replan_steps < 1 or args.replan_steps > adapter.action_horizon:
        raise ValueError(
            f"--replan-steps must be in [1, {adapter.action_horizon}], got {args.replan_steps}"
        )

    probe = PolicyClientAsync(
        host=args.policy_host,
        port=args.policy_port,
        timeout_ms=args.policy_timeout_ms,
        identity="dexjoco-probe",
    )
    try:
        if not probe.ping():
            raise RuntimeError(f"policy server ping failed at {args.policy_host}:{args.policy_port}")
    finally:
        probe.close()

    all_configs = load_task_configs(args.task_config_dir)
    if args.tasks:
        selected = set(args.tasks)
        task_configs = [cfg for cfg in all_configs if cfg["env_name"] in selected]
        missing = selected - {cfg["env_name"] for cfg in task_configs}
        if missing:
            raise ValueError(f"Unknown task names: {sorted(missing)}")
    else:
        task_configs = all_configs

    label = (
        f"{args.control_mode}_stride{args.replan_steps}"
        + ("_lpf" if args.low_pass_alpha is not None else "")
        + ("_clip" if args.action_clip else "")
    )
    session_dir = args.output_dir.expanduser().resolve()
    session_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[dexjoco-control] label={label} run_dir={run_dir} "
        f"policy={args.policy_host}:{args.policy_port}",
        flush=True,
    )
    print(
        f"[dexjoco-control] mode={args.control_mode} replan={args.replan_steps} "
        f"horizon={adapter.action_horizon} overlap={adapter.action_horizon - args.replan_steps} "
        f"lpf={args.low_pass_alpha}",
        flush=True,
    )
    print(f"[dexjoco-control] tasks={len(task_configs)} episodes={args.episodes}", flush=True)
    print(f"[dexjoco-control] output={session_dir}", flush=True)

    tasks = []
    total_success = 0
    total_episodes = 0
    for task_cfg in task_configs:
        env_name = task_cfg["env_name"]
        print(f"\n[dexjoco-control] task={env_name}", flush=True)
        task_summary = evaluate_task(
            task_cfg,
            adapter,
            args,
            task_output_dir=session_dir / env_name,
        )
        tasks.append(task_summary)
        total_success += int(task_summary["successes"])
        total_episodes += int(task_summary["episodes"])
        print(
            f"[dexjoco-control] {env_name}: {task_summary['successes']}/{task_summary['episodes']} "
            f"({100 * task_summary['success_rate']:.1f}%)",
            flush=True,
        )

    summary = {
        "label": label,
        "run_dir": str(run_dir),
        "policy_host": args.policy_host,
        "policy_port": args.policy_port,
        "control_mode": args.control_mode,
        "async_fallback": args.async_fallback,
        "replan_steps": int(args.replan_steps),
        "action_horizon": int(adapter.action_horizon),
        "overlap_steps": int(max(0, adapter.action_horizon - args.replan_steps)),
        "overlap_ratio": float(max(0, adapter.action_horizon - args.replan_steps) / adapter.action_horizon),
        "low_pass_alpha": args.low_pass_alpha,
        "low_pass_continuous_dim": (
            args.low_pass_continuous_dim
            if args.low_pass_continuous_dim is not None
            else adapter.action_output_dim
        ),
        "episodes_per_task": int(args.episodes),
        "num_tasks": len(task_configs),
        "total_episodes": int(total_episodes),
        "total_successes": int(total_success),
        "overall_success_rate": float(total_success / total_episodes if total_episodes else 0.0),
        "randomize": bool(args.randomize),
        "randomize_dynamics": bool(args.randomize_dynamics),
        "seed": int(args.seed),
        "save_actions": bool(args.save_actions),
        "action_clip": bool(args.action_clip),
        "clip_max_xyz_step": float(args.clip_max_xyz_step),
        "clip_max_dz_down": float(args.clip_max_dz_down),
        "tasks": tasks,
    }

    summary_path = session_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    write_flat_csv(summary, session_dir / "summary.csv")
    write_video_manifest(summary, session_dir / "video_manifest.csv")

    print(
        f"\n[dexjoco-control] overall: {total_success}/{total_episodes} "
        f"({100 * summary['overall_success_rate']:.1f}%)",
        flush=True,
    )
    print(f"[dexjoco-control] summary={summary_path}", flush=True)


if __name__ == "__main__":
    main()
