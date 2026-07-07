#!/usr/bin/env python3
"""Multi-GPU parallel DexJoCo eval orchestrator for FastWAM.

Launches one async policy server per GPU and one eval client per shard, splits
the total episode budget across them, waits for everything to finish, then
merges the per-shard summaries into a single combined report.

Typical usage (4 GPUs, 100 episodes of ``water_plant``)::

    python scripts/dexjoco_async/run_multi_gpu_dexjoco_eval.py \
      --gpus 4,5,6,7 \
      --run-dir runs/water_plant_uncond_2cam_384_1e-4/2026-06-29_16-38-39 \
      --checkpoint runs/.../checkpoints/weights/step_006500.pt \
      --no-load-text-encoder \
      --task-config-dir third_party/dexjoco/configs/rand_obj \
      --tasks water_plant \
      --episodes 100 --seed 0 \
      --replan-steps 24 --control-mode blocking \
      --max-env-steps 1500 \
      --output-dir evaluate_results/dexjoco/water_plant/step_006500

The orchestrator itself runs in an env with ``zmq``+``msgpack`` (e.g. ``fastwam``)
and only needs ``conda`` on PATH. Each server is launched in ``--server-conda-env``
(default ``fastwam``) and each eval client in ``--client-conda-env`` (default
``dexjoco``), mirroring the manual two-terminal workflow.

Architecture / decoupling:

* ``multi_gpu_eval_utils.py`` — sharding, port allocation, ping, conda subprocess
  launch. No torch/mujoco dependency.
* ``eval_summary_aggregator.py`` — merges per-shard ``summary.json`` files. No
  torch/mujoco dependency; reusable as a library or CLI.
* This orchestrator — only glue: spawns processes, waits, calls the aggregator,
  cleans up. The underlying eval/server scripts are NOT modified.
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
THIS_DIR = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
for _p in (PROJECT_ROOT, SRC_ROOT, SCRIPTS_ROOT, THIS_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from eval_summary_aggregator import merge_shard_summaries, write_combined  # noqa: E402
from multi_gpu_eval_utils import (  # noqa: E402
    ServerSpec,
    ShardSpec,
    build_conda_command,
    find_free_ports,
    launch_subprocess,
    locate_conda_sh,
    shard_episodes,
    terminate_process,
    wait_for_server,
)

DEFAULT_SERVER_SCRIPT = SCRIPTS_ROOT / "run_fastwam_server_async.py"
DEFAULT_CLIENT_SCRIPT = THIS_DIR / "eval_dexjoco_fastwam_control.py"
DEFAULT_DEXJOCO_PY_ROOT = PROJECT_ROOT / "third_party" / "dexjoco" / "dexjoco"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a multi-GPU parallel DexJoCo eval (N servers + N sharded clients).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Multi-GPU orchestration ---
    gpus = parser.add_argument_group("Multi-GPU orchestration")
    gpus.add_argument(
        "--gpus",
        type=str,
        required=True,
        help="Comma-separated GPU ids, e.g. 4,5,6,7. One server is started per GPU.",
    )
    gpus.add_argument(
        "--ports",
        type=str,
        default=None,
        help="Comma-separated TCP ports (one per GPU). If omitted, free ports are auto-allocated from --base-port.",
    )
    gpus.add_argument("--base-port", type=int, default=5570, help="Lower bound for auto port allocation.")
    gpus.add_argument("--host", type=str, default="0.0.0.0", help="Server bind host.")
    gpus.add_argument("--client-host", type=str, default="127.0.0.1", help="Host clients use to reach servers.")
    gpus.add_argument("--episodes", type=int, required=True, help="Total episodes (split across shards).")
    gpus.add_argument("--seed", type=int, default=0, help="Base seed; shard seeds are contiguous from here.")
    gpus.add_argument(
        "--launch-servers",
        dest="launch_servers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If set, start a server per GPU. Use --no-launch-servers to reuse already-running servers on --ports.",
    )
    gpus.add_argument("--server-startup-timeout", type=float, default=1200.0, help="Seconds to wait per server for ping.")
    gpus.add_argument("--server-poll-interval", type=float, default=5.0, help="Seconds between server pings.")
    gpus.add_argument("--server-conda-env", type=str, default="fastwam", help="Conda env for policy servers.")
    gpus.add_argument("--client-conda-env", type=str, default="dexjoco", help="Conda env for eval clients.")
    gpus.add_argument("--server-script", type=Path, default=DEFAULT_SERVER_SCRIPT, help="Server entrypoint.")
    gpus.add_argument("--client-script", type=Path, default=DEFAULT_CLIENT_SCRIPT, help="Eval client entrypoint.")
    gpus.add_argument("--dexjoco-py-root", type=Path, default=DEFAULT_DEXJOCO_PY_ROOT, help="DexJoCo python package root for PYTHONPATH.")
    gpus.add_argument("--server-num-workers", type=int, default=8, help="Server worker threads (passed through).")
    gpus.add_argument(
        "--shard-dir-fmt",
        type=str,
        default="shard_{i}",
        help="Per-shard output subdirectory template ({{i}} = shard index).",
    )

    # --- Model / checkpoint (server) ---
    model = parser.add_argument_group("Model / server")
    model.add_argument("--run-dir", type=Path, required=True, help="FastWAM training run directory.")
    model.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path or step spec.")
    model.add_argument("--dataset-stats-path", type=str, default=None, help="Override dataset stats path.")
    model.add_argument("--action-horizon", type=int, default=None, help="Override action horizon.")
    model.add_argument("--num-inference-steps", type=int, default=None, help="Override inference steps.")
    model.add_argument(
        "--load-text-encoder",
        dest="load_text_encoder",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Forward --load-text-encoder / --no-load-text-encoder to servers.",
    )
    model.add_argument("--mock", action="store_true", help="Start mock servers (no checkpoint).")
    model.add_argument("--api-token", type=str, default=None, help="Optional shared API token.")

    # --- Eval client pass-through ---
    ev = parser.add_argument_group("Eval client (pass-through)")
    ev.add_argument("--task-config-dir", type=Path, default=Path("third_party/dexjoco/configs/rand_obj"))
    ev.add_argument("--tasks", type=str, nargs="*", default=None)
    ev.add_argument("--replan-steps", type=int, required=True)
    ev.add_argument("--control-mode", choices=["blocking", "overlap"], default="blocking")
    ev.add_argument("--async-fallback", choices=["wait", "hold_last"], default="wait")
    ev.add_argument("--max-env-steps", type=int, default=1500)
    ev.add_argument("--policy-timeout-ms", type=int, default=300000)
    ev.add_argument("--video-fps", type=int, default=30)
    ev.add_argument("--low-pass-alpha", type=float, default=None)
    ev.add_argument("--low-pass-continuous-dim", type=int, default=None)
    ev.add_argument("--randomize", dest="randomize", action=argparse.BooleanOptionalAction, default=False)
    ev.add_argument("--randomize-dynamics", dest="randomize_dynamics", action=argparse.BooleanOptionalAction, default=False)
    ev.add_argument("--save-video", dest="save_video", action=argparse.BooleanOptionalAction, default=True)
    ev.add_argument("--save-actions", dest="save_actions", action=argparse.BooleanOptionalAction, default=True)
    ev.add_argument("--action-clip", dest="action_clip", action=argparse.BooleanOptionalAction, default=False)
    ev.add_argument("--clip-max-xyz-step", type=float, default=0.05)
    ev.add_argument("--clip-max-dz-down", type=float, default=0.03)

    parser.add_argument("--output-dir", type=Path, required=True, help="Top-level output dir for the combined report.")
    return parser.parse_args()


def _parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _bool_flag(name: str, value: bool | None) -> list[str]:
    if value is None:
        return []
    return [f"--{name}" if value else f"--no-{name}"]


def _build_server_argv(args: argparse.Namespace, server: ServerSpec) -> list[str]:
    argv: list[str] = [
        "python",
        str(Path(args.server_script).resolve()),
        "--device", server.device,
        "--host", server.bind_host,
        "--port", str(server.port),
        "--num-workers", str(args.server_num_workers),
    ]
    if args.mock:
        argv.append("--mock")
    else:
        if not args.run_dir or not args.checkpoint:
            raise ValueError("--run-dir and --checkpoint are required unless --mock is set.")
        argv += ["--run-dir", str(args.run_dir.resolve()), "--checkpoint", str(args.checkpoint)]
    if args.dataset_stats_path is not None:
        argv += ["--dataset-stats-path", str(args.dataset_stats_path)]
    if args.action_horizon is not None:
        argv += ["--action-horizon", str(args.action_horizon)]
    if args.num_inference_steps is not None:
        argv += ["--num-inference-steps", str(args.num_inference_steps)]
    argv += _bool_flag("load-text-encoder", args.load_text_encoder)
    if args.api_token:
        argv += ["--api-token", str(args.api_token)]
    return argv


def _build_client_argv(args: argparse.Namespace, shard: ShardSpec, shard_out_dir: Path) -> list[str]:
    argv: list[str] = [
        "python",
        str(Path(args.client_script).resolve()),
        "--run-dir", str(args.run_dir.resolve()),
        "--policy-host", shard.server.connect_host,
        "--policy-port", str(shard.server.port),
        "--policy-timeout-ms", str(args.policy_timeout_ms),
        "--task-config-dir", str(args.task_config_dir.resolve()),
        "--episodes", str(shard.num_episodes),
        "--seed", str(shard.base_seed),
        "--replan-steps", str(args.replan_steps),
        "--control-mode", str(args.control_mode),
        "--async-fallback", str(args.async_fallback),
        "--max-env-steps", str(args.max_env_steps),
        "--output-dir", str(shard_out_dir),
        "--video-fps", str(args.video_fps),
    ]
    if args.tasks:
        argv += ["--tasks"] + list(args.tasks)
    if args.low_pass_alpha is not None:
        argv += ["--low-pass-alpha", str(args.low_pass_alpha)]
    if args.low_pass_continuous_dim is not None:
        argv += ["--low-pass-continuous-dim", str(args.low_pass_continuous_dim)]
    argv += _bool_flag("randomize", args.randomize)
    argv += _bool_flag("randomize-dynamics", args.randomize_dynamics)
    argv += _bool_flag("save-video", args.save_video)
    argv += _bool_flag("save-actions", args.save_actions)
    argv += _bool_flag("action-clip", args.action_clip)
    if args.action_clip:
        argv += ["--clip-max-xyz-step", str(args.clip_max_xyz_step)]
        argv += ["--clip-max-dz-down", str(args.clip_max_dz_down)]
    if args.api_token:
        argv += ["--api-token", str(args.api_token)]
    return argv


def _server_exports(args: argparse.Namespace, server: ServerSpec) -> tuple[dict[str, str], dict[str, str]]:
    exports: dict[str, str] = {
        "CUDA_VISIBLE_DEVICES": str(server.gpu),
        "PYTHONPATH": _dedupe_path(f"{SRC_ROOT}:{SCRIPTS_ROOT}:{_retain(args, 'PYTHONPATH')}"),
    }
    return exports, {}


def _client_exports(args: argparse.Namespace) -> tuple[dict[str, str], dict[str, str]]:
    pyroot = str(Path(args.dexjoco_py_root).resolve())
    py_path = f"{SRC_ROOT}:{SCRIPTS_ROOT}:{pyroot}:{_retain(args, 'PYTHONPATH')}"
    exports: dict[str, str] = {
        "MUJOCO_GL": "egl",
        "PYTHONPATH": _dedupe_path(py_path),
    }
    # LD_LIBRARY_PATH references $CONDA_PREFIX which is only defined after
    # `conda activate <client env>` runs, so it must be expanded by bash (raw).
    raw_exports: dict[str, str] = {
        "LD_LIBRARY_PATH": '"${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"',
    }
    return exports, raw_exports


def _retain(args: argparse.Namespace, var: str) -> str:
    """Return the orchestrator's current value of env var ``var`` (may be empty)."""
    import os

    return os.environ.get(var, "")


def _dedupe_path(path: str) -> str:
    """Remove duplicate entries from a colon-separated PATH-style string, keeping order."""
    seen: set[str] = set()
    out: list[str] = []
    for part in path.split(":"):
        if part and part not in seen:
            seen.add(part)
            out.append(part)
    return ":".join(out)


def _inject_shard_metadata(summary_path: Path, shard: ShardSpec) -> bool:
    """Add ``shard_id`` / ``base_seed`` into a shard's summary.json in place.

    Keeps the eval script untouched: the orchestrator annotates the summary it
    already wrote so the aggregator has stable provenance.
    """
    if not summary_path.exists():
        return False
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    payload["shard_id"] = int(shard.shard_id)
    payload.setdefault("base_seed", int(shard.base_seed))
    payload.setdefault("policy_port", int(shard.server.port))
    payload.setdefault("global_episode_start", int(shard.global_episode_start))
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return True


def _stream_log_tail(path: Path, *, prefix: str, last_pos: int) -> int:
    """Print newly appended lines from ``path``; return the new file offset."""
    if not path.exists():
        return last_pos
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            f.seek(last_pos)
            chunk = f.read()
            new_pos = f.tell()
    except Exception:
        return last_pos
    for line in chunk.splitlines():
        if line.strip():
            print(f"  [{prefix}] {line}", flush=True)
    return new_pos


def main() -> int:
    args = parse_args()
    gpus = _parse_int_list(args.gpus)
    if not gpus:
        raise ValueError("--gpus must list at least one GPU id")

    # Resolve ports (explicit or auto-allocated).
    if args.ports:
        ports = _parse_int_list(args.ports)
        if len(ports) != len(gpus):
            raise ValueError(
                f"--ports has {len(ports)} entries but --gpus has {len(gpus)}; they must match"
            )
    else:
        ports = find_free_ports(len(gpus))

    servers = [
        ServerSpec(gpu=g, port=p, bind_host=args.host, connect_host=args.client_host)
        for g, p in zip(gpus, ports)
    ]

    shards_all = shard_episodes(servers, total_episodes=args.episodes, base_seed=args.seed)
    # Only run shards that actually have episodes.
    shards = [s for s in shards_all if s.num_episodes > 0]
    if len(shards) != len(shards_all):
        idle = [s.shard_id for s in shards_all if s.num_episodes == 0]
        print(f"[multi-gpu] episodes={args.episodes} < gpus={len(gpus)}; "
              f"skipping idle shards {idle}", flush=True)

    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[multi-gpu] gpus={gpus} ports={ports} episodes={args.episodes} "
          f"shards={len(shards)} output={out_dir}", flush=True)
    for s in shards:
        print(f"[multi-gpu]   shard {s.shard_id}: gpu={s.server.gpu} port={s.server.port} "
              f"eps={s.num_episodes} seed={s.base_seed}..{s.base_seed + s.num_episodes - 1} "
              f"(global {s.global_episode_start}..{s.global_episode_start + s.num_episodes - 1})",
              flush=True)

    conda_sh = locate_conda_sh()

    server_procs: list[tuple[ShardSpec, Any, Path]] = []
    client_procs: list[tuple[ShardSpec, Any, Path]] = []
    exit_code = 0

    # Graceful cleanup on Ctrl-C / errors.
    def _cleanup_servers() -> None:
        for shard, proc, _ in server_procs:
            terminate_process(proc, label=f"server-shard{shard.shard_id}")

    def _signal_handler(signum, _frame):
        print(f"\n[multi-gpu] signal {signum} received, shutting down...", flush=True)
        for shard, proc, _ in client_procs:
            terminate_process(proc, label=f"client-shard{shard.shard_id}")
        _cleanup_servers()
        raise SystemExit(130)

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    try:
        # 1) Launch servers.
        if args.launch_servers:
            if args.mock:
                print("[multi-gpu] launching mock servers (no checkpoint)", flush=True)
            for shard in shards:
                shard_dir = out_dir / args.shard_dir_fmt.format(i=shard.shard_id)
                log_path = shard_dir / "server.log"
                argv = _build_server_argv(args, shard.server)
                exports, raw_exports = _server_exports(args, shard.server)
                cmd = build_conda_command(
                    conda_sh,
                    args.server_conda_env,
                    exports,
                    argv,
                    raw_exports=raw_exports,
                )
                proc = launch_subprocess(cmd, log_path=log_path, cwd=PROJECT_ROOT)
                server_procs.append((shard, proc, log_path))
                print(f"[multi-gpu] started server shard={shard.shard_id} gpu={shard.server.gpu} "
                      f"port={shard.server.port} pid={proc.pid} log={log_path}", flush=True)

            # 2) Wait for all servers to become ready.
            print("[multi-gpu] waiting for servers to become ready...", flush=True)
            ready_all = True
            for shard, _, _ in server_procs:
                ok = wait_for_server(
                    shard.server.connect_host,
                    shard.server.port,
                    timeout_s=args.server_startup_timeout,
                    poll_interval_s=args.server_poll_interval,
                )
                # If the process already died, treat as not ready.
                proc = next(p for s, p, _ in server_procs if s.shard_id == shard.shard_id)
                if proc.poll() is not None:
                    ok = False
                status = "ready" if ok else "FAILED"
                print(f"[multi-gpu]   server shard={shard.shard_id} port={shard.server.port}: {status}",
                      flush=True)
                if not ok:
                    ready_all = False
            if not ready_all:
                print("[multi-gpu] one or more servers failed to start; aborting", flush=True)
                exit_code = 1
                _cleanup_servers()
                return exit_code
        else:
            # Reuse existing servers: just verify readiness.
            print("[multi-gpu] --no-launch-servers: verifying existing servers...", flush=True)
            for shard in shards:
                ok = wait_for_server(
                    shard.server.connect_host,
                    shard.server.port,
                    timeout_s=args.server_startup_timeout,
                    poll_interval_s=args.server_poll_interval,
                )
                print(f"[multi-gpu]   server port={shard.server.port}: "
                      f"{'ready' if ok else 'NOT REACHABLE'}", flush=True)
                if not ok:
                    print("[multi-gpu] existing server not reachable; aborting", flush=True)
                    return 1

        # 3) Launch eval clients (one per shard, in parallel).
        print("[multi-gpu] launching eval clients...", flush=True)
        for shard in shards:
            shard_dir = out_dir / args.shard_dir_fmt.format(i=shard.shard_id)
            log_path = shard_dir / "client.log"
            argv = _build_client_argv(args, shard, shard_dir)
            exports, raw_exports = _client_exports(args)
            cmd = build_conda_command(
                conda_sh,
                args.client_conda_env,
                exports,
                argv,
                raw_exports=raw_exports,
            )
            proc = launch_subprocess(cmd, log_path=log_path, cwd=PROJECT_ROOT)
            client_procs.append((shard, proc, log_path))
            print(f"[multi-gpu] started client shard={shard.shard_id} port={shard.server.port} "
                  f"eps={shard.num_episodes} pid={proc.pid} log={log_path}", flush=True)

        # 4) Wait for clients to finish, streaming log tails for visibility.
        print("[multi-gpu] waiting for eval clients to finish...", flush=True)
        offsets: dict[int, int] = {}
        remaining = list(client_procs)
        while remaining:
            time.sleep(5.0)
            still_running: list[tuple[ShardSpec, Any, Path]] = []
            for shard, proc, log_path in remaining:
                # Surface recent log lines so the user sees per-episode progress.
                key = shard.shard_id
                offsets[key] = _stream_log_tail(
                    log_path, prefix=f"shard{key}", last_pos=offsets.get(key, 0)
                )
                rc = proc.poll()
                if rc is None:
                    still_running.append((shard, proc, log_path))
                else:
                    tag = "ok" if rc == 0 else f"exit={rc}"
                    print(f"[multi-gpu] client shard={shard.shard_id} finished ({tag})", flush=True)
                    offsets[key] = _stream_log_tail(
                        log_path, prefix=f"shard{key}", last_pos=offsets.get(key, 0)
                    )
            remaining = still_running
            if remaining:
                names = ",".join(str(s.shard_id) for s, _, _ in remaining)
                print(f"[multi-gpu] still running: shards [{names}]", flush=True)

        # 5) Inject shard provenance into each shard summary, then aggregate.
        shard_summary_paths: list[Path] = []
        for shard, proc, _ in client_procs:
            if proc.returncode != 0:
                print(f"[multi-gpu] WARNING: shard {shard.shard_id} exited with "
                      f"{proc.returncode}; attempting partial aggregation", flush=True)
                exit_code = 1
            shard_dir = out_dir / args.shard_dir_fmt.format(i=shard.shard_id)
            summary_path = shard_dir / "summary.json"
            if _inject_shard_metadata(summary_path, shard):
                shard_summary_paths.append(summary_path)
            else:
                print(f"[multi-gpu] WARNING: no summary.json for shard {shard.shard_id} "
                      f"at {summary_path}", flush=True)

        if not shard_summary_paths:
            print("[multi-gpu] no shard summaries produced; nothing to aggregate", flush=True)
            if exit_code == 0:
                exit_code = 1
            return exit_code

        label = (
            f"{args.control_mode}_stride{args.replan_steps}"
            + ("_lpf" if args.low_pass_alpha is not None else "")
            + ("_clip" if args.action_clip else "")
            + f"_gpus{len(gpus)}"
        )
        combined = merge_shard_summaries(
            shard_summary_paths,
            label=label,
            extra_top_level={
                "policy_host": args.client_host,
                "seed": int(args.seed),
                "randomize": bool(args.randomize),
                "randomize_dynamics": bool(args.randomize_dynamics),
                "save_actions": bool(args.save_actions),
                "action_clip": bool(args.action_clip),
                "clip_max_xyz_step": float(args.clip_max_xyz_step),
                "clip_max_dz_down": float(args.clip_max_dz_down),
                "gpus": gpus,
                "ports": ports,
            },
        )
        paths = write_combined(combined, out_dir)
        total = combined["total_episodes"]
        succ = combined["total_successes"]
        rate = combined["overall_success_rate"]
        print(f"\n[multi-gpu] combined: {succ}/{total} ({100 * rate:.1f}%) "
              f"across {combined['num_shards']} shards / {combined['num_tasks']} task(s)",
              flush=True)
        for task in combined["tasks"]:
            print(f"[multi-gpu]   {task['env_name']}: {task['successes']}/{task['episodes']} "
                  f"({100 * task['success_rate']:.1f}%)", flush=True)
        print(f"[multi-gpu] summary={paths['summary']}", flush=True)
        print(f"[multi-gpu] csv={paths['csv']}", flush=True)
        return exit_code

    finally:
        if args.launch_servers:
            _cleanup_servers()


if __name__ == "__main__":
    raise SystemExit(main())
