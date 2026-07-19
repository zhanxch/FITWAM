#!/usr/bin/env python3
"""Multi-GPU DexJoCo rollout collection orchestrator for FastWAM.

This mirrors ``run_multi_gpu_dexjoco_eval.py`` but writes LeRobot datasets
instead of eval summaries:

* one async FastWAM policy server per GPU;
* one collection client per shard;
* optional merge into a single raw rollout dataset;
* optional failure-tail trimming into a second dataset.

Run it inside tmux for long jobs.
"""

from __future__ import annotations

import argparse
import json
import signal
import subprocess
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
DEFAULT_COLLECT_SCRIPT = SCRIPTS_ROOT / "collect_dexjoco_rollouts.py"
DEFAULT_BUILD_SCRIPT = SCRIPTS_ROOT / "build_rollout_datasets.py"
DEFAULT_DEXJOCO_PY_ROOT = PROJECT_ROOT / "third_party" / "dexjoco" / "dexjoco"
DEFAULT_SOURCE_DATASET_ROOT = PROJECT_ROOT / "data" / "dexjoco" / "dexjoco_lerobot_datasets"
DEFAULT_SOURCE_DATASET = DEFAULT_SOURCE_DATASET_ROOT / "water_plant"
DEFAULT_SUCCESS_PROMPTS = {
    "water_plant": "Grasp the watering can and apply water to the plant.",
    "hammer_nail": "Pick up the hammer and hammer the nail into the board.",
}
SHARD_LAUNCH_FRESH = "fresh"
SHARD_LAUNCH_OVERWRITE = "overwrite"
SHARD_LAUNCH_RESUME = "resume"
_INITIALIZED_SHARD_DIRS = (
    Path("meta"),
    Path("data") / "chunk-000",
    Path("videos") / "chunk-000" / "observation.images.front",
    Path("videos") / "chunk-000" / "observation.images.wrist",
)
_INITIALIZED_SHARD_FILES = (
    Path("meta") / "info.json",
    Path("meta") / "tasks.jsonl",
    Path("meta") / "episode_outcomes.jsonl",
    Path("collection_summary.json"),
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run multi-GPU DexJoCo rollout collection into LeRobot datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    orch = parser.add_argument_group("Multi-GPU orchestration")
    orch.add_argument("--gpus", type=str, required=True, help="Comma-separated GPU ids, e.g. 0,1,2,3.")
    orch.add_argument("--ports", type=str, default=None, help="Comma-separated TCP ports, one per GPU.")
    orch.add_argument("--base-port", type=int, default=5590)
    orch.add_argument("--host", type=str, default="0.0.0.0")
    orch.add_argument("--client-host", type=str, default="127.0.0.1")
    orch.add_argument("--episodes", type=int, required=True, help="Total rollout attempts to save.")
    orch.add_argument("--seed", type=int, default=0, help="Base seed; shard seeds are contiguous.")
    orch.add_argument("--launch-servers", action=argparse.BooleanOptionalAction, default=True)
    orch.add_argument("--server-startup-timeout", type=float, default=1200.0)
    orch.add_argument("--server-poll-interval", type=float, default=5.0)
    orch.add_argument("--server-conda-env", type=str, default="fastwam")
    orch.add_argument("--client-conda-env", type=str, default="dexjoco")
    orch.add_argument("--server-script", type=Path, default=DEFAULT_SERVER_SCRIPT)
    orch.add_argument("--collect-script", type=Path, default=DEFAULT_COLLECT_SCRIPT)
    orch.add_argument("--build-script", type=Path, default=DEFAULT_BUILD_SCRIPT)
    orch.add_argument("--dexjoco-py-root", type=Path, default=DEFAULT_DEXJOCO_PY_ROOT)
    orch.add_argument("--server-num-workers", type=int, default=8)
    orch.add_argument("--shard-dir-fmt", type=str, default="shard_{i}")

    model = parser.add_argument_group("Model / server")
    model.add_argument("--run-dir", type=Path, required=True)
    model.add_argument("--checkpoint", type=str, required=True)
    normalization = model.add_mutually_exclusive_group()
    normalization.add_argument("--dataset-stats-path", type=str, default=None)
    normalization.add_argument("--norm-stats-meta-dir", type=Path, default=None)
    model.add_argument(
        "--text-embedding-cache-dir",
        type=Path,
        default=None,
        help="Runtime relocation of cached task contexts used by collectors.",
    )
    model.add_argument("--action-horizon", type=int, default=None)
    model.add_argument("--num-inference-steps", type=int, default=None)
    model.add_argument("--load-text-encoder", action=argparse.BooleanOptionalAction, default=True)

    collect = parser.add_argument_group("Collection")
    collect.add_argument("--task-config-dir", type=Path, default=PROJECT_ROOT / "third_party" / "dexjoco" / "configs" / "rand_obj")
    collect.add_argument("--tasks", type=str, nargs="*", default=["water_plant"], help="Currently exactly one task is supported.")
    collect.add_argument(
        "--source-dataset",
        type=Path,
        default=None,
        help="Template LeRobot dataset. Defaults to data/dexjoco/dexjoco_lerobot_datasets/<task>.",
    )
    collect.add_argument(
        "--success-prompt",
        type=str,
        default=None,
        help="Instruction text for successful trajectories. Defaults to a known task prompt when available.",
    )
    collect.add_argument("--replan-steps", type=int, required=True)
    collect.add_argument("--max-env-steps", type=int, default=600)
    collect.add_argument("--policy-timeout-ms", type=int, default=300000)
    collect.add_argument("--video-fps", type=int, default=30)
    collect.add_argument("--randomize", action=argparse.BooleanOptionalAction, default=False)
    collect.add_argument("--randomize-dynamics", action=argparse.BooleanOptionalAction, default=False)
    collect.add_argument("--action-clip", action=argparse.BooleanOptionalAction, default=False)
    collect.add_argument("--clip-max-xyz-step", type=float, default=0.05)
    collect.add_argument("--clip-max-dz-down", type=float, default=0.03)
    collect.add_argument("--failure-phrase", type=str, default="Failed to finish the whole process.")
    collect.add_argument(
        "--outcome-task-mode",
        choices=("task-marker", "clean"),
        default="task-marker",
        help=(
            "task-marker preserves the legacy failure phrase in failed task text; "
            "clean keeps one instruction for both outcomes and relies on the outcome ledger."
        ),
    )

    outputs = parser.add_argument_group("Outputs")
    outputs.add_argument("--output-dir", type=Path, required=True, help="Top-level log/shard directory.")
    outputs.add_argument("--raw-output-dataset", type=Path, default=None, help="Optional merged raw dataset path.")
    outputs.add_argument("--trimmed-output-dataset", type=Path, default=None, help="Optional trimmed dataset path.")
    outputs.add_argument("--trim-failure-seconds", type=float, default=None, help="If set, trim this many seconds from failed rollouts.")
    output_mode = outputs.add_mutually_exclusive_group()
    output_mode.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Replace existing shard and derived outputs. This remains the default for fresh runs.",
    )
    output_mode.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume every deterministic shard in place, then rebuild any requested "
            "merged raw and trimmed datasets from the completed shards."
        ),
    )
    args = parser.parse_args(argv)
    if args.resume:
        args.overwrite = False
    elif args.overwrite is None:
        args.overwrite = True
    return args


def _parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _bool_flag(name: str, value: bool | None) -> list[str]:
    if value is None:
        return []
    return [f"--{name}" if value else f"--no-{name}"]


def _dedupe_path(path: str) -> str:
    seen: set[str] = set()
    out: list[str] = []
    for part in path.split(":"):
        if part and part not in seen:
            seen.add(part)
            out.append(part)
    return ":".join(out)


def _retain(var: str) -> str:
    import os

    return os.environ.get(var, "")


def _server_exports(server: ServerSpec) -> tuple[dict[str, str], dict[str, str]]:
    return {
        "CUDA_VISIBLE_DEVICES": str(server.gpu),
        "PYTHONPATH": _dedupe_path(f"{SRC_ROOT}:{SCRIPTS_ROOT}:{_retain('PYTHONPATH')}"),
    }, {}


def _client_exports(args: argparse.Namespace) -> tuple[dict[str, str], dict[str, str]]:
    pyroot = str(Path(args.dexjoco_py_root).resolve())
    return {
        "MUJOCO_GL": "egl",
        "PYTHONPATH": _dedupe_path(f"{SRC_ROOT}:{SCRIPTS_ROOT}:{pyroot}:{_retain('PYTHONPATH')}"),
    }, {
        "LD_LIBRARY_PATH": '"${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"',
    }


def _build_server_argv(args: argparse.Namespace, server: ServerSpec) -> list[str]:
    argv = [
        "python",
        str(Path(args.server_script).resolve()),
        "--device", server.device,
        "--host", server.bind_host,
        "--port", str(server.port),
        "--num-workers", str(args.server_num_workers),
        "--run-dir", str(args.run_dir.resolve()),
        "--checkpoint", str(args.checkpoint),
    ]
    dataset_stats_path = getattr(args, "dataset_stats_path", None)
    norm_stats_meta_dir = getattr(args, "norm_stats_meta_dir", None)
    if dataset_stats_path is not None and norm_stats_meta_dir is not None:
        raise ValueError(
            "--dataset-stats-path and --norm-stats-meta-dir are mutually exclusive"
        )
    if dataset_stats_path is not None:
        argv += ["--dataset-stats-path", str(dataset_stats_path)]
    if norm_stats_meta_dir is not None:
        argv += [
            "--norm-stats-meta-dir",
            str(Path(norm_stats_meta_dir).expanduser().resolve()),
        ]
    if args.action_horizon is not None:
        argv += ["--action-horizon", str(args.action_horizon)]
    if args.num_inference_steps is not None:
        argv += ["--num-inference-steps", str(args.num_inference_steps)]
    argv += _bool_flag("load-text-encoder", args.load_text_encoder)
    return argv


def _resolve_task_config(args: argparse.Namespace) -> Path:
    tasks = list(args.tasks or [])
    if len(tasks) != 1:
        raise ValueError("Collection currently supports exactly one task, e.g. --tasks water_plant")
    task_config = args.task_config_dir.expanduser().resolve() / f"{tasks[0]}.yaml"
    if not task_config.exists():
        raise FileNotFoundError(task_config)
    return task_config


def _resolve_task_name(args: argparse.Namespace) -> str:
    tasks = list(args.tasks or [])
    if len(tasks) != 1:
        raise ValueError("Collection currently supports exactly one task, e.g. --tasks water_plant")
    return tasks[0]


def _resolve_success_prompt(args: argparse.Namespace) -> str:
    if args.success_prompt:
        return args.success_prompt
    task_name = _resolve_task_name(args)
    if task_name in DEFAULT_SUCCESS_PROMPTS:
        return DEFAULT_SUCCESS_PROMPTS[task_name]
    raise ValueError(
        f"No default success prompt for task {task_name!r}; pass --success-prompt explicitly."
    )


def _resolve_source_dataset(args: argparse.Namespace) -> Path:
    if args.source_dataset is not None:
        return args.source_dataset.expanduser().resolve()
    return (DEFAULT_SOURCE_DATASET_ROOT / _resolve_task_name(args)).resolve()


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return value


def _validate_jsonl_objects(path: Path, *, label: str) -> None:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"{label} is not readable: {path}") from exc
    rows = 0
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"{label} has invalid JSON at line {line_number}: {path}"
            ) from exc
        if not isinstance(value, dict):
            raise ValueError(
                f"{label} row {line_number} must be a JSON object: {path}"
            )
        rows += 1
    if rows == 0:
        raise ValueError(f"{label} must contain at least one row: {path}")


def _classify_resume_shard_dataset(
    args: argparse.Namespace,
    shard: ShardSpec,
    shard_dataset_dir: Path,
) -> str:
    """Choose resume or protected fresh launch for one deterministic shard."""
    dataset_dir = shard_dataset_dir.expanduser().resolve()
    if not dataset_dir.exists():
        if shard_dataset_dir.is_symlink():
            raise ValueError(
                f"Shard dataset is a broken symlink; refusing recovery: {shard_dataset_dir}"
            )
        return SHARD_LAUNCH_FRESH
    if not dataset_dir.is_dir():
        raise ValueError(
            f"Shard dataset path exists but is not a directory; refusing recovery: {dataset_dir}"
        )

    missing_or_wrong: list[str] = []
    for relative in _INITIALIZED_SHARD_DIRS:
        path = dataset_dir / relative
        if not path.is_dir():
            missing_or_wrong.append(f"directory:{relative}")
    for relative in _INITIALIZED_SHARD_FILES:
        path = dataset_dir / relative
        if not path.is_file():
            missing_or_wrong.append(f"file:{relative}")
    if missing_or_wrong:
        raise ValueError(
            "Shard dataset exists but is only partially initialized; refusing recovery "
            f"without deleting data: {dataset_dir} ({', '.join(missing_or_wrong)})"
        )

    _read_json_object(dataset_dir / "meta" / "info.json", label="meta/info.json")
    _validate_jsonl_objects(
        dataset_dir / "meta" / "tasks.jsonl",
        label="meta/tasks.jsonl",
    )
    summary = _read_json_object(
        dataset_dir / "collection_summary.json",
        label="collection_summary.json",
    )
    expected = {
        "mode": "save_all",
        "outcome_task_mode": str(args.outcome_task_mode),
        "target_episodes": int(shard.num_episodes),
        "max_attempts": int(shard.num_episodes),
        "base_seed": int(shard.base_seed),
        "output_dataset": str(dataset_dir),
    }
    mismatches = [
        f"{key}={summary.get(key)!r} expected={value!r}"
        for key, value in expected.items()
        if summary.get(key) != value
    ]
    if mismatches:
        raise ValueError(
            "Shard dataset identity conflicts with the frozen shard assignment; "
            f"refusing recovery without deleting data: {dataset_dir} "
            f"({'; '.join(mismatches)})"
        )
    return SHARD_LAUNCH_RESUME


def _plan_resume_shard_launches(
    args: argparse.Namespace,
    shards: list[ShardSpec],
    out_dir: Path,
) -> dict[int, str]:
    if not args.resume:
        return {}
    plan: dict[int, str] = {}
    for shard in shards:
        shard_dir = out_dir / args.shard_dir_fmt.format(i=shard.shard_id)
        dataset_dir = shard_dir / "dataset"
        plan[shard.shard_id] = _classify_resume_shard_dataset(
            args,
            shard,
            dataset_dir,
        )
    return plan


def _build_collect_argv(
    args: argparse.Namespace,
    shard: ShardSpec,
    shard_dataset_dir: Path,
    *,
    shard_launch_mode: str | None = None,
) -> list[str]:
    argv = [
        "python",
        str(Path(args.collect_script).resolve()),
        "--run-dir", str(args.run_dir.resolve()),
        "--policy-host", shard.server.connect_host,
        "--policy-port", str(shard.server.port),
        "--policy-timeout-ms", str(args.policy_timeout_ms),
        "--task-config", str(_resolve_task_config(args)),
        "--source-dataset", str(_resolve_source_dataset(args)),
        "--output-dataset", str(shard_dataset_dir),
        "--save-all-trajectories",
        "--target-episodes", str(shard.num_episodes),
        "--max-attempts", str(shard.num_episodes),
        "--seed", str(shard.base_seed),
        "--replan-steps", str(args.replan_steps),
        "--max-env-steps", str(args.max_env_steps),
        "--video-fps", str(args.video_fps),
        "--success-prompt", _resolve_success_prompt(args),
        "--failure-phrase", str(args.failure_phrase),
        "--outcome-task-mode", str(args.outcome_task_mode),
    ]
    if args.text_embedding_cache_dir is not None:
        argv += [
            "--text-embedding-cache-dir",
            str(args.text_embedding_cache_dir.expanduser().resolve()),
        ]
    argv += _bool_flag("randomize", args.randomize)
    argv += _bool_flag("randomize-dynamics", args.randomize_dynamics)
    argv += _bool_flag("action-clip", args.action_clip)
    if args.action_clip:
        argv += ["--clip-max-xyz-step", str(args.clip_max_xyz_step)]
        argv += ["--clip-max-dz-down", str(args.clip_max_dz_down)]
    if shard_launch_mode is None:
        if args.resume:
            shard_launch_mode = SHARD_LAUNCH_RESUME
        elif args.overwrite:
            shard_launch_mode = SHARD_LAUNCH_OVERWRITE
        else:
            shard_launch_mode = SHARD_LAUNCH_FRESH
    if shard_launch_mode == SHARD_LAUNCH_RESUME:
        argv.append("--resume")
    elif shard_launch_mode == SHARD_LAUNCH_OVERWRITE:
        argv.append("--overwrite")
    elif shard_launch_mode != SHARD_LAUNCH_FRESH:
        raise ValueError(f"Unsupported shard launch mode: {shard_launch_mode!r}")
    return argv


def _stream_log(path: Path, *, prefix: str, last_pos: int) -> int:
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


def _run_dataset_build(args: argparse.Namespace, shard_dataset_dirs: list[Path]) -> int:
    if args.raw_output_dataset is None:
        return 0
    raw_output = args.raw_output_dataset.expanduser().resolve()
    cmd = [
        sys.executable,
        str(args.build_script.expanduser().resolve()),
        "merge-shards",
        "--shard-datasets",
        *[str(path) for path in shard_dataset_dirs],
        "--output-dataset",
        str(raw_output),
        "--failure-phrase",
        str(args.failure_phrase),
    ]
    if args.overwrite or args.resume:
        cmd.append("--overwrite")
    print(f"[collect] merging raw dataset -> {raw_output}", flush=True)
    rc = subprocess.run(cmd, cwd=str(PROJECT_ROOT)).returncode
    if rc != 0:
        return rc

    if args.trim_failure_seconds is None:
        return 0
    if args.trimmed_output_dataset is None:
        raise ValueError("--trimmed-output-dataset is required when --trim-failure-seconds is set")
    trimmed_output = args.trimmed_output_dataset.expanduser().resolve()
    cmd = [
        sys.executable,
        str(args.build_script.expanduser().resolve()),
        "trim-failures",
        "--source-dataset",
        str(raw_output),
        "--output-dataset",
        str(trimmed_output),
        "--trim-failure-seconds",
        str(args.trim_failure_seconds),
        "--failure-phrase",
        str(args.failure_phrase),
    ]
    if args.overwrite or args.resume:
        cmd.append("--overwrite")
    print(f"[collect] building trimmed dataset -> {trimmed_output}", flush=True)
    return subprocess.run(cmd, cwd=str(PROJECT_ROOT)).returncode


def main() -> int:
    args = parse_args()
    gpus = _parse_int_list(args.gpus)
    if not gpus:
        raise ValueError("--gpus must list at least one GPU")
    if args.ports:
        ports = _parse_int_list(args.ports)
        if len(ports) != len(gpus):
            raise ValueError("--ports and --gpus must have the same length")
    else:
        ports = find_free_ports(len(gpus))

    servers = [
        ServerSpec(gpu=g, port=p, bind_host=args.host, connect_host=args.client_host)
        for g, p in zip(gpus, ports)
    ]
    shards_all = shard_episodes(servers, total_episodes=args.episodes, base_seed=args.seed)
    shards = [s for s in shards_all if s.num_episodes > 0]
    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[collect] gpus={gpus} ports={ports} episodes={args.episodes} output={out_dir}", flush=True)
    for shard in shards:
        print(
            f"[collect]   shard {shard.shard_id}: gpu={shard.server.gpu} port={shard.server.port} "
            f"eps={shard.num_episodes} seed={shard.base_seed}..{shard.base_seed + shard.num_episodes - 1}",
            flush=True,
        )
    shard_launch_modes = _plan_resume_shard_launches(args, shards, out_dir)
    for shard in shards:
        if shard.shard_id in shard_launch_modes:
            print(
                f"[collect]   shard {shard.shard_id}: "
                f"dataset_launch={shard_launch_modes[shard.shard_id]}",
                flush=True,
            )

    conda_sh = locate_conda_sh()
    server_procs: list[tuple[ShardSpec, Any, Path]] = []
    collect_procs: list[tuple[ShardSpec, Any, Path, Path]] = []

    def cleanup_servers() -> None:
        for shard, proc, _ in server_procs:
            terminate_process(proc, label=f"server-shard{shard.shard_id}")

    def signal_handler(signum, _frame):
        print(f"\n[collect] signal {signum} received, shutting down...", flush=True)
        for shard, proc, _, _ in collect_procs:
            terminate_process(proc, label=f"collect-shard{shard.shard_id}")
        cleanup_servers()
        raise SystemExit(130)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        if args.launch_servers:
            for shard in shards:
                shard_dir = out_dir / args.shard_dir_fmt.format(i=shard.shard_id)
                log_path = shard_dir / "server.log"
                exports, raw_exports = _server_exports(shard.server)
                cmd = build_conda_command(
                    conda_sh,
                    args.server_conda_env,
                    exports,
                    _build_server_argv(args, shard.server),
                    raw_exports=raw_exports,
                )
                proc = launch_subprocess(cmd, log_path=log_path, cwd=PROJECT_ROOT)
                server_procs.append((shard, proc, log_path))
                print(
                    f"[collect] started server shard={shard.shard_id} gpu={shard.server.gpu} "
                    f"port={shard.server.port} pid={proc.pid} log={log_path}",
                    flush=True,
                )
            print("[collect] waiting for servers...", flush=True)
            ready_all = True
            for shard, proc, _ in server_procs:
                ok = wait_for_server(
                    shard.server.connect_host,
                    shard.server.port,
                    timeout_s=args.server_startup_timeout,
                    poll_interval_s=args.server_poll_interval,
                )
                if proc.poll() is not None:
                    ok = False
                print(f"[collect]   server shard={shard.shard_id}: {'ready' if ok else 'FAILED'}", flush=True)
                ready_all = ready_all and ok
            if not ready_all:
                cleanup_servers()
                return 1
        else:
            print("[collect] --no-launch-servers: verifying existing servers...", flush=True)
            for shard in shards:
                ok = wait_for_server(
                    shard.server.connect_host,
                    shard.server.port,
                    timeout_s=args.server_startup_timeout,
                    poll_interval_s=args.server_poll_interval,
                )
                print(f"[collect]   server port={shard.server.port}: {'ready' if ok else 'NOT REACHABLE'}", flush=True)
                if not ok:
                    return 1

        print("[collect] launching collectors...", flush=True)
        shard_dataset_dirs: list[Path] = []
        for shard in shards:
            shard_dir = out_dir / args.shard_dir_fmt.format(i=shard.shard_id)
            shard_dataset_dir = shard_dir / "dataset"
            shard_dataset_dirs.append(shard_dataset_dir)
            log_path = shard_dir / "collect.log"
            exports, raw_exports = _client_exports(args)
            cmd = build_conda_command(
                conda_sh,
                args.client_conda_env,
                exports,
                _build_collect_argv(
                    args,
                    shard,
                    shard_dataset_dir,
                    shard_launch_mode=shard_launch_modes.get(shard.shard_id),
                ),
                raw_exports=raw_exports,
            )
            proc = launch_subprocess(cmd, log_path=log_path, cwd=PROJECT_ROOT)
            collect_procs.append((shard, proc, log_path, shard_dataset_dir))
            print(
                f"[collect] started collector shard={shard.shard_id} eps={shard.num_episodes} "
                f"pid={proc.pid} log={log_path} dataset={shard_dataset_dir}",
                flush=True,
            )

        exit_code = 0
        offsets: dict[int, int] = {}
        remaining = list(collect_procs)
        while remaining:
            time.sleep(5.0)
            still_running = []
            for shard, proc, log_path, dataset_dir in remaining:
                offsets[shard.shard_id] = _stream_log(
                    log_path,
                    prefix=f"shard{shard.shard_id}",
                    last_pos=offsets.get(shard.shard_id, 0),
                )
                rc = proc.poll()
                if rc is None:
                    still_running.append((shard, proc, log_path, dataset_dir))
                else:
                    print(f"[collect] collector shard={shard.shard_id} finished ({'ok' if rc == 0 else f'exit={rc}'})", flush=True)
                    offsets[shard.shard_id] = _stream_log(
                        log_path,
                        prefix=f"shard{shard.shard_id}",
                        last_pos=offsets.get(shard.shard_id, 0),
                    )
                    if rc != 0:
                        exit_code = 1
            remaining = still_running
            if remaining:
                names = ",".join(str(item[0].shard_id) for item in remaining)
                print(f"[collect] still running: shards [{names}]", flush=True)

        if exit_code != 0:
            return exit_code
        build_rc = _run_dataset_build(args, shard_dataset_dirs)
        if build_rc != 0:
            return build_rc
        print("[collect] done", flush=True)
        return 0
    finally:
        if args.launch_servers:
            cleanup_servers()


if __name__ == "__main__":
    raise SystemExit(main())
