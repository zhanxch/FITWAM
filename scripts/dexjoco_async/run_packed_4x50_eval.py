#!/usr/bin/env python3
"""Fastest single-ckpt official 4×50 DexJoCo eval via packed servers.

Starts ``servers_per_gpu`` policy servers on each physical GPU and runs all
4 official repeats concurrently. Each repeat shards its 50 episodes
(seeds 0..49) across ``N * (servers_per_gpu // 4)`` servers.

With the default ``servers_per_gpu=4`` on N GPUs this yields:
  N×4 servers total, 4 concurrent runs, each run uses all N GPUs once
  → exactly 4 servers packed on every GPU.

DexJoCo / DEWO v2 official eval uses the opensource stack
(``scripts/dexjoco/eval_opensource_4x50.sh``), not this packed local-async path.
"""

from __future__ import annotations

import argparse
import json
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
THIS_DIR = Path(__file__).resolve().parent
for _p in (PROJECT_ROOT, SCRIPTS_ROOT, THIS_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from multi_gpu_eval_utils import find_free_ports, terminate_process  # noqa: E402

NUM_RUNS = 4
EPISODES_PER_RUN = 50
BASE_SEED = 0
MULTI_GPU_SCRIPT = THIS_DIR / "run_multi_gpu_dexjoco_eval.py"


def _parse_int_list(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Packed official 4×50 eval: servers_per_gpu × N GPUs servers, "
            "all 4 runs concurrent."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--gpus", type=str, required=True, help="Physical GPU ids, e.g. 4,5,6,7")
    p.add_argument(
        "--servers-per-gpu",
        type=int,
        default=4,
        help="Policy servers packed onto each physical GPU. Must be divisible by 4.",
    )
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--run-dir", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--ckpt-tag", type=str, default=None, help="Label in aggregate.json")
    p.add_argument("--method-name", type=str, default="packed_4x50")
    p.add_argument("--server-conda-env", type=str, default="fastwam")
    p.add_argument("--client-conda-env", type=str, default="dexjoco")
    p.add_argument("--norm-stats-meta-dir", type=str, default=None)
    p.add_argument("--dataset-stats-path", type=str, default=None)
    p.add_argument("--text-embedding-cache-dir", type=str, default=None)
    p.add_argument("--task-config-dir", type=str, required=True)
    p.add_argument("--tasks", type=str, default="water_plant")
    p.add_argument("--dexjoco-py-root", type=str, default=str(PROJECT_ROOT / "third_party/dexjoco/dexjoco"))
    p.add_argument("--replan-steps", type=int, default=25)
    p.add_argument("--control-mode", type=str, default="blocking")
    p.add_argument("--max-env-steps", type=int, default=1500)
    p.add_argument("--video-fps", type=int, default=30)
    p.add_argument("--num-inference-steps", type=int, default=None)
    p.add_argument("--server-num-workers", type=int, default=8)
    p.add_argument("--server-startup-timeout", type=float, default=1200.0)
    p.add_argument("--no-load-text-encoder", action="store_true")
    p.add_argument("--no-randomize", action="store_true")
    p.add_argument("--no-randomize-dynamics", action="store_true")
    p.add_argument("--save-video", action="store_true")
    p.add_argument("--save-actions", action="store_true")
    p.add_argument("--no-action-clip", action="store_true")
    p.add_argument(
        "--protocol-label",
        type=str,
        default="official_4x50_seeds_0_49",
        help="Written into aggregate.json (override if max_env_steps != 1500).",
    )
    return p.parse_args()


def _run_gpu_list(physical: list[int], copies_per_gpu: int) -> list[int]:
    out: list[int] = []
    for g in physical:
        out.extend([g] * copies_per_gpu)
    return out


def _build_run_argv(
    args: argparse.Namespace,
    *,
    run_gpus: list[int],
    ports: list[int],
    out_dir: Path,
    eval_repeat: int = 0,
) -> list[str]:
    argv = [
        sys.executable,
        str(MULTI_GPU_SCRIPT),
        "--gpus",
        ",".join(str(g) for g in run_gpus),
        "--ports",
        ",".join(str(p) for p in ports),
        "--episodes",
        str(EPISODES_PER_RUN),
        "--seed",
        str(BASE_SEED),
        "--eval-repeat",
        str(int(eval_repeat)),
        "--server-conda-env",
        args.server_conda_env,
        "--client-conda-env",
        args.client_conda_env,
        "--run-dir",
        args.run_dir,
        "--checkpoint",
        args.checkpoint,
        "--task-config-dir",
        args.task_config_dir,
        "--tasks",
        args.tasks,
        "--dexjoco-py-root",
        args.dexjoco_py_root,
        "--replan-steps",
        str(args.replan_steps),
        "--control-mode",
        args.control_mode,
        "--max-env-steps",
        str(args.max_env_steps),
        "--video-fps",
        str(args.video_fps),
        "--server-num-workers",
        str(args.server_num_workers),
        "--server-startup-timeout",
        str(args.server_startup_timeout),
        "--output-dir",
        str(out_dir),
    ]
    if args.norm_stats_meta_dir:
        argv += ["--norm-stats-meta-dir", args.norm_stats_meta_dir]
    if args.dataset_stats_path:
        argv += ["--dataset-stats-path", args.dataset_stats_path]
    if args.text_embedding_cache_dir:
        argv += ["--text-embedding-cache-dir", args.text_embedding_cache_dir]
    if args.num_inference_steps is not None:
        argv += ["--num-inference-steps", str(args.num_inference_steps)]
    if args.no_load_text_encoder:
        argv.append("--no-load-text-encoder")
    if args.no_randomize:
        argv.append("--no-randomize")
    if args.no_randomize_dynamics:
        argv.append("--no-randomize-dynamics")
    if args.save_video:
        argv.append("--save-video")
    if args.save_actions:
        argv.append("--save-actions")
    if args.no_action_clip:
        argv.append("--no-action-clip")
    return argv


def _aggregate(out_root: Path, *, method: str, ckpt_tag: str, protocol: str, max_env_steps: int) -> dict[str, Any]:
    rates: list[float] = []
    pooled_s = pooled_n = 0
    rows: list[dict[str, Any]] = []
    for i in range(1, NUM_RUNS + 1):
        d = json.loads((out_root / f"run{i}" / "summary.json").read_text())
        rate = float(d["overall_success_rate"])
        s = int(d["total_successes"])
        n = int(d["total_episodes"])
        rates.append(rate)
        pooled_s += s
        pooled_n += n
        rows.append({"run": i, "successes": s, "episodes": n, "rate": rate, "seed": d.get("seed")})
    mean = statistics.fmean(rates)
    var = statistics.pvariance(rates) if len(rates) > 1 else 0.0
    std = statistics.pstdev(rates) if len(rates) > 1 else 0.0
    agg = {
        "method": method,
        "ckpt_tag": ckpt_tag,
        "protocol": protocol,
        "max_env_steps": max_env_steps,
        "packing": "servers_per_gpu_concurrent_runs",
        "runs": rows,
        "mean_success_rate": mean,
        "var_success_rate": var,
        "std_success_rate": std,
        "pooled_successes": pooled_s,
        "pooled_episodes": pooled_n,
        "pooled_success_rate": (pooled_s / pooled_n if pooled_n else None),
    }
    (out_root / "aggregate.json").write_text(json.dumps(agg, indent=2) + "\n")
    return agg


def main() -> int:
    args = parse_args()
    physical = _parse_int_list(args.gpus)
    if not physical:
        raise SystemExit("--gpus must list ≥1 GPU")
    spg = int(args.servers_per_gpu)
    if spg < 1:
        raise SystemExit("--servers-per-gpu must be ≥1")
    if spg % NUM_RUNS != 0:
        raise SystemExit(
            f"--servers-per-gpu={spg} must be divisible by {NUM_RUNS} "
            f"so each of the {NUM_RUNS} runs gets the same per-GPU packing"
        )
    copies = spg // NUM_RUNS
    run_gpus = _run_gpu_list(physical, copies)
    workers_per_run = len(run_gpus)
    total_servers = workers_per_run * NUM_RUNS

    out_root = args.output_dir.expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    ckpt_tag = args.ckpt_tag or Path(args.checkpoint).stem

    print(
        f"[packed-4x50] gpus={physical} servers_per_gpu={spg} "
        f"→ {total_servers} servers ({workers_per_run}/run × {NUM_RUNS} runs)",
        flush=True,
    )
    print(f"[packed-4x50] per-run gpu list={run_gpus}", flush=True)
    print(f"[packed-4x50] ckpt={args.checkpoint}", flush=True)
    print(f"[packed-4x50] out={out_root}", flush=True)
    print(
        f"[packed-4x50] protocol=4×50 seeds={BASE_SEED}..{BASE_SEED + EPISODES_PER_RUN - 1} "
        f"max_env_steps={args.max_env_steps} replan={args.replan_steps} mode={args.control_mode}",
        flush=True,
    )

    # Pre-allocate all ports so concurrent orchestrators cannot collide.
    all_ports = find_free_ports(total_servers)
    procs: list[tuple[int, subprocess.Popen[Any], Path]] = []

    def _cleanup() -> None:
        for run_i, proc, _ in procs:
            terminate_process(proc, label=f"packed-run{run_i}")

    def _on_signal(signum, _frame):
        print(f"\n[packed-4x50] signal {signum}, shutting down...", flush=True)
        _cleanup()
        raise SystemExit(130)

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    try:
        for run_i in range(1, NUM_RUNS + 1):
            run_dir = out_root / f"run{run_i}"
            run_dir.mkdir(parents=True, exist_ok=True)
            log_path = run_dir / "orchestrator.log"
            port_slice = all_ports[(run_i - 1) * workers_per_run : run_i * workers_per_run]
            argv = _build_run_argv(
                args,
                run_gpus=run_gpus,
                ports=port_slice,
                out_dir=run_dir,
                eval_repeat=run_i - 1,
            )
            with log_path.open("w", encoding="utf-8") as logf:
                proc = subprocess.Popen(
                    argv,
                    cwd=str(PROJECT_ROOT),
                    stdout=logf,
                    stderr=subprocess.STDOUT,
                )
            procs.append((run_i, proc, log_path))
            print(
                f"[packed-4x50] launched run{run_i}/{NUM_RUNS} pid={proc.pid} "
                f"gpus={run_gpus} ports={port_slice} log={log_path}",
                flush=True,
            )

        # Wait; stream a heartbeat from each run log.
        offsets = {run_i: 0 for run_i, _, _ in procs}
        pending = list(procs)
        while pending:
            time.sleep(5.0)
            still: list[tuple[int, subprocess.Popen[Any], Path]] = []
            for run_i, proc, log_path in pending:
                if log_path.exists():
                    try:
                        text = log_path.read_text(encoding="utf-8", errors="replace")
                        chunk = text[offsets[run_i] :]
                        offsets[run_i] = len(text)
                        for line in chunk.splitlines():
                            if line.strip() and (
                                "combined:" in line
                                or "success=" in line
                                or "EXIT" in line
                                or "ready" in line
                                or "FAILED" in line
                            ):
                                print(f"  [run{run_i}] {line}", flush=True)
                    except Exception:
                        pass
                rc = proc.poll()
                if rc is None:
                    still.append((run_i, proc, log_path))
                elif rc != 0:
                    print(f"[packed-4x50] run{run_i} FAILED rc={rc}; see {log_path}", flush=True)
                    _cleanup()
                    return int(rc)
                else:
                    print(f"[packed-4x50] run{run_i} finished OK", flush=True)
            pending = still

        for run_i in range(1, NUM_RUNS + 1):
            sp = out_root / f"run{run_i}" / "summary.json"
            if not sp.exists():
                print(f"[packed-4x50] missing {sp}", flush=True)
                return 3

        protocol = args.protocol_label
        if int(args.max_env_steps) != 1500 and protocol == "official_4x50_seeds_0_49":
            protocol = f"4x50_seeds_0_49_max_env_steps_{args.max_env_steps}"

        agg = _aggregate(
            out_root,
            method=args.method_name,
            ckpt_tag=ckpt_tag,
            protocol=protocol,
            max_env_steps=int(args.max_env_steps),
        )
        print(
            f"[packed-4x50] aggregate mean={agg['mean_success_rate']:.4f} "
            f"std={agg['std_success_rate']:.4f} "
            f"pooled={agg['pooled_successes']}/{agg['pooled_episodes']}",
            flush=True,
        )
        print(json.dumps(agg, indent=2), flush=True)
        return 0
    except Exception:
        _cleanup()
        raise


if __name__ == "__main__":
    raise SystemExit(main())
