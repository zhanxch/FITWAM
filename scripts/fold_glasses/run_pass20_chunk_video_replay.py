#!/usr/bin/env python3
"""Launch Pass@20 chunk-video replay workers. Does not write into scan shards."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
WORKER = ROOT / "scripts" / "fold_glasses" / "replay_pass20_chunk_videos.py"


def parse_gpus(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", default="2,3")
    parser.add_argument("--workers-per-gpu", type=int, default=3)
    parser.add_argument("--scan-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--log-dir", type=Path, default=None)
    args, passthrough = parser.parse_known_args()
    gpus = parse_gpus(args.gpus)
    if not gpus:
        raise SystemExit("No GPUs specified")
    if args.workers_per_gpu < 1:
        raise SystemExit("--workers-per-gpu must be positive")
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    log_dir = (args.log_dir or (output / "logs")).expanduser().resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    world = len(gpus) * int(args.workers_per_gpu)
    procs: list[subprocess.Popen] = []
    for rank in range(world):
        gpu = gpus[rank % len(gpus)]
        log_path = log_dir / f"replay_rank{rank}.log"
        cmd = [
            args.python,
            str(WORKER),
            "--scan-root",
            str(args.scan_root.expanduser().resolve()),
            "--output",
            str(output),
            "--shard-rank",
            str(rank),
            "--shard-world",
            str(world),
            *passthrough,
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        env.setdefault("MUJOCO_GL", "egl")
        env.setdefault("PYOPENGL_PLATFORM", "egl")
        log_file = log_path.open("w", encoding="utf-8")
        print(f"[video-orch] rank={rank} gpu={gpu} log={log_path}", flush=True)
        procs.append(
            subprocess.Popen(
                cmd,
                cwd=str(ROOT),
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )
        )
    failed = False
    for rank, proc in enumerate(procs):
        code = proc.wait()
        if code != 0:
            failed = True
            print(f"[video-orch] rank {rank} failed with {code}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
