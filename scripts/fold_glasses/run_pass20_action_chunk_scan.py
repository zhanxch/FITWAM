#!/usr/bin/env python3
"""Launch Pass@20 action-chunk analysis across GPUs and merge shard outputs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCANNER = ROOT / "scripts" / "fold_glasses" / "scan_failure_pass20_action_chunks.py"


def parse_gpus(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def merge_shards(output: Path, world: int) -> None:
    prefixes: list[dict] = []
    ledger: list[dict] = []
    summaries: list[dict] = []
    for rank in range(world):
        shard = output / f"shard{rank}"
        prefixes.extend(load_jsonl(shard / "prefix_results.jsonl"))
        ledger.extend(load_jsonl(shard / "action_chunk_ledger.jsonl"))
        summary_path = shard / "summary.json"
        if summary_path.exists():
            summaries.append(json.loads(summary_path.read_text(encoding="utf-8")))
        for name in ("prefixes", "episodes"):
            src = shard / name
            dst = output / name
            if not src.exists():
                continue
            dst.mkdir(parents=True, exist_ok=True)
            for child in src.iterdir():
                target = dst / child.name
                if target.exists() or target.is_symlink():
                    continue
                os.symlink(child.resolve(), target)
    prefixes.sort(
        key=lambda row: (
            int(row.get("source_failure_episode_index", -1)),
            int(row.get("prefix_frame", -1)),
        )
    )
    ledger.sort(
        key=lambda row: (
            int(row.get("episode_index", -1)),
            int(row.get("prefix_frame", -1)),
            int(row.get("replicate_index", -1)),
        )
    )
    write_jsonl(output / "prefix_results.jsonl", prefixes)
    write_jsonl(output / "action_chunk_ledger.jsonl", ledger)
    payload = {
        "format": "FoldGlassesPassAtKActionChunkScan",
        "status": "complete",
        "num_shards": world,
        "num_prefix_results": len(prefixes),
        "num_action_chunks": len(ledger),
        "shards": summaries,
    }
    (output / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, sort_keys=True), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", default="4,5,6,7")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    args, passthrough = parser.parse_known_args()
    gpus = parse_gpus(args.gpus)
    if not gpus:
        raise SystemExit("No GPUs specified")
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    world = len(gpus)
    procs: list[subprocess.Popen] = []
    for rank, gpu in enumerate(gpus):
        shard_out = output / f"shard{rank}"
        shard_out.mkdir(parents=True, exist_ok=True)
        log_path = output / "logs" / f"scan_shard{rank}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            args.python,
            str(SCANNER),
            "--dataset",
            str(args.dataset),
            "--output",
            str(shard_out),
            "--device",
            "cuda:0",
            "--shard-rank",
            str(rank),
            "--shard-world",
            str(world),
            *passthrough,
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        log_file = log_path.open("w", encoding="utf-8")
        print(f"[scan-orch] rank={rank} gpu={gpu} log={log_path}", flush=True)
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
            print(f"[scan-orch] shard {rank} failed with {code}", flush=True)
    if failed:
        return 1
    merge_shards(output, world)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
