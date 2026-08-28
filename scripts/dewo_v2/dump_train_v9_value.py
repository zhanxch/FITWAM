#!/usr/bin/env python3
"""Dump recoverability value V(s) on the DEWO v9 training pool (offline, no env).

Queries: succ events (D+), fail events (D_fail), zero prefixes on succ episodes (D0).
Use merge + --plot for succ/fail/zero histograms to tune CFG_GROWTH_TAU / drop_delta.

  python scripts/dewo_v2/dump_train_v9_value.py --run-dir RUN --ckpt CKPT \\
    --out-dir OUT --shard-index 0 --num-shards 4 --device cuda:0 ...

  python scripts/dewo_v2/dump_train_v9_value.py --merge --out-dir OUT --plot OUT/succ_fail_value.png
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
SRC = ROOT / "src"
DEXJOCO_ASYNC = SCRIPTS / "dexjoco_async"
for path in (ROOT, SRC, SCRIPTS, DEXJOCO_ASYNC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from fastwam.datasets.eve.tau_queries import (  # noqa: E402
    TauQuery,
    collect_v9_value_queries,
    shard_queries,
)
from scripts.dewo_v2.dump_train_cfg_gate_rms import (  # noqa: E402
    CAMERA_VIDEO_KEYS,
    _build_env_obs,
    _dataset_chunks_size,
    _decode_rgb_frames,
    _load_manifest_units,
    _load_proprio_frames,
    _parquet_path,
    _unwrap_action,
    _video_path,
)

DEFAULT_STATS = ROOT / "artifacts" / "mixed_5task" / "dataset_stats.json"


def _cfg_value(pred: dict[str, Any]) -> float:
    value = pred.get("cfg_value")
    if value is None:
        raise RuntimeError("infer_action returned no cfg_value (is value_head enabled?)")
    return float(np.asarray(value, dtype=np.float64).reshape(-1)[0])


def _queries_from_manifest(args: argparse.Namespace) -> list[TauQuery]:
    manifest_path = Path(args.manifest).expanduser().resolve()
    units = _load_manifest_units(manifest_path)
    queries = collect_v9_value_queries(
        units,
        replan_steps=int(args.replan_steps),
        prefix_fraction=float(args.prefix_fraction),
        max_zero_per_episode=args.max_zero_per_episode,
    )
    print(
        f"[v9-value] manifest={manifest_path} units={len(units)} "
        f"succ={sum(q.kind == 'succ' for q in queries)} "
        f"fail={sum(q.kind == 'fail' for q in queries)} "
        f"zero={sum(q.kind == 'zero' for q in queries)} total={len(queries)}",
        flush=True,
    )
    return queries


def _run_dump(args: argparse.Namespace) -> None:
    from dexjoco_fastwam_adapter import DexJoCoFastWAMAdapter, load_dexjoco_eval_settings, load_task_configs
    from run_fastwam_server import _build_policy_from_run

    run_dir = Path(args.run_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    queries = shard_queries(
        _queries_from_manifest(args),
        shard_index=int(args.shard_index),
        num_shards=int(args.num_shards),
    )
    shard_path = out_dir / f"shard_{int(args.shard_index):02d}.jsonl"
    if args.list_only:
        print(json.dumps({"shard": int(args.shard_index), "n": len(queries)}, indent=2))
        for query in queries:
            print(json.dumps(query.as_dict()))
        return

    print(f"Loading FastWAM model on {args.device} ...", flush=True)
    policy = _build_policy_from_run(
        run_dir=run_dir,
        checkpoint=str(args.ckpt),
        dataset_stats_path=str(args.dataset_stats),
        norm_stats_meta_dir=None,
        device=str(args.device),
        action_horizon=int(args.action_horizon),
        num_inference_steps=int(args.num_inference_steps),
        load_text_encoder=False,
        inference_seed=int(args.inference_seed),
        text_cfg_scale=1.0,
        negative_prompt=None,
        backbone_checkpoint=str(args.backbone) if args.backbone else None,
        uncond_adapter=None,
        adaptive_cfg_tau=0.0,
        cfg_epsilon_l=None,
        cfg_residual_clip_mode="rms",
    )
    settings = load_dexjoco_eval_settings(
        run_dir,
        action_horizon_override=int(args.action_horizon),
        text_embedding_cache_dir_override=args.text_embedding_cache_dir,
    )
    adapter = DexJoCoFastWAMAdapter(settings)
    task_cfg_dir = Path(args.task_config_dir).expanduser().resolve()
    task_cfgs = load_task_configs(task_cfg_dir)
    matched = [cfg for cfg in task_cfgs if str(cfg.get("env_name")) == str(args.task)]
    if not matched:
        raise ValueError(f"Task {args.task} not in {task_cfg_dir}")
    task_cfg = matched[0]
    camera_mapping = {str(k): str(v) for k, v in (task_cfg.get("camera_mapping") or {}).items()}
    camera_key = str(camera_mapping.get("base", "front"))
    image_keys = list(adapter.image_keys) or ["front", "wrist"]

    grouped: dict[tuple[str, int], list[TauQuery]] = defaultdict(list)
    for query in queries:
        grouped[(query.dataset_root, query.episode_index)].append(query)

    rows: list[dict[str, Any]] = []
    for (root_s, episode_index), group in sorted(grouped.items()):
        root = Path(root_s)
        chunks_size = _dataset_chunks_size(root)
        frame_ids = [q.frame_index for q in group]
        cameras: dict[str, dict[int, np.ndarray]] = {}
        for cam in image_keys:
            video_key = CAMERA_VIDEO_KEYS.get(cam, f"observation.images.{cam}")
            video_path = _video_path(root, episode_index, video_key, chunks_size)
            cameras[cam] = _decode_rgb_frames(video_path, frame_ids)
        proprio = _load_proprio_frames(
            _parquet_path(root, episode_index, chunks_size),
            frame_ids,
        )
        for query in group:
            env_obs = _build_env_obs(
                cameras={cam: cameras[cam][query.frame_index] for cam in image_keys},
                proprio=proprio[query.frame_index],
            )
            observation = adapter.env_obs_to_policy_obs(
                env_obs,
                camera_key=camera_key,
                camera_mapping=camera_mapping,
                task_prompt=str(task_cfg["prompt"]),
                cfg_base_prompt=(
                    None
                    if task_cfg.get("cfg_base_prompt") is None
                    else str(task_cfg["cfg_base_prompt"])
                ),
                cfg_failure_prompt=(
                    None
                    if task_cfg.get("cfg_failure_prompt") is None
                    else str(task_cfg["cfg_failure_prompt"])
                ),
            )
            pred = _unwrap_action(
                policy.get_action(
                    observation,
                    options={
                        "seed": int(args.inference_seed),
                        "cfg_exec_horizon": int(args.replan_steps),
                    },
                )
            )
            value = _cfg_value(pred)
            row = query.as_dict()
            row["cfg_value"] = value
            rows.append(row)
            print(
                f"[dump] {query.kind} ep={query.episode_index} t={query.frame_index} V={value:.6f}",
                flush=True,
            )

    with shard_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    print(f"[dump] wrote {shard_path} n={len(rows)}", flush=True)


def _merge(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir).expanduser().resolve()
    rows: list[dict[str, Any]] = []
    shard_files = sorted(out_dir.glob("shard_*.jsonl"))
    if not shard_files:
        raise FileNotFoundError(f"No shard_*.jsonl under {out_dir}")
    for path in shard_files:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))

    def _values(kind: str) -> list[float]:
        return [float(r["cfg_value"]) for r in rows if r.get("kind") == kind]

    buckets = {kind: _values(kind) for kind in ("succ", "fail", "zero")}
    summary = {
        "n_total": len(rows),
        "n_succ": len(buckets["succ"]),
        "n_fail": len(buckets["fail"]),
        "n_zero": len(buckets["zero"]),
        "succ_mean": float(np.mean(buckets["succ"])) if buckets["succ"] else None,
        "fail_mean": float(np.mean(buckets["fail"])) if buckets["fail"] else None,
        "zero_mean": float(np.mean(buckets["zero"])) if buckets["zero"] else None,
        "succ_p50": float(np.median(buckets["succ"])) if buckets["succ"] else None,
        "fail_p50": float(np.median(buckets["fail"])) if buckets["fail"] else None,
        "zero_p50": float(np.median(buckets["zero"])) if buckets["zero"] else None,
        "shards": [str(p) for p in shard_files],
    }
    for kind in ("succ", "fail", "zero"):
        (out_dir / f"v_{kind}.json").write_text(
            json.dumps(buckets[kind], indent=2) + "\n", encoding="utf-8"
        )
    (out_dir / "dump_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)

    if args.plot:
        import matplotlib.pyplot as plt

        plot_path = Path(args.plot).expanduser().resolve()
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        labels = []
        data = []
        colors = {"succ": "#2ca02c", "fail": "#d62728", "zero": "#1f77b4"}
        for kind in ("succ", "fail", "zero"):
            if buckets[kind]:
                labels.append(f"{kind} (n={len(buckets[kind])})")
                data.append(buckets[kind])
        if data:
            for kind in ("succ", "fail", "zero"):
                if buckets[kind]:
                    ax.hist(
                        buckets[kind],
                        bins=30,
                        alpha=0.55,
                        label=f"{kind} (n={len(buckets[kind])})",
                        color=colors[kind],
                    )
            ax.set_xlabel("V(s) — recoverability value")
            ax.set_ylabel("count")
            ax.set_title("DEWO v9 training pool value dump")
            ax.legend()
            fig.tight_layout()
            fig.savefig(plot_path, dpi=160)
            plt.close(fig)
            print(f"[v9-value] wrote plot {plot_path}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--backbone", type=str, default=None)
    parser.add_argument("--dataset-stats", type=Path, default=DEFAULT_STATS)
    parser.add_argument("--text-embedding-cache-dir", type=str, required=True)
    parser.add_argument("--task-config-dir", type=Path, required=True)
    parser.add_argument("--task", type=str, default="fold_glasses")
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--replan-steps", type=int, default=24)
    parser.add_argument("--prefix-fraction", type=float, default=0.5)
    parser.add_argument("--max-zero-per-episode", type=int, default=None)
    parser.add_argument("--action-horizon", type=int, default=32)
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--inference-seed", type=int, default=20260812)
    parser.add_argument("--list-only", action="store_true")
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--plot", type=str, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.merge:
        _merge(args)
        return 0
    _run_dump(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
