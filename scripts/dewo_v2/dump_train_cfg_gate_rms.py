#!/usr/bin/env python3
"""Dump NFE0 exec RMS on the DEWO v6/v7 *training pool* (D+ events, D0 prefixes).

This is not the 200-episode collect set. Queries come from the v6 Eve manifest
(success-event primaries + success episodes). Does not step the env.

v7 energy is RMS(ε_+ − ε_-); the eval yaml must include cfg_failure_prompt.

  python scripts/dewo_v2/dump_train_cfg_gate_rms.py \
    --run-dir RUN --ckpt ADAPTER --backbone BACKBONE \
    --shard-index 0 --num-shards 4 --device cuda:0 --out-dir ...

  python scripts/dewo_v2/dump_train_cfg_gate_rms.py --merge --out-dir ... \
    --calibrate-json RUN/adaptive_cfg_tau.json
"""

from __future__ import annotations

import argparse
import json
import os
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
    collect_v6_tau_queries,
    shard_queries,
)
from fastwam.models.wan22.uncond_adapter import write_adaptive_cfg_tau_json  # noqa: E402

DEFAULT_RUN = (
    ROOT
    / "runs"
    / "dexjoco_water_plant_dewo_v6"
    / "2026-08-24_14-51-57_B1-jump-fast-v6-uncond-adapter"
)
DEFAULT_TASK_CFG_DIR = ROOT / "configs" / "eval" / "dexjoco" / "water_plant_dewo_v6_cfg"
DEFAULT_STATS = ROOT / "artifacts" / "mixed_5task" / "dataset_stats.json"
DEFAULT_TEXT_CACHE = (
    ROOT
    / "data"
    / "water_plant_mixed_s0_dewo_v2_pair_20260820_182236"
    / "text_embeds_cache"
)
CAMERA_VIDEO_KEYS = {
    "front": "observation.images.front",
    "wrist": "observation.images.wrist",
}


def _load_manifest_units(manifest_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    units = payload.get("samples") or payload.get("units") or []
    if not isinstance(units, list):
        raise ValueError(f"No sample list in {manifest_path}")
    return list(units)


def _nfe0_exec_rms(pred: dict[str, Any], exec_horizon: int) -> float:
    token_nfe = pred.get("cfg_token_rms_nfe")
    if token_nfe is not None:
        arr = np.asarray(token_nfe, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] < 1:
            raise ValueError(f"cfg_token_rms_nfe shape {arr.shape} is not [nfe, T]")
        horizon = min(int(exec_horizon), int(arr.shape[1]))
        return float(arr[0, :horizon].mean())
    gate = pred.get("cfg_gate_exec_rms")
    if gate is None:
        raise RuntimeError("infer_action returned neither cfg_token_rms_nfe nor cfg_gate_exec_rms")
    return float(np.asarray(gate, dtype=np.float32).reshape(-1)[0])


def _unwrap_action(pred: Any) -> dict[str, Any]:
    if isinstance(pred, tuple):
        pred = pred[0]
    if not isinstance(pred, dict):
        raise TypeError(f"get_action returned {type(pred)}")
    return pred


def _video_path(root: Path, episode_index: int, video_key: str, chunks_size: int) -> Path:
    chunk = int(episode_index) // int(chunks_size)
    return root / "videos" / f"chunk-{chunk:03d}" / video_key / f"episode_{int(episode_index):06d}.mp4"


def _parquet_path(root: Path, episode_index: int, chunks_size: int) -> Path:
    chunk = int(episode_index) // int(chunks_size)
    return root / "data" / f"chunk-{chunk:03d}" / f"episode_{int(episode_index):06d}.parquet"


def _dataset_chunks_size(root: Path) -> int:
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        return 1000
    info = json.loads(info_path.read_text(encoding="utf-8"))
    return int(info.get("chunks_size", info.get("chunks_size", 1000)) or 1000)


def _decode_rgb_frames(video_path: Path, frame_indices: list[int]) -> dict[int, np.ndarray]:
    import av

    needed = set(int(i) for i in frame_indices)
    got: dict[int, np.ndarray] = {}
    with av.open(str(video_path)) as container:
        for i, frame in enumerate(container.decode(video=0)):
            if i in needed:
                got[i] = frame.to_ndarray(format="rgb24")
                if len(got) == len(needed):
                    break
    missing = sorted(needed - got.keys())
    if missing:
        raise FileNotFoundError(f"{video_path} missing frames {missing[:8]}")
    return got


def _load_proprio_frames(parquet_path: Path, frame_indices: list[int]) -> dict[int, np.ndarray]:
    import pyarrow.parquet as pq

    table = pq.read_table(parquet_path)
    names = set(table.column_names)
    state_col = "observation.state" if "observation.state" in names else "state"
    if state_col not in names:
        raise KeyError(f"{parquet_path} has no state column; columns={sorted(names)[:20]}")
    states = table.column(state_col).to_pylist()
    needed = [int(i) for i in frame_indices]
    out: dict[int, np.ndarray] = {}
    for idx in needed:
        if idx < 0 or idx >= len(states):
            raise IndexError(f"{parquet_path} frame {idx} out of range {len(states)}")
        out[idx] = np.asarray(states[idx], dtype=np.float32).reshape(-1)
    return out


def _build_env_obs(
    *,
    cameras: dict[str, np.ndarray],
    proprio: np.ndarray,
) -> dict[str, Any]:
    obs: dict[str, Any] = {"state": np.asarray(proprio, dtype=np.float32)}
    obs.update(cameras)
    return obs


def _uncond_recipe(run_dir: Path) -> str:
    cfg_path = run_dir / "config.yaml"
    if not cfg_path.is_file():
        return "v6"
    import yaml

    payload = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    return str(
        ((payload.get("model") or {}).get("uncond_adapter") or {}).get("recipe") or "v6"
    )


def _queries_from_run(run_dir: Path, args: argparse.Namespace) -> list[TauQuery]:
    cfg_path = run_dir / "config.yaml"
    import yaml

    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    manifest_path = Path(
        args.manifest
        or cfg["data"]["train"]["manifest_path"]
    ).expanduser().resolve()
    units = _load_manifest_units(manifest_path)
    queries = collect_v6_tau_queries(
        units,
        replan_steps=int(args.replan_steps),
        prefix_fraction=float(args.prefix_fraction),
        max_zero_per_episode=args.max_zero_per_episode,
    )
    print(
        f"[tau] manifest={manifest_path} units={len(units)} "
        f"plus={sum(q.kind == 'plus' for q in queries)} "
        f"zero={sum(q.kind == 'zero' for q in queries)}",
        flush=True,
    )
    return queries


def _run_dump(args: argparse.Namespace) -> None:
    from dexjoco_fastwam_adapter import (  # noqa: E402
        DexJoCoFastWAMAdapter,
        load_dexjoco_eval_settings,
        load_task_configs,
    )
    from run_fastwam_server import _build_policy_from_run  # noqa: E402

    run_dir = Path(args.run_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    queries = shard_queries(
        _queries_from_run(run_dir, args),
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
        text_cfg_scale=float(args.text_cfg_scale),
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
    recipe = _uncond_recipe(run_dir)
    if recipe == "v7" and task_cfg_dir == DEFAULT_TASK_CFG_DIR.resolve():
        task_cfg_dir = (
            ROOT / "configs" / "eval" / "dexjoco" / f"{args.task}_dewo_v7_cfg"
        ).resolve()
    task_cfgs = load_task_configs(task_cfg_dir)
    matched = [cfg for cfg in task_cfgs if str(cfg.get("env_name")) == str(args.task)]
    if not matched:
        raise ValueError(f"Task {args.task} not in {task_cfg_dir}")
    task_cfg = matched[0]
    if recipe == "v7" and not task_cfg.get("cfg_failure_prompt"):
        raise ValueError(
            f"v7 dump requires cfg_failure_prompt in {task_cfg_dir} "
            f"(energy is RMS(ε_+ − ε_-), not RMS(ε_+ − ε_0))."
        )
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
                        "return_cfg_residual": True,
                        "cfg_exec_horizon": int(args.replan_steps),
                        "adaptive_cfg_tau": 0.0,
                    },
                )
            )
            rms = _nfe0_exec_rms(pred, int(args.replan_steps))
            row = query.as_dict()
            row["nfe0_exec_rms"] = rms
            rows.append(row)
            print(
                f"[dump] {query.kind} ep={query.episode_index} t={query.frame_index} E={rms:.6f}",
                flush=True,
            )

    with shard_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    print(f"[dump] wrote {shard_path} n={len(rows)}", flush=True)


def _merge_and_calibrate(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir).expanduser().resolve()
    rows: list[dict[str, Any]] = []
    shard_files = sorted(out_dir.glob("shard_*.jsonl"))
    if not shard_files:
        raise FileNotFoundError(f"No shard_*.jsonl under {out_dir}")
    for path in shard_files:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    plus = [float(r["nfe0_exec_rms"]) for r in rows if r.get("kind") == "plus"]
    zero = [float(r["nfe0_exec_rms"]) for r in rows if r.get("kind") == "zero"]
    (out_dir / "e_plus.json").write_text(json.dumps(plus, indent=2) + "\n", encoding="utf-8")
    (out_dir / "e_zero.json").write_text(json.dumps(zero, indent=2) + "\n", encoding="utf-8")
    summary = {
        "n_plus": len(plus),
        "n_zero": len(zero),
        "e_plus_mean": float(np.mean(plus)) if plus else None,
        "e_zero_mean": float(np.mean(zero)) if zero else None,
        "shards": [str(p) for p in shard_files],
    }
    (out_dir / "dump_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    if args.calibrate_json:
        recipe = _uncond_recipe(Path(args.run_dir).expanduser())
        payload = write_adaptive_cfg_tau_json(
            args.calibrate_json,
            plus,
            zero,
            recall=float(args.recall),
            max_fpr0=float(args.max_fpr0),
            recipe=recipe,
        )
        print(json.dumps(payload, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--backbone", type=str, default=None)
    parser.add_argument("--dataset-stats", type=Path, default=DEFAULT_STATS)
    parser.add_argument("--text-embedding-cache-dir", type=str, default=str(DEFAULT_TEXT_CACHE))
    parser.add_argument("--task-config-dir", type=Path, default=DEFAULT_TASK_CFG_DIR)
    parser.add_argument("--task", type=str, default="water_plant")
    parser.add_argument("--manifest", type=str, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--replan-steps", type=int, default=24)
    parser.add_argument("--prefix-fraction", type=float, default=0.5)
    parser.add_argument("--max-zero-per-episode", type=int, default=None)
    parser.add_argument("--action-horizon", type=int, default=32)
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--text-cfg-scale", type=float, default=1.2)
    parser.add_argument("--inference-seed", type=int, default=20260812)
    parser.add_argument("--list-only", action="store_true")
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--calibrate-json", type=str, default=None)
    parser.add_argument("--recall", type=float, default=0.90)
    parser.add_argument("--max-fpr0", type=float, default=0.05)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.merge:
        _merge_and_calibrate(args)
        return 0
    if not args.ckpt:
        args.ckpt = str(Path(args.run_dir) / "checkpoints" / "weights" / "step_001500.pt")
    _run_dump(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
