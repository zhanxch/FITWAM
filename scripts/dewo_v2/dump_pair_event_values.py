#!/usr/bin/env python3
"""Dump V on pair failure prefixes: label t* vs earlier replans (true event labels)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
SRC = ROOT / "src"
DEXJOCO_ASYNC = SCRIPTS / "dexjoco_async"
for path in (ROOT, SRC, SCRIPTS, DEXJOCO_ASYNC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from fastwam.datasets.eve.tau_queries import TauQuery  # noqa: E402
from scripts.dewo_v2.dump_train_cfg_gate_rms import (  # noqa: E402
    CAMERA_VIDEO_KEYS,
    _build_env_obs,
    _dataset_chunks_size,
    _decode_rgb_frames,
    _load_proprio_frames,
    _parquet_path,
    _unwrap_action,
    _video_path,
)
from scripts.dewo_v2.dump_train_v9_value import _cfg_value  # noqa: E402


def _build_queries(
    pair_index: Path,
    dataset_root: Path,
    replan: int,
    episodes_len: dict[int, int] | None = None,
) -> list[tuple[TauQuery, dict]]:
    payload = json.loads(pair_index.read_text(encoding="utf-8"))
    pairs = payload["pairs"] if isinstance(payload, dict) else payload
    root_s = str(dataset_root.resolve())
    out: list[tuple[TauQuery, dict]] = []
    for pair in pairs:
        pid = str(pair["pair_id"])
        fail_ep = int(pair["failure_episode_index"])
        t_star = int(pair["t_star_last_recoverable"])
        succ_ep = int(pair["success_episode_index"])
        succ_len = int(pair["success_length"])
        fail_len = int(pair.get("failure_length") or 0)
        if episodes_len and fail_ep in episodes_len:
            fail_len = int(episodes_len[fail_ep])
        if fail_len <= 0:
            fail_len = max(t_star, replan)
        # Last in-episode frame index (0-based)
        last_frame = max(0, min(t_star, fail_len) - 1)
        t_event = last_frame - (last_frame % replan)
        meta_event = {"pair_id": pid, "is_event": True, "role": "fail_tstar"}
        out.append(
            (
                TauQuery(
                    kind="fail",
                    sample_id=pid,
                    dataset_root=root_s,
                    episode_index=fail_ep,
                    frame_index=t_event,
                ),
                meta_event,
            )
        )
        neg_frames = [0]
        mid = max(0, t_event - replan)
        if mid not in neg_frames and mid < t_event:
            neg_frames.append(mid)
        for frame in neg_frames:
            out.append(
                (
                    TauQuery(
                        kind="zero",
                        sample_id=pid,
                        dataset_root=root_s,
                        episode_index=fail_ep,
                        frame_index=frame,
                    ),
                    {"pair_id": pid, "is_event": False, "role": "fail_prefix"},
                )
            )
        t_norm = t_star / max(succ_len, 1)
        f_succ = int(t_norm * succ_len)
        f_succ = min(max(0, f_succ - (f_succ % replan)), max(0, succ_len - 1))
        out.append(
            (
                TauQuery(
                    kind="succ",
                    sample_id=pid,
                    dataset_root=root_s,
                    episode_index=succ_ep,
                    frame_index=f_succ,
                ),
                {"pair_id": pid, "is_event": False, "role": "succ_same_norm_t"},
            )
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pair-index",
        type=Path,
        default=ROOT / "data/fold_glasses_dewo_v9_pair_full_lerobot/pair_index.json",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=ROOT / "data/fold_glasses_dewo_v9_pair_full_lerobot",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=ROOT
        / "runs/dexjoco_fold_glasses_dewo_v9/2026-08-27_11-21-16_B1-jump-fast-v9-uncond-adapter",
    )
    parser.add_argument("--ckpt", type=Path, default=Path("checkpoints/weights/step_005000.pt"))
    parser.add_argument(
        "--dataset-stats",
        type=Path,
        default=ROOT / "artifacts/mixed_5task/dataset_stats.json",
    )
    parser.add_argument(
        "--text-embedding-cache-dir",
        type=Path,
        default=ROOT / "data/text_embeds_cache/fold_glasses_dewo_v9_pair",
    )
    parser.add_argument(
        "--task-config-dir",
        type=Path,
        default=ROOT / "configs/eval/dexjoco/fold_glasses_dewo_v9_cfg",
    )
    parser.add_argument("--task", default="fold_glasses")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--replan-steps", type=int, default=24)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    from dexjoco_fastwam_adapter import DexJoCoFastWAMAdapter, load_dexjoco_eval_settings, load_task_configs
    from run_fastwam_server import _build_policy_from_run

    run_dir = args.run_dir.expanduser().resolve()
    ckpt = args.ckpt if args.ckpt.is_absolute() else (run_dir / args.ckpt)
    episodes_len: dict[int, int] = {}
    ep_path = args.dataset_root / "meta" / "episodes.jsonl"
    if ep_path.is_file():
        for line in ep_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                episodes_len[int(row["episode_index"])] = int(row["length"])
    queries = _build_queries(args.pair_index, args.dataset_root, int(args.replan_steps), episodes_len)
    print(f"[pair-dump] queries={len(queries)}", flush=True)

    policy = _build_policy_from_run(
        run_dir=run_dir,
        checkpoint=str(ckpt),
        dataset_stats_path=str(args.dataset_stats),
        norm_stats_meta_dir=None,
        device=str(args.device),
        action_horizon=32,
        num_inference_steps=10,
        load_text_encoder=False,
        inference_seed=0,
        text_cfg_scale=1.0,
        negative_prompt=None,
        backbone_checkpoint=None,
        uncond_adapter=None,
        adaptive_cfg_tau=None,
        cfg_epsilon_l=None,
        cfg_residual_clip_mode="rms",
    )
    settings = load_dexjoco_eval_settings(
        run_dir,
        text_embedding_cache_dir_override=str(args.text_embedding_cache_dir),
    )
    adapter = DexJoCoFastWAMAdapter(settings)
    task_cfgs = load_task_configs(args.task_config_dir.expanduser().resolve())
    matched = [c for c in task_cfgs if str(c.get("env_name")) == str(args.task)]
    if not matched:
        raise ValueError(f"task {args.task} not found")
    task_cfg = matched[0]
    camera_mapping = {str(k): str(v) for k, v in (task_cfg.get("camera_mapping") or {}).items()}
    camera_key = str(camera_mapping.get("base", "front"))
    image_keys = list(adapter.image_keys) or ["front", "wrist"]
    chunks_size = _dataset_chunks_size(args.dataset_root)

    rows: list[dict] = []
    grouped: dict[tuple[int, int], list[tuple[TauQuery, dict]]] = {}
    for query, meta in queries:
        grouped.setdefault((query.episode_index, query.frame_index), []).append((query, meta))

    for (episode_index, frame_index), group in sorted(grouped.items()):
        frame_ids = [frame_index]
        cameras: dict[str, dict[int, np.ndarray]] = {}
        for cam in image_keys:
            video_key = CAMERA_VIDEO_KEYS.get(cam, f"observation.images.{cam}")
            video_path = _video_path(args.dataset_root, episode_index, video_key, chunks_size)
            cameras[cam] = _decode_rgb_frames(video_path, frame_ids)
        proprio = _load_proprio_frames(
            _parquet_path(args.dataset_root, episode_index, chunks_size),
            frame_ids,
        )
        env_obs = _build_env_obs(
            cameras={cam: cameras[cam][frame_index] for cam in image_keys},
            proprio=proprio[frame_index],
        )
        observation = adapter.env_obs_to_policy_obs(
            env_obs,
            camera_key=camera_key,
            camera_mapping=camera_mapping,
            task_prompt=str(task_cfg["prompt"]),
            cfg_base_prompt=(
                None if task_cfg.get("cfg_base_prompt") is None else str(task_cfg["cfg_base_prompt"])
            ),
            cfg_failure_prompt=(
                None
                if task_cfg.get("cfg_failure_prompt") is None
                else str(task_cfg["cfg_failure_prompt"])
            ),
        )
        pred = _unwrap_action(policy.get_action(observation, options={"seed": 0}))
        value = _cfg_value(pred)
        for query, meta in group:
            row = query.as_dict()
            row.update(meta)
            row["cfg_value"] = value
            rows.append(row)
            print(
                f"[pair-dump] {meta['role']} pair={meta['pair_id']} ep={episode_index} "
                f"t={frame_index} event={meta['is_event']} V={value:.6f}",
                flush=True,
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    print(f"[pair-dump] wrote {args.out} n={len(rows)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
