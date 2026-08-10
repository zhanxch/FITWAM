#!/usr/bin/env python3
"""Screen expert episodes for the open-loop width story.

Target pattern
--------------
1. B1: wide in early progress, narrower in mid
2. S0: more stable width across the trajectory (low CV in mid / whole)

Uses a coarse open-loop probe (larger stride, fewer seeds) and ranks episodes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))

from openloop_action_interval_fixed_episode import (  # noqa: E402
    B1_TASK_PROMPT,
    DEFAULT_B1_CKPT,
    DEFAULT_B1_RUN,
    DEFAULT_EXPERT_DS,
    DEFAULT_NORM,
    DEFAULT_S0_CKPT,
    DEFAULT_S0_RUN,
    DEFAULT_TEXT,
    S0_TASK_PROMPT,
    _denormalize_action,
    _hwc_to_model_image,
    _instruction,
    _load_model,
    _load_text_context,
    _normalize_proprio,
    _read_video_rgb,
)

DEFAULT_OUT = ROOT / "results/openloop_width_episode_screen_20260808"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--expert-dataset", type=Path, default=DEFAULT_EXPERT_DS)
    p.add_argument("--output", type=Path, default=DEFAULT_OUT)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--episodes", type=str, default="all", help="comma list or 'all'")
    p.add_argument("--max-episodes", type=int, default=0, help="0=all selected")
    p.add_argument("--stride", type=int, default=12)
    p.add_argument("--num-samples", type=int, default=4)
    p.add_argument("--num-inference-steps", type=int, default=8)
    p.add_argument("--action-horizon", type=int, default=32)
    p.add_argument("--sample-seed0", type=int, default=20260808)
    p.add_argument("--early", type=str, default="0.10,0.35")
    p.add_argument("--mid", type=str, default="0.35,0.70")
    p.add_argument("--s0-run-dir", type=Path, default=DEFAULT_S0_RUN)
    p.add_argument("--s0-checkpoint", type=Path, default=DEFAULT_S0_CKPT)
    p.add_argument("--b1-run-dir", type=Path, default=DEFAULT_B1_RUN)
    p.add_argument("--b1-checkpoint", type=Path, default=DEFAULT_B1_CKPT)
    p.add_argument("--norm-stats-meta-dir", type=Path, default=DEFAULT_NORM)
    p.add_argument("--text-embedding-cache-dir", type=Path, default=DEFAULT_TEXT)
    return p.parse_args()


def parse_range(s: str) -> tuple[float, float]:
    a, b = s.split(",")
    return float(a), float(b)


def list_episodes(ds: Path, spec: str, max_n: int) -> list[int]:
    if spec.strip() == "all":
        eps = sorted(int(p.stem.split("_")[-1]) for p in (ds / "data" / "chunk-000").glob("episode_*.parquet"))
    else:
        eps = [int(x) for x in spec.split(",") if x.strip()]
    if max_n > 0:
        eps = eps[:max_n]
    return eps


def load_episode_arrays(
    ds: Path, ep: int, frame_idx: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    import pyarrow.parquet as pq

    parquet = ds / "data" / "chunk-000" / f"episode_{ep:06d}.parquet"
    front_v = ds / "videos" / "chunk-000" / "observation.images.front" / f"episode_{ep:06d}.mp4"
    wrist_v = ds / "videos" / "chunk-000" / "observation.images.wrist" / f"episode_{ep:06d}.mp4"
    table = pq.read_table(parquet, columns=["action", "observation.state"])
    actions = np.asarray(table.column("action").to_pylist(), dtype=np.float32)
    states = np.asarray(table.column("observation.state").to_pylist(), dtype=np.float32)
    if actions.shape[1] == 23:
        actions = actions[:, 1:]
    front = _read_video_rgb(front_v)
    wrist = _read_video_rgb(wrist_v)
    n = min(len(actions), len(states), len(front), len(wrist))
    frame_idx = np.asarray(frame_idx[frame_idx < n], dtype=np.int64)
    imgs = []
    for t in frame_idx:
        im = _hwc_to_model_image(front[int(t)], wrist[int(t)])
        if im.ndim == 4 and im.shape[0] == 1:
            im = im[0]
        imgs.append(im)
    images = np.stack(imgs, axis=0).astype(np.float32)
    return (
        images,
        states[frame_idx].astype(np.float32),
        actions[frame_idx].astype(np.float32),
        frame_idx,
        int(n),
    )


def infer_a0(
    *,
    model,
    processor,
    images: np.ndarray,
    proprios: np.ndarray,
    context,
    context_mask,
    num_samples: int,
    sample_seed0: int,
    action_horizon: int,
    num_inference_steps: int,
    device: str,
    ep_tag: int,
) -> np.ndarray:
    import torch

    T = len(images)
    rows = []
    for i in range(T):
        img = torch.from_numpy(images[i]).to(device=device, dtype=model.torch_dtype)
        if img.ndim == 3:
            img = img.unsqueeze(0)
        prop = _normalize_proprio(processor, proprios[i], device, model.torch_dtype)
        samples = []
        for k in range(num_samples):
            seed = int(sample_seed0 + 10007 * k + 17 * ep_tag + i)
            with torch.no_grad():
                pred = model.infer_action(
                    prompt=None,
                    input_image=img,
                    action_horizon=action_horizon,
                    proprio=prop,
                    context=context,
                    context_mask=context_mask,
                    num_inference_steps=num_inference_steps,
                    seed=seed,
                    rand_device="cpu",
                    tiled=False,
                )
            act = _denormalize_action(processor, pred["action"])
            if act.shape[-1] > 22:
                act = act[..., :22]
            samples.append(act[0].astype(np.float32))  # a0
        rows.append(np.stack(samples, axis=0))
    return np.stack(rows, axis=0)  # [T,K,D]


def residual_width(a0: np.ndarray, gt: np.ndarray) -> np.ndarray:
    r = np.linalg.norm(a0 - gt[:, None, :], axis=-1)
    return np.quantile(r, 0.9, axis=1) - np.quantile(r, 0.1, axis=1)


def sigma_l2(a0: np.ndarray) -> np.ndarray:
    return np.linalg.norm(a0.std(axis=1), axis=-1)


def score_episode(
    progress: np.ndarray,
    w_s0: np.ndarray,
    w_b1: np.ndarray,
    early: tuple[float, float],
    mid: tuple[float, float],
) -> dict[str, float]:
    e = (progress >= early[0]) & (progress < early[1])
    m = (progress >= mid[0]) & (progress < mid[1])
    allm = progress >= early[0]
    if e.sum() < 2 or m.sum() < 2:
        return {"valid": 0.0}

    s0_e, s0_m = float(w_s0[e].mean()), float(w_s0[m].mean())
    b1_e, b1_m = float(w_b1[e].mean()), float(w_b1[m].mean())
    s0_cv_mid = float(w_s0[m].std() / max(s0_m, 1e-9))
    b1_cv_mid = float(w_b1[m].std() / max(b1_m, 1e-9))
    s0_cv_all = float(w_s0[allm].std() / max(w_s0[allm].mean(), 1e-9))
    b1_narrow = b1_e / max(b1_m, 1e-9)  # >1 => mid narrower than early
    s0_level = s0_e / max(s0_m, 1e-9)  # ~1 => S0 early≈mid (stable level)
    # Desire: B1_narrow high, S0_cv low, S0_level near 1, and B1 actually narrower mid than S0 mid
    abs_gap = s0_m / max(b1_m, 1e-9)
    # penalize if B1 does not narrow
    narrow_term = max(b1_narrow - 1.0, 0.0)
    stability_term = 1.0 / max(s0_cv_mid, 0.08)
    level_term = 1.0 / max(abs(np.log(s0_level + 1e-9)), 0.15)
    score = narrow_term * stability_term * level_term * np.log1p(abs_gap)
    return {
        "valid": 1.0,
        "S0_early": s0_e,
        "S0_mid": s0_m,
        "S0_cv_mid": s0_cv_mid,
        "S0_cv_all": s0_cv_all,
        "S0_early_over_mid": s0_level,
        "B1_early": b1_e,
        "B1_mid": b1_m,
        "B1_cv_mid": b1_cv_mid,
        "B1_early_over_mid": b1_narrow,
        "S0_mid_over_B1_mid": abs_gap,
        "score": float(score),
    }


def main() -> None:
    import torch

    args = parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    early = parse_range(args.early)
    mid = parse_range(args.mid)
    episodes = list_episodes(args.expert_dataset, args.episodes, args.max_episodes)
    print(f"[screen] n_episodes={len(episodes)} stride={args.stride} K={args.num_samples} device={args.device}", flush=True)

    # Load both models once
    models = {}
    for name, run_dir, ckpt, prompt in [
        ("S0", args.s0_run_dir, args.s0_checkpoint, S0_TASK_PROMPT),
        ("B1", args.b1_run_dir, args.b1_checkpoint, B1_TASK_PROMPT),
    ]:
        print(f"[load] {name}", flush=True)
        model, processor, _ = _load_model(
            run_dir=run_dir,
            checkpoint=ckpt,
            norm_stats_meta_dir=args.norm_stats_meta_dir,
            device=args.device,
        )
        ctx, mask = _load_text_context(
            args.text_embedding_cache_dir,
            _instruction(prompt),
            args.device,
            model.torch_dtype,
        )
        models[name] = (model, processor, ctx, mask)

    rows: list[dict[str, Any]] = []
    for j, ep in enumerate(episodes):
        # probe length cheaply
        import pyarrow.parquet as pq

        n = len(pq.read_table(args.expert_dataset / "data" / "chunk-000" / f"episode_{ep:06d}.parquet", columns=["frame_index"]))
        frame_idx = np.arange(0, n, max(args.stride, 1), dtype=np.int64)
        if frame_idx[-1] != n - 1:
            frame_idx = np.unique(np.concatenate([frame_idx, [n - 1]]))
        progress = frame_idx.astype(np.float64) / max(n - 1, 1)
        print(f"[ep {j+1}/{len(episodes)}] expert {ep} n={n} queries={len(frame_idx)}", flush=True)
        try:
            images, proprios, gt, frame_idx, n = load_episode_arrays(args.expert_dataset, ep, frame_idx)
        except Exception as exc:  # noqa: BLE001
            print(f"  skip load: {exc}", flush=True)
            continue
        progress = frame_idx.astype(np.float64) / max(n - 1, 1)

        widths = {}
        for name in ("S0", "B1"):
            model, processor, ctx, mask = models[name]
            a0 = infer_a0(
                model=model,
                processor=processor,
                images=images,
                proprios=proprios,
                context=ctx,
                context_mask=mask,
                num_samples=args.num_samples,
                sample_seed0=args.sample_seed0,
                action_horizon=args.action_horizon,
                num_inference_steps=args.num_inference_steps,
                device=args.device,
                ep_tag=ep,
            )
            widths[f"{name}_residual"] = residual_width(a0, gt)
            widths[f"{name}_sigma"] = sigma_l2(a0)

        # Primary story metric = predicted-action interval width ||σ(a0)||_2
        metrics = score_episode(progress, widths["S0_sigma"], widths["B1_sigma"], early, mid)
        metrics_res = score_episode(progress, widths["S0_residual"], widths["B1_residual"], early, mid)
        row = {
            "episode": int(ep),
            "n_frames": int(n),
            "n_query": int(len(progress)),
            "width_metric": "sigma_l2",
            **metrics,
            "residual": {k: v for k, v in metrics_res.items() if k != "valid"},
        }
        rows.append(row)
        print(
            f"  score={row.get('score', 0):.3f}  "
            f"B1 early/mid={row.get('B1_early_over_mid', 0):.2f}  "
            f"S0_cv_all={row.get('S0_cv_all', 0):.2f}  "
            f"S0_e/m={row.get('S0_early_over_mid', 0):.2f}",
            flush=True,
        )
        # incremental save
        def _write_rankings(cur: list[dict[str, Any]]) -> None:
            valid = [r for r in cur if r.get("valid", 0)]
            by_joint = sorted(valid, key=lambda x: -x["score"])
            by_b1 = sorted(valid, key=lambda x: -x["B1_early_over_mid"])
            by_s0 = sorted(
                valid,
                key=lambda x: (x["S0_cv_all"], abs(float(np.log(max(x["S0_early_over_mid"], 1e-9))))),
            )
            payload = {
                "early": early,
                "mid": mid,
                "width_metric": "sigma_l2",
                "rows_by_joint_score": by_joint,
                "rows_by_b1_early_over_mid": by_b1,
                "rows_by_s0_stability": by_s0,
            }
            (out / "screen_ranking.json").write_text(json.dumps(payload, indent=2) + "\n")

        _write_rankings(rows)
        np.savez_compressed(
            out / f"ep{ep:03d}_widths.npz",
            progress=progress,
            S0_residual_width=widths["S0_residual"],
            B1_residual_width=widths["B1_residual"],
            S0_sigma_l2=widths["S0_sigma"],
            B1_sigma_l2=widths["B1_sigma"],
        )

    valid = [r for r in rows if r.get("valid", 0)]
    by_joint = sorted(valid, key=lambda x: -x["score"])
    by_b1 = sorted(valid, key=lambda x: -x["B1_early_over_mid"])
    by_s0 = sorted(
        valid,
        key=lambda x: (x["S0_cv_all"], abs(float(np.log(max(x["S0_early_over_mid"], 1e-9))))),
    )
    # Separate story picks (may be two different episodes)
    b1_ok = [r for r in by_b1 if r["B1_early_over_mid"] >= 1.20 and r["S0_mid_over_B1_mid"] >= 1.5]
    s0_ok = [
        r
        for r in by_s0
        if r["S0_cv_all"] <= 0.30 and 0.75 <= r["S0_early_over_mid"] <= 1.30 and r["S0_mid"] >= 0.15
    ]
    both = [
        r
        for r in by_joint
        if r["B1_early_over_mid"] >= 1.20
        and r["S0_cv_all"] <= 0.35
        and 0.75 <= r["S0_early_over_mid"] <= 1.30
        and r["S0_mid_over_B1_mid"] >= 2.0
    ]
    report = {
        "protocol": {
            "dataset": str(args.expert_dataset),
            "stride": args.stride,
            "num_samples": args.num_samples,
            "num_inference_steps": args.num_inference_steps,
            "early": early,
            "mid": mid,
            "width": "||std_k(a0)||_2 over K seeds (sigma_l2); residual also stored",
        },
        "n_scored": len(valid),
        "best_joint": by_joint[0] if by_joint else None,
        "best_b1_front_wide_mid_narrow": b1_ok[0] if b1_ok else (by_b1[0] if by_b1 else None),
        "best_s0_stable": s0_ok[0] if s0_ok else (by_s0[0] if by_s0 else None),
        "both_on_same_episode": both[:5],
        "top10_b1_narrow": by_b1[:10],
        "top10_s0_stable": by_s0[:10],
        "top10_joint": by_joint[:10],
        "n_b1_ok": len(b1_ok),
        "n_s0_ok": len(s0_ok),
        "n_both": len(both),
    }
    (out / "screen_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "best_b1": report["best_b1_front_wide_mid_narrow"],
                "best_s0": report["best_s0_stable"],
                "best_joint": report["best_joint"],
                "n_b1_ok": len(b1_ok),
                "n_s0_ok": len(s0_ok),
                "n_both": len(both),
            },
            indent=2,
        )
    )
    print(f"[done] {out}", flush=True)


if __name__ == "__main__":
    main()
