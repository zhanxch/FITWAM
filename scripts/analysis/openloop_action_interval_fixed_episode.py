#!/usr/bin/env python3
"""Open-loop multi-sample action interval on one fixed S0 success eval episode.

Protocol (discussion-aligned)
-----------------------------
1. Take the shortest successful S0 official-eval episode (seed=9, 217 steps).
2. Replay that episode's executed actions in DexJoCo to recover policy observations
   {input_image, proprio} at each timestep (same scene layout / trajectory).
3. For each sampled frame, run infer_action K times with distinct seeds on both
   S0 and B1-remap-cfg (same observations).
4. Plot predicted action-interval width over episode progress (S0 vs B1 only).
   No B/C/D panels; no closed-loop intervention.

Example:
  # Phase 1 — dexjoco env (CPU/sim)
  conda activate dexjoco
  python scripts/analysis/openloop_action_interval_fixed_episode.py --phase replay

  # Phase 2+3 — fastwam/web + GPU
  conda activate fastwam
  CUDA_VISIBLE_DEVICES=1 python scripts/analysis/openloop_action_interval_fixed_episode.py \\
    --phase infer_plot --device cuda:0 --num-samples 8 --stride 5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ACTIONS = (
    ROOT
    / "evaluate_results/dexjoco/official_4x50_remaining_skip_early_20260807_122616"
    / "S0/step_006500/run1/shard_0/water_plant/episode_09_success_actions.npz"
)
DEFAULT_TASK_YAML = ROOT / "third_party/dexjoco/configs/multi_task/water_plant.yaml"
DEFAULT_S0_RUN = ROOT / "runs/water_plant_uncond_2cam_384_1e-4/2026-06-29_16-38-39"
DEFAULT_S0_CKPT = DEFAULT_S0_RUN / "checkpoints/weights/step_006500.pt"
DEFAULT_B1_RUN = (
    ROOT / "runs/dexjoco_water_plant_offline_self_improving/2026-08-06_22-08-51_B1-remap-cfg"
)
DEFAULT_B1_CKPT = DEFAULT_B1_RUN / "checkpoints/weights/step_006500.pt"
DEFAULT_NORM = ROOT / "data/water_plant_fastwam/meta"
DEFAULT_TEXT = ROOT / "data/text_embeds_cache/water_plant"
DEFAULT_OUT = ROOT / "results/openloop_action_interval_s0ep09_20260808"
DEFAULT_EXPERT_DS = ROOT / "data/water_plant_fastwam"
DEFAULT_EXPERT_OUT = ROOT / "results/openloop_action_interval_expert_ep22_20260808"

S0_TASK_PROMPT = "Grasp the watering can and apply water to the plant."
B1_TASK_PROMPT = "Grasp the watering can and apply water to the plant. Successful execution."


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--phase",
        choices=["replay", "prepare_expert", "infer", "plot", "infer_plot", "all"],
        default="infer_plot",
    )
    p.add_argument("--actions-npz", type=Path, default=DEFAULT_ACTIONS)
    p.add_argument("--seed", type=int, default=9)
    p.add_argument("--expert-dataset", type=Path, default=DEFAULT_EXPERT_DS)
    p.add_argument(
        "--expert-episode",
        type=int,
        default=22,
        help="Expert episode index (default: shortest water_plant_fastwam ep 22, len=203).",
    )
    p.add_argument("--task-yaml", type=Path, default=DEFAULT_TASK_YAML)
    p.add_argument("--s0-run-dir", type=Path, default=DEFAULT_S0_RUN)
    p.add_argument("--s0-checkpoint", type=Path, default=DEFAULT_S0_CKPT)
    p.add_argument("--b1-run-dir", type=Path, default=DEFAULT_B1_RUN)
    p.add_argument("--b1-checkpoint", type=Path, default=DEFAULT_B1_CKPT)
    p.add_argument("--norm-stats-meta-dir", type=Path, default=DEFAULT_NORM)
    p.add_argument("--text-embedding-cache-dir", type=Path, default=DEFAULT_TEXT)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--num-samples", type=int, default=8)
    p.add_argument("--stride", type=int, default=5)
    p.add_argument("--action-horizon", type=int, default=32)
    p.add_argument("--num-inference-steps", type=int, default=10)
    p.add_argument("--sample-seed0", type=int, default=20260808)
    p.add_argument("--dpi", type=int, default=300)
    args = p.parse_args()
    if args.output is None:
        # Expert pipeline uses a separate results dir; S0-eval replay keeps the old default.
        args.output = (
            DEFAULT_EXPERT_OUT
            if args.phase in {"prepare_expert"}
            else DEFAULT_OUT
        )
    return args


def json_dump(obj: Any, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True) + "\n")


def _load_source_label(out: Path) -> str:
    meta_path = out / "replay_meta.json"
    if not meta_path.exists():
        return "trajectory"
    meta = json.loads(meta_path.read_text())
    return str(meta.get("source_label") or meta.get("source") or "trajectory")


def _read_video_rgb(path: Path) -> np.ndarray:
    import av

    container = av.open(str(path))
    frames: list[np.ndarray] = []
    for frame in container.decode(video=0):
        frames.append(frame.to_ndarray(format="rgb24"))
    if not frames:
        raise RuntimeError(f"Empty video: {path}")
    return np.stack(frames, axis=0)


def _hwc_to_model_image(front: np.ndarray, wrist: np.ndarray) -> np.ndarray:
    """Match S0 2cam horizontal concat: each cam 384x384 → [1,3,384,768] in [-1,1]."""
    from PIL import Image

    f = Image.fromarray(front, mode="RGB").resize((384, 384), Image.Resampling.BILINEAR)
    w = Image.fromarray(wrist, mode="RGB").resize((384, 384), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (768, 384))
    canvas.paste(f, (0, 0))
    canvas.paste(w, (384, 0))
    arr = np.asarray(canvas, dtype=np.float32) / 127.5 - 1.0
    return np.transpose(arr, (2, 0, 1))[None, ...].astype(np.float32)


def phase_prepare_expert(args: argparse.Namespace) -> Path:
    import pyarrow.parquet as pq

    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    ds = args.expert_dataset
    ep = int(args.expert_episode)
    parquet = ds / "data" / "chunk-000" / f"episode_{ep:06d}.parquet"
    front_v = ds / "videos" / "chunk-000" / "observation.images.front" / f"episode_{ep:06d}.mp4"
    wrist_v = ds / "videos" / "chunk-000" / "observation.images.wrist" / f"episode_{ep:06d}.mp4"
    table = pq.read_table(parquet, columns=["action", "observation.state", "frame_index"])
    actions = np.asarray(table.column("action").to_pylist(), dtype=np.float32)
    states = np.asarray(table.column("observation.state").to_pylist(), dtype=np.float32)
    if actions.shape[1] == 23:
        # soft-event datasets prepend score
        actions = actions[:, 1:]
    front = _read_video_rgb(front_v)
    wrist = _read_video_rgb(wrist_v)
    n = min(len(actions), len(states), len(front), len(wrist))
    print(
        f"[expert] ep={ep} n={n} action={actions.shape} state={states.shape} "
        f"front={front.shape} wrist={wrist.shape}",
        flush=True,
    )
    images = np.stack([_hwc_to_model_image(front[t], wrist[t]) for t in range(n)], axis=0)
    # images currently [T,1,3,H,W] if _hwc returns with batch — squeeze
    if images.ndim == 5 and images.shape[1] == 1:
        images = images[:, 0]
    # ensure [T,1,3,H,W] for infer loader compatibility with replay format
    if images.ndim == 4:
        images = images[:, None, ...]
    proprios = states[:n].astype(np.float32)
    executed = actions[:n].astype(np.float32)

    obs_path = out / "replay_obs.npz"
    np.savez_compressed(
        obs_path,
        input_image=images,
        proprio=proprios,
        executed_actions=executed,
        seed=np.asarray(int(ep)),
        initial_state_mae=np.asarray(0.0),
    )
    meta = {
        "source": "expert",
        "source_label": f"expert ep{ep} (water_plant_fastwam)",
        "expert_dataset": str(ds),
        "expert_episode": ep,
        "n_frames": int(n),
        "image_shape": list(images.shape),
        "proprio_dim": int(proprios.shape[-1]),
        "action_dim": int(executed.shape[-1]),
        "gt_name": "expert demonstration action",
    }
    json_dump(meta, out / "replay_meta.json")
    print(f"[expert] wrote {obs_path} frames={n} image={images.shape}", flush=True)
    return obs_path


def phase_replay(args: argparse.Namespace) -> Path:
    sys.path.insert(0, str(ROOT / "scripts" / "dexjoco_async"))
    sys.path.insert(0, str(ROOT / "third_party" / "dexjoco" / "dexjoco"))
    from dexjoco_fastwam_adapter import (  # type: ignore
        DexJoCoFastWAMAdapter,
        DexJoCoFastWAMEvalEnv,
        DexJoCoTaskConfig,
        load_dexjoco_eval_settings,
    )

    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    z = np.load(args.actions_npz)
    executed = np.asarray(z["executed_actions"], dtype=np.float32)
    initial = np.asarray(z["initial_state"], dtype=np.float32).reshape(-1)

    eval_settings = load_dexjoco_eval_settings(
        args.s0_run_dir,
        action_horizon_override=args.action_horizon,
        text_embedding_cache_dir_override=args.text_embedding_cache_dir,
    )
    adapter = DexJoCoFastWAMAdapter(eval_settings)
    import yaml

    with args.task_yaml.open("r", encoding="utf-8") as f:
        task_cfg = yaml.safe_load(f)
    task = DexJoCoTaskConfig.from_yaml(task_cfg)
    env = DexJoCoFastWAMEvalEnv(task, seed=int(args.seed), randomize=False, randomize_dynamics=False)
    try:
        obs0 = env.reset()
        state0 = np.asarray(obs0["state"], dtype=np.float32).reshape(-1)
        # Sanity: initial proprio should match logged initial_state (up to shared prefix).
        n = min(len(state0), len(initial))
        mae = float(np.mean(np.abs(state0[:n] - initial[:n])))
        print(f"[replay] seed={args.seed} steps={len(executed)} initial_state mae={mae:.6g}", flush=True)

        images: list[np.ndarray] = []
        proprios: list[np.ndarray] = []
        for t in range(len(executed)):
            policy_obs = env.build_policy_obs(adapter)
            images.append(np.asarray(policy_obs["input_image"], dtype=np.float32).copy())
            proprios.append(np.asarray(policy_obs["proprio"], dtype=np.float32).copy())
            env.step_rotvec(executed[t])
            if env.is_done and t + 1 < len(executed):
                print(f"[replay] WARN env done early at t={t}", flush=True)
                break
        images_arr = np.stack(images, axis=0)
        proprios_arr = np.stack(proprios, axis=0)
    finally:
        env.close()

    obs_path = out / "replay_obs.npz"
    np.savez_compressed(
        obs_path,
        input_image=images_arr,
        proprio=proprios_arr,
        executed_actions=executed[: len(images)],
        seed=np.asarray(int(args.seed)),
        initial_state_mae=np.asarray(mae),
    )
    meta = {
        "source": "s0_eval_success",
        "source_label": f"S0 success eval seed={int(args.seed)}",
        "actions_npz": str(args.actions_npz),
        "seed": int(args.seed),
        "n_frames": int(len(images)),
        "image_shape": list(images_arr.shape),
        "proprio_dim": int(proprios_arr.shape[-1]),
        "initial_state_mae": mae,
        "task_yaml": str(args.task_yaml),
        "s0_run_dir": str(args.s0_run_dir),
        "gt_name": "S0 executed action",
    }
    json_dump(meta, out / "replay_meta.json")
    print(f"[replay] wrote {obs_path} frames={len(images)} image={images_arr.shape}", flush=True)
    return obs_path


def _load_model(
    *,
    run_dir: Path,
    checkpoint: Path,
    norm_stats_meta_dir: Path,
    device: str,
):
    import torch
    from omegaconf import OmegaConf
    from hydra.utils import instantiate

    src = str(ROOT / "src")
    if src not in sys.path:
        sys.path.insert(0, src)

    cfg = OmegaConf.load(run_dir / "config.yaml")
    model_cfg = OmegaConf.create(OmegaConf.to_container(cfg.model, resolve=True))
    model_cfg.load_text_encoder = False
    model_dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
    processor_cfg = OmegaConf.create(OmegaConf.to_container(cfg.data.train.processor, resolve=True))
    processor_cfg.norm_stats_source = "meta"
    processor_cfg.norm_stats_meta_dir = str(norm_stats_meta_dir.resolve())
    print(f"[infer] load {checkpoint.name} from {run_dir.name} on {device}", flush=True)
    model = instantiate(model_cfg, model_dtype=model_dtype, device=device)
    model.load_checkpoint(str(checkpoint))
    model.eval()
    processor = instantiate(processor_cfg)
    processor.eval()
    processor.set_normalizer_from_modality_stats()
    return model, processor, cfg


def _load_text_context(cache_dir: Path, instruction: str, device: str, dtype):
    import hashlib
    import torch

    hashed = hashlib.sha256(instruction.encode("utf-8")).hexdigest()
    pt = cache_dir / f"{hashed}.t5_len128.wan22ti2v5b.pt"
    if not pt.exists():
        raise FileNotFoundError(f"Missing text cache: {pt}")
    payload = torch.load(pt, map_location="cpu", weights_only=False)
    context = payload["context"].unsqueeze(0).to(device=device, dtype=dtype)
    mask = payload["mask"].unsqueeze(0).to(device=device, dtype=torch.bool)
    return context, mask


def _normalize_proprio(processor, proprio: np.ndarray, device: str, dtype):
    import torch

    tensor = torch.from_numpy(np.asarray(proprio, dtype=np.float32)).float().view(1, -1)
    out = processor.normalizer.forward({"state": {"default": tensor}})
    return out["state"]["default"].to(device=device, dtype=dtype)


def _denormalize_action(processor, action_norm, last_proprio_norm=None) -> np.ndarray:
    import torch

    # action_norm: [H,D] or [1,H,D]
    if action_norm.ndim == 3:
        action_norm = action_norm[0]
    action_key = "default"
    # Match server: normalizers["action"][key].backward
    normalizer = processor.normalizer.normalizers["action"][action_key]
    denorm = normalizer.backward(action_norm.detach().to(dtype=torch.float32, device="cpu"))
    return np.asarray(denorm.numpy(), dtype=np.float32)


def _instruction(task_prompt: str) -> str:
    return (
        "A video recorded from a robot's point of view executing the following instruction: "
        f"{task_prompt}"
    )


def _sample_actions_for_frames(
    *,
    model,
    processor,
    images: np.ndarray,
    proprios: np.ndarray,
    frame_idx: np.ndarray,
    context,
    context_mask,
    num_samples: int,
    sample_seed0: int,
    action_horizon: int,
    num_inference_steps: int,
    device: str,
) -> np.ndarray:
    """Return actions [T,K,H,D] in denormalized env space (first control dims)."""
    import torch

    T = len(frame_idx)
    outs: list[np.ndarray] = []
    for i, t in enumerate(frame_idx):
        img = torch.from_numpy(images[t]).to(device=device, dtype=model.torch_dtype)
        if img.ndim == 3:
            img = img.unsqueeze(0)
        prop = _normalize_proprio(processor, proprios[t], device, model.torch_dtype)
        sample_rows = []
        for k in range(num_samples):
            seed = int(sample_seed0 + 10007 * k + t)
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
            # control dims: if model outputs more than proprio action, keep first 22
            if act.shape[-1] > 22:
                act = act[..., -22:] if act.shape[-1] == 23 else act[..., :22]
            sample_rows.append(act.astype(np.float32))
        outs.append(np.stack(sample_rows, axis=0))  # [K,H,D]
        if (i + 1) % 5 == 0 or i == 0 or i + 1 == T:
            print(f"[infer] frame {i+1}/{T} t={int(t)}", flush=True)
    return np.stack(outs, axis=0)  # [T,K,H,D]


def phase_infer(args: argparse.Namespace) -> Path:
    import torch

    out = args.output
    obs_path = out / "replay_obs.npz"
    if not obs_path.exists():
        raise FileNotFoundError(f"Missing {obs_path}; run --phase replay first")
    pack = np.load(obs_path)
    images = np.asarray(pack["input_image"], dtype=np.float32)
    proprios = np.asarray(pack["proprio"], dtype=np.float32)
    executed = np.asarray(pack["executed_actions"], dtype=np.float32)
    n = len(images)
    frame_idx = np.arange(0, n, max(int(args.stride), 1), dtype=np.int64)
    if frame_idx[-1] != n - 1:
        frame_idx = np.unique(np.concatenate([frame_idx, [n - 1]]))

    results: dict[str, Any] = {
        "frame_idx": frame_idx,
        "progress": frame_idx.astype(np.float64) / max(n - 1, 1),
        "executed_at_frames": executed[frame_idx],
        "n_frames_full": n,
        "num_samples": int(args.num_samples),
        "stride": int(args.stride),
        "num_inference_steps": int(args.num_inference_steps),
        "sample_seed0": int(args.sample_seed0),
    }

    for name, run_dir, ckpt, task_prompt in [
        ("S0", args.s0_run_dir, args.s0_checkpoint, S0_TASK_PROMPT),
        ("B1", args.b1_run_dir, args.b1_checkpoint, B1_TASK_PROMPT),
    ]:
        model, processor, _cfg = _load_model(
            run_dir=run_dir,
            checkpoint=ckpt,
            norm_stats_meta_dir=args.norm_stats_meta_dir,
            device=args.device,
        )
        context, context_mask = _load_text_context(
            args.text_embedding_cache_dir,
            _instruction(task_prompt),
            args.device,
            model.torch_dtype,
        )
        actions = _sample_actions_for_frames(
            model=model,
            processor=processor,
            images=images,
            proprios=proprios,
            frame_idx=frame_idx,
            context=context,
            context_mask=context_mask,
            num_samples=args.num_samples,
            sample_seed0=args.sample_seed0,
            action_horizon=args.action_horizon,
            num_inference_steps=args.num_inference_steps,
            device=args.device,
        )
        # Interval from first-step action a0 across K samples
        a0 = actions[:, :, 0, :]  # [T,K,D]
        mu = a0.mean(axis=1)
        std = a0.std(axis=1)
        sigma_l2 = np.linalg.norm(std, axis=-1)
        # Also radius of sample cloud: mean L2 distance to mean
        radius = np.mean(np.linalg.norm(a0 - mu[:, None, :], axis=-1), axis=1)
        p10 = np.quantile(np.linalg.norm(a0 - mu[:, None, :], axis=-1), 0.10, axis=1)
        p90 = np.quantile(np.linalg.norm(a0 - mu[:, None, :], axis=-1), 0.90, axis=1)
        results[f"{name}_actions_a0"] = a0.astype(np.float32)
        results[f"{name}_mu_a0"] = mu.astype(np.float32)
        results[f"{name}_std_a0"] = std.astype(np.float32)
        results[f"{name}_sigma_l2"] = sigma_l2.astype(np.float32)
        results[f"{name}_radius_mean"] = radius.astype(np.float32)
        results[f"{name}_radius_p10"] = p10.astype(np.float32)
        results[f"{name}_radius_p90"] = p90.astype(np.float32)
        results[f"{name}_task_prompt"] = np.asarray(task_prompt)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Soft-event criticality on executed trajectory (for interaction highlight)
    try:
        sys.path.insert(0, str(ROOT / "scripts" / "analysis"))
        from build_interaction_sensitivity_evidence import (  # type: ignore
            SoftEventScorer,
            load_soft_event_module,
        )

        soft_mod = load_soft_event_module(ROOT / "src/fastwam/__pycache__/soft_event.cpython-310.pyc")
        scorer = SoftEventScorer(
            ROOT / "data/water_plant_soft_event_v1/meta/eve/soft_event_motif_model.json",
            soft_mod,
        )
        score = scorer.score(executed)["score"]
        results["soft_event_full"] = score
        results["soft_event_at_frames"] = score[frame_idx]
        peak = int(np.nanargmax(score))
        results["interaction_peak_progress"] = float(peak / max(n - 1, 1))
        half = 0.12
        results["interaction_onset"] = float(max(0.0, results["interaction_peak_progress"] - half))
        results["interaction_offset"] = float(min(1.0, results["interaction_peak_progress"] + half))
    except Exception as exc:  # noqa: BLE001
        print(f"[infer] soft-event skipped: {exc}", flush=True)

    out_path = out / "openloop_interval.npz"
    # numpy can't save str arrays mixed easily — drop prompt strings
    save = {k: v for k, v in results.items() if not isinstance(v, str)}
    np.savez_compressed(out_path, **save)
    replay_meta = {}
    meta_path = out / "replay_meta.json"
    if meta_path.exists():
        replay_meta = json.loads(meta_path.read_text())
    episode_info: dict[str, Any] = {
        "source": replay_meta.get("source"),
        "source_label": replay_meta.get("source_label"),
        "n_frames": int(n),
        "n_query_frames": int(len(frame_idx)),
        "gt_name": replay_meta.get("gt_name"),
    }
    if replay_meta.get("source") == "expert":
        episode_info["expert_episode"] = replay_meta.get("expert_episode")
        episode_info["expert_dataset"] = replay_meta.get("expert_dataset")
    else:
        episode_info["seed"] = int(args.seed)
        episode_info["actions_npz"] = str(args.actions_npz)
    report = {
        "episode": episode_info,
        "infer": {
            "num_samples": int(args.num_samples),
            "stride": int(args.stride),
            "num_inference_steps": int(args.num_inference_steps),
            "sample_seed0": int(args.sample_seed0),
            "device": args.device,
            "s0_checkpoint": str(args.s0_checkpoint),
            "b1_checkpoint": str(args.b1_checkpoint),
            "interval_definition": (
                "At each query frame t, K=num_samples infer_action seeds; "
                "interval width = L2(std_k(a0)); band = p10/p90 of ||a0-μ||"
            ),
        },
        "summary": {
            "S0_sigma_l2_mean": float(np.mean(results["S0_sigma_l2"])),
            "B1_sigma_l2_mean": float(np.mean(results["B1_sigma_l2"])),
            "S0_sigma_l2_at_peak_bin": None,
            "B1_sigma_l2_at_peak_bin": None,
        },
    }
    if "interaction_peak_progress" in results:
        prog = results["progress"]
        peak_p = float(results["interaction_peak_progress"])
        j = int(np.argmin(np.abs(prog - peak_p)))
        report["summary"]["S0_sigma_l2_at_peak_bin"] = float(results["S0_sigma_l2"][j])
        report["summary"]["B1_sigma_l2_at_peak_bin"] = float(results["B1_sigma_l2"][j])
        report["interaction_peak_progress"] = peak_p
        report["interaction_onset"] = float(results["interaction_onset"])
        report["interaction_offset"] = float(results["interaction_offset"])
    json_dump(report, out / "openloop_report.json")
    print(json.dumps(report["summary"], indent=2), flush=True)
    print(f"[infer] wrote {out_path}", flush=True)
    return out_path


def phase_plot(args: argparse.Namespace) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = args.output
    z = np.load(out / "openloop_interval.npz")
    report = json.loads((out / "openloop_report.json").read_text())
    progress = z["progress"]
    c_s0, c_b1 = "#4C78A8", "#54A24B"
    source_label = _load_source_label(out)

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4), dpi=args.dpi, sharey=True)
    fig.suptitle(
        "Open-loop predicted action interval on a fixed episode\n"
        f"{source_label}  ·  {int(z['n_frames_full'])} steps  ·  "
        f"K={int(z['num_samples'])} samples/frame  ·  stride={int(z['stride'])}",
        fontsize=10,
        fontweight="semibold",
        y=1.02,
    )
    for ax, name, color in [(axes[0], "S0", c_s0), (axes[1], "B1", c_b1)]:
        title = "Expert-only Baseline (S0)" if name == "S0" else "Rollout-Retrained (B1-remap-cfg)"
        lo = z[f"{name}_radius_p10"]
        hi = z[f"{name}_radius_p90"]
        mid = z[f"{name}_sigma_l2"]
        ax.fill_between(progress, lo, hi, color=color, alpha=0.25, linewidth=0, label=r"sample radius [p10,p90]")
        ax.plot(progress, mid, color=color, lw=1.8, label=r"$\|\sigma(a_0)\|_2$")
        ax.legend(fontsize=6.5, frameon=False, loc="upper right")
        ax.set_title(title, fontsize=9, color=color, loc="left", fontweight="semibold")
        ax.set_xlabel("episode progress", fontsize=8)
        ax.set_xlim(0, 1)
        ax.grid(True, axis="y", color="#E5E7EB", lw=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel("predicted action interval", fontsize=8)
    ymax = max(
        float(np.nanmax(z["S0_radius_p90"])),
        float(np.nanmax(z["B1_radius_p90"])),
        float(np.nanmax(z["S0_sigma_l2"])),
        float(np.nanmax(z["B1_sigma_l2"])),
    )
    for ax in axes:
        ax.set_ylim(0, ymax * 1.15)

    fig.text(
        0.5,
        -0.08,
        f"Observations from {source_label}. "
        "Interval = multi-seed open-loop infer_action at each frame (denormalized a0).",
        ha="center",
        fontsize=7,
        color="#4B5563",
    )
    stem = out / "fig_openloop_action_interval_S0_vs_B1"
    for ext in ("png", "pdf"):
        path = Path(str(stem) + f".{ext}")
        fig.savefig(path, dpi=args.dpi, bbox_inches="tight", facecolor="white")
        print("wrote", path, flush=True)
    plt.close(fig)
    report["figure"] = str(stem) + ".png"
    json_dump(report, out / "openloop_report.json")

    # Merged figure: GT executed action + S0/B1 sample intervals
    phase_plot_merged(args, z, report)
    return Path(str(stem) + ".png")


def phase_plot_merged(args: argparse.Namespace, z: Any, report: dict[str, Any]) -> Path:
    """One merged panel: GT action vs S0/B1 open-loop sample intervals."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = args.output
    progress = np.asarray(z["progress"], dtype=np.float64)
    gt = np.asarray(z["executed_at_frames"], dtype=np.float64)  # [T,D]
    gt_norm = np.linalg.norm(gt, axis=-1)
    source_label = _load_source_label(out)
    meta = json.loads((out / "replay_meta.json").read_text()) if (out / "replay_meta.json").exists() else {}
    gt_name = str(meta.get("gt_name") or "GT action")

    def sample_norm_stats(actions_a0: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # actions_a0: [T,K,D] -> norms [T,K]
        norms = np.linalg.norm(actions_a0, axis=-1)
        return (
            np.quantile(norms, 0.10, axis=1),
            np.median(norms, axis=1),
            np.quantile(norms, 0.90, axis=1),
        )

    def residual_stats(actions_a0: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # ||a_k - a_gt|| over K samples
        res = np.linalg.norm(actions_a0 - gt[:, None, :], axis=-1)
        return (
            np.quantile(res, 0.10, axis=1),
            np.median(res, axis=1),
            np.quantile(res, 0.90, axis=1),
        )

    s0_lo, s0_mid, s0_hi = sample_norm_stats(np.asarray(z["S0_actions_a0"], dtype=np.float64))
    b1_lo, b1_mid, b1_hi = sample_norm_stats(np.asarray(z["B1_actions_a0"], dtype=np.float64))
    s0_rlo, s0_rmid, s0_rhi = residual_stats(np.asarray(z["S0_actions_a0"], dtype=np.float64))
    b1_rlo, b1_rmid, b1_rhi = residual_stats(np.asarray(z["B1_actions_a0"], dtype=np.float64))

    c_s0, c_b1, c_gt = "#4C78A8", "#54A24B", "#111827"

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(8.2, 5.2),
        dpi=args.dpi,
        sharex=True,
        gridspec_kw={"hspace": 0.18},
    )
    fig.suptitle(
        f"GT action vs open-loop sample intervals\n"
        f"{source_label}  ·  K={int(z['num_samples'])}  ·  stride={int(z['stride'])}",
        fontsize=10,
        fontweight="semibold",
        y=0.98,
    )

    ax = axes[0]
    ax.plot(progress, gt_norm, color=c_gt, lw=1.8, label=rf"GT $\|a\|_2$ ({gt_name})", zorder=3)
    ax.fill_between(progress, s0_lo, s0_hi, color=c_s0, alpha=0.22, linewidth=0, label="S0 sample [p10,p90]")
    ax.plot(progress, s0_mid, color=c_s0, lw=1.4, label="S0 sample median")
    ax.fill_between(progress, b1_lo, b1_hi, color=c_b1, alpha=0.22, linewidth=0, label="B1 sample [p10,p90]")
    ax.plot(progress, b1_mid, color=c_b1, lw=1.4, label="B1 sample median")
    ax.set_ylabel(r"$\|a\|_2$", fontsize=8)
    ax.set_title("A  Action magnitude", fontsize=9, loc="left", fontweight="semibold")
    ax.legend(fontsize=6.5, frameon=False, ncol=2, loc="upper right")
    ax.grid(True, axis="y", color="#E5E7EB", lw=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=7)

    ax = axes[1]
    ax.axhline(0.0, color=c_gt, lw=1.0, ls="--", label="GT (residual = 0)", zorder=2)
    ax.fill_between(progress, s0_rlo, s0_rhi, color=c_s0, alpha=0.22, linewidth=0, label="S0 $\|a-a^{\\mathrm{GT}}\|$ [p10,p90]")
    ax.plot(progress, s0_rmid, color=c_s0, lw=1.4, label="S0 residual median")
    ax.fill_between(progress, b1_rlo, b1_rhi, color=c_b1, alpha=0.22, linewidth=0, label="B1 $\|a-a^{\\mathrm{GT}}\|$ [p10,p90]")
    ax.plot(progress, b1_rmid, color=c_b1, lw=1.4, label="B1 residual median")
    ax.set_ylabel(r"$\|a-a^{\mathrm{GT}}\|_2$", fontsize=8)
    ax.set_xlabel("episode progress", fontsize=8)
    ax.set_title(f"B  Residual to GT ({gt_name})", fontsize=9, loc="left", fontweight="semibold")
    ax.legend(fontsize=6.5, frameon=False, ncol=2, loc="upper right")
    ax.grid(True, axis="y", color="#E5E7EB", lw=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=7)
    ax.set_xlim(0, 1)

    fig.text(
        0.5,
        0.01,
        f"Observations from {source_label}. Shaded bands = open-loop multi-seed sample intervals of denormalized a0.",
        ha="center",
        fontsize=7,
        color="#4B5563",
    )

    stem = out / "fig_openloop_gt_vs_sample_intervals"
    for ext in ("png", "pdf"):
        path = Path(str(stem) + f".{ext}")
        fig.savefig(path, dpi=args.dpi, bbox_inches="tight", facecolor="white")
        print("wrote", path, flush=True)
    plt.close(fig)
    report["figure_merged"] = str(stem) + ".png"
    report["merged_summary"] = {
        "S0_residual_median_mean": float(np.mean(s0_rmid)),
        "B1_residual_median_mean": float(np.mean(b1_rmid)),
        "S0_residual_p90_mean": float(np.mean(s0_rhi)),
        "B1_residual_p90_mean": float(np.mean(b1_rhi)),
        "gt_action_l2_mean": float(np.mean(gt_norm)),
    }
    json_dump(report, out / "openloop_report.json")
    return Path(str(stem) + ".png")


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.phase in {"replay", "all"}:
        phase_replay(args)
    if args.phase in {"prepare_expert"}:
        phase_prepare_expert(args)
    if args.phase in {"infer", "infer_plot", "all"}:
        phase_infer(args)
    if args.phase in {"plot", "infer_plot", "all"}:
        phase_plot(args)


if __name__ == "__main__":
    main()
