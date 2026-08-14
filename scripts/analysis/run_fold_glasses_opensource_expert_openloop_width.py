#!/usr/bin/env python3
"""fold_glasses S0 open-loop action-interval width on ALL expert episodes.

Aligns inference with the open-source stack that was already validated:
  FastWAM-infer-in-DexJoco FastWAMDexJocoPolicy
  + FastWAM pin 45d8e14
  + configs/fastwam_dexjoco.yaml  (224, z-score)
  + artifacts/fold_glasses/{dataset_stats.json, text embedding}
  + action_horizon=32, num_inference_steps=10, rand_device=cpu, tiled=False

Do NOT reuse water_plant openloop helpers (384 / min-max / current src / meta stats).

Protocol (plotting; same as water_plant expert width)
----------------------------------------------------
- Dataset: data/fold_glasses_fastwam (expert demos)
- K=20 multi-seed samples per query frame via policy.infer(..., noise_seed=...)
- stride=5, progress in [0,1]
- Width: ||std_k(a0)||_2 over denormalized first-step actions
- Plots: per-episode + collage with shared ymax

Launch (pack GPUs 4-7):
  bash scripts/analysis/run_fold_glasses_opensource_expert_openloop_width.sh
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OPEN = Path(os.environ.get("FASTWAM_OPEN_REPO", "/data_all/xiangchengzhan/FastWAM-infer-in-DexJoco"))
FASTWAM_PIN = Path(
    os.environ.get("FASTWAM_PIN", str(ROOT / "third_party/FastWAM_pin_45d8e14"))
)

DEFAULT_EXPERT_DS = ROOT / "data/fold_glasses_fastwam"
DEFAULT_OUT = ROOT / "results/fold_glasses_opensource_expert_openloop_width_K20_stride5_20260810"
DEFAULT_CKPT = OPEN / "checkpoints/fold_glasses/step_010000.pt"
DEFAULT_MODEL_CFG = OPEN / "configs/fastwam_dexjoco.yaml"
DEFAULT_STATS = OPEN / "artifacts/fold_glasses/dataset_stats.json"
DEFAULT_TEXT = (
    OPEN
    / "artifacts/fold_glasses"
    / "0c3367ce1d74848cc46b93c6d2eee5e2097dca410a2c95f3da48bd8c8673fa20.t5_len128.wan22ti2v5b.pt"
)
EXPECTED_PIN_HEAD = "45d8e1458921d83f8ad6cf9ce993d371208dabd0"

C_S0 = "#4C78A8"
C_MUTED = "#5B6B7A"
C_GRID = "#E6E9ED"


def ensure_opensource_pythonpath() -> None:
    """Prefer OPEN src + FastWAM pin over workspace src (must match eval_opensource)."""
    paths = [
        str(OPEN / "src"),
        str(FASTWAM_PIN / "src"),
        str(ROOT / "third_party" / "dexjoco" / "dexjoco"),
    ]
    # Prepend so they win over any ambient FastWAM/src.
    for p in reversed(paths):
        if p in sys.path:
            sys.path.remove(p)
        sys.path.insert(0, p)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["worker", "collage", "orchestrate"], required=True)
    p.add_argument("--expert-dataset", type=Path, default=DEFAULT_EXPERT_DS)
    p.add_argument("--output", type=Path, default=DEFAULT_OUT)
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    p.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CFG)
    p.add_argument("--dataset-stats", type=Path, default=DEFAULT_STATS)
    p.add_argument("--text-embedding", type=Path, default=DEFAULT_TEXT)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--episodes", type=str, default="", help="comma list for worker")
    p.add_argument("--worker-id", type=str, default="w0")
    p.add_argument("--num-samples", type=int, default=20)
    p.add_argument("--stride", type=int, default=5)
    p.add_argument("--num-inference-steps", type=int, default=10)
    p.add_argument("--action-horizon", type=int, default=32)
    p.add_argument("--replan-steps", type=int, default=24, help="unused for openloop; kept for OS parity")
    p.add_argument("--sample-seed0", type=int, default=20260808)
    p.add_argument("--gpus", type=str, default="4,5,6,7")
    p.add_argument("--workers-per-gpu", type=int, default=5)
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--collage-cols", type=int, default=4)
    p.add_argument("--ymax", type=float, default=0.0, help="0 = auto robust quantile")
    p.add_argument(
        "--ymax-quantile",
        type=float,
        default=0.95,
        help="shared ymax = quantile(all ||σ(a0)||_2) * pad",
    )
    p.add_argument("--skip-pin-check", action="store_true")
    return p.parse_args()


def assert_pin_head() -> None:
    head = subprocess.check_output(
        ["git", "-C", str(FASTWAM_PIN), "rev-parse", "HEAD"], text=True
    ).strip()
    if head != EXPECTED_PIN_HEAD:
        raise RuntimeError(
            f"FastWAM pin HEAD mismatch: got {head}, expected {EXPECTED_PIN_HEAD}. "
            "Refuse to run non-opensource-aligned code."
        )


def list_all_episodes(ds: Path) -> list[int]:
    return sorted(
        int(p.stem.split("_")[-1]) for p in (ds / "data" / "chunk-000").glob("episode_*.parquet")
    )


def parse_episode_list(spec: str) -> list[int]:
    return [int(x) for x in spec.split(",") if x.strip()]


def json_dump(obj: Any, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True) + "\n")


def _read_video_rgb(path: Path) -> np.ndarray:
    import av

    container = av.open(str(path))
    frames: list[np.ndarray] = []
    for frame in container.decode(video=0):
        frames.append(frame.to_ndarray(format="rgb24"))
    if not frames:
        raise RuntimeError(f"Empty video: {path}")
    return np.stack(frames, axis=0)


def load_episode_raw(ds: Path, ep: int, frame_idx: np.ndarray):
    """Load raw HWC uint8 cams + state + GT action (opensource policy does its own preprocess)."""
    import pyarrow.parquet as pq

    parquet = ds / "data" / "chunk-000" / f"episode_{ep:06d}.parquet"
    front_v = ds / "videos" / "chunk-000" / "observation.images.front" / f"episode_{ep:06d}.mp4"
    wrist_v = ds / "videos" / "chunk-000" / "observation.images.wrist" / f"episode_{ep:06d}.mp4"
    table = pq.read_table(parquet, columns=["action", "observation.state"])
    actions = np.asarray(table.column("action").to_pylist(), dtype=np.float32)
    states = np.asarray(table.column("observation.state").to_pylist(), dtype=np.float32)
    if actions.ndim != 2:
        raise ValueError(f"Unexpected action shape {actions.shape}")
    if actions.shape[1] == 23:
        actions = actions[:, 1:]
    front = _read_video_rgb(front_v)
    wrist = _read_video_rgb(wrist_v)
    n = min(len(actions), len(states), len(front), len(wrist))
    frame_idx = np.asarray(frame_idx[frame_idx < n], dtype=np.int64)
    return (
        front,
        wrist,
        states.astype(np.float32),
        actions.astype(np.float32),
        frame_idx,
        int(n),
    )


def load_policy(args: argparse.Namespace):
    ensure_opensource_pythonpath()
    from fastwam_dexjoco.policy import FastWAMDexJocoPolicy  # noqa: WPS451

    print(
        f"[policy] OPEN={OPEN} PIN={FASTWAM_PIN}\n"
        f"  cfg={args.model_config}\n"
        f"  ckpt={args.checkpoint}\n"
        f"  stats={args.dataset_stats}\n"
        f"  text={args.text_embedding}",
        flush=True,
    )
    return FastWAMDexJocoPolicy(
        model_config=args.model_config,
        checkpoint=args.checkpoint,
        dataset_stats=args.dataset_stats,
        text_embedding=args.text_embedding,
        device=args.device,
        action_horizon=args.action_horizon,
        replan_steps=args.replan_steps,
        num_inference_steps=args.num_inference_steps,
    )


def infer_a0_episode(
    *,
    policy,
    front: np.ndarray,
    wrist: np.ndarray,
    states: np.ndarray,
    frame_idx: np.ndarray,
    num_samples: int,
    sample_seed0: int,
    ep: int,
) -> np.ndarray:
    T = len(frame_idx)
    rows = []
    for i in range(T):
        t = int(frame_idx[i])
        obs = {
            "front": front[t],
            "wrist": wrist[t],
            "state": states[t],
        }
        samples = []
        for k in range(num_samples):
            seed = int(sample_seed0 + 10007 * k + 100003 * ep + t)
            act = policy.infer(obs, noise_seed=seed)  # [H,22] denormed
            samples.append(np.asarray(act[0], dtype=np.float32))
        rows.append(np.stack(samples, axis=0))
        if (i + 1) % 10 == 0 or i == 0 or i + 1 == T:
            print(f"  [ep{ep}] frame {i+1}/{T} t={t}", flush=True)
    return np.stack(rows, axis=0)  # [T,K,D]


def sigma_l2(a0: np.ndarray) -> np.ndarray:
    return np.linalg.norm(a0.std(axis=1), axis=-1).astype(np.float32)


def radius_band(a0: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = a0.mean(axis=1)
    r = np.linalg.norm(a0 - mu[:, None, :], axis=-1)
    return (
        np.mean(r, axis=1).astype(np.float32),
        np.quantile(r, 0.10, axis=1).astype(np.float32),
        np.quantile(r, 0.90, axis=1).astype(np.float32),
    )


def episode_done(out: Path, ep: int) -> bool:
    return (out / "npz" / f"ep{ep:03d}_widths.npz").exists()


def run_worker(args: argparse.Namespace) -> None:
    import torch

    if not args.skip_pin_check:
        assert_pin_head()

    out = args.output
    (out / "npz").mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    eps = parse_episode_list(args.episodes)
    print(
        f"[worker {args.worker_id}] device={args.device} n_eps={len(eps)} "
        f"K={args.num_samples} stride={args.stride} steps={args.num_inference_steps} "
        f"(opensource FastWAMDexJocoPolicy)",
        flush=True,
    )
    policy = load_policy(args)
    meta_rows = []
    for j, ep in enumerate(eps):
        npz_path = out / "npz" / f"ep{ep:03d}_widths.npz"
        if npz_path.exists():
            print(f"[worker {args.worker_id}] skip done ep{ep}", flush=True)
            continue
        import pyarrow.parquet as pq

        n_full = len(
            pq.read_table(
                args.expert_dataset / "data" / "chunk-000" / f"episode_{ep:06d}.parquet",
                columns=["frame_index"],
            )
        )
        frame_idx = np.arange(0, n_full, max(args.stride, 1), dtype=np.int64)
        if len(frame_idx) == 0 or frame_idx[-1] != n_full - 1:
            frame_idx = np.unique(np.concatenate([frame_idx, [n_full - 1]]))
        t0 = time.time()
        print(
            f"[worker {args.worker_id}] ({j+1}/{len(eps)}) ep={ep} n={n_full} "
            f"queries={len(frame_idx)}",
            flush=True,
        )
        front, wrist, states, gt, frame_idx, n = load_episode_raw(
            args.expert_dataset, ep, frame_idx
        )
        progress = frame_idx.astype(np.float64) / max(n - 1, 1)
        a0 = infer_a0_episode(
            policy=policy,
            front=front,
            wrist=wrist,
            states=states,
            frame_idx=frame_idx,
            num_samples=args.num_samples,
            sample_seed0=args.sample_seed0,
            ep=ep,
        )
        s = sigma_l2(a0)
        r_mean, r_p10, r_p90 = radius_band(a0)
        np.savez_compressed(
            npz_path,
            episode=np.asarray(ep, dtype=np.int32),
            n_frames=np.asarray(n, dtype=np.int32),
            frame_idx=frame_idx.astype(np.int64),
            progress=progress.astype(np.float64),
            S0_sigma_l2=s,
            S0_radius_mean=r_mean,
            S0_radius_p10=r_p10,
            S0_radius_p90=r_p90,
            S0_actions_a0=a0.astype(np.float32),
            executed_at_frames=gt[frame_idx].astype(np.float32),
            num_samples=np.asarray(args.num_samples, dtype=np.int32),
            stride=np.asarray(args.stride, dtype=np.int32),
            num_inference_steps=np.asarray(args.num_inference_steps, dtype=np.int32),
            sample_seed0=np.asarray(args.sample_seed0, dtype=np.int32),
            image_size=np.asarray(int(getattr(policy, "image_size", 224)), dtype=np.int32),
            inference_stack=np.asarray("opensource_FastWAMDexJocoPolicy"),
        )
        row = {
            "episode": int(ep),
            "n_frames": int(n),
            "n_query": int(len(frame_idx)),
            "S0_sigma_l2_mean": float(np.mean(s)),
            "S0_sigma_l2_std": float(np.std(s)),
            "S0_cv": float(np.std(s) / max(float(np.mean(s)), 1e-9)),
            "elapsed_sec": float(time.time() - t0),
            "worker_id": args.worker_id,
        }
        meta_rows.append(row)
        json_dump(row, out / "npz" / f"ep{ep:03d}_summary.json")
        print(
            f"[worker {args.worker_id}] wrote ep{ep} mean={row['S0_sigma_l2_mean']:.4f} "
            f"cv={row['S0_cv']:.3f} elapsed={row['elapsed_sec']:.1f}s",
            flush=True,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    json_dump(
        {"worker_id": args.worker_id, "device": args.device, "rows": meta_rows},
        out / "logs" / f"worker_{args.worker_id}_done.json",
    )
    print(f"[worker {args.worker_id}] DONE", flush=True)


def collect_npz(out: Path) -> list[Path]:
    return sorted((out / "npz").glob("ep*_widths.npz"))


def compute_global_ymax(npzs: list[Path], *, q: float = 0.95, pad: float = 1.08) -> float:
    vals: list[np.ndarray] = []
    for p in npzs:
        z = np.load(p)
        vals.append(np.asarray(z["S0_sigma_l2"], dtype=np.float64).ravel())
    if not vals:
        return 1.0
    allv = np.concatenate(vals)
    allv = allv[np.isfinite(allv)]
    if allv.size == 0:
        return 1.0
    return float(max(float(np.quantile(allv, q)) * pad, 1e-3))


def plot_one_episode(z: dict[str, Any], stem: Path, ymax: float, dpi: int) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    progress = np.asarray(z["progress"], dtype=np.float64)
    width = np.asarray(z["S0_sigma_l2"], dtype=np.float64)
    ep = int(np.asarray(z["episode"]))
    n = int(np.asarray(z["n_frames"]))
    k = int(np.asarray(z["num_samples"]))
    stride = int(np.asarray(z["stride"]))
    body = progress >= 0.05
    w_body = width[body] if body.any() else width
    mean_w = float(np.mean(w_body))
    cv_w = float(np.std(w_body) / max(mean_w, 1e-9))

    fig, ax = plt.subplots(figsize=(5.2, 2.2), dpi=dpi)
    ax.plot(progress, width, color=C_S0, lw=1.7)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, ymax)
    ax.set_xlabel("episode progress", fontsize=7, color=C_MUTED)
    ax.set_ylabel(r"width $\|\sigma(a_0)\|_2$", fontsize=7, color=C_MUTED)
    ax.set_title(
        f"fold_glasses OS S0 expert ep{ep}  ·  n={n}  ·  K={k}  ·  stride={stride}  ·  "
        f"mean={mean_w:.3f}  cv={cv_w:.2f}",
        fontsize=7.5,
        loc="left",
        color="#1F2A33",
        fontweight="semibold",
    )
    ax.grid(True, axis="y", color=C_GRID, lw=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=6.5, colors=C_MUTED)
    fig.tight_layout()
    png = Path(str(stem) + ".png")
    fig.savefig(png, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png


def build_collage(pngs: list[Path], out_path: Path, cols: int, dpi: int) -> Path:
    from PIL import Image

    if not pngs:
        raise RuntimeError("No per-episode PNGs to collage")
    imgs = [Image.open(p).convert("RGB") for p in pngs]
    cell_w = max(im.width for im in imgs)
    cell_h = max(im.height for im in imgs)
    rows = (len(imgs) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * cell_w, rows * cell_h), (255, 255, 255))
    for i, im in enumerate(imgs):
        r, c = divmod(i, cols)
        x = c * cell_w + (cell_w - im.width) // 2
        y = r * cell_h + (cell_h - im.height) // 2
        canvas.paste(im, (x, y))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, dpi=(dpi, dpi))
    strip = Image.new("RGB", (cell_w, len(imgs) * cell_h), (255, 255, 255))
    for i, im in enumerate(imgs):
        x = (cell_w - im.width) // 2
        y = i * cell_h + (cell_h - im.height) // 2
        strip.paste(im, (x, y))
    strip_path = out_path.with_name(out_path.stem + "_vertical.png")
    strip.save(strip_path, dpi=(dpi, dpi))
    return out_path


def run_collage(args: argparse.Namespace) -> None:
    out = args.output
    plots = out / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    npzs = collect_npz(out)
    if not npzs:
        raise FileNotFoundError(f"No npz under {out / 'npz'}")
    ymax = (
        float(args.ymax)
        if args.ymax > 0
        else compute_global_ymax(npzs, q=float(args.ymax_quantile))
    )
    print(
        f"[collage] n={len(npzs)} global_ymax={ymax:.4f} "
        f"(q={args.ymax_quantile if args.ymax <= 0 else 'fixed'})",
        flush=True,
    )

    summaries = []
    pngs: list[Path] = []
    for p in npzs:
        z = dict(np.load(p, allow_pickle=True))
        ep = int(np.asarray(z["episode"]))
        png = plot_one_episode(z, plots / f"ep{ep:03d}_S0_width", ymax=ymax, dpi=args.dpi)
        pngs.append(png)
        prog = np.asarray(z["progress"], dtype=np.float64)
        mid = np.asarray(z["S0_sigma_l2"], dtype=np.float64)
        body = mid[prog >= 0.05] if (prog >= 0.05).any() else mid
        mean_w = float(np.mean(body))
        cv_w = float(np.std(body) / max(mean_w, 1e-9))
        summaries.append(
            {
                "episode": ep,
                "n_frames": int(np.asarray(z["n_frames"])),
                "n_query": int(len(prog)),
                "S0_sigma_l2_mean": mean_w,
                "S0_sigma_l2_std": float(np.std(body)),
                "S0_cv": cv_w,
                "S0_sigma_l2_mean_full": float(np.mean(mid)),
                "S0_cv_full": float(np.std(mid) / max(float(np.mean(mid)), 1e-9)),
                "plot": str(png),
            }
        )
        print(f"[collage] plotted ep{ep}", flush=True)

    grid = out / "fig_S0_all_expert_width_grid.png"
    build_collage(pngs, grid, cols=args.collage_cols, dpi=args.dpi)
    report = {
        "protocol": {
            "task": "fold_glasses",
            "inference_stack": "opensource_FastWAMDexJocoPolicy",
            "fastwam_pin": str(FASTWAM_PIN),
            "fastwam_pin_head": EXPECTED_PIN_HEAD,
            "model_config": str(args.model_config),
            "checkpoint": str(args.checkpoint),
            "dataset_stats": str(args.dataset_stats),
            "text_embedding": str(args.text_embedding),
            "image_size": 224,
            "norm": "z-score (opensource artifacts)",
            "dataset": str(args.expert_dataset),
            "num_samples": args.num_samples,
            "stride": args.stride,
            "num_inference_steps": args.num_inference_steps,
            "action_horizon": args.action_horizon,
            "replan_steps": args.replan_steps,
            "sample_seed0": args.sample_seed0,
            "width": "||std_k(a0)||_2",
            "progress": "[0,1]",
            "stats_progress": ">=0.05",
            "ymax_shared": ymax,
            "ymax_quantile": float(args.ymax_quantile) if args.ymax <= 0 else None,
        },
        "n_episodes": len(summaries),
        "episodes": summaries,
        "grid": str(grid),
        "vertical": str(grid.with_name(grid.stem + "_vertical.png")),
        "mean_of_means": float(np.mean([r["S0_sigma_l2_mean"] for r in summaries])),
        "mean_of_cvs": float(np.mean([r["S0_cv"] for r in summaries])),
        "median_of_cvs": float(np.median([r["S0_cv"] for r in summaries])),
    }
    json_dump(report, out / "report.json")
    print(
        json.dumps(
            {k: report[k] for k in ("n_episodes", "mean_of_means", "mean_of_cvs", "grid", "vertical")},
            indent=2,
        )
    )
    print(f"[collage] DONE -> {out}", flush=True)


def shard_episodes(episodes: list[int], n_shards: int) -> list[list[int]]:
    shards: list[list[int]] = [[] for _ in range(n_shards)]
    for i, ep in enumerate(episodes):
        shards[i % n_shards].append(ep)
    return shards


def run_orchestrate(args: argparse.Namespace) -> None:
    if not args.skip_pin_check:
        assert_pin_head()

    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    (out / "npz").mkdir(parents=True, exist_ok=True)

    for label, path in {
        "checkpoint": args.checkpoint,
        "model_config": args.model_config,
        "dataset_stats": args.dataset_stats,
        "text_embedding": args.text_embedding,
    }.items():
        if not Path(path).is_file():
            raise FileNotFoundError(f"Missing {label}: {path}")

    gpus = [int(x) for x in args.gpus.split(",") if x.strip()]
    wpg = int(args.workers_per_gpu)
    n_workers = len(gpus) * wpg
    all_eps = list_all_episodes(args.expert_dataset)
    pending = [ep for ep in all_eps if not episode_done(out, ep)]
    shards = shard_episodes(pending, n_workers)
    protocol = {
        "task": "fold_glasses",
        "inference_stack": "opensource_FastWAMDexJocoPolicy",
        "fastwam_pin": str(FASTWAM_PIN),
        "fastwam_pin_head": EXPECTED_PIN_HEAD,
        "open_repo": str(OPEN),
        "gpus": gpus,
        "workers_per_gpu": wpg,
        "n_workers": n_workers,
        "n_episodes_total": len(all_eps),
        "n_episodes_pending": len(pending),
        "num_samples": args.num_samples,
        "stride": args.stride,
        "num_inference_steps": args.num_inference_steps,
        "action_horizon": args.action_horizon,
        "replan_steps": args.replan_steps,
        "sample_seed0": args.sample_seed0,
        "checkpoint": str(args.checkpoint),
        "model_config": str(args.model_config),
        "dataset_stats": str(args.dataset_stats),
        "text_embedding": str(args.text_embedding),
        "dataset": str(args.expert_dataset),
        "image_size": 224,
        "norm": "z-score",
    }
    json_dump(protocol, out / "launch_protocol.json")
    print(json.dumps(protocol, indent=2), flush=True)

    if not pending:
        print("[orch] all episodes already done; running collage", flush=True)
        run_collage(args)
        return

    py = sys.executable
    script = str(Path(__file__).resolve())
    procs: list[subprocess.Popen] = []
    wid = 0
    for gpu in gpus:
        for local_i in range(wpg):
            eps = shards[wid]
            wid_name = f"g{gpu}_w{local_i}"
            log_path = out / "logs" / f"{wid_name}.log"
            if not eps:
                print(f"[orch] {wid_name}: empty shard, skip", flush=True)
                wid += 1
                continue
            cmd = [
                py,
                script,
                "--mode",
                "worker",
                "--device",
                "cuda:0",
                "--worker-id",
                wid_name,
                "--episodes",
                ",".join(str(e) for e in eps),
                "--output",
                str(out),
                "--expert-dataset",
                str(args.expert_dataset),
                "--checkpoint",
                str(args.checkpoint),
                "--model-config",
                str(args.model_config),
                "--dataset-stats",
                str(args.dataset_stats),
                "--text-embedding",
                str(args.text_embedding),
                "--num-samples",
                str(args.num_samples),
                "--stride",
                str(args.stride),
                "--num-inference-steps",
                str(args.num_inference_steps),
                "--action-horizon",
                str(args.action_horizon),
                "--replan-steps",
                str(args.replan_steps),
                "--sample-seed0",
                str(args.sample_seed0),
            ]
            if args.skip_pin_check:
                cmd.append("--skip-pin-check")
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            env["PYTHONUNBUFFERED"] = "1"
            env["TOKENIZERS_PARALLELISM"] = "false"
            env["DIFFSYNTH_SKIP_DOWNLOAD"] = "true"
            env["DIFFSYNTH_MODEL_BASE_PATH"] = env.get(
                "DIFFSYNTH_MODEL_BASE_PATH", str(ROOT / "checkpoints")
            )
            # Critical: workers must see OPEN + pin, not workspace src.
            env["PYTHONPATH"] = (
                f"{OPEN / 'src'}:{FASTWAM_PIN / 'src'}:"
                f"{ROOT / 'third_party' / 'dexjoco' / 'dexjoco'}:"
                f"{env.get('PYTHONPATH', '')}"
            )
            env["FASTWAM_OPEN_REPO"] = str(OPEN)
            env["FASTWAM_PIN"] = str(FASTWAM_PIN)
            print(f"[orch] launch {wid_name} gpu={gpu} n_eps={len(eps)} -> {log_path}", flush=True)
            log_f = open(log_path, "w", encoding="utf-8")
            procs.append(
                subprocess.Popen(
                    cmd,
                    env=env,
                    stdout=log_f,
                    stderr=subprocess.STDOUT,
                    cwd=str(ROOT),
                )
            )
            wid += 1
            time.sleep(3.0)  # stagger loads; ~13GB each on A100-80G

    print(f"[orch] waiting for {len(procs)} workers...", flush=True)
    codes = [proc.wait() for proc in procs]
    bad = [c for c in codes if c != 0]
    print(f"[orch] workers finished codes={codes}", flush=True)
    if bad:
        raise RuntimeError(f"{len(bad)} workers failed; see {out / 'logs'}")

    missing = [ep for ep in all_eps if not episode_done(out, ep)]
    if missing:
        raise RuntimeError(f"Missing episodes after workers: {missing[:20]}...")

    run_collage(args)


def main() -> None:
    args = parse_args()
    if args.mode == "worker":
        run_worker(args)
    elif args.mode == "collage":
        run_collage(args)
    else:
        run_orchestrate(args)


if __name__ == "__main__":
    main()
