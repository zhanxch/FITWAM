#!/usr/bin/env python3
"""Precompute per-window frozen-VAE latents for Eve/B1 offline training.

Wan2.2 VAE is causal/chunked, so each training window must be encoded
independently (slicing a long-episode encode is not training-equivalent).

Usage (after sourcing the DEWO v9 protocol env):

  python scripts/precompute_vae_latents.py \\
    task=dexjoco/dexjoco_dewo_v9_offline_b1_jump_fast_uncond \\
    +vae_latent_cache_dir=/path/to/vae_latent_cache

Or set env VAE_LATENT_CACHE_DIR. Multi-GPU:

  torchrun --standalone --nproc_per_node=4 scripts/precompute_vae_latents.py ...
"""

from __future__ import annotations

import json
import os
from collections import OrderedDict
from pathlib import Path
from typing import Any

import hydra
import torch
import torch.distributed as dist
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from fastwam.datasets.vae_latent_cache import vae_latent_cache_path
from fastwam.models.wan22.helpers.loader import _resolve_configs, _load_registered_model
from fastwam.utils.config_resolvers import register_default_resolvers
from fastwam.utils.logging_config import get_logger, setup_logging

register_default_resolvers()
logger = get_logger(__name__)

DEFAULT_MODEL_ID = "Wan-AI/Wan2.2-TI2V-5B"
DEFAULT_TOKENIZER_MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B"


def _init_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return False, 0, 1, 0
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend=backend, init_method="env://")
    return True, dist.get_rank(), dist.get_world_size(), local_rank


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n", "", "null", "none"}:
        return False
    raise ValueError(f"Cannot parse bool: {value!r}")


def _resolve_cache_dir(cfg: DictConfig) -> Path:
    raw = OmegaConf.select(cfg, "vae_latent_cache_dir", default=None)
    if raw is None or str(raw).strip().lower() in {"", "null", "none"}:
        env = os.environ.get("VAE_LATENT_CACHE_DIR", "").strip()
        if not env:
            raise ValueError(
                "Set `+vae_latent_cache_dir=...` or env VAE_LATENT_CACHE_DIR."
            )
        raw = env
    path = Path(str(raw)).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _load_vae_only(*, model_id: str, device: str, dtype: torch.dtype, redirect: bool):
    _, _, vae_config, _ = _resolve_configs(
        model_id=model_id,
        tokenizer_model_id=DEFAULT_TOKENIZER_MODEL_ID,
        redirect_common_files=redirect,
    )
    vae_config.download_if_necessary()
    vae = _load_registered_model(
        vae_config.path,
        "wan_video_vae",
        torch_dtype=dtype,
        device=device,
    )
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    return vae, str(vae_config.path)


@torch.no_grad()
def _encode_video(vae, video: torch.Tensor, device: str, dtype: torch.dtype) -> torch.Tensor:
    # Match FastWAM._encode_video_latents: video [B,3,T,H,W] in [-1,1]
    if video.ndim != 4:
        raise ValueError(f"Expected video [3,T,H,W], got {tuple(video.shape)}")
    batch = video.unsqueeze(0).to(device=device, dtype=dtype, non_blocking=True)
    z = vae.encode(batch, device=device, tiled=False)
    if isinstance(z, list):
        z = z[0]
    if z.ndim == 5 and z.shape[0] == 1:
        z = z[0]
    if z.ndim != 4:
        raise ValueError(f"VAE encode expected [C,T,H,W], got {tuple(z.shape)}")
    return z.detach().to(dtype=torch.float16).contiguous().cpu()


def _resolve_shard(cfg: DictConfig, *, dist_rank: int, dist_world: int) -> tuple[int, int]:
    """Optional file-shard overrides for multi-proc-per-GPU launches."""

    shard_rank = OmegaConf.select(cfg, "vae_shard_rank", default=None)
    shard_world = OmegaConf.select(cfg, "vae_shard_world", default=None)
    if shard_rank is None and shard_world is None:
        return int(dist_rank), int(dist_world)
    if shard_rank is None or shard_world is None:
        raise ValueError("Set both +vae_shard_rank and +vae_shard_world together.")
    rank = int(shard_rank)
    world = int(shard_world)
    if world < 1 or rank < 0 or rank >= world:
        raise ValueError(f"Invalid VAE shard rank/world: {rank}/{world}")
    return rank, world


def _group_indices_by_episode(samples_meta, indices: list[int]) -> OrderedDict:
    """Group Eve windows by (dataset_root, episode_index), preserving first-seen order."""
    groups: OrderedDict[tuple[str, int] | None, list[int]] = OrderedDict()
    for idx in indices:
        key = None
        if isinstance(samples_meta, list) and 0 <= idx < len(samples_meta):
            sample_ref = samples_meta[idx]
            if isinstance(sample_ref, dict):
                root = sample_ref.get("dataset_root")
                episode_index = sample_ref.get("episode_index")
                if root not in {None, ""} and episode_index is not None:
                    key = (str(root), int(episode_index))
        groups.setdefault(key, []).append(idx)
    return groups


def _cache_key_from_meta(samples_meta, idx: int) -> tuple[str | None, int | None]:
    if not isinstance(samples_meta, list) or not (0 <= idx < len(samples_meta)):
        return None, None
    sample_ref = samples_meta[idx]
    if not isinstance(sample_ref, dict):
        return None, None
    unit = sample_ref.get("unit", {}) if isinstance(sample_ref.get("unit"), dict) else {}
    sample_id = str(unit.get("sample_id", unit.get("event_id", f"idx_{idx}")))
    window_start = int(sample_ref.get("window_start", -1))
    return sample_id, window_start


def _build_dataset(node_cfg: DictConfig, *, force_decode_video: bool):
    # Precompute must decode pixels even if drop_video is configured for training.
    overrides = {
        "vae_latent_cache_dir": None,
        "require_vae_latent_cache": False,
        "drop_video_when_latents_cached": False,
        "is_training_set": False,  # disable CFG dropout randomness
    }
    if force_decode_video:
        overrides["drop_video_when_latents_cached"] = False
    cfg = OmegaConf.merge(node_cfg, OmegaConf.create(overrides))
    dataset = instantiate(cfg)
    # VAE precompute only needs pixels; skip T5 cache I/O and FAST tokenization.
    if hasattr(dataset, "force_skip_text"):
        dataset.force_skip_text = True
    return dataset


def _write_window_latent(
    *,
    dataset,
    vae,
    cache_dir: Path,
    device: str,
    dtype: torch.dtype,
    idx: int,
    sample_id: str | None,
    window_start: int | None,
    overwrite: bool,
) -> str:
    sample = dataset[idx]
    sample_id = str(sample.get("eve_sample_id", sample_id or f"idx_{idx}"))
    window_start = int(sample.get("eve_window_start", window_start if window_start is not None else -1))
    if window_start < 0:
        raise ValueError(f"Sample {idx} missing eve_window_start")
    out_path = vae_latent_cache_path(cache_dir, sample_id, window_start)
    if out_path.exists() and not overwrite:
        return "skipped"
    video = sample["video"]
    if not torch.is_tensor(video) or video.ndim != 4 or int(video.shape[-1]) < 64:
        raise ValueError(
            f"Sample {idx} ({sample_id}) has invalid video for VAE encode: "
            f"{None if video is None else tuple(getattr(video, 'shape', ()))}"
        )
    latents = _encode_video(vae, video, device=device, dtype=dtype)
    tmp = out_path.with_suffix(".pt.tmp")
    torch.save(
        {
            "input_latents": latents,
            "sample_id": sample_id,
            "window_start": window_start,
            "video_shape": list(video.shape),
            "latent_shape": list(latents.shape),
        },
        tmp,
    )
    os.replace(tmp, out_path)
    return "wrote"


def _encode_split(
    *,
    dataset,
    vae,
    cache_dir: Path,
    device: str,
    dtype: torch.dtype,
    rank: int,
    world_size: int,
    overwrite: bool,
    max_samples: int | None,
) -> dict[str, int]:
    n = len(dataset)
    indices = list(range(rank, n, world_size))
    if max_samples is not None:
        indices = indices[: max(0, int(max_samples))]
    wrote = skipped = errors = 0

    # Fast resume: resolve cache keys from Eve metadata without decoding video.
    samples_meta = getattr(dataset, "_samples", None)
    groups = _group_indices_by_episode(samples_meta, indices)
    can_prefetch = hasattr(dataset, "prefetch_episode_video") and hasattr(
        dataset, "enable_video_decode_cache"
    )
    if can_prefetch:
        dataset.enable_video_decode_cache(True)
        if rank == 0:
            n_eps = sum(1 for key in groups if key is not None)
            logger.info(
                "VAE episode prefetch enabled: episodes=%d windows=%d",
                n_eps,
                len(indices),
            )

    progress = None
    if rank == 0:
        progress = tqdm(total=len(indices), desc=f"encode rank{rank}", leave=True)

    try:
        for ep_key, ep_indices in groups.items():
            pending: list[tuple[int, str | None, int | None]] = []
            for idx in ep_indices:
                sample_id, window_start = _cache_key_from_meta(samples_meta, idx)
                if window_start is not None and window_start >= 0 and sample_id is not None:
                    out_path = vae_latent_cache_path(cache_dir, sample_id, window_start)
                    if out_path.exists() and not overwrite:
                        skipped += 1
                        if progress is not None:
                            progress.update(1)
                        continue
                pending.append((idx, sample_id, window_start))
            if not pending:
                continue
            if can_prefetch and ep_key is not None:
                try:
                    dataset.prefetch_episode_video(ep_key[0], ep_key[1])
                except Exception as err:
                    logger.warning(
                        "VAE episode prefetch failed for %s: %s; "
                        "falling back to per-window decode",
                        ep_key,
                        err,
                    )
            try:
                for idx, sample_id, window_start in pending:
                    status = _write_window_latent(
                        dataset=dataset,
                        vae=vae,
                        cache_dir=cache_dir,
                        device=device,
                        dtype=dtype,
                        idx=idx,
                        sample_id=sample_id,
                        window_start=window_start,
                        overwrite=overwrite,
                    )
                    if status == "skipped":
                        skipped += 1
                    else:
                        wrote += 1
                    if progress is not None:
                        progress.update(1)
            finally:
                if can_prefetch:
                    dataset.clear_video_decode_cache()
    finally:
        if progress is not None:
            progress.close()
        if can_prefetch:
            dataset.enable_video_decode_cache(False)

    return {"wrote": wrote, "skipped": skipped, "errors": errors, "total_seen": len(indices)}


@hydra.main(version_base="1.3", config_name="train", config_path="../configs")
def main(cfg: DictConfig) -> None:
    setup_logging()
    distributed, dist_rank, dist_world, local_rank = _init_distributed()
    # When launched with CUDA_VISIBLE_DEVICES=<one gpu>, always use cuda:0.
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if distributed and torch.cuda.device_count() > 1:
        device = f"cuda:{local_rank}"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    rank, world_size = _resolve_shard(cfg, dist_rank=dist_rank, dist_world=dist_world)

    cache_dir = _resolve_cache_dir(cfg)
    overwrite = _as_bool(OmegaConf.select(cfg, "overwrite", default=False))
    max_samples = OmegaConf.select(cfg, "max_samples", default=None)
    max_samples = None if max_samples is None else int(max_samples)
    encode_val = _as_bool(OmegaConf.select(cfg, "encode_val", default=True))

    model_cfg = cfg.get("model", {})
    model_id = str(model_cfg.get("model_id", DEFAULT_MODEL_ID))
    redirect = bool(model_cfg.get("redirect_common_files", True))

    if rank == 0:
        logger.info("VAE latent cache dir: %s", cache_dir)
        logger.info(
            "shard=%d/%d dist=%s overwrite=%s encode_val=%s device=%s",
            rank,
            world_size,
            f"{dist_rank}/{dist_world}" if distributed else "off",
            overwrite,
            encode_val,
            device,
        )

    vae, vae_path = _load_vae_only(
        model_id=model_id,
        device=device,
        dtype=dtype,
        redirect=redirect,
    )

    splits = [("train", cfg.data.train)]
    if encode_val and OmegaConf.select(cfg, "data.val", default=None) is not None:
        splits.append(("val", cfg.data.val))

    summary = {
        "cache_dir": str(cache_dir),
        "vae_path": vae_path,
        "model_id": model_id,
        "splits": {},
    }

    for split_name, node_cfg in splits:
        if rank == 0:
            logger.info("Building %s dataset...", split_name)
        dataset = _build_dataset(node_cfg, force_decode_video=True)
        stats = _encode_split(
            dataset=dataset,
            vae=vae,
            cache_dir=cache_dir,
            device=device,
            dtype=dtype,
            rank=rank,
            world_size=world_size,
            overwrite=overwrite,
            max_samples=max_samples,
        )
        summary["splits"][split_name] = {
            "num_samples": len(dataset),
            **stats,
            "video_size": list(getattr(dataset, "video_size", [])),
            "num_frames": int(getattr(dataset, "num_frames", -1)),
            "action_video_freq_ratio": int(getattr(dataset, "action_video_freq_ratio", -1)),
        }
        if rank == 0:
            logger.info("[%s] %s", split_name, stats)

    if distributed:
        dist.barrier()

    if rank == 0:
        index_path = cache_dir / "index.json"
        # Merge prior index if present (resume-friendly metadata).
        prior = {}
        if index_path.exists():
            try:
                prior = json.loads(index_path.read_text(encoding="utf-8"))
            except Exception:
                prior = {}
        prior.update(summary)
        prior["num_cache_files"] = len(list(cache_dir.glob("*.pt")))
        index_path.write_text(json.dumps(prior, indent=2) + "\n", encoding="utf-8")
        logger.info("Wrote %s (files=%s)", index_path, prior["num_cache_files"])

    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
