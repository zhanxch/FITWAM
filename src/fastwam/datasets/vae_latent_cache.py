"""Shared helpers for per-window frozen-VAE latent caches."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import torch
from torch.utils.data._utils.collate import default_collate


def vae_latent_cache_key(sample_id: str, window_start: int) -> str:
    raw = f"{sample_id}__w{int(window_start)}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def vae_latent_cache_path(
    cache_dir: str | Path,
    sample_id: str,
    window_start: int,
) -> Path:
    return Path(cache_dir).expanduser() / f"{vae_latent_cache_key(sample_id, window_start)}.pt"


def resolve_optional_path(value: Any) -> Path | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"null", "none"}:
        return None
    return Path(text).expanduser()


def save_vae_latent_cache(
    cache_dir: str | Path,
    *,
    sample_id: str,
    window_start: int,
    latents: torch.Tensor,
    video_shape: list[int] | tuple[int, ...] | None = None,
) -> Path:
    """Atomically write a per-window VAE latent cache entry (CPU float16)."""
    if not torch.is_tensor(latents):
        raise TypeError(f"`latents` must be a tensor, got {type(latents)}")
    if latents.ndim == 5 and latents.shape[0] == 1:
        latents = latents[0]
    if latents.ndim != 4:
        raise ValueError(f"`latents` must be [C,T,H,W], got {tuple(latents.shape)}")

    out_path = vae_latent_cache_path(cache_dir, sample_id=str(sample_id), window_start=int(window_start))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload_latents = latents.detach().to(dtype=torch.float16).contiguous().cpu()
    payload: dict[str, Any] = {
        "input_latents": payload_latents,
        "sample_id": str(sample_id),
        "window_start": int(window_start),
        "latent_shape": list(payload_latents.shape),
    }
    if video_shape is not None:
        payload["video_shape"] = [int(x) for x in video_shape]

    # Unique tmp avoids clobber when two ranks race the same key.
    tmp = out_path.parent / (
        f"{out_path.stem}.{os.getpid()}.{torch.randint(0, 1_000_000, (1,)).item()}.pt.tmp"
    )
    try:
        torch.save(payload, tmp)
        os.replace(tmp, out_path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
    return out_path


def collate_robot_video_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate that tolerates partial `input_latents` (cache hit/miss mix).

    Cache hits use a tiny placeholder video (16x16) while misses keep full-res
    pixels, so `video` may also be heterogeneous:

    - all hit  -> stacked float tensor [B,C,T,H,W] for latents; video stacked stubs
    - all miss -> latents omitted; video stacked full-res
    - mixed    -> `input_latents` is list[Tensor|None]; `video` is list[Tensor]
    """
    if not batch:
        raise ValueError("Cannot collate an empty batch.")

    latents = [sample.pop("input_latents", None) for sample in batch]
    videos = [sample.pop("video", None) for sample in batch]
    collated = default_collate(batch)

    present = [z for z in latents if z is not None]
    if present:
        if len(present) == len(latents):
            stacked = torch.stack(present, dim=0)
            if stacked.ndim != 5:
                raise ValueError(
                    f"Stacked `input_latents` must be [B,C,T,H,W], got {tuple(stacked.shape)}"
                )
            collated["input_latents"] = stacked
        else:
            normalized: list[torch.Tensor | None] = []
            for z in latents:
                if z is None:
                    normalized.append(None)
                    continue
                if not torch.is_tensor(z):
                    raise TypeError(f"`input_latents` entry must be a tensor, got {type(z)}")
                if z.ndim != 4:
                    raise ValueError(
                        f"`input_latents` entry must be [C,T,H,W], got {tuple(z.shape)}"
                    )
                normalized.append(z)
            collated["input_latents"] = normalized

    if all(v is None for v in videos):
        return collated
    if any(v is None for v in videos):
        raise ValueError("Batch mixes missing and present `video` entries.")
    shapes = [tuple(v.shape) for v in videos]
    if len(set(shapes)) == 1:
        collated["video"] = torch.stack(videos, dim=0)
    else:
        for i, v in enumerate(videos):
            if not torch.is_tensor(v) or v.ndim != 4:
                raise ValueError(
                    f"`video[{i}]` must be [C,T,H,W], got "
                    f"{None if v is None else tuple(getattr(v, 'shape', ()))}"
                )
        collated["video"] = videos
    return collated
