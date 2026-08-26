"""Tests for train-time VAE latent cache fill helpers."""

from __future__ import annotations

from pathlib import Path

import torch

from fastwam.datasets.vae_latent_cache import (
    collate_robot_video_batch,
    save_vae_latent_cache,
    vae_latent_cache_path,
)
from fastwam.models.wan22.fastwam import FastWAM


def test_save_vae_latent_cache_roundtrip(tmp_path: Path):
    latents = torch.randn(16, 3, 8, 8)
    path = save_vae_latent_cache(
        tmp_path,
        sample_id="ep0",
        window_start=12,
        latents=latents,
        video_shape=[3, 9, 64, 64],
    )
    assert path == vae_latent_cache_path(tmp_path, "ep0", 12)
    assert path.exists()
    payload = torch.load(path, map_location="cpu")
    assert payload["sample_id"] == "ep0"
    assert payload["window_start"] == 12
    assert torch.equal(payload["input_latents"], latents.to(torch.float16))


def test_collate_all_hit_all_miss_mixed():
    t_video = 5
    hit = {
        "video": torch.zeros(3, t_video, 16, 16),
        "action": torch.zeros(4, 2),
        "input_latents": torch.randn(8, 2, 4, 4),
        "eve_sample_id": "a",
        "eve_window_start": 1,
    }
    miss = {
        "video": torch.randn(3, t_video, 64, 64),
        "action": torch.zeros(4, 2),
        "eve_sample_id": "b",
        "eve_window_start": 2,
    }

    all_hit = collate_robot_video_batch(
        [
            {**hit, "input_latents": torch.randn(8, 2, 4, 4), "eve_sample_id": "h0"},
            {**hit, "input_latents": torch.randn(8, 2, 4, 4), "eve_sample_id": "h1"},
        ]
    )
    assert torch.is_tensor(all_hit["input_latents"])
    assert all_hit["input_latents"].shape == (2, 8, 2, 4, 4)
    assert torch.is_tensor(all_hit["video"])
    assert all_hit["video"].shape[-1] == 16

    all_miss = collate_robot_video_batch(
        [
            {**miss, "eve_sample_id": "m0"},
            {**miss, "eve_sample_id": "m1"},
        ]
    )
    assert "input_latents" not in all_miss
    assert torch.is_tensor(all_miss["video"])
    assert all_miss["video"].shape[-1] == 64

    mixed = collate_robot_video_batch(
        [
            {**hit, "input_latents": torch.randn(8, 2, 4, 4)},
            dict(miss),
        ]
    )
    assert isinstance(mixed["input_latents"], list)
    assert mixed["input_latents"][0] is not None
    assert mixed["input_latents"][1] is None
    assert isinstance(mixed["video"], list)
    assert mixed["video"][0].shape[-1] == 16
    assert mixed["video"][1].shape[-1] == 64
    assert mixed["eve_sample_id"] == ["a", "b"]


def test_resolve_training_input_latents_mixed_fill(tmp_path: Path):
    model = FastWAM.__new__(FastWAM)
    model.device = torch.device("cpu")
    model.torch_dtype = torch.float32
    model.vae = object()
    model.fill_vae_latent_cache = True
    model.vae_latent_cache_dir = str(tmp_path)

    encoded = torch.randn(1, 8, 2, 4, 4)

    def _fake_encode(video_tensor, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)):
        assert video_tensor.ndim == 5
        assert video_tensor.shape[0] == 1
        assert int(video_tensor.shape[-1]) >= 64
        return encoded.clone()

    model._encode_video_latents = _fake_encode  # type: ignore[method-assign]

    cached = torch.randn(8, 2, 4, 4)
    sample = {
        "video": [
            torch.zeros(3, 5, 16, 16),
            torch.randn(3, 5, 64, 64),
        ],
        "action": torch.randn(2, 4, 8),
        "input_latents": [cached, None],
        "eve_sample_id": ["s0", "s1"],
        "eve_window_start": torch.tensor([0, 10], dtype=torch.long),
    }

    latents, batch_size, num_frames = model._resolve_training_input_latents(sample)
    assert batch_size == 2
    assert num_frames == 5
    assert latents.shape == (2, 8, 2, 4, 4)
    assert torch.allclose(latents[0].cpu(), cached)
    assert torch.allclose(latents[1].cpu(), encoded[0])

    out = vae_latent_cache_path(tmp_path, "s1", 10)
    assert out.exists()
    payload = torch.load(out, map_location="cpu")
    assert payload["sample_id"] == "s1"
    assert payload["window_start"] == 10


def _bare_robot_dataset():
    from collections import OrderedDict

    from fastwam.datasets.lerobot.robot_video_dataset import RobotVideoDataset

    dataset = RobotVideoDataset.__new__(RobotVideoDataset)
    dataset._text_context_mem_cache = OrderedDict()
    dataset._text_embedding_cache_dir_ready = False
    dataset.require_vae_latent_cache = False
    dataset.drop_video_when_latents_cached = True
    dataset.video_sample_indices = [0, 4, 8]
    return dataset


def test_text_context_lru_avoids_second_torch_load(tmp_path: Path, monkeypatch):
    import hashlib

    dataset = _bare_robot_dataset()
    dataset.text_embedding_cache_dir = str(tmp_path)
    dataset.context_len = 4
    prompt = "water the plant"
    hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    cache_path = tmp_path / f"{hashed}.t5_len4.wan22ti2v5b.pt"
    torch.save(
        {
            "context": torch.ones(4, 2),
            "mask": torch.ones(4, dtype=torch.bool),
        },
        cache_path,
    )
    loads = {"n": 0}
    real_load = torch.load

    def counting_load(*args, **kwargs):
        loads["n"] += 1
        return real_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", counting_load)
    first, first_mask = dataset._get_cached_text_context(prompt)
    second, second_mask = dataset._get_cached_text_context(prompt)
    assert loads["n"] == 1
    assert torch.equal(first, second)
    first[0, 0] = 0
    assert second[0, 0] == 1
    assert torch.equal(first_mask, second_mask)


def test_vae_attach_skip_exists_does_not_stat(tmp_path: Path, monkeypatch):
    dataset = _bare_robot_dataset()
    dataset.vae_latent_cache_dir = tmp_path
    latents = torch.randn(8, 2, 4, 4)
    path = save_vae_latent_cache(
        tmp_path, sample_id="ep0", window_start=3, latents=latents
    )
    stats = {"n": 0}
    real_exists = Path.exists

    def counting_exists(self):
        stats["n"] += 1
        return real_exists(self)

    monkeypatch.setattr(Path, "exists", counting_exists)
    data = dataset._maybe_attach_vae_latents(
        {"video": torch.ones(3, 3, 64, 64)},
        sample_id="ep0",
        window_start=3,
        cache_path=path,
        skip_exists_check=True,
    )
    assert stats["n"] == 0
    assert "input_latents" in data
    miss = dataset._maybe_attach_vae_latents(
        {},
        sample_id="missing",
        window_start=0,
        cache_path=None,
        skip_exists_check=True,
    )
    assert "input_latents" not in miss
    assert stats["n"] == 0
