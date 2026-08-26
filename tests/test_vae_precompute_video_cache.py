"""Tests for episode-span VAE precompute video decode cache."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from fastwam.datasets.lerobot.lerobot.datasets.video_utils import (  # noqa: E402
    select_closest_video_frames,
)
from fastwam.datasets.lerobot.lerobot.lerobot_dataset import (  # noqa: E402
    LeRobotDataset,
)
from precompute_vae_latents import (  # noqa: E402
    _cache_key_from_meta,
    _group_indices_by_episode,
)


class SelectClosestVideoFramesTests(unittest.TestCase):
    def test_uint8_to_float_and_nearest_timestamp(self) -> None:
        frames = torch.arange(10, dtype=torch.uint8).view(10, 1, 1, 1).expand(10, 3, 2, 2).contiguous()
        loaded_ts = torch.linspace(0.0, 0.9, 10)
        out = select_closest_video_frames(
            frames,
            loaded_ts,
            [0.0, 0.51, 0.9],
            tolerance_s=0.1,
            video_path="dummy.mp4",
            backend="pyav",
        )
        self.assertEqual(tuple(out.shape), (3, 3, 2, 2))
        self.assertEqual(out.dtype, torch.float32)
        self.assertAlmostEqual(float(out[0, 0, 0, 0]), 0.0)
        self.assertAlmostEqual(float(out[1, 0, 0, 0]), 5.0 / 255.0)
        self.assertAlmostEqual(float(out[2, 0, 0, 0]), 9.0 / 255.0)

    def test_rejects_out_of_tolerance(self) -> None:
        frames = torch.zeros(2, 3, 2, 2, dtype=torch.uint8)
        loaded_ts = torch.tensor([0.0, 0.1], dtype=torch.float32)
        with self.assertRaises(AssertionError):
            select_closest_video_frames(
                frames,
                loaded_ts,
                [5.0],
                tolerance_s=0.1,
                video_path="dummy.mp4",
            )


class GroupEpisodeIndicesTests(unittest.TestCase):
    def test_groups_preserve_order_and_fallback(self) -> None:
        samples = [
            {"dataset_root": "/a", "episode_index": 1, "unit": {"sample_id": "s0"}, "window_start": 0},
            {"dataset_root": "/a", "episode_index": 1, "unit": {"sample_id": "s0"}, "window_start": 4},
            {"dataset_root": "/b", "episode_index": 0, "unit": {"sample_id": "s1"}, "window_start": 8},
            {"unit": {"sample_id": "orphan"}, "window_start": 0},
        ]
        groups = _group_indices_by_episode(samples, [0, 1, 2, 3])
        self.assertEqual(list(groups.keys()), [("/a", 1), ("/b", 0), None])
        self.assertEqual(groups[("/a", 1)], [0, 1])
        self.assertEqual(groups[("/b", 0)], [2])
        self.assertEqual(groups[None], [3])
        sample_id, window_start = _cache_key_from_meta(samples, 1)
        self.assertEqual(sample_id, "s0")
        self.assertEqual(window_start, 4)


class QueryVideosCacheTests(unittest.TestCase):
    def test_query_videos_uses_prefetched_span(self) -> None:
        ds = LeRobotDataset.__new__(LeRobotDataset)
        ds.root = Path("/tmp/ds")
        ds.tolerance_s = 0.05
        ds.video_backend = "pyav"
        ds.meta = mock.Mock()
        ds.meta.video_keys = ["observation.images.front"]
        ds.meta.get_video_file_path.return_value = Path("videos/front.mp4")
        frames = torch.arange(4, dtype=torch.uint8).view(4, 1, 1, 1).expand(4, 3, 2, 2).contiguous()
        loaded_ts = torch.tensor([0.0, 0.1, 0.2, 0.3], dtype=torch.float32)
        video_path = str(ds.root / "videos/front.mp4")
        ds._video_decode_cache = {video_path: (frames, loaded_ts)}

        with mock.patch(
            "fastwam.datasets.lerobot.lerobot.lerobot_dataset.decode_video_frames",
            side_effect=AssertionError("should not decode on cache hit"),
        ):
            item = ds._query_videos(
                {"observation.images.front": [0.0, 0.2]},
                ep_idx=0,
            )
        video = item["observation.images.front"]
        self.assertEqual(tuple(video.shape), (2, 3, 2, 2))
        self.assertAlmostEqual(float(video[0, 0, 0, 0]), 0.0)
        self.assertAlmostEqual(float(video[1, 0, 0, 0]), 2.0 / 255.0)


if __name__ == "__main__":
    unittest.main()
