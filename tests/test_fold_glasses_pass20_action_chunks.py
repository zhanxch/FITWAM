from __future__ import annotations

import unittest

import numpy as np

from scripts.fold_glasses.scan_failure_pass20_action_chunks import (
    action_chunk_mse,
    consecutive_zero_nodes_should_stop,
    gt_window,
    select_all_failures,
)


class SelectAllFailuresTest(unittest.TestCase):
    def test_keeps_every_failure_on_mixed_and_all_failure_seeds(self) -> None:
        attempts = [
            {"seed": 1, "repeat": 0, "success": True, "saved_episode_index": 0},
            {"seed": 2, "repeat": 0, "success": False, "saved_episode_index": 1},
            {"seed": 2, "repeat": 1, "success": True, "saved_episode_index": 2},
            {"seed": 2, "repeat": 2, "success": False, "saved_episode_index": 3},
            {"seed": 3, "repeat": 0, "success": False, "saved_episode_index": 4},
            {"seed": 3, "repeat": 1, "success": False, "saved_episode_index": 5},
        ]
        selected, audit = select_all_failures(attempts)
        self.assertEqual(
            [row["saved_episode_index"] for row in selected],
            [1, 3, 4, 5],
        )
        by_seed = {row["seed"]: row for row in audit}
        self.assertEqual(by_seed[1]["selection_reason"], "all_success_excluded")
        self.assertEqual(by_seed[2]["selected_failure_episode_indices"], [1, 3])
        self.assertEqual(by_seed[3]["selected_failure_episode_indices"], [4, 5])


class ConsecutiveZeroStopTest(unittest.TestCase):
    def test_requires_three_adjacent_zeros(self) -> None:
        rows = [
            {"prefix_frame": 48, "success_count": 4},
            {"prefix_frame": 72, "success_count": 0},
            {"prefix_frame": 96, "success_count": 0},
        ]
        self.assertFalse(consecutive_zero_nodes_should_stop(rows, consecutive_zeros=3))
        rows.append({"prefix_frame": 120, "success_count": 0})
        self.assertTrue(consecutive_zero_nodes_should_stop(rows, consecutive_zeros=3))

    def test_a_hit_resets_the_run(self) -> None:
        rows = [
            {"prefix_frame": 48, "success_count": 0},
            {"prefix_frame": 72, "success_count": 0},
            {"prefix_frame": 96, "success_count": 2},
            {"prefix_frame": 120, "success_count": 0},
        ]
        self.assertFalse(consecutive_zero_nodes_should_stop(rows, consecutive_zeros=3))


class ActionChunkLossTest(unittest.TestCase):
    def test_mse_is_zero_when_prediction_matches_gt(self) -> None:
        mean = np.zeros(22, dtype=np.float32)
        std = np.ones(22, dtype=np.float32)
        gt = np.full((32, 22), 0.5, dtype=np.float32)
        loss, count = action_chunk_mse(gt, gt, mean, std, horizon=32)
        self.assertEqual(count, 32)
        self.assertAlmostEqual(loss, 0.0, places=6)

    def test_uses_overlapping_gt_window(self) -> None:
        mean = np.zeros(22, dtype=np.float32)
        std = np.ones(22, dtype=np.float32)
        pred = np.ones((32, 22), dtype=np.float32)
        gt = np.zeros((10, 22), dtype=np.float32)
        loss, count = action_chunk_mse(pred, gt, mean, std, horizon=32)
        self.assertEqual(count, 10)
        self.assertAlmostEqual(loss, 1.0, places=6)

    def test_gt_window_clips_to_episode_end(self) -> None:
        actions = np.arange(20, dtype=np.float32).reshape(20, 1)
        actions = np.repeat(actions, 22, axis=1)
        window = gt_window(actions, 16, 32)
        self.assertEqual(len(window), 4)


if __name__ == "__main__":
    unittest.main()
