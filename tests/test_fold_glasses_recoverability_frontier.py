from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "fold_glasses"))

import scan_failure_recoverability_frontier as frontier  # noqa: E402


def prefix(frame: int, successes: int, pass_m: int = 4) -> dict[str, object]:
    return {
        "prefix_frame": frame,
        "success_count": successes,
        "pass_m": pass_m,
        "pass_at_m_hit": successes > 0,
    }


class ScanFrameValidationTest(unittest.TestCase):
    def test_rejects_non_replan_aligned_prefix(self) -> None:
        with self.assertRaisesRegex(ValueError, "aligned"):
            frontier.validate_scan_frames(
                [0, 24, 50], replan_steps=24, max_steps=1000
            )

    def test_clips_scan_frames_to_recorded_horizon(self) -> None:
        self.assertEqual(
            frontier.clip_scan_frames([48, 72, 192, 216], horizon=198),
            [48, 72, 192],
        )
        self.assertEqual(
            frontier.recorded_horizon([0] * 1000, [0] * 1000, max_steps=1200),
            1000,
        )
        self.assertEqual(
            frontier.recorded_horizon([0] * 198, [0] * 198, max_steps=1000),
            198,
        )


class RecoverabilityFrontierTest(unittest.TestCase):
    def test_first_adjacent_zero_is_the_failure_frame(self) -> None:
        rows = [prefix(24, 1), prefix(48, 2), prefix(72, 0), prefix(96, 0)]
        found = frontier.find_recoverability_frontiers(rows, max_steps=1200)

        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]["last_recoverable_frame"], 48)
        self.assertEqual(found[0]["t_frame"], 48)
        self.assertEqual(found[0]["failure_frame"], 72)
        self.assertEqual(found[0]["t_plus_24_frame"], 72)
        self.assertEqual(found[0]["first_zero_frame"], 72)
        self.assertEqual(found[0]["event_start"], 39)
        self.assertEqual(found[0]["event_end_exclusive"], 72)
        self.assertEqual(found[0]["event_window"], [39, 72])
        self.assertFalse(found[0]["absolute_irreversibility_claimed"])

    def test_isolated_zero_after_a_hit_is_the_cliff(self) -> None:
        rows = [prefix(48, 1), prefix(72, 0), prefix(96, 2), prefix(120, 1)]
        found = frontier.find_recoverability_frontiers(rows)
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]["failure_frame"], 72)
        self.assertEqual(found[0]["last_recoverable_frame"], 48)

    def test_later_recovery_island_is_ignored(self) -> None:
        rows = [
            prefix(48, 1),
            prefix(72, 0),
            prefix(96, 0),
            prefix(120, 2),
            prefix(144, 1),
            prefix(168, 0),
        ]
        found = frontier.find_recoverability_frontiers(rows, max_steps=1200)
        self.assertEqual([row["first_zero_frame"] for row in found], [72])
        self.assertEqual(found[0]["last_recoverable_frame"], 48)

    def test_all_success_prefix_then_zero_is_a_valid_pair(self) -> None:
        rows = [prefix(48, 4), prefix(72, 0), prefix(96, 0)]
        found = frontier.find_recoverability_frontiers(rows, max_steps=1200)
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]["last_recoverable_success_count"], 4)
        self.assertTrue(
            frontier.training_pair_eligible(
                {"seed_classification": "mixed"}, found[0], pass_m=4
            )
        )

    def test_training_pair_does_not_require_mixed_seed(self) -> None:
        event = frontier.find_recoverability_frontiers(
            [prefix(48, 2), prefix(72, 0)]
        )[0]
        self.assertTrue(
            frontier.training_pair_eligible(
                {"seed_classification": "all_failure"}, event, pass_m=4
            )
        )

    def test_unrecoverable_from_the_first_scan_point_has_no_pair(self) -> None:
        rows = [prefix(48, 0), prefix(72, 0)]
        self.assertEqual(frontier.find_recoverability_frontiers(rows), [])


class PrefixPassAtMTest(unittest.TestCase):
    def test_does_not_stop_after_first_success(self) -> None:
        rows = [{"success": False}, {"success": True}]
        self.assertFalse(frontier.prefix_scan_should_stop(rows, pass_m=4))

    def test_stops_only_after_all_trials(self) -> None:
        rows = [{"success": True}] * 3
        self.assertFalse(frontier.prefix_scan_should_stop(rows, pass_m=4))
        self.assertTrue(
            frontier.prefix_scan_should_stop(rows + [{"success": False}], pass_m=4)
        )

    def test_success_candidates_are_ordered_by_replicate(self) -> None:
        rows = [
            {"prefix_frame": 48, "replicate_index": 2, "success": True},
            {"prefix_frame": 48, "replicate_index": 0, "success": True},
            {"prefix_frame": 48, "replicate_index": 1, "success": False},
            {"prefix_frame": 72, "replicate_index": 0, "success": True},
        ]
        candidates = frontier.ordered_success_candidates(rows, prefix_frame=48)
        self.assertEqual([row["replicate_index"] for row in candidates], [0, 2])


class CropSavedSuccessRolloutTest(unittest.TestCase):
    def test_event_uses_factual_prefix_then_saved_continuation(self) -> None:
        prefix = 72
        event_start = 63
        event_end = 96
        factual_actions = np.zeros((120, 22), dtype=np.float32)
        factual_states = np.zeros((120, 23), dtype=np.float32)
        factual_actions[:, 0] = np.arange(120)
        factual_states[:, 0] = np.arange(120) + 1000
        continuation_len = 80
        continuation_actions = np.zeros((continuation_len, 22), dtype=np.float32)
        continuation_states = np.zeros((continuation_len, 23), dtype=np.float32)
        continuation_actions[:, 0] = np.arange(continuation_len) + 5000
        continuation_states[:, 0] = np.arange(continuation_len) + 6000
        factual_front = {
            frame: np.full((2, 2, 3), frame, dtype=np.uint8)
            for frame in range(event_start, prefix)
        }
        factual_wrist = {
            frame: np.full((2, 2, 3), frame + 10, dtype=np.uint8)
            for frame in range(event_start, prefix)
        }
        continuation_fronts = np.stack(
            [
                np.full((2, 2, 3), 200 + index, dtype=np.uint8)
                for index in range(continuation_len)
            ]
        )
        continuation_wrists = np.stack(
            [
                np.full((2, 2, 3), 220 + index, dtype=np.uint8)
                for index in range(continuation_len)
            ]
        )

        cropped = frontier.crop_counterfactual_success_event(
            event_start=event_start,
            event_end=event_end,
            prefix_frame=prefix,
            factual_actions=factual_actions,
            factual_states=factual_states,
            factual_front=factual_front,
            factual_wrist=factual_wrist,
            continuation_actions=continuation_actions,
            continuation_states=continuation_states,
            continuation_fronts=continuation_fronts,
            continuation_wrists=continuation_wrists,
        )

        self.assertEqual(list(cropped["frame_indices"]), list(range(63, 96)))
        self.assertEqual(int(cropped["materialized_end"]), 96)
        np.testing.assert_array_equal(cropped["actions"][:9, 0], np.arange(63, 72))
        np.testing.assert_array_equal(
            cropped["actions"][9:, 0], np.arange(24) + 5000
        )
        np.testing.assert_array_equal(cropped["front"][0], factual_front[63])
        np.testing.assert_array_equal(cropped["front"][9], continuation_fronts[0])
        np.testing.assert_array_equal(cropped["front"][-1], continuation_fronts[23])

    def test_failure_cache_does_not_need_rollout_videos(self) -> None:
        self.assertTrue(
            frontier.saved_success_rollout_videos_complete({"success": False})
        )
        self.assertFalse(
            frontier.saved_success_rollout_videos_complete({"success": True})
        )


class SeedSelectionTest(unittest.TestCase):
    def test_selects_one_failure_including_all_failure_seeds(self) -> None:
        attempts = [
            {"seed": 1, "repeat": 0, "success": True, "saved_episode_index": 0},
            {"seed": 1, "repeat": 1, "success": True, "saved_episode_index": 1},
            {"seed": 2, "repeat": 0, "success": False, "saved_episode_index": 2},
            {"seed": 2, "repeat": 1, "success": True, "saved_episode_index": 3},
            {"seed": 2, "repeat": 2, "success": False, "saved_episode_index": 4},
            {
                "seed": 10106,
                "repeat": 0,
                "success": False,
                "saved_episode_index": 5,
            },
            {"seed": 9, "repeat": 0, "success": False, "saved_episode_index": 6},
        ]
        selected, audit = frontier.select_one_failure_per_seed(
            attempts,
            preferred_episode_indices={4, 5, 6},
        )

        self.assertEqual(
            [(row["seed"], row["saved_episode_index"]) for row in selected],
            [(2, 4), (9, 6), (10106, 5)],
        )
        self.assertTrue(all(row["training_eligible"] for row in selected))
        by_seed = {row["seed"]: row for row in audit}
        self.assertEqual(by_seed[1]["selection_reason"], "all_success_excluded")
        self.assertEqual(by_seed[9]["selection_reason"], "selected")
        self.assertFalse(by_seed[10106]["evaluation_only"])

    def test_estimate_scan_cost_is_available_without_gpu(self) -> None:
        estimate = frontier.estimate_scan_cost(
            num_episodes=2,
            scan_frames=[48, 72],
            pass_m=4,
            max_steps=1200,
            replan_steps=24,
        )
        self.assertEqual(estimate["policy_load_count"], 1)
        self.assertEqual(estimate["factual_replay_steps"], 2400)
        self.assertGreater(estimate["continuation_policy_replans"], 0)


if __name__ == "__main__":
    unittest.main()
