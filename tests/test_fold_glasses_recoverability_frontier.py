from __future__ import annotations

import unittest

from scripts.fold_glasses import scan_failure_recoverability_frontier as frontier


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
                [0, 24, 50], replan_steps=24, max_steps=1200
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


class PrefixEarlyStopTest(unittest.TestCase):
    def test_stops_after_first_success(self) -> None:
        rows = [{"success": False}, {"success": True}]
        self.assertTrue(frontier.prefix_scan_should_stop(rows, pass_m=4))

    def test_requires_all_failures_to_declare_unrecoverable(self) -> None:
        rows = [{"success": False}] * 3
        self.assertFalse(frontier.prefix_scan_should_stop(rows, pass_m=4))
        self.assertTrue(
            frontier.prefix_scan_should_stop(rows + [{"success": False}], pass_m=4)
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
