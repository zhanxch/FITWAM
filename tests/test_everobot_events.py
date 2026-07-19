from __future__ import annotations

import unittest

import numpy as np

from fastwam.everobot_events import (
    exponential_moving_average,
    extract_candidate_windows,
    hysteresis_mask,
    merge_short_gaps,
    remove_short_runs,
    trailing_median,
)


class EveRobotEventsTest(unittest.TestCase):
    def test_trailing_median_uses_only_finite_history(self) -> None:
        values = np.asarray([np.nan, 1.0, np.inf, 3.0, 100.0])

        actual = trailing_median(values, window_size=3)

        np.testing.assert_allclose(
            actual,
            np.asarray([np.nan, 1.0, 1.0, 2.0, 51.5]),
            equal_nan=True,
        )

    def test_ema_preserves_invalid_positions_and_finite_state(self) -> None:
        values = np.asarray([np.nan, 1.0, 3.0, np.inf, 5.0])

        actual = exponential_moving_average(values, alpha=0.5)

        np.testing.assert_allclose(
            actual,
            np.asarray([np.nan, 1.0, 2.0, np.nan, 3.5]),
            equal_nan=True,
        )

    def test_hysteresis_starts_high_and_stops_below_low(self) -> None:
        scores = np.asarray([np.nan, 0.2, 0.7, 0.5, 0.3, np.nan, 0.8, 0.1])

        actual = hysteresis_mask(
            scores,
            high_threshold=0.6,
            low_threshold=0.4,
        )

        np.testing.assert_array_equal(
            actual,
            np.asarray([False, False, True, True, False, False, True, False]),
        )

    def test_gap_merge_does_not_cross_invalid_frames(self) -> None:
        mask = np.asarray(
            [True, False, True, True, False, True, True, False, False, True]
        )
        valid = np.asarray(
            [True, True, True, True, False, True, True, True, True, True]
        )

        actual = merge_short_gaps(mask, max_gap=2, valid_mask=valid)

        np.testing.assert_array_equal(
            actual,
            np.asarray([True, True, True, True, False, True, True, True, True, True]),
        )

    def test_extraction_does_not_bridge_raw_invalid_frames(self) -> None:
        scores = np.asarray([0.8, 0.8, np.nan, 0.8, 0.8])

        result = extract_candidate_windows(
            scores,
            median_window=3,
            high_threshold=0.7,
            low_threshold=0.4,
            max_gap=1,
        )

        self.assertTrue(np.isnan(result.smoothed_scores[2]))
        self.assertEqual(
            [
                (candidate.core_start_frame, candidate.core_end_frame)
                for candidate in result.candidates
            ],
            [(0, 2), (3, 5)],
        )

    def test_short_runs_are_removed_at_episode_boundaries(self) -> None:
        mask = np.asarray([True, False, True, True, False, True])

        actual = remove_short_runs(mask, min_run=2)

        np.testing.assert_array_equal(
            actual,
            np.asarray([False, False, True, True, False, False]),
        )

    def test_candidate_padding_and_minimum_expansion_clip_to_episode(self) -> None:
        scores = np.asarray([0.8, 0.8, 0.0, 0.0, 0.0, 0.0])

        result = extract_candidate_windows(
            scores,
            high_threshold=0.7,
            low_threshold=0.4,
            pre_padding=3,
            post_padding=0,
            min_window=5,
        )

        self.assertEqual(len(result.candidates), 1)
        candidate = result.candidates[0]
        self.assertEqual(
            (
                candidate.core_start_frame,
                candidate.core_end_frame,
                candidate.start_frame,
                candidate.end_frame,
            ),
            (0, 2, 0, 5),
        )

    def test_short_pulse_and_gap_controls_produce_sustained_candidates(self) -> None:
        scores = np.asarray(
            [0.0, 0.9, 0.0, 0.0, 0.8, 0.7, 0.0, 0.8, 0.7, 0.0]
        )

        result = extract_candidate_windows(
            scores,
            high_threshold=0.75,
            low_threshold=0.5,
            max_gap=1,
            min_run=2,
        )

        self.assertEqual(
            [
                (candidate.core_start_frame, candidate.core_end_frame)
                for candidate in result.candidates
            ],
            [(4, 9)],
        )
        np.testing.assert_array_equal(
            result.active_mask,
            np.asarray(
                [False, False, False, False, True, True, True, True, True, False]
            ),
        )

    def test_short_pulse_is_not_merged_into_a_nearby_valid_event(self) -> None:
        scores = np.asarray([0.9, 0.0, 0.0, 0.8, 0.8, 0.0])

        result = extract_candidate_windows(
            scores,
            high_threshold=0.75,
            low_threshold=0.5,
            max_gap=2,
            min_run=2,
        )

        self.assertEqual(
            [
                (candidate.core_start_frame, candidate.core_end_frame)
                for candidate in result.candidates
            ],
            [(3, 5)],
        )

    def test_confidence_and_episode_weights_are_deterministic(self) -> None:
        scores = np.asarray([0.8, 0.6, 0.0, 1.0, 0.8])
        kwargs = {
            "high_threshold": 0.6,
            "low_threshold": 0.5,
        }

        first = extract_candidate_windows(scores, **kwargs)
        second = extract_candidate_windows(scores.copy(), **kwargs)

        self.assertEqual(first.candidates, second.candidates)
        self.assertEqual(len(first.candidates), 2)
        np.testing.assert_allclose(
            [candidate.confidence for candidate in first.candidates],
            [0.7, 0.9],
        )
        np.testing.assert_allclose(
            [candidate.peak_score for candidate in first.candidates],
            [0.8, 1.0],
        )
        np.testing.assert_allclose(
            [candidate.episode_weight for candidate in first.candidates],
            [0.4375, 0.5625],
        )
        self.assertAlmostEqual(
            sum(candidate.episode_weight for candidate in first.candidates),
            1.0,
        )

    def test_candidate_cap_keeps_strongest_events_in_temporal_order(self) -> None:
        scores = np.asarray([0.6, 0.0, 0.9, 0.0, 0.8])

        result = extract_candidate_windows(
            scores,
            high_threshold=0.5,
            low_threshold=0.5,
            max_candidates_per_episode=2,
        )

        self.assertEqual(
            [
                (candidate.core_start_frame, candidate.core_end_frame)
                for candidate in result.candidates
            ],
            [(2, 3), (4, 5)],
        )
        np.testing.assert_array_equal(
            result.active_mask,
            np.asarray([False, False, True, False, True]),
        )
        self.assertAlmostEqual(
            sum(candidate.episode_weight for candidate in result.candidates),
            1.0,
        )

    def test_all_invalid_or_subthreshold_scores_return_no_candidates(self) -> None:
        for scores in (
            np.asarray([np.nan, np.nan]),
            np.asarray([0.1, 0.2, 0.3]),
            np.asarray([], dtype=np.float64),
        ):
            with self.subTest(scores=scores):
                result = extract_candidate_windows(
                    scores,
                    high_threshold=0.6,
                    low_threshold=0.4,
                )
                self.assertEqual(result.candidates, ())
                self.assertFalse(result.active_mask.any())

    def test_invalid_parameters_are_rejected(self) -> None:
        scores = np.asarray([0.1, 0.9])
        invalid_calls = (
            lambda: trailing_median(scores, 0),
            lambda: exponential_moving_average(scores, 0.0),
            lambda: hysteresis_mask(
                scores,
                high_threshold=0.4,
                low_threshold=0.5,
            ),
            lambda: merge_short_gaps(np.asarray([True]), -1),
            lambda: remove_short_runs(np.asarray([True]), 0),
            lambda: extract_candidate_windows(scores, min_window=0),
            lambda: extract_candidate_windows(
                scores, max_candidates_per_episode=0
            ),
        )

        for call in invalid_calls:
            with self.subTest(call=call):
                with self.assertRaises(ValueError):
                    call()


if __name__ == "__main__":
    unittest.main()
