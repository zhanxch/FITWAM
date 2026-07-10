from __future__ import annotations

import unittest

import numpy as np

from scripts.probe_state_line_distance import line_distances, mean_finite, robust_score


class StateLineProbeTest(unittest.TestCase):
    def test_line_distances_match_time_value_plane_formula(self) -> None:
        values = np.asarray(
            [
                [1.0, 2.0],
                [2.0, 4.0],
                [5.0, 6.0],
                [8.0, 9.0],
            ],
            dtype=np.float32,
        )
        scale = np.asarray([1.0, 2.0], dtype=np.float32)

        actual = line_distances(values, scale)
        normalized = values / scale
        expected = np.full_like(normalized, np.nan)
        expected[2:] = np.abs(
            2.0 * normalized[1:-1] - normalized[:-2] - normalized[2:]
        ) / np.sqrt((normalized[1:-1] - normalized[:-2]) ** 2 + 1.0)

        np.testing.assert_allclose(actual[2:], expected[2:], rtol=1e-6, atol=1e-6)
        self.assertTrue(np.isnan(actual[:2]).all())

    def test_score_uses_finite_dimension_mean_and_global_quantiles(self) -> None:
        per_dim = np.asarray(
            [
                [np.nan, np.nan],
                [np.nan, np.nan],
                [1.0, 3.0],
                [2.0, np.nan],
                [4.0, 6.0],
            ],
            dtype=np.float32,
        )
        distance = mean_finite(per_dim, axis=1)
        np.testing.assert_allclose(distance[2:], np.asarray([2.0, 2.0, 5.0], dtype=np.float32))

        score, calibration = robust_score(distance, low_q=0.0, high_q=1.0)
        np.testing.assert_allclose(score[2:], np.asarray([0.0, 0.0, 1.0], dtype=np.float32))
        self.assertTrue(np.isnan(score[:2]).all())
        self.assertEqual(calibration, {"low": 2.0, "high": 5.0, "median": 2.0})


if __name__ == "__main__":
    unittest.main()
