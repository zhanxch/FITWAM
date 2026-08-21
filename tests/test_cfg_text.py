"""Unit tests for ternary DEWO CFG channel scheduling."""

from __future__ import annotations

import unittest
from unittest import mock
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fastwam.datasets.cfg_text import (
    apply_ternary_cfg_suffix,
    normalize_cfg_channel_probs,
    sample_cfg_channel,
    select_cfg_schedule_probs,
)


class CfgTextTests(unittest.TestCase):
    def test_normalize_probs(self) -> None:
        probs = normalize_cfg_channel_probs({"outcome": 4, "fast": 2, "base": 4})
        assert probs is not None
        self.assertAlmostEqual(probs["outcome"], 0.4)
        self.assertAlmostEqual(probs["fast"], 0.2)
        self.assertAlmostEqual(probs["base"], 0.4)

    def test_sample_channel_frequencies(self) -> None:
        probs = {"outcome": 0.4, "fast": 0.2, "base": 0.4}
        rng = np.random.RandomState(0)
        counts = {"outcome": 0, "fast": 0, "base": 0}
        n = 5000
        for _ in range(n):
            counts[sample_cfg_channel(probs, rng=rng)] += 1
        self.assertAlmostEqual(counts["outcome"] / n, 0.4, delta=0.03)
        self.assertAlmostEqual(counts["fast"] / n, 0.2, delta=0.03)
        self.assertAlmostEqual(counts["base"] / n, 0.4, delta=0.03)

    def test_val_samples_cfg_channels_not_success_only(self) -> None:
        """Val/test must keep ternary CFG, not collapse to success/outcome."""

        with mock.patch(
            "fastwam.datasets.cfg_text.sample_cfg_channel", return_value="base"
        ):
            text, channel = apply_ternary_cfg_suffix(
                "Fold the glasses.",
                outcome_flag=0,
                success_suffix=" Successful execution.",
                failure_suffix=None,
                channel_probs={"outcome": 0.4, "fast": 0.2, "base": 0.4},
                failure_channel_probs={"outcome": 0.0, "fast": 0.5, "base": 0.5},
                actions=np.zeros((8, 22), dtype=np.float32),
                is_training=False,
                fast_fail_closed=True,
            )
        self.assertEqual((text, channel), ("Fold the glasses.", "base"))

        with mock.patch(
            "fastwam.datasets.cfg_text.sample_cfg_channel", return_value="fast"
        ), mock.patch(
            "fastwam.datasets.cfg_text.format_fast_action_suffix",
            return_value=" Action codes: 7 8",
        ):
            text, channel = apply_ternary_cfg_suffix(
                "Fold the glasses.",
                outcome_flag=0,
                success_suffix=" Successful execution.",
                failure_suffix=None,
                channel_probs={"outcome": 0.4, "fast": 0.2, "base": 0.4},
                actions=np.zeros((8, 22), dtype=np.float32),
                is_training=False,
                fast_fail_closed=True,
            )
        self.assertEqual(channel, "fast")
        self.assertIn("Action codes", text)

        with mock.patch(
            "fastwam.datasets.cfg_text.sample_cfg_channel", return_value="outcome"
        ):
            text, channel = apply_ternary_cfg_suffix(
                "Fold the glasses.",
                outcome_flag=0,
                success_suffix=" Successful execution.",
                failure_suffix=None,
                channel_probs={"outcome": 0.4, "fast": 0.2, "base": 0.4},
                actions=np.zeros((8, 22), dtype=np.float32),
                is_training=False,
                fast_fail_closed=True,
            )
        self.assertEqual(channel, "outcome")
        self.assertTrue(text.endswith(" Successful execution."))

    def test_legacy_dropout_base(self) -> None:
        rng_state = np.random.get_state()
        try:
            np.random.seed(1)
            # With dropout=1.0, legacy path always returns base.
            text, channel = apply_ternary_cfg_suffix(
                "Fold the glasses.",
                outcome_flag=1,
                success_suffix=" Successful execution.",
                failure_suffix=" Failed execution.",
                channel_probs=None,
                is_training=True,
                legacy_dropout_prob=1.0,
            )
            self.assertEqual(channel, "base")
            self.assertEqual(text, "Fold the glasses.")
        finally:
            np.random.set_state(rng_state)

    def test_formal_fast_channel_fails_closed_without_actions(self) -> None:
        with mock.patch(
            "fastwam.datasets.cfg_text.sample_cfg_channel",
            return_value="fast",
        ):
            with self.assertRaisesRegex(ValueError, "actions are missing"):
                apply_ternary_cfg_suffix(
                    "Fold the glasses.",
                    outcome_flag=0,
                    success_suffix=" Successful execution.",
                    failure_suffix=" Failed execution.",
                    channel_probs={"outcome": 0.4, "fast": 0.2, "base": 0.4},
                    actions=None,
                    is_training=True,
                    fast_fail_closed=True,
                )

    def test_legacy_fast_channel_still_falls_back_to_outcome(self) -> None:
        with mock.patch(
            "fastwam.datasets.cfg_text.sample_cfg_channel",
            return_value="fast",
        ):
            text, channel = apply_ternary_cfg_suffix(
                "Fold the glasses.",
                outcome_flag=1,
                success_suffix=" Successful execution.",
                failure_suffix=" Failed execution.",
                channel_probs={"outcome": 0.4, "fast": 0.2, "base": 0.4},
                actions=None,
                is_training=True,
            )

        self.assertEqual(channel, "outcome")
        self.assertTrue(text.endswith(" Failed execution."))

    def test_failure_uses_fast_or_base_without_failure_suffix(self) -> None:
        failure_probs = {"outcome": 0.0, "fast": 0.5, "base": 0.5}
        with mock.patch(
            "fastwam.datasets.cfg_text.sample_cfg_channel", return_value="fast"
        ), mock.patch(
            "fastwam.datasets.cfg_text.format_fast_action_suffix",
            return_value=" Action codes: 1 2 3",
        ):
            text, channel = apply_ternary_cfg_suffix(
                "Fold the glasses.",
                outcome_flag=1,
                success_suffix=" Successful execution.",
                failure_suffix=None,
                channel_probs={"outcome": 0.4, "fast": 0.2, "base": 0.4},
                failure_channel_probs=failure_probs,
                actions=np.zeros((8, 22), dtype=np.float32),
                is_training=True,
                fast_fail_closed=True,
            )
        self.assertEqual(channel, "fast")
        self.assertIn("Action codes", text)

        with mock.patch(
            "fastwam.datasets.cfg_text.sample_cfg_channel", return_value="base"
        ):
            text, channel = apply_ternary_cfg_suffix(
                "Fold the glasses.",
                outcome_flag=1,
                success_suffix=" Successful execution.",
                failure_suffix=None,
                channel_probs={"outcome": 0.4, "fast": 0.2, "base": 0.4},
                failure_channel_probs=failure_probs,
                actions=np.zeros((8, 22), dtype=np.float32),
                is_training=True,
                fast_fail_closed=True,
            )
        self.assertEqual((text, channel), ("Fold the glasses.", "base"))

    def test_success_keeps_outcome_channel(self) -> None:
        with mock.patch(
            "fastwam.datasets.cfg_text.sample_cfg_channel", return_value="outcome"
        ):
            text, channel = apply_ternary_cfg_suffix(
                "Fold the glasses.",
                outcome_flag=0,
                success_suffix=" Successful execution.",
                failure_suffix=None,
                channel_probs={"outcome": 0.4, "fast": 0.2, "base": 0.4},
                failure_channel_probs={"outcome": 0.0, "fast": 0.5, "base": 0.5},
                actions=np.zeros((8, 22), dtype=np.float32),
                is_training=True,
                fast_fail_closed=True,
            )
        self.assertEqual(channel, "outcome")
        self.assertTrue(text.endswith(" Successful execution."))

    def test_primary_uses_outcome_and_base_without_fast(self) -> None:
        primary = {"outcome": 0.5, "fast": 0.0, "base": 0.5}
        aux = {"outcome": 0.4, "fast": 0.2, "base": 0.4}
        fail = {"outcome": 0.0, "fast": 0.2, "base": 0.4}
        probs = select_cfg_schedule_probs(
            cfg_schedule="primary",
            outcome_flag=0,
            channel_probs=aux,
            failure_channel_probs=fail,
            primary_channel_probs=primary,
        )
        assert probs is not None
        self.assertAlmostEqual(probs["outcome"], 0.5)
        self.assertAlmostEqual(probs["fast"], 0.0)
        self.assertAlmostEqual(probs["base"], 0.5)

        with mock.patch(
            "fastwam.datasets.cfg_text.sample_cfg_channel", return_value="outcome"
        ):
            text, channel = apply_ternary_cfg_suffix(
                "Hammer the nail.",
                outcome_flag=0,
                success_suffix=" Successful execution.",
                failure_suffix=None,
                channel_probs=aux,
                failure_channel_probs=fail,
                primary_channel_probs=primary,
                cfg_schedule="primary",
                actions=np.zeros((8, 22), dtype=np.float32),
                is_training=True,
                fast_fail_closed=True,
            )
        self.assertEqual(channel, "outcome")
        self.assertTrue(text.endswith(" Successful execution."))

    def test_primary_rejects_fast_probability(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot include FAST"):
            select_cfg_schedule_probs(
                cfg_schedule="primary",
                outcome_flag=0,
                channel_probs={"outcome": 0.4, "fast": 0.2, "base": 0.4},
                primary_channel_probs={"outcome": 0.4, "fast": 0.2, "base": 0.4},
            )

    def test_aux_schedules_keep_fast(self) -> None:
        aux = {"outcome": 0.4, "fast": 0.2, "base": 0.4}
        fail = {"outcome": 0.0, "fast": 0.2, "base": 0.4}
        success_probs = select_cfg_schedule_probs(
            cfg_schedule="aux_success",
            outcome_flag=0,
            channel_probs=aux,
            failure_channel_probs=fail,
            primary_channel_probs={"outcome": 0.5, "fast": 0.0, "base": 0.5},
        )
        fail_probs = select_cfg_schedule_probs(
            cfg_schedule="aux_failure",
            outcome_flag=1,
            channel_probs=aux,
            failure_channel_probs=fail,
            primary_channel_probs={"outcome": 0.5, "fast": 0.0, "base": 0.5},
        )
        assert success_probs is not None and fail_probs is not None
        self.assertAlmostEqual(success_probs["fast"], 0.2)
        self.assertAlmostEqual(fail_probs["fast"], 0.2 / 0.6)
        self.assertAlmostEqual(fail_probs["base"], 0.4 / 0.6)
        self.assertAlmostEqual(fail_probs["outcome"], 0.0)


if __name__ == "__main__":
    unittest.main()
