from __future__ import annotations

import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from scripts.everobot import validate_recoverability_event_pairs as validator


class RecoverabilityPairFixture:
    def __init__(self, root: Path, *, t: int = 48) -> None:
        self.root = root
        self.pair_dir = root / "event_pairs" / "ep000010_frontier_00_f0072"
        self.pair_dir.mkdir(parents=True)
        self.t = t
        self.start = max(0, t - 9)
        self.end = t + 24
        self.frames = np.arange(self.start, self.end, dtype=np.int32)
        self.success_actions = np.zeros((33, 22), dtype=np.float32)
        self.failure_actions = np.zeros((33, 22), dtype=np.float32)
        # Preserve a byte-identical factual prefix, then make the branches
        # observably different in the event core/continuation.
        self.success_actions[9:, 0] = 1.0
        self.failure_actions[9:, 0] = -1.0
        self.success_states = np.zeros((33, 23), dtype=np.float32)
        self.failure_states = np.zeros((33, 23), dtype=np.float32)
        self.success_states[9:, 0] = 2.0
        self.failure_states[9:, 0] = -2.0
        np.savez_compressed(
            self.pair_dir / "success_event.npz",
            frame_indices=self.frames,
            actions=self.success_actions,
            states=self.success_states,
        )
        np.savez_compressed(
            self.pair_dir / "failure_event.npz",
            frame_indices=self.frames,
            actions=self.failure_actions,
            states=self.failure_states,
        )
        self.success_descriptor = {
            "format": "FoldGlassesCounterfactualSuccessEvent",
            "version": "1.0",
            "seed": 10001,
            "source_failure_episode_index": 10,
            "frame_start": self.start,
            "frame_end_exclusive": self.end,
            "num_frames": 33,
            "exact_counterfactual_prefix_frame": self.t,
            "frontier_first_zero_frame": self.t + 24,
            "outcome": "success",
            "arrays": str((self.pair_dir / "success_event.npz").resolve()),
            "successful_replicate_index": 0,
            "action_loss": "enabled",
            "action_loss_window": [self.t, self.t + 24],
            "deterministic_rerun_succeeded": True,
        }
        self.failure_descriptor = {
            "format": "FoldGlassesFactualFailureEvent",
            "version": "1.0",
            "seed": 10001,
            "source_failure_episode_index": 10,
            "frame_start": self.start,
            "frame_end_exclusive": self.end,
            "num_frames": 33,
            "exact_counterfactual_prefix_frame": self.t,
            "frontier_first_zero_frame": self.t + 24,
            "outcome": "failure",
            "arrays": str((self.pair_dir / "failure_event.npz").resolve()),
            "action_loss": "disabled",
        }
        (self.pair_dir / "success_event.json").write_text(
            json.dumps(self.success_descriptor), encoding="utf-8"
        )
        (self.pair_dir / "failure_event.json").write_text(
            json.dumps(self.failure_descriptor), encoding="utf-8"
        )
        self.pair = {
            "format": "FoldGlassesRecoverabilityEventPair",
            "version": "1.0",
            "status": "complete",
            "pair_id": "seed10001_ep10_frontier_00_f0072",
            "seed": 10001,
            "seed_classification": "mixed",
            "source_failure_episode_index": 10,
            "source_repeat": 2,
            "frontier": {
                "frontier_id": "frontier_00_f0072",
                "t_frame": self.t,
                "t_plus_24_frame": self.t + 24,
                "last_recoverable_frame": self.t,
                "last_recoverable_success_count": 2,
                "first_zero_frame": self.t + 24,
                "failure_frame": self.t + 24,
                "event_start": self.start,
                "event_end_exclusive": self.end,
                "event_window": [self.start, self.end],
                "event_pre_frames": 9,
                "event_post_frames": 24,
                "core_event_start": self.t,
                "core_event_end": self.t + 24,
                "snapshot_frame": self.t,
                "pass_m": 4,
                "prefix_is_mixed": True,
            },
            "factual_failure_event": str(
                (self.pair_dir / "failure_event.json").resolve()
            ),
            "counterfactual_success_event": str(
                (self.pair_dir / "success_event.json").resolve()
            ),
            "successful_replicate_index": 0,
            "training_eligible": True,
            "evaluation_only": False,
            "run_signature": {"checkpoint": "unit-test"},
        }
        self.pair_path = self.pair_dir / "pair.json"
        self.write()

    def write(self) -> None:
        self.pair_path.write_text(json.dumps(self.pair), encoding="utf-8")
        (self.pair_dir / "success_event.json").write_text(
            json.dumps(self.success_descriptor), encoding="utf-8"
        )
        (self.pair_dir / "failure_event.json").write_text(
            json.dumps(self.failure_descriptor), encoding="utf-8"
        )


class ValidateRecoverabilityEventPairsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = TemporaryDirectory()
        self.fixture = RecoverabilityPairFixture(Path(self.temporary.name))

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_valid_pair_has_exact_interval_and_safe_roles(self) -> None:
        report = validator.validate_pair(self.fixture.pair_path)
        self.assertEqual(report["status"], "valid")
        self.assertTrue(report["trainable"])
        self.assertEqual(report["event_interval"], [39, 72])
        self.assertEqual(report["num_frames"], 33)
        self.assertEqual(report["success_count_at_t"], 2)
        self.assertEqual(report["artifacts"]["success"]["action_loss"], "enabled")
        self.assertEqual(report["artifacts"]["failure"]["action_loss"], "disabled")

    def test_all_success_prefix_is_accepted(self) -> None:
        self.fixture.pair["frontier"]["last_recoverable_success_count"] = 4
        self.fixture.pair["frontier"]["prefix_is_mixed"] = False
        self.fixture.write()
        report = validator.inspect_pair(self.fixture.pair_path)
        self.assertEqual(report["status"], "valid")

    def test_all_failure_seed_is_accepted_but_evaluation_only_is_not(self) -> None:
        self.fixture.pair["seed_classification"] = "all_failure"
        self.fixture.write()
        report = validator.inspect_pair(self.fixture.pair_path)
        self.assertEqual(report["status"], "valid")

        self.fixture.pair["evaluation_only"] = True
        self.fixture.write()
        report = validator.inspect_pair(self.fixture.pair_path)
        self.assertEqual(report["status"], "rejected")
        self.assertIn("evaluation_only", report["errors"][0])

    def test_non_adjacent_zero_is_rejected(self) -> None:
        self.fixture.pair["frontier"]["first_zero_frame"] = self.fixture.t + 48
        self.fixture.write()
        report = validator.inspect_pair(self.fixture.pair_path)
        self.assertEqual(report["status"], "rejected")
        self.assertIn("first_zero_frame", report["errors"][0])

    def test_widened_interval_is_rejected(self) -> None:
        self.fixture.pair["frontier"]["event_start"] = self.fixture.t - 33
        self.fixture.pair["frontier"]["event_window"] = [self.fixture.t - 33, self.fixture.end]
        self.fixture.write()
        report = validator.inspect_pair(self.fixture.pair_path)
        self.assertEqual(report["status"], "rejected")
        self.assertIn("event_window", report["errors"][0])

    def test_early_clamped_window_is_reported_but_not_trainable(self) -> None:
        # t=0 gives only 24 observations in the requested half-open interval.
        early = RecoverabilityPairFixture(Path(self.temporary.name) / "early", t=0)
        report = validator.inspect_pair(early.pair_path)
        self.assertEqual(report["status"], "rejected")
        self.assertIn("33 frames", report["errors"][0])

    def test_prefix_action_mismatch_is_rejected(self) -> None:
        path = self.fixture.pair_dir / "failure_event.npz"
        with np.load(path) as loaded:
            arrays = {key: np.asarray(loaded[key]) for key in loaded.files}
        arrays["actions"][0, 0] = 1.0
        np.savez_compressed(path, **arrays)
        report = validator.inspect_pair(self.fixture.pair_path)
        self.assertEqual(report["status"], "rejected")
        self.assertIn("prefix actions", report["errors"][0])

    def test_prefix_state_mismatch_is_rejected(self) -> None:
        path = self.fixture.pair_dir / "failure_event.npz"
        with np.load(path) as loaded:
            arrays = {key: np.asarray(loaded[key]) for key in loaded.files}
        arrays["states"][8, 0] = 1.0
        np.savez_compressed(path, **arrays)
        report = validator.inspect_pair(self.fixture.pair_path)
        self.assertEqual(report["status"], "rejected")
        self.assertIn("prefix states", report["errors"][0])

    def test_failure_action_enabled_is_rejected(self) -> None:
        self.fixture.failure_descriptor["action_loss"] = "enabled"
        self.fixture.write()
        report = validator.inspect_pair(self.fixture.pair_path)
        self.assertEqual(report["status"], "rejected")
        self.assertIn("action_loss", report["errors"][0])

    def test_failure_action_window_is_rejected(self) -> None:
        self.fixture.failure_descriptor["action_loss_window"] = [self.fixture.t, self.fixture.t + 24]
        self.fixture.write()
        report = validator.inspect_pair(self.fixture.pair_path)
        self.assertEqual(report["status"], "rejected")
        self.assertIn("action_loss_window", report["errors"][0])

    def test_success_action_window_must_be_core(self) -> None:
        self.fixture.success_descriptor["action_loss_window"] = [self.fixture.t - 1, self.fixture.t + 24]
        self.fixture.write()
        report = validator.inspect_pair(self.fixture.pair_path)
        self.assertEqual(report["status"], "rejected")
        self.assertIn("action_loss_window", report["errors"][0])

    def test_conflicting_snapshot_hashes_are_rejected(self) -> None:
        self.fixture.success_descriptor["snapshot_hash"] = "a" * 64
        self.fixture.failure_descriptor["snapshot_hash"] = "b" * 64
        self.fixture.write()
        report = validator.inspect_pair(self.fixture.pair_path)
        self.assertEqual(report["status"], "rejected")
        self.assertIn("snapshot_hash mismatch", report["errors"][0])

    def test_strict_mode_rejects_legacy_missing_action_loss(self) -> None:
        self.fixture.success_descriptor.pop("action_loss")
        self.fixture.failure_descriptor.pop("action_loss")
        self.fixture.write()
        report = validator.inspect_pair(
            self.fixture.pair_path, require_explicit_action_loss=True
        )
        self.assertEqual(report["status"], "rejected")
        self.assertIn("explicit action_loss", report["errors"][0])

    def test_training_api_is_strict_by_default_but_inspect_is_legacy_compatible(self) -> None:
        self.fixture.success_descriptor.pop("action_loss")
        self.fixture.success_descriptor.pop("action_loss_window")
        self.fixture.failure_descriptor.pop("action_loss")
        self.fixture.write()

        audit = validator.inspect_pair(self.fixture.pair_path)
        self.assertEqual(audit["status"], "valid")
        self.assertTrue(any("lacks explicit action_loss" in warning for warning in audit["warnings"]))

        strict = validator.inspect_pair(
            self.fixture.pair_path,
            require_explicit_action_loss=True,
            require_action_loss_window=True,
        )
        self.assertEqual(strict["status"], "rejected")
        self.assertIn("explicit action_loss", strict["errors"][0])

    def test_training_api_rejects_missing_success_core_window(self) -> None:
        self.fixture.success_descriptor.pop("action_loss_window")
        self.fixture.write()
        with self.assertRaises(validator.PairValidationError):
            validator.validate_pair(self.fixture.pair_path)

    def test_frame_ids_must_match_exact_event_interval(self) -> None:
        path = self.fixture.pair_dir / "success_event.npz"
        with np.load(path) as loaded:
            arrays = {key: np.asarray(loaded[key]) for key in loaded.files}
        arrays["frame_indices"][0] += 1
        np.savez_compressed(path, **arrays)
        report = validator.inspect_pair(self.fixture.pair_path)
        self.assertEqual(report["status"], "rejected")
        self.assertIn("frame_indices", report["errors"][0])


if __name__ == "__main__":
    unittest.main()
