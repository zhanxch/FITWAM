import copy
import math
import unittest
from pathlib import Path

from fastwam.everobot_schema import (
    SCHEMA_VERSION,
    compute_manifest_hash,
    resolve_manifest_dataset_root,
    resolve_dataset_root,
    resolve_dataset_roots,
    validate_manifest,
    with_manifest_hash,
)


def make_manifest():
    manifest = {
        "format": "EveRobotTrainManifest",
        "schema_version": SCHEMA_VERSION,
        "manifest_name": "round-1-train",
        "frame_interval": "half_open",
        "selection": {"include_outcomes": ["success", "failure"]},
        "eve_root": "/machine-a/eve",
        "dataset_roots": {
            "base": "/machine-a/data/base",
            "round1": "/machine-a/data/round1",
        },
        "num_samples": 2,
        "source_round_ids": ["base:round:-1", "round1:round:1"],
        "source_hashes": {
            "round_meta_sha256": "1" * 64,
            "episode_meta_sha256": "2" * 64,
            "event_meta_sha256": "3" * 64,
        },
        "samples": [
            {
                "sample_type": "episode",
                "sample_id": "base_ep000000",
                "dataset_id": "base",
                "dataset_root": "/machine-a/data/base",
                "episode_id": "base:episode:000000",
                "episode_index": 0,
                "round_id": "base:round:-1",
                "collection_round": -1,
                "start_frame": 0,
                "end_frame": 24,
                "sample_stride": 2,
                "action_loss": "enabled",
            },
            {
                "sample_type": "event",
                "sample_id": "round1_ep000003_failure",
                "event_id": "round1_ep000003_failure",
                "effector": "global",
                "dataset_id": "round1",
                "dataset_root": "/machine-a/data/round1",
                "episode_id": "round1:episode:000003",
                "episode_index": 3,
                "round_id": "round1:round:1",
                "collection_round": 1,
                "start_frame": 4,
                "end_frame": 20,
                "sample_stride": 1,
                "action_loss": "disabled",
                "action_loss_window": [8, 16],
            },
        ],
    }
    return with_manifest_hash(manifest)


class EveRobotSchemaTest(unittest.TestCase):
    def test_legacy_v01_manifest_remains_readable(self):
        manifest = {
            "format": "EveRobotTrainManifest",
            "schema_version": "0.1",
            "dataset_roots": {"base": "/legacy/base"},
            "samples": [
                {
                    "sample_type": "episode",
                    "dataset_id": "base",
                    "dataset_root": "/legacy/base",
                    "episode_index": 0,
                    "start_frame": 0,
                    "end_frame": 12,
                }
            ],
        }

        self.assertIs(validate_manifest(manifest), manifest)
        manifest["samples"][0].pop("start_frame")
        manifest["samples"][0].pop("end_frame")
        self.assertIs(validate_manifest(manifest), manifest)

    def test_hash_is_stable_under_runtime_relocation(self):
        original = make_manifest()
        relocated = copy.deepcopy(original)
        relocated["eve_root"] = "/machine-b/work/eve"
        relocated["dataset_roots"] = {
            "base": "/machine-b/datasets/base",
            "round1": "/machine-b/datasets/round1",
        }
        relocated["samples"][0]["dataset_root"] = "/machine-b/datasets/base"
        relocated["samples"][1]["dataset_root"] = "/machine-b/datasets/round1"

        self.assertEqual(
            compute_manifest_hash(original), compute_manifest_hash(relocated)
        )
        validate_manifest(relocated)

    def test_invalid_half_open_intervals_are_rejected(self):
        for start, end in ((5, 5), (8, 3), (-1, 3)):
            with self.subTest(start=start, end=end):
                manifest = make_manifest()
                manifest["samples"][0]["start_frame"] = start
                manifest["samples"][0]["end_frame"] = end
                manifest["manifest_hash"] = compute_manifest_hash(manifest)
                with self.assertRaisesRegex(ValueError, "half-open interval"):
                    validate_manifest(manifest)

    def test_duplicate_sample_ids_are_rejected(self):
        manifest = make_manifest()
        manifest["samples"][1]["sample_id"] = manifest["samples"][0]["sample_id"]
        manifest["manifest_hash"] = compute_manifest_hash(manifest)

        with self.assertRaisesRegex(ValueError, "duplicate sample_id"):
            validate_manifest(manifest)

    def test_manifest_hash_tamper_is_rejected(self):
        manifest = make_manifest()
        manifest["samples"][0]["end_frame"] = 23

        with self.assertRaisesRegex(ValueError, "manifest_hash mismatch"):
            validate_manifest(manifest)

    def test_missing_schema_version_cannot_fall_back_to_legacy(self):
        manifest = make_manifest()
        manifest.pop("schema_version")

        with self.assertRaisesRegex(ValueError, "schema_version is required"):
            validate_manifest(manifest)

    def test_required_references_stride_and_action_loss_are_strict(self):
        invalid_changes = (
            ("episode_id", None),
            ("round_id", ""),
            ("collection_round", None),
            ("sample_stride", 0),
            ("action_loss", "automatic"),
        )
        for field, value in invalid_changes:
            with self.subTest(field=field, value=value):
                manifest = make_manifest()
                if value is None:
                    manifest["samples"][0].pop(field)
                else:
                    manifest["samples"][0][field] = value
                manifest["manifest_hash"] = compute_manifest_hash(manifest)
                with self.assertRaises(ValueError):
                    validate_manifest(manifest)

    def test_optional_soft_event_fields_are_valid(self):
        manifest = make_manifest()
        sample = manifest["samples"][1]
        sample.update(
            {
                "event_weight": 0.75,
                "core_start_frame": 6,
                "core_end_frame": 18,
                "episode_outcome": "failure",
                "event_outcome": "unknown",
            }
        )
        manifest["manifest_hash"] = compute_manifest_hash(manifest)

        self.assertIs(validate_manifest(manifest), manifest)

    def test_event_weight_must_be_finite_and_bounded(self):
        for value in (-0.01, 1.01, math.nan, math.inf, True, "0.5"):
            with self.subTest(value=value):
                manifest = make_manifest()
                manifest["samples"][1]["event_weight"] = value
                with self.assertRaisesRegex(ValueError, "event_weight"):
                    validate_manifest(manifest)

        for value in (0.0, 1.0):
            with self.subTest(value=value):
                manifest = make_manifest()
                manifest["samples"][1]["event_weight"] = value
                manifest["manifest_hash"] = compute_manifest_hash(manifest)
                self.assertIs(validate_manifest(manifest), manifest)

    def test_candidate_confidence_fields_must_be_finite_and_bounded(self):
        for field in ("absolute_confidence", "episode_sampling_weight"):
            for value in (-0.1, 1.1, float("nan"), True):
                with self.subTest(field=field, value=value):
                    manifest = make_manifest()
                    manifest["samples"][1][field] = value
                    with self.assertRaisesRegex(ValueError, field):
                        validate_manifest(manifest)

            for value in (0.0, 1.0):
                with self.subTest(field=field, value=value):
                    manifest = make_manifest()
                    manifest["samples"][1][field] = value
                    manifest["manifest_hash"] = compute_manifest_hash(manifest)
                    self.assertIs(validate_manifest(manifest), manifest)

    def test_pair_fields_require_explicit_valid_pair(self):
        for value in (-0.01, 1.01, math.nan, math.inf, True, "0.5"):
            with self.subTest(value=value):
                manifest = make_manifest()
                manifest["samples"][1]["pair_weight"] = value
                with self.assertRaisesRegex(ValueError, "pair_weight"):
                    validate_manifest(manifest)

        for value in ("", "   ", 7, False):
            with self.subTest(value=value):
                manifest = make_manifest()
                manifest["samples"][1]["pair_id"] = value
                with self.assertRaisesRegex(ValueError, "pair_id"):
                    validate_manifest(manifest)

        manifest = make_manifest()
        manifest["samples"][1]["pair_weight"] = 0.5
        with self.assertRaisesRegex(ValueError, "pair_id"):
            validate_manifest(manifest)

        manifest["samples"][1]["pair_id"] = "pair-1"
        manifest["manifest_hash"] = compute_manifest_hash(manifest)
        self.assertIs(validate_manifest(manifest), manifest)

    def test_core_interval_must_be_complete_and_inside_sample(self):
        invalid_updates = (
            {"core_start_frame": 6},
            {"core_end_frame": 18},
            {"core_start_frame": 3, "core_end_frame": 18},
            {"core_start_frame": 6, "core_end_frame": 21},
            {"core_start_frame": 8, "core_end_frame": 8},
        )
        for updates in invalid_updates:
            with self.subTest(updates=updates):
                manifest = make_manifest()
                manifest["samples"][1].update(updates)
                manifest["manifest_hash"] = compute_manifest_hash(manifest)
                with self.assertRaisesRegex(ValueError, "core"):
                    validate_manifest(manifest)

    def test_optional_outcomes_use_declared_vocabularies(self):
        invalid_updates = (
            {"episode_outcome": "unknown"},
            {"event_outcome": "partial"},
        )
        for updates in invalid_updates:
            with self.subTest(updates=updates):
                manifest = make_manifest()
                manifest["samples"][1].update(updates)
                manifest["manifest_hash"] = compute_manifest_hash(manifest)
                with self.assertRaisesRegex(ValueError, "outcome"):
                    validate_manifest(manifest)

    def test_explicit_dataset_root_overrides_take_precedence(self):
        manifest = make_manifest()
        override = Path("~/relocated/base").expanduser()
        roots = resolve_dataset_roots(manifest, {"base": override})

        self.assertEqual(roots["base"], str(override.resolve()))
        self.assertEqual(
            roots["round1"], str(Path("/machine-a/data/round1").resolve())
        )
        self.assertEqual(
            resolve_dataset_root(manifest, manifest["samples"][0], {"base": override}),
            str(override.resolve()),
        )
        self.assertEqual(
            resolve_manifest_dataset_root(
                manifest, manifest["samples"][0], {"base": override}
            ),
            str(override.resolve()),
        )


if __name__ == "__main__":
    unittest.main()
