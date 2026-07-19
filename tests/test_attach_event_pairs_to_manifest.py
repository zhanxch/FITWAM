import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from fastwam.everobot_schema import (
    SCHEMA_VERSION,
    compute_manifest_hash,
    validate_manifest,
    with_manifest_hash,
)


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "everobot"
    / "attach_event_pairs_to_manifest.py"
)
SPEC = importlib.util.spec_from_file_location("attach_event_pairs", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
attach = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(attach)

HASH = "a" * 64


def make_sample(event_id: str, outcome: str, episode_index: int, split: str = "train"):
    return {
        "sample_type": "event",
        "sample_id": event_id,
        "event_id": event_id,
        "effector": "global",
        "dataset_id": "dataset",
        "episode_id": f"dataset:episode:{episode_index:06d}",
        "episode_index": episode_index,
        "round_id": "dataset:round:0",
        "collection_round": 0,
        "start_frame": 0,
        "end_frame": 16,
        "sample_stride": 1,
        "action_loss": "enabled" if outcome == "success" else "disabled",
        "episode_outcome": outcome,
        "event_outcome": "unknown",
        "split": split,
    }


def make_manifest(split: str = "train"):
    samples = [
        make_sample("success-0", "success", 0, split),
        make_sample("failure-0", "failure", 1, split),
        make_sample("unpaired-0", "success", 2, split),
    ]
    return with_manifest_hash(
        {
            "format": "EveRobotTrainManifest",
            "schema_version": SCHEMA_VERSION,
            "manifest_name": f"{split}-manifest",
            "frame_interval": "half_open",
            "selection": {"splits": [split]},
            "dataset_roots": {"dataset": "/data/dataset"},
            "source_round_ids": ["dataset:round:0"],
            "source_hashes": {
                "round_meta_sha256": "1" * 64,
                "episode_meta_sha256": "2" * 64,
                "event_meta_sha256": "3" * 64,
            },
            "num_samples": len(samples),
            "samples": samples,
        }
    )


def pair_row(pair_id="pair-0", success="success-0", failure="failure-0"):
    return {
        "pair_id": pair_id,
        "success_event_id": success,
        "failure_event_id": failure,
        "pair_weight": 0.75,
    }


def write_targets(
    path: Path,
    *,
    pair_ids=("pair-0",),
    success_ids=("success-0",),
    failure_ids=("failure-0",),
    splits=("train",),
    weights=(0.75,),
):
    count = len(pair_ids)
    if len(splits) == 1 and count > 1:
        splits = splits * count
    np.savez_compressed(
        path,
        pair_id=np.asarray(pair_ids, dtype="<U32"),
        success_event_id=np.asarray(success_ids, dtype="<U32"),
        failure_event_id=np.asarray(failure_ids, dtype="<U32"),
        split=np.asarray(splits, dtype="<U16"),
        pair_weight=np.asarray(weights, dtype=np.float32),
        z_plus=np.ones((count, 3), dtype=np.float32),
        z_minus=np.zeros((count, 3), dtype=np.float32),
        teacher_sha256=np.asarray([HASH] * count, dtype="<U64"),
    )


class AttachEventPairsTest(unittest.TestCase):
    def test_attaches_both_events_rehashes_and_keeps_targets_external(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target_path = root / "targets.npz"
            write_targets(target_path)
            original = make_manifest()
            before = copy.deepcopy(original)
            with attach.PairTargetStore(target_path) as targets:
                result = attach.attach_pairs(original, [pair_row()], targets)

            self.assertEqual(original, before)
            self.assertNotEqual(result["manifest_hash"], before["manifest_hash"])
            self.assertEqual(result["manifest_hash"], compute_manifest_hash(result))
            validate_manifest(result, strict=True)
            paired = {
                sample["event_id"]: sample
                for sample in result["samples"]
                if sample.get("pair_id")
            }
            self.assertEqual(set(paired), {"success-0", "failure-0"})
            for sample in paired.values():
                self.assertEqual(sample["pair_id"], "pair-0")
                self.assertAlmostEqual(sample["pair_weight"], 0.75)
                self.assertNotIn("z_plus", sample)
                self.assertNotIn("z_minus", sample)
            unpaired = next(
                sample
                for sample in result["samples"]
                if sample["event_id"] == "unpaired-0"
            )
            self.assertNotIn("pair_id", unpaired)
            self.assertNotIn("pair_weight", unpaired)

    def test_split_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target_path = Path(directory) / "targets.npz"
            write_targets(target_path, splits=("val",))
            with attach.PairTargetStore(target_path) as targets:
                with self.assertRaisesRegex(ValueError, "target split"):
                    attach.attach_pairs(make_manifest("train"), [pair_row()], targets)

    def test_failure_only_mode_does_not_require_success_event_in_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target_path = Path(directory) / "targets.npz"
            write_targets(target_path)
            manifest = make_manifest()
            manifest["samples"] = [
                sample
                for sample in manifest["samples"]
                if sample["event_id"] != "success-0"
            ]
            manifest["num_samples"] = len(manifest["samples"])
            manifest["manifest_hash"] = compute_manifest_hash(manifest)
            with attach.PairTargetStore(target_path) as targets:
                result = attach.attach_pairs(
                    manifest,
                    [pair_row()],
                    targets,
                    attach_side="failure",
                )

        failure = next(
            sample
            for sample in result["samples"]
            if sample["event_id"] == "failure-0"
        )
        self.assertEqual(failure["pair_id"], "pair-0")
        self.assertAlmostEqual(failure["pair_weight"], 0.75)
        self.assertFalse(
            any(
                sample.get("pair_id")
                for sample in result["samples"]
                if sample["event_id"] != "failure-0"
            )
        )

    def test_event_reuse_is_rejected_instead_of_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target_path = Path(directory) / "targets.npz"
            write_targets(
                target_path,
                pair_ids=("pair-0", "pair-1"),
                success_ids=("success-0", "success-0"),
                failure_ids=("failure-0", "unpaired-0"),
                weights=(0.75, 0.75),
            )
            rows = [
                pair_row(),
                pair_row("pair-1", "success-0", "unpaired-0"),
            ]
            with attach.PairTargetStore(target_path) as targets:
                with self.assertRaisesRegex(ValueError, "reused"):
                    attach.attach_pairs(make_manifest(), rows, targets)

    def test_missing_manifest_sample_and_missing_target_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target_path = root / "targets.npz"
            write_targets(target_path)
            incomplete = make_manifest()
            incomplete["samples"] = [
                sample
                for sample in incomplete["samples"]
                if sample["event_id"] != "failure-0"
            ]
            incomplete["num_samples"] = len(incomplete["samples"])
            incomplete["manifest_hash"] = compute_manifest_hash(incomplete)
            with attach.PairTargetStore(target_path) as targets:
                with self.assertRaisesRegex(ValueError, "partially represented"):
                    attach.attach_pairs(incomplete, [pair_row()], targets)

            other_target_path = root / "other_targets.npz"
            write_targets(
                other_target_path,
                pair_ids=("other-pair",),
                success_ids=("success-x",),
                failure_ids=("failure-x",),
            )
            with attach.PairTargetStore(other_target_path) as targets:
                with self.assertRaisesRegex(ValueError, "no exported target"):
                    attach.attach_pairs(make_manifest(), [pair_row()], targets)

    def test_cli_writes_atomically_and_refuses_existing_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = root / "manifest.json"
            ledger_path = root / "pairs.jsonl"
            target_path = root / "targets.npz"
            output_path = root / "attached.json"
            manifest_path.write_text(
                json.dumps(make_manifest()), encoding="utf-8"
            )
            ledger_path.write_text(
                json.dumps(pair_row()) + "\n", encoding="utf-8"
            )
            write_targets(target_path)

            exit_code = attach.main(
                [
                    "--manifest",
                    str(manifest_path),
                    "--pair-ledger",
                    str(ledger_path),
                    "--pair-targets",
                    str(target_path),
                    "--output",
                    str(output_path),
                    "--expected-teacher-sha256",
                    HASH,
                    "--attach-side",
                    "both",
                ]
            )
            self.assertEqual(exit_code, 0)
            output = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(output["manifest_hash"], compute_manifest_hash(output))
            with self.assertRaisesRegex(FileExistsError, "overwrite"):
                attach.main(
                    [
                        "--manifest",
                        str(manifest_path),
                        "--pair-ledger",
                        str(ledger_path),
                        "--pair-targets",
                        str(target_path),
                        "--output",
                        str(output_path),
                    ]
                )


if __name__ == "__main__":
    unittest.main()
