import argparse
import copy
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from fastwam.datasets.eve.pair_targets import PairTargetStore
from fastwam.everobot_schema import compute_manifest_hash, validate_manifest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "everobot"
    / "build_pair_shuffle_control.py"
)
SPEC = importlib.util.spec_from_file_location("build_pair_shuffle_control", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
shuffle = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = shuffle
SPEC.loader.exec_module(shuffle)


TEACHER_HASH = "a" * 64


def make_sample(
    index: int,
    *,
    task: str,
    split: str = "train",
    pair_weight: float | None = None,
) -> dict:
    failure = pair_weight is not None
    event_id = f"failure-{index}" if failure else f"success-primary-{index}"
    sample = {
        "sample_type": "event",
        "sample_id": event_id,
        "event_id": event_id,
        "effector": "global",
        "dataset_id": f"dataset-{task}",
        "dataset_root": f"/data/{task}",
        "episode_id": f"dataset-{task}:episode:{index:06d}",
        "episode_index": index,
        "round_id": f"dataset-{task}:round:0",
        "collection_round": 0,
        "start_frame": index * 20,
        "end_frame": index * 20 + 16,
        "core_start_frame": index * 20 + 2,
        "core_end_frame": index * 20 + 14,
        "valid_intervals": [[index * 20, index * 20 + 16]],
        "sample_stride": 1,
        "action_loss": "disabled" if failure else "enabled",
        "batch_role": "auxiliary" if failure else "primary",
        "episode_outcome": "failure" if failure else "success",
        "event_outcome": "failure" if failure else "success",
        "split": split,
        "task_name": task,
        "task": f"Perform {task}.",
    }
    if failure:
        sample["pair_id"] = f"pair-{index}"
        sample["pair_weight"] = pair_weight
    return sample


def make_manifest(specs: list[tuple[str, str]], weights: list[float]) -> dict:
    samples = [
        make_sample(index, task=task, split=split, pair_weight=weights[index])
        for index, (task, split) in enumerate(specs)
    ]
    samples.append(make_sample(100, task="unpaired"))
    round_ids = sorted({sample["round_id"] for sample in samples})
    manifest = {
        "format": "EveRobotTrainManifest",
        "schema_version": "0.2",
        "manifest_name": "formal-m",
        "frame_interval": "half_open",
        "selection": {"splits": sorted({split for _, split in specs})},
        "source_round_ids": round_ids,
        "source_hashes": {
            "round_meta_sha256": "1" * 64,
            "episode_meta_sha256": "2" * 64,
            "event_meta_sha256": "3" * 64,
        },
        "num_samples": len(samples),
        "samples": samples,
    }
    manifest["manifest_hash"] = compute_manifest_hash(manifest)
    return manifest


def make_arrays(
    specs: list[tuple[str, str]],
    weights: list[float],
    *,
    success_ids: list[str] | None = None,
) -> dict[str, np.ndarray]:
    count = len(specs)
    if success_ids is None:
        success_ids = [f"success-{index}" for index in range(count)]
    return {
        "pair_id": np.asarray([f"pair-{index}" for index in range(count)]),
        "success_event_id": np.asarray(success_ids),
        "failure_event_id": np.asarray([f"failure-{index}" for index in range(count)]),
        "split": np.asarray([split for _, split in specs]),
        "pair_weight": np.asarray(weights, dtype=np.float32),
        "z_plus": np.asarray(
            [[float(index), float(index) + 0.25] for index in range(count)],
            dtype=np.float32,
        ),
        "z_minus": np.asarray(
            [[-float(index), -float(index) - 0.5] for index in range(count)],
            dtype=np.float32,
        ),
        "teacher_sha256": np.asarray([TEACHER_HASH] * count),
    }


def write_sources(
    root: Path,
    specs: list[tuple[str, str]],
    weights: list[float],
    *,
    success_ids: list[str] | None = None,
) -> tuple[Path, Path, dict, dict[str, np.ndarray]]:
    manifest = make_manifest(specs, weights)
    arrays = make_arrays(specs, weights, success_ids=success_ids)
    manifest_path = root / "manifest.json"
    targets_path = root / "targets.npz"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    np.savez_compressed(targets_path, **arrays)
    return manifest_path, targets_path, manifest, arrays


def make_args(root: Path, manifest_path: Path, targets_path: Path, suffix: str):
    return argparse.Namespace(
        manifest=manifest_path,
        pair_targets=targets_path,
        output_manifest=root / f"shuffled-{suffix}.json",
        output_pair_targets=root / f"shuffled-targets-{suffix}.npz",
        proof_output=root / f"proof-{suffix}.json",
        shuffle_seed=1729,
        expected_teacher_sha256=TEACHER_HASH,
    )


class PairShuffleControlTest(unittest.TestCase):
    def test_derangement_never_swaps_candidates_within_one_episode(self) -> None:
        specs = [("water_plant", "train")] * 4
        weights = [1.0] * 4
        success_ids = [
            "rollout_ep000001_candidate_000",
            "rollout_ep000001_candidate_001",
            "rollout_ep000002_candidate_000",
            "rollout_ep000002_candidate_001",
        ]
        manifest = make_manifest(specs, weights)
        arrays = make_arrays(specs, weights, success_ids=success_ids)
        _, output, proof = shuffle.build_control(
            manifest,
            arrays,
            shuffle_seed=1729,
            source_manifest_sha256="1" * 64,
            source_targets_sha256="2" * 64,
        )

        for original, shuffled in zip(
            arrays["success_event_id"], output["success_event_id"], strict=True
        ):
            self.assertNotEqual(
                shuffle._event_episode_identity(str(original)),
                shuffle._event_episode_identity(str(shuffled)),
            )
        self.assertTrue(
            proof["invariant_checks"]["all_referenced_success_episodes_deranged"]
        )

    def test_is_deterministic_and_preserves_strict_invariants(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            specs = [("water_plant", "train")] * 4
            weights = [0.25, 0.5, 0.75, 1.0]
            manifest_path, targets_path, source_manifest, source_arrays = write_sources(
                root, specs, weights
            )
            args_a = make_args(root, manifest_path, targets_path, "a")
            args_b = make_args(root, manifest_path, targets_path, "b")

            report_a = shuffle.run(args_a)
            report_b = shuffle.run(args_b)

            self.assertEqual(report_a["proof_sha256"], report_b["proof_sha256"])
            self.assertEqual(
                args_a.output_pair_targets.read_bytes(),
                args_b.output_pair_targets.read_bytes(),
            )
            output_a = json.loads(args_a.output_manifest.read_text(encoding="utf-8"))
            output_b = json.loads(args_b.output_manifest.read_text(encoding="utf-8"))
            self.assertEqual(output_a, output_b)
            validate_manifest(output_a, strict=True, verify_hash=True)

            before = copy.deepcopy(source_manifest["samples"])
            after = copy.deepcopy(output_a["samples"])
            for sample in before:
                sample.pop("pair_id", None)
            for sample in after:
                sample.pop("pair_id", None)
            self.assertEqual(before, after)

            with np.load(args_a.output_pair_targets, allow_pickle=False) as output:
                self.assertTrue(
                    np.array_equal(source_arrays["pair_weight"], output["pair_weight"])
                )
                self.assertTrue(
                    np.array_equal(
                        source_arrays["failure_event_id"], output["failure_event_id"]
                    )
                )
                self.assertTrue(
                    np.array_equal(source_arrays["z_minus"], output["z_minus"])
                )
                self.assertTrue(
                    all(
                        original != shuffled
                        for original, shuffled in zip(
                            source_arrays["success_event_id"],
                            output["success_event_id"],
                            strict=True,
                        )
                    )
                )
                source_items = sorted(
                    (str(event_id), tuple(vector.tolist()))
                    for event_id, vector in zip(
                        source_arrays["success_event_id"],
                        source_arrays["z_plus"],
                        strict=True,
                    )
                )
                output_items = sorted(
                    (str(event_id), tuple(vector.tolist()))
                    for event_id, vector in zip(
                        output["success_event_id"], output["z_plus"], strict=True
                    )
                )
                self.assertEqual(source_items, output_items)

            with PairTargetStore(args_a.output_pair_targets) as targets:
                for sample in output_a["samples"][:4]:
                    target = targets.get(sample["pair_id"])
                    self.assertEqual(target.failure_event_id, sample["event_id"])

            proof = json.loads(args_a.proof_output.read_text(encoding="utf-8"))
            self.assertTrue(all(proof["invariant_checks"].values()))
            self.assertEqual(proof["source"]["referenced_pair_count"], 4)
            self.assertEqual(proof["source"]["target_row_count"], 4)
            self.assertEqual(
                proof["source"]["passthrough_unreferenced_target_count"], 0
            )
            self.assertEqual(len(proof["mapping"]), 4)
            self.assertEqual(
                proof["proof_sha256"],
                shuffle._sha256_json(
                    {
                        key: value
                        for key, value in proof.items()
                        if key != "proof_sha256"
                    }
                ),
            )
            self.assertEqual(
                proof["output"]["pair_targets_file_sha256"],
                shuffle._sha256_file(args_a.output_pair_targets),
            )

    def test_derangement_stays_inside_each_task_and_split_group(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            specs = [
                ("water_plant", "train"),
                ("water_plant", "train"),
                ("hammer_nail", "train"),
                ("hammer_nail", "train"),
                ("water_plant", "val"),
                ("water_plant", "val"),
            ]
            weights = [0.4] * len(specs)
            manifest_path, targets_path, _, _ = write_sources(root, specs, weights)
            args = make_args(root, manifest_path, targets_path, "groups")
            shuffle.run(args)
            proof = json.loads(args.proof_output.read_text(encoding="utf-8"))
            mapping_by_source = {row["source_pair_id"]: row for row in proof["mapping"]}
            for index, (task, split) in enumerate(specs):
                row = mapping_by_source[f"pair-{index}"]
                donor_index = int(row["donor_source_pair_id"].split("-")[-1])
                self.assertEqual(specs[donor_index], (task, split))
                self.assertNotEqual(donor_index, index)

    def test_duplicate_success_identities_derange_when_feasible(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            specs = [("water_plant", "train")] * 4
            weights = [0.5] * 4
            success_ids = ["success-a", "success-a", "success-b", "success-b"]
            manifest_path, targets_path, _, _ = write_sources(
                root, specs, weights, success_ids=success_ids
            )
            args = make_args(root, manifest_path, targets_path, "duplicates")
            shuffle.run(args)
            with np.load(args.output_pair_targets, allow_pickle=False) as output:
                self.assertEqual(
                    sorted(str(value) for value in output["success_event_id"]),
                    sorted(success_ids),
                )
                self.assertTrue(
                    all(
                        original != shuffled
                        for original, shuffled in zip(
                            success_ids, output["success_event_id"], strict=True
                        )
                    )
                )

    def test_singleton_and_non_derangeable_identity_groups_fail_closed(self) -> None:
        cases = [
            (["only"], "only one pair"),
            (["same", "same", "different"], "Hall's condition"),
        ]
        for case_index, (success_ids, message) in enumerate(cases):
            with self.subTest(success_ids=success_ids):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    specs = [("water_plant", "train")] * len(success_ids)
                    weights = [0.5] * len(success_ids)
                    manifest_path, targets_path, _, _ = write_sources(
                        root, specs, weights, success_ids=success_ids
                    )
                    args = make_args(
                        root, manifest_path, targets_path, f"closed-{case_index}"
                    )
                    with self.assertRaisesRegex(ValueError, message):
                        shuffle.run(args)
                    self.assertFalse(args.output_manifest.exists())
                    self.assertFalse(args.output_pair_targets.exists())
                    self.assertFalse(args.proof_output.exists())

    def test_rejects_unreferenced_target_and_non_failure_attachment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            specs = [("water_plant", "train")] * 2
            weights = [0.5, 0.75]
            manifest_path, targets_path, manifest, _ = write_sources(
                root, specs, weights
            )
            manifest["samples"][1]["pair_weight"] = 0.0
            manifest["samples"][1].pop("pair_id")
            manifest["manifest_hash"] = compute_manifest_hash(manifest)
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            args = make_args(root, manifest_path, targets_path, "orphan")
            with self.assertRaisesRegex(ValueError, "train target rows unreferenced"):
                shuffle.run(args)

            manifest_path, targets_path, manifest, _ = write_sources(
                root, specs, weights
            )
            manifest["samples"][0]["episode_outcome"] = "success"
            manifest["manifest_hash"] = compute_manifest_hash(manifest)
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            args = make_args(root, manifest_path, targets_path, "success-side")
            with self.assertRaisesRegex(ValueError, "not a failure event"):
                shuffle.run(args)

    def test_unreferenced_validation_targets_are_preserved_as_passthrough(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            specs = [
                ("water_plant", "train"),
                ("water_plant", "train"),
                ("water_plant", "val"),
            ]
            weights = [0.5, 0.75, 1.0]
            manifest_path, targets_path, manifest, arrays = write_sources(
                root, specs, weights
            )
            manifest["samples"] = [
                sample
                for sample in manifest["samples"]
                if sample.get("pair_id") != "pair-2"
            ]
            manifest["num_samples"] = len(manifest["samples"])
            manifest["source_round_ids"] = sorted(
                {sample["round_id"] for sample in manifest["samples"]}
            )
            manifest["manifest_hash"] = compute_manifest_hash(manifest)
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            args = make_args(root, manifest_path, targets_path, "val-passthrough")
            shuffle.run(args)

            with np.load(args.output_pair_targets, allow_pickle=False) as output:
                self.assertEqual(str(output["pair_id"][2]), "pair-2")
                self.assertEqual(
                    str(output["success_event_id"][2]),
                    str(arrays["success_event_id"][2]),
                )
                np.testing.assert_array_equal(output["z_plus"][2], arrays["z_plus"][2])
                np.testing.assert_array_equal(
                    output["z_minus"][2], arrays["z_minus"][2]
                )
            proof = json.loads(args.proof_output.read_text(encoding="utf-8"))
            self.assertEqual(
                proof["source"]["passthrough_unreferenced_target_count"], 1
            )

    def test_refuses_overwrite_before_publishing_any_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            specs = [("water_plant", "train")] * 2
            weights = [0.5, 0.75]
            manifest_path, targets_path, _, _ = write_sources(root, specs, weights)
            args = make_args(root, manifest_path, targets_path, "existing")
            args.proof_output.write_text("occupied", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "overwrite"):
                shuffle.run(args)
            self.assertFalse(args.output_manifest.exists())
            self.assertFalse(args.output_pair_targets.exists())
            self.assertEqual(args.proof_output.read_text(encoding="utf-8"), "occupied")


if __name__ == "__main__":
    unittest.main()
