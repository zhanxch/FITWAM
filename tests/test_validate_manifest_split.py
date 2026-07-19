from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

from fastwam.everobot_schema import SCHEMA_VERSION, with_manifest_hash


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "everobot"
    / "validate_manifest_split.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "_test_validate_manifest_split", MODULE_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


audit = _load_module()


def make_sample(
    *,
    dataset_id: str,
    episode_index: int,
    outcome: str,
    action_loss: str,
    sample_id: str | None = None,
) -> dict[str, object]:
    episode_id = f"{dataset_id}:episode:{episode_index:06d}"
    return {
        "sample_type": "episode",
        "sample_id": sample_id or f"{dataset_id}_ep{episode_index:06d}",
        "dataset_id": dataset_id,
        "dataset_root": f"/data/{dataset_id}",
        "episode_id": episode_id,
        "episode_index": episode_index,
        "round_id": f"{dataset_id}:round:0",
        "collection_round": 0,
        "start_frame": 0,
        "end_frame": 32,
        "sample_stride": 1,
        "episode_outcome": outcome,
        "action_loss": action_loss,
        "sample_role": f"{outcome}_episode",
    }


def make_manifest(name: str, samples: list[dict[str, object]]) -> dict[str, object]:
    dataset_roots = {
        str(sample["dataset_id"]): str(sample["dataset_root"])
        for sample in samples
    }
    round_ids = sorted({str(sample["round_id"]) for sample in samples})
    return with_manifest_hash(
        {
            "format": "EveRobotTrainManifest",
            "schema_version": SCHEMA_VERSION,
            "manifest_name": name,
            "frame_interval": "half_open",
            "selection": {},
            "dataset_roots": dataset_roots,
            "source_round_ids": round_ids,
            "source_hashes": {
                "round_meta_sha256": "1" * 64,
                "episode_meta_sha256": "2" * 64,
                "event_meta_sha256": "3" * 64,
            },
            "num_samples": len(samples),
            "samples": samples,
        }
    )


def write_manifest(path: Path, manifest: dict[str, object]) -> None:
    path.write_text(
        json.dumps(manifest, ensure_ascii=True) + "\n", encoding="utf-8"
    )


class ValidateManifestSplitTest(unittest.TestCase):
    def test_clean_split_reports_counts_roles_and_hash(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_path = root / "train.json"
            val_path = root / "val.json"
            write_manifest(
                train_path,
                make_manifest(
                    "train",
                    [
                        make_sample(
                            dataset_id="train",
                            episode_index=0,
                            outcome="success",
                            action_loss="enabled",
                        ),
                        make_sample(
                            dataset_id="train",
                            episode_index=1,
                            outcome="failure",
                            action_loss="disabled",
                        ),
                    ],
                ),
            )
            write_manifest(
                val_path,
                make_manifest(
                    "val",
                    [
                        make_sample(
                            dataset_id="val",
                            episode_index=0,
                            outcome="success",
                            action_loss="enabled",
                        )
                    ],
                ),
            )

            report = audit.audit_manifest_splits(
                {"train": [train_path], "val": [val_path]}
            )

        self.assertEqual(report["status"], "ok")
        self.assertFalse(report["cross_split_overlap"]["has_episode_leakage"])
        train = next(
            item for item in report["manifests"] if item["split"] == "train"
        )
        self.assertTrue(train["manifest_hash"]["matches"])
        self.assertEqual(train["sample_count"]["actual"], 2)
        self.assertEqual(
            train["action_loss_counts"], {"disabled": 1, "enabled": 1}
        )
        self.assertEqual(
            train["normalized_batch_role_counts"],
            {"auxiliary": 1, "primary": 1},
        )

    def test_episode_id_overlap_fails_even_when_dataset_identity_differs(self) -> None:
        train_sample = make_sample(
            dataset_id="train",
            episode_index=0,
            outcome="success",
            action_loss="enabled",
        )
        val_sample = make_sample(
            dataset_id="val",
            episode_index=9,
            outcome="success",
            action_loss="enabled",
        )
        val_sample["episode_id"] = train_sample["episode_id"]
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_path = root / "train.json"
            val_path = root / "val.json"
            write_manifest(train_path, make_manifest("train", [train_sample]))
            write_manifest(val_path, make_manifest("val", [val_sample]))
            report = audit.audit_manifest_splits(
                {"train": [train_path], "val": [val_path]}
            )

        overlaps = report["cross_split_overlap"]["episode_id"]
        self.assertEqual(overlaps[0]["count"], 1)
        self.assertEqual(overlaps[0]["examples"], [train_sample["episode_id"]])
        self.assertEqual(report["status"], "error")

    def test_dataset_episode_overlap_fails_even_when_episode_id_differs(self) -> None:
        train_sample = make_sample(
            dataset_id="shared",
            episode_index=4,
            outcome="success",
            action_loss="enabled",
        )
        val_sample = dict(train_sample)
        val_sample["sample_id"] = "different-sample"
        val_sample["episode_id"] = "different-episode-id"
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_path = root / "train.json"
            val_path = root / "val.json"
            write_manifest(train_path, make_manifest("train", [train_sample]))
            write_manifest(val_path, make_manifest("val", [val_sample]))
            report = audit.audit_manifest_splits(
                {"train": [train_path], "val": [val_path]}
            )

        overlaps = report["cross_split_overlap"]["dataset_episode"]
        self.assertEqual(overlaps[0]["count"], 1)
        self.assertEqual(overlaps[0]["examples"], [["shared", 4]])

    def test_multiple_manifests_per_split_are_aggregated(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths: list[Path] = []
            for index in range(3):
                path = root / f"manifest-{index}.json"
                split = "train" if index < 2 else "val"
                write_manifest(
                    path,
                    make_manifest(
                        f"{split}-{index}",
                        [
                            make_sample(
                                dataset_id=split,
                                episode_index=index,
                                outcome="success",
                                action_loss="enabled",
                            )
                        ],
                    ),
                )
                paths.append(path)
            report = audit.audit_manifest_splits(
                {"train": paths[:2], "val": paths[2:]}
            )

        self.assertEqual(report["manifest_count"], 3)
        self.assertEqual(report["splits"]["train"]["manifest_count"], 2)
        self.assertEqual(report["splits"]["train"]["episode_id_count"], 2)

    def test_manifest_hash_and_declared_count_are_audited(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_path = root / "train.json"
            val_path = root / "val.json"
            train = make_manifest(
                "train",
                [
                    make_sample(
                        dataset_id="train",
                        episode_index=0,
                        outcome="success",
                        action_loss="enabled",
                    )
                ],
            )
            train["num_samples"] = 7
            write_manifest(train_path, train)
            write_manifest(
                val_path,
                make_manifest(
                    "val",
                    [
                        make_sample(
                            dataset_id="val",
                            episode_index=0,
                            outcome="success",
                            action_loss="enabled",
                        )
                    ],
                ),
            )
            report = audit.audit_manifest_splits(
                {"train": [train_path], "val": [val_path]}
            )

        train_report = next(
            item for item in report["manifests"] if item["split"] == "train"
        )
        self.assertFalse(train_report["manifest_hash"]["matches"])
        self.assertFalse(train_report["sample_count"]["matches"])
        self.assertFalse(train_report["schema_valid"])

    def test_nonfinite_manifest_value_is_reported_without_crashing(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_path = root / "train.json"
            val_path = root / "val.json"
            train = make_manifest(
                "train",
                [
                    make_sample(
                        dataset_id="train",
                        episode_index=0,
                        outcome="success",
                        action_loss="enabled",
                    )
                ],
            )
            train["samples"][0]["event_weight"] = float("nan")
            write_manifest(train_path, train)
            write_manifest(
                val_path,
                make_manifest(
                    "val",
                    [
                        make_sample(
                            dataset_id="val",
                            episode_index=0,
                            outcome="success",
                            action_loss="enabled",
                        )
                    ],
                ),
            )
            report = audit.audit_manifest_splits(
                {"train": [train_path], "val": [val_path]}
            )

        train_report = next(
            item for item in report["manifests"] if item["split"] == "train"
        )
        self.assertIsNone(train_report["manifest_hash"]["computed"])
        self.assertIsNotNone(train_report["manifest_hash"]["error"])
        self.assertEqual(report["status"], "error")

    def test_pair_id_and_weight_consistency(self) -> None:
        success = make_sample(
            dataset_id="train",
            episode_index=0,
            outcome="success",
            action_loss="enabled",
        )
        failure = make_sample(
            dataset_id="train",
            episode_index=1,
            outcome="failure",
            action_loss="disabled",
        )
        for sample in (success, failure):
            sample["pair_id"] = "pair-1"
            sample["pair_weight"] = 0.6
        valid_pair_report, valid_errors = audit.audit_pair_fields(
            [success, failure]
        )
        self.assertEqual(valid_errors, [])
        self.assertEqual(valid_pair_report["complete_pair_ids"], 1)

        failure["pair_weight"] = 0.4
        invalid_pair_report, invalid_errors = audit.audit_pair_fields(
            [success, failure]
        )
        self.assertEqual(
            invalid_pair_report["weight_mismatch_pair_ids"], ["pair-1"]
        )
        self.assertTrue(any("inconsistent pair_weight" in error for error in invalid_errors))

    def test_pair_members_may_be_split_across_manifests_in_one_split(self) -> None:
        success = make_sample(
            dataset_id="train-success",
            episode_index=0,
            outcome="success",
            action_loss="enabled",
        )
        failure = make_sample(
            dataset_id="train-failure",
            episode_index=0,
            outcome="failure",
            action_loss="disabled",
        )
        for sample in (success, failure):
            sample["pair_id"] = "pair-across-manifests"
            sample["pair_weight"] = 0.7
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            success_path = root / "success.json"
            failure_path = root / "failure.json"
            val_path = root / "val.json"
            write_manifest(
                success_path,
                make_manifest("train-success", [success]),
            )
            write_manifest(
                failure_path,
                make_manifest("train-failure", [failure]),
            )
            write_manifest(
                val_path,
                make_manifest(
                    "val",
                    [
                        make_sample(
                            dataset_id="val",
                            episode_index=0,
                            outcome="success",
                            action_loss="enabled",
                        )
                    ],
                ),
            )
            report = audit.audit_manifest_splits(
                {
                    "train": [success_path, failure_path],
                    "val": [val_path],
                }
            )

        self.assertEqual(report["status"], "ok")
        self.assertEqual(
            report["split_pair_consistency"]["train"]["complete_pair_ids"],
            1,
        )

    def test_optional_dataset_root_existence_check(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_root = root / "dataset"
            dataset_root.mkdir()
            train = make_manifest(
                "train",
                [
                    make_sample(
                        dataset_id="train",
                        episode_index=0,
                        outcome="success",
                        action_loss="enabled",
                    )
                ],
            )
            train["dataset_roots"]["train"] = str(dataset_root)
            train["manifest_hash"] = audit.compute_manifest_hash(train)
            train_path = root / "train.json"
            val_path = root / "val.json"
            write_manifest(train_path, train)
            write_manifest(
                val_path,
                make_manifest(
                    "val",
                    [
                        make_sample(
                            dataset_id="val",
                            episode_index=0,
                            outcome="success",
                            action_loss="enabled",
                        )
                    ],
                ),
            )
            report = audit.audit_manifest_splits(
                {"train": [train_path], "val": [val_path]},
                check_dataset_roots=True,
            )

        train_report = next(
            item for item in report["manifests"] if item["split"] == "train"
        )
        val_report = next(
            item for item in report["manifests"] if item["split"] == "val"
        )
        self.assertTrue(
            train_report["dataset_roots"]["roots"]["train"]["exists"]
        )
        self.assertFalse(val_report["dataset_roots"]["roots"]["val"]["exists"])
        self.assertEqual(report["status"], "error")

    def test_cli_writes_report_and_exit_codes_follow_overlap_policy(self) -> None:
        shared = make_sample(
            dataset_id="shared",
            episode_index=0,
            outcome="success",
            action_loss="enabled",
        )
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_path = root / "train.json"
            val_path = root / "val.json"
            output_path = root / "report.json"
            write_manifest(train_path, make_manifest("train", [shared]))
            val_sample = dict(shared)
            val_sample["sample_id"] = "val-sample"
            write_manifest(val_path, make_manifest("val", [val_sample]))

            args = [
                f"train={train_path}",
                f"val={val_path}",
                "--output",
                str(output_path),
            ]
            with mock.patch("sys.stdout", new_callable=io.StringIO):
                self.assertEqual(audit.main(args), 1)
            saved = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertTrue(saved["cross_split_overlap"]["has_episode_leakage"])

            with mock.patch("sys.stdout", new_callable=io.StringIO):
                self.assertEqual(audit.main([*args, "--allow-overlap"]), 0)

    def test_cli_rejects_unlabeled_or_single_split_inputs(self) -> None:
        with mock.patch("sys.stdout", new_callable=io.StringIO):
            self.assertEqual(audit.main(["manifest.json"]), 2)
        with mock.patch("sys.stdout", new_callable=io.StringIO):
            self.assertEqual(
                audit.main(["train=a.json", "train=b.json"]),
                2,
            )


if __name__ == "__main__":
    unittest.main()
