from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

from fastwam.everobot_schema import (
    compute_manifest_hash,
    validate_manifest,
    with_manifest_hash,
)
from scripts.everobot import match_auxiliary_manifest_budget as matcher


SOURCE_HASHES = {
    "round_meta_sha256": "a" * 64,
    "episode_meta_sha256": "b" * 64,
    "event_meta_sha256": "c" * 64,
}


def primary_sample(sample_id: str, *, split: str = "train") -> dict[str, object]:
    return {
        "sample_type": "episode",
        "sample_id": sample_id,
        "dataset_id": "expert",
        "episode_id": f"expert:episode:{sample_id}",
        "episode_index": int(sample_id.rsplit("-", 1)[-1]),
        "round_id": "expert:round:0",
        "collection_round": 0,
        "episode_outcome": "success",
        "event_outcome": "success",
        "start_frame": 0,
        "end_frame": 100,
        "action_loss": "enabled",
        "batch_role": "primary",
        "sample_stride": 1,
        "split": split,
    }


def auxiliary_sample(
    sample_id: str,
    *,
    dataset_id: str,
    episode_index: int,
    outcome: str,
    split: str,
    core_start: int,
    window_selection: str = "core_start_anchor",
) -> dict[str, object]:
    return {
        "sample_type": "event",
        "sample_id": sample_id,
        "event_id": sample_id,
        "dataset_id": dataset_id,
        "episode_id": f"{dataset_id}:episode:{episode_index:06d}",
        "episode_index": episode_index,
        "round_id": f"{dataset_id}:round:0",
        "collection_round": 0,
        "episode_outcome": outcome,
        "event_outcome": "unknown",
        "event_type": "interaction_candidate",
        "event_level": "candidate",
        "effector": "global",
        "start_frame": max(core_start - 5, 0),
        "end_frame": core_start + 20,
        "core_start_frame": core_start,
        "core_end_frame": core_start + 5,
        "action_loss": "disabled",
        "batch_role": "auxiliary",
        "sample_stride": 1,
        "split": split,
        "window_selection": window_selection,
    }


def manifest(
    name: str,
    samples: list[dict[str, object]],
) -> dict[str, object]:
    round_ids = sorted({str(sample["round_id"]) for sample in samples})
    dataset_ids = sorted({str(sample["dataset_id"]) for sample in samples})
    return with_manifest_hash(
        {
            "format": "EveRobotTrainManifest",
            "schema_version": "0.2",
            "frame_interval": "half_open",
            "manifest_name": name,
            "selection": {"purpose": name},
            "samples": samples,
            "num_samples": len(samples),
            "dataset_roots": {
                dataset_id: f"/datasets/{dataset_id}"
                for dataset_id in dataset_ids
            },
            "source_round_ids": round_ids,
            "source_hashes": dict(SOURCE_HASHES),
        }
    )


def episode_row(sample: dict[str, object], *, length: int = 100) -> dict[str, object]:
    return {
        "dataset_id": sample["dataset_id"],
        "episode_id": sample["episode_id"],
        "episode_index": sample["episode_index"],
        "length": length,
        "split": sample["split"],
    }


class MatchAuxiliaryManifestBudgetTest(unittest.TestCase):
    def setUp(self) -> None:
        self.primary_train = primary_sample("primary-0", split="train")
        self.primary_val = primary_sample("primary-1", split="val")
        self.control_aux = [
            auxiliary_sample(
                "control-train-010",
                dataset_id="control",
                episode_index=0,
                outcome="success",
                split="train",
                core_start=10,
            ),
            auxiliary_sample(
                "control-train-040",
                dataset_id="control",
                episode_index=1,
                outcome="success",
                split="train",
                core_start=40,
            ),
            auxiliary_sample(
                "control-val-030",
                dataset_id="control",
                episode_index=2,
                outcome="success",
                split="val",
                core_start=30,
            ),
            auxiliary_sample(
                "control-val-090",
                dataset_id="control",
                episode_index=3,
                outcome="success",
                split="val",
                core_start=90,
            ),
        ]
        self.reference_aux = [
            auxiliary_sample(
                "reference-train-035",
                dataset_id="reference",
                episode_index=0,
                outcome="failure",
                split="train",
                core_start=35,
            ),
            auxiliary_sample(
                "reference-val-080",
                dataset_id="reference",
                episode_index=1,
                outcome="failure",
                split="val",
                core_start=80,
            ),
        ]
        self.control = manifest(
            "B0",
            [self.primary_train, self.primary_val, *self.control_aux],
        )
        self.reference = manifest(
            "B1",
            [self.primary_train, self.primary_val, *self.reference_aux],
        )
        self.episodes = [
            episode_row(sample)
            for sample in [
                self.primary_train,
                self.primary_val,
                *self.control_aux,
                *self.reference_aux,
            ]
        ]

    def test_selects_equal_budget_globally_and_preserves_all_primary(self) -> None:
        result, diagnostics = matcher.match_auxiliary_budget(
            self.control,
            self.reference,
            self.episodes,
            seed=20260718,
        )

        validate_manifest(result, strict=True, verify_hash=True)
        self.assertEqual(result["manifest_hash"], compute_manifest_hash(result))
        primary_ids = [
            sample["sample_id"]
            for sample in result["samples"]
            if sample["batch_role"] == "primary"
        ]
        result_primary = [
            sample
            for sample in result["samples"]
            if sample["batch_role"] == "primary"
        ]
        auxiliary_ids = [
            sample["sample_id"]
            for sample in result["samples"]
            if sample["batch_role"] == "auxiliary"
        ]
        self.assertEqual(primary_ids, ["primary-0", "primary-1"])
        self.assertEqual(result_primary, [self.primary_train, self.primary_val])
        self.assertEqual(
            auxiliary_ids, ["control-train-040", "control-val-090"]
        )
        self.assertEqual(diagnostics["counts"]["reference_auxiliary"], 2)
        self.assertEqual(diagnostics["counts"]["selected_control_auxiliary"], 2)
        self.assertEqual(
            diagnostics["selected_sample_ids_sha256"],
            result["selection"]["auxiliary_budget_match"][
                "selected_sample_ids_sha256"
            ],
        )
        self.assertAlmostEqual(
            diagnostics["progress"]["mean_absolute_distance"], 0.075
        )
        self.assertAlmostEqual(
            diagnostics["progress"]["median_absolute_distance"], 0.075
        )
        self.assertAlmostEqual(
            diagnostics["progress"]["max_abs_decile_percentage_points"],
            50.0,
        )

    def test_same_seed_is_deterministic_for_equal_progress_ties(self) -> None:
        tied = auxiliary_sample(
            "control-train-040-b",
            dataset_id="control",
            episode_index=4,
            outcome="success",
            split="train",
            core_start=40,
        )
        control = manifest(
            "B0-tied",
            [
                self.primary_train,
                self.primary_val,
                *self.control_aux,
                tied,
            ],
        )
        episodes = [*self.episodes, episode_row(tied)]
        first, first_diagnostics = matcher.match_auxiliary_budget(
            control, self.reference, episodes, seed=17
        )
        second, second_diagnostics = matcher.match_auxiliary_budget(
            control, self.reference, episodes, seed=17
        )
        self.assertEqual(first, second)
        self.assertEqual(first_diagnostics, second_diagnostics)
        self.assertEqual(
            len(first_diagnostics["selected_sample_ids"]),
            len(set(first_diagnostics["selected_sample_ids"])),
        )

    def test_rejects_insufficient_control_auxiliary_without_outputs(self) -> None:
        reference_extra = auxiliary_sample(
            "reference-train-060",
            dataset_id="reference",
            episode_index=2,
            outcome="failure",
            split="train",
            core_start=60,
        )
        reference = manifest(
            "B1-too-large",
            [
                self.primary_train,
                self.primary_val,
                *self.reference_aux,
                reference_extra,
            ],
        )
        control = manifest(
            "B0-too-small",
            [
                self.primary_train,
                self.primary_val,
                self.control_aux[0],
                self.control_aux[2],
            ],
        )
        with self.assertRaisesRegex(ValueError, "insufficient"):
            matcher.match_auxiliary_budget(
                control,
                reference,
                [*self.episodes, episode_row(reference_extra)],
                seed=1,
            )

    def test_rejects_split_mixing_and_non_anchor_windows(self) -> None:
        train_only_control = manifest(
            "B0-train-only",
            [
                self.primary_train,
                self.primary_val,
                self.control_aux[0],
                self.control_aux[1],
            ],
        )
        with self.assertRaisesRegex(ValueError, "split sets differ"):
            matcher.match_auxiliary_budget(
                train_only_control,
                self.reference,
                self.episodes,
                seed=1,
            )

        invalid_auxiliary = dict(self.control_aux[0])
        invalid_auxiliary["window_selection"] = "all"
        invalid_control = manifest(
            "B0-invalid-window",
            [
                self.primary_train,
                self.primary_val,
                invalid_auxiliary,
                *self.control_aux[1:],
            ],
        )
        with self.assertRaisesRegex(ValueError, "core_start_anchor"):
            matcher.match_auxiliary_budget(
                invalid_control,
                self.reference,
                self.episodes,
                seed=1,
            )

    def test_cli_atomically_writes_both_outputs_and_refuses_overwrite(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            eve_root = root / "eve"
            eve_root.mkdir()
            control_path = root / "B0.json"
            reference_path = root / "B1.json"
            output_path = root / "B0_matched.json"
            diagnostics_path = root / "B0_matched.diagnostics.json"
            control_path.write_text(json.dumps(self.control), encoding="utf-8")
            reference_path.write_text(
                json.dumps(self.reference), encoding="utf-8"
            )
            (eve_root / "episode_meta.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in self.episodes),
                encoding="utf-8",
            )

            exit_code = matcher.main(
                [
                    "--control-manifest",
                    str(control_path),
                    "--reference-manifest",
                    str(reference_path),
                    "--eve-root",
                    str(eve_root),
                    "--output",
                    str(output_path),
                    "--diagnostics-output",
                    str(diagnostics_path),
                    "--seed",
                    "20260718",
                ]
            )
            self.assertEqual(exit_code, 0)
            self.assertTrue(output_path.is_file())
            self.assertTrue(diagnostics_path.is_file())

            with self.assertRaisesRegex(FileExistsError, "overwrite"):
                matcher.main(
                    [
                        "--control-manifest",
                        str(control_path),
                        "--reference-manifest",
                        str(reference_path),
                        "--eve-root",
                        str(eve_root),
                        "--output",
                        str(output_path),
                        "--diagnostics-output",
                        str(root / "unused.json"),
                        "--seed",
                        "20260718",
                    ]
                )
            self.assertFalse((root / "unused.json").exists())

            second_output = root / "second.json"
            with self.assertRaisesRegex(FileExistsError, "overwrite"):
                matcher.main(
                    [
                        "--control-manifest",
                        str(control_path),
                        "--reference-manifest",
                        str(reference_path),
                        "--eve-root",
                        str(eve_root),
                        "--output",
                        str(second_output),
                        "--diagnostics-output",
                        str(diagnostics_path),
                        "--seed",
                        "20260718",
                    ]
                )
            self.assertFalse(second_output.exists())

    def test_atomic_publish_rolls_back_if_second_link_fails(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            output_path = root / "output.json"
            diagnostics_path = root / "diagnostics.json"
            real_link = matcher.os.link
            calls = 0

            def fail_second_link(source: Path, destination: Path) -> None:
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("simulated diagnostics publish failure")
                real_link(source, destination)

            with (
                mock.patch.object(matcher.os, "link", side_effect=fail_second_link),
                self.assertRaisesRegex(OSError, "simulated"),
            ):
                matcher.write_outputs_atomically_new(
                    output_path,
                    {"output": True},
                    diagnostics_path,
                    {"diagnostics": True},
                )

            self.assertFalse(output_path.exists())
            self.assertFalse(diagnostics_path.exists())
            self.assertEqual(list(root.glob(".*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
