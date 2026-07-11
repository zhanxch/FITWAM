from __future__ import annotations

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from fastwam.everobot_schema import compute_manifest_hash, validate_manifest
from scripts.everobot import build_eve_sidecar


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def make_dataset(
    root: Path,
    episodes: list[dict[str, object]],
    *,
    collection_summary: dict[str, object] | None = None,
) -> None:
    write_json(root / "meta" / "info.json", {"fps": 30})
    write_jsonl(root / "meta" / "episodes.jsonl", episodes)
    data_path = root / "data" / "chunk-000" / "episode_000000.parquet"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    data_path.write_bytes(b"synthetic-action-data")
    if collection_summary is not None:
        write_json(root / "collection_summary.json", collection_summary)


class EveSidecarBuilderTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = TemporaryDirectory()
        root = Path(self.tmp.name)
        self.base_root = root / "base"
        self.rollout_root = root / "rollout"
        self.trimmed_root = root / "trimmed"
        self.eve_root = root / "eve"

        make_dataset(
            self.base_root,
            [
                {"episode_index": 0, "length": 40, "tasks": ["Water plant"]},
                {"episode_index": 1, "length": 48, "tasks": ["Water plant"]},
            ],
        )
        make_dataset(
            self.rollout_root,
            [
                {"episode_index": 0, "length": 50, "tasks": ["Water plant"]},
                {
                    "episode_index": 1,
                    "length": 60,
                    "tasks": [
                        "Water plant. " + build_eve_sidecar.FAILURE_PHRASE
                    ],
                },
            ],
            collection_summary={
                "attempt_log": [
                    {
                        "saved_episode_index": 0,
                        "attempt_index": 10,
                        "seed": 101,
                        "success": True,
                    },
                    {
                        "saved_episode_index": 1,
                        "attempt_index": 11,
                        "seed": 102,
                        "success": False,
                    },
                ]
            },
        )
        write_json(
            self.trimmed_root / "trim_summary.json",
            {"episodes": [{"episode_index": 1, "trimmed_length": 36}]},
        )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def init_args(self) -> argparse.Namespace:
        return argparse.Namespace(
            dataset_root=self.base_root,
            dataset_id="expert_base",
            eve_root=self.eve_root,
            task_name="water_plant",
            source_type="expert_success",
            source_policy="expert",
            collection_round=-1,
            split="train",
            failure_phrase=build_eve_sidecar.FAILURE_PHRASE,
            default_failure_type="unknown_failure",
            force_success=True,
            dataset_uri="dataset://water-plant/base",
            dataset_fingerprint_sha256=None,
            parent_round_ids=None,
            config_path=None,
            code_commit="test-commit",
            created_at="2026-07-11T00:00:00+00:00",
        )

    def append_args(self, *, source_policy: str = "fastwam-step6650") -> argparse.Namespace:
        return argparse.Namespace(
            base_eve_root=self.eve_root,
            rollout_root=self.rollout_root,
            trimmed_event_root=self.trimmed_root,
            dataset_id="rollout_round0",
            task_name="water_plant",
            source_policy=source_policy,
            source_checkpoint="step_6650.pt",
            source_checkpoint_sha256="a" * 64,
            collection_round=0,
            split="train",
            failure_phrase=build_eve_sidecar.FAILURE_PHRASE,
            default_failure_type="timeout_or_incomplete",
            failure_action_loss="disabled",
            annotation_source="auto",
            annotation_method=None,
            annotation_version="event_window_v1",
            annotation_confidence=None,
            dataset_uri="dataset://water-plant/round0",
            dataset_fingerprint_sha256=None,
            parent_round_ids=["expert_base:round:-1"],
            config_path=None,
            code_commit="test-commit",
            created_at="2026-07-11T01:00:00+00:00",
        )

    def manifest_args(
        self, name: str, *, collection_rounds: list[int] | None = None
    ) -> argparse.Namespace:
        return argparse.Namespace(
            eve_root=self.eve_root,
            manifest_name=name,
            include_outcomes=["success", "failure"],
            success_dataset_ids=None,
            failure_dataset_ids=None,
            failure_sample_mode="event_only",
            collection_rounds=collection_rounds,
            splits=["train"],
            include_sample_ids=None,
            exclude_sample_ids=None,
            success_sample_stride=1,
            failure_sample_stride=1,
            failure_action_loss="disabled",
        )

    def test_two_round_build_is_immutable_and_filterable(self) -> None:
        build_eve_sidecar.init_base(self.init_args())
        build_eve_sidecar.append_rollout(self.append_args())
        build_eve_sidecar.append_rollout(self.append_args())

        rounds = build_eve_sidecar.load_jsonl(self.eve_root / "round_meta.jsonl")
        episodes = build_eve_sidecar.load_jsonl(
            self.eve_root / "episode_meta.jsonl"
        )
        events = build_eve_sidecar.load_jsonl(self.eve_root / "event_meta.jsonl")
        self.assertEqual(len(rounds), 2)
        self.assertEqual(len(episodes), 4)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["end_frame"], 36)
        self.assertIsNone(events[0]["failure_frame"])
        self.assertEqual(events[0]["annotation"]["source"], "auto")

        build_eve_sidecar.build_manifest(self.manifest_args("all_rounds"))
        manifest = build_eve_sidecar.read_json(
            self.eve_root / "manifests" / "all_rounds.json"
        )
        validate_manifest(manifest)
        self.assertEqual(manifest["manifest_hash"], compute_manifest_hash(manifest))
        self.assertEqual(manifest["num_samples"], 4)
        self.assertTrue(
            all(
                "round_id" in sample and "collection_round" in sample
                for sample in manifest["samples"]
            )
        )

        build_eve_sidecar.build_manifest(
            self.manifest_args("round0_only", collection_rounds=[0])
        )
        round0_manifest = build_eve_sidecar.read_json(
            self.eve_root / "manifests" / "round0_only.json"
        )
        validate_manifest(round0_manifest)
        self.assertEqual(round0_manifest["num_samples"], 2)
        self.assertEqual(
            {sample["episode_outcome"] for sample in round0_manifest["samples"]},
            {"success", "failure"},
        )
        self.assertEqual(round0_manifest["source_round_ids"], ["rollout_round0:round:0"])

        with self.assertRaisesRegex(ValueError, "identity collision"):
            build_eve_sidecar.append_rollout(
                self.append_args(source_policy="different-policy")
            )

        reused_dataset_args = self.append_args()
        reused_dataset_args.collection_round = 1
        with self.assertRaisesRegex(ValueError, "identity collision"):
            build_eve_sidecar.append_rollout(reused_dataset_args)
        rounds_after_failed_transaction = build_eve_sidecar.load_jsonl(
            self.eve_root / "round_meta.jsonl"
        )
        self.assertEqual(len(rounds_after_failed_transaction), 2)

    def test_manual_head_and_tail_trim_keeps_raw_frame_coordinates(self) -> None:
        start, end, rule = build_eve_sidecar.trim_frame_interval(
            episode_index=4,
            raw_length=600,
            trim_report={
                4: {
                    "new_length": 480,
                    "trim_start_frame": 60,
                    "trim_end_frame": 540,
                }
            },
        )

        self.assertEqual((start, end), (60, 540))
        self.assertEqual(rule, "trimmed_failure_window")

    def test_dataset_fingerprint_covers_action_data(self) -> None:
        before = build_eve_sidecar.dataset_content_fingerprint(self.base_root)
        data_path = (
            self.base_root / "data" / "chunk-000" / "episode_000000.parquet"
        )
        data_path.write_bytes(b"changed-action-data")
        after = build_eve_sidecar.dataset_content_fingerprint(self.base_root)

        self.assertNotEqual(before, after)

    def test_manifest_rejects_event_outside_its_episode(self) -> None:
        build_eve_sidecar.init_base(self.init_args())
        build_eve_sidecar.append_rollout(self.append_args())
        event_path = self.eve_root / "event_meta.jsonl"
        events = build_eve_sidecar.load_jsonl(event_path)
        events[0]["end_frame"] = 999
        write_jsonl(event_path, events)

        with self.assertRaisesRegex(ValueError, "exceeds episode length"):
            build_eve_sidecar.build_manifest(self.manifest_args("invalid-event"))

    def test_invalid_trim_bounds_are_rejected_instead_of_clamped(self) -> None:
        with self.assertRaisesRegex(ValueError, "Invalid trim interval"):
            build_eve_sidecar.trim_frame_interval(
                episode_index=0,
                raw_length=100,
                trim_report={
                    0: {"trim_start_frame": -1, "trim_end_frame": 101}
                },
            )


if __name__ == "__main__":
    unittest.main()
