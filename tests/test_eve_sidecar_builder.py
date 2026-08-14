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
    outcome_rows: list[dict[str, object]] | None = None,
) -> None:
    write_json(root / "meta" / "info.json", {"fps": 30})
    write_jsonl(root / "meta" / "episodes.jsonl", episodes)
    data_path = root / "data" / "chunk-000" / "episode_000000.parquet"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    data_path.write_bytes(b"synthetic-action-data")
    if collection_summary is not None:
        write_json(root / "collection_summary.json", collection_summary)
    if outcome_rows is not None:
        write_jsonl(root / "meta" / "episode_outcomes.jsonl", outcome_rows)


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
            success_auxiliary_dataset_ids=None,
            failure_dataset_ids=None,
            success_sample_mode="episode_only",
            failure_sample_mode="event_only",
            failure_window_selection="core_start_anchor",
            failure_source_window_rules=None,
            event_types=None,
            collection_rounds=collection_rounds,
            splits=["train"],
            include_sample_ids=None,
            exclude_sample_ids=None,
            success_sample_stride=1,
            failure_sample_stride=1,
            failure_action_loss="disabled",
        )

    def test_failure_sliding_selection_preserves_all_event_windows(self) -> None:
        build_eve_sidecar.init_base(self.init_args())
        build_eve_sidecar.append_rollout(self.append_args())

        args = self.manifest_args("failure_sliding", collection_rounds=[0])
        args.include_outcomes = ["failure"]
        args.failure_window_selection = "sliding"
        build_eve_sidecar.build_manifest(args)

        manifest = build_eve_sidecar.read_json(
            self.eve_root / "manifests" / "failure_sliding.json"
        )
        self.assertEqual(
            manifest["selection"]["failure_window_selection"], "sliding"
        )
        self.assertEqual(manifest["num_samples"], 1)
        self.assertNotIn("window_selection", manifest["samples"][0])

    def test_failure_source_window_rule_excludes_full_episode_fallback(self) -> None:
        build_eve_sidecar.init_base(self.init_args())
        build_eve_sidecar.append_rollout(self.append_args())

        event_path = self.eve_root / "event_meta.jsonl"
        events = build_eve_sidecar.load_jsonl(event_path)
        events[0]["source_window_rule"] = "full_failure_episode"
        write_jsonl(event_path, events)

        args = self.manifest_args("no_fallbacks")
        args.failure_source_window_rules = ["trimmed_failure_window"]
        build_eve_sidecar.build_manifest(args)

        manifest = build_eve_sidecar.read_json(
            self.eve_root / "manifests" / "no_fallbacks.json"
        )
        self.assertEqual(
            manifest["selection"]["failure_source_window_rules"],
            ["trimmed_failure_window"],
        )
        self.assertNotIn("failure", {
            sample["episode_outcome"] for sample in manifest["samples"]
        })

    def candidate_event_rows(self) -> list[dict[str, object]]:
        common = {
            "schema_version": build_eve_sidecar.SCHEMA_VERSION,
            "round_id": "rollout_round0:round:0",
            "dataset_id": "rollout_round0",
            "dataset_root": str(self.rollout_root.resolve()),
            "task_name": "water_plant",
            "task": "Water plant",
            "event_type": "interaction_candidate",
            "event_level": "candidate",
            "event_label": "state_line_transition",
            "effector": "global",
            "event_outcome": "unknown",
            "source_policy": "fastwam-step6650",
            "collection_round": 0,
            "split": "train",
        }
        return [
            {
                **common,
                "event_id": "rollout_round0_ep000000_candidate_000",
                "episode_id": "rollout_round0:episode:000000",
                "episode_index": 0,
                "episode_outcome": "success",
                "start_frame": 5,
                "end_frame": 20,
                "core_start_frame": 8,
                "core_end_frame": 16,
                "core_interval": [8, 16],
                "event_weight": 0.75,
                "action_loss": "disabled",
                "annotation": {
                    "source": "auto",
                    "method": "state_line",
                    "version": "state_line_v1",
                    "confidence": 0.8,
                },
            },
            {
                **common,
                "event_id": "rollout_round0_ep000001_candidate_000",
                "episode_id": "rollout_round0:episode:000001",
                "episode_index": 1,
                "episode_outcome": "failure",
                "start_frame": 20,
                "end_frame": 45,
                "core_start_frame": 25,
                "core_end_frame": 38,
                "core_interval": [25, 38],
                "event_weight": 0.9,
                "action_loss": "enabled",
                "annotation": {
                    "source": "auto",
                    "method": "state_line",
                    "version": "state_line_v1",
                    "confidence": 0.85,
                },
            },
        ]

    def append_candidate_events(
        self, rows: list[dict[str, object]] | None = None
    ) -> None:
        build_eve_sidecar.append_immutable_jsonl_group(
            [
                (
                    self.eve_root / "event_meta.jsonl",
                    self.candidate_event_rows() if rows is None else rows,
                    ("event_id",),
                    ("dataset_root",),
                )
            ]
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

    def test_append_rollout_uses_required_structured_outcomes_with_clean_task(
        self,
    ) -> None:
        write_jsonl(
            self.rollout_root / "meta" / "episodes.jsonl",
            [
                {"episode_index": 0, "length": 50, "tasks": ["Water plant"]},
                {"episode_index": 1, "length": 60, "tasks": ["Water plant"]},
            ],
        )
        write_jsonl(
            self.rollout_root / "meta" / "episode_outcomes.jsonl",
            [
                {
                    "episode_index": 0,
                    "outcome": "success",
                    "success": True,
                    "attempt_index": 10,
                    "seed": 101,
                    "source": "dexjoco_env",
                },
                {
                    "episode_index": 1,
                    "outcome": "failure",
                    "success": False,
                    "attempt_index": 11,
                    "seed": 102,
                    "source": "dexjoco_env",
                },
            ],
        )
        build_eve_sidecar.init_base(self.init_args())
        args = self.append_args()
        args.require_explicit_outcomes = True
        build_eve_sidecar.append_rollout(args)

        rows = [
            row
            for row in build_eve_sidecar.load_jsonl(
                self.eve_root / "episode_meta.jsonl"
            )
            if row["dataset_id"] == "rollout_round0"
        ]
        self.assertEqual(
            [row["episode_outcome"] for row in rows], ["success", "failure"]
        )
        self.assertEqual(
            {row["outcome_source"] for row in rows},
            {"structured_outcome_ledger"},
        )
        self.assertTrue(all(row["task"] == "Water plant" for row in rows))

    def test_append_rollout_rejects_missing_required_outcome_ledger(self) -> None:
        build_eve_sidecar.init_base(self.init_args())
        args = self.append_args()
        args.require_explicit_outcomes = True
        with self.assertRaisesRegex(
            FileNotFoundError, "structured outcome ledger"
        ):
            build_eve_sidecar.append_rollout(args)

    def test_outcome_ledger_rejects_partial_coverage(self) -> None:
        write_jsonl(
            self.rollout_root / "meta" / "episode_outcomes.jsonl",
            [
                {
                    "episode_index": 0,
                    "outcome": "success",
                    "success": True,
                }
            ],
        )
        with self.assertRaisesRegex(ValueError, "coverage mismatch"):
            build_eve_sidecar.load_episode_outcome_ledger(
                self.rollout_root,
                required=True,
                expected_episode_indices=[0, 1],
            )

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

    def test_manifest_include_exclude_filters_are_strict(self) -> None:
        build_eve_sidecar.init_base(self.init_args())
        build_eve_sidecar.append_rollout(self.append_args())
        build_eve_sidecar.build_manifest(self.manifest_args("all"))
        manifest = build_eve_sidecar.read_json(
            self.eve_root / "manifests" / "all.json"
        )
        sample_ids = [sample["sample_id"] for sample in manifest["samples"]]

        include_args = self.manifest_args("included")
        include_args.include_sample_ids = [sample_ids[0]]
        build_eve_sidecar.build_manifest(include_args)
        included = build_eve_sidecar.read_json(
            self.eve_root / "manifests" / "included.json"
        )
        self.assertEqual(
            [sample["sample_id"] for sample in included["samples"]],
            [sample_ids[0]],
        )

        exclude_args = self.manifest_args("excluded")
        exclude_args.exclude_sample_ids = [sample_ids[0]]
        build_eve_sidecar.build_manifest(exclude_args)
        excluded = build_eve_sidecar.read_json(
            self.eve_root / "manifests" / "excluded.json"
        )
        self.assertNotIn(
            sample_ids[0],
            {sample["sample_id"] for sample in excluded["samples"]},
        )

        overlap_args = self.manifest_args("overlap")
        overlap_args.include_sample_ids = [sample_ids[0]]
        overlap_args.exclude_sample_ids = [sample_ids[0]]
        with self.assertRaisesRegex(ValueError, "both included and excluded"):
            build_eve_sidecar.build_manifest(overlap_args)

        missing_args = self.manifest_args("missing")
        missing_args.include_sample_ids = ["missing:sample"]
        with self.assertRaisesRegex(ValueError, "Requested sample IDs are absent"):
            build_eve_sidecar.build_manifest(missing_args)

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

    def test_interaction_candidates_support_success_and_failure(self) -> None:
        build_eve_sidecar.init_base(self.init_args())
        build_eve_sidecar.append_rollout(self.append_args())
        self.append_candidate_events()

        args = self.manifest_args("interaction_candidates", collection_rounds=[0])
        args.success_sample_mode = "event_only"
        args.event_types = ["interaction_candidate"]
        build_eve_sidecar.build_manifest(args)

        manifest = build_eve_sidecar.read_json(
            self.eve_root / "manifests" / "interaction_candidates.json"
        )
        validate_manifest(manifest)
        self.assertEqual(manifest["num_samples"], 2)
        self.assertEqual(
            manifest["selection"]["success_sample_mode"], "event_only"
        )
        self.assertEqual(
            manifest["selection"]["event_types"], ["interaction_candidate"]
        )

        by_outcome = {
            sample["episode_outcome"]: sample for sample in manifest["samples"]
        }
        success = by_outcome["success"]
        failure = by_outcome["failure"]
        self.assertEqual(success["event_outcome"], "unknown")
        self.assertEqual(failure["event_outcome"], "unknown")
        self.assertEqual(success["action_loss"], "enabled")
        self.assertEqual(failure["action_loss"], "disabled")
        self.assertEqual(success["batch_role"], "primary")
        self.assertEqual(failure["batch_role"], "auxiliary")
        self.assertNotIn("window_selection", success)
        self.assertEqual(
            failure["window_selection"], "core_start_anchor"
        )
        self.assertEqual(success["sample_role"], "success_candidate")
        self.assertEqual(failure["sample_role"], "failure_candidate")
        self.assertEqual(success["event_weight"], 0.75)
        self.assertEqual(failure["event_weight"], 0.9)
        self.assertEqual(success["core_start_frame"], 8)
        self.assertEqual(success["core_end_frame"], 16)
        self.assertEqual(success["core_interval"], [8, 16])
        self.assertEqual(success["annotation"]["method"], "state_line")

    def test_success_sample_modes_and_event_type_filter(self) -> None:
        build_eve_sidecar.init_base(self.init_args())
        build_eve_sidecar.append_rollout(self.append_args())
        self.append_candidate_events()

        default_args = self.manifest_args(
            "success_episode_default", collection_rounds=[0]
        )
        default_args.include_outcomes = ["success"]
        build_eve_sidecar.build_manifest(default_args)
        default_manifest = build_eve_sidecar.read_json(
            self.eve_root / "manifests" / "success_episode_default.json"
        )
        self.assertEqual(default_manifest["num_samples"], 1)
        self.assertEqual(default_manifest["samples"][0]["sample_type"], "episode")

        both_args = self.manifest_args("success_both", collection_rounds=[0])
        both_args.include_outcomes = ["success"]
        both_args.success_sample_mode = "both"
        both_args.event_types = ["interaction_candidate"]
        build_eve_sidecar.build_manifest(both_args)
        both_manifest = build_eve_sidecar.read_json(
            self.eve_root / "manifests" / "success_both.json"
        )
        self.assertEqual(both_manifest["num_samples"], 2)
        self.assertEqual(
            {sample["sample_type"] for sample in both_manifest["samples"]},
            {"episode", "event"},
        )

        legacy_args = self.manifest_args(
            "legacy_failure_only", collection_rounds=[0]
        )
        legacy_args.success_sample_mode = "event_only"
        legacy_args.event_types = ["failure_event"]
        build_eve_sidecar.build_manifest(legacy_args)
        legacy_manifest = build_eve_sidecar.read_json(
            self.eve_root / "manifests" / "legacy_failure_only.json"
        )
        self.assertEqual(legacy_manifest["num_samples"], 1)
        self.assertEqual(
            legacy_manifest["samples"][0]["event_type"], "failure_event"
        )
        self.assertEqual(
            legacy_manifest["samples"][0]["event_outcome"], "failure"
        )

    def test_candidate_ledger_is_immutable_and_idempotent(self) -> None:
        build_eve_sidecar.init_base(self.init_args())
        build_eve_sidecar.append_rollout(self.append_args())
        rows = self.candidate_event_rows()
        self.append_candidate_events(rows)
        self.append_candidate_events(rows)
        events = build_eve_sidecar.load_jsonl(
            self.eve_root / "event_meta.jsonl"
        )
        self.assertEqual(len(events), 3)

        changed = [dict(row) for row in rows]
        changed[0]["event_weight"] = 0.5
        with self.assertRaisesRegex(ValueError, "identity collision"):
            self.append_candidate_events(changed)

        events_after_collision = build_eve_sidecar.load_jsonl(
            self.eve_root / "event_meta.jsonl"
        )
        self.assertEqual(events_after_collision, events)

    def test_candidate_episode_outcome_must_match_linked_episode(self) -> None:
        build_eve_sidecar.init_base(self.init_args())
        build_eve_sidecar.append_rollout(self.append_args())
        mismatched = self.candidate_event_rows()[1]
        mismatched["episode_outcome"] = "success"
        self.append_candidate_events([mismatched])

        args = self.manifest_args("mismatched_candidate", collection_rounds=[0])
        args.include_outcomes = ["success"]
        args.success_sample_mode = "event_only"
        args.event_types = ["interaction_candidate"]
        with self.assertRaisesRegex(
            ValueError, "episode_outcome does not match its episode"
        ):
            build_eve_sidecar.build_manifest(args)

    def test_invalid_trim_bounds_are_rejected_instead_of_clamped(self) -> None:
        with self.assertRaisesRegex(ValueError, "Invalid trim interval"):
            build_eve_sidecar.trim_frame_interval(
                episode_index=0,
                raw_length=100,
                trim_report={
                    0: {"trim_start_frame": -1, "trim_end_frame": 101}
                },
            )

    def test_frozen_split_map_is_applied_to_episode_and_event_ledgers(self) -> None:
        split_map = self.eve_root.parent / "episode_splits.jsonl"
        write_jsonl(
            split_map,
            [
                {
                    "dataset_id": "expert_base",
                    "episode_index": 0,
                    "split": "train",
                },
                {
                    "dataset_id": "expert_base",
                    "episode_index": 1,
                    "split": "val",
                },
                {
                    "dataset_id": "rollout_round0",
                    "episode_index": 0,
                    "split": "val",
                },
                {
                    "dataset_id": "rollout_round0",
                    "episode_index": 1,
                    "split": "train",
                },
            ],
        )
        init_args = self.init_args()
        init_args.split_map = split_map
        append_args = self.append_args()
        append_args.split_map = split_map

        build_eve_sidecar.init_base(init_args)
        build_eve_sidecar.append_rollout(append_args)

        episodes = {
            (row["dataset_id"], row["episode_index"]): row["split"]
            for row in build_eve_sidecar.load_jsonl(
                self.eve_root / "episode_meta.jsonl"
            )
        }
        self.assertEqual(episodes[("expert_base", 0)], "train")
        self.assertEqual(episodes[("expert_base", 1)], "val")
        self.assertEqual(episodes[("rollout_round0", 0)], "val")
        self.assertEqual(episodes[("rollout_round0", 1)], "train")
        event = build_eve_sidecar.load_jsonl(
            self.eve_root / "event_meta.jsonl"
        )[0]
        self.assertEqual(event["split"], "train")

    def test_split_map_must_cover_every_episode(self) -> None:
        split_map = self.eve_root.parent / "incomplete_splits.jsonl"
        write_jsonl(
            split_map,
            [
                {
                    "dataset_id": "expert_base",
                    "episode_index": 0,
                    "split": "train",
                }
            ],
        )
        args = self.init_args()
        args.split_map = split_map
        with self.assertRaisesRegex(ValueError, "no assignment"):
            build_eve_sidecar.init_base(args)

    def test_success_auxiliary_dataset_is_action_disabled(self) -> None:
        build_eve_sidecar.init_base(self.init_args())
        build_eve_sidecar.append_rollout(self.append_args())
        args = self.manifest_args("success_auxiliary_control")
        args.include_outcomes = ["success"]
        args.success_dataset_ids = ["expert_base"]
        args.success_auxiliary_dataset_ids = ["rollout_round0"]

        build_eve_sidecar.build_manifest(args)
        manifest = json.loads(
            (
                self.eve_root
                / "manifests"
                / "success_auxiliary_control.json"
            ).read_text(encoding="utf-8")
        )
        primary = [
            sample
            for sample in manifest["samples"]
            if sample["dataset_id"] == "expert_base"
        ]
        auxiliary = [
            sample
            for sample in manifest["samples"]
            if sample["dataset_id"] == "rollout_round0"
        ]
        self.assertEqual(len(primary), 2)
        self.assertEqual(len(auxiliary), 1)
        self.assertTrue(
            all(sample["action_loss"] == "enabled" for sample in primary)
        )
        self.assertTrue(
            all(sample["batch_role"] == "primary" for sample in primary)
        )
        self.assertEqual(auxiliary[0]["action_loss"], "disabled")
        self.assertEqual(auxiliary[0]["batch_role"], "auxiliary")

    def test_success_auxiliary_event_keeps_expert_episode_primary(self) -> None:
        build_eve_sidecar.init_base(self.init_args())
        build_eve_sidecar.append_rollout(self.append_args())
        self.append_candidate_events()
        for mode, expected_auxiliary_count in (("event_only", 1), ("both", 2)):
            with self.subTest(mode=mode):
                manifest_name = f"success_auxiliary_events_{mode}"
                args = self.manifest_args(manifest_name)
                args.include_outcomes = ["success"]
                args.success_dataset_ids = ["expert_base"]
                args.success_auxiliary_dataset_ids = ["rollout_round0"]
                args.success_sample_mode = mode
                args.event_types = ["interaction_candidate"]

                build_eve_sidecar.build_manifest(args)
                manifest = build_eve_sidecar.read_json(
                    self.eve_root / "manifests" / f"{manifest_name}.json"
                )
                primary = [
                    sample
                    for sample in manifest["samples"]
                    if sample["batch_role"] == "primary"
                ]
                auxiliary = [
                    sample
                    for sample in manifest["samples"]
                    if sample["batch_role"] == "auxiliary"
                ]

                self.assertEqual(len(primary), 2)
                self.assertTrue(
                    all(
                        sample["dataset_id"] == "expert_base"
                        for sample in primary
                    )
                )
                self.assertTrue(
                    all(sample["sample_type"] == "episode" for sample in primary)
                )
                self.assertTrue(
                    all(sample["action_loss"] == "enabled" for sample in primary)
                )
                self.assertEqual(len(auxiliary), expected_auxiliary_count)
                event_auxiliary = [
                    sample
                    for sample in auxiliary
                    if sample["sample_type"] == "event"
                ]
                self.assertEqual(len(event_auxiliary), 1)
                event = event_auxiliary[0]
                self.assertEqual(event["dataset_id"], "rollout_round0")
                self.assertEqual(event["event_type"], "interaction_candidate")
                self.assertEqual(event["episode_outcome"], "success")
                self.assertEqual(event["action_loss"], "disabled")
                self.assertEqual(event["sample_role"], "success_auxiliary")
                self.assertEqual(
                    event["window_selection"], "core_start_anchor"
                )


if __name__ == "__main__":
    unittest.main()
