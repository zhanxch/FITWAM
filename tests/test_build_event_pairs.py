from __future__ import annotations

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts.everobot import build_event_pairs


def episode(
    dataset_id: str,
    episode_index: int,
    outcome: str,
    task: str,
) -> dict[str, object]:
    return {
        "episode_id": f"{dataset_id}:episode:{episode_index:06d}",
        "dataset_id": dataset_id,
        "episode_index": episode_index,
        "episode_outcome": outcome,
        "task_name": task,
        "length": 100,
    }


def event(
    dataset_id: str,
    episode_index: int,
    outcome: str,
    task: str,
    *,
    suffix: str = "0",
) -> dict[str, object]:
    return {
        "event_id": f"{dataset_id}:event:{episode_index:06d}:{suffix}",
        "episode_id": f"{dataset_id}:episode:{episode_index:06d}",
        "dataset_id": dataset_id,
        "episode_index": episode_index,
        "episode_outcome": outcome,
        "task_name": task,
        "event_type": "interaction_candidate",
        "split": "train",
        "start_frame": 10,
        "end_frame": 30,
        "absolute_confidence": 0.9,
        "event_weight": 0.8,
    }


def feature(
    event_id: str,
    progress: float,
    pre_state: list[float],
    action: list[float],
) -> dict[str, object]:
    return {
        "event_id": event_id,
        "progress": progress,
        "pre_state_embedding": pre_state,
        "action_embedding": action,
    }


def config(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "pairing_version": "test_v1",
        "matching": "one_to_one",
        "max_success_uses": 1,
        "max_failure_uses": 1,
        "max_progress_delta": 0.12,
        "max_pre_state_distance": 0.5,
        "min_action_divergence": 0.5,
        "tau_progress": 0.08,
        "tau_state": 1.0,
        "event_types": ["interaction_candidate"],
        "splits": ["train"],
    }
    values.update(overrides)
    return values


class BuildEventPairsTest(unittest.TestCase):
    def basic_rows(
        self,
    ) -> tuple[
        list[dict[str, object]],
        list[dict[str, object]],
        list[dict[str, object]],
    ]:
        episodes = [
            episode("successes", 0, "success", "water_plant"),
            episode("successes", 1, "success", "water_plant"),
            episode("failures", 0, "failure", "water_plant"),
            episode("failures", 1, "failure", "water_plant"),
        ]
        events = [
            event("successes", 0, "success", "water_plant"),
            event("successes", 1, "success", "water_plant"),
            event("failures", 0, "failure", "water_plant"),
            event("failures", 1, "failure", "water_plant"),
        ]
        features = [
            feature(events[0]["event_id"], 0.20, [0.0, 0.0], [0.0, 0.0]),
            feature(events[1]["event_id"], 0.80, [1.0, 1.0], [1.0, 1.0]),
            feature(events[2]["event_id"], 0.21, [0.1, 0.1], [1.0, 1.0]),
            feature(events[3]["event_id"], 0.79, [0.9, 0.9], [0.0, 0.0]),
        ]
        return events, episodes, features

    def test_matching_is_deterministic_and_one_to_one(self) -> None:
        events, episodes, features = self.basic_rows()
        first, first_diagnostics = (
            build_event_pairs.build_event_pairs_with_diagnostics(
                list(reversed(events)),
                list(reversed(episodes)),
                list(reversed(features)),
                config=config(),
                created_at="2026-07-17T00:00:00+00:00",
            )
        )
        second, second_diagnostics = (
            build_event_pairs.build_event_pairs_with_diagnostics(
                events,
                episodes,
                features,
                config=config(),
                created_at="2026-07-17T00:00:00+00:00",
            )
        )
        self.assertEqual(first, second)
        self.assertEqual(first_diagnostics, second_diagnostics)
        self.assertEqual(len(first), 2)
        self.assertEqual(len({row["success_event_id"] for row in first}), 2)
        self.assertEqual(len({row["failure_event_id"] for row in first}), 2)

    def test_pair_weight_uses_absolute_confidence_not_episode_weight(self) -> None:
        events, episodes, features = self.basic_rows()
        for row in events:
            row["absolute_confidence"] = 0.5
            row["event_weight"] = 0.01
        rows = build_event_pairs.build_event_pairs(
            list(reversed(events)),
            list(reversed(episodes)),
            list(reversed(features)),
            config=config(),
            created_at="2026-07-17T00:00:00+00:00",
        )
        reference_events = [
            {**row, "event_weight": 1.0}
            for row in events
        ]
        reference = build_event_pairs.build_event_pairs(
            reference_events,
            episodes,
            features,
            config=config(),
            created_at="2026-07-17T00:00:00+00:00",
        )
        self.assertEqual(
            [row["pair_weight"] for row in rows],
            [row["pair_weight"] for row in reference],
        )

    def test_missing_absolute_confidence_fails_closed(self) -> None:
        events, episodes, features = self.basic_rows()
        events[0].pop("absolute_confidence")
        with self.assertRaisesRegex(ValueError, "requires absolute_confidence"):
            build_event_pairs.build_event_pairs(
                events,
                episodes,
                features,
                config=config(),
            )

    def test_annotation_confidence_is_an_explicit_compatible_source(self) -> None:
        events, episodes, features = self.basic_rows()
        events[0].pop("absolute_confidence")
        events[0]["annotation"] = {"confidence": 0.9}
        rows = build_event_pairs.build_event_pairs(
            events,
            episodes,
            features,
            config=config(),
        )
        self.assertEqual(len(rows), 2)

    def test_never_matches_across_tasks(self) -> None:
        success_episode = episode("successes", 0, "success", "water_plant")
        failure_episode = episode("failures", 0, "failure", "hammer_nail")
        success_event = event("successes", 0, "success", "water_plant")
        failure_event = event("failures", 0, "failure", "hammer_nail")
        rows = build_event_pairs.build_event_pairs(
            [success_event, failure_event],
            [success_episode, failure_episode],
            [
                feature(success_event["event_id"], 0.5, [0.0], [0.0]),
                feature(failure_event["event_id"], 0.5, [0.0], [1.0]),
            ],
            config=config(),
        )
        self.assertEqual(rows, [])

    def test_never_matches_across_splits(self) -> None:
        success_episode = {
            **episode("successes", 0, "success", "water_plant"),
            "split": "train",
        }
        failure_episode = {
            **episode("failures", 0, "failure", "water_plant"),
            "split": "val",
        }
        success_event = event("successes", 0, "success", "water_plant")
        failure_event = {
            **event("failures", 0, "failure", "water_plant"),
            "split": "val",
        }
        rows = build_event_pairs.build_event_pairs(
            [success_event, failure_event],
            [success_episode, failure_episode],
            [
                feature(success_event["event_id"], 0.5, [0.0], [0.0]),
                feature(failure_event["event_id"], 0.5, [0.0], [1.0]),
            ],
            config=config(splits=["train", "val"]),
        )
        self.assertEqual(rows, [])

    def test_all_thresholds_are_strictly_enforced(self) -> None:
        success_episode = episode("successes", 0, "success", "water_plant")
        failure_episode = episode("failures", 0, "failure", "water_plant")
        success_event = event("successes", 0, "success", "water_plant")
        failure_event = event("failures", 0, "failure", "water_plant")

        def pair_for(
            failure_progress: float,
            failure_pre_state: list[float],
            failure_action: list[float],
        ) -> list[dict[str, object]]:
            return build_event_pairs.build_event_pairs(
                [success_event, failure_event],
                [success_episode, failure_episode],
                [
                    feature(success_event["event_id"], 0.5, [0.0], [0.0]),
                    feature(
                        failure_event["event_id"],
                        failure_progress,
                        failure_pre_state,
                        failure_action,
                    ),
                ],
                config=config(),
            )

        self.assertEqual(pair_for(0.63, [0.0], [1.0]), [])
        self.assertEqual(pair_for(0.5, [0.6], [1.0]), [])
        self.assertEqual(pair_for(0.5, [0.0], [0.4]), [])
        accepted = pair_for(0.62, [0.5], [0.5])
        self.assertEqual(len(accepted), 1)
        self.assertAlmostEqual(accepted[0]["components"]["progress_delta"], 0.12)
        self.assertAlmostEqual(accepted[0]["components"]["pre_state_distance"], 0.5)
        self.assertAlmostEqual(accepted[0]["components"]["action_divergence"], 0.5)

    def test_missing_features_produces_no_fabricated_pair(self) -> None:
        success_episode = episode("successes", 0, "success", "water_plant")
        failure_episode = episode("failures", 0, "failure", "water_plant")
        success_event = event("successes", 0, "success", "water_plant")
        failure_event = event("failures", 0, "failure", "water_plant")
        rows, diagnostics = build_event_pairs.build_event_pairs_with_diagnostics(
            [success_event, failure_event],
            [success_episode, failure_episode],
            [],
            config=config(),
        )
        self.assertEqual(rows, [])
        self.assertEqual(
            diagnostics["failure_events"][0]["rejection_reason"],
            "missing_feature",
        )
        self.assertEqual(
            diagnostics["coverage"]["rejection_reason_counts"],
            {"missing_feature": 1},
        )

    def test_bounded_matching_honors_each_usage_cap(self) -> None:
        events, episodes, features = self.basic_rows()
        rows = build_event_pairs.build_event_pairs(
            events,
            episodes,
            features,
            config=config(
                matching="bounded",
                max_success_uses=2,
                max_failure_uses=1,
                max_progress_delta=1.0,
                max_pre_state_distance=2.0,
            ),
        )
        failure_counts: dict[str, int] = {}
        success_counts: dict[str, int] = {}
        for row in rows:
            success_counts[row["success_event_id"]] = (
                success_counts.get(row["success_event_id"], 0) + 1
            )
            failure_counts[row["failure_event_id"]] = (
                failure_counts.get(row["failure_event_id"], 0) + 1
            )
        self.assertTrue(all(count <= 2 for count in success_counts.values()))
        self.assertTrue(all(count <= 1 for count in failure_counts.values()))

    def test_min_pair_weight_rejects_instead_of_flooring_low_quality_pair(
        self,
    ) -> None:
        events, episodes, features = self.basic_rows()
        for row in events:
            row["absolute_confidence"] = 0.1
        rows, diagnostics = build_event_pairs.build_event_pairs_with_diagnostics(
            events,
            episodes,
            features,
            config=config(min_pair_weight=0.05),
        )
        self.assertEqual(rows, [])
        self.assertEqual(
            diagnostics["coverage"]["rejection_reason_counts"],
            {"pair_weight_threshold": 2},
        )

    def test_mutual_nearest_keeps_only_reciprocal_pairs_and_diagnoses_rest(
        self,
    ) -> None:
        episodes = [
            episode("successes", 0, "success", "water_plant"),
            episode("successes", 1, "success", "water_plant"),
            episode("failures", 0, "failure", "water_plant"),
            episode("failures", 1, "failure", "water_plant"),
        ]
        events = [
            event("successes", 0, "success", "water_plant"),
            event("successes", 1, "success", "water_plant"),
            event("failures", 0, "failure", "water_plant"),
            event("failures", 1, "failure", "water_plant"),
        ]
        features = [
            feature(events[0]["event_id"], 0.5, [0.0], [0.0]),
            feature(events[1]["event_id"], 0.5, [10.0], [0.0]),
            feature(events[2]["event_id"], 0.5, [1.0], [1.0]),
            feature(events[3]["event_id"], 0.5, [2.0], [1.0]),
        ]
        rows, diagnostics = build_event_pairs.build_event_pairs_with_diagnostics(
            events,
            episodes,
            features,
            config=config(
                matching="mutual_nearest",
                max_pre_state_distance=20.0,
            ),
            created_at="2026-07-17T00:00:00+00:00",
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["success_event_id"], events[0]["event_id"])
        self.assertEqual(rows[0]["failure_event_id"], events[2]["event_id"])
        by_failure = {
            row["failure_event_id"]: row
            for row in diagnostics["failure_events"]
        }
        self.assertTrue(by_failure[events[2]["event_id"]]["selected"])
        self.assertEqual(
            by_failure[events[3]["event_id"]]["rejection_reason"],
            "not_mutual_nearest",
        )
        self.assertEqual(
            diagnostics["coverage"]["failure_event_coverage"],
            0.5,
        )
        self.assertEqual(
            diagnostics["pair_weight_distribution"]["count"],
            1,
        )

    def calibration_rows(
        self,
    ) -> tuple[
        list[dict[str, object]],
        list[dict[str, object]],
        list[dict[str, object]],
    ]:
        episodes = [
            {**episode("train_success", 0, "success", "water_plant"), "split": "train"},
            {**episode("train_failure", 0, "failure", "water_plant"), "split": "train"},
            {**episode("val_success", 0, "success", "water_plant"), "split": "val"},
            {**episode("val_failure", 0, "failure", "water_plant"), "split": "val"},
        ]
        events = [
            event("train_success", 0, "success", "water_plant"),
            event("train_failure", 0, "failure", "water_plant"),
            {**event("val_success", 0, "success", "water_plant"), "split": "val"},
            {**event("val_failure", 0, "failure", "water_plant"), "split": "val"},
        ]
        features = [
            feature(events[0]["event_id"], 0.5, [0.0], [0.0]),
            feature(events[1]["event_id"], 0.5, [0.4], [1.2]),
            feature(events[2]["event_id"], 0.5, [10.0], [0.0]),
            feature(events[3]["event_id"], 0.5, [11.0], [3.0]),
        ]
        return events, episodes, features

    def test_calibration_is_deterministic_and_has_no_val_leakage(self) -> None:
        events, episodes, features = self.calibration_rows()
        calibration_config = config(
            splits=["train", "val"],
            max_pre_state_distance=2.0,
        )
        records, _ = build_event_pairs._collect_event_records(
            events,
            episodes,
            features,
            config=calibration_config,
        )
        first = build_event_pairs.fit_pairing_calibration(
            records,
            config=calibration_config,
        )
        changed_val_features = [
            (
                {
                    **row,
                    "pre_state_embedding": [999.0],
                    "action_embedding": [999.0],
                }
                if str(row["event_id"]).startswith("val_")
                else row
            )
            for row in features
        ]
        changed_records, _ = build_event_pairs._collect_event_records(
            list(reversed(events)),
            list(reversed(episodes)),
            list(reversed(changed_val_features)),
            config=calibration_config,
        )
        second = build_event_pairs.fit_pairing_calibration(
            changed_records,
            config=calibration_config,
        )
        self.assertEqual(first, second)
        self.assertEqual(
            first["thresholds"]["min_action_divergence"],
            1.2,
        )
        self.assertEqual(first["thresholds"]["tau_state"], 0.4)

    def test_frozen_calibration_reuses_exact_input_and_rejects_conflict(
        self,
    ) -> None:
        events, episodes, features = self.calibration_rows()
        calibration_config = config(
            splits=["train", "val"],
            max_pre_state_distance=2.0,
        )
        records, _ = build_event_pairs._collect_event_records(
            events,
            episodes,
            features,
            config=calibration_config,
        )
        calibration = build_event_pairs.fit_pairing_calibration(
            records,
            config=calibration_config,
        )
        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "pair_calibration.json"
            self.assertTrue(
                build_event_pairs.write_json_atomic_frozen(path, calibration)
            )
            self.assertFalse(
                build_event_pairs.write_json_atomic_frozen(path, calibration)
            )
            changed_features = [
                (
                    {**row, "action_embedding": [2.4]}
                    if row["event_id"] == events[1]["event_id"]
                    else row
                )
                for row in features
            ]
            changed_records, _ = build_event_pairs._collect_event_records(
                events,
                episodes,
                changed_features,
                config=calibration_config,
            )
            with self.assertRaisesRegex(ValueError, "current train inputs"):
                build_event_pairs.validate_pairing_calibration(
                    calibration,
                    changed_records,
                    config=calibration_config,
                )
            conflicting = {
                **calibration,
                "input_sha256": "0" * 64,
            }
            with self.assertRaisesRegex(ValueError, "Frozen artifact conflict"):
                build_event_pairs.write_json_atomic_frozen(path, conflicting)

    def test_immutable_prepare_is_idempotent_and_rejects_collision(self) -> None:
        events, episodes, features = self.basic_rows()
        rows = build_event_pairs.build_event_pairs(
            events,
            episodes,
            features,
            config=config(),
            created_at="2026-07-17T00:00:00+00:00",
        )
        merged, appended = build_event_pairs.prepare_immutable_pairs([], rows)
        self.assertEqual(appended, 2)
        repeated = [{**row, "created_at": "2026-07-18T00:00:00+00:00"} for row in rows]
        merged_again, appended_again = build_event_pairs.prepare_immutable_pairs(
            merged, repeated
        )
        self.assertEqual(appended_again, 0)
        self.assertEqual(merged_again, merged)

        collision = dict(rows[0])
        collision["pair_weight"] = 0.0
        with self.assertRaisesRegex(ValueError, "collision"):
            build_event_pairs.prepare_immutable_pairs(merged, [collision])

    def test_cli_rejects_changed_frozen_pair_selection_without_partial_write(
        self,
    ) -> None:
        events, episodes, features = self.basic_rows()
        with TemporaryDirectory() as temporary_directory:
            eve_root = Path(temporary_directory) / "eve"
            eve_root.mkdir()
            event_path = eve_root / "event_meta.jsonl"
            episode_path = eve_root / "episode_meta.jsonl"
            feature_path = eve_root / "features.jsonl"
            output_path = eve_root / "pairs" / "test_v1.jsonl"
            diagnostics_path = eve_root / "pairs" / "test_v1.diagnostics.json"

            def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
                path.write_text(
                    "".join(json.dumps(row) + "\n" for row in rows),
                    encoding="utf-8",
                )

            write_rows(event_path, [events[0], events[2]])
            write_rows(episode_path, [episodes[0], episodes[2]])
            write_rows(feature_path, [features[0], features[2]])
            args = argparse.Namespace(
                eve_root=eve_root,
                event_ledger=None,
                episode_ledger=None,
                features=feature_path,
                output=output_path,
                pairing_version="test_v1",
                matching="one_to_one",
                max_success_uses=1,
                max_failure_uses=1,
                max_progress_delta=0.12,
                max_pre_state_distance=0.5,
                min_action_divergence=0.5,
                tau_progress=0.08,
                tau_state=1.0,
                event_types=["interaction_candidate"],
                splits=["train"],
                fit_calibration=None,
                calibration=None,
                diagnostics_output=diagnostics_path,
                created_at="2026-07-17T00:00:00+00:00",
            )
            _, selected, appended = build_event_pairs.run(args)
            self.assertEqual((selected, appended), (1, 1))
            original_ledger = output_path.read_bytes()
            original_diagnostics = diagnostics_path.read_bytes()

            write_rows(event_path, events)
            write_rows(episode_path, episodes)
            write_rows(feature_path, features)
            with self.assertRaisesRegex(ValueError, "Frozen pair ledger conflicts"):
                build_event_pairs.run(args)

            self.assertEqual(output_path.read_bytes(), original_ledger)
            self.assertEqual(diagnostics_path.read_bytes(), original_diagnostics)

    def test_cli_writes_versioned_ledger_and_is_idempotent(self) -> None:
        events, episodes, features = self.basic_rows()
        with TemporaryDirectory() as temporary_directory:
            eve_root = Path(temporary_directory) / "eve"
            eve_root.mkdir()
            event_path = eve_root / "event_meta.jsonl"
            episode_path = eve_root / "episode_meta.jsonl"
            feature_path = eve_root / "features.jsonl"
            calibration_path = eve_root / "pair_calibration.json"
            diagnostics_path = eve_root / "pair_diagnostics.json"
            for path, rows in (
                (event_path, events),
                (episode_path, episodes),
                (feature_path, features),
            ):
                path.write_text(
                    "".join(json.dumps(row) + "\n" for row in rows),
                    encoding="utf-8",
                )
            args = argparse.Namespace(
                eve_root=eve_root,
                event_ledger=None,
                episode_ledger=None,
                features=feature_path,
                output=None,
                pairing_version="test_v1",
                matching="one_to_one",
                max_success_uses=1,
                max_failure_uses=1,
                max_progress_delta=0.12,
                max_pre_state_distance=0.5,
                min_action_divergence=0.5,
                event_types=["interaction_candidate"],
                splits=["train"],
                fit_calibration=calibration_path,
                calibration=None,
                diagnostics_output=diagnostics_path,
                created_at="2026-07-17T00:00:00+00:00",
            )
            output_path, selected, appended = build_event_pairs.run(args)
            self.assertEqual(
                output_path,
                (eve_root / "pairs" / "test_v1.jsonl").resolve(),
            )
            self.assertEqual((selected, appended), (2, 2))
            self.assertTrue(calibration_path.is_file())
            self.assertTrue(diagnostics_path.is_file())
            calibration_payload = json.loads(
                calibration_path.read_text(encoding="utf-8")
            )
            diagnostics_payload = json.loads(
                diagnostics_path.read_text(encoding="utf-8")
            )
            self.assertEqual(
                calibration_payload["calibration_split"],
                "train",
            )
            self.assertEqual(
                diagnostics_payload["pairing_config"]["threshold_source"],
                "train_calibrated",
            )
            self.assertEqual(
                diagnostics_payload["coverage"]["failure_event_coverage"],
                1.0,
            )
            _, selected_again, appended_again = build_event_pairs.run(args)
            self.assertEqual((selected_again, appended_again), (2, 0))

            written = build_event_pairs.read_jsonl(output_path)
            self.assertEqual(len(written), 2)
            self.assertTrue(
                all(row["pairing_version"] == "test_v1" for row in written)
            )
            self.assertTrue(
                all(
                    row["provenance"]["feature_table_sha256"]
                    for row in written
                )
            )

    def test_cli_accepts_multiple_split_feature_tables(self) -> None:
        args = build_event_pairs.parse_args(
            [
                "--eve-root",
                "/tmp/eve",
                "--features",
                "/tmp/train.jsonl",
                "/tmp/val.jsonl",
                "--splits",
                "train",
                "val",
            ]
        )
        self.assertEqual(
            args.features,
            [Path("/tmp/train.jsonl"), Path("/tmp/val.jsonl")],
        )
        self.assertEqual(args.splits, ["train", "val"])


if __name__ == "__main__":
    unittest.main()
