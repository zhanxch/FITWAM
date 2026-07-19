from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from scripts.everobot import extract_state_line_events as extractor


def episode_row(
    index: int,
    *,
    outcome: str,
    split: str = "train",
    length: int = 16,
) -> dict[str, object]:
    dataset_id = "rollout_r0"
    return {
        "episode_id": f"{dataset_id}:episode:{index:06d}",
        "round_id": f"{dataset_id}:round:0",
        "dataset_id": dataset_id,
        "dataset_root": "/dataset",
        "episode_index": index,
        "task_name": "water_plant",
        "task": "Water plant",
        "source_policy": "fastwam",
        "collection_round": 0,
        "episode_outcome": outcome,
        "length": length,
        "split": split,
    }


def states_with_turn(length: int, *, turn_frame: int, amplitude: float) -> np.ndarray:
    values = np.arange(length, dtype=np.float64)
    states = np.stack([values, values * 0.5], axis=1)
    states[turn_frame:, 0] += amplitude
    states[turn_frame:, 1] -= amplitude * 0.25
    return states


class StateLineExtractionTest(unittest.TestCase):
    def test_calibration_identity_is_stable_and_split_specific(self) -> None:
        train = states_with_turn(18, turn_frame=8, amplitude=4.0)
        first = extractor.fit_robust_calibration(
            [("train:0", train)],
            low_quantile=0.10,
            high_quantile=0.95,
        )
        second = extractor.fit_robust_calibration(
            [("train:0", train.copy())],
            low_quantile=0.10,
            high_quantile=0.95,
        )
        changed = extractor.fit_robust_calibration(
            [("train:0", train), ("val:0", train * 10.0)],
            low_quantile=0.10,
            high_quantile=0.95,
        )

        self.assertEqual(first["calibration_id"], second["calibration_id"])
        self.assertNotEqual(first["calibration_id"], changed["calibration_id"])
        self.assertEqual(first["calibration_split"], "train")
        same_statistics_different_content = extractor.fit_robust_calibration(
            [("train:0", train[::-1].copy())],
            low_quantile=0.10,
            high_quantile=0.95,
        )
        self.assertNotEqual(
            first["calibration_input_sha256"],
            same_statistics_different_content["calibration_input_sha256"],
        )
        self.assertNotEqual(
            first["calibration_id"],
            same_statistics_different_content["calibration_id"],
        )

    def test_success_and_failure_candidates_are_unknown_with_safe_action_loss(self) -> None:
        calibration = {
            **extractor.fit_robust_calibration(
                [
                    (
                        "fit:0",
                        states_with_turn(20, turn_frame=8, amplitude=3.0),
                    )
                ],
                low_quantile=0.0,
                high_quantile=1.0,
            )
        }
        parameters = extractor.ExtractionParameters(
            median_window=1,
            ema_alpha=1.0,
            high_threshold=0.2,
            low_threshold=0.1,
            max_gap=0,
            min_run=1,
            pre_padding=1,
            post_padding=1,
            min_window=3,
        )
        state = states_with_turn(16, turn_frame=7, amplitude=6.0)

        success_rows, success_scores = extractor.extract_episode_rows(
            episode_row(0, outcome="success"),
            state,
            calibration,
            parameters=parameters,
            scores_artifact="annotations/event_scores.parquet",
            method_id=extractor.make_method_id(
                algorithm_version=extractor.ALGORITHM_VERSION,
                calibration_id=str(calibration["calibration_id"]),
                parameters=parameters,
            ),
        )
        failure_rows, _ = extractor.extract_episode_rows(
            episode_row(1, outcome="failure"),
            state,
            calibration,
            parameters=parameters,
            scores_artifact="annotations/event_scores.parquet",
            method_id=extractor.make_method_id(
                algorithm_version=extractor.ALGORITHM_VERSION,
                calibration_id=str(calibration["calibration_id"]),
                parameters=parameters,
            ),
        )

        self.assertTrue(success_rows)
        self.assertTrue(failure_rows)
        self.assertTrue(all(row["event_outcome"] == "unknown" for row in success_rows))
        self.assertTrue(all(row["event_outcome"] == "unknown" for row in failure_rows))
        self.assertTrue(all(row["action_loss"] == "enabled" for row in success_rows))
        self.assertTrue(all(row["action_loss"] == "disabled" for row in failure_rows))
        self.assertTrue(
            all(row["event_type"] == "interaction_candidate" for row in success_rows)
        )
        self.assertEqual(len(success_scores), 16)
        self.assertEqual(
            sum(float(row["event_weight"]) for row in success_rows), 1.0
        )
        self.assertEqual(
            sum(float(row["episode_sampling_weight"]) for row in success_rows),
            1.0,
        )
        self.assertTrue(
            all(
                float(row["absolute_confidence"])
                == float(row["annotation"]["confidence"])
                for row in success_rows
            )
        )
        event_id = str(success_rows[0]["event_id"])
        self.assertIn(calibration["calibration_id"], event_id)
        self.assertIn(extractor.ALGORITHM_VERSION, event_id)

    def test_long_candidate_is_preserved_and_marked_for_loader_windowing(self) -> None:
        states = states_with_turn(20, turn_frame=5, amplitude=6.0)
        calibration = extractor.fit_robust_calibration(
            [("fit:0", states)],
            low_quantile=0.0,
            high_quantile=1.0,
        )
        parameters = extractor.ExtractionParameters(
            median_window=1,
            ema_alpha=1.0,
            high_threshold=0.0,
            low_threshold=0.0,
            max_gap=0,
            min_run=1,
            pre_padding=0,
            post_padding=0,
            min_window=1,
            max_candidate=5,
        )

        rows, _ = extractor.extract_episode_rows(
            episode_row(0, outcome="success", length=20),
            states,
            calibration,
            parameters=parameters,
            scores_artifact="annotations/event_scores.parquet",
            method_id=extractor.make_method_id(
                algorithm_version=extractor.ALGORITHM_VERSION,
                calibration_id=str(calibration["calibration_id"]),
                parameters=parameters,
            ),
        )

        self.assertEqual(len(rows), 1)
        self.assertGreater(int(rows[0]["end_frame"]) - int(rows[0]["start_frame"]), 5)
        self.assertIs(rows[0]["exceeds_max_candidate"], True)
        self.assertEqual(
            rows[0]["annotation"]["parameters"]["max_candidate"], 5
        )
        self.assertEqual(
            rows[0]["annotation"]["long_candidate_policy"],
            "preserve_coarse_event_and_defer_sliding_window_to_loader",
        )

    def test_run_uses_only_train_rows_for_calibration_and_is_idempotent(self) -> None:
        rows = [
            episode_row(0, outcome="success", split="train"),
            episode_row(1, outcome="failure", split="train"),
            episode_row(2, outcome="success", split="val"),
        ]
        states = {
            str(rows[0]["episode_id"]): states_with_turn(
                16, turn_frame=6, amplitude=3.0
            ),
            str(rows[1]["episode_id"]): states_with_turn(
                16, turn_frame=8, amplitude=5.0
            ),
            str(rows[2]["episode_id"]): states_with_turn(
                16, turn_frame=4, amplitude=100.0
            ),
        }
        captured_scores: list[dict[str, object]] = []

        def score_writer(
            path: Path, score_rows: list[dict[str, object]]
        ) -> None:
            del path
            captured_scores[:] = score_rows

        with TemporaryDirectory() as temporary:
            eve_root = Path(temporary)
            parameters = extractor.ExtractionParameters(
                median_window=1,
                ema_alpha=1.0,
                high_threshold=0.2,
                low_threshold=0.1,
                max_gap=0,
                min_run=1,
                pre_padding=0,
                post_padding=0,
                min_window=1,
            )
            kwargs = {
                "eve_root": eve_root,
                "episode_rows": rows,
                "state_loader": lambda row: states[str(row["episode_id"])],
                "parameters": parameters,
                "calibration_split": "train",
                "low_quantile": 0.0,
                "high_quantile": 1.0,
                "algorithm_version": extractor.ALGORITHM_VERSION,
                "scores_path": None,
                "append_ledger": True,
                "scores_writer": score_writer,
            }
            first = extractor.run_extraction(**kwargs)
            second = extractor.run_extraction(**kwargs)

            self.assertEqual(first["num_episodes"], 3)
            self.assertEqual(first["calibration"]["num_episodes"], 2)
            self.assertEqual(first["num_appended_candidates"], first["num_candidates"])
            self.assertEqual(second["num_appended_candidates"], 0)
            self.assertEqual(first["method_id"], second["method_id"])
            self.assertIn(
                first["method_id"], Path(first["scores_path"]).name
            )
            self.assertEqual(len(captured_scores), 48)
            ledger_lines = (
                eve_root / "event_meta.jsonl"
            ).read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(ledger_lines), first["num_candidates"])
            self.assertTrue(
                all(json.loads(line)["event_outcome"] == "unknown" for line in ledger_lines)
            )

    def test_append_detects_identity_collision_without_changing_old_content(self) -> None:
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "event_meta.jsonl"
            original = {"event_id": "event-1", "value": 1, "dataset_root": "/old"}
            path.write_text(json.dumps(original) + "\n", encoding="utf-8")
            before = path.read_bytes()

            self.assertEqual(
                extractor.append_event_rows(
                    path,
                    [{"event_id": "event-1", "value": 1, "dataset_root": "/new"}],
                ),
                0,
            )
            self.assertEqual(path.read_bytes(), before)
            with self.assertRaisesRegex(ValueError, "identity collision"):
                extractor.append_event_rows(
                    path,
                    [{"event_id": "event-1", "value": 2, "dataset_root": "/new"}],
                )
            self.assertEqual(path.read_bytes(), before)

    def test_parameter_change_versions_method_event_id_and_score_path(self) -> None:
        calibration = extractor.fit_robust_calibration(
            [
                (
                    "fit:0",
                    states_with_turn(16, turn_frame=7, amplitude=4.0),
                )
            ],
            low_quantile=0.0,
            high_quantile=1.0,
        )
        first_parameters = extractor.ExtractionParameters(max_candidate=96)
        second_parameters = extractor.ExtractionParameters(
            high_threshold=0.60,
            max_candidate=96,
        )
        capped_parameters = extractor.ExtractionParameters(
            max_candidate=96,
            max_candidates_per_episode=10,
        )
        first_method = extractor.make_method_id(
            algorithm_version=extractor.ALGORITHM_VERSION,
            calibration_id=str(calibration["calibration_id"]),
            parameters=first_parameters,
        )
        second_method = extractor.make_method_id(
            algorithm_version=extractor.ALGORITHM_VERSION,
            calibration_id=str(calibration["calibration_id"]),
            parameters=second_parameters,
        )
        self.assertNotEqual(first_method, second_method)
        self.assertNotEqual(
            first_method,
            extractor.make_method_id(
                algorithm_version=extractor.ALGORITHM_VERSION,
                calibration_id=str(calibration["calibration_id"]),
                parameters=capped_parameters,
            ),
        )
        self.assertNotEqual(
            f"event_scores_{first_method}.parquet",
            f"event_scores_{second_method}.parquet",
        )
        self.assertNotEqual(
            extractor.stable_event_id(
                episode_row(0, outcome="success"),
                candidate_index=0,
                algorithm_version=extractor.ALGORITHM_VERSION,
                calibration_id=str(calibration["calibration_id"]),
                method_id=first_method,
            ),
            extractor.stable_event_id(
                episode_row(0, outcome="success"),
                candidate_index=0,
                algorithm_version=extractor.ALGORITHM_VERSION,
                calibration_id=str(calibration["calibration_id"]),
                method_id=second_method,
            ),
        )

    def test_ledger_collision_preflight_prevents_score_write(self) -> None:
        rows = [episode_row(0, outcome="success", split="train")]
        states = {
            str(rows[0]["episode_id"]): states_with_turn(
                16, turn_frame=6, amplitude=5.0
            )
        }
        writes: list[Path] = []

        def score_writer(
            path: Path, score_rows: list[dict[str, object]]
        ) -> None:
            del score_rows
            writes.append(path)

        with TemporaryDirectory() as temporary:
            eve_root = Path(temporary)
            kwargs = {
                "eve_root": eve_root,
                "episode_rows": rows,
                "state_loader": lambda row: states[str(row["episode_id"])],
                "parameters": extractor.ExtractionParameters(
                    median_window=1,
                    ema_alpha=1.0,
                    high_threshold=0.2,
                    low_threshold=0.1,
                    min_run=1,
                ),
                "calibration_split": "train",
                "low_quantile": 0.0,
                "high_quantile": 1.0,
                "algorithm_version": extractor.ALGORITHM_VERSION,
                "scores_path": None,
                "append_ledger": True,
                "scores_writer": score_writer,
            }
            first = extractor.run_extraction(**kwargs)
            ledger_path = eve_root / "event_meta.jsonl"
            ledger_rows = [
                json.loads(line)
                for line in ledger_path.read_text(encoding="utf-8").splitlines()
            ]
            ledger_rows[0]["action_loss"] = "disabled"
            ledger_path.write_text(
                "".join(json.dumps(row) + "\n" for row in ledger_rows),
                encoding="utf-8",
            )
            corrupted_ledger = ledger_path.read_bytes()
            writes_before = list(writes)

            with self.assertRaisesRegex(ValueError, "identity collision"):
                extractor.run_extraction(**kwargs)

            self.assertEqual(writes, writes_before)
            self.assertEqual(ledger_path.read_bytes(), corrupted_ledger)
            self.assertEqual(first["num_appended_candidates"], first["num_candidates"])

    def test_select_rejects_duplicate_episode_identity(self) -> None:
        row = episode_row(0, outcome="success")
        with self.assertRaisesRegex(ValueError, "Duplicate episode_id"):
            extractor.select_episode_rows([row, row])


if __name__ == "__main__":
    unittest.main()
