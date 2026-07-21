from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from scripts.everobot import extract_state_line_events as extractor
from fastwam.everobot_schema import SCHEMA_VERSION, with_manifest_hash


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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest_sample(row: dict[str, object]) -> dict[str, object]:
    outcome = str(row["episode_outcome"])
    return {
        "sample_type": "episode",
        "sample_id": f"sample-{int(row['episode_index']):06d}",
        "dataset_id": str(row["dataset_id"]),
        "episode_id": str(row["episode_id"]),
        "episode_index": int(row["episode_index"]),
        "round_id": str(row["round_id"]),
        "collection_round": int(row["collection_round"]),
        "task": str(row["task"]),
        "episode_outcome": outcome,
        "event_outcome": outcome,
        "start_frame": 0,
        "end_frame": int(row["length"]),
        "action_loss": "enabled" if outcome == "success" else "disabled",
        "sample_role": "success_episode" if outcome == "success" else "failure_episode",
        "sample_stride": 1,
        "split": str(row["split"]),
    }


def _write_tail_inputs(
    root: Path,
    *,
    episodes: list[dict[str, object]],
    cutoff_rows: list[dict[str, object]],
    report_updates: dict[str, object] | None = None,
) -> tuple[Path, Path, Path]:
    source_manifest_path = root / "source_manifest.json"
    samples = [_manifest_sample(row) for row in episodes]
    source_manifest = with_manifest_hash(
        {
            "format": "EveRobotTrainManifest",
            "schema_version": SCHEMA_VERSION,
            "manifest_name": "tail-source",
            "eve_root": str(root / "eve"),
            "frame_interval": "half_open",
            "selection": {},
            "dataset_roots": {"rollout_r0": "/dataset"},
            "source_round_ids": sorted({str(row["round_id"]) for row in episodes}),
            "source_hashes": {
                "round_meta_sha256": "1" * 64,
                "episode_meta_sha256": "2" * 64,
                "event_meta_sha256": "3" * 64,
            },
            "num_samples": len(samples),
            "samples": samples,
        }
    )
    source_manifest_path.write_text(
        json.dumps(source_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    cutoffs_path = root / "tail_consensus_cutoffs.jsonl"
    cutoffs_path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in cutoff_rows
        ),
        encoding="utf-8",
    )
    report: dict[str, object] = {
        "format": extractor.TAIL_CONSENSUS_FORMAT,
        "schema_version": extractor.TAIL_CONSENSUS_SCHEMA_VERSION,
        "status": "ok",
        "rule": dict(extractor.TAIL_CONSENSUS_RULE),
        "manifest_sha256": _sha256(source_manifest_path),
        "outputs": {
            "cutoff_records_file": cutoffs_path.name,
            "cutoff_records_sha256": _sha256(cutoffs_path),
            "cutoff_records_count": len(cutoff_rows),
        },
    }
    if report_updates:
        report.update(report_updates)
    report_path = root / "tail_consensus_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report_path, cutoffs_path, source_manifest_path


def _cutoff_row(
    index: int, *, length: int, cutoff: int, should_trim: bool
) -> dict[str, object]:
    return {
        "dataset_id": "rollout_r0",
        "episode_index": index,
        "num_frames": length,
        "cutoff_frame": cutoff,
        "should_trim": should_trim,
        "consensus_material": should_trim,
        "instability_flags": (
            [] if should_trim else ["decision_disagreement", "cutoff_disagreement"]
        ),
    }


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

    def test_success_and_failure_candidates_are_unknown_with_safe_action_loss(
        self,
    ) -> None:
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
        self.assertEqual(sum(float(row["event_weight"]) for row in success_rows), 1.0)
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
        self.assertEqual(rows[0]["annotation"]["parameters"]["max_candidate"], 5)
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

        def score_writer(path: Path, score_rows: list[dict[str, object]]) -> None:
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
            self.assertIn(first["method_id"], Path(first["scores_path"]).name)
            self.assertEqual(len(captured_scores), 48)
            ledger_lines = (
                (eve_root / "event_meta.jsonl").read_text(encoding="utf-8").splitlines()
            )
            self.assertEqual(len(ledger_lines), first["num_candidates"])
            self.assertTrue(
                all(
                    json.loads(line)["event_outcome"] == "unknown"
                    for line in ledger_lines
                )
            )

    def test_tail_consensus_is_failure_only_and_preserves_full_scores(self) -> None:
        length = 24
        episodes = [
            episode_row(0, outcome="success", length=length),
            episode_row(1, outcome="failure", length=length),
            episode_row(128, outcome="failure", length=length),
        ]
        states = {
            str(episodes[0]["episode_id"]): states_with_turn(
                length, turn_frame=7, amplitude=5.0
            ),
            str(episodes[1]["episode_id"]): states_with_turn(
                length, turn_frame=7, amplitude=7.0
            ),
            str(episodes[2]["episode_id"]): states_with_turn(
                length, turn_frame=10, amplitude=6.0
            ),
        }
        states[str(episodes[1]["episode_id"])][18:, 0] -= 12.0
        calibration = extractor.fit_robust_calibration(
            [(str(episodes[0]["episode_id"]), states[str(episodes[0]["episode_id"])])],
            low_quantile=0.0,
            high_quantile=1.0,
        )
        parameters = extractor.ExtractionParameters(
            median_window=1,
            ema_alpha=1.0,
            high_threshold=0.15,
            low_threshold=0.05,
            max_gap=0,
            min_run=1,
            pre_padding=1,
            post_padding=1,
            min_window=3,
        )

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            report_path, cutoffs_path, source_manifest_path = _write_tail_inputs(
                root,
                episodes=episodes,
                cutoff_rows=[
                    _cutoff_row(1, length=length, cutoff=14, should_trim=True),
                    _cutoff_row(128, length=length, cutoff=length, should_trim=False),
                ],
            )
            bundle = extractor.load_tail_consensus_bundle(
                report_path=report_path,
                cutoffs_path=cutoffs_path,
                source_manifest_path=source_manifest_path,
                episode_rows=episodes,
            )
            calibration_path = root / "frozen_calibration.json"
            extractor.persist_calibration(calibration_path, calibration)
            loaded_calibration = extractor.load_calibration(
                calibration_path,
                expected_algorithm_version=extractor.ALGORITHM_VERSION,
            )
            captured: dict[str, list[dict[str, object]]] = {}

            def writer(label: str):
                def capture(path: Path, score_rows: list[dict[str, object]]) -> None:
                    del path
                    captured[label] = [dict(row) for row in score_rows]

                return capture

            common = {
                "episode_rows": episodes,
                "state_loader": lambda row: states[str(row["episode_id"])],
                "parameters": parameters,
                "calibration_split": "train",
                "low_quantile": 0.0,
                "high_quantile": 1.0,
                "algorithm_version": extractor.ALGORITHM_VERSION,
                "scores_path": None,
                "append_ledger": True,
                "calibration": loaded_calibration,
                "calibration_source_path": calibration_path,
            }
            base_root = root / "base"
            tail_root = root / "tail"
            base = extractor.run_extraction(
                eve_root=base_root,
                scores_writer=writer("base"),
                **common,
            )
            tail = extractor.run_extraction(
                eve_root=tail_root,
                scores_writer=writer("tail"),
                tail_consensus=bundle,
                **common,
            )

            self.assertEqual(base["calibration_path"], str(calibration_path.resolve()))
            self.assertEqual(tail["calibration_path"], str(calibration_path.resolve()))
            self.assertNotEqual(base["method_id"], tail["method_id"])
            self.assertEqual(
                tail["tail_consensus"]["tail_consensus_cutoffs_sha256"],
                _sha256(cutoffs_path),
            )
            self.assertEqual(len(captured["base"]), length * len(episodes))
            self.assertEqual(len(captured["tail"]), length * len(episodes))

            base_events = [
                json.loads(line)
                for line in (base_root / "event_meta.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            tail_events = [
                json.loads(line)
                for line in (tail_root / "event_meta.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            trimmed_events = [
                row for row in tail_events if int(row["episode_index"]) == 1
            ]
            self.assertTrue(trimmed_events)
            self.assertTrue(all(int(row["end_frame"]) <= 14 for row in trimmed_events))
            self.assertTrue(
                all(
                    row["annotation"]["tail_consensus"]["effective_end_frame"] == 14
                    for row in trimmed_events
                )
            )

            def normalized_event(row: dict[str, object]) -> dict[str, object]:
                normalized = copy.deepcopy(row)
                normalized.pop("event_id")
                annotation = normalized["annotation"]
                annotation.pop("method_id")
                annotation.pop("scores_artifact")
                annotation.pop("scores_sha256")
                annotation.pop("tail_consensus", None)
                return normalized

            def normalized_score(row: dict[str, object]) -> dict[str, object]:
                normalized = dict(row)
                for field in (
                    "method_id",
                    "effective_end_frame",
                    "visible_for_candidate_extraction",
                    "tail_should_trim",
                    "tail_consensus_report_sha256",
                    "tail_consensus_cutoffs_sha256",
                    "tail_source_manifest_sha256",
                ):
                    normalized.pop(field, None)
                return normalized

            for episode_index in (0, 128):
                base_semantics = [
                    normalized_event(row)
                    for row in base_events
                    if int(row["episode_index"]) == episode_index
                ]
                tail_semantics = [
                    normalized_event(row)
                    for row in tail_events
                    if int(row["episode_index"]) == episode_index
                ]
                self.assertEqual(base_semantics, tail_semantics)

            for episode_index in (0, 128):
                episode_id = str(episodes[0 if episode_index == 0 else 2]["episode_id"])
                base_scores = [
                    normalized_score(row)
                    for row in captured["base"]
                    if row["episode_id"] == episode_id
                ]
                tail_scores = [
                    normalized_score(row)
                    for row in captured["tail"]
                    if row["episode_id"] == episode_id
                ]
                self.assertEqual(base_scores, tail_scores)
            episode_128_scores = [
                row
                for row in captured["tail"]
                if row["episode_id"] == episodes[2]["episode_id"]
            ]
            self.assertTrue(
                all(row["effective_end_frame"] == length for row in episode_128_scores)
            )
            self.assertTrue(
                all(
                    row["visible_for_candidate_extraction"]
                    for row in episode_128_scores
                )
            )
            trimmed_scores = [
                row
                for row in captured["tail"]
                if row["episode_id"] == episodes[1]["episode_id"]
            ]
            self.assertTrue(
                all(
                    not row["visible_for_candidate_extraction"]
                    and not row["active_candidate"]
                    for row in trimmed_scores[14:]
                )
            )

    def test_tail_consensus_inputs_fail_closed(self) -> None:
        episodes = [
            episode_row(0, outcome="success", length=24),
            episode_row(1, outcome="failure", length=24),
            episode_row(128, outcome="failure", length=24),
        ]
        valid_cutoffs = [
            _cutoff_row(1, length=24, cutoff=14, should_trim=True),
            _cutoff_row(128, length=24, cutoff=24, should_trim=False),
        ]
        scenarios = (
            "format",
            "schema",
            "status",
            "rule",
            "cutoff_hash",
            "cutoff_count",
            "source_hash",
            "duplicate",
            "missing",
            "extra",
            "frame_count",
            "cutoff_bounds",
        )
        for scenario in scenarios:
            with self.subTest(scenario=scenario), TemporaryDirectory() as temporary:
                root = Path(temporary)
                cutoffs = [dict(row) for row in valid_cutoffs]
                updates: dict[str, object] = {}
                if scenario == "format":
                    updates["format"] = "WrongFormat"
                elif scenario == "schema":
                    updates["schema_version"] = "2.0"
                elif scenario == "status":
                    updates["status"] = "failed"
                elif scenario == "rule":
                    updates["rule"] = {"trim_condition": "any_input_should_trim"}
                elif scenario == "duplicate":
                    cutoffs.append(dict(cutoffs[0]))
                elif scenario == "missing":
                    cutoffs.pop()
                elif scenario == "extra":
                    cutoffs.append(
                        _cutoff_row(999, length=24, cutoff=24, should_trim=False)
                    )
                elif scenario == "frame_count":
                    cutoffs[0]["num_frames"] = 23
                    cutoffs[0]["cutoff_frame"] = 13
                elif scenario == "cutoff_bounds":
                    cutoffs[0]["cutoff_frame"] = 0
                report_path, cutoffs_path, manifest_path = _write_tail_inputs(
                    root,
                    episodes=episodes,
                    cutoff_rows=cutoffs,
                    report_updates=updates,
                )
                if scenario == "cutoff_hash":
                    cutoffs_path.write_text(
                        cutoffs_path.read_text(encoding="utf-8") + "\n",
                        encoding="utf-8",
                    )
                elif scenario == "cutoff_count":
                    report = json.loads(report_path.read_text(encoding="utf-8"))
                    report["outputs"]["cutoff_records_count"] += 1
                    report_path.write_text(json.dumps(report), encoding="utf-8")
                elif scenario == "source_hash":
                    report = json.loads(report_path.read_text(encoding="utf-8"))
                    report["manifest_sha256"] = "f" * 64
                    report_path.write_text(json.dumps(report), encoding="utf-8")

                with self.assertRaises(ValueError):
                    extractor.load_tail_consensus_bundle(
                        report_path=report_path,
                        cutoffs_path=cutoffs_path,
                        source_manifest_path=manifest_path,
                        episode_rows=episodes,
                    )

    def test_cli_accepts_frozen_calibration_and_complete_tail_bundle(self) -> None:
        args = extractor.parse_args(
            [
                "--eve-root",
                "/tmp/eve",
                "--calibration",
                "/tmp/calibration.json",
                "--tail-consensus-report",
                "/tmp/report.json",
                "--tail-consensus-cutoffs",
                "/tmp/cutoffs.jsonl",
                "--source-manifest",
                "/tmp/source.json",
            ]
        )
        self.assertEqual(args.calibration, Path("/tmp/calibration.json"))
        self.assertEqual(args.tail_consensus_report, Path("/tmp/report.json"))
        self.assertEqual(args.tail_consensus_cutoffs, Path("/tmp/cutoffs.jsonl"))
        self.assertEqual(args.source_manifest, Path("/tmp/source.json"))

    def test_append_detects_identity_collision_without_changing_old_content(
        self,
    ) -> None:
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

        def score_writer(path: Path, score_rows: list[dict[str, object]]) -> None:
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
