from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from scripts.everobot import build_event_pairs
from scripts.everobot import extract_event_pair_features as extractor


try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    pa = None
    pq = None


def fixed_list(values: np.ndarray, dim: int):
    flat = pa.array(np.asarray(values, dtype=np.float32).reshape(-1))
    return pa.FixedSizeListArray.from_arrays(flat, dim)


@unittest.skipIf(pa is None or pq is None, "pyarrow is required")
class ExtractEventPairFeaturesTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.eve_root = self.root / "eve"
        self.eve_root.mkdir()
        self.dataset_root = self.root / "dataset"
        (self.dataset_root / "meta").mkdir(parents=True)
        (self.dataset_root / "meta" / "info.json").write_text(
            json.dumps(
                {
                    "chunks_size": 1000,
                    "data_path": (
                        "data/chunk-{episode_chunk:03d}/"
                        "episode_{episode_index:06d}.parquet"
                    ),
                }
            ),
            encoding="utf-8",
        )
        self.episodes: list[dict[str, object]] = []
        self.events: list[dict[str, object]] = []

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def add_episode(
        self,
        episode_index: int,
        split: str,
        states: np.ndarray,
        actions: np.ndarray,
    ) -> dict[str, object]:
        path = (
            self.dataset_root
            / "data"
            / "chunk-000"
            / f"episode_{episode_index:06d}.parquet"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            pa.table(
                {
                    extractor.STATE_COLUMN: fixed_list(states, states.shape[1]),
                    extractor.ACTION_COLUMN: fixed_list(actions, actions.shape[1]),
                }
            ),
            path,
        )
        row: dict[str, object] = {
            "episode_id": f"robot:episode:{episode_index:06d}",
            "dataset_id": "robot",
            "dataset_root": str(self.dataset_root),
            "episode_index": episode_index,
            "task_name": "water_plant",
            "split": split,
            "length": len(states),
        }
        self.episodes.append(row)
        return row

    def add_event(
        self,
        episode: dict[str, object],
        *,
        start: int,
        end: int,
        suffix: str = "0",
        split: str | None = None,
    ) -> dict[str, object]:
        row: dict[str, object] = {
            "event_id": f"event:{episode['episode_index']}:{suffix}",
            "episode_id": episode["episode_id"],
            "dataset_id": episode["dataset_id"],
            "episode_index": episode["episode_index"],
            "task_name": episode["task_name"],
            "split": episode["split"] if split is None else split,
            "core_start_frame": start,
            "core_end_frame": end,
        }
        self.events.append(row)
        return row

    def write_ledgers(self) -> None:
        for name, rows in (
            ("episode_meta.jsonl", self.episodes),
            ("event_meta.jsonl", self.events),
        ):
            (self.eve_root / name).write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )

    @staticmethod
    def state_rows(length: int, offset: float = 0.0) -> np.ndarray:
        base = np.arange(length, dtype=np.float64)[:, None]
        dims = np.arange(extractor.STATE_DIM, dtype=np.float64)[None, :]
        return offset + base + dims * 0.1

    @staticmethod
    def action_rows(length: int, offset: float = 0.0) -> np.ndarray:
        base = np.arange(length, dtype=np.float64)[:, None]
        dims = np.arange(extractor.ACTION_DIM, dtype=np.float64)[None, :]
        return offset + base * 0.5 + dims * 0.2

    def fit_train(self, output_name: str = "train.jsonl") -> tuple[dict, Path, Path]:
        calibration = self.root / "calibration.json"
        output = self.root / output_name
        result = extractor.run(
            eve_root=self.eve_root,
            split="train",
            output=output,
            fit_calibration_path=calibration,
            calibration_path=None,
        )
        return result, calibration, output

    def test_train_fit_extracts_expected_progress_and_dimensions(self) -> None:
        train = self.add_episode(
            0, "train", self.state_rows(8), self.action_rows(8)
        )
        self.add_event(train, start=3, end=6)
        self.write_ledgers()
        result, calibration_path, output = self.fit_train()
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
        row = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(result["num_events"], 1)
        self.assertEqual(calibration["fit_split"], "train")
        self.assertEqual(calibration["num_frames"], 8)
        self.assertEqual(len(row["pre_state_embedding"]), 23)
        self.assertEqual(len(row["action_embedding"]), 66)
        self.assertEqual(row["pre_state_window"], [0, 3])
        self.assertEqual(row["core_interval"], [3, 6])
        self.assertAlmostEqual(row["progress"], 9 / 16)
        extractor.validate_calibration(calibration)

    def test_val_loads_train_calibration_without_refitting(self) -> None:
        train = self.add_episode(
            0, "train", self.state_rows(6), self.action_rows(6)
        )
        validation = self.add_episode(
            1, "val", self.state_rows(6, 100.0), self.action_rows(6, 50.0)
        )
        self.add_event(train, start=2, end=4)
        self.add_event(validation, start=2, end=5)
        self.write_ledgers()
        _, calibration_path, _ = self.fit_train()
        calibration_before = calibration_path.read_bytes()
        val_output = self.root / "val.jsonl"
        result = extractor.run(
            eve_root=self.eve_root,
            split="val",
            output=val_output,
            fit_calibration_path=None,
            calibration_path=calibration_path,
        )
        self.assertEqual(result["num_events"], 1)
        self.assertEqual(calibration_path.read_bytes(), calibration_before)
        row = json.loads(val_output.read_text(encoding="utf-8"))
        self.assertEqual(row["split"], "val")
        self.assertEqual(row["event_id"], "event:1:0")

    def test_event_type_filter_excludes_coarse_failure_event(self) -> None:
        train = self.add_episode(
            0, "train", self.state_rows(8), self.action_rows(8)
        )
        candidate = self.add_event(train, start=3, end=6)
        candidate["event_type"] = "interaction_candidate"
        self.events.append(
            {
                "event_id": "event:0:coarse-failure",
                "episode_id": train["episode_id"],
                "dataset_id": train["dataset_id"],
                "episode_index": train["episode_index"],
                "task_name": train["task_name"],
                "split": train["split"],
                "event_type": "failure_event",
                "start_frame": 0,
                "end_frame": 8,
            }
        )
        self.write_ledgers()

        calibration = self.root / "calibration.json"
        output = self.root / "filtered.jsonl"
        result = extractor.run(
            eve_root=self.eve_root,
            split="train",
            output=output,
            fit_calibration_path=calibration,
            calibration_path=None,
            event_types=["interaction_candidate"],
        )

        self.assertEqual(result["num_events"], 1)
        row = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(row["event_id"], candidate["event_id"])

    def test_split_leakage_guards(self) -> None:
        validation = self.add_episode(
            0, "val", self.state_rows(5), self.action_rows(5)
        )
        self.add_event(validation, start=1, end=3)
        self.write_ledgers()
        with self.assertRaisesRegex(ValueError, "only for split='train'"):
            extractor.run(
                eve_root=self.eve_root,
                split="val",
                output=self.root / "val.jsonl",
                fit_calibration_path=self.root / "bad.json",
                calibration_path=None,
            )

        fake = {
            "format": extractor.CALIBRATION_FORMAT,
            "schema_version": extractor.SCHEMA_VERSION,
            "algorithm_version": extractor.ALGORITHM_VERSION,
            "fit_split": "val",
        }
        with self.assertRaisesRegex(ValueError, "split='train'"):
            extractor.validate_calibration(fake)

    def test_boundary_uses_core_start_when_no_prior_frame(self) -> None:
        train = self.add_episode(
            0, "train", self.state_rows(4), self.action_rows(4)
        )
        self.add_event(train, start=0, end=1)
        self.write_ledgers()
        _, _, output = self.fit_train()
        row = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(row["pre_state_window"], [0, 1])
        self.assertEqual(row["core_interval"], [0, 1])
        self.assertEqual(row["action_embedding"][22:44], [0.0] * 22)
        self.assertEqual(row["action_embedding"][44:66], [0.0] * 22)

    def test_rejects_out_of_bounds_wrong_dimension_and_nan(self) -> None:
        train = self.add_episode(
            0, "train", self.state_rows(4), self.action_rows(4)
        )
        self.add_event(train, start=2, end=5)
        self.write_ledgers()
        with self.assertRaisesRegex(ValueError, "exceeds episode length"):
            self.fit_train()

        self.temporary.cleanup()
        self.setUp()
        bad_states = np.zeros((4, 22), dtype=np.float64)
        train = self.add_episode(0, "train", bad_states, self.action_rows(4))
        self.add_event(train, start=1, end=3)
        self.write_ledgers()
        with self.assertRaisesRegex(ValueError, "dimension 22 != 23"):
            self.fit_train()

        self.temporary.cleanup()
        self.setUp()
        states = self.state_rows(4)
        actions = self.action_rows(4)
        actions[2, 0] = np.nan
        train = self.add_episode(0, "train", states, actions)
        self.add_event(train, start=1, end=3)
        self.write_ledgers()
        with self.assertRaisesRegex(ValueError, "NaN or infinite"):
            self.fit_train()

    def test_deterministic_loaded_calibration_outputs(self) -> None:
        train = self.add_episode(
            0, "train", self.state_rows(7), self.action_rows(7)
        )
        self.add_event(train, start=2, end=5)
        self.write_ledgers()
        first_result, calibration_path, first_output = self.fit_train("first.jsonl")
        second_output = self.root / "second.jsonl"
        second_result = extractor.run(
            eve_root=self.eve_root,
            split="train",
            output=second_output,
            fit_calibration_path=None,
            calibration_path=calibration_path,
        )
        self.assertEqual(first_output.read_bytes(), second_output.read_bytes())
        self.assertEqual(first_result["method_id"], second_result["method_id"])
        self.assertEqual(first_result["rows_sha256"], second_result["rows_sha256"])

    def test_atomic_outputs_refuse_overwrite(self) -> None:
        train = self.add_episode(
            0, "train", self.state_rows(5), self.action_rows(5)
        )
        self.add_event(train, start=1, end=4)
        self.write_ledgers()
        _, calibration_path, output = self.fit_train()
        original = output.read_bytes()
        with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
            extractor.run(
                eve_root=self.eve_root,
                split="train",
                output=output,
                fit_calibration_path=None,
                calibration_path=calibration_path,
            )
        self.assertEqual(output.read_bytes(), original)

        existing_calibration = self.root / "existing-calibration.json"
        existing_calibration.write_text("sentinel", encoding="utf-8")
        with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
            extractor.run(
                eve_root=self.eve_root,
                split="train",
                output=self.root / "unused.jsonl",
                fit_calibration_path=existing_calibration,
                calibration_path=None,
            )
        self.assertEqual(
            existing_calibration.read_text(encoding="utf-8"), "sentinel"
        )

    def test_parquet_default_companion_is_pair_builder_compatible(self) -> None:
        train = self.add_episode(
            0, "train", self.state_rows(5), self.action_rows(5)
        )
        self.add_event(train, start=1, end=4)
        self.write_ledgers()
        calibration = self.root / "calibration.json"
        parquet = self.root / "features.parquet"
        result = extractor.run(
            eve_root=self.eve_root,
            split="train",
            output=parquet,
            fit_calibration_path=calibration,
            calibration_path=None,
        )
        companion = parquet.with_suffix(".jsonl")
        self.assertTrue(parquet.is_file())
        self.assertTrue(companion.is_file())
        self.assertEqual(result["pairing_input"], str(companion.resolve()))
        rows = build_event_pairs.read_feature_table(companion)
        self.assertEqual(len(rows), 1)
        self.assertIsInstance(rows[0]["pre_state_embedding"], list)
        self.assertEqual(len(rows[0]["pre_state_embedding"]), 23)
        self.assertEqual(len(rows[0]["action_embedding"]), 66)

    def test_cli_help(self) -> None:
        with self.assertRaises(SystemExit) as caught:
            extractor.parse_args(["--help"])
        self.assertEqual(caught.exception.code, 0)


if __name__ == "__main__":
    unittest.main()
