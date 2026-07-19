from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts.everobot import validate_offline_event_pair_quality as validator


def episode(index: int, *, split: str, outcome: str) -> dict[str, object]:
    return {
        "episode_id": f"{split}-{outcome}-episode-{index}",
        "dataset_id": f"{split}-{outcome}",
        "episode_index": index,
        "split": split,
        "episode_outcome": outcome,
        "length": 100,
    }


def event(
    row: dict[str, object],
    index: int,
    *,
    long: bool = False,
) -> dict[str, object]:
    return {
        "event_id": f"{row['episode_id']}-event-{index}",
        "episode_id": row["episode_id"],
        "split": row["split"],
        "episode_outcome": row["episode_outcome"],
        "event_type": "interaction_candidate",
        "start_frame": 10,
        "end_frame": 80 if long else 30,
        "exceeds_max_candidate": long,
    }


class ValidateOfflineEventPairQualityTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.episode_path = self.root / "episode_meta.jsonl"
        self.event_path = self.root / "event_meta.jsonl"
        self.pair_path = self.root / "pairs.jsonl"
        self.diagnostics_path = self.root / "pair_diagnostics.json"
        self.output_path = self.root / "quality.json"

        self.episodes: list[dict[str, object]] = []
        self.events: list[dict[str, object]] = []
        for split in ("train", "val"):
            for outcome in ("success", "failure"):
                for index in range(2):
                    episode_row = episode(index, split=split, outcome=outcome)
                    self.episodes.append(episode_row)
                    self.events.append(event(episode_row, 0))

        by_class = {
            (str(row["split"]), str(row["episode_outcome"])): row
            for row in self.events
        }
        self.pairs: list[dict[str, object]] = []
        for split in ("train", "val"):
            pair_id = f"pair-{split}"
            self.pairs.append(
                {
                    "pair_id": pair_id,
                    "split": split,
                    "success_event_id": by_class[(split, "success")]["event_id"],
                    "failure_event_id": by_class[(split, "failure")]["event_id"],
                    "pair_weight": 0.8,
                }
            )
        paired_by_failure = {
            str(row["failure_event_id"]): [str(row["pair_id"])]
            for row in self.pairs
        }
        failure_events = [
            {
                "failure_event_id": row["event_id"],
                "split": row["split"],
                "selected": str(row["event_id"]) in paired_by_failure,
                "selected_pair_ids": paired_by_failure.get(
                    str(row["event_id"]), []
                ),
            }
            for row in self.events
            if row["episode_outcome"] == "failure"
        ]
        self.diagnostics = {
            "coverage": {
                "total_failure_events": len(failure_events),
                "selected_failure_events": len(self.pairs),
                "failure_event_coverage": len(self.pairs) / len(failure_events),
                "selected_pairs": len(self.pairs),
            },
            "pair_weight_distribution": {"count": len(self.pairs)},
            "failure_events": failure_events,
        }
        self._write_inputs()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )

    def _write_inputs(self) -> None:
        self._write_jsonl(self.episode_path, self.episodes)
        self._write_jsonl(self.event_path, self.events)
        self._write_jsonl(self.pair_path, self.pairs)
        self.diagnostics_path.write_text(
            json.dumps(self.diagnostics), encoding="utf-8"
        )

    def _relaxed_args(self) -> list[str]:
        return [
            "--episode-meta",
            str(self.episode_path),
            "--event-meta",
            str(self.event_path),
            "--pair-ledger",
            str(self.pair_path),
            "--pair-diagnostics",
            str(self.diagnostics_path),
            "--output",
            str(self.output_path),
            "--min-candidate-episode-coverage",
            "0",
            "--min-train-outcome-candidates",
            "0",
            "--min-val-outcome-candidates",
            "0",
            "--min-events-per-episode-median",
            "0",
            "--max-events-per-episode-median",
            "10",
            "--max-events-per-episode-p95",
            "10",
            "--max-long-candidate-ratio",
            "1",
            "--min-train-pairs",
            "0",
            "--min-val-pairs",
            "0",
            "--min-train-failure-coverage",
            "0",
            "--min-val-failure-coverage",
            "0",
            "--min-train-failure-episodes",
            "0",
            "--min-val-failure-episodes",
            "0",
            "--max-train-single-episode-pair-share",
            "1",
            "--max-val-single-episode-pair-share",
            "1",
            "--min-pair-weight-median",
            "0",
            "--max-low-pair-weight-ratio",
            "1",
        ]

    @staticmethod
    def _run(args: list[str]) -> int:
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            return validator.main(args)

    def test_relaxed_fixture_passes_and_reports_all_metrics(self) -> None:
        inputs_before = {
            path: path.read_bytes()
            for path in (
                self.episode_path,
                self.event_path,
                self.pair_path,
                self.diagnostics_path,
            )
        }

        self.assertEqual(self._run(self._relaxed_args()), 0)

        report = json.loads(self.output_path.read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "passed")
        train_success = report["candidate_metrics"]["train"]["success"]
        self.assertEqual(train_success["episode_count"], 2)
        self.assertEqual(train_success["candidate_count"], 2)
        self.assertEqual(train_success["candidate_episode_coverage"], 1.0)
        self.assertEqual(train_success["events_per_episode"]["median"], 1.0)
        self.assertEqual(train_success["events_per_episode"]["p95"], 1.0)
        train_pairs = report["pair_metrics"]["train"]
        self.assertEqual(train_pairs["pair_count"], 1)
        self.assertEqual(train_pairs["failure_event_coverage"], 0.5)
        self.assertEqual(train_pairs["unique_failure_episode_count"], 1)
        self.assertEqual(
            train_pairs["max_single_failure_episode_pair_share"], 1.0
        )
        self.assertEqual(train_pairs["pair_weight_median"], 0.8)
        self.assertEqual(train_pairs["low_pair_weight_ratio"], 0.0)
        self.assertEqual(report["diagnostics_consistency"]["status"], "consistent")
        self.assertTrue(all(check["passed"] for check in report["checks"]))
        self.assertEqual(
            inputs_before,
            {path: path.read_bytes() for path in inputs_before},
        )
        self.assertEqual(
            list(self.output_path.parent.glob(f".{self.output_path.name}.*.tmp")),
            [],
        )

    def test_default_thresholds_fail_and_still_write_atomic_report(self) -> None:
        args = self._relaxed_args()[:10]
        self.assertEqual(self._run(args), 1)

        report = json.loads(self.output_path.read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "failed")
        failed = {
            (row["name"], row["scope"]) for row in report["failed_checks"]
        }
        self.assertIn(("candidate_count", "train/success"), failed)
        self.assertIn(("pair_count", "train"), failed)
        self.assertIn(("max_single_failure_episode_pair_share", "train"), failed)
        self.assertEqual(
            list(self.output_path.parent.glob(f".{self.output_path.name}.*.tmp")),
            [],
        )

    def test_long_candidate_ratio_uses_explicit_marker(self) -> None:
        self.events[0]["exceeds_max_candidate"] = True
        self._write_inputs()
        args = self._relaxed_args()
        cutoff_index = args.index("--max-long-candidate-ratio") + 1
        args[cutoff_index] = "0.4"

        self.assertEqual(self._run(args), 1)
        report = json.loads(self.output_path.read_text(encoding="utf-8"))
        metric = report["candidate_metrics"]["train"]["success"]
        self.assertEqual(metric["long_candidate_count"], 1)
        self.assertEqual(metric["long_candidate_ratio"], 0.5)
        self.assertTrue(
            any(
                check["name"] == "long_candidate_ratio"
                and check["scope"] == "train/success"
                and not check["passed"]
                for check in report["checks"]
            )
        )

    def test_diagnostics_mismatch_is_input_error(self) -> None:
        self.diagnostics["coverage"]["selected_pairs"] = 999
        self._write_inputs()

        self.assertEqual(self._run(self._relaxed_args()), 2)
        report = json.loads(self.output_path.read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "error")
        self.assertIn("selected_pairs", report["error"])

    def test_pair_metrics_are_computed_per_split(self) -> None:
        val_failure_id = str(self.pairs[1]["failure_event_id"])
        val_failure_event = next(
            row for row in self.events if row["event_id"] == val_failure_id
        )
        second_val_failure = next(
            row
            for row in self.events
            if row["split"] == "val"
            and row["episode_outcome"] == "failure"
            and row["event_id"] != val_failure_id
        )
        self.pairs.append(
            {
                "pair_id": "pair-val-second",
                "split": "val",
                "success_event_id": self.pairs[1]["success_event_id"],
                "failure_event_id": second_val_failure["event_id"],
                "pair_weight": 0.05,
            }
        )
        del val_failure_event
        self.diagnostics["coverage"]["selected_pairs"] = len(self.pairs)
        self.diagnostics["coverage"]["selected_failure_events"] = len(self.pairs)
        self.diagnostics["coverage"]["failure_event_coverage"] = (
            len(self.pairs) / len(self.diagnostics["failure_events"])
        )
        self.diagnostics["pair_weight_distribution"]["count"] = len(self.pairs)
        for row in self.diagnostics["failure_events"]:
            matching = [
                pair["pair_id"]
                for pair in self.pairs
                if pair["failure_event_id"] == row["failure_event_id"]
            ]
            row["selected"] = bool(matching)
            row["selected_pair_ids"] = matching
        self._write_inputs()

        self.assertEqual(self._run(self._relaxed_args()), 0)
        report = json.loads(self.output_path.read_text(encoding="utf-8"))
        train = report["pair_metrics"]["train"]
        val = report["pair_metrics"]["val"]
        self.assertEqual(train["pair_count"], 1)
        self.assertEqual(train["failure_event_coverage"], 0.5)
        self.assertEqual(val["pair_count"], 2)
        self.assertEqual(val["failure_event_coverage"], 1.0)
        self.assertEqual(val["unique_failure_episode_count"], 2)
        self.assertEqual(val["max_single_failure_episode_pair_share"], 0.5)
        self.assertAlmostEqual(val["pair_weight_median"], 0.425)
        self.assertEqual(val["low_pair_weight_ratio"], 0.5)

    def test_default_thresholds_match_formal_recommendation(self) -> None:
        args = validator.parse_args(self._relaxed_args()[:10])
        thresholds = validator._thresholds_from_args(args)
        self.assertEqual(thresholds["min_candidate_episode_coverage"], 0.8)
        self.assertEqual(thresholds["min_train_outcome_candidates"], 32)
        self.assertEqual(thresholds["min_val_outcome_candidates"], 8)
        self.assertEqual(thresholds["min_events_per_episode_median"], 1.0)
        self.assertEqual(thresholds["max_events_per_episode_median"], 6.0)
        self.assertEqual(thresholds["max_events_per_episode_p95"], 10.0)
        self.assertEqual(thresholds["max_long_candidate_ratio"], 0.1)
        self.assertEqual(thresholds["min_train_pairs"], 32)
        self.assertEqual(thresholds["min_val_pairs"], 8)
        self.assertEqual(thresholds["min_train_failure_coverage"], 0.30)
        self.assertEqual(thresholds["min_val_failure_coverage"], 0.25)
        self.assertEqual(thresholds["min_train_failure_episodes"], 16)
        self.assertEqual(thresholds["min_val_failure_episodes"], 4)
        self.assertEqual(
            thresholds["max_train_single_episode_pair_share"],
            0.10,
        )
        self.assertEqual(
            thresholds["max_val_single_episode_pair_share"],
            0.25,
        )
        self.assertEqual(thresholds["min_pair_weight_median"], 0.10)
        self.assertEqual(thresholds["max_low_pair_weight_ratio"], 0.25)

    def test_output_cannot_overwrite_read_only_input(self) -> None:
        args = self._relaxed_args()
        output_index = args.index("--output") + 1
        args[output_index] = str(self.event_path)
        before = self.event_path.read_bytes()

        self.assertEqual(self._run(args), 2)
        self.assertEqual(self.event_path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
