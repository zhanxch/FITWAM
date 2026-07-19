from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "validate_s0_formal_outputs_under_test",
    ROOT / "scripts" / "water_plant" / "validate_s0_formal_outputs.py",
)
validate = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(validate)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ValidateS0FormalOutputsTest(unittest.TestCase):
    def fixture(self, root: Path):
        dataset = root / "dataset"
        (dataset / "meta").mkdir(parents=True)
        (dataset / "data" / "chunk-000").mkdir(parents=True)
        (dataset / "videos" / "chunk-000" / "observation.images.front").mkdir(
            parents=True
        )
        (dataset / "meta" / "info.json").write_text(
            json.dumps({"total_episodes": 200}),
            encoding="utf-8",
        )
        episodes = [
            {"episode_index": episode, "length": 1, "tasks": ["Water plant."]}
            for episode in range(200)
        ]
        outcomes = [
            {
                "episode_index": episode,
                "attempt_index": episode,
                "seed": 1000 + episode,
                "success": episode % 2 == 0,
                "outcome": "success" if episode % 2 == 0 else "failure",
                "source": "dexjoco_env",
            }
            for episode in range(200)
        ]
        write_jsonl(dataset / "meta" / "episodes.jsonl", episodes)
        ledger = dataset / "meta" / "episode_outcomes.jsonl"
        write_jsonl(ledger, outcomes)
        (dataset / "data" / "chunk-000" / "episode_000000.parquet").write_bytes(
            b"parquet"
        )
        (
            dataset
            / "videos"
            / "chunk-000"
            / "observation.images.front"
            / "episode_000000.mp4"
        ).write_bytes(b"video")

        protocol = root / "collection_protocol.json"
        protocol.write_text(
            json.dumps(
                {
                    "collection": {
                        "kind": "formal",
                        "episodes": 200,
                        "base_seed": 1000,
                        "seed_stop_exclusive": 1200,
                    }
                }
            ),
            encoding="utf-8",
        )
        outcome_validation = root / "outcome_validation.json"
        outcome_validation.write_text(
            json.dumps(
                {
                    "status": "valid",
                    "dataset_root": str(dataset.resolve()),
                    "episodes": 200,
                    "successes": 100,
                    "failures": 100,
                    "check_media": True,
                    "outcome_ledger": str(ledger.resolve()),
                    "outcome_ledger_sha256": sha256(ledger),
                    "physical_validation": {
                        "episodes": 200,
                        "frames": 200,
                        "videos": 200,
                    },
                }
            ),
            encoding="utf-8",
        )
        return dataset, protocol, outcome_validation

    def test_valid_outputs_bind_protocol_seeds_and_all_dataset_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset, protocol, outcome_validation = self.fixture(Path(tmp))
            report = validate.validate_formal_outputs(
                protocol, dataset, outcome_validation
            )
            self.assertEqual(report["status"], "valid")
            self.assertEqual(report["expected_episodes"], 200)
            self.assertEqual(report["protocol"]["sha256"], sha256(protocol))
            self.assertEqual(report["protocol"]["base_seed"], 1000)
            files = report["dataset"]["fingerprint"]["files"]
            self.assertEqual(
                [record["path"] for record in files],
                sorted(
                    path.relative_to(dataset).as_posix()
                    for path in dataset.rglob("*")
                    if path.is_file()
                ),
            )
            for record in files:
                path = dataset / record["path"]
                self.assertEqual(record["size_bytes"], path.stat().st_size)
                self.assertEqual(record["sha256"], sha256(path))
            self.assertTrue(any(record["path"].startswith("meta/") for record in files))
            self.assertTrue(any(record["path"].endswith(".parquet") for record in files))
            self.assertTrue(any(record["path"].endswith(".mp4") for record in files))
            self.assertEqual(
                report["dataset"]["episode_outcomes_sha256"],
                sha256(dataset / "meta" / "episode_outcomes.jsonl"),
            )

    def test_fingerprint_is_deterministic_and_changes_with_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset, protocol, outcome_validation = self.fixture(Path(tmp))
            first = validate.validate_formal_outputs(
                protocol, dataset, outcome_validation
            )["dataset"]["fingerprint"]["sha256"]
            second = validate.validate_formal_outputs(
                protocol, dataset, outcome_validation
            )["dataset"]["fingerprint"]["sha256"]
            self.assertEqual(first, second)
            parquet = dataset / "data" / "chunk-000" / "episode_000000.parquet"
            parquet.write_bytes(b"changed parquet")
            third = validate.validate_formal_outputs(
                protocol, dataset, outcome_validation
            )["dataset"]["fingerprint"]["sha256"]
            self.assertNotEqual(first, third)

    def test_duplicate_seed_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset, protocol, outcome_validation = self.fixture(Path(tmp))
            ledger = dataset / "meta" / "episode_outcomes.jsonl"
            rows = [
                json.loads(line)
                for line in ledger.read_text(encoding="utf-8").splitlines()
            ]
            rows[-1]["seed"] = rows[-2]["seed"]
            write_jsonl(ledger, rows)
            payload = json.loads(outcome_validation.read_text(encoding="utf-8"))
            payload["outcome_ledger_sha256"] = sha256(ledger)
            outcome_validation.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate seed"):
                validate.validate_formal_outputs(
                    protocol, dataset, outcome_validation
                )

    def test_noncontiguous_episode_coverage_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset, protocol, outcome_validation = self.fixture(Path(tmp))
            ledger = dataset / "meta" / "episode_outcomes.jsonl"
            rows = [
                json.loads(line)
                for line in ledger.read_text(encoding="utf-8").splitlines()
            ]
            rows[-1]["episode_index"] = 200
            write_jsonl(ledger, rows)
            payload = json.loads(outcome_validation.read_text(encoding="utf-8"))
            payload["outcome_ledger_sha256"] = sha256(ledger)
            outcome_validation.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "cover episode_index"):
                validate.validate_formal_outputs(
                    protocol, dataset, outcome_validation
                )

    def test_protocol_episode_count_and_seed_stop_are_strict(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset, protocol, outcome_validation = self.fixture(Path(tmp))
            payload = json.loads(protocol.read_text(encoding="utf-8"))
            payload["collection"]["episodes"] = 199
            protocol.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "episodes must be 200"):
                validate.validate_formal_outputs(
                    protocol, dataset, outcome_validation
                )

            payload["collection"]["episodes"] = 200
            payload["collection"]["seed_stop_exclusive"] = 1199
            protocol.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "seed_stop_exclusive"):
                validate.validate_formal_outputs(
                    protocol, dataset, outcome_validation
                )

    def test_stale_outcome_validation_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset, protocol, outcome_validation = self.fixture(Path(tmp))
            payload = json.loads(outcome_validation.read_text(encoding="utf-8"))
            payload["outcome_ledger_sha256"] = "0" * 64
            outcome_validation.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "ledger SHA256"):
                validate.validate_formal_outputs(
                    protocol, dataset, outcome_validation
                )

    def test_missing_media_validation_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset, protocol, outcome_validation = self.fixture(Path(tmp))
            payload = json.loads(outcome_validation.read_text(encoding="utf-8"))
            payload["check_media"] = False
            outcome_validation.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "--check-media"):
                validate.validate_formal_outputs(
                    protocol, dataset, outcome_validation
                )

    def test_failed_revalidation_removes_stale_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset, protocol, outcome_validation = self.fixture(root)
            report_path = root / "formal_protocol_validation.json"
            report_path.write_text('{"status":"valid"}', encoding="utf-8")
            payload = json.loads(protocol.read_text(encoding="utf-8"))
            payload["collection"]["episodes"] = 199
            protocol.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "episodes must be 200"):
                validate.validate_and_write(
                    protocol,
                    dataset,
                    outcome_validation,
                    report_path,
                )
            self.assertFalse(report_path.exists())

    def test_atomic_report_is_written_outside_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset, protocol, outcome_validation = self.fixture(root)
            report_path = root / "formal_protocol_validation.json"
            report = validate.validate_and_write(
                protocol,
                dataset,
                outcome_validation,
                report_path,
            )
            self.assertEqual(json.loads(report_path.read_text())["status"], "valid")
            self.assertEqual(report["status"], "valid")
            self.assertEqual(
                list(root.glob(".formal_protocol_validation.json.incomplete-*")),
                [],
            )
            with self.assertRaisesRegex(ValueError, "outside the dataset"):
                validate.validate_and_write(
                    protocol,
                    dataset,
                    outcome_validation,
                    dataset / "formal_protocol_validation.json",
                )


if __name__ == "__main__":
    unittest.main()
