from __future__ import annotations

import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "validate_offline_rollout_eval_under_test",
    ROOT / "scripts" / "water_plant" / "validate_offline_rollout_eval.py",
)
validate = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(validate)


class ValidateOfflineRolloutEvalTest(unittest.TestCase):
    def fixture(self, root: Path) -> tuple[Path, Path]:
        base_seed = 2026071900
        inference_seed = 314159
        shard_counts = [13, 13, 12, 12]
        assignments = []
        seed_cursor = base_seed
        episode_cursor = 0
        for shard_id, count in enumerate(shard_counts):
            assignments.append(
                {
                    "shard_id": shard_id,
                    "gpu": shard_id,
                    "episodes": count,
                    "base_seed": seed_cursor,
                    "seed_stop_exclusive": seed_cursor + count,
                    "global_episode_start": episode_cursor,
                }
            )
            seed_cursor += count
            episode_cursor += count

        protocol = {
            "variant": "B0",
            "episodes": 50,
            "gpus": [0, 1, 2, 3],
            "replan_steps": 25,
            "max_env_steps": 1500,
            "task": "water_plant",
            "save_video": True,
            "save_actions": True,
            "randomize": False,
            "randomize_dynamics": False,
            "action_clip": False,
            "base_seed": base_seed,
            "seed_stop_exclusive": base_seed + 50,
            "inference_seed": inference_seed,
            "shards": assignments,
            "provenance": {
                "run_id": "wp_B0_step_006500_eval",
                "checkpoint_sha256": "a" * 64,
                "resolved_config_sha256": "b" * 64,
            },
        }
        protocol_path = root / "protocol.json"
        protocol_path.write_text(json.dumps(protocol), encoding="utf-8")

        episode_results = []
        shard_rows: list[list[dict]] = [[], [], [], []]
        for episode in range(50):
            if episode < 10:
                success, steps = True, 500
            elif episode < 20:
                success, steps = True, 800
            elif episode < 30:
                success, steps = True, 1200
            else:
                success, steps = False, 1500
            shard = next(
                assignment["shard_id"]
                for assignment in assignments
                if assignment["base_seed"]
                <= base_seed + episode
                < assignment["seed_stop_exclusive"]
            )
            video = root / "artifacts" / f"episode_{episode:03d}.mp4"
            actions = root / "artifacts" / f"episode_{episode:03d}_actions.npz"
            video.parent.mkdir(parents=True, exist_ok=True)
            video.write_bytes(b"video")
            actions.write_bytes(b"actions")
            row = {
                "episode": episode,
                "seed": base_seed + episode,
                "shard": shard,
                "success": success,
                "steps": steps,
                "video_path": str(video),
                "actions_path": str(actions),
            }
            episode_results.append(row)
            shard_rows[shard].append(row)

        shards = []
        for assignment in assignments:
            rows = shard_rows[assignment["shard_id"]]
            successes = sum(1 for row in rows if row["success"])
            shards.append(
                {
                    "shard_id": assignment["shard_id"],
                    "episodes": len(rows),
                    "successes": successes,
                    "success_rate": successes / len(rows),
                    "base_seed": assignment["base_seed"],
                    "summary_path": str(root / f"shard_{assignment['shard_id']}" / "summary.json"),
                }
            )

        summary = {
            "total_episodes": 50,
            "episodes_per_task": 50,
            "num_tasks": 1,
            "num_shards": 4,
            "gpus": [0, 1, 2, 3],
            "replan_steps": 25,
            "max_env_steps": 1500,
            "task": "water_plant",
            "seed": base_seed,
            "inference_seed": inference_seed,
            "save_video": True,
            "save_video": True,
            "save_actions": True,
            "randomize": False,
            "randomize_dynamics": False,
            "action_clip": False,
            "total_successes": 30,
            "overall_success_rate": 0.6,
            "shards": shards,
            "tasks": [
                {
                    "env_name": "water_plant",
                    "episodes": 50,
                    "successes": 30,
                    "success_rate": 0.6,
                    "episode_results": episode_results,
                }
            ],
        }
        summary_path = root / "summary.json"
        summary_path.write_text(json.dumps(summary), encoding="utf-8")
        return summary_path, protocol_path

    @staticmethod
    def load(path: Path) -> dict:
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def write(path: Path, payload: dict) -> None:
        path.write_text(json.dumps(payload), encoding="utf-8")

    def test_valid_13_13_12_12_shards_and_milestones_are_written_atomically(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary, protocol = self.fixture(root)
            report_json = root / "reports" / "validated.json"
            report_csv = root / "reports" / "horizons.csv"
            episodes_csv = root / "reports" / "episodes.csv"

            report = validate.validate_and_write(
                summary, protocol, report_json, report_csv, episodes_csv
            )

            self.assertEqual(report["status"], "valid")
            self.assertEqual(
                [row["episodes"] for row in report["settings"]["shards"]],
                [13, 13, 12, 12],
            )
            self.assertEqual(report["horizons"]["600"]["successes"], 10)
            self.assertEqual(report["horizons"]["1000"]["successes"], 20)
            self.assertEqual(report["horizons"]["1500"]["successes"], 30)
            self.assertEqual(report["horizons"]["600"]["rate"], 0.2)
            self.assertEqual(report["horizons"]["1000"]["rate"], 0.4)
            self.assertEqual(report["horizons"]["1500"]["rate"], 0.6)
            self.assertEqual(report["median_successful_step"], 800.0)
            self.assertEqual(report["provenance"]["inference_seed"], 314159)
            self.assertIsNone(report["episodes"][-1]["success_step"])
            self.assertEqual(report["episodes"][0]["success_step"], 500)

            self.assertEqual(self.load(report_json), report)
            with report_csv.open(newline="", encoding="utf-8") as stream:
                horizon_rows = list(csv.DictReader(stream))
            self.assertEqual([row["successes"] for row in horizon_rows], ["10", "20", "30"])
            with episodes_csv.open(newline="", encoding="utf-8") as stream:
                episode_rows = list(csv.DictReader(stream))
            self.assertEqual(len(episode_rows), 50)
            self.assertEqual(episode_rows[-1]["outcome"], "failure")
            self.assertEqual(episode_rows[-1]["success_step"], "")
            self.assertEqual(list((root / "reports").glob("*.incomplete-*")), [])

    def test_missing_seed_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary_path, protocol = self.fixture(root)
            summary = self.load(summary_path)
            summary["tasks"][0]["episode_results"][-1]["seed"] += 1000
            self.write(summary_path, summary)
            with self.assertRaisesRegex(ValueError, "seed set mismatch.*missing=.*unexpected="):
                validate.validate_offline_rollout_eval(summary_path, protocol)

    def test_duplicate_seed_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary_path, protocol = self.fixture(root)
            summary = self.load(summary_path)
            summary["tasks"][0]["episode_results"][-1]["seed"] = summary["tasks"][0][
                "episode_results"
            ][-2]["seed"]
            self.write(summary_path, summary)
            with self.assertRaisesRegex(ValueError, "duplicate seed"):
                validate.validate_offline_rollout_eval(summary_path, protocol)

    def test_protocol_or_inference_seed_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary_path, protocol = self.fixture(root)
            summary = self.load(summary_path)
            summary["inference_seed"] += 1
            self.write(summary_path, summary)
            with self.assertRaisesRegex(ValueError, "summary.inference_seed"):
                validate.validate_offline_rollout_eval(summary_path, protocol)

            summary, _ = self.fixture(root)
            protocol_payload = self.load(protocol)
            protocol_payload["replan_steps"] = 24
            self.write(protocol, protocol_payload)
            with self.assertRaisesRegex(ValueError, "protocol.replan_steps"):
                validate.validate_offline_rollout_eval(summary, protocol)

    def test_missing_or_empty_artifact_is_rejected_and_stale_outputs_removed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary, protocol = self.fixture(root)
            payload = self.load(summary)
            artifact = Path(payload["tasks"][0]["episode_results"][3]["video_path"])
            artifact.unlink()
            outputs = [root / "valid.json", root / "horizons.csv", root / "episodes.csv"]
            for output in outputs:
                output.write_text("stale", encoding="utf-8")
            with self.assertRaises(FileNotFoundError):
                validate.validate_and_write(summary, protocol, *outputs)
            self.assertTrue(all(not output.exists() for output in outputs))

            summary, protocol = self.fixture(root)
            payload = self.load(summary)
            Path(payload["tasks"][0]["episode_results"][4]["actions_path"]).write_bytes(b"")
            with self.assertRaisesRegex(ValueError, "actions_path must be nonempty"):
                validate.validate_offline_rollout_eval(summary, protocol)

    def test_stale_shard_metadata_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary_path, protocol = self.fixture(root)
            summary = self.load(summary_path)
            summary["shards"][0]["episodes"] = 12
            self.write(summary_path, summary)
            with self.assertRaisesRegex(ValueError, "summary shard 0.episodes"):
                validate.validate_offline_rollout_eval(summary_path, protocol)

            summary_path, protocol = self.fixture(root)
            summary = self.load(summary_path)
            summary["tasks"][0]["episode_results"][13]["shard"] = 0
            self.write(summary_path, summary)
            with self.assertRaisesRegex(ValueError, "shard 0 episode row count"):
                validate.validate_offline_rollout_eval(summary_path, protocol)


if __name__ == "__main__":
    unittest.main()
