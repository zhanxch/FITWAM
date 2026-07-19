from __future__ import annotations

import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_offline_rollout_eval_under_test",
    ROOT / "scripts" / "water_plant" / "run_offline_rollout_eval.py",
)
rollout = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(rollout)

VALIDATOR_SPEC = importlib.util.spec_from_file_location(
    "validate_offline_rollout_eval_integration",
    ROOT / "scripts" / "water_plant" / "validate_offline_rollout_eval.py",
)
validator = importlib.util.module_from_spec(VALIDATOR_SPEC)
assert VALIDATOR_SPEC.loader is not None
VALIDATOR_SPEC.loader.exec_module(validator)


class OfflineRolloutFixture:
    def __init__(self, root: Path):
        self.root = root
        self.repo = root / "repo"
        self.run_dir = self.repo / "runs" / "B1"
        self.checkpoint = (
            self.run_dir / "checkpoints" / "weights" / "step_006500.pt"
        )
        self.output_root = root / "outputs" / "B1_step6500"
        self.env_prefix = root / "env"
        self.dexjoco_root = root / "dexjoco"
        self.stats = root / "stats.json"
        self.text_cache = root / "text_cache"

        self._write(self.run_dir / "config.yaml", b"variant: B1\n")
        self._write(self.checkpoint, b"checkpoint")
        self._write(self.env_prefix / "bin" / "python", b"#!/bin/sh\n")
        os.chmod(self.env_prefix / "bin" / "python", 0o755)
        self._write(
            self.dexjoco_root / "configs" / "rand_obj" / "water_plant.yaml",
            b"task: water_plant\n",
        )
        (self.dexjoco_root / "dexjoco").mkdir(parents=True)
        self._write(self.stats, b'{"mean": 0, "std": 1}\n')
        self._write(self.text_cache / "water_plant.pt", b"text-cache")

        for path in rollout.CODE_FILE_PATHS.values():
            if path.is_absolute():
                continue
            self._write(self.repo / path, f"# {path}\n".encode())

    @staticmethod
    def _write(path: Path, payload: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)

    def argv(
        self,
        *,
        checkpoint: Path | None = None,
        checkpoint_step: int = 6500,
        output_root: Path | None = None,
        inference_seed: int | None = 314159,
    ) -> list[str]:
        argv = [
            "--variant",
            "B1",
            "--run-dir",
            str(self.run_dir),
            "--checkpoint",
            str(checkpoint or self.checkpoint),
            "--checkpoint-step",
            str(checkpoint_step),
            "--output-root",
            str(output_root or self.output_root),
            "--base-seed",
            "20260718",
            "--gpus",
            "0,1,2,3",
            "--env-prefix",
            str(self.env_prefix),
            "--dexjoco-root",
            str(self.dexjoco_root),
            "--dataset-stats-path",
            str(self.stats),
            "--text-cache-dir",
            str(self.text_cache),
        ]
        if inference_seed is not None:
            argv.extend(["--inference-seed", str(inference_seed)])
        return argv

    def successful_subprocess(self, command, *, cwd, check):
        self.last_cwd = cwd
        self.commands.append(list(command))
        self.assert_protocol_frozen()
        if Path(command[1]).name == "run_multi_gpu_dexjoco_eval.py":
            (self.output_root / "eval").mkdir(parents=True)
            (self.output_root / "eval" / "summary.json").write_text(
                "{}\n", encoding="utf-8"
            )
        elif Path(command[1]).name == "validate_offline_rollout_eval.py":
            for filename in (
                "validated_summary.json",
                "validated_summary.csv",
                "episodes.csv",
            ):
                (self.output_root / filename).write_text(
                    "validated\n", encoding="utf-8"
                )
        return mock.Mock(returncode=0)

    def assert_protocol_frozen(self) -> None:
        protocol = self.output_root / "protocol.json"
        resolved_config = self.output_root / "resolved_config.yaml"
        if not protocol.is_file() or not resolved_config.is_file():
            raise AssertionError("Protocol and config must be frozen before launch")

    def install_subprocess_state(self) -> None:
        self.commands: list[list[str]] = []
        self.last_cwd: Path | None = None


class RunOfflineRolloutEvalTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.fixture = OfflineRolloutFixture(
            Path(self.temporary_directory.name)
        )
        self.fixture.install_subprocess_state()
        self.project_patch = mock.patch.object(
            rollout, "PROJECT_ROOT", self.fixture.repo
        )
        self.project_patch.start()
        self.addCleanup(self.project_patch.stop)

    def run_success(self, argv: list[str] | None = None) -> int:
        with mock.patch.object(
            rollout.subprocess,
            "run",
            side_effect=self.fixture.successful_subprocess,
        ):
            return rollout.main(argv or self.fixture.argv())

    def test_refuses_stale_output_without_touching_it(self):
        self.fixture.output_root.mkdir(parents=True)
        marker = self.fixture.output_root / "keep.txt"
        marker.write_text("do not touch\n", encoding="utf-8")
        with mock.patch.object(rollout.subprocess, "run") as subprocess_run:
            result = rollout.main(self.fixture.argv())
        self.assertEqual(result, 2)
        self.assertEqual(marker.read_text(encoding="utf-8"), "do not touch\n")
        self.assertFalse((self.fixture.output_root / "failure.json").exists())
        subprocess_run.assert_not_called()

    def test_checkpoint_step_mismatch_fails_before_launch(self):
        wrong_checkpoint = (
            self.fixture.run_dir
            / "checkpoints"
            / "weights"
            / "step_006000.pt"
        )
        wrong_checkpoint.write_bytes(b"wrong-step")
        with mock.patch.object(rollout.subprocess, "run") as subprocess_run:
            result = rollout.main(
                self.fixture.argv(
                    checkpoint=wrong_checkpoint,
                    checkpoint_step=6500,
                )
            )
        self.assertEqual(result, 1)
        failure = json.loads(
            (self.fixture.output_root / "failure.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(failure["stage"], "preflight")
        self.assertIn("does not match", failure["message"])
        subprocess_run.assert_not_called()

    def test_evaluator_command_hard_enforces_formal_protocol(self):
        result = self.run_success()
        self.assertEqual(result, 0)
        evaluator = self.fixture.commands[0]
        expected = [
            str((self.fixture.env_prefix / "bin" / "python").resolve()),
            str(
                (
                    self.fixture.repo
                    / "scripts/dexjoco_async/run_multi_gpu_dexjoco_eval.py"
                ).resolve()
            ),
            "--gpus",
            "0,1,2,3",
            "--episodes",
            "50",
            "--seed",
            "20260718",
            "--inference-seed",
            "314159",
            "--server-conda-env",
            str(self.fixture.env_prefix.resolve()),
            "--client-conda-env",
            str(self.fixture.env_prefix.resolve()),
            "--run-dir",
            str(self.fixture.run_dir.resolve()),
            "--checkpoint",
            str(self.fixture.checkpoint.resolve()),
            "--dataset-stats-path",
            str(self.fixture.stats.resolve()),
            "--text-embedding-cache-dir",
            str(self.fixture.text_cache.resolve()),
            "--no-load-text-encoder",
            "--task-config-dir",
            str(
                (
                    self.fixture.dexjoco_root / "configs" / "rand_obj"
                ).resolve()
            ),
            "--tasks",
            "water_plant",
            "--dexjoco-py-root",
            str((self.fixture.dexjoco_root / "dexjoco").resolve()),
            "--replan-steps",
            "25",
            "--control-mode",
            "blocking",
            "--async-fallback",
            "wait",
            "--max-env-steps",
            "1500",
            "--save-video",
            "--save-actions",
            "--no-randomize",
            "--no-randomize-dynamics",
            "--no-action-clip",
            "--output-dir",
            str((self.fixture.output_root / "eval").resolve()),
        ]
        self.assertEqual(evaluator, expected)
        self.assertEqual(self.fixture.last_cwd, self.fixture.repo)

    def test_protocol_freezes_inputs_seeds_and_exact_shards(self):
        result = self.run_success()
        self.assertEqual(result, 0)
        protocol = json.loads(
            (self.fixture.output_root / "protocol.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(protocol["variant"], "B1")
        self.assertEqual(protocol["task"], "water_plant")
        self.assertEqual(protocol["checkpoint"]["step"], 6500)
        self.assertEqual(
            protocol["checkpoint"]["path"],
            str(self.fixture.checkpoint.resolve()),
        )
        self.assertGreater(protocol["checkpoint"]["size_bytes"], 0)
        self.assertEqual(len(protocol["checkpoint"]["sha256"]), 64)
        self.assertEqual(
            protocol["config"]["copied_path"],
            str((self.fixture.output_root / "resolved_config.yaml").resolve()),
        )
        self.assertEqual(
            protocol["config"]["sha256"],
            protocol["config"]["copied_sha256"],
        )
        self.assertEqual(protocol["evaluation"]["episodes"], 50)
        self.assertEqual(protocol["evaluation"]["gpus"], [0, 1, 2, 3])
        self.assertEqual(protocol["evaluation"]["inference_seed"], 314159)
        self.assertEqual(
            protocol["evaluation"]["expected_seeds"],
            list(range(20260718, 20260768)),
        )
        shards = protocol["evaluation"]["shards"]
        self.assertEqual(
            [shard["episodes"] for shard in shards],
            [13, 13, 12, 12],
        )
        self.assertEqual(
            [shard["base_seed"] for shard in shards],
            [20260718, 20260731, 20260744, 20260756],
        )
        self.assertEqual(
            [shard["seed_stop_exclusive"] for shard in shards],
            [20260731, 20260744, 20260756, 20260768],
        )
        self.assertEqual(protocol["normalization"]["kind"], "dataset_stats")
        self.assertEqual(protocol["text_cache"]["file_count"], 1)
        self.assertIn("multi_gpu_evaluator", protocol["code_files"])
        self.assertIn("validator", protocol["code_files"])
        self.assertEqual(
            protocol["provenance"]["checkpoint_sha256"],
            protocol["checkpoint"]["sha256"],
        )
        self.assertEqual(
            protocol["provenance"]["inference_seed"],
            protocol["evaluation"]["inference_seed"],
        )
        self.assertEqual(
            protocol["argv"]["evaluator"], self.fixture.commands[0]
        )

    def test_default_inference_seed_and_validator_invocation(self):
        result = self.run_success(
            self.fixture.argv(inference_seed=None)
        )
        self.assertEqual(result, 0)
        protocol = json.loads(
            (self.fixture.output_root / "protocol.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(
            protocol["evaluation"]["inference_seed"],
            protocol["evaluation"]["base_seed"],
        )
        validator = self.fixture.commands[1]
        self.assertEqual(
            validator,
            [
                str((self.fixture.env_prefix / "bin" / "python").resolve()),
                str(
                    (
                        self.fixture.repo
                        / "scripts/water_plant/validate_offline_rollout_eval.py"
                    ).resolve()
                ),
                "--summary",
                str(
                    (
                        self.fixture.output_root / "eval" / "summary.json"
                    ).resolve()
                ),
                "--protocol",
                str((self.fixture.output_root / "protocol.json").resolve()),
                "--report-json",
                str(
                    (
                        self.fixture.output_root / "validated_summary.json"
                    ).resolve()
                ),
                "--report-csv",
                str(
                    (
                        self.fixture.output_root / "validated_summary.csv"
                    ).resolve()
                ),
                "--episodes-csv",
                str((self.fixture.output_root / "episodes.csv").resolve()),
            ],
        )
        for filename in (
            "validated_summary.json",
            "validated_summary.csv",
            "episodes.csv",
        ):
            self.assertGreater(
                (self.fixture.output_root / filename).stat().st_size, 0
            )

    def test_frozen_wrapper_protocol_is_accepted_by_real_validator(self):
        output_root = rollout._create_output_root(self.fixture.output_root)
        args = rollout.parse_args(self.fixture.argv())
        protocol, _, _ = rollout._prepare_protocol(
            args,
            output_root=output_root,
            wrapper_argv=["run_offline_rollout_eval.py", *self.fixture.argv()],
        )
        protocol_path = output_root / "protocol.json"
        rollout._atomic_write_json(protocol_path, protocol)

        assignments = protocol["evaluation"]["shards"]
        episode_results = []
        shard_rows = {assignment["shard_id"]: [] for assignment in assignments}
        for episode in range(50):
            seed = args.base_seed + episode
            assignment = next(
                item
                for item in assignments
                if item["base_seed"] <= seed < item["seed_stop_exclusive"]
            )
            video = output_root / "artifacts" / f"episode_{episode:03d}.mp4"
            actions = (
                output_root / "artifacts" / f"episode_{episode:03d}_actions.npz"
            )
            video.parent.mkdir(parents=True, exist_ok=True)
            video.write_bytes(b"video")
            actions.write_bytes(b"actions")
            row = {
                "episode": episode,
                "seed": seed,
                "shard": assignment["shard_id"],
                "success": episode < 25,
                "steps": 500 if episode < 25 else 1500,
                "video_path": str(video),
                "actions_path": str(actions),
            }
            episode_results.append(row)
            shard_rows[assignment["shard_id"]].append(row)

        shards = []
        for assignment in assignments:
            rows = shard_rows[assignment["shard_id"]]
            successes = sum(int(row["success"]) for row in rows)
            shards.append(
                {
                    "shard_id": assignment["shard_id"],
                    "episodes": len(rows),
                    "successes": successes,
                    "success_rate": successes / len(rows),
                    "base_seed": assignment["base_seed"],
                    "global_episode_start": assignment[
                        "global_episode_start"
                    ],
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
            "seed": args.base_seed,
            "inference_seed": 314159,
            "save_video": True,
            "save_actions": True,
            "randomize": False,
            "randomize_dynamics": False,
            "action_clip": False,
            "total_successes": 25,
            "overall_success_rate": 0.5,
            "shards": shards,
            "tasks": [
                {
                    "env_name": "water_plant",
                    "episodes": 50,
                    "successes": 25,
                    "success_rate": 0.5,
                    "episode_results": episode_results,
                }
            ],
        }
        summary_path = output_root / "eval" / "summary.json"
        summary_path.parent.mkdir()
        summary_path.write_text(json.dumps(summary), encoding="utf-8")

        report = validator.validate_offline_rollout_eval(
            summary_path, protocol_path
        )
        self.assertEqual(report["status"], "valid")
        self.assertEqual(report["horizons"]["600"]["successes"], 25)
        self.assertEqual(
            report["provenance"]["checkpoint_sha256"],
            protocol["checkpoint"]["sha256"],
        )


if __name__ == "__main__":
    unittest.main()
