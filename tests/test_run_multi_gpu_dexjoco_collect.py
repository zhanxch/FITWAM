from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    utils = types.ModuleType("multi_gpu_eval_utils")
    utils.ServerSpec = type("ServerSpec", (), {})
    utils.ShardSpec = type("ShardSpec", (), {})
    for name in (
        "build_conda_command",
        "find_free_ports",
        "launch_subprocess",
        "locate_conda_sh",
        "shard_episodes",
        "terminate_process",
        "wait_for_server",
    ):
        setattr(utils, name, lambda *_args, **_kwargs: None)
    previous = sys.modules.get("multi_gpu_eval_utils")
    sys.modules["multi_gpu_eval_utils"] = utils
    try:
        spec = importlib.util.spec_from_file_location(
            "run_multi_gpu_dexjoco_collect_under_test",
            ROOT / "scripts" / "dexjoco_async" / "run_multi_gpu_dexjoco_collect.py",
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        if previous is None:
            sys.modules.pop("multi_gpu_eval_utils", None)
        else:
            sys.modules["multi_gpu_eval_utils"] = previous


orchestrator = _load_module()


class MultiGpuCollectCliTest(unittest.TestCase):
    def _collect_args(self, root: Path, **overrides):
        task_dir = root / "tasks"
        task_dir.mkdir()
        (task_dir / "water_plant.yaml").write_text("{}\n", encoding="utf-8")
        source = root / "source"
        source.mkdir()
        run_dir = root / "run"
        run_dir.mkdir()
        collect_script = root / "collect.py"
        collect_script.write_text("", encoding="utf-8")
        values = {
            "collect_script": collect_script,
            "run_dir": run_dir,
            "policy_timeout_ms": 100,
            "task_config_dir": task_dir,
            "tasks": ["water_plant"],
            "source_dataset": source,
            "text_embedding_cache_dir": None,
            "replan_steps": 25,
            "max_env_steps": 600,
            "video_fps": 30,
            "success_prompt": "Clean instruction.",
            "failure_phrase": "Failed.",
            "outcome_task_mode": "clean",
            "randomize": False,
            "randomize_dynamics": False,
            "action_clip": False,
            "clip_max_xyz_step": 0.05,
            "clip_max_dz_down": 0.03,
            "overwrite": False,
            "resume": False,
            "api_token": None,
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    @staticmethod
    def _shard(*, shard_id=0, num_episodes=50, base_seed=1000):
        return SimpleNamespace(
            shard_id=shard_id,
            num_episodes=num_episodes,
            base_seed=base_seed,
            server=SimpleNamespace(
                connect_host="127.0.0.1",
                bind_host="0.0.0.0",
                port=5590,
                device="cuda",
            ),
        )

    @staticmethod
    def _initialize_shard_dataset(
        dataset_dir: Path,
        *,
        num_episodes=50,
        base_seed=1000,
        outcome_task_mode="clean",
    ):
        (dataset_dir / "meta").mkdir(parents=True)
        (dataset_dir / "data" / "chunk-000").mkdir(parents=True)
        for video_key in (
            "observation.images.front",
            "observation.images.wrist",
        ):
            (dataset_dir / "videos" / "chunk-000" / video_key).mkdir(parents=True)
        (dataset_dir / "meta" / "info.json").write_text(
            json.dumps({"total_episodes": 0}),
            encoding="utf-8",
        )
        (dataset_dir / "meta" / "tasks.jsonl").write_text(
            json.dumps({"task_index": 0, "task": "Clean instruction."}) + "\n",
            encoding="utf-8",
        )
        (dataset_dir / "meta" / "episode_outcomes.jsonl").write_text(
            "",
            encoding="utf-8",
        )
        (dataset_dir / "collection_summary.json").write_text(
            json.dumps(
                {
                    "status": "running",
                    "mode": "save_all",
                    "outcome_task_mode": outcome_task_mode,
                    "target_episodes": num_episodes,
                    "max_attempts": num_episodes,
                    "base_seed": base_seed,
                    "output_dataset": str(dataset_dir.resolve()),
                    "episodes": 0,
                    "attempt_log": [],
                }
            ),
            encoding="utf-8",
        )

    def test_clean_outcome_task_mode_is_forwarded_to_each_collector(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = self._collect_args(root)
            shard = self._shard()
            argv = orchestrator._build_collect_argv(
                args,
                shard,
                root / "shard-dataset",
            )

            mode_index = argv.index("--outcome-task-mode")
            self.assertEqual(argv[mode_index + 1], "clean")

    def test_text_cache_relocation_is_forwarded_to_each_collector(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = root / "text-cache"
            cache.mkdir()
            args = self._collect_args(root, text_embedding_cache_dir=cache)
            argv = orchestrator._build_collect_argv(
                args,
                self._shard(),
                root / "shard-dataset",
            )
            self.assertEqual(
                argv[argv.index("--text-embedding-cache-dir") + 1],
                str(cache.resolve()),
            )

    def test_resume_is_forwarded_without_overwrite_and_keeps_shard_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = self._collect_args(root, resume=True, overwrite=False)
            shard = self._shard()
            argv = orchestrator._build_collect_argv(args, shard, root / "shard-dataset")

            self.assertIn("--resume", argv)
            self.assertNotIn("--overwrite", argv)
            self.assertEqual(argv[argv.index("--target-episodes") + 1], "50")
            self.assertEqual(argv[argv.index("--max-attempts") + 1], "50")
            self.assertEqual(argv[argv.index("--seed") + 1], "1000")

    def test_resume_plan_uses_protected_fresh_for_uninitialized_shard(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = self._collect_args(root, resume=True, overwrite=False)
            shard = self._shard()
            dataset_dir = root / "shard_0" / "dataset"

            mode = orchestrator._classify_resume_shard_dataset(
                args,
                shard,
                dataset_dir,
            )
            argv = orchestrator._build_collect_argv(
                args,
                shard,
                dataset_dir,
                shard_launch_mode=mode,
            )

            self.assertEqual(mode, orchestrator.SHARD_LAUNCH_FRESH)
            self.assertNotIn("--resume", argv)
            self.assertNotIn("--overwrite", argv)
            self.assertEqual(argv[argv.index("--target-episodes") + 1], "50")
            self.assertEqual(argv[argv.index("--seed") + 1], "1000")

    def test_resume_plan_resumes_fully_initialized_shard(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = self._collect_args(root, resume=True, overwrite=False)
            shard = self._shard()
            dataset_dir = root / "shard_0" / "dataset"
            self._initialize_shard_dataset(dataset_dir)

            mode = orchestrator._classify_resume_shard_dataset(
                args,
                shard,
                dataset_dir,
            )
            argv = orchestrator._build_collect_argv(
                args,
                shard,
                dataset_dir,
                shard_launch_mode=mode,
            )

            self.assertEqual(mode, orchestrator.SHARD_LAUNCH_RESUME)
            self.assertIn("--resume", argv)
            self.assertNotIn("--overwrite", argv)

    def test_resume_plan_handles_mixed_initialized_and_missing_shards(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = self._collect_args(root, resume=True, overwrite=False)
            args.shard_dir_fmt = "shard_{i}"
            initialized = self._shard(shard_id=0, base_seed=1000)
            missing = self._shard(shard_id=1, base_seed=1050)
            self._initialize_shard_dataset(
                root / "shard_0" / "dataset",
                base_seed=1000,
            )

            plan = orchestrator._plan_resume_shard_launches(
                args,
                [initialized, missing],
                root,
            )

            self.assertEqual(
                plan,
                {
                    0: orchestrator.SHARD_LAUNCH_RESUME,
                    1: orchestrator.SHARD_LAUNCH_FRESH,
                },
            )

    def test_resume_plan_fails_closed_on_partial_dataset_without_deleting_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = self._collect_args(root, resume=True, overwrite=False)
            shard = self._shard()
            dataset_dir = root / "shard_0" / "dataset"
            dataset_dir.mkdir(parents=True)
            marker = dataset_dir / "do-not-delete.txt"
            marker.write_text("preserve", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "partially initialized"):
                orchestrator._classify_resume_shard_dataset(
                    args,
                    shard,
                    dataset_dir,
                )

            self.assertEqual(marker.read_text(encoding="utf-8"), "preserve")

    def test_resume_plan_fails_closed_on_conflicting_shard_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = self._collect_args(root, resume=True, overwrite=False)
            shard = self._shard(base_seed=1000)
            dataset_dir = root / "shard_0" / "dataset"
            self._initialize_shard_dataset(dataset_dir, base_seed=999)
            marker = dataset_dir / "do-not-delete.txt"
            marker.write_text("preserve", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "frozen shard assignment"):
                orchestrator._classify_resume_shard_dataset(
                    args,
                    shard,
                    dataset_dir,
                )

            self.assertEqual(marker.read_text(encoding="utf-8"), "preserve")

    def test_non_resume_overwrite_and_no_overwrite_semantics_are_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shard = self._shard()

            overwrite_argv = orchestrator._build_collect_argv(
                self._collect_args(root, overwrite=True),
                shard,
                root / "overwrite-dataset",
            )
            self.assertIn("--overwrite", overwrite_argv)
            self.assertNotIn("--resume", overwrite_argv)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            no_overwrite_argv = orchestrator._build_collect_argv(
                self._collect_args(root, overwrite=False),
                self._shard(),
                root / "no-overwrite-dataset",
            )
            self.assertNotIn("--overwrite", no_overwrite_argv)
            self.assertNotIn("--resume", no_overwrite_argv)

    def test_resume_and_overwrite_are_cli_mutually_exclusive(self):
        common = [
            "--gpus", "0",
            "--episodes", "1",
            "--run-dir", "/tmp/run",
            "--checkpoint", "step_000001.pt",
            "--replan-steps", "25",
            "--output-dir", "/tmp/out",
        ]
        with self.assertRaises(SystemExit):
            orchestrator.parse_args(common + ["--resume", "--overwrite"])

        resumed = orchestrator.parse_args(common + ["--resume"])
        self.assertTrue(resumed.resume)
        self.assertFalse(resumed.overwrite)

        fresh = orchestrator.parse_args(common)
        self.assertFalse(fresh.resume)
        self.assertTrue(fresh.overwrite)

        protected = orchestrator.parse_args(common + ["--no-overwrite"])
        self.assertFalse(protected.resume)
        self.assertFalse(protected.overwrite)

    def test_normalization_overrides_are_cli_mutually_exclusive(self):
        common = [
            "--gpus", "0",
            "--episodes", "1",
            "--run-dir", "/tmp/run",
            "--checkpoint", "step_000001.pt",
            "--replan-steps", "25",
            "--output-dir", "/tmp/out",
        ]
        with self.assertRaises(SystemExit):
            orchestrator.parse_args(
                common
                + [
                    "--dataset-stats-path", "/tmp/dataset_stats.json",
                    "--norm-stats-meta-dir", "/tmp/meta",
                ]
            )

    def test_meta_stats_override_is_forwarded_to_each_server(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            meta_dir = root / "meta"
            meta_dir.mkdir()
            args = SimpleNamespace(
                server_script=root / "server.py",
                server_num_workers=8,
                run_dir=root,
                checkpoint="step_000001.pt",
                dataset_stats_path=None,
                norm_stats_meta_dir=meta_dir,
                action_horizon=None,
                num_inference_steps=None,
                load_text_encoder=False,
                api_token=None,
            )
            argv = orchestrator._build_server_argv(args, self._shard().server)
            self.assertEqual(
                argv[argv.index("--norm-stats-meta-dir") + 1],
                str(meta_dir.resolve()),
            )
            self.assertNotIn("--dataset-stats-path", argv)

    def test_server_rejects_ambiguous_normalization_overrides(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = SimpleNamespace(
                server_script=root / "server.py",
                server_num_workers=8,
                run_dir=root,
                checkpoint="step_000001.pt",
                dataset_stats_path=root / "dataset_stats.json",
                norm_stats_meta_dir=root / "meta",
                action_horizon=None,
                num_inference_steps=None,
                load_text_encoder=False,
                api_token=None,
            )
            with self.assertRaisesRegex(ValueError, "mutually exclusive"):
                orchestrator._build_server_argv(args, self._shard().server)

    def test_resume_rebuilds_existing_raw_and_trimmed_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build_script = root / "build.py"
            build_script.write_text("", encoding="utf-8")
            args = SimpleNamespace(
                raw_output_dataset=root / "raw",
                trimmed_output_dataset=root / "trimmed",
                trim_failure_seconds=8.0,
                build_script=build_script,
                failure_phrase="Failed.",
                overwrite=False,
                resume=True,
            )
            completed = subprocess.CompletedProcess(args=[], returncode=0)
            with mock.patch.object(
                orchestrator.subprocess,
                "run",
                side_effect=[completed, completed],
            ) as run:
                rc = orchestrator._run_dataset_build(
                    args,
                    [root / "shard_0" / "dataset", root / "shard_1" / "dataset"],
                )

            self.assertEqual(rc, 0)
            self.assertEqual(run.call_count, 2)
            merge_cmd = run.call_args_list[0].args[0]
            trim_cmd = run.call_args_list[1].args[0]
            self.assertEqual(merge_cmd[-1], "--overwrite")
            self.assertEqual(trim_cmd[-1], "--overwrite")


if __name__ == "__main__":
    unittest.main()
