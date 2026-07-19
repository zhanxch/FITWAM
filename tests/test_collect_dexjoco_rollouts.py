from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    stubs = {}
    for name in ("av", "numpy", "pyarrow", "pyarrow.parquet"):
        if name not in sys.modules:
            stubs[name] = types.ModuleType(name)
    if "pyarrow" in stubs:
        stubs["pyarrow"].__path__ = []
    if "dexjoco_fastwam_adapter" not in sys.modules:
        adapter = types.ModuleType("dexjoco_fastwam_adapter")
        adapter.DEFAULT_TASK_CONFIG_DIR = Path(".")
        for name in (
            "ActionConstraintConfig",
            "DexJoCoFastWAMAdapter",
            "DexJoCoFastWAMEvalEnv",
            "DexJoCoTaskConfig",
        ):
            setattr(adapter, name, type(name, (), {}))
        adapter._safe_rgb_uint8 = lambda value: value
        adapter.constrain_rotvec_action = lambda value, *_args, **_kwargs: value
        adapter.load_dexjoco_eval_settings = lambda _path: None
        stubs["dexjoco_fastwam_adapter"] = adapter
    if "policy_client_async" not in sys.modules:
        client = types.ModuleType("policy_client_async")
        client.PolicyClientAsync = type("PolicyClientAsync", (), {})
        stubs["policy_client_async"] = client

    saved = {name: sys.modules.get(name) for name in stubs}
    sys.modules.update(stubs)
    try:
        spec = importlib.util.spec_from_file_location(
            "collect_dexjoco_rollouts_under_test",
            ROOT / "scripts" / "collect_dexjoco_rollouts.py",
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        for name, previous in saved.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


collect = _load_module()


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _episode_row(episode_index: int, length: int = 4) -> dict:
    return {
        "episode_index": episode_index,
        "tasks": ["Clean instruction."],
        "length": length,
    }


def _stats_row(episode_index: int) -> dict:
    return {"episode_index": episode_index, "stats": {}}


def _attempt_row(episode_index: int, *, base_seed: int = 100) -> dict:
    return {
        "attempt_index": episode_index,
        "seed": base_seed + episode_index,
        "success": episode_index % 2 == 0,
        "done": True,
        "steps": 4,
        "elapsed_s": 1.0,
        "saved_failure_index": None,
        "saved_episode_index": episode_index,
    }


def _create_resume_dataset(
    root: Path,
    *,
    committed: int,
    materialized: int,
    episodes_rows: int | None = None,
    stats_rows: int | None = None,
    outcome_rows: int | None = None,
    info_episodes: int | None = None,
    malformed_episode_tail: bool = False,
    include_outcome_ledger: bool = True,
) -> Path:
    output = root / "output"
    meta = output / "meta"
    meta.mkdir(parents=True)
    info_episodes = materialized if info_episodes is None else info_episodes
    info = {
        "total_episodes": info_episodes,
        "total_frames": info_episodes * 4,
        "total_tasks": 1,
        "total_videos": info_episodes * 2,
        "total_chunks": 1,
        "splits": {"train": f"0:{info_episodes}"},
        "chunks_size": 1000,
        "features": {
            "observation.images.front": {},
            "observation.images.wrist": {},
        },
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": (
            "videos/chunk-{episode_chunk:03d}/{video_key}/"
            "episode_{episode_index:06d}.mp4"
        ),
    }
    (meta / "info.json").write_text(json.dumps(info), encoding="utf-8")
    _write_jsonl(
        meta / "tasks.jsonl",
        [{"task_index": 0, "task": "Clean instruction."}],
    )

    episodes_rows = materialized if episodes_rows is None else episodes_rows
    stats_rows = materialized if stats_rows is None else stats_rows
    outcome_rows = materialized if outcome_rows is None else outcome_rows
    _write_jsonl(
        meta / "episodes.jsonl",
        [_episode_row(index) for index in range(episodes_rows)],
    )
    if malformed_episode_tail:
        with (meta / "episodes.jsonl").open("a", encoding="utf-8") as f:
            f.write('{"episode_index":')
    _write_jsonl(
        meta / "episodes_stats.jsonl",
        [_stats_row(index) for index in range(stats_rows)],
    )
    attempts = [_attempt_row(index) for index in range(committed)]
    if include_outcome_ledger:
        _write_jsonl(
            meta / "episode_outcomes.jsonl",
            [
                collect.make_outcome_row(
                    episode_index=index,
                    success=index % 2 == 0,
                    attempt_index=index,
                    seed=100 + index,
                )
                for index in range(outcome_rows)
            ],
        )
    summary = {
        "status": "running",
        "mode": "save_all",
        "outcome_task_mode": "clean",
        "base_seed": 100,
        "next_attempt_index": committed,
        "episodes": committed,
        "frames": committed * 4,
        "attempts": committed,
        "failures": sum(not row["success"] for row in attempts),
        "successes_saved": sum(row["success"] for row in attempts),
        "attempt_log": attempts,
    }
    (output / "collection_summary.json").write_text(
        json.dumps(summary),
        encoding="utf-8",
    )

    for episode_index in range(materialized):
        parquet = (
            output
            / "data"
            / "chunk-000"
            / f"episode_{episode_index:06d}.parquet"
        )
        parquet.parent.mkdir(parents=True, exist_ok=True)
        parquet.write_bytes(b"parquet")
        for video_key in ("observation.images.front", "observation.images.wrist"):
            video = (
                output
                / "videos"
                / "chunk-000"
                / video_key
                / f"episode_{episode_index:06d}.mp4"
            )
            video.parent.mkdir(parents=True, exist_ok=True)
            video.write_bytes(b"video")
    return output


class CollectOutcomeLedgerTest(unittest.TestCase):
    def test_cli_keeps_task_marker_as_backward_compatible_default(self):
        with mock.patch.object(
            sys,
            "argv",
            ["collect_dexjoco_rollouts.py", "--run-dir", "/tmp/run"],
        ):
            args = collect.parse_args()
        self.assertEqual(args.outcome_task_mode, "task-marker")

    def test_outcome_row_has_required_structured_fields(self):
        row = collect.make_outcome_row(
            episode_index=3,
            success=False,
            attempt_index=7,
            seed=10007,
        )
        self.assertEqual(
            row,
            {
                "episode_index": 3,
                "outcome": "failure",
                "success": False,
                "attempt_index": 7,
                "seed": 10007,
                "source": "dexjoco_env",
            },
        )

    def test_validation_rejects_duplicate_and_inconsistent_rows(self):
        duplicate = [
            collect.make_outcome_row(
                episode_index=0, success=True, attempt_index=0, seed=1
            ),
            collect.make_outcome_row(
                episode_index=0, success=False, attempt_index=1, seed=2
            ),
        ]
        with self.assertRaisesRegex(ValueError, "Duplicate outcome"):
            collect.validate_outcome_rows(duplicate, expected_episode_count=2)

        inconsistent = [
            {
                **collect.make_outcome_row(
                    episode_index=0, success=False, attempt_index=0, seed=1
                ),
                "outcome": "success",
            }
        ]
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            collect.validate_outcome_rows(inconsistent, expected_episode_count=1)

    def test_clean_task_mode_creates_one_task_and_empty_ledger(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            output = root / "output"
            (source / "meta").mkdir(parents=True)
            (source / "meta" / "info.json").write_text(
                json.dumps(
                    {
                        "fps": 30,
                        "chunks_size": 1000,
                        "features": {},
                    }
                ),
                encoding="utf-8",
            )

            info, episodes, frames, stats, attempts = collect.prepare_dataset(
                source,
                output,
                "Clean instruction.",
                "Clean instruction. Failed.",
                overwrite=False,
                resume=False,
                save_all_trajectories=True,
                outcome_task_mode="clean",
                base_seed=100,
            )

            self.assertEqual(info["total_tasks"], 1)
            self.assertEqual((episodes, frames, stats, attempts), (0, 0, [], []))
            tasks = collect.load_jsonl(output / "meta" / "tasks.jsonl")
            self.assertEqual(tasks, [{"task_index": 0, "task": "Clean instruction."}])
            self.assertTrue((output / "meta" / "episode_outcomes.jsonl").exists())
            self.assertEqual(
                (output / "meta" / "episode_outcomes.jsonl").read_text(encoding="utf-8"),
                "",
            )

    def test_legacy_resume_backfills_ledger_from_attempt_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = _create_resume_dataset(
                root,
                committed=1,
                materialized=1,
                include_outcome_ledger=False,
            )
            (output / "meta" / "tasks.jsonl").write_text(
                json.dumps({"task_index": 0, "task": "Task. Failed."}) + "\n",
                encoding="utf-8",
            )
            summary = json.loads(
                (output / "collection_summary.json").read_text(encoding="utf-8")
            )
            summary["attempt_log"][0]["success"] = False
            summary["failures"] = 1
            summary["successes_saved"] = 0
            summary["mode"] = "failures_only"
            summary["outcome_task_mode"] = "task-marker"
            (output / "collection_summary.json").write_text(
                json.dumps(summary),
                encoding="utf-8",
            )

            with (
                mock.patch.object(collect, "_parquet_num_rows", return_value=4),
                mock.patch.object(collect, "get_video_info", return_value={}),
            ):
                collect.prepare_dataset(
                    root / "unused-source",
                    output,
                    "Task.",
                    "Task. Failed.",
                    overwrite=False,
                    resume=True,
                    save_all_trajectories=False,
                    outcome_task_mode="task-marker",
                    base_seed=100,
                )

            rows = collect.load_jsonl(output / "meta" / "episode_outcomes.jsonl")
            self.assertEqual(
                rows,
                [
                    {
                        "episode_index": 0,
                        "outcome": "failure",
                        "success": False,
                        "attempt_index": 0,
                        "seed": 100,
                        "source": "dexjoco_env",
                    }
                ],
            )

    def test_resume_reconciles_each_uncommitted_save_stage(self):
        stages = {
            "media_only": {
                "episodes_rows": 1,
                "stats_rows": 1,
                "outcome_rows": 1,
                "info_episodes": 1,
            },
            "episodes_written": {
                "episodes_rows": 2,
                "stats_rows": 1,
                "outcome_rows": 1,
                "info_episodes": 1,
            },
            "stats_written": {
                "episodes_rows": 2,
                "stats_rows": 2,
                "outcome_rows": 1,
                "info_episodes": 1,
            },
            "outcome_and_info_written": {
                "episodes_rows": 2,
                "stats_rows": 2,
                "outcome_rows": 2,
                "info_episodes": 2,
            },
        }
        for stage, kwargs in stages.items():
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as tmp:
                output = _create_resume_dataset(
                    Path(tmp),
                    committed=1,
                    materialized=2,
                    **kwargs,
                )
                with (
                    mock.patch.object(collect, "_parquet_num_rows", return_value=4),
                    mock.patch.object(collect, "get_video_info", return_value={}),
                ):
                    info, episodes, frames, stats, attempts = collect.prepare_dataset(
                        Path(tmp) / "unused-source",
                        output,
                        "Clean instruction.",
                        "Clean instruction. Failed.",
                        overwrite=False,
                        resume=True,
                        save_all_trajectories=True,
                        outcome_task_mode="clean",
                        base_seed=100,
                    )

                self.assertEqual((episodes, frames), (1, 4))
                self.assertEqual(len(stats), 1)
                self.assertEqual(len(attempts), 1)
                self.assertEqual(collect._next_attempt_index(attempts), 1)
                self.assertEqual(100 + collect._next_attempt_index(attempts), 101)
                self.assertEqual(info["total_episodes"], 1)
                self.assertEqual(info["total_frames"], 4)
                for name in (
                    "episodes.jsonl",
                    "episodes_stats.jsonl",
                    "episode_outcomes.jsonl",
                ):
                    self.assertEqual(
                        len(collect.load_jsonl(output / "meta" / name)),
                        1,
                    )
                self.assertFalse(
                    (
                        output
                        / "data"
                        / "chunk-000"
                        / "episode_000001.parquet"
                    ).exists()
                )
                for video_key in (
                    "observation.images.front",
                    "observation.images.wrist",
                ):
                    self.assertFalse(
                        (
                            output
                            / "videos"
                            / "chunk-000"
                            / video_key
                            / "episode_000001.mp4"
                        ).exists()
                    )

    def test_resume_before_first_commit_restarts_original_seed(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = _create_resume_dataset(
                Path(tmp),
                committed=0,
                materialized=1,
                episodes_rows=0,
                stats_rows=0,
                outcome_rows=0,
                info_episodes=1,
            )
            info, episodes, frames, stats, attempts = collect.prepare_dataset(
                Path(tmp) / "unused-source",
                output,
                "Clean instruction.",
                "Clean instruction. Failed.",
                overwrite=False,
                resume=True,
                save_all_trajectories=True,
                outcome_task_mode="clean",
                base_seed=100,
            )
            self.assertEqual((episodes, frames, stats, attempts), (0, 0, [], []))
            self.assertEqual(info["total_episodes"], 0)
            self.assertEqual(collect._next_attempt_index(attempts), 0)
            self.assertEqual(100 + collect._next_attempt_index(attempts), 100)
            self.assertFalse(
                (
                    output
                    / "data"
                    / "chunk-000"
                    / "episode_000000.parquet"
                ).exists()
            )
            for video_key in (
                "observation.images.front",
                "observation.images.wrist",
            ):
                self.assertFalse(
                    (
                        output
                        / "videos"
                        / "chunk-000"
                        / video_key
                        / "episode_000000.mp4"
                    ).exists()
                )

    def test_resume_discards_malformed_uncommitted_jsonl_tail(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = _create_resume_dataset(
                Path(tmp),
                committed=1,
                materialized=2,
                episodes_rows=1,
                stats_rows=1,
                outcome_rows=1,
                malformed_episode_tail=True,
            )
            with (
                mock.patch.object(collect, "_parquet_num_rows", return_value=4),
                mock.patch.object(collect, "get_video_info", return_value={}),
            ):
                collect.prepare_dataset(
                    Path(tmp) / "unused-source",
                    output,
                    "Clean instruction.",
                    "Clean instruction. Failed.",
                    overwrite=False,
                    resume=True,
                    save_all_trajectories=True,
                    outcome_task_mode="clean",
                    base_seed=100,
                )
            self.assertEqual(
                collect.load_jsonl(output / "meta" / "episodes.jsonl"),
                [_episode_row(0)],
            )

    def test_resume_keeps_episode_after_atomic_summary_commit(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = _create_resume_dataset(
                Path(tmp),
                committed=2,
                materialized=2,
            )
            with (
                mock.patch.object(collect, "_parquet_num_rows", return_value=4),
                mock.patch.object(collect, "get_video_info", return_value={}),
            ):
                info, episodes, frames, _stats, attempts = collect.prepare_dataset(
                    Path(tmp) / "unused-source",
                    output,
                    "Clean instruction.",
                    "Clean instruction. Failed.",
                    overwrite=False,
                    resume=True,
                    save_all_trajectories=True,
                    outcome_task_mode="clean",
                    base_seed=100,
                )
            self.assertEqual((episodes, frames), (2, 8))
            self.assertEqual(info["total_episodes"], 2)
            self.assertEqual(collect._next_attempt_index(attempts), 2)
            self.assertTrue(
                (
                    output
                    / "data"
                    / "chunk-000"
                    / "episode_000001.parquet"
                ).exists()
            )

    def test_resume_rejects_missing_committed_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = _create_resume_dataset(
                Path(tmp),
                committed=1,
                materialized=1,
            )
            (
                output
                / "videos"
                / "chunk-000"
                / "observation.images.wrist"
                / "episode_000000.mp4"
            ).unlink()
            with (
                mock.patch.object(collect, "_parquet_num_rows", return_value=4),
                self.assertRaisesRegex(ValueError, "missing video"),
            ):
                collect.prepare_dataset(
                    Path(tmp) / "unused-source",
                    output,
                    "Clean instruction.",
                    "Clean instruction. Failed.",
                    overwrite=False,
                    resume=True,
                    save_all_trajectories=True,
                    outcome_task_mode="clean",
                    base_seed=100,
                )

    def test_resume_rejects_changed_base_seed(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = _create_resume_dataset(
                Path(tmp),
                committed=1,
                materialized=1,
            )
            with self.assertRaisesRegex(ValueError, "Resume seed mismatch"):
                collect.prepare_dataset(
                    Path(tmp) / "unused-source",
                    output,
                    "Clean instruction.",
                    "Clean instruction. Failed.",
                    overwrite=False,
                    resume=True,
                    save_all_trajectories=True,
                    outcome_task_mode="clean",
                    base_seed=999,
                )


if __name__ == "__main__":
    unittest.main()
