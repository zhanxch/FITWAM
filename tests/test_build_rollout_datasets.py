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
    saved = {name: sys.modules.get(name) for name in stubs}
    sys.modules.update(stubs)
    try:
        spec = importlib.util.spec_from_file_location(
            "build_rollout_datasets_under_test",
            ROOT / "scripts" / "build_rollout_datasets.py",
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


build = _load_module()


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _outcome(ep: int, success: bool, attempt: int, seed: int) -> dict:
    return {
        "episode_index": ep,
        "outcome": "success" if success else "failure",
        "success": success,
        "attempt_index": attempt,
        "seed": seed,
        "source": "dexjoco_env",
    }


class OutcomeLedgerValidationTest(unittest.TestCase):
    def test_missing_ledger_is_rejected_for_shard_merge(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "meta").mkdir(parents=True)
            (root / "meta" / "info.json").write_text(
                json.dumps({"features": {}}),
                encoding="utf-8",
            )
            _write_jsonl(
                root / "meta" / "episodes.jsonl",
                [{"episode_index": 0, "tasks": ["Task."], "length": 3}],
            )
            _write_jsonl(
                root / "meta" / "tasks.jsonl",
                [{"task_index": 0, "task": "Task."}],
            )
            episodes = build.load_jsonl(root / "meta" / "episodes.jsonl")
            with self.assertRaisesRegex(FileNotFoundError, "episode_outcomes"):
                build.load_outcome_ledger(
                    root,
                    episodes,
                    failure_phrase=build.FAILURE_PHRASE,
                    required=True,
                )
            with self.assertRaisesRegex(FileNotFoundError, "episode_outcomes"):
                build.merge_shards(
                    [root],
                    root / "merged",
                    overwrite=False,
                    failure_phrase=build.FAILURE_PHRASE,
                )

    def test_duplicate_missing_and_inconsistent_rows_are_rejected(self):
        episodes = [
            {"episode_index": 0, "tasks": ["Task."], "length": 3},
            {"episode_index": 1, "tasks": ["Task."], "length": 3},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            duplicate = [_outcome(0, True, 0, 10), _outcome(0, False, 1, 11)]
            with self.assertRaisesRegex(ValueError, "duplicate outcome"):
                build.validate_outcome_rows(
                    duplicate,
                    episodes,
                    dataset_root=root,
                    failure_phrase=build.FAILURE_PHRASE,
                )

            with self.assertRaisesRegex(ValueError, "missing=\\[1\\]"):
                build.validate_outcome_rows(
                    [_outcome(0, True, 0, 10)],
                    episodes,
                    dataset_root=root,
                    failure_phrase=build.FAILURE_PHRASE,
                )

            inconsistent = [_outcome(0, True, 0, 10), _outcome(1, False, 1, 11)]
            inconsistent[1]["outcome"] = "success"
            with self.assertRaisesRegex(ValueError, "inconsistent"):
                build.validate_outcome_rows(
                    inconsistent,
                    episodes,
                    dataset_root=root,
                    failure_phrase=build.FAILURE_PHRASE,
                )

    def test_clean_task_failure_is_valid_and_marker_conflict_is_rejected(self):
        clean_failure = [{"episode_index": 0, "tasks": ["Task."], "length": 3}]
        rows = [_outcome(0, False, 0, 10)]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build.validate_outcome_rows(
                rows,
                clean_failure,
                dataset_root=root,
                failure_phrase=build.FAILURE_PHRASE,
            )
            marker_success = [
                {
                    "episode_index": 0,
                    "tasks": [f"Task. {build.FAILURE_PHRASE}"],
                    "length": 3,
                }
            ]
            with self.assertRaisesRegex(ValueError, "failure task marker"):
                build.validate_outcome_rows(
                    [_outcome(0, True, 0, 10)],
                    marker_success,
                    dataset_root=root,
                    failure_phrase=build.FAILURE_PHRASE,
                )


class MergeAndTrimLedgerTest(unittest.TestCase):
    def test_finalize_writes_episode_outcome_ledger(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "meta").mkdir(parents=True)
            row = _outcome(0, False, 2, 102)
            build.finalize_dataset(
                root,
                {
                    "features": {},
                    "video_path": "",
                },
                [{"episode_index": 0, "tasks": ["Task."], "length": 3}],
                [{}],
                total_frames=3,
                extra_summary={"status": "complete"},
                episode_outcomes=[row],
            )
            self.assertEqual(
                build.load_jsonl(root / "meta" / "episode_outcomes.jsonl"),
                [row],
            )

    def _make_shard(
        self,
        root: Path,
        *,
        outcomes: list[dict],
        task_mode: str = "clean",
    ) -> Path:
        (root / "meta").mkdir(parents=True)
        (root / "meta" / "info.json").write_text(
            json.dumps(
                {
                    "chunks_size": 1000,
                    "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
                    "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
                    "features": {},
                    "fps": 30,
                    "total_episodes": len(outcomes),
                    "total_frames": 3 * len(outcomes),
                }
            ),
            encoding="utf-8",
        )
        _write_jsonl(
            root / "meta" / "episodes.jsonl",
            [
                {"episode_index": index, "tasks": ["Task."], "length": 3}
                for index in range(len(outcomes))
            ],
        )
        _write_jsonl(root / "meta" / "episode_outcomes.jsonl", outcomes)
        _write_jsonl(root / "meta" / "tasks.jsonl", [{"task_index": 0, "task": "Task."}])
        (root / "collection_summary.json").write_text(
            json.dumps(
                {
                    "status": "complete",
                    "mode": "save_all",
                    "outcome_task_mode": task_mode,
                    "target_episodes": len(outcomes),
                    "episodes": len(outcomes),
                    "attempts": len(outcomes),
                    "successes_saved": sum(row["success"] for row in outcomes),
                    "failures": sum(not row["success"] for row in outcomes),
                    "attempt_log": [
                        {
                            "attempt_index": row["attempt_index"],
                            "seed": row["seed"],
                            "success": row["success"],
                            "saved_episode_index": row["episode_index"],
                        }
                        for row in outcomes
                    ],
                }
            ),
            encoding="utf-8",
        )
        return root

    def test_merge_remaps_local_episode_indexes_and_writes_complete_ledger(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shard0 = self._make_shard(
                root / "s0",
                outcomes=[_outcome(0, True, 0, 100)],
            )
            shard1 = self._make_shard(
                root / "s1",
                outcomes=[
                    _outcome(0, False, 0, 200),
                    _outcome(1, True, 1, 201),
                ],
            )
            captured = {}
            fake_data = {
                "action": [0, 1, 2],
                "observation.state": [0, 1, 2],
                "timestamp": [0, 1, 2],
                "frame_index": [0, 1, 2],
                "episode_index": [0, 0, 0],
                "index": [0, 1, 2],
                "task_index": [0, 0, 0],
            }

            class FakeArray(list):
                def astype(self, _dtype):
                    return self

                def __truediv__(self, value):
                    return FakeArray(item / value for item in self)

            fake_np = types.SimpleNamespace(
                int64=object(),
                float32=object(),
                arange=lambda start, stop=None, dtype=None: FakeArray(
                    range(start if stop is None else start, start if stop is None else stop)
                ),
                full=lambda length, value, dtype=None: FakeArray([value] * length),
            )

            def capture_finalize(*_args, **kwargs):
                captured.update(kwargs)

            with (
                mock.patch.object(build, "prepare_output", return_value=json.loads(
                    (shard0 / "meta" / "info.json").read_text(encoding="utf-8")
                )),
                mock.patch.object(build.shutil, "copy2"),
                mock.patch.object(build.pq, "read_table", return_value=object(), create=True),
                mock.patch.object(build, "table_to_numpy_dict", return_value=fake_data),
                mock.patch.object(build, "write_episode_parquet"),
                mock.patch.object(build, "video_keys", return_value=[]),
                mock.patch.object(build, "compute_episode_stats", return_value={}),
                mock.patch.object(build, "finalize_dataset", side_effect=capture_finalize),
                mock.patch.object(build, "np", fake_np),
            ):
                build.merge_shards(
                    [shard0, shard1],
                    root / "merged",
                    overwrite=False,
                    failure_phrase=build.FAILURE_PHRASE,
                )

            self.assertEqual(
                [row["episode_index"] for row in captured["episode_outcomes"]],
                [0, 1, 2],
            )
            self.assertEqual(
                [row["outcome"] for row in captured["episode_outcomes"]],
                ["success", "failure", "success"],
            )
            self.assertEqual(
                [
                    row["saved_episode_index"]
                    for row in captured["extra_summary"]["attempt_log"]
                ],
                [0, 1, 2],
            )
            self.assertEqual(
                [row["attempt_index"] for row in captured["episode_outcomes"]],
                [0, 1, 2],
            )
            self.assertEqual(
                [
                    (
                        row["source_shard_id"],
                        row["source_episode_index"],
                        row["source_attempt_index"],
                    )
                    for row in captured["episode_outcomes"]
                ],
                [(0, 0, 0), (1, 0, 0), (1, 1, 1)],
            )
            self.assertEqual(
                [row["attempt_index"] for row in captured["extra_summary"]["attempt_log"]],
                [0, 1, 2],
            )
            self.assertEqual(
                [
                    (
                        row["source_shard_id"],
                        row["source_episode_index"],
                        row["source_attempt_index"],
                    )
                    for row in captured["extra_summary"]["attempt_log"]
                ],
                [(0, 0, 0), (1, 0, 0), (1, 1, 1)],
            )
            self.assertEqual(captured["extra_summary"]["outcome_task_mode"], "clean")

    def test_merge_rejects_missing_incomplete_or_count_mismatched_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shard = self._make_shard(
                root / "shard",
                outcomes=[_outcome(0, True, 0, 100)],
            )
            summary_path = shard / "collection_summary.json"

            summary_path.unlink()
            with self.assertRaisesRegex(FileNotFoundError, "collection_summary"):
                build.merge_shards(
                    [shard],
                    root / "missing-summary",
                    overwrite=False,
                    failure_phrase=build.FAILURE_PHRASE,
                )

            shard = self._make_shard(
                root / "shard-incomplete",
                outcomes=[_outcome(0, True, 0, 100)],
            )
            summary_path = shard / "collection_summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["status"] = "running"
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "expected 'complete'"):
                build.merge_shards(
                    [shard],
                    root / "incomplete",
                    overwrite=False,
                    failure_phrase=build.FAILURE_PHRASE,
                )

            shard = self._make_shard(
                root / "shard-count",
                outcomes=[_outcome(0, True, 0, 100)],
            )
            summary_path = shard / "collection_summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["target_episodes"] = 2
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "counts disagree"):
                build.merge_shards(
                    [shard],
                    root / "count-mismatch",
                    overwrite=False,
                    failure_phrase=build.FAILURE_PHRASE,
                )

    def test_trim_uses_ledger_over_clean_task_text_and_propagates_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self._make_shard(
                root / "source",
                outcomes=[_outcome(0, False, 0, 100)],
            )
            captured = {}
            fake_data = {
                key: list(range(6))
                for key in (
                    "action",
                    "observation.state",
                    "timestamp",
                    "frame_index",
                    "episode_index",
                    "index",
                    "task_index",
                )
            }

            class FakeArray(list):
                def astype(self, _dtype):
                    return self

                def __truediv__(self, value):
                    return FakeArray(item / value for item in self)

            fake_np = types.SimpleNamespace(
                int64=object(),
                float32=object(),
                arange=lambda start, stop=None, dtype=None: FakeArray(
                    range(start if stop is None else start, start if stop is None else stop)
                ),
                full=lambda length, value, dtype=None: FakeArray([value] * length),
            )

            def capture_finalize(*_args, **kwargs):
                captured.update(kwargs)

            with (
                mock.patch.object(build, "prepare_output", return_value=json.loads(
                    (source / "meta" / "info.json").read_text(encoding="utf-8")
                )),
                mock.patch.object(build.shutil, "copy2"),
                mock.patch.object(build.pq, "read_table", return_value=object(), create=True),
                mock.patch.object(build, "table_to_numpy_dict", return_value=fake_data),
                mock.patch.object(build, "write_episode_parquet"),
                mock.patch.object(build, "video_keys", return_value=[]),
                mock.patch.object(build, "compute_episode_stats", return_value={}),
                mock.patch.object(build, "finalize_dataset", side_effect=capture_finalize),
                mock.patch.object(build, "np", fake_np),
            ):
                build.trim_failures(
                    source,
                    root / "trimmed",
                    trim_seconds=1 / 30,
                    trim_only_length=6,
                    failure_phrase=build.FAILURE_PHRASE,
                    overwrite=False,
                )

            self.assertEqual(captured["episode_outcomes"], [_outcome(0, False, 0, 100)])
            self.assertEqual(captured["extra_summary"]["outcome_source"], "episode_outcomes.jsonl")
            self.assertEqual(captured["extra_summary"]["trimmed_failures"], 1)
            self.assertEqual(captured["extra_summary"]["outcome_task_mode"], "clean")

    def test_trim_falls_back_to_task_marker_for_legacy_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self._make_shard(
                root / "source",
                outcomes=[_outcome(0, False, 0, 100)],
                task_mode="task-marker",
            )
            (source / "meta" / "episode_outcomes.jsonl").unlink()
            _write_jsonl(
                source / "meta" / "episodes.jsonl",
                [
                    {
                        "episode_index": 0,
                        "tasks": [f"Task. {build.FAILURE_PHRASE}"],
                        "length": 6,
                    }
                ],
            )
            captured = {}
            fake_data = {
                key: list(range(6))
                for key in (
                    "action",
                    "observation.state",
                    "timestamp",
                    "frame_index",
                    "episode_index",
                    "index",
                    "task_index",
                )
            }

            class FakeArray(list):
                def astype(self, _dtype):
                    return self

                def __truediv__(self, value):
                    return FakeArray(item / value for item in self)

            fake_np = types.SimpleNamespace(
                int64=object(),
                float32=object(),
                arange=lambda start, stop=None, dtype=None: FakeArray(
                    range(start if stop is None else start, start if stop is None else stop)
                ),
                full=lambda length, value, dtype=None: FakeArray([value] * length),
            )

            def capture_finalize(*_args, **kwargs):
                captured.update(kwargs)

            with (
                mock.patch.object(build, "prepare_output", return_value=json.loads(
                    (source / "meta" / "info.json").read_text(encoding="utf-8")
                )),
                mock.patch.object(build.shutil, "copy2"),
                mock.patch.object(build.pq, "read_table", return_value=object(), create=True),
                mock.patch.object(build, "table_to_numpy_dict", return_value=fake_data),
                mock.patch.object(build, "write_episode_parquet"),
                mock.patch.object(build, "video_keys", return_value=[]),
                mock.patch.object(build, "compute_episode_stats", return_value={}),
                mock.patch.object(build, "finalize_dataset", side_effect=capture_finalize),
                mock.patch.object(build, "np", fake_np),
            ):
                build.trim_failures(
                    source,
                    root / "trimmed",
                    trim_seconds=1 / 30,
                    trim_only_length=6,
                    failure_phrase=build.FAILURE_PHRASE,
                    overwrite=False,
                )

            self.assertIsNone(captured["episode_outcomes"])
            self.assertEqual(captured["extra_summary"]["outcome_source"], "legacy_task_marker")
            self.assertEqual(captured["extra_summary"]["trimmed_failures"], 1)

    def test_validate_outcome_dataset_checks_counts_and_hashes_ledger(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self._make_shard(
                root / "source",
                outcomes=[
                    _outcome(0, True, 0, 100),
                    _outcome(1, False, 1, 101),
                ],
            )
            info_path = source / "meta" / "info.json"
            info = json.loads(info_path.read_text(encoding="utf-8"))
            info["total_episodes"] = 2
            info_path.write_text(json.dumps(info), encoding="utf-8")
            (source / "collection_summary.json").write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "episodes": 2,
                        "successes_saved": 1,
                        "failures": 1,
                        "outcome_task_mode": "clean",
                        "attempt_log": [
                            {
                                "attempt_index": 0,
                                "seed": 100,
                                "success": True,
                                "saved_episode_index": 0,
                            },
                            {
                                "attempt_index": 1,
                                "seed": 101,
                                "success": False,
                                "saved_episode_index": 1,
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            report = build.validate_outcome_dataset(
                source,
                failure_phrase=build.FAILURE_PHRASE,
                expected_episodes=2,
            )

            self.assertEqual(report["status"], "valid")
            self.assertEqual(report["successes"], 1)
            self.assertEqual(report["failures"], 1)
            self.assertEqual(len(report["outcome_ledger_sha256"]), 64)
            self.assertFalse(report["check_media"])

    def test_validate_outcome_dataset_check_media_validates_physical_files(self):
        class FakeColumn:
            def __init__(self, values):
                self.values = values

            def combine_chunks(self):
                return self

            def to_pylist(self):
                return self.values

        class FakeTable:
            def __init__(self, columns):
                self.columns = columns
                self.column_names = list(columns)
                self.num_rows = len(columns["frame_index"])

            def __getitem__(self, name):
                return FakeColumn(self.columns[name])

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self._make_shard(
                root / "source",
                outcomes=[
                    _outcome(0, True, 0, 100),
                    _outcome(1, False, 1, 101),
                ],
            )
            info_path = source / "meta" / "info.json"
            info = json.loads(info_path.read_text(encoding="utf-8"))
            info["features"] = {
                "action": {"dtype": "float32"},
                "observation.images.front": {"dtype": "video"},
                "observation.images.wrist": {"dtype": "video"},
            }
            info["total_videos"] = 4
            info_path.write_text(json.dumps(info), encoding="utf-8")
            _write_jsonl(
                source / "meta" / "episodes_stats.jsonl",
                [
                    {"episode_index": 0, "stats": {"action": {"count": [3]}}},
                    {"episode_index": 1, "stats": {"action": {"count": [3]}}},
                ],
            )
            tables = {}
            for episode_index in range(2):
                parquet_path = source / "data" / "chunk-000" / (
                    f"episode_{episode_index:06d}.parquet"
                )
                parquet_path.parent.mkdir(parents=True, exist_ok=True)
                parquet_path.touch()
                tables[str(parquet_path.resolve())] = FakeTable(
                    {
                        "frame_index": [0, 1, 2],
                        "episode_index": [episode_index] * 3,
                        "index": list(
                            range(episode_index * 3, episode_index * 3 + 3)
                        ),
                    }
                )
                for key in info["features"]:
                    video_path = (
                        source
                        / "videos"
                        / "chunk-000"
                        / key
                        / f"episode_{episode_index:06d}.mp4"
                    )
                    video_path.parent.mkdir(parents=True, exist_ok=True)
                    video_path.touch()

            with (
                mock.patch.object(
                    build.pq,
                    "read_table",
                    side_effect=lambda path: tables[str(Path(path).resolve())],
                    create=True,
                ),
                mock.patch.object(build, "count_video_frames", return_value=3),
            ):
                report = build.validate_outcome_dataset(
                    source,
                    failure_phrase=build.FAILURE_PHRASE,
                    expected_episodes=2,
                    check_media=True,
                )

            self.assertTrue(report["check_media"])
            self.assertEqual(report["physical_validation"]["frames"], 6)
            self.assertEqual(report["physical_validation"]["videos"], 4)

            stats_rows = build.load_jsonl(
                source / "meta" / "episodes_stats.jsonl"
            )
            stats_rows[0]["stats"]["action"]["count"] = [2]
            _write_jsonl(source / "meta" / "episodes_stats.jsonl", stats_rows)
            with self.assertRaisesRegex(ValueError, "has count=\\[2\\]"):
                build.validate_outcome_dataset(
                    source,
                    failure_phrase=build.FAILURE_PHRASE,
                    expected_episodes=2,
                    check_media=True,
                )

    def test_check_media_rejects_bad_indexes_and_video_frame_count(self):
        class FakeColumn:
            def __init__(self, values):
                self.values = values

            def combine_chunks(self):
                return self

            def to_pylist(self):
                return self.values

        class FakeTable:
            column_names = ["frame_index", "episode_index", "index"]

            def __init__(self, frame_indexes, *, global_indexes=None, num_rows=3):
                self.num_rows = num_rows
                self.columns = {
                    "frame_index": frame_indexes,
                    "episode_index": [0, 0, 0],
                    "index": global_indexes or [0, 1, 2],
                }

            def __getitem__(self, name):
                return FakeColumn(self.columns[name])

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self._make_shard(
                root / "source",
                outcomes=[_outcome(0, True, 0, 100)],
            )
            info_path = source / "meta" / "info.json"
            info = json.loads(info_path.read_text(encoding="utf-8"))
            info["features"] = {"observation.images.front": {"dtype": "video"}}
            info_path.write_text(json.dumps(info), encoding="utf-8")
            _write_jsonl(
                source / "meta" / "episodes_stats.jsonl",
                [{"episode_index": 0, "stats": {}}],
            )
            parquet_path = (
                source / "data" / "chunk-000" / "episode_000000.parquet"
            )
            parquet_path.parent.mkdir(parents=True, exist_ok=True)
            parquet_path.touch()
            video_path = (
                source
                / "videos"
                / "chunk-000"
                / "observation.images.front"
                / "episode_000000.mp4"
            )
            video_path.parent.mkdir(parents=True, exist_ok=True)
            video_path.touch()

            with mock.patch.object(
                build.pq,
                "read_table",
                return_value=FakeTable([0, 2, 1]),
                create=True,
            ):
                with self.assertRaisesRegex(ValueError, "frame_index is not contiguous"):
                    build.validate_outcome_dataset(
                        source,
                        failure_phrase=build.FAILURE_PHRASE,
                        check_media=True,
                    )

            with mock.patch.object(
                build.pq,
                "read_table",
                return_value=FakeTable([0, 1, 2], num_rows=2),
                create=True,
            ):
                with self.assertRaisesRegex(ValueError, "row count 2"):
                    build.validate_outcome_dataset(
                        source,
                        failure_phrase=build.FAILURE_PHRASE,
                        check_media=True,
                    )

            with mock.patch.object(
                build.pq,
                "read_table",
                return_value=FakeTable(
                    [0, 1, 2],
                    global_indexes=[0, 2, 1],
                ),
                create=True,
            ):
                with self.assertRaisesRegex(ValueError, "global index is not contiguous"):
                    build.validate_outcome_dataset(
                        source,
                        failure_phrase=build.FAILURE_PHRASE,
                        check_media=True,
                    )

            video_path.unlink()
            with mock.patch.object(
                build.pq,
                "read_table",
                return_value=FakeTable([0, 1, 2]),
                create=True,
            ):
                with self.assertRaises(FileNotFoundError):
                    build.validate_outcome_dataset(
                        source,
                        failure_phrase=build.FAILURE_PHRASE,
                        check_media=True,
                    )
            video_path.touch()

            with (
                mock.patch.object(
                    build.pq,
                    "read_table",
                    return_value=FakeTable([0, 1, 2]),
                    create=True,
                ),
                mock.patch.object(build, "count_video_frames", return_value=2),
            ):
                with self.assertRaisesRegex(ValueError, "decoded frame count 2"):
                    build.validate_outcome_dataset(
                        source,
                        failure_phrase=build.FAILURE_PHRASE,
                        check_media=True,
                    )


if __name__ == "__main__":
    unittest.main()
