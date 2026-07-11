from __future__ import annotations

import importlib.util
import logging
import sys
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "fastwam"
    / "datasets"
    / "eve"
    / "manifest_dataset.py"
)


class _RobotVideoDataset:
    init_calls: list[dict[str, object]] = []

    def __init__(self, *args, dataset_dirs, **kwargs):
        del args
        self.__class__.init_calls.append(
            {"dataset_dirs": dataset_dirs, "kwargs": dict(kwargs)}
        )
        self.num_frames = int(kwargs.get("num_frames", 4))
        self.global_sample_stride = int(kwargs.get("global_sample_stride", 1))


SCHEMA_CALLS: list[tuple[dict[str, object], bool, bool]] = []


def _validate_manifest(manifest, *, strict, verify_hash):
    SCHEMA_CALLS.append((manifest, strict, verify_hash))


def _resolve_manifest_dataset_root(manifest, unit, overrides):
    del manifest
    dataset_id = str(unit["dataset_id"])
    return (overrides or {}).get(dataset_id, unit["dataset_root"])


def _load_module():
    torch_stub = types.ModuleType("torch")
    base_stub = types.ModuleType(
        "fastwam.datasets.lerobot.robot_video_dataset"
    )
    base_stub.RobotVideoDataset = _RobotVideoDataset
    schema_stub = types.ModuleType("fastwam.everobot_schema")
    schema_stub.validate_manifest = _validate_manifest
    schema_stub.resolve_manifest_dataset_root = _resolve_manifest_dataset_root
    logging_stub = types.ModuleType("fastwam.utils.logging_config")
    logging_stub.get_logger = logging.getLogger

    module_name = "_test_eve_manifest_dataset"
    spec = importlib.util.spec_from_file_location(module_name, MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(
        sys.modules,
        {
            "torch": torch_stub,
            "fastwam.datasets.lerobot.robot_video_dataset": base_stub,
            "fastwam.everobot_schema": schema_stub,
            "fastwam.utils.logging_config": logging_stub,
        },
    ):
        spec.loader.exec_module(module)
    return module


manifest_dataset = _load_module()
EveManifestRobotVideoDataset = manifest_dataset.EveManifestRobotVideoDataset


class EveManifestLoaderTest(unittest.TestCase):
    def setUp(self) -> None:
        SCHEMA_CALLS.clear()
        _RobotVideoDataset.init_calls.clear()

    @staticmethod
    def _manifest(dataset_root: str) -> dict[str, object]:
        return {
            "format": "EveRobotTrainManifest",
            "schema_version": "0.1",
            "dataset_roots": {"robot": dataset_root},
            "samples": [
                {
                    "sample_type": "episode",
                    "dataset_id": "robot",
                    "dataset_root": dataset_root,
                    "episode_index": 3,
                    "start_frame": 0,
                    "end_frame": 8,
                }
            ],
        }

    def test_init_validates_v01_manifest_and_uses_relocated_root(self) -> None:
        with TemporaryDirectory() as tmp:
            old_root = str(Path(tmp) / "old")
            new_root = str(Path(tmp) / "relocated")
            manifest = self._manifest(old_root)
            episode_index = {(str(Path(new_root).resolve()), 3): (10, 8)}

            with (
                mock.patch.object(manifest_dataset, "_load_json", return_value=manifest),
                mock.patch.object(
                    manifest_dataset, "_load_eve_action_schema", return_value=None
                ),
                mock.patch.object(
                    EveManifestRobotVideoDataset,
                    "_build_episode_index",
                    return_value=episode_index,
                ),
            ):
                dataset = EveManifestRobotVideoDataset(
                    manifest_path=str(Path(tmp) / "manifest.json"),
                    dataset_root_overrides={"robot": new_root},
                    num_frames=4,
                )

        self.assertEqual(SCHEMA_CALLS, [(manifest, True, True)])
        self.assertEqual(
            _RobotVideoDataset.init_calls[0]["dataset_dirs"],
            [str(Path(new_root).resolve())],
        )
        self.assertEqual(len(dataset._samples), 5)
        self.assertTrue(
            all(sample["dataset_root"] == str(Path(new_root).resolve()) for sample in dataset._samples)
        )

    def test_missing_episode_reference_raises_by_default(self) -> None:
        dataset = EveManifestRobotVideoDataset.__new__(
            EveManifestRobotVideoDataset
        )
        dataset.manifest = self._manifest("/old/root")
        dataset.dataset_root_overrides = {}
        dataset.strict_manifest_references = True
        dataset.manifest_splits = None
        dataset.manifest_collection_iters = None
        dataset.event_sample_stride = None
        dataset.episode_sample_stride = None
        dataset.num_frames = 4
        dataset._episode_index = {}

        with self.assertRaisesRegex(ValueError, "episode absent"):
            dataset._expand_manifest_samples()

    def test_missing_episode_reference_can_warn_and_skip(self) -> None:
        dataset = EveManifestRobotVideoDataset.__new__(
            EveManifestRobotVideoDataset
        )
        dataset.manifest = self._manifest("/old/root")
        dataset.dataset_root_overrides = {}
        dataset.strict_manifest_references = False
        dataset.manifest_splits = None
        dataset.manifest_collection_iters = None
        dataset.event_sample_stride = None
        dataset.episode_sample_stride = None
        dataset.num_frames = 4
        dataset._episode_index = {}

        with self.assertLogs(manifest_dataset.logger, level="WARNING") as logs:
            samples = dataset._expand_manifest_samples()

        self.assertEqual(samples, [])
        self.assertTrue(
            any("Skipping missing Eve episode reference" in line for line in logs.output)
        )
        self.assertTrue(
            any("Skipped 1 Eve sample units with missing episodes" in line for line in logs.output)
        )

    def test_opt_out_is_forwarded_to_manifest_validation(self) -> None:
        with TemporaryDirectory() as tmp:
            root = str(Path(tmp) / "dataset")
            manifest = self._manifest(root)
            with (
                mock.patch.object(manifest_dataset, "_load_json", return_value=manifest),
                mock.patch.object(
                    manifest_dataset, "_load_eve_action_schema", return_value=None
                ),
                mock.patch.object(
                    EveManifestRobotVideoDataset,
                    "_build_episode_index",
                    return_value={(str(Path(root).resolve()), 3): (0, 8)},
                ),
            ):
                EveManifestRobotVideoDataset(
                    manifest_path=str(Path(tmp) / "manifest.json"),
                    strict_manifest_references=False,
                    num_frames=4,
                )

        self.assertEqual(SCHEMA_CALLS, [(manifest, True, True)])

    def test_hash_verification_has_an_independent_opt_out(self) -> None:
        with TemporaryDirectory() as tmp:
            root = str(Path(tmp) / "dataset")
            manifest = self._manifest(root)
            with (
                mock.patch.object(manifest_dataset, "_load_json", return_value=manifest),
                mock.patch.object(
                    manifest_dataset, "_load_eve_action_schema", return_value=None
                ),
                mock.patch.object(
                    EveManifestRobotVideoDataset,
                    "_build_episode_index",
                    return_value={(str(Path(root).resolve()), 3): (0, 8)},
                ),
            ):
                EveManifestRobotVideoDataset(
                    manifest_path=str(Path(tmp) / "manifest.json"),
                    verify_manifest_hash=False,
                    num_frames=4,
                )

        self.assertEqual(SCHEMA_CALLS, [(manifest, True, False)])

    def test_source_stride_limits_windows_to_the_manifest_interval(self) -> None:
        with TemporaryDirectory() as tmp:
            root = str(Path(tmp) / "dataset")
            manifest = self._manifest(root)
            episode_index = {(str(Path(root).resolve()), 3): (10, 8)}
            with (
                mock.patch.object(manifest_dataset, "_load_json", return_value=manifest),
                mock.patch.object(
                    manifest_dataset, "_load_eve_action_schema", return_value=None
                ),
                mock.patch.object(
                    EveManifestRobotVideoDataset,
                    "_build_episode_index",
                    return_value=episode_index,
                ),
            ):
                dataset = EveManifestRobotVideoDataset(
                    manifest_path=str(Path(tmp) / "manifest.json"),
                    num_frames=4,
                    global_sample_stride=2,
                )

        self.assertEqual(len(dataset._samples), 2)
        self.assertEqual(
            [(sample["window_start"], sample["window_end"]) for sample in dataset._samples],
            [(0, 7), (1, 8)],
        )

    def test_overrides_replace_explicit_legacy_dataset_dirs(self) -> None:
        with TemporaryDirectory() as tmp:
            old_root = str(Path(tmp) / "old")
            new_root = str(Path(tmp) / "relocated")
            manifest = self._manifest(old_root)
            with (
                mock.patch.object(manifest_dataset, "_load_json", return_value=manifest),
                mock.patch.object(
                    manifest_dataset, "_load_eve_action_schema", return_value=None
                ),
                mock.patch.object(
                    EveManifestRobotVideoDataset,
                    "_build_episode_index",
                    return_value={(str(Path(new_root).resolve()), 3): (0, 8)},
                ),
            ):
                EveManifestRobotVideoDataset(
                    manifest_path=str(Path(tmp) / "manifest.json"),
                    dataset_dirs=[old_root],
                    dataset_root_overrides={"robot": new_root},
                    num_frames=4,
                )

        self.assertEqual(
            _RobotVideoDataset.init_calls[0]["dataset_dirs"],
            [str(Path(new_root).resolve())],
        )

    def test_excluded_split_roots_are_not_loaded(self) -> None:
        with TemporaryDirectory() as tmp:
            train_root = str(Path(tmp) / "train")
            val_root = str(Path(tmp) / "unavailable-val")
            manifest = self._manifest(train_root)
            manifest["samples"][0]["split"] = "train"
            manifest["samples"].append(
                {
                    "sample_type": "episode",
                    "dataset_id": "val-robot",
                    "dataset_root": val_root,
                    "episode_index": 0,
                    "start_frame": 0,
                    "end_frame": 8,
                    "split": "val",
                }
            )
            manifest["dataset_roots"]["val-robot"] = val_root
            with (
                mock.patch.object(manifest_dataset, "_load_json", return_value=manifest),
                mock.patch.object(
                    manifest_dataset, "_load_eve_action_schema", return_value=None
                ),
                mock.patch.object(
                    EveManifestRobotVideoDataset,
                    "_build_episode_index",
                    return_value={(str(Path(train_root).resolve()), 3): (0, 8)},
                ),
            ):
                EveManifestRobotVideoDataset(
                    manifest_path=str(Path(tmp) / "manifest.json"),
                    manifest_splits=["train"],
                    num_frames=4,
                )

        self.assertEqual(
            _RobotVideoDataset.init_calls[0]["dataset_dirs"],
            [str(Path(train_root).resolve())],
        )

    def test_valid_intervals_limit_window_expansion(self) -> None:
        with TemporaryDirectory() as tmp:
            root = str(Path(tmp) / "dataset")
            manifest = self._manifest(root)
            manifest["samples"][0]["valid_intervals"] = [[0, 4], [6, 8]]
            with (
                mock.patch.object(manifest_dataset, "_load_json", return_value=manifest),
                mock.patch.object(
                    manifest_dataset, "_load_eve_action_schema", return_value=None
                ),
                mock.patch.object(
                    EveManifestRobotVideoDataset,
                    "_build_episode_index",
                    return_value={(str(Path(root).resolve()), 3): (0, 8)},
                ),
            ):
                dataset = EveManifestRobotVideoDataset(
                    manifest_path=str(Path(tmp) / "manifest.json"),
                    num_frames=4,
                )

        self.assertEqual(len(dataset._samples), 1)
        self.assertEqual(dataset._samples[0]["window_start"], 0)


if __name__ == "__main__":
    unittest.main()
