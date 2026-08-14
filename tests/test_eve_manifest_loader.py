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
        self.vae_latent_cache_dir = kwargs.get("vae_latent_cache_dir")
        self.require_vae_latent_cache = bool(
            kwargs.get("require_vae_latent_cache", False)
        )
        self.drop_video_when_latents_cached = bool(
            kwargs.get("drop_video_when_latents_cached", False)
        )

    def _maybe_attach_vae_latents(self, data, *, sample_id, window_start):
        del sample_id, window_start
        return data


SCHEMA_CALLS: list[tuple[dict[str, object], bool, bool]] = []


def _validate_manifest(manifest, *, strict, verify_hash):
    SCHEMA_CALLS.append((manifest, strict, verify_hash))


def _resolve_manifest_dataset_root(manifest, unit, overrides):
    del manifest
    dataset_id = str(unit["dataset_id"])
    return (overrides or {}).get(dataset_id, unit["dataset_root"])


def _load_module():
    torch_stub = types.ModuleType("torch")
    torch_stub.float32 = "float32"
    torch_stub.long = "long"
    torch_stub.tensor = lambda value, dtype=None: value
    torch_stub.zeros = lambda size, dtype=None: [0.0] * int(size)
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
        self.assertEqual(dataset.global_sample_stride, 2)
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

    def test_paired_failure_event_expands_only_nearest_core_anchor(self) -> None:
        with TemporaryDirectory() as tmp:
            root = str(Path(tmp) / "dataset")
            manifest = self._manifest(root)
            manifest["samples"][0].update(
                {
                    "sample_type": "event",
                    "event_id": "failure-event",
                    "episode_outcome": "failure",
                    "event_outcome": "failure",
                    "core_start_frame": 5,
                    "window_selection": "core_start_anchor",
                    "pair_id": "pair-0",
                    "pair_weight": 0.8,
                }
            )
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
        self.assertEqual(dataset._samples[0]["window_start"], 4)

    def test_paired_failure_without_window_selection_keeps_sliding_windows(self) -> None:
        with TemporaryDirectory() as tmp:
            root = str(Path(tmp) / "dataset")
            manifest = self._manifest(root)
            manifest["samples"][0].update(
                {
                    "sample_type": "event",
                    "event_id": "legacy-failure-event",
                    "episode_outcome": "failure",
                    "event_outcome": "failure",
                    "core_start_frame": 5,
                    "pair_id": "pair-0",
                    "pair_weight": 0.8,
                }
            )
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

        self.assertEqual(
            [sample["window_start"] for sample in dataset._samples],
            [0, 1, 2, 3, 4],
        )

    def test_success_auxiliary_event_selects_one_33_frame_anchor_window(self) -> None:
        with TemporaryDirectory() as tmp:
            root = str(Path(tmp) / "dataset")
            manifest = self._manifest(root)
            manifest["samples"][0].update(
                {
                    "sample_type": "event",
                    "event_id": "success-event",
                    "episode_outcome": "success",
                    "event_outcome": "unknown",
                    "batch_role": "auxiliary",
                    "action_loss": "disabled",
                    "start_frame": 0,
                    "end_frame": 50,
                    "core_start_frame": 25,
                    "window_selection": "core_start_anchor",
                }
            )
            with (
                mock.patch.object(manifest_dataset, "_load_json", return_value=manifest),
                mock.patch.object(
                    manifest_dataset, "_load_eve_action_schema", return_value=None
                ),
                mock.patch.object(
                    EveManifestRobotVideoDataset,
                    "_build_episode_index",
                    return_value={(str(Path(root).resolve()), 3): (0, 50)},
                ),
            ):
                dataset = EveManifestRobotVideoDataset(
                    manifest_path=str(Path(tmp) / "manifest.json"),
                    num_frames=33,
                )

        self.assertEqual(len(dataset._samples), 1)
        self.assertEqual(dataset._samples[0]["window_start"], 17)
        self.assertEqual(dataset._samples[0]["window_end"], 50)

    def test_failure_event_selects_one_33_frame_anchor_window(self) -> None:
        with TemporaryDirectory() as tmp:
            root = str(Path(tmp) / "dataset")
            manifest = self._manifest(root)
            manifest["samples"][0].update(
                {
                    "sample_type": "event",
                    "event_id": "failure-event",
                    "episode_outcome": "failure",
                    "event_outcome": "unknown",
                    "batch_role": "auxiliary",
                    "action_loss": "disabled",
                    "start_frame": 0,
                    "end_frame": 50,
                    "core_start_frame": 25,
                    "window_selection": "core_start_anchor",
                    "pair_weight": 0.0,
                }
            )
            with (
                mock.patch.object(manifest_dataset, "_load_json", return_value=manifest),
                mock.patch.object(
                    manifest_dataset, "_load_eve_action_schema", return_value=None
                ),
                mock.patch.object(
                    EveManifestRobotVideoDataset,
                    "_build_episode_index",
                    return_value={(str(Path(root).resolve()), 3): (0, 50)},
                ),
            ):
                dataset = EveManifestRobotVideoDataset(
                    manifest_path=str(Path(tmp) / "manifest.json"),
                    num_frames=33,
                )

        self.assertEqual(len(dataset._samples), 1)
        self.assertEqual(dataset._samples[0]["window_start"], 17)
        self.assertEqual(dataset._samples[0]["window_end"], 50)

    def test_unpaired_event_keeps_all_sliding_windows(self) -> None:
        with TemporaryDirectory() as tmp:
            root = str(Path(tmp) / "dataset")
            manifest = self._manifest(root)
            manifest["samples"][0].update(
                {
                    "sample_type": "event",
                    "event_id": "failure-event",
                    "episode_outcome": "failure",
                    "event_outcome": "failure",
                    "core_start_frame": 5,
                    "pair_weight": 0.0,
                }
            )
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

        self.assertEqual(
            [sample["window_start"] for sample in dataset._samples],
            [0, 1, 2, 3, 4],
        )

    def test_paired_success_event_keeps_all_sliding_windows(self) -> None:
        with TemporaryDirectory() as tmp:
            root = str(Path(tmp) / "dataset")
            manifest = self._manifest(root)
            manifest["samples"][0].update(
                {
                    "sample_type": "event",
                    "event_id": "success-event",
                    "episode_outcome": "success",
                    "event_outcome": "success",
                    "core_start_frame": 5,
                    "pair_id": "pair-0",
                    "pair_weight": 0.8,
                }
            )
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

        self.assertEqual(
            [sample["window_start"] for sample in dataset._samples],
            [0, 1, 2, 3, 4],
        )

    @staticmethod
    def _retry_dataset() -> EveManifestRobotVideoDataset:
        dataset = EveManifestRobotVideoDataset.__new__(
            EveManifestRobotVideoDataset
        )
        dataset._samples = [{}, {}, {}, {}]
        dataset.sampling_roles = (
            "primary",
            "auxiliary",
            "primary",
            "primary",
        )
        dataset._sampling_role_indices = {
            "primary": (0, 2, 3),
            "auxiliary": (1,),
        }
        dataset.max_load_retry = 2
        return dataset

    def test_structural_error_fails_hard_without_retry(self) -> None:
        dataset = self._retry_dataset()
        calls: list[int] = []

        def get_eve(index: int):
            calls.append(index)
            raise ValueError("pair target split mismatch")

        dataset._get_eve = get_eve
        with self.assertRaisesRegex(ValueError, "pair target split mismatch"):
            dataset[0]
        self.assertEqual(calls, [0])

    def test_non_decode_runtime_error_fails_hard_without_retry(self) -> None:
        dataset = self._retry_dataset()
        calls: list[int] = []

        def get_eve(index: int):
            calls.append(index)
            raise RuntimeError("pair target hash mismatch")

        dataset._get_eve = get_eve
        with self.assertRaisesRegex(RuntimeError, "pair target hash mismatch"):
            dataset[0]
        self.assertEqual(calls, [0])

    def test_io_retry_uses_deterministic_same_role_order(self) -> None:
        dataset = self._retry_dataset()
        calls: list[int] = []

        def get_eve(index: int):
            calls.append(index)
            if len(calls) < 3:
                raise OSError("temporary video read failure")
            return {"index": index}

        dataset._get_eve = get_eve
        self.assertEqual(dataset[0], {"index": 3})
        self.assertEqual(calls, [0, 2, 3])

    def test_explicit_video_decode_runtime_error_is_retryable(self) -> None:
        dataset = self._retry_dataset()
        calls: list[int] = []

        def get_eve(index: int):
            calls.append(index)
            if len(calls) == 1:
                raise RuntimeError("failed to decode video stream")
            return {"index": index}

        dataset._get_eve = get_eve
        self.assertEqual(dataset[0], {"index": 2})
        self.assertEqual(calls, [0, 2])

    def test_unknown_event_outcome_falls_back_to_episode_outcome(self) -> None:
        self.assertEqual(
            EveManifestRobotVideoDataset._effective_outcome(
                {
                    "event_outcome": "unknown",
                    "episode_outcome": "failure",
                }
            ),
            "failure",
        )
        self.assertEqual(
            EveManifestRobotVideoDataset._outcome_flag(
                {
                    "event_outcome": "unknown",
                    "episode_outcome": "failure",
                }
            ),
            1,
        )
        self.assertEqual(
            EveManifestRobotVideoDataset._effective_outcome(
                {
                    "event_outcome": "success",
                    "episode_outcome": "failure",
                }
            ),
            "success",
        )

    def test_get_eve_returns_soft_event_defaults(self) -> None:
        dataset = EveManifestRobotVideoDataset.__new__(
            EveManifestRobotVideoDataset
        )
        dataset.manifest_path = "/tmp/manifest.json"
        dataset.global_sample_stride = 1
        dataset._samples = [
            {
                "unit": {
                    "sample_type": "event",
                    "sample_id": "event-1",
                    "dataset_id": "robot",
                    "collection_round": 1,
                    "event_outcome": "unknown",
                    "episode_outcome": "failure",
                    "event_weight": None,
                    "pair_weight": None,
                    "pair_id": None,
                },
                "episode_index": 3,
                "global_frame_idx": 10,
                "window_start": 2,
                "window_end": 6,
            }
        ]
        dataset.vae_latent_cache_dir = None
        dataset.drop_video_when_latents_cached = False
        dataset._get = lambda *args, **kwargs: {}

        data = dataset._get_eve(0)

        self.assertEqual(data["event_weight"], 1.0)
        self.assertEqual(data["pair_weight"], 0.0)
        self.assertEqual(data["pair_id"], "")
        self.assertEqual(data["outcome_flag"], 1)
        self.assertEqual(data["eve_event_outcome"], "failure")

    def test_get_eve_returns_explicit_soft_event_fields(self) -> None:
        dataset = EveManifestRobotVideoDataset.__new__(
            EveManifestRobotVideoDataset
        )
        dataset.manifest_path = "/tmp/manifest.json"
        dataset.global_sample_stride = 1
        dataset._samples = [
            {
                "unit": {
                    "sample_type": "event",
                    "sample_id": "event-1",
                    "dataset_id": "robot",
                    "collection_round": 1,
                    "event_outcome": "failure",
                    "episode_outcome": "failure",
                    "event_weight": 0.6,
                    "pair_weight": 0.4,
                    "pair_id": "pair-7",
                },
                "episode_index": 3,
                "global_frame_idx": 10,
                "window_start": 2,
                "window_end": 6,
            }
        ]
        dataset.vae_latent_cache_dir = None
        dataset.drop_video_when_latents_cached = False
        dataset._get = lambda *args, **kwargs: {}

        data = dataset._get_eve(0)

        self.assertEqual(data["event_weight"], 0.6)
        self.assertEqual(data["pair_weight"], 0.4)
        self.assertEqual(data["pair_id"], "pair-7")


if __name__ == "__main__":
    unittest.main()
