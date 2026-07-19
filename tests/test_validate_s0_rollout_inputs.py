from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "validate_s0_rollout_inputs_under_test",
    ROOT / "scripts" / "water_plant" / "validate_s0_rollout_inputs.py",
)
validate = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(validate)


def _stat_block(stepwise_length: int, dimension: int) -> dict:
    stepwise_min = [[-2.0] * dimension for _ in range(stepwise_length)]
    stepwise_max = [[2.0] * dimension for _ in range(stepwise_length)]
    stepwise_q01 = [[-1.0] * dimension for _ in range(stepwise_length)]
    stepwise_q99 = [[1.0] * dimension for _ in range(stepwise_length)]
    stepwise_mean = [[0.0] * dimension for _ in range(stepwise_length)]
    stepwise_std = [[0.5] * dimension for _ in range(stepwise_length)]
    return {
        "stepwise_min": stepwise_min,
        "stepwise_max": stepwise_max,
        "stepwise_q01": stepwise_q01,
        "stepwise_q99": stepwise_q99,
        "stepwise_mean": stepwise_mean,
        "stepwise_std": stepwise_std,
        "global_min": [-2.0] * dimension,
        "global_max": [2.0] * dimension,
        "global_q01": [-1.0] * dimension,
        "global_q99": [1.0] * dimension,
        "global_mean": [0.0] * dimension,
        "global_std": [0.5] * dimension,
    }


def _dataset_stats() -> dict:
    return {
        "action": {"default": _stat_block(32, 22)},
        "state": {"default": _stat_block(1, 23)},
        "num_episodes": 100,
        "num_transition": 12000,
    }


def _meta_feature_stats(dimension: int) -> dict:
    return {
        "min": [-2.0] * dimension,
        "max": [2.0] * dimension,
        "q01": [-1.0] * dimension,
        "q99": [1.0] * dimension,
        "mean": [0.0] * dimension,
        "std": [0.5] * dimension,
        "count": [100],
    }


def _meta_stats() -> dict:
    return {
        "action": _meta_feature_stats(22),
        "observation.state": _meta_feature_stats(23),
    }


def _modality() -> dict:
    return {
        "action": {
            "arm": {"start": 0, "end": 6},
            "hand": {"start": 6, "end": 22},
        },
        "state": {
            "eef_pose": {"start": 0, "end": 7},
            "hand_joints": {"start": 7, "end": 23},
        },
    }


def _config() -> dict:
    return {
        "data": {
            "train": {
                "video_size": [384, 768],
                "concat_multi_camera": "horizontal",
                "num_frames": 33,
                "context_len": 128,
                "shape_meta": {
                    "images": [
                        {"key": "front", "shape": [3, 384, 384]},
                        {"key": "wrist", "shape": [3, 384, 384]},
                    ]
                },
                "processor": {
                    "num_output_cameras": 2,
                    "action_output_dim": 22,
                    "proprio_output_dim": 23,
                    "norm_stats_source": "compute",
                },
            }
        },
        "model": {"proprio_dim": 23, "load_text_encoder": False},
    }


class ValidateS0RolloutInputsTest(unittest.TestCase):
    def _fixture(self, root: Path) -> Namespace:
        run_dir = root / "run"
        source = root / "source"
        base_checkpoints = root / "base-checkpoints"
        dexjoco_root = root / "dexjoco"
        (run_dir / "checkpoints" / "weights").mkdir(parents=True)
        (source / "meta").mkdir(parents=True)
        for relative in validate.BASE_MODEL_FILES:
            path = base_checkpoints / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"base-model:{relative}".encode("utf-8"))
        for relative in (
            "configs/rand_obj/water_plant.yaml",
            "dexjoco/dexjoco/tasks/mappings.py",
        ):
            path = dexjoco_root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"dexjoco:{relative}".encode("utf-8"))
        (run_dir / "config.yaml").write_text(
            yaml.safe_dump(_config()),
            encoding="utf-8",
        )
        checkpoint = run_dir / "checkpoints" / "weights" / "step_006500.pt"
        checkpoint.write_bytes(b"checkpoint")
        stats = run_dir / "dataset_stats.json"
        stats.write_text(
            json.dumps(_dataset_stats()),
            encoding="utf-8",
        )
        text_cache = run_dir / "text_cache"
        text_cache.mkdir()
        instruction = validate.MODEL_PROMPT.format(task=validate.SUCCESS_PROMPT)
        digest = hashlib.sha256(instruction.encode("utf-8")).hexdigest()
        (text_cache / f"{digest}.t5_len128.npz").write_bytes(b"npz")
        config = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
        config["data"]["train"]["text_embedding_cache_dir"] = str(text_cache)
        (run_dir / "config.yaml").write_text(
            yaml.safe_dump(config),
            encoding="utf-8",
        )
        (source / "meta" / "info.json").write_text(
            json.dumps(
                {
                    "fps": 30,
                    "total_episodes": 100,
                    "features": {
                        "action": {"shape": [22]},
                        "observation.state": {"shape": [23]},
                        "observation.images.front": {"dtype": "video"},
                        "observation.images.wrist": {"dtype": "video"},
                    },
                }
            ),
            encoding="utf-8",
        )
        (source / "meta" / "episodes.jsonl").write_text(
            json.dumps({"episode_index": 0}) + "\n",
            encoding="utf-8",
        )
        return Namespace(
            run_dir=run_dir,
            checkpoint=checkpoint,
            dataset_stats=stats,
            norm_stats_meta_dir=None,
            text_cache_dir=None,
            source_dataset=source,
            base_checkpoints_dir=base_checkpoints,
            dexjoco_root=dexjoco_root,
            protocol_out=root / "collection_protocol.json",
            expected_checkpoint_name="step_006500.pt",
            collection_kind="formal",
            episodes=200,
            base_seed=20260718,
            gpus="0,1,2,3",
            replan_steps=25,
            max_env_steps=1500,
            video_fps=30,
            outcome_task_mode="clean",
            resume=False,
        )

    @staticmethod
    def _enable_meta_mode(args: Namespace) -> Path:
        config_path = args.run_dir / "config.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        config["data"]["train"]["processor"]["norm_stats_source"] = "meta"
        config["data"]["train"]["processor"]["norm_stats_meta_dir"] = "unavailable/meta"
        config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
        meta_dir = args.run_dir.parent / "frozen-meta"
        meta_dir.mkdir()
        (meta_dir / "stats.json").write_text(
            json.dumps(_meta_stats()),
            encoding="utf-8",
        )
        (meta_dir / "modality.json").write_text(
            json.dumps(_modality()),
            encoding="utf-8",
        )
        args.dataset_stats = None
        args.norm_stats_meta_dir = meta_dir
        return meta_dir

    def test_valid_inputs_are_bound_into_protocol(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self._fixture(Path(tmp))
            payload = validate.validate_inputs(args)
            self.assertEqual(payload["model"]["config"]["action_horizon"], 32)
            self.assertEqual(
                payload["model"]["normalization"]["fields"]["action"][
                    "stepwise_shape"
                ],
                [32, 22],
            )
            self.assertEqual(
                payload["model"]["normalization"]["fields"]["state"]["global_shape"],
                [23],
            )
            self.assertEqual(payload["source_dataset"]["camera_keys"], [
                "observation.images.front",
                "observation.images.wrist",
            ])
            self.assertEqual(len(payload["model"]["checkpoint_sha256"]), 64)
            self.assertEqual(
                set(payload["model"]["base_model"]["files"]),
                set(validate.BASE_MODEL_FILES),
            )
            self.assertIn(
                "configs/rand_obj/water_plant.yaml",
                payload["environment"]["files"],
            )

    def test_missing_base_model_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self._fixture(Path(tmp))
            missing = args.base_checkpoints_dir / validate.BASE_MODEL_FILES[0]
            missing.unlink()
            with self.assertRaisesRegex(
                FileNotFoundError,
                "Missing non-empty FastWAM base model file",
            ):
                validate.validate_inputs(args)

    def test_unvalidated_7500_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self._fixture(Path(tmp))
            checkpoint_7500 = args.checkpoint.with_name("step_007500.pt")
            args.checkpoint.rename(checkpoint_7500)
            args.checkpoint = checkpoint_7500

            with self.assertRaisesRegex(
                ValueError,
                "Expected checkpoint 'step_006500.pt'",
            ):
                validate.validate_inputs(args)

    def test_meta_mode_binds_stats_and_modality_without_dataset_stats(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self._fixture(Path(tmp))
            meta_dir = self._enable_meta_mode(args)
            args.run_dir.joinpath("dataset_stats.json").unlink()

            payload = validate.validate_inputs(args)
            normalization = payload["model"]["normalization"]

            self.assertEqual(normalization["source"], "meta")
            self.assertEqual(normalization["meta_dir"], str(meta_dir.resolve()))
            self.assertEqual(normalization["fields"]["action"]["shape"], [22])
            self.assertEqual(normalization["fields"]["state"]["shape"], [23])
            self.assertEqual(
                normalization["stats"]["sha256"],
                validate.sha256_file(meta_dir / "stats.json"),
            )
            self.assertEqual(
                normalization["modality"]["sha256"],
                validate.sha256_file(meta_dir / "modality.json"),
            )

    def test_dataset_mode_still_requires_dataset_stats(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self._fixture(Path(tmp))
            args.dataset_stats = None
            with self.assertRaisesRegex(ValueError, "requires --dataset-stats"):
                validate.validate_inputs(args)

    def test_meta_mode_rejects_dataset_stats_argument(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self._fixture(Path(tmp))
            dataset_stats = args.dataset_stats
            self._enable_meta_mode(args)
            args.dataset_stats = dataset_stats
            with self.assertRaisesRegex(ValueError, "forbids --dataset-stats"):
                validate.validate_inputs(args)

    def test_meta_mode_requires_explicit_meta_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self._fixture(Path(tmp))
            self._enable_meta_mode(args)
            args.norm_stats_meta_dir = None
            with self.assertRaisesRegex(
                ValueError,
                "requires an explicit --norm-stats-meta-dir",
            ):
                validate.validate_inputs(args)

    def test_meta_mode_requires_explicit_relocatable_meta_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self._fixture(Path(tmp))
            self._enable_meta_mode(args)
            args.norm_stats_meta_dir = None
            with self.assertRaisesRegex(
                ValueError,
                "requires an explicit --norm-stats-meta-dir",
            ):
                validate.validate_inputs(args)

    def test_meta_mode_rejects_empty_required_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self._fixture(Path(tmp))
            meta_dir = self._enable_meta_mode(args)
            (meta_dir / "stats.json").write_bytes(b"")
            with self.assertRaisesRegex(
                FileNotFoundError,
                "requires a non-empty file",
            ):
                validate.validate_inputs(args)

    def test_dataset_mode_rejects_meta_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self._fixture(Path(tmp))
            args.norm_stats_meta_dir = args.run_dir.parent / "meta"
            with self.assertRaisesRegex(ValueError, "forbids --norm-stats-meta-dir"):
                validate.validate_inputs(args)

    def test_meta_mode_rejects_non_contiguous_modality_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self._fixture(Path(tmp))
            meta_dir = self._enable_meta_mode(args)
            modality = _modality()
            modality["action"]["hand"]["start"] = 7
            (meta_dir / "modality.json").write_text(
                json.dumps(modality),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "ordered contiguous"):
                validate.validate_inputs(args)

    def test_meta_mode_rejects_wrong_stats_dimension(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self._fixture(Path(tmp))
            meta_dir = self._enable_meta_mode(args)
            stats = _meta_stats()
            stats["observation.state"]["mean"].pop()
            (meta_dir / "stats.json").write_text(
                json.dumps(stats),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ValueError,
                r"observation\.state\.mean must have shape \[23\].*got \[22\]",
            ):
                validate.validate_inputs(args)

    def test_meta_mode_rejects_non_finite_or_unordered_stats(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self._fixture(Path(tmp))
            meta_dir = self._enable_meta_mode(args)
            stats = _meta_stats()
            stats["action"]["q01"][3] = float("nan")
            (meta_dir / "stats.json").write_text(
                json.dumps(stats),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, r"action\.q01\[3\].*finite"):
                validate.validate_inputs(args)

    def test_meta_mode_rejects_invalid_quantile_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self._fixture(Path(tmp))
            meta_dir = self._enable_meta_mode(args)
            stats = _meta_stats()
            stats["action"]["q01"][3] = 1.5
            (meta_dir / "stats.json").write_text(
                json.dumps(stats),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ValueError,
                r"action statistics at index 3.*min <= q01 <= q99 <= max",
            ):
                validate.validate_inputs(args)

    def test_wrong_camera_protocol_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self._fixture(Path(tmp))
            config_path = args.run_dir / "config.yaml"
            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            config["data"]["train"]["shape_meta"]["images"].pop()
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "front\\+wrist"):
                validate.validate_inputs(args)

    def test_resume_requires_identical_immutable_protocol(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self._fixture(Path(tmp))
            immutable = validate.validate_inputs(args)
            args.protocol_out.write_text(
                json.dumps(
                    {
                        **immutable,
                        "created_at_utc": "2026-07-17T00:00:00+00:00",
                        "created_by_host": "test",
                    }
                ),
                encoding="utf-8",
            )
            args.resume = True
            immutable_after = validate.validate_inputs(args)
            existing = json.loads(args.protocol_out.read_text(encoding="utf-8"))
            existing.pop("created_at_utc")
            existing.pop("created_by_host")
            self.assertEqual(existing, immutable_after)

    def test_wrong_action_dimension_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self._fixture(Path(tmp))
            stats = _dataset_stats()
            stats["action"]["default"]["global_mean"] = [0.0] * 21
            args.dataset_stats.write_text(json.dumps(stats), encoding="utf-8")
            with self.assertRaisesRegex(
                ValueError,
                r"action\.default\.global_mean must have shape \[22\].*got \[21\]",
            ):
                validate.validate_inputs(args)

    def test_wrong_observation_state_dimension_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self._fixture(Path(tmp))
            stats = _dataset_stats()
            stats["state"]["default"]["stepwise_std"] = [[0.5] * 22]
            args.dataset_stats.write_text(json.dumps(stats), encoding="utf-8")
            with self.assertRaisesRegex(
                ValueError,
                r"state\.default\.stepwise_std must have shape \[1, 23\].*got \[1, 22\]",
            ):
                validate.validate_inputs(args)

    def test_missing_required_stat_field_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self._fixture(Path(tmp))
            stats = _dataset_stats()
            del stats["action"]["default"]["global_q99"]
            args.dataset_stats.write_text(json.dumps(stats), encoding="utf-8")
            with self.assertRaisesRegex(
                ValueError,
                "action.default is missing required field 'global_q99'",
            ):
                validate.validate_inputs(args)

    def test_non_finite_stat_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self._fixture(Path(tmp))
            stats = _dataset_stats()
            stats["state"]["default"]["global_mean"][3] = float("nan")
            args.dataset_stats.write_text(json.dumps(stats), encoding="utf-8")
            with self.assertRaisesRegex(
                ValueError,
                r"state\.default\.global_mean\[3\].*finite",
            ):
                validate.validate_inputs(args)

    def test_ragged_stepwise_stat_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self._fixture(Path(tmp))
            stats = _dataset_stats()
            stats["action"]["default"]["stepwise_min"][4].pop()
            args.dataset_stats.write_text(json.dumps(stats), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "ragged nested arrays"):
                validate.validate_inputs(args)

    def test_negative_standard_deviation_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self._fixture(Path(tmp))
            stats = _dataset_stats()
            stats["action"]["default"]["global_std"][7] = -0.1
            args.dataset_stats.write_text(json.dumps(stats), encoding="utf-8")
            with self.assertRaisesRegex(
                ValueError,
                r"action\.default\.global_std\[7\] must be non-negative",
            ):
                validate.validate_inputs(args)

    def test_inconsistent_quantile_order_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self._fixture(Path(tmp))
            stats = _dataset_stats()
            stats["state"]["default"]["global_q01"][2] = 1.5
            args.dataset_stats.write_text(json.dumps(stats), encoding="utf-8")
            with self.assertRaisesRegex(
                ValueError,
                r"state\.default\.global statistics at flat index 2.*"
                r"min <= q01 <= q99 <= max",
            ):
                validate.validate_inputs(args)

    def test_missing_positive_counts_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self._fixture(Path(tmp))
            stats = _dataset_stats()
            stats["num_transition"] = 0
            args.dataset_stats.write_text(json.dumps(stats), encoding="utf-8")
            with self.assertRaisesRegex(
                ValueError,
                "num_transition must be a positive integer",
            ):
                validate.validate_inputs(args)


if __name__ == "__main__":
    unittest.main()
