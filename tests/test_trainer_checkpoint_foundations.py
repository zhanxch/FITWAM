from __future__ import annotations

import importlib.util
import json
import logging
import math
import random
import sys
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock


MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "src" / "fastwam" / "trainer.py"
)


def _load_trainer_module():
    numpy_stub = types.ModuleType("numpy")
    numpy_stub.isfinite = math.isfinite
    numpy_stub.uint8 = "uint8"
    torch_stub = types.ModuleType("torch")
    torch_stub.Tensor = type("Tensor", (), {})
    torch_stub.float32 = "float32"
    torch_stub.int64 = "int64"
    torch_stub.cuda = types.SimpleNamespace(is_available=lambda: False)
    torch_stub.no_grad = lambda: (lambda fn: fn)

    class _Generator:
        def __init__(self, device=None):
            del device
            self.rng = random.Random()

        def manual_seed(self, seed):
            self.rng.seed(seed)
            return self

    class _Scalar:
        def __init__(self, value):
            self.value = value

        def item(self):
            return self.value

    class _Vector:
        def __init__(self, values):
            self.values = values

        def tolist(self):
            return list(self.values)

    torch_stub.Generator = _Generator
    torch_stub.randint = lambda low, high, shape, generator: _Scalar(
        generator.rng.randrange(low, high)
    )

    def _randperm(size, generator):
        values = list(range(size))
        generator.rng.shuffle(values)
        return _Vector(values)

    torch_stub.randperm = _randperm

    lr_scheduler_stub = types.ModuleType("torch.optim.lr_scheduler")
    for name in ("ConstantLR", "CosineAnnealingLR", "LinearLR", "SequentialLR"):
        setattr(lr_scheduler_stub, name, type(name, (), {}))
    torch_optim_stub = types.ModuleType("torch.optim")
    torch_optim_stub.lr_scheduler = lr_scheduler_stub
    torch_utils_data_stub = types.ModuleType("torch.utils.data")
    torch_utils_data_stub.DataLoader = type("DataLoader", (), {})
    torch_utils_stub = types.ModuleType("torch.utils")
    torch_utils_stub.data = torch_utils_data_stub

    accelerate_stub = types.ModuleType("accelerate")
    accelerate_stub.Accelerator = type("Accelerator", (), {})
    omegaconf_stub = types.ModuleType("omegaconf")
    omegaconf_stub.DictConfig = dict
    omegaconf_stub.OmegaConf = types.SimpleNamespace()
    pil_stub = types.ModuleType("PIL")
    pil_stub.Image = types.SimpleNamespace()

    fastwam_stub = types.ModuleType("fastwam")
    fastwam_stub.__path__ = []
    utils_stub = types.ModuleType("fastwam.utils")
    utils_stub.__path__ = []
    fs_stub = types.ModuleType("fastwam.utils.fs")
    fs_stub.ensure_dir = lambda path: Path(path).mkdir(parents=True, exist_ok=True)
    logging_stub = types.ModuleType("fastwam.utils.logging_config")
    logging_stub.get_logger = logging.getLogger
    logging_stub.setup_logging = lambda *args, **kwargs: None
    pytorch_stub = types.ModuleType("fastwam.utils.pytorch_utils")
    pytorch_stub.set_global_seed = lambda *args, **kwargs: None
    samplers_stub = types.ModuleType("fastwam.utils.samplers")
    samplers_stub.ResumableEpochSampler = type("ResumableEpochSampler", (), {})
    video_io_stub = types.ModuleType("fastwam.utils.video_io")
    video_io_stub.save_mp4 = lambda *args, **kwargs: None
    video_metrics_stub = types.ModuleType("fastwam.utils.video_metrics")
    video_metrics_stub.pil_frames_to_video_tensor = lambda value: value
    video_metrics_stub.video_psnr = lambda **kwargs: 0.0
    video_metrics_stub.video_ssim = lambda **kwargs: 0.0

    module_name = "fastwam._trainer_checkpoint_test"
    spec = importlib.util.spec_from_file_location(module_name, MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(
        sys.modules,
        {
            "numpy": numpy_stub,
            "torch": torch_stub,
            "torch.optim": torch_optim_stub,
            "torch.optim.lr_scheduler": lr_scheduler_stub,
            "torch.utils": torch_utils_stub,
            "torch.utils.data": torch_utils_data_stub,
            "accelerate": accelerate_stub,
            "omegaconf": omegaconf_stub,
            "PIL": pil_stub,
            "fastwam": fastwam_stub,
            "fastwam.utils": utils_stub,
            "fastwam.utils.fs": fs_stub,
            "fastwam.utils.logging_config": logging_stub,
            "fastwam.utils.pytorch_utils": pytorch_stub,
            "fastwam.utils.samplers": samplers_stub,
            "fastwam.utils.video_io": video_io_stub,
            "fastwam.utils.video_metrics": video_metrics_stub,
        },
    ):
        spec.loader.exec_module(module)
    return module


trainer_module = _load_trainer_module()
Wan22Trainer = trainer_module.Wan22Trainer


class _FakeModel:
    def __init__(self):
        self.save_calls = 0

    def save_checkpoint(self, path, optimizer=None, step=None):
        del optimizer
        self.save_calls += 1
        Path(path).write_text(str(step), encoding="utf-8")


class _FakeAccelerator:
    is_main_process = True

    def __init__(self, model):
        self.model = model
        self.save_state_calls = 0

    def wait_for_everyone(self):
        return None

    def unwrap_model(self, model):
        return model

    def save_state(self, output_dir):
        self.save_state_calls += 1
        Path(output_dir, "state.bin").write_text("state", encoding="utf-8")


class TrainerCheckpointFoundationsTest(unittest.TestCase):
    def test_legacy_save_interval_and_deterministic_eval_indices(self):
        cfg = {"save_weights_every": None}
        self.assertEqual(
            trainer_module._optional_interval(cfg, "save_weights_every", 500),
            500,
        )
        self.assertEqual(
            trainer_module._optional_interval(
                {"save_weights_every": 0},
                "save_weights_every",
                500,
            ),
            0,
        )

        kwargs = {
            "dataset_size": 3,
            "eval_seed": 20260717,
            "process_index": 2,
            "num_processes": 4,
            "num_samples": 7,
        }
        indices = trainer_module._deterministic_eval_indices(**kwargs)
        self.assertEqual(indices, trainer_module._deterministic_eval_indices(**kwargs))
        self.assertEqual(len(indices), 7)
        self.assertTrue(all(0 <= index < 3 for index in indices))
        self.assertEqual(
            len(
                trainer_module._deterministic_eval_indices(
                    5,
                    eval_seed=20260717,
                    process_index=0,
                    num_processes=4,
                    num_samples=1,
                )
            ),
            1,
        )
        rank_zero = trainer_module._deterministic_eval_indices(
            32,
            eval_seed=20260717,
            process_index=0,
            num_processes=4,
            num_samples=4,
        )
        rank_one = trainer_module._deterministic_eval_indices(
            32,
            eval_seed=20260717,
            process_index=1,
            num_processes=4,
            num_samples=4,
        )
        self.assertFalse(set(rank_zero) & set(rank_one))

    def test_explicit_weight_steps_are_strictly_validated(self):
        self.assertEqual(
            trainer_module._optional_positive_step_list(
                {"save_weight_steps": None},
                "save_weight_steps",
            ),
            (),
        )
        self.assertEqual(
            trainer_module._optional_positive_step_list(
                {"save_weight_steps": [500, 1000, 3000, 5000, 6000, 6500]},
                "save_weight_steps",
            ),
            (500, 1000, 3000, 5000, 6000, 6500),
        )
        self.assertEqual(
            trainer_module._optional_positive_step_list(
                {"save_weight_steps": []},
                "save_weight_steps",
            ),
            (),
        )

        invalid_values = (
            "500,1000",
            (500, 1000),
            [0],
            [-1],
            [500, 500],
            [1000, 500],
            [True],
            [500.0],
            ["500"],
        )
        for value in invalid_values:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    trainer_module._optional_positive_step_list(
                        {"save_weight_steps": value},
                        "save_weight_steps",
                    )

    def test_weight_save_schedule_unions_interval_explicit_and_final(self):
        kwargs = {
            "save_weights_every": 500,
            "save_weight_steps": (300, 500, 6500),
        }
        self.assertFalse(
            trainer_module._should_save_weights_at_step(
                299, is_final_step=False, **kwargs
            )
        )
        self.assertTrue(
            trainer_module._should_save_weights_at_step(
                300, is_final_step=False, **kwargs
            )
        )
        self.assertTrue(
            trainer_module._should_save_weights_at_step(
                500, is_final_step=False, **kwargs
            )
        )
        self.assertTrue(
            trainer_module._should_save_weights_at_step(
                6500, is_final_step=True, **kwargs
            )
        )
        self.assertTrue(
            trainer_module._should_save_weights_at_step(
                6501,
                save_weights_every=0,
                save_weight_steps=(),
                is_final_step=True,
            )
        )
        self.assertTrue(
            trainer_module._should_save_weights_at_step(
                300,
                save_weights_every=0,
                save_weight_steps=(300,),
                is_final_step=False,
            )
        )

    def test_checkpoint_is_not_written_twice_and_retention_is_owner_scoped(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            weights_dir = root / "weights"
            state_dir = root / "state"
            weights_dir.mkdir()
            state_dir.mkdir()

            model = _FakeModel()
            accelerator = _FakeAccelerator(model)
            trainer = object.__new__(Wan22Trainer)
            trainer.model = model
            trainer.accelerator = accelerator
            trainer.weights_dir = str(weights_dir)
            trainer.state_dir = str(state_dir)
            trainer.checkpoint_owner_id = "owner-a"
            trainer.global_step = 300
            trainer.epoch = 1
            trainer.batch_in_epoch = 2
            trainer.state_keep_last = 2
            trainer.best_checkpoint = None
            trainer.experiment_provenance = {
                "protocol": "fitwam_offline_self_improving_v1",
                "variant": "M",
            }

            first = trainer.save_checkpoint()
            second = trainer.save_checkpoint()
            self.assertEqual(first, second)
            self.assertEqual(model.save_calls, 1)
            self.assertEqual(accelerator.save_state_calls, 1)
            checkpoint_meta = json.loads(
                (state_dir / "step_000300" / "checkpoint_meta.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                checkpoint_meta["experiment_provenance"]["variant"], "M"
            )
            self.assertEqual(
                checkpoint_meta["experiment_provenance_sha256"],
                trainer_module._sha256_json(
                    checkpoint_meta["experiment_provenance"]
                ),
            )

            for step in (100, 200):
                path = state_dir / f"step_{step:06d}"
                path.mkdir()
                trainer_module._atomic_write_json(
                    path / "checkpoint_meta.json",
                    {
                        "complete": True,
                        "owner_id": "owner-a",
                        "global_step": step,
                    },
                )
            foreign = state_dir / "step_000050"
            foreign.mkdir()
            trainer_module._atomic_write_json(
                foreign / "checkpoint_meta.json",
                {
                    "complete": True,
                    "owner_id": "owner-b",
                    "global_step": 50,
                },
            )
            incomplete = state_dir / "step_000075"
            incomplete.mkdir()

            trainer._retain_state_checkpoints()
            self.assertFalse((state_dir / "step_000100").exists())
            self.assertTrue((state_dir / "step_000200").exists())
            self.assertTrue((state_dir / "step_000300").exists())
            self.assertTrue(foreign.exists())
            self.assertTrue(incomplete.exists())

    def test_best_index_and_stop_reason_are_explicit(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            weights_dir = root / "weights"
            weights_dir.mkdir()
            step_weights = weights_dir / "step_000500.pt"
            step_weights.write_text("weights", encoding="utf-8")

            model = _FakeModel()
            trainer = object.__new__(Wan22Trainer)
            trainer.model = model
            trainer.accelerator = _FakeAccelerator(model)
            trainer.weights_dir = str(weights_dir)
            trainer.checkpoint_owner_id = "owner-a"
            trainer.best_checkpoint_path = root / "best_checkpoint.json"
            trainer.stop_reason_path = root / "stop_reason.json"
            trainer.best_checkpoint = None
            trainer.best_metric = "val_loss"
            trainer.best_metric_mode = "min"
            trainer.global_step = 500
            trainer.max_steps = 6500
            trainer.epoch = 2
            trainer.batch_in_epoch = 8

            trainer._update_best_checkpoint_index({"val_loss": 1.25})
            with trainer.best_checkpoint_path.open("r", encoding="utf-8") as f:
                best = json.load(f)
            self.assertEqual(best["step"], 500)
            self.assertEqual(best["weights_path"], str(step_weights))
            self.assertTrue(best["weights_available"])

            trainer._update_best_checkpoint_index({"val_loss": 1.5})
            with trainer.best_checkpoint_path.open("r", encoding="utf-8") as f:
                self.assertEqual(json.load(f)["value"], 1.25)

            trainer._write_stop_reason(status="completed", reason="max_steps")
            with trainer.stop_reason_path.open("r", encoding="utf-8") as f:
                stop = json.load(f)
            self.assertEqual(stop["reason"], "max_steps")
            self.assertEqual(stop["global_step"], 500)


if __name__ == "__main__":
    unittest.main()
