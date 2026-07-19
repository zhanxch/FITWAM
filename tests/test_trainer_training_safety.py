from __future__ import annotations

import importlib.util
import inspect
import logging
import math
import random
import sys
import types
import unittest
from pathlib import Path
from unittest import mock


MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "src" / "fastwam" / "trainer.py"
)


class _FakeTensor:
    def __init__(self, value):
        self.value = value

    def detach(self):
        return self

    def all(self):
        return self

    def to(self, **kwargs):
        del kwargs
        return self

    def item(self):
        return self.value

    def __invert__(self):
        return _FakeTensor(not self.value)


def _load_trainer_module():
    numpy_stub = types.ModuleType("numpy")
    numpy_stub.isfinite = math.isfinite
    numpy_stub.uint8 = "uint8"

    torch_stub = types.ModuleType("torch")
    torch_stub.Tensor = _FakeTensor
    torch_stub.float32 = "float32"
    torch_stub.int64 = "int64"
    torch_stub.OutOfMemoryError = type("OutOfMemoryError", (RuntimeError,), {})
    torch_stub.cuda = types.SimpleNamespace(is_available=lambda: False)
    torch_stub.no_grad = lambda: (lambda fn: fn)
    torch_stub.as_tensor = lambda value, device=None: (
        value if isinstance(value, _FakeTensor) else _FakeTensor(value)
    )
    torch_stub.isfinite = lambda value: _FakeTensor(math.isfinite(value.value))

    class _Generator:
        def __init__(self, device=None):
            del device
            self.rng = random.Random()

        def manual_seed(self, seed):
            self.rng.seed(seed)
            return self

    torch_stub.Generator = _Generator
    torch_stub.randperm = lambda size, generator: types.SimpleNamespace(
        tolist=lambda: list(range(size))
    )

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
    fs_stub.ensure_dir = lambda path: None
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

    module_name = "fastwam._trainer_training_safety_test"
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


class _FakeOptimizer:
    def __init__(self):
        self.zero_grad_calls = 0

    def zero_grad(self, set_to_none):
        self.zero_grad_calls += 1
        self.set_to_none = set_to_none


class _FakeAccelerator:
    def __init__(self, reduced_value=None):
        self.reduced_value = reduced_value
        self.reductions = []

    def reduce(self, value, reduction):
        self.reductions.append((value.item(), reduction))
        if self.reduced_value is None:
            return value
        return _FakeTensor(self.reduced_value)


def _trainer_for_safety(*, reduced_value=None):
    trainer = object.__new__(Wan22Trainer)
    trainer.accelerator = _FakeAccelerator(reduced_value=reduced_value)
    trainer.optimizer = _FakeOptimizer()
    trainer.global_step = 17
    trainer.batch_in_epoch = 3
    trainer._prepare_train_sample = lambda sample: sample
    return trainer


class TrainerTrainingSafetyTest(unittest.TestCase):
    def test_non_finite_loss_is_synchronized_and_clears_gradients(self):
        trainer = _trainer_for_safety()
        with self.assertRaises(trainer_module.NonFiniteTrainingError):
            trainer._raise_if_non_finite(
                _FakeTensor(float("nan")),
                value_name="loss",
                device="cpu",
            )
        self.assertEqual(trainer.optimizer.zero_grad_calls, 1)
        self.assertEqual(trainer.accelerator.reductions, [(True, "max")])

    def test_remote_rank_non_finite_also_stops_local_rank(self):
        trainer = _trainer_for_safety(reduced_value=1)
        with self.assertRaises(trainer_module.NonFiniteTrainingError):
            trainer._raise_if_non_finite(
                _FakeTensor(1.0),
                value_name="gradient norm",
                device="cpu",
            )
        self.assertEqual(trainer.optimizer.zero_grad_calls, 1)

    def test_finite_value_leaves_gradients_unchanged(self):
        trainer = _trainer_for_safety()
        trainer._raise_if_non_finite(
            _FakeTensor(1.0),
            value_name="loss",
            device="cpu",
        )
        self.assertEqual(trainer.optimizer.zero_grad_calls, 0)

    def test_data_pipeline_errors_are_wrapped_but_oom_is_not(self):
        trainer = _trainer_for_safety()
        with self.assertRaises(trainer_module.TrainingDataError) as context:
            trainer._next_train_sample(iter(_RaisingIterator(OSError("bad shard"))))
        self.assertIsInstance(context.exception.__cause__, OSError)

        oom = trainer_module.torch.OutOfMemoryError("CUDA out of memory")
        with self.assertRaises(trainer_module.torch.OutOfMemoryError):
            trainer._next_train_sample(iter(_RaisingIterator(oom)))

    def test_stop_reason_classification(self):
        cases = [
            (trainer_module.NonFiniteTrainingError("nan"), "nan"),
            (trainer_module.torch.OutOfMemoryError("oom"), "oom"),
            (RuntimeError("CUDA out of memory"), "oom"),
            (trainer_module.TrainingDataError("input"), "data_error"),
            (KeyboardInterrupt(), "manual"),
            (ValueError("model bug"), "exception"),
        ]
        for error, expected in cases:
            with self.subTest(error=type(error).__name__):
                self.assertEqual(
                    trainer_module._classify_stop_reason(error),
                    expected,
                )

    def test_train_records_reason_and_reraises(self):
        for error, expected_reason, expected_status in [
            (trainer_module.NonFiniteTrainingError("nan"), "nan", "failed"),
            (trainer_module.torch.OutOfMemoryError("oom"), "oom", "failed"),
            (trainer_module.TrainingDataError("input"), "data_error", "failed"),
            (KeyboardInterrupt(), "manual", "interrupted"),
            (RuntimeError("other"), "exception", "failed"),
        ]:
            with self.subTest(reason=expected_reason):
                trainer = object.__new__(Wan22Trainer)
                trainer._train_impl = mock.Mock(side_effect=error)
                trainer._write_stop_reason = mock.Mock()
                trainer._finish_wandb = mock.Mock()
                with self.assertRaises(type(error)):
                    trainer.train()
                trainer._write_stop_reason.assert_called_once_with(
                    status=expected_status,
                    reason=expected_reason,
                    error=error,
                )
                trainer._finish_wandb.assert_called_once_with()

    def test_checks_precede_backward_and_optimizer_step(self):
        source = inspect.getsource(Wan22Trainer._train_impl)
        loss_check = source.index('value_name="loss"')
        backward = source.index("self.accelerator.backward(loss)")
        grad_check = source.index('value_name="gradient norm"')
        optimizer_step = source.index("self.optimizer.step()")
        self.assertLess(loss_check, backward)
        self.assertLess(grad_check, optimizer_step)


class _RaisingIterator:
    def __init__(self, error):
        self.error = error

    def __iter__(self):
        return self

    def __next__(self):
        raise self.error


if __name__ == "__main__":
    unittest.main()
