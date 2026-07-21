from __future__ import annotations

import importlib.util
import logging
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
RUNTIME_PATH = ROOT / "src" / "fastwam" / "runtime.py"


class _RandomStream:
    def __init__(self, name: str, events: list[tuple[str, int] | tuple[str]]) -> None:
        self.name = name
        self.events = events
        self.state = 0

    def seed(self, seed: int) -> None:
        self.events.append((f"{self.name}_seed", int(seed)))
        self.state = int(seed)

    def random(self) -> float:
        self.state = (1103515245 * self.state + 12345) % (2**31)
        return self.state / float(2**31)


def _load_runtime_module():
    events: list[tuple[str, int] | tuple[str]] = []

    numpy_stream = _RandomStream("numpy", events)
    numpy_stub = types.ModuleType("numpy")
    numpy_stub.random = numpy_stream
    numpy_stub.float32 = "float32"

    torch_stream = _RandomStream("torch", events)
    distributed_stub = types.ModuleType("torch.distributed")
    distributed_stub.is_initialized = lambda: False
    distributed_stub.get_rank = lambda: 0

    cuda_stub = types.SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 4,
        manual_seed_all=lambda seed: events.append(("cuda_seed", int(seed))),
    )
    torch_stub = types.ModuleType("torch")
    torch_stub.dtype = object
    torch_stub.float32 = "float32"
    torch_stub.float16 = "float16"
    torch_stub.bfloat16 = "bfloat16"
    torch_stub.distributed = distributed_stub
    torch_stub.cuda = cuda_stub
    torch_stub.manual_seed = torch_stream.seed
    torch_stub.random_value = torch_stream.random

    hydra_stub = types.ModuleType("hydra")
    hydra_utils_stub = types.ModuleType("hydra.utils")
    hydra_utils_stub.instantiate = lambda *_args, **_kwargs: None
    hydra_stub.utils = hydra_utils_stub

    class _OmegaConf:
        @staticmethod
        def to_container(_cfg, resolve=True):
            del resolve
            return {}

        @staticmethod
        def save(_payload, handle):
            handle.write("{}\n")

    omegaconf_stub = types.ModuleType("omegaconf")
    omegaconf_stub.DictConfig = object
    omegaconf_stub.OmegaConf = _OmegaConf

    pil_stub = types.ModuleType("PIL")
    pil_image_stub = types.ModuleType("PIL.Image")
    pil_image_stub.Image = type("Image", (), {})
    pil_stub.Image = pil_image_stub

    einops_stub = types.ModuleType("einops")
    einops_stub.repeat = lambda value, *_args, **_kwargs: value

    fastwam_stub = types.ModuleType("fastwam")
    fastwam_stub.__path__ = []
    trainer_stub = types.ModuleType("fastwam.trainer")
    trainer_stub.Wan22Trainer = type("Wan22Trainer", (), {})

    utils_stub = types.ModuleType("fastwam.utils")
    utils_stub.__path__ = []
    logging_stub = types.ModuleType("fastwam.utils.logging_config")
    logging_stub.get_logger = logging.getLogger
    logging_stub.setup_logging = lambda *_args, **_kwargs: None
    video_io_stub = types.ModuleType("fastwam.utils.video_io")
    video_io_stub.save_mp4 = lambda *_args, **_kwargs: None
    misc_stub = types.ModuleType("fastwam.utils.misc")
    misc_stub.register_work_dir = lambda _path: None
    misc_stub.get_work_dir = lambda: "."
    utils_stub.misc = misc_stub

    module_name = "fastwam._runtime_training_seed_test"
    spec = importlib.util.spec_from_file_location(module_name, RUNTIME_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(
        sys.modules,
        {
            "numpy": numpy_stub,
            "torch": torch_stub,
            "torch.distributed": distributed_stub,
            "hydra": hydra_stub,
            "hydra.utils": hydra_utils_stub,
            "omegaconf": omegaconf_stub,
            "PIL": pil_stub,
            "PIL.Image": pil_image_stub,
            "einops": einops_stub,
            "fastwam": fastwam_stub,
            "fastwam.trainer": trainer_stub,
            "fastwam.utils": utils_stub,
            "fastwam.utils.logging_config": logging_stub,
            "fastwam.utils.video_io": video_io_stub,
            "fastwam.utils.misc": misc_stub,
        },
    ):
        spec.loader.exec_module(module)

    module.random = _RandomStream("python", events)
    return module, numpy_stub, torch_stub, events


class RuntimeTrainingSeedTest(unittest.TestCase):
    def test_model_is_instantiated_after_seeding_and_repeats_exactly(self) -> None:
        runtime, numpy_stub, torch_stub, events = _load_runtime_module()
        initialized_models = []
        instantiation_devices = []

        class _Trainer:
            def __init__(self, *, cfg, model, train_dataset, val_dataset):
                del cfg, train_dataset, val_dataset
                initialized_models.append(model)

            def train(self):
                return None

        def instantiate_model(_config, **kwargs):
            events.append(("instantiate",))
            instantiation_devices.append(kwargs["device"])
            return (
                runtime.random.random(),
                numpy_stub.random.random(),
                torch_stub.random_value(),
            )

        runtime.Wan22Trainer = _Trainer
        runtime.instantiate = instantiate_model
        runtime.build_datasets = lambda _cfg: (object(), object())

        with tempfile.TemporaryDirectory() as output_dir:
            cfg = SimpleNamespace(
                output_dir=output_dir,
                mixed_precision="bf16",
                seed=73,
                model=object(),
                data=object(),
            )
            with mock.patch.dict(os.environ, {"LOCAL_RANK": "2"}):
                runtime.run_training(cfg)
                first_events = list(events)

                runtime.random.seed(999)
                numpy_stub.random.seed(999)
                torch_stub.manual_seed(999)
                events.clear()
                runtime.run_training(cfg)
                second_events = list(events)

        expected_prefix = [
            ("python_seed", 73),
            ("numpy_seed", 73),
            ("torch_seed", 73),
            ("cuda_seed", 73),
            ("instantiate",),
        ]
        self.assertEqual(first_events[:5], expected_prefix)
        self.assertEqual(second_events[:5], expected_prefix)
        self.assertEqual(instantiation_devices, ["cuda:2", "cuda:2"])
        self.assertEqual(initialized_models[0], initialized_models[1])

    def test_trainer_keeps_rank_aware_worker_seeding(self) -> None:
        trainer_source = (ROOT / "src" / "fastwam" / "trainer.py").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            "worker_init_fn = set_global_seed(self.seed, get_worker_init_fn=True)",
            trainer_source,
        )


if __name__ == "__main__":
    unittest.main()
