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


ROOT = Path(__file__).resolve().parents[1]
SAMPLER_PATH = ROOT / "src" / "fastwam" / "utils" / "role_balanced_sampler.py"
TRAINER_PATH = ROOT / "src" / "fastwam" / "trainer.py"
DATASET_PATH = (
    ROOT / "src" / "fastwam" / "datasets" / "eve" / "manifest_dataset.py"
)


class _Sampler:
    def __init__(self, data_source=None):
        self.data_source = data_source

    @classmethod
    def __class_getitem__(cls, item):
        del item
        return cls


class _FakeTensor:
    def __init__(self, value):
        self.value = value
        self.moves = []

    def to(self, *, device, non_blocking):
        self.moves.append((device, non_blocking))
        return self


class _GatheredHashes:
    def __init__(self, values):
        self.values = list(values)

    def reshape(self, *shape):
        del shape
        return self

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return list(self.values)


class _FakeDataLoader:
    def __init__(self, dataset, **kwargs):
        self.dataset = dataset
        self.kwargs = kwargs
        self.batch_sampler = kwargs.get("batch_sampler")


def _torch_modules():
    torch_stub = types.ModuleType("torch")
    torch_stub.Tensor = _FakeTensor
    torch_stub.float32 = "float32"
    torch_stub.int64 = "int64"
    torch_stub.long = "long"
    torch_stub.cuda = types.SimpleNamespace(is_available=lambda: False)
    torch_stub.no_grad = lambda: (lambda fn: fn)
    torch_stub.is_tensor = lambda value: isinstance(value, _FakeTensor)
    torch_stub.tensor = lambda value, dtype=None: value

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

    torch_utils_data = types.ModuleType("torch.utils.data")
    torch_utils_data.DataLoader = _FakeDataLoader
    torch_utils_data.Sampler = _Sampler
    torch_utils = types.ModuleType("torch.utils")
    torch_utils.data = torch_utils_data

    lr_scheduler = types.ModuleType("torch.optim.lr_scheduler")
    for name in ("ConstantLR", "CosineAnnealingLR", "LinearLR", "SequentialLR"):
        setattr(lr_scheduler, name, type(name, (), {}))
    torch_optim = types.ModuleType("torch.optim")
    torch_optim.lr_scheduler = lr_scheduler

    return {
        "torch": torch_stub,
        "torch.utils": torch_utils,
        "torch.utils.data": torch_utils_data,
        "torch.optim": torch_optim,
        "torch.optim.lr_scheduler": lr_scheduler,
    }


TORCH_MODULES = _torch_modules()


def _load_sampler_module():
    module_name = "fastwam.utils.role_balanced_sampler"
    spec = importlib.util.spec_from_file_location(module_name, SAMPLER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, TORCH_MODULES):
        spec.loader.exec_module(module)
    return module


role_sampler_module = _load_sampler_module()
RoleBalancedBatchSampler = role_sampler_module.RoleBalancedBatchSampler


def _load_trainer_module():
    numpy_stub = types.ModuleType("numpy")
    numpy_stub.isfinite = math.isfinite
    numpy_stub.uint8 = "uint8"
    accelerate_stub = types.ModuleType("accelerate")
    accelerate_stub.Accelerator = type("Accelerator", (), {})
    omegaconf_stub = types.ModuleType("omegaconf")
    omegaconf_stub.DictConfig = dict
    omegaconf_stub.OmegaConf = types.SimpleNamespace()
    pil_stub = types.ModuleType("PIL")
    pil_stub.Image = types.SimpleNamespace()

    fastwam_stub = types.ModuleType("fastwam")
    fastwam_stub.__path__ = []
    datasets_stub = types.ModuleType("fastwam.datasets")
    datasets_stub.__path__ = []
    vae_stub = types.ModuleType("fastwam.datasets.vae_latent_cache")
    vae_stub.collate_robot_video_batch = lambda batch: batch
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

    module_name = "fastwam._trainer_role_balanced_test"
    spec = importlib.util.spec_from_file_location(module_name, TRAINER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    modules = {
        **TORCH_MODULES,
        "numpy": numpy_stub,
        "accelerate": accelerate_stub,
        "omegaconf": omegaconf_stub,
        "PIL": pil_stub,
        "fastwam": fastwam_stub,
        "fastwam.datasets": datasets_stub,
        "fastwam.datasets.vae_latent_cache": vae_stub,
        "fastwam.utils": utils_stub,
        "fastwam.utils.fs": fs_stub,
        "fastwam.utils.logging_config": logging_stub,
        "fastwam.utils.pytorch_utils": pytorch_stub,
        "fastwam.utils.role_balanced_sampler": role_sampler_module,
        "fastwam.utils.samplers": samplers_stub,
        "fastwam.utils.video_io": video_io_stub,
        "fastwam.utils.video_metrics": video_metrics_stub,
    }
    with mock.patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
    return module


trainer_module = _load_trainer_module()
Wan22Trainer = trainer_module.Wan22Trainer


def _load_manifest_dataset_module():
    base_stub = types.ModuleType(
        "fastwam.datasets.lerobot.robot_video_dataset"
    )
    base_stub.RobotVideoDataset = type("RobotVideoDataset", (), {})
    schema_stub = types.ModuleType("fastwam.everobot_schema")
    schema_stub.resolve_manifest_dataset_root = lambda *args, **kwargs: ""
    schema_stub.validate_manifest = lambda *args, **kwargs: None
    logging_stub = types.ModuleType("fastwam.utils.logging_config")
    logging_stub.get_logger = logging.getLogger

    module_name = "_test_role_balanced_eve_manifest"
    spec = importlib.util.spec_from_file_location(module_name, DATASET_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    modules = {
        **TORCH_MODULES,
        "fastwam.datasets.lerobot.robot_video_dataset": base_stub,
        "fastwam.everobot_schema": schema_stub,
        "fastwam.utils.logging_config": logging_stub,
    }
    with mock.patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
    return module


manifest_dataset_module = _load_manifest_dataset_module()
EveManifestRobotVideoDataset = (
    manifest_dataset_module.EveManifestRobotVideoDataset
)


class _Dataset:
    def __init__(self):
        self.sampling_roles = tuple(
            ["primary"] * 64 + ["auxiliary"] * 64
        )

    def __len__(self):
        return len(self.sampling_roles)


class _Accelerator:
    num_processes = 4
    process_index = 2
    device = "cuda:2"

    def __init__(self):
        self.prepare_calls = []
        self.gathered_hashes = None

    def prepare(self, *values):
        self.prepare_calls.append(values)
        return values

    def gather(self, value):
        hashes = self.gathered_hashes
        if hashes is None:
            hashes = value
        return _GatheredHashes(hashes)

    def wait_for_everyone(self):
        return None


def _trainer(*, enabled: bool) -> Wan22Trainer:
    trainer = object.__new__(Wan22Trainer)
    trainer.role_balanced_sampling_enabled = enabled
    trainer.role_balanced_primary_per_batch = 2
    trainer.role_balanced_seed = 19
    trainer.batch_size = 4
    trainer.num_workers = 0
    trainer.seed = 11
    trainer.accelerator = _Accelerator()
    trainer.model = object()
    trainer.optimizer = object()
    trainer.scheduler = object()
    trainer.train_loader = object()
    return trainer


def _build_role_loader(trainer, dataset):
    fastwam_package = types.ModuleType("fastwam")
    fastwam_package.__path__ = []
    utils_package = types.ModuleType("fastwam.utils")
    utils_package.__path__ = []
    with mock.patch.dict(
        sys.modules,
        {
            "fastwam": fastwam_package,
            "fastwam.utils": utils_package,
            "fastwam.utils.role_balanced_sampler": role_sampler_module,
        },
    ):
        return trainer._build_loader(dataset)


class TrainerRoleBalancedLoaderTest(unittest.TestCase):
    def test_role_balanced_loader_sets_explicit_deepspeed_batch_contract(self):
        trainer = _trainer(enabled=True)
        trainer.gradient_accumulation_steps = 1
        trainer.accelerator.state = types.SimpleNamespace(
            deepspeed_plugin=types.SimpleNamespace(
                deepspeed_config={
                    "train_batch_size": "auto",
                    "train_micro_batch_size_per_gpu": "auto",
                    "gradient_accumulation_steps": "auto",
                }
            )
        )

        trainer._configure_role_balanced_deepspeed_batch_contract()

        self.assertEqual(
            trainer.accelerator.state.deepspeed_plugin.deepspeed_config,
            {
                "train_batch_size": 16,
                "train_micro_batch_size_per_gpu": 4,
                "gradient_accumulation_steps": 1,
            },
        )

    def test_role_balanced_loader_rejects_conflicting_deepspeed_batch(self):
        trainer = _trainer(enabled=True)
        trainer.gradient_accumulation_steps = 1
        trainer.accelerator.state = types.SimpleNamespace(
            deepspeed_plugin=types.SimpleNamespace(
                deepspeed_config={
                    "train_micro_batch_size_per_gpu": 2,
                }
            )
        )

        with self.assertRaisesRegex(ValueError, "batch contract disagrees"):
            trainer._configure_role_balanced_deepspeed_batch_contract()

    def test_loader_enables_persistent_prefetch_when_num_workers_positive(self):
        trainer = _trainer(enabled=True)
        trainer.num_workers = 4
        loader = _build_role_loader(trainer, _Dataset())
        self.assertTrue(loader.kwargs["persistent_workers"])
        self.assertEqual(loader.kwargs["prefetch_factor"], 4)

    def test_loader_omits_persistent_prefetch_when_num_workers_zero(self):
        trainer = _trainer(enabled=True)
        loader = _build_role_loader(trainer, _Dataset())
        self.assertNotIn("persistent_workers", loader.kwargs)
        self.assertNotIn("prefetch_factor", loader.kwargs)

    def test_eve_sampling_roles_follow_manifest_and_keep_failures_auxiliary(self):
        role = EveManifestRobotVideoDataset._sampling_role

        self.assertEqual(
            role({"episode_outcome": "success", "action_loss": "enabled"}),
            "primary",
        )
        self.assertEqual(
            role({"episode_outcome": "success", "action_loss": "disabled"}),
            "auxiliary_success",
        )
        self.assertEqual(
            role(
                {
                    "episode_outcome": "success",
                    "action_loss": "enabled",
                    "batch_role": "auxiliary",
                }
            ),
            "auxiliary_success",
        )
        self.assertEqual(
            role({"episode_outcome": "failure", "action_loss": "disabled"}),
            "auxiliary",
        )
        self.assertEqual(
            role(
                {
                    "episode_outcome": "failure",
                    "event_outcome": "success",
                    "action_loss": "enabled",
                }
            ),
            "auxiliary",
        )
        with self.assertRaisesRegex(ValueError, "Failure"):
            role(
                {
                    "episode_outcome": "failure",
                    "action_loss": "disabled",
                    "batch_role": "primary",
                }
            )

        schedule = EveManifestRobotVideoDataset._cfg_schedule
        self.assertEqual(
            schedule({"episode_outcome": "success", "action_loss": "enabled"}),
            "primary",
        )
        self.assertEqual(
            schedule({"episode_outcome": "success", "action_loss": "disabled"}),
            "aux_success",
        )
        self.assertEqual(
            schedule({"episode_outcome": "failure", "action_loss": "disabled"}),
            "aux_failure",
        )

    def test_each_rank_local_batch_is_exactly_two_primary_two_auxiliary(self):
        dataset = _Dataset()
        trainers = []
        for rank in range(4):
            trainer = _trainer(enabled=True)
            trainer.accelerator.process_index = rank
            _build_role_loader(trainer, dataset)
            trainers.append(trainer)

        for trainer in trainers:
            for batch in trainer.train_sampler:
                roles = [dataset.sampling_roles[index] for index in batch]
                self.assertEqual(len(batch), 4)
                self.assertEqual(roles.count("primary"), 2)
                self.assertEqual(roles.count("auxiliary"), 2)

    def test_accelerate_does_not_receive_the_rank_sharded_loader(self):
        trainer = _trainer(enabled=True)
        local_loader = trainer.train_loader

        trainer._prepare_training_components()

        self.assertIs(trainer.train_loader, local_loader)
        self.assertEqual(len(trainer.accelerator.prepare_calls), 1)
        self.assertEqual(len(trainer.accelerator.prepare_calls[0]), 3)
        self.assertNotIn(local_loader, trainer.accelerator.prepare_calls[0])

        legacy = _trainer(enabled=False)
        legacy_loader = legacy.train_loader
        legacy._prepare_training_components()
        self.assertEqual(len(legacy.accelerator.prepare_calls[0]), 4)
        self.assertIn(legacy_loader, legacy.accelerator.prepare_calls[0])

    def test_raw_loader_batch_is_moved_to_the_rank_device(self):
        trainer = _trainer(enabled=True)
        tensor = _FakeTensor(3)
        sample = {"tensor": tensor, "metadata": ["keep", {"id": "x"}]}

        moved = trainer._prepare_train_sample(sample)

        self.assertIs(moved["tensor"], tensor)
        self.assertEqual(tensor.moves, [("cuda:2", True)])
        self.assertEqual(moved["metadata"], ["keep", {"id": "x"}])

    def test_sampler_epoch_and_batch_offset_round_trip_with_trainer_state(self):
        trainer = _trainer(enabled=True)
        _build_role_loader(trainer, _Dataset())
        trainer.global_step = 12
        trainer.epoch = 3
        trainer.batch_in_epoch = 2

        trainer.train_sampler.set_epoch(3)
        full_epoch = list(trainer.train_sampler)
        self.assertGreater(len(full_epoch), 2)

        with TemporaryDirectory() as temp_dir:
            trainer._save_trainer_state(temp_dir)
            payload = json.loads(
                (Path(temp_dir) / "trainer_state.json").read_text(
                    encoding="utf-8"
                )
            )

        restored = _trainer(enabled=True)
        _build_role_loader(restored, _Dataset())
        restored._restore_train_sampler_progress(
            epoch=payload["epoch"],
            batch_in_epoch=payload["batch_in_epoch"],
            sampler_state=payload["train_sampler_state"],
        )

        self.assertEqual(restored.train_sampler.epoch, 3)
        self.assertEqual(restored.train_sampler.batch_offset, 2)
        self.assertEqual(list(restored.train_sampler), full_epoch[2:])

    def test_output_dir_gate_accepts_one_path_and_rejects_rank_split(self):
        trainer = _trainer(enabled=True)
        trainer.output_dir = "./runs/shared-run"
        path_hash = trainer_module._stable_output_dir_hash(trainer.output_dir)
        trainer.accelerator.gathered_hashes = [path_hash] * 4

        with mock.patch.object(
            trainer_module.torch,
            "tensor",
            side_effect=lambda values, **kwargs: list(values),
            create=True,
        ):
            trainer._assert_output_dir_consistent()

            trainer.accelerator.gathered_hashes = [
                path_hash,
                path_hash,
                path_hash + 1,
                path_hash,
            ]
            with self.assertRaisesRegex(
                RuntimeError,
                "Inconsistent `output_dir` across ranks",
            ):
                trainer._assert_output_dir_consistent()

if __name__ == "__main__":
    unittest.main()
