from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fastwam.trainer import (  # noqa: E402
    Wan22Trainer,
    _validate_pair_loss_accumulation_protocol,
)


class _FakeAccelerator:
    def __init__(self, model, *, num_processes: int, global_weight: float):
        self._model = model
        self.num_processes = num_processes
        self.device = torch.device("cpu")
        self.global_weight = float(global_weight)

    def unwrap_model(self, model):
        self.test_model_identity = model is self._model
        return model

    def reduce(self, value, reduction):
        if reduction != "sum":
            raise AssertionError(reduction)
        return torch.tensor(
            self.global_weight,
            device=value.device,
            dtype=value.dtype,
        )


class TrainerOfflinePairScalingTest(unittest.TestCase):
    def test_pair_loss_rejects_gradient_accumulation_above_one(self):
        model = SimpleNamespace(
            offline_steer_enabled=True,
            offline_steer_config={"pair_loss_weight": 0.1},
        )
        _validate_pair_loss_accumulation_protocol(model, 1)
        with self.assertRaisesRegex(
            ValueError,
            "gradient_accumulation_steps=1",
        ):
            _validate_pair_loss_accumulation_protocol(model, 2)

    def make_trainer(
        self,
        *,
        pair_loss_weight: float = 0.1,
        warmup_steps: int = 500,
        global_weight: float = 4.0,
    ):
        trainer = object.__new__(Wan22Trainer)
        model = SimpleNamespace(
            offline_steer_enabled=True,
            offline_steer_config={
                "pair_loss_weight": pair_loss_weight,
                "pair_loss_warmup_steps": warmup_steps,
            },
        )
        trainer.model = model
        trainer.accelerator = _FakeAccelerator(
            model,
            num_processes=4,
            global_weight=global_weight,
        )
        trainer.global_step = 0
        return trainer

    def test_pair_warmup_reaches_one_at_configured_step(self):
        trainer = self.make_trainer()
        self.assertAlmostEqual(trainer._offline_pair_loss_scale(), 1.0 / 500.0)
        trainer.global_step = 499
        self.assertEqual(trainer._offline_pair_loss_scale(), 1.0)
        trainer.global_step = 700
        self.assertEqual(trainer._offline_pair_loss_scale(), 1.0)

    def test_ddp_scale_recovers_global_weighted_denominator(self):
        trainer = self.make_trainer(global_weight=4.0)
        scale = trainer._offline_pair_loss_ddp_scale(
            {"pair_weight": torch.tensor([0.75, 0.25, 0.0, 0.0])}
        )
        self.assertEqual(scale, 1.0)

        trainer.accelerator.global_weight = 8.0
        scale = trainer._offline_pair_loss_ddp_scale(
            {"pair_weight": torch.tensor([0.75, 0.25, 0.0, 0.0])}
        )
        self.assertEqual(scale, 0.5)

    def test_no_global_pair_weight_returns_zero_scale(self):
        trainer = self.make_trainer(global_weight=0.0)
        scale = trainer._offline_pair_loss_ddp_scale(
            {"pair_weight": torch.zeros(4)}
        )
        self.assertEqual(scale, 0.0)


if __name__ == "__main__":
    unittest.main()
