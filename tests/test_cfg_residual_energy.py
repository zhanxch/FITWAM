from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fastwam.models.wan22.fastwam import action_cfg_residual_energy
from fastwam.models.wan22.uncond_adapter import (
    bound_cfg_residual,
    normalize_cfg_residual_clip_mode,
)


class CfgResidualEnergyTests(unittest.TestCase):
    def test_per_token_rms(self) -> None:
        delta = torch.zeros(2, 4, 3)
        delta[0, 1] = torch.tensor([3.0, 4.0, 0.0])
        rms = action_cfg_residual_energy(delta)
        self.assertEqual(tuple(rms.shape), (2, 4))
        self.assertAlmostEqual(float(rms[0, 0]), 0.0)
        self.assertAlmostEqual(float(rms[0, 1]), ((9.0 + 16.0) / 3.0) ** 0.5, places=5)
        self.assertAlmostEqual(float(rms[1].max()), 0.0)

    def test_rejects_wrong_rank(self) -> None:
        with self.assertRaises(ValueError):
            action_cfg_residual_energy(torch.zeros(4, 3))

    def test_epsilon_l_rms_bound_preserves_direction(self) -> None:
        delta = torch.tensor([[[3.0, 4.0], [0.1, 0.0]]])
        bounded = bound_cfg_residual(delta, 1.0, mode="rms")
        self.assertAlmostEqual(float(action_cfg_residual_energy(bounded)[0, 0]), 1.0, places=5)
        self.assertTrue(torch.allclose(bounded[0, 0] / bounded[0, 0].norm(), delta[0, 0] / delta[0, 0].norm()))
        self.assertTrue(torch.allclose(bounded[0, 1], delta[0, 1]))

    def test_epsilon_l_elementwise_and_zero(self) -> None:
        delta = torch.tensor([[[-2.0, 0.25, 3.0]]])
        bounded = bound_cfg_residual(delta, 0.5, mode="elementwise")
        self.assertTrue(torch.equal(bounded, torch.tensor([[[-0.5, 0.25, 0.5]]])))
        self.assertTrue(torch.equal(bound_cfg_residual(delta, 0.0), torch.zeros_like(delta)))

    def test_none_is_identity_and_mode_aliases(self) -> None:
        delta = torch.randn(1, 2, 3)
        self.assertIs(bound_cfg_residual(delta, None), delta)
        self.assertEqual(normalize_cfg_residual_clip_mode("token_rms"), "rms")
        with self.assertRaises(ValueError):
            bound_cfg_residual(delta, 0.1, mode="bad")
        with self.assertRaises(ValueError):
            bound_cfg_residual(delta, -0.1)


if __name__ == "__main__":
    unittest.main()
