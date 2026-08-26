from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn

from fastwam.models.wan22.dewo_v5_train_mode import (
    GROUP_ADAPTER,
    apply_dewo_v5_uncond_adapter_mode,
    classify_dewo_v5_parameter,
)
from fastwam.models.wan22.dewo_v6_train_mode import (
    apply_dewo_v6_uncond_adapter_mode,
    classify_dewo_v6_parameter,
    collect_dewo_v6_param_groups,
)
from fastwam.models.wan22.uncond_adapter import (
    CHECKPOINT_FORMAT_UNCOND_ADAPTER_V1,
    LinearWithUncondAdapter,
    inject_uncond_adapter,
    normalize_uncond_adapter_config,
    pin_video_context_per_sample,
    recommend_adaptive_cfg_tau,
    uncond_adapter_payload,
    uncond_adapter_residual_mse,
    write_adaptive_cfg_tau_json,
)


class _FakeAttn(nn.Module):
    def __init__(self, dim: int = 8) -> None:
        super().__init__()
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)


class _FakeBlock(nn.Module):
    def __init__(self, dim: int = 8) -> None:
        super().__init__()
        self.self_attn = _FakeAttn(dim)
        self.cross_attn = _FakeAttn(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))


class _FakeExpert(nn.Module):
    def __init__(self, dim: int = 8) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([_FakeBlock(dim)])
        self.head = nn.Linear(dim, dim)


class _TinyMot(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mixtures = nn.ModuleDict(
            {
                "video": _FakeExpert(),
                "action": _FakeExpert(),
            }
        )


class _TinyFastWAM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mot = _TinyMot()
        self.dit = self.mot
        self.video_expert = self.mot.mixtures["video"]
        self.action_expert = self.mot.mixtures["action"]
        self.proprio_encoder = nn.Linear(3, 8)


class DewoV6UncondAdapterTests(unittest.TestCase):
    def test_v6_freeze_alias_matches_v5(self) -> None:
        self.assertIs(apply_dewo_v6_uncond_adapter_mode, apply_dewo_v5_uncond_adapter_mode)
        self.assertIs(classify_dewo_v6_parameter, classify_dewo_v5_parameter)
        model = _TinyFastWAM()
        inject_uncond_adapter(model, rank=2, alpha=2.0)
        apply_dewo_v6_uncond_adapter_mode(model)
        trainable = [name for name, param in model.named_parameters() if param.requires_grad]
        self.assertTrue(trainable)
        self.assertTrue(all(classify_dewo_v6_parameter(name) == GROUP_ADAPTER for name in trainable))
        groups = collect_dewo_v6_param_groups(model, lr=1e-4, weight_decay=0.01)
        self.assertEqual(groups[0]["name"], GROUP_ADAPTER)

    def test_normalize_config_keeps_v6_fields(self) -> None:
        cfg = normalize_uncond_adapter_config(
            {
                "enabled": True,
                "rank": 16,
                "alpha": 16,
                "pin_video_context_to_base": True,
                "identity_lock_lambda": 0.1,
                "action_residual_lock_lambda": 0.05,
                "video_bc_on_zero_action": True,
                "recipe": "v6",
            }
        )
        self.assertEqual(cfg["recipe"], "v6")
        self.assertEqual(cfg["action_residual_lock_lambda"], 0.05)
        self.assertTrue(cfg["video_bc_on_zero_action"])
        self.assertTrue(cfg["pin_video_context_to_base"])

    def test_payload_writes_recipe_v6_on_v5_format(self) -> None:
        model = _TinyFastWAM()
        inject_uncond_adapter(model, rank=2, alpha=2.0)
        model.uncond_adapter_config["recipe"] = "v6"
        payload = uncond_adapter_payload(model, step=12, source_checkpoint="/tmp/base.pt")
        self.assertEqual(payload["format"], CHECKPOINT_FORMAT_UNCOND_ADAPTER_V1)
        self.assertEqual(payload["recipe"], "v6")

    def test_residual_is_per_sample_then_weighted_mean(self) -> None:
        model = _TinyFastWAM()
        inject_uncond_adapter(model, rank=2, alpha=2.0)
        apply_dewo_v6_uncond_adapter_mode(model)
        linear = model.action_expert.blocks[0].cross_attn.k
        self.assertIsInstance(linear, LinearWithUncondAdapter)
        linear.uncond_up.data.fill_(0.5)
        model.train()
        _ = linear(torch.randn(4, 3, 8))
        per = linear._last_delta_mse
        self.assertIsNotNone(per)
        self.assertEqual(tuple(per.shape), (4,))
        self.assertGreater(float(per.mean().detach()), 0.0)
        weight = torch.tensor([1.0, 0.0, 0.0, 0.0])
        weighted = uncond_adapter_residual_mse(
            model, expert="action", sample_weight=weight
        )
        self.assertIsNotNone(weighted)
        self.assertTrue(torch.allclose(weighted, per[0], atol=1e-5))
        unweighted = uncond_adapter_residual_mse(model, expert="action")
        self.assertGreater(float(unweighted.detach()), 0.0)

    def test_pin_only_dplus_rows_to_base(self) -> None:
        context = torch.tensor([[[1.0]], [[2.0]]])
        context_mask = torch.tensor([[True], [True]])
        base_context = torch.tensor([[[9.0]], [[8.0]]])
        base_mask = torch.tensor([[False], [False]])
        weight = torch.tensor([1.0, 0.0])
        out_ctx, out_mask = pin_video_context_per_sample(
            context, context_mask, base_context, base_mask, weight
        )
        self.assertTrue(torch.equal(out_ctx[0], base_context[0]))
        self.assertTrue(torch.equal(out_ctx[1], context[1]))
        self.assertFalse(bool(out_mask[0, 0].item()))
        self.assertTrue(bool(out_mask[1, 0].item()))

    def test_recommend_tau_is_plus_quantile_and_fpr_check(self) -> None:
        e_plus = [float(i) for i in range(1, 11)]
        expected = float(torch.quantile(torch.tensor(e_plus, dtype=torch.float64), 0.10).item())
        ok = recommend_adaptive_cfg_tau(e_plus, [0.0] * 20, recall=0.90, max_fpr0=0.05)
        self.assertTrue(ok["separable"])
        self.assertAlmostEqual(ok["tau"], expected, places=6)
        self.assertGreaterEqual(ok["recall_plus"], 0.90)
        self.assertLessEqual(ok["fpr0"], 0.05)
        self.assertEqual(ok["quantile"], 0.10)

        blocked = recommend_adaptive_cfg_tau(e_plus, [9.0] * 20, recall=0.90, max_fpr0=0.05)
        self.assertFalse(blocked["separable"])
        self.assertEqual(blocked["reason"], "fpr0_above_max")
        self.assertAlmostEqual(blocked["tau"], expected, places=6)

        empty = recommend_adaptive_cfg_tau([], [1.0], recall=0.90, max_fpr0=0.05)
        self.assertFalse(empty["separable"])
        self.assertIsNone(empty["tau"])
        self.assertEqual(empty["reason"], "empty_e_plus")

    def test_write_adaptive_cfg_tau_json(self) -> None:
        e_plus = [0.2, 0.3, 0.4, 0.5]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "adaptive_cfg_tau.json"
            payload = write_adaptive_cfg_tau_json(
                path, e_plus, [0.01, 0.02], recall=0.90, max_fpr0=0.05, recipe="v6"
            )
            loaded = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(loaded["recipe"], "v6")
        self.assertIn("separable", loaded)
        self.assertIn("tau", loaded)
        self.assertEqual(payload["n_plus"], 4)


if __name__ == "__main__":
    unittest.main()
