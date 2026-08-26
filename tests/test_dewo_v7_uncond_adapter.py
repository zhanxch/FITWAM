from __future__ import annotations

import unittest

import torch

from fastwam.models.wan22.dewo_v5_train_mode import (
    GROUP_ADAPTER,
    UNCOND_ADAPTER_TRAIN_MODES,
    apply_dewo_v5_uncond_adapter_mode,
    classify_dewo_v5_parameter,
)
from fastwam.models.wan22.dewo_v7_train_mode import (
    apply_dewo_v7_uncond_adapter_mode,
    classify_dewo_v7_parameter,
    collect_dewo_v7_param_groups,
)
from fastwam.models.wan22.uncond_adapter import (
    CHECKPOINT_FORMAT_UNCOND_ADAPTER_V1,
    CFG_MIX_SUBTRACT_BASE,
    CFG_MIX_SUBTRACT_FAIL,
    cfg_mix_subtract_branch,
    inject_uncond_adapter,
    mix_guided_action_epsilon,
    normalize_uncond_adapter_config,
    uncond_adapter_payload,
)


class _FakeAttn(torch.nn.Module):
    def __init__(self, dim: int = 8) -> None:
        super().__init__()
        self.q = torch.nn.Linear(dim, dim)
        self.k = torch.nn.Linear(dim, dim)
        self.v = torch.nn.Linear(dim, dim)
        self.o = torch.nn.Linear(dim, dim)


class _FakeBlock(torch.nn.Module):
    def __init__(self, dim: int = 8) -> None:
        super().__init__()
        self.self_attn = _FakeAttn(dim)
        self.cross_attn = _FakeAttn(dim)


class _FakeExpert(torch.nn.Module):
    def __init__(self, dim: int = 8) -> None:
        super().__init__()
        self.blocks = torch.nn.ModuleList([_FakeBlock(dim)])
        self.head = torch.nn.Linear(dim, dim)


class _TinyMot(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mixtures = torch.nn.ModuleDict(
            {
                "video": _FakeExpert(),
                "action": _FakeExpert(),
            }
        )


class _TinyFastWAM(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mot = _TinyMot()
        self.dit = self.mot
        self.video_expert = self.mot.mixtures["video"]
        self.action_expert = self.mot.mixtures["action"]
        self.proprio_encoder = torch.nn.Linear(3, 8)


class DewoV7UncondAdapterTests(unittest.TestCase):
    def test_v7_is_an_uncond_adapter_train_mode(self) -> None:
        self.assertIn("dewo_v7_uncond_adapter", UNCOND_ADAPTER_TRAIN_MODES)

    def test_v7_freeze_alias_matches_v5(self) -> None:
        self.assertIs(apply_dewo_v7_uncond_adapter_mode, apply_dewo_v5_uncond_adapter_mode)
        self.assertIs(classify_dewo_v7_parameter, classify_dewo_v5_parameter)
        model = _TinyFastWAM()
        inject_uncond_adapter(model, rank=2, alpha=2.0)
        apply_dewo_v7_uncond_adapter_mode(model)
        trainable = [name for name, param in model.named_parameters() if param.requires_grad]
        self.assertTrue(trainable)
        self.assertTrue(all(classify_dewo_v7_parameter(name) == GROUP_ADAPTER for name in trainable))
        groups = collect_dewo_v7_param_groups(model, lr=1e-4, weight_decay=0.01)
        self.assertEqual(groups[0]["name"], GROUP_ADAPTER)

    def test_normalize_config_keeps_v7_fields(self) -> None:
        cfg = normalize_uncond_adapter_config(
            {
                "enabled": True,
                "rank": 16,
                "alpha": 16,
                "pin_video_context_to_base": True,
                "identity_lock_lambda": 0.1,
                "action_residual_lock_lambda": 0.05,
                "video_bc_on_zero_action": False,
                "recipe": "v7",
            }
        )
        self.assertEqual(cfg["recipe"], "v7")
        self.assertFalse(cfg["video_bc_on_zero_action"])
        self.assertTrue(cfg["pin_video_context_to_base"])

    def test_payload_writes_recipe_v7_on_v5_format(self) -> None:
        model = _TinyFastWAM()
        inject_uncond_adapter(model, rank=2, alpha=2.0)
        model.uncond_adapter_config["recipe"] = "v7"
        payload = uncond_adapter_payload(model, step=12, source_checkpoint="/tmp/base.pt")
        self.assertEqual(payload["format"], CHECKPOINT_FORMAT_UNCOND_ADAPTER_V1)
        self.assertEqual(payload["recipe"], "v7")

    def test_mix_subtracts_fail_with_base_origin(self) -> None:
        self.assertEqual(cfg_mix_subtract_branch("v7"), CFG_MIX_SUBTRACT_FAIL)
        self.assertEqual(cfg_mix_subtract_branch("v6"), CFG_MIX_SUBTRACT_BASE)
        base = torch.tensor([[[1.0, 0.0]]])
        posi = torch.tensor([[[4.0, 1.0]]])
        fail = torch.tensor([[[2.0, 3.0]]])
        cfg, delta = mix_guided_action_epsilon(
            base,
            posi,
            mix_weight=1.0,
            subtract=CFG_MIX_SUBTRACT_FAIL,
            epsilon_fail=fail,
        )
        self.assertTrue(torch.equal(delta, posi - fail))
        self.assertTrue(torch.equal(cfg, base + (posi - fail)))
        self.assertFalse(torch.equal(cfg, posi))
        zero, _ = mix_guided_action_epsilon(
            base,
            posi,
            mix_weight=0.0,
            subtract=CFG_MIX_SUBTRACT_FAIL,
            epsilon_fail=fail,
        )
        self.assertTrue(torch.equal(zero, base))
        with self.assertRaisesRegex(ValueError, "epsilon_fail"):
            mix_guided_action_epsilon(
                base,
                posi,
                mix_weight=1.0,
                subtract=CFG_MIX_SUBTRACT_FAIL,
            )

    def test_action_residual_lock_excludes_failure_rows(self) -> None:
        # Mirrors FastWAM.compute_loss: fail rows must not L2-lock Δ_action(c_fail).
        primary_lock_w = torch.tensor([1.0, 1.0, 1.0])
        video_w = torch.tensor([1.0, 0.0, 0.0])
        outcome_flag = torch.tensor([0, 0, 1])
        action_lock_w = (primary_lock_w * (1.0 - video_w)).clamp(min=0)
        fail_row = (outcome_flag.to(dtype=torch.float32) == 1).to(dtype=action_lock_w.dtype)
        action_lock_w = (action_lock_w * (1.0 - fail_row)).clamp(min=0)
        self.assertTrue(torch.equal(action_lock_w, torch.tensor([0.0, 1.0, 0.0])))


if __name__ == "__main__":
    unittest.main()
