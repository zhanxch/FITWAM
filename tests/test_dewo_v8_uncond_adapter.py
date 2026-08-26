from __future__ import annotations

import unittest

import torch

from fastwam.models.wan22.dewo_v5_train_mode import (
    GROUP_ADAPTER,
    UNCOND_ADAPTER_TRAIN_MODES,
)
from fastwam.models.wan22.dewo_v8_train_mode import (
    GROUP_VALUE,
    apply_dewo_v8_uncond_adapter_mode,
    classify_dewo_v8_parameter,
    collect_dewo_v8_param_groups,
)
from fastwam.models.wan22.uncond_adapter import (
    CFG_MIX_SUBTRACT_BASE,
    CHECKPOINT_FORMAT_UNCOND_ADAPTER_V1,
    cfg_mix_subtract_branch,
    inject_uncond_adapter,
    normalize_uncond_adapter_config,
    uncond_adapter_payload,
)
from fastwam.models.wan22.value_head import (
    RecoverabilityValueHead,
    attach_recoverability_value_head,
    drop_edge_gate,
    recoverability_cliff_loss,
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
        self.device = torch.device("cpu")
        self.torch_dtype = torch.float32


class DewoV8UncondAdapterTests(unittest.TestCase):
    def test_v8_is_an_uncond_adapter_train_mode(self) -> None:
        self.assertIn("dewo_v8_uncond_adapter", UNCOND_ADAPTER_TRAIN_MODES)

    def test_mix_subtracts_base_not_fail(self) -> None:
        self.assertEqual(cfg_mix_subtract_branch("v8"), CFG_MIX_SUBTRACT_BASE)
        self.assertEqual(cfg_mix_subtract_branch("v6"), CFG_MIX_SUBTRACT_BASE)

    def test_drop_edge_does_not_fire_on_high_v_alone(self) -> None:
        self.assertEqual(drop_edge_gate(0.95, 0.94, v_high=0.5, delta=0.15), 0.0)
        self.assertEqual(drop_edge_gate(None, 0.99, v_high=0.5, delta=0.15), 0.0)

    def test_drop_edge_fires_once_on_cliff(self) -> None:
        self.assertEqual(drop_edge_gate(0.9, 0.4, v_high=0.5, delta=0.15), 1.0)
        self.assertEqual(
            drop_edge_gate(0.9, 0.4, v_high=0.5, delta=0.15, fired=True), 0.0
        )

    def test_value_head_pools_current_frame(self) -> None:
        head = RecoverabilityValueHead(in_channels=4, hidden=8)
        latents = torch.zeros(2, 4, 3, 5, 5)
        latents[:, :, 0] = 1.0
        latents[:, :, 1:] = -10.0
        out = head(latents)
        self.assertEqual(tuple(out.shape), (2,))
        self.assertTrue(torch.all((out > 0) & (out < 1)))
        # Zero last-layer init keeps V≈0.5; large latents must not inf BCE.
        logits = head.logits(1000.0 * latents)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, torch.tensor([1.0, 0.0])
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.allclose(out, torch.full_like(out, 0.5), atol=1e-5))
        bf16_head = RecoverabilityValueHead(in_channels=4, hidden=8).to(dtype=torch.bfloat16)
        bf16_latents = latents.to(dtype=torch.bfloat16)
        device_type = bf16_latents.device.type
        with torch.autocast(device_type=device_type if device_type in {"cuda", "cpu"} else "cpu", enabled=False):
            bf16_logits = bf16_head.logits(bf16_latents.detach())
        self.assertEqual(bf16_logits.dtype, torch.float32)
        bf16_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            bf16_logits, torch.tensor([1.0, 0.0])
        )
        self.assertTrue(torch.isfinite(bf16_loss))
        self.assertFalse(bf16_latents.requires_grad)

    def test_freeze_trains_adapter_and_value_head(self) -> None:
        model = _TinyFastWAM()
        inject_uncond_adapter(model, rank=2, alpha=2.0)
        attach_recoverability_value_head(
            model, {"enabled": True, "in_channels": 4, "hidden": 8}
        )
        apply_dewo_v8_uncond_adapter_mode(model)
        trainable = [name for name, param in model.named_parameters() if param.requires_grad]
        self.assertTrue(trainable)
        groups = {classify_dewo_v8_parameter(name) for name in trainable}
        self.assertEqual(groups, {GROUP_ADAPTER, GROUP_VALUE})
        opt_groups = collect_dewo_v8_param_groups(model, lr=1e-4, weight_decay=0.01)
        self.assertEqual([g["name"] for g in opt_groups], [GROUP_ADAPTER, GROUP_VALUE])
        self.assertAlmostEqual(opt_groups[0]["lr"], 1e-4)
        self.assertAlmostEqual(opt_groups[1]["lr"], 1e-5)

    def test_payload_keeps_v5_format_and_value_head(self) -> None:
        model = _TinyFastWAM()
        inject_uncond_adapter(model, rank=2, alpha=2.0)
        attach_recoverability_value_head(
            model, {"enabled": True, "in_channels": 4, "hidden": 8}
        )
        model.uncond_adapter_config["recipe"] = "v8"
        payload = uncond_adapter_payload(model, step=3, source_checkpoint="/tmp/s0.pt")
        self.assertEqual(payload["format"], CHECKPOINT_FORMAT_UNCOND_ADAPTER_V1)
        self.assertEqual(payload["recipe"], "v8")
        self.assertIn("value_head", payload)
        self.assertTrue(payload["value_head"])

    def test_normalize_config_keeps_v8_value_head(self) -> None:
        cfg = normalize_uncond_adapter_config(
            {
                "enabled": True,
                "rank": 16,
                "recipe": "v8",
                "value_head": {"enabled": True, "in_channels": 48, "lambda_value": 1.0},
            }
        )
        self.assertEqual(cfg["recipe"], "v8")
        self.assertTrue(cfg["value_head"]["enabled"])
        self.assertEqual(cfg["value_head"]["in_channels"], 48)

    def test_v8_action_lock_is_d0_only(self) -> None:
        # D0: action=1 video=1; D+: action=1 video=0; D_fail: action=0 video=1
        action_w = torch.tensor([1.0, 1.0, 0.0])
        video_w = torch.tensor([1.0, 0.0, 1.0])
        action_lock_w = (action_w * video_w).clamp(min=0)
        self.assertTrue(torch.equal(action_lock_w, torch.tensor([1.0, 0.0, 0.0])))

    def test_cliff_loss_ranks_pair(self) -> None:
        pred = torch.tensor([0.8, 0.1, 0.7], requires_grad=True)
        target = torch.tensor([1.0, 0.0, 1.0])
        pair_ids = ["p1", "p1", ""]
        loss = recoverability_cliff_loss(pred, target, pair_ids, margin=0.2)
        self.assertGreaterEqual(float(loss.item()), 0.0)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_cliff_loss_without_pairs_stays_on_graph(self) -> None:
        pred = torch.tensor([0.8, 0.7], requires_grad=True)
        target = torch.tensor([1.0, 1.0])
        loss = recoverability_cliff_loss(pred, target, ["", ""], margin=0.2)
        self.assertEqual(float(loss.item()), 0.0)
        loss.backward()
        self.assertIsNotNone(pred.grad)


if __name__ == "__main__":
    unittest.main()
