from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn

from fastwam.models.wan22.dewo_v5_train_mode import (
    GROUP_ADAPTER,
    GROUP_FROZEN,
    apply_dewo_v5_uncond_adapter_mode,
    classify_dewo_v5_parameter,
    collect_dewo_v5_param_groups,
)
from fastwam.models.wan22.fastwam import FastWAM
from fastwam.models.wan22.uncond_adapter import (
    CHECKPOINT_FORMAT_UNCOND_ADAPTER_V1,
    CFG_MIX_WEIGHT_BASE,
    LinearWithUncondAdapter,
    adaptive_cfg_mix_weight,
    inject_uncond_adapter,
    is_uncond_adapter_checkpoint,
    load_uncond_adapter_state_dict,
    resolve_backbone_and_adapter_paths,
    set_uncond_adapter_enabled,
    uncond_adapter_enabled,
    uncond_adapter_payload,
    uncond_adapter_residual_mse,
    v5_infer_remap_to_base_context,
    v5_infer_use_adapter,
    v5_infer_video_uses_base_context,
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


class DewoV5UncondAdapterTests(unittest.TestCase):
    def test_classifies_adapter_vs_frozen(self) -> None:
        self.assertEqual(
            classify_dewo_v5_parameter(
                "dit.mixtures.action.blocks.0.cross_attn.k.uncond_down"
            ),
            GROUP_ADAPTER,
        )
        self.assertEqual(
            classify_dewo_v5_parameter(
                "dit.mixtures.action.blocks.0.cross_attn.k.uncond_up"
            ),
            GROUP_ADAPTER,
        )
        self.assertEqual(
            classify_dewo_v5_parameter(
                "dit.mixtures.action.blocks.0.cross_attn.k.weight"
            ),
            GROUP_FROZEN,
        )
        self.assertEqual(classify_dewo_v5_parameter("proprio_encoder.weight"), GROUP_FROZEN)

    def test_injects_kv_on_video_and_action(self) -> None:
        model = _TinyFastWAM()
        inject_uncond_adapter(model, rank=2, alpha=2.0)
        action_attn = model.action_expert.blocks[0].cross_attn
        video_attn = model.video_expert.blocks[0].cross_attn
        self.assertIsInstance(action_attn.k, LinearWithUncondAdapter)
        self.assertIsInstance(action_attn.v, LinearWithUncondAdapter)
        self.assertIsInstance(video_attn.k, LinearWithUncondAdapter)
        self.assertIsInstance(video_attn.v, LinearWithUncondAdapter)
        self.assertNotIsInstance(action_attn.q, LinearWithUncondAdapter)
        self.assertNotIsInstance(action_attn.o, LinearWithUncondAdapter)
        self.assertNotIsInstance(video_attn.q, LinearWithUncondAdapter)

    def test_gate_off_matches_backbone_and_zero_up_is_identity(self) -> None:
        model = _TinyFastWAM()
        linear = model.action_expert.blocks[0].cross_attn.k
        x = torch.randn(2, 8)
        y_base = linear(x).detach().clone()
        inject_uncond_adapter(model, rank=2, alpha=2.0)
        wrapped = model.action_expert.blocks[0].cross_attn.k
        self.assertIsInstance(wrapped, LinearWithUncondAdapter)
        self.assertFalse(model.uncond_adapter_gate.enabled)
        self.assertTrue(torch.allclose(wrapped(x), y_base))
        set_uncond_adapter_enabled(model, True)
        self.assertTrue(torch.allclose(wrapped(x), y_base))
        wrapped.uncond_up.data.fill_(0.25)
        with uncond_adapter_enabled(model, True):
            y_on = wrapped(x)
        self.assertFalse(torch.allclose(y_on, y_base))
        with uncond_adapter_enabled(model, False):
            y_off = wrapped(x)
        self.assertTrue(torch.allclose(y_off, y_base))

    def test_train_mode_only_adapter_requires_grad(self) -> None:
        model = _TinyFastWAM()
        inject_uncond_adapter(model, rank=2, alpha=2.0)
        apply_dewo_v5_uncond_adapter_mode(model)
        trainable = [
            name for name, param in model.named_parameters() if param.requires_grad
        ]
        self.assertTrue(trainable)
        self.assertTrue(all(classify_dewo_v5_parameter(name) == GROUP_ADAPTER for name in trainable))
        self.assertFalse(model.proprio_encoder.weight.requires_grad)
        self.assertFalse(model.action_expert.blocks[0].self_attn.q.weight.requires_grad)
        self.assertFalse(model.action_expert.blocks[0].cross_attn.k.weight.requires_grad)
        self.assertTrue(model.action_expert.blocks[0].cross_attn.k.uncond_down.requires_grad)
        self.assertTrue(model.uncond_adapter_gate.enabled)
        groups = collect_dewo_v5_param_groups(model, lr=1e-4, weight_decay=0.01)
        self.assertEqual(groups[0]["name"], GROUP_ADAPTER)
        opt_ids = {id(p) for p in groups[0]["params"]}
        self.assertNotIn(id(model.action_expert.blocks[0].cross_attn.k.weight), opt_ids)
        self.assertIn(id(model.action_expert.blocks[0].cross_attn.k.uncond_down), opt_ids)

    def test_export_load_roundtrip_and_payload_format(self) -> None:
        model = _TinyFastWAM()
        inject_uncond_adapter(model, rank=2, alpha=2.0)
        model.action_expert.blocks[0].cross_attn.k.uncond_up.data.fill_(0.5)
        payload = uncond_adapter_payload(
            model, step=12, source_checkpoint="/tmp/base.pt"
        )
        self.assertEqual(payload["format"], CHECKPOINT_FORMAT_UNCOND_ADAPTER_V1)
        self.assertTrue(is_uncond_adapter_checkpoint(payload))
        self.assertGreater(payload["n_params"], 0)

        restored = _TinyFastWAM()
        inject_uncond_adapter(restored, rank=2, alpha=2.0)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "adapter.pt"
            torch.save(payload, path)
            load_uncond_adapter_state_dict(restored, str(path))
            stub = object.__new__(FastWAM)
            with self.assertRaises(ValueError) as ctx:
                FastWAM.load_checkpoint(stub, str(path))
            self.assertIn("uncond-adapter", str(ctx.exception))
            backbone, adapter = resolve_backbone_and_adapter_paths(
                str(path),
                config_resume="/tmp/base.pt",
            )
            self.assertEqual(adapter, str(path))
            self.assertEqual(backbone, "/tmp/base.pt")
        self.assertTrue(
            torch.allclose(
                restored.action_expert.blocks[0].cross_attn.k.uncond_up,
                model.action_expert.blocks[0].cross_attn.k.uncond_up,
            )
        )

    def test_v5_infer_video_pin_and_residual_lock(self) -> None:
        self.assertTrue(
            v5_infer_video_uses_base_context(
                adapter_injected=True,
                pin_video_context_to_base=True,
                has_negative_context=True,
            )
        )
        self.assertFalse(
            v5_infer_video_uses_base_context(
                adapter_injected=True,
                pin_video_context_to_base=True,
                has_negative_context=False,
            )
        )
        model = _TinyFastWAM()
        inject_uncond_adapter(model, rank=2, alpha=2.0)
        apply_dewo_v5_uncond_adapter_mode(model)
        linear = model.video_expert.blocks[0].cross_attn.k
        linear.uncond_up.data.fill_(0.5)
        linear.train()
        _ = linear(torch.randn(2, 8))
        mse = uncond_adapter_residual_mse(model, expert="video")
        self.assertIsNotNone(mse)
        self.assertGreater(float(mse.detach()), 0.0)
        action_mse = uncond_adapter_residual_mse(model, expert="action")
        self.assertIsNone(action_mse)

    def test_v5_infer_pairs_adapter_with_success_not_base(self) -> None:
        self.assertFalse(v5_infer_use_adapter(branch="base", use_text_cfg=True))
        self.assertFalse(v5_infer_use_adapter(branch="base", use_text_cfg=False))
        self.assertTrue(v5_infer_use_adapter(branch="posi", use_text_cfg=True))
        self.assertFalse(v5_infer_use_adapter(branch="posi", use_text_cfg=False))
        self.assertTrue(
            v5_infer_remap_to_base_context(
                adapter_injected=True,
                use_text_cfg=False,
                has_negative_context=True,
            )
        )
        self.assertFalse(
            v5_infer_remap_to_base_context(
                adapter_injected=True,
                use_text_cfg=True,
                has_negative_context=True,
            )
        )
        self.assertFalse(
            v5_infer_remap_to_base_context(
                adapter_injected=False,
                use_text_cfg=False,
                has_negative_context=True,
            )
        )

    def test_adaptive_mix_high_energy_keeps_guided_scale(self) -> None:
        self.assertEqual(
            adaptive_cfg_mix_weight(exec_rms=0.051, tau=0.05, guided_scale=2.0),
            2.0,
        )

    def test_adaptive_mix_low_energy_is_base_not_mix_one(self) -> None:
        weight = adaptive_cfg_mix_weight(exec_rms=0.035, tau=0.05, guided_scale=2.0)
        self.assertEqual(weight, CFG_MIX_WEIGHT_BASE)
        self.assertEqual(weight, 0.0)
        self.assertNotEqual(weight, 1.0)

    def test_adaptive_mix_rejects_scale_one_as_guided(self) -> None:
        with self.assertRaises(ValueError):
            adaptive_cfg_mix_weight(exec_rms=0.1, tau=0.05, guided_scale=1.0)

    def test_resolve_explicit_adapter_skips_peek(self) -> None:
        backbone, adapter = resolve_backbone_and_adapter_paths(
            "/tmp/base.pt",
            adapter_checkpoint="/tmp/adapter.pt",
        )
        self.assertEqual(backbone, "/tmp/base.pt")
        self.assertEqual(adapter, "/tmp/adapter.pt")


if __name__ == "__main__":
    unittest.main()
