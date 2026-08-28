from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import torch

from fastwam.models.wan22.dewo_v9_train_mode import (
    GROUP_ADAPTER,
    GROUP_VALUE,
    UNCOND_ADAPTER_TRAIN_MODES,
    apply_dewo_v9_uncond_adapter_mode,
    classify_dewo_v9_parameter,
    collect_dewo_v9_param_groups,
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
    progress_return,
    relative_growth,
    relative_growth_gate,
    low_value_growth_gate,
)
from fastwam.datasets.eve.manifest_dataset import EveManifestRobotVideoDataset

_GEO = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "dewo_v2"
    / "v9_pair_geometry.py"
)
_spec = importlib.util.spec_from_file_location("v9_pair_geometry", _GEO)
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
sys.modules.setdefault("v9_pair_geometry", _mod)
_spec.loader.exec_module(_mod)
fail_cliff_span = _mod.fail_cliff_span
stitch_prefix_plus_continuation = _mod.stitch_prefix_plus_continuation


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
        self.hidden_dim = dim
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


class DewoV9UncondAdapterTests(unittest.TestCase):
    def test_v9_is_an_uncond_adapter_train_mode(self) -> None:
        self.assertIn("dewo_v9_uncond_adapter", UNCOND_ADAPTER_TRAIN_MODES)

    def test_mix_subtracts_base_not_fail(self) -> None:
        self.assertEqual(cfg_mix_subtract_branch("v9"), CFG_MIX_SUBTRACT_BASE)

    def test_progress_return_rises_and_fail_is_zero(self) -> None:
        early = progress_return(0, 250, gamma=0.99, failed=False)
        late = progress_return(249, 250, gamma=0.99, failed=False)
        mid = progress_return(72, 250, gamma=0.99, failed=False)
        self.assertLess(early, mid)
        self.assertLess(mid, late)
        self.assertAlmostEqual(late, 1.0)
        self.assertEqual(progress_return(72, 250, gamma=0.99, failed=True), 0.0)
        # No event floor: mid-episode G is the discounted remainder, not ≥0.5.
        self.assertLess(mid, 0.5)

    def test_drop_edge_fires_on_drop_without_v_high(self) -> None:
        self.assertEqual(drop_edge_gate(0.30, 0.10, v_high=None, delta=0.15), 1.0)
        self.assertEqual(drop_edge_gate(0.30, 0.29, v_high=None, delta=0.15), 0.0)
        self.assertEqual(drop_edge_gate(None, 0.10, v_high=None, delta=0.15), 0.0)
        self.assertEqual(
            drop_edge_gate(0.30, 0.10, v_high=None, delta=0.15, fired=True), 0.0
        )

    def test_optional_v_high_floor(self) -> None:
        self.assertEqual(drop_edge_gate(0.30, 0.10, v_high=0.5, delta=0.15), 0.0)
        self.assertEqual(drop_edge_gate(0.90, 0.40, v_high=0.5, delta=0.15), 1.0)

    def test_relative_growth_gate_skips_early_and_fires_on_stall(self) -> None:
        self.assertIsNone(relative_growth(None, 0.1))
        self.assertAlmostEqual(relative_growth(0.10, 0.104), 0.04)
        # Before start_replan: never fire.
        self.assertEqual(
            relative_growth_gate(0.10, 0.10, tau=0.05, replan_index=1, start_replan=2),
            0.0,
        )
        # Stall after start: fire.
        self.assertEqual(
            relative_growth_gate(0.1274, 0.1274, tau=0.05, replan_index=4, start_replan=2),
            1.0,
        )
        # Healthy rise: no fire.
        self.assertEqual(
            relative_growth_gate(0.14, 0.22, tau=0.05, replan_index=5, start_replan=2),
            0.0,
        )
        # Success-path dip may fire (allowed).
        self.assertEqual(
            relative_growth_gate(0.22, 0.215, tau=0.05, replan_index=6, start_replan=2),
            1.0,
        )
        # Optional once-fire and stop_replan cut later stalls.
        self.assertEqual(
            relative_growth_gate(
                0.10, 0.10, tau=0.05, replan_index=2, start_replan=2, fired=True
            ),
            0.0,
        )
        self.assertEqual(
            relative_growth_gate(
                0.10,
                0.10,
                tau=0.05,
                replan_index=4,
                start_replan=2,
                stop_replan=3,
            ),
            0.0,
        )
        self.assertEqual(
            relative_growth_gate(
                0.10,
                0.10,
                tau=0.05,
                replan_index=3,
                start_replan=2,
                stop_replan=3,
            ),
            1.0,
        )

    def test_value_head_pools_tokens(self) -> None:
        head = RecoverabilityValueHead(in_channels=8, hidden=8)
        tokens = torch.zeros(2, 4, 8)
        tokens[:, 0] = 1.0
        out = head(tokens)
        self.assertEqual(tuple(out.shape), (2,))
        self.assertTrue(torch.allclose(out, torch.full_like(out, 0.5), atol=1e-5))

    def test_value_lr_matches_adapter(self) -> None:
        model = _TinyFastWAM()
        inject_uncond_adapter(model, rank=2, alpha=2.0)
        attach_recoverability_value_head(
            model,
            {
                "enabled": True,
                "encoder": "video_dit",
                "in_channels": 8,
                "hidden": 8,
                "v_high": None,
                "lambda_cliff": 0.0,
            },
            recipe="v9",
        )
        apply_dewo_v9_uncond_adapter_mode(model)
        trainable = [
            name for name, param in model.named_parameters() if param.requires_grad
        ]
        groups = {classify_dewo_v9_parameter(name) for name in trainable}
        self.assertEqual(groups, {GROUP_ADAPTER, GROUP_VALUE})
        opt_groups = collect_dewo_v9_param_groups(
            model, lr=1e-4, weight_decay=0.01, value_lr_scale=1.0
        )
        self.assertAlmostEqual(opt_groups[0]["lr"], 1e-4)
        self.assertAlmostEqual(opt_groups[1]["lr"], 1e-4)

    def test_payload_keeps_v5_format_and_recipe_v9(self) -> None:
        model = _TinyFastWAM()
        inject_uncond_adapter(model, rank=2, alpha=2.0)
        attach_recoverability_value_head(
            model,
            {"enabled": True, "encoder": "video_dit", "in_channels": 8, "hidden": 8},
            recipe="v9",
        )
        model.uncond_adapter_config["recipe"] = "v9"
        payload = uncond_adapter_payload(model, step=3, source_checkpoint="/tmp/s0.pt")
        self.assertEqual(payload["format"], CHECKPOINT_FORMAT_UNCOND_ADAPTER_V1)
        self.assertEqual(payload["recipe"], "v9")
        self.assertIn("value_head", payload)

    def test_normalize_config_v9_defaults(self) -> None:
        cfg = normalize_uncond_adapter_config(
            {"enabled": True, "rank": 16, "recipe": "v9"}
        )
        self.assertEqual(cfg["recipe"], "v9")
        self.assertTrue(cfg["value_head"]["enabled"])
        self.assertEqual(cfg["value_head"]["encoder"], "video_dit")
        self.assertEqual(cfg["value_head"]["in_channels"], 3072)
        self.assertEqual(cfg["value_head"]["loss"], "huber")
        self.assertIsNone(cfg["value_head"]["v_high"])
        self.assertEqual(cfg["value_head"]["lambda_cliff"], 0.0)
        self.assertAlmostEqual(cfg["value_head"]["gamma"], 0.99)

    def test_v9_pool_progress_target(self) -> None:
        dataset = EveManifestRobotVideoDataset.__new__(EveManifestRobotVideoDataset)
        dataset.unit_filter = "dewo_v9_pool"
        dataset.value_gamma = 0.99
        episode = {
            "sample_type": "episode",
            "episode_outcome": "success",
            "batch_role": "primary",
            "action_loss": "enabled",
            "end_frame": 250,
        }
        failure = {
            "sample_type": "event",
            "episode_outcome": "failure",
            "event_outcome": "failure",
            "batch_role": "auxiliary",
            "action_loss": "disabled",
            "end_frame": 48,
        }
        self.assertTrue(dataset._passes_unit_filter(episode))
        self.assertTrue(dataset._passes_unit_filter(failure))
        g0 = dataset._v9_value_target(episode, 0, gamma=0.99)
        g_end = dataset._v9_value_target(episode, 249, gamma=0.99)
        self.assertLess(g0, g_end)
        self.assertAlmostEqual(g_end, 1.0)
        self.assertEqual(dataset._v9_value_target(failure, 0, gamma=0.99), 0.0)

    def test_fail_cliff_does_not_eat_shared_prefix(self) -> None:
        lo, hi = fail_cliff_span(72, 96, 198, min_len=33, post=24)
        self.assertEqual(lo, 72)
        self.assertEqual(hi, 120)
        lo2, hi2 = fail_cliff_span(90, 96, 100, min_len=33, post=24)
        self.assertEqual(lo2, 67)
        self.assertEqual(hi2, 100)

    def test_stitch_prefix_plus_continuation(self) -> None:
        prefix = torch.arange(3).numpy()
        cont = torch.arange(3, 7).numpy()
        stitched = stitch_prefix_plus_continuation(prefix, cont)
        self.assertEqual(list(stitched), [0, 1, 2, 3, 4, 5, 6])


if __name__ == "__main__":
    unittest.main()
