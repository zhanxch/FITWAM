from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fastwam.models.wan22.fastwam import FastWAM  # noqa: E402


class DummyVideoExpert(nn.Module):
    hidden_dim = 6
    video_attention_mask_mode = "first_frame_causal"
    fuse_vae_embedding_in_latents = True

    def pre_dit(
        self,
        *,
        x,
        timestep,
        context,
        context_mask,
        action,
        fuse_vae_embedding_in_latents,
    ):
        del timestep, action, fuse_vae_embedding_in_latents
        batch_size = x.shape[0]
        tokens = torch.arange(
            batch_size * 2 * self.hidden_dim,
            device=x.device,
            dtype=x.dtype,
        ).reshape(batch_size, 2, self.hidden_dim)
        return {
            "tokens": tokens,
            "freqs": torch.zeros(2, 1, 2, device=x.device, dtype=x.dtype),
            "t_mod": torch.zeros(
                batch_size,
                2,
                6,
                self.hidden_dim,
                device=x.device,
                dtype=x.dtype,
            ),
            "context": context,
            "context_mask": context_mask.unsqueeze(1).expand(-1, 2, -1),
            "meta": {"tokens_per_frame": 1},
        }

    @staticmethod
    def build_video_to_video_mask(
        *,
        video_seq_len,
        video_tokens_per_frame,
        device,
    ):
        del video_tokens_per_frame
        return torch.ones(
            video_seq_len,
            video_seq_len,
            dtype=torch.bool,
            device=device,
        )

    @staticmethod
    def post_dit(tokens, pre_state):
        del pre_state
        return tokens


class DummyActionExpert(nn.Module):
    hidden_dim = 4
    action_dim = 2

    def pre_dit(
        self,
        *,
        action_tokens,
        timestep,
        context,
        context_mask,
    ):
        del timestep
        padding = torch.zeros(
            *action_tokens.shape[:-1],
            self.hidden_dim - self.action_dim,
            device=action_tokens.device,
            dtype=action_tokens.dtype,
        )
        tokens = torch.cat([action_tokens, padding], dim=-1)
        return {
            "tokens": tokens,
            "freqs": torch.zeros(
                tokens.shape[1],
                1,
                2,
                device=tokens.device,
                dtype=tokens.dtype,
            ),
            "t_mod": torch.zeros(
                tokens.shape[0],
                6,
                self.hidden_dim,
                device=tokens.device,
                dtype=tokens.dtype,
            ),
            "context": context,
            "context_mask": context_mask.unsqueeze(1).expand(
                -1,
                tokens.shape[1],
                -1,
            ),
        }

    @staticmethod
    def post_dit(tokens, pre_state):
        del pre_state
        return tokens[..., :2]


class DummyMoT(nn.Module):
    @staticmethod
    def forward(
        *,
        embeds_all,
        attention_mask,
        freqs_all,
        context_all,
        t_mod_all,
    ):
        del attention_mask, freqs_all, context_all, t_mod_all
        return embeds_all

    @staticmethod
    def prefill_video_cache(**kwargs):
        del kwargs
        return []

    @staticmethod
    def forward_action_with_video_cache(*, action_tokens, **kwargs):
        del kwargs
        return action_tokens


class DummyVAE(nn.Module):
    temporal_downsample_factor = 4
    upsampling_factor = 16

    def __init__(self):
        super().__init__()
        self.model = SimpleNamespace(z_dim=2)

    @staticmethod
    def encode(value, **kwargs):
        del kwargs
        if isinstance(value, list):
            source = value[0]
            return [
                torch.zeros(
                    2,
                    1,
                    1,
                    1,
                    device=source.device,
                    dtype=source.dtype,
                )
            ]
        return torch.zeros(
            value.shape[0],
            2,
            2,
            1,
            1,
            device=value.device,
            dtype=value.dtype,
        )


class CountingStudent(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.calls = 0
        self.last_video_tokens = None
        self.last_context_tokens = None

    def forward(
        self,
        video_tokens,
        context_tokens,
        *,
        video_mask,
        context_mask,
    ):
        del video_mask, context_mask
        self.calls += 1
        self.last_video_tokens = video_tokens.detach().clone()
        self.last_context_tokens = context_tokens.detach().clone()
        output = torch.ones(
            video_tokens.shape[0],
            self.embedding_dim,
            device=video_tokens.device,
            dtype=video_tokens.dtype,
        )
        return torch.nn.functional.normalize(output, dim=-1)


def make_model(
    *,
    steer_enabled: bool,
    proprio_dim: int | None = None,
    outcome_num_classes: int = 0,
    pair_loss_weight: float = 0.0,
    detach_backbone_inputs: bool = True,
) -> FastWAM:
    return FastWAM(
        video_expert=DummyVideoExpert(),
        action_expert=DummyActionExpert(),
        mot=DummyMoT(),
        vae=DummyVAE(),
        text_dim=5,
        proprio_dim=proprio_dim,
        outcome_num_classes=outcome_num_classes,
        device="cpu",
        torch_dtype=torch.float32,
        offline_steer_config={
            "enabled": steer_enabled,
            "hidden_dim": 6,
            "embedding_dim": 3,
            "num_heads": 2,
            "dropout": 0.0,
            "pair_loss_weight": pair_loss_weight,
            "pair_loss_margin": 0.2,
            "pair_loss_warmup_steps": 500,
            "detach_backbone_inputs": detach_backbone_inputs,
        },
    )


class FastWAMOfflineSteerIntegrationTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(17)

    def test_disabled_path_keeps_action_token_object_unchanged(self) -> None:
        model = make_model(steer_enabled=False)
        tokens = torch.randn(2, 3, 4)
        action_pre = {"tokens": tokens}

        model._inject_offline_steer(action_pre, steer_embedding=None)

        self.assertIs(action_pre["tokens"], tokens)
        self.assertIsNone(model.offline_steer_student)
        self.assertIsNone(model.offline_steer_residual)

    def test_student_uses_only_current_frame_and_zero_init_is_exact_noop(self) -> None:
        model = make_model(steer_enabled=True)
        student = CountingStudent(embedding_dim=3)
        model.offline_steer_student = student
        video_tokens = torch.randn(2, 6, 6)
        context = torch.randn(2, 3, 5)
        context_mask = torch.ones(2, 3, dtype=torch.bool)
        action_tokens = torch.randn(2, 4, 4)
        original = action_tokens.clone()

        embedding = model._compute_offline_steer_embedding(
            video_tokens=video_tokens,
            video_tokens_per_frame=2,
            context=context,
            context_mask=context_mask,
        )
        action_pre = {"tokens": action_tokens}
        model._inject_offline_steer(
            action_pre,
            steer_embedding=embedding,
        )

        self.assertEqual(student.calls, 1)
        self.assertTrue(torch.equal(student.last_video_tokens, video_tokens[:, :2]))
        self.assertTrue(torch.equal(action_pre["tokens"], original))

    def test_student_backbone_inputs_are_detached_by_default(self) -> None:
        model = make_model(steer_enabled=True)
        video_tokens = torch.randn(2, 4, 6, requires_grad=True)
        context = torch.randn(2, 3, 5, requires_grad=True)
        context_mask = torch.ones(2, 3, dtype=torch.bool)

        embedding = model._compute_offline_steer_embedding(
            video_tokens=video_tokens,
            video_tokens_per_frame=2,
            context=context,
            context_mask=context_mask,
        )
        embedding[:, 0].sum().backward()

        self.assertIsNone(video_tokens.grad)
        self.assertIsNone(context.grad)
        self.assertTrue(
            any(
                parameter.grad is not None
                for parameter in model.offline_steer_student.parameters()
            )
        )

    def test_pair_loss_uses_only_positive_pairs_and_detaches_targets(self) -> None:
        model = make_model(steer_enabled=True, pair_loss_weight=0.1)
        student = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            requires_grad=True,
        )
        success = torch.tensor(
            [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            requires_grad=True,
        )
        failure = torch.tensor(
            [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            requires_grad=True,
        )

        loss, weight_sum, enabled_frac = model._compute_offline_pair_loss(
            steer_embedding=student,
            sample={
                "pair_weight": torch.tensor([0.75, 0.0]),
                "steer_success_target": success,
                "steer_failure_target": failure,
            },
        )
        self.assertIsNotNone(loss)
        self.assertAlmostEqual(weight_sum, 0.75)
        self.assertAlmostEqual(enabled_frac, 0.5)
        loss.backward()

        self.assertIsNotNone(student.grad)
        self.assertIsNone(success.grad)
        self.assertIsNone(failure.grad)

    def test_all_zero_pair_weights_produce_connected_zero_loss(self) -> None:
        model = make_model(steer_enabled=True, pair_loss_weight=0.1)
        student = torch.randn(2, 3, requires_grad=True)

        loss, weight_sum, enabled_frac = model._compute_offline_pair_loss(
            steer_embedding=student,
            sample={"pair_weight": torch.zeros(2)},
        )
        self.assertEqual(float(loss.detach().item()), 0.0)
        self.assertEqual(weight_sum, 0.0)
        self.assertEqual(enabled_frac, 0.0)
        loss.backward()
        self.assertTrue(torch.equal(student.grad, torch.zeros_like(student)))

    def test_positive_pair_weight_requires_targets(self) -> None:
        model = make_model(steer_enabled=True, pair_loss_weight=0.1)
        with self.assertRaisesRegex(ValueError, "steer_success_target"):
            model._compute_offline_pair_loss(
                steer_embedding=torch.randn(2, 3),
                sample={"pair_weight": torch.tensor([1.0, 0.0])},
            )

    def test_zero_scale_validation_may_omit_pair_fields(self) -> None:
        model = make_model(steer_enabled=True, pair_loss_weight=0.1)

        loss, weight_sum, enabled_frac = model._compute_offline_pair_loss(
            steer_embedding=torch.randn(2, 3),
            sample={},
            require_pair_fields=False,
        )

        self.assertIsNone(loss)
        self.assertEqual(weight_sum, 0.0)
        self.assertEqual(enabled_frac, 0.0)

        with self.assertRaisesRegex(ValueError, "pair_weight"):
            model._compute_offline_pair_loss(
                steer_embedding=torch.randn(2, 3),
                sample={},
                require_pair_fields=True,
            )

    def test_student_context_keeps_proprio_but_excludes_outcome_token(self) -> None:
        model = make_model(
            steer_enabled=True,
            proprio_dim=2,
            outcome_num_classes=2,
        )
        sample = {
            "video": torch.zeros(1, 3, 5, 16, 16),
            "action": torch.zeros(1, 4, 2),
            "context": torch.randn(1, 2, 5),
            "context_mask": torch.ones(1, 2, dtype=torch.bool),
            "proprio": torch.randn(1, 5, 2),
            "outcome_flag": torch.ones(1, dtype=torch.long),
        }

        inputs = model.build_inputs(sample)

        self.assertEqual(inputs["context"].shape[1], 4)
        self.assertEqual(inputs["steer_context"].shape[1], 3)
        self.assertTrue(
            torch.equal(
                inputs["steer_context"][:, :2],
                sample["context"],
            )
        )

    def test_joint_action_and_cache_paths_apply_the_same_residual(self) -> None:
        model = make_model(steer_enabled=True)
        with torch.no_grad():
            model.offline_steer_residual.projection.bias.fill_(1.0)
        first_frame = torch.zeros(1, 2, 1, 1, 1)
        action = torch.zeros(1, 2, 2)
        timestep = torch.zeros(1)
        context = torch.zeros(1, 2, 5)
        context_mask = torch.ones(1, 2, dtype=torch.bool)
        steer = torch.ones(1, 3)

        _, joint_action = model._predict_joint_noise(
            latents_video=first_frame,
            latents_action=action,
            timestep_video=timestep,
            timestep_action=timestep,
            context=context,
            context_mask=context_mask,
            fuse_vae_embedding_in_latents=True,
            steer_embedding=steer,
        )
        action_only = model._predict_action_noise(
            first_frame_latents=first_frame,
            latents_action=action,
            timestep_action=timestep,
            context=context,
            context_mask=context_mask,
            fuse_vae_embedding_in_latents=True,
            steer_embedding=steer,
        )
        cached = model._predict_action_noise_with_cache(
            latents_action=action,
            timestep_action=timestep,
            context=context,
            context_mask=context_mask,
            video_kv_cache=[],
            attention_mask=torch.ones(4, 4, dtype=torch.bool),
            video_seq_len=2,
            steer_embedding=steer,
        )

        expected = torch.ones_like(action)
        self.assertTrue(torch.equal(joint_action, expected))
        self.assertTrue(torch.equal(action_only, expected))
        self.assertTrue(torch.equal(cached, expected))

    def test_infer_action_computes_student_once_for_all_denoising_steps(self) -> None:
        model = make_model(steer_enabled=True)
        student = CountingStudent(embedding_dim=3)
        model.offline_steer_student = student

        output = model.infer_action(
            prompt=None,
            input_image=torch.zeros(3, 16, 16),
            action_horizon=2,
            context=torch.zeros(2, 5),
            context_mask=torch.ones(2, dtype=torch.bool),
            num_inference_steps=3,
            seed=5,
        )

        self.assertEqual(tuple(output["action"].shape), (2, 2))
        self.assertEqual(student.calls, 1)

    def test_checkpoint_roundtrip_restores_steer_modules(self) -> None:
        source = make_model(steer_enabled=True)
        with torch.no_grad():
            source.offline_steer_residual.projection.bias.fill_(0.75)
        target = make_model(steer_enabled=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "steer.pt"
            source.save_checkpoint(checkpoint, step=12)
            target.load_checkpoint(checkpoint)
            torch.testing.assert_close(
                target.offline_steer_residual.projection.bias,
                torch.full((4,), 0.75),
            )

    def test_steer_checkpoint_is_rejected_when_target_disables_steer(self) -> None:
        source = make_model(steer_enabled=True, proprio_dim=2)
        target = make_model(steer_enabled=False, proprio_dim=2)
        with torch.no_grad():
            for parameter in source.proprio_encoder.parameters():
                parameter.fill_(0.75)
            for parameter in target.proprio_encoder.parameters():
                parameter.zero_()
        target_before = {
            key: value.clone()
            for key, value in target.proprio_encoder.state_dict().items()
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "steer.pt"
            source.save_checkpoint(checkpoint)

            with self.assertRaisesRegex(
                ValueError,
                r"contains `offline_steer` weights.*enabled=false",
            ):
                target.load_checkpoint(checkpoint)

        for key, value in target.proprio_encoder.state_dict().items():
            torch.testing.assert_close(value, target_before[key])

    def test_incomplete_steer_checkpoint_is_rejected(self) -> None:
        source = make_model(steer_enabled=True)
        target = make_model(steer_enabled=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "incomplete-steer.pt"
            payload = {
                "mot": source.mot.state_dict(),
                "offline_steer_student": source.offline_steer_student.state_dict(),
            }
            torch.save(payload, checkpoint)

            with self.assertRaisesRegex(
                ValueError,
                r"incomplete `offline_steer` state",
            ):
                target.load_checkpoint(checkpoint)

    def test_vanilla_checkpoint_loads_into_enabled_steer_model(self) -> None:
        vanilla = make_model(steer_enabled=False)
        target = make_model(steer_enabled=True)
        student_before = {
            key: value.clone()
            for key, value in target.offline_steer_student.state_dict().items()
        }
        residual_before = {
            key: value.clone()
            for key, value in target.offline_steer_residual.state_dict().items()
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "vanilla.pt"
            vanilla.save_checkpoint(checkpoint)
            target.load_checkpoint(checkpoint)

        for key, value in target.offline_steer_student.state_dict().items():
            torch.testing.assert_close(value, student_before[key])
        for key, value in target.offline_steer_residual.state_dict().items():
            torch.testing.assert_close(value, residual_before[key])


if __name__ == "__main__":
    unittest.main()
