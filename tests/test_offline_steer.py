from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fastwam.models.wan22.offline_steer import (  # noqa: E402
    ObservationSteerStudent,
    TrajectoryTeacher,
    ZeroInitSteerResidual,
    weighted_pair_loss,
)


class OfflineSteerTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)

    def test_zero_initialized_residual_is_an_exact_no_op(self) -> None:
        residual = ZeroInitSteerResidual(
            embedding_dim=8,
            action_hidden_dim=16,
        )
        action_tokens = torch.randn(3, 5, 16)
        steer_embedding = torch.randn(3, 8)

        actual = residual.add_to_action_tokens(action_tokens, steer_embedding)

        self.assertTrue(torch.equal(residual(steer_embedding), torch.zeros(3, 16)))
        self.assertTrue(torch.equal(actual, action_tokens))

    def test_student_masks_exclude_video_and_context_padding(self) -> None:
        student = ObservationSteerStudent(
            video_dim=6,
            context_dim=5,
            hidden_dim=12,
            embedding_dim=8,
            num_heads=3,
        ).eval()
        video = torch.randn(2, 4, 6)
        context = torch.randn(2, 3, 5)
        video_mask = torch.tensor(
            [[True, True, False, False], [True, False, False, False]]
        )
        context_mask = torch.tensor(
            [[True, False, False], [True, True, False]]
        )
        changed_video = video.clone()
        changed_context = context.clone()
        changed_video[~video_mask] = torch.randn_like(changed_video[~video_mask]) * 100
        changed_context[~context_mask] = (
            torch.randn_like(changed_context[~context_mask]) * 100
        )

        with torch.no_grad():
            expected = student(
                video,
                context,
                video_mask=video_mask,
                context_mask=context_mask,
            )
            actual = student(
                changed_video,
                changed_context,
                video_mask=video_mask,
                context_mask=context_mask,
            )

        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(
            actual.norm(dim=-1),
            torch.ones(2),
            rtol=1e-6,
            atol=1e-6,
        )

        with self.assertRaisesRegex(ValueError, "at least one valid"):
            student(
                video,
                context,
                video_mask=torch.zeros_like(video_mask),
                context_mask=torch.zeros_like(context_mask),
            )

    def test_weighted_pair_loss_only_backpropagates_through_student(self) -> None:
        student = ObservationSteerStudent(
            video_dim=6,
            context_dim=5,
            hidden_dim=12,
            embedding_dim=8,
            num_heads=3,
        )
        video = torch.randn(3, 4, 6)
        context = torch.randn(3, 2, 5)
        success_target = torch.randn(3, 8, requires_grad=True)
        failure_target = torch.randn(3, 8, requires_grad=True)
        sample_weight = torch.tensor(
            [1.0, 0.25, 0.0],
            requires_grad=True,
        )

        student_embedding = student(video, context)
        loss = weighted_pair_loss(
            student_embedding,
            success_target,
            failure_target,
            sample_weight,
            margin=0.2,
        )
        loss.backward()

        student_gradient = sum(
            parameter.grad.abs().sum().item()
            for parameter in student.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(student_gradient, 0.0)
        self.assertIsNone(success_target.grad)
        self.assertIsNone(failure_target.grad)
        self.assertIsNone(sample_weight.grad)

    def test_teacher_is_invariant_to_masked_padding(self) -> None:
        teacher = TrajectoryTeacher(
            action_dim=7,
            hidden_dim=16,
            embedding_dim=8,
            num_heads=4,
            num_layers=2,
        ).eval()
        actions = torch.randn(2, 3, 7)
        padded_actions = torch.cat(
            [actions, torch.randn(2, 4, 7) * 100],
            dim=1,
        )
        valid_mask = torch.ones(2, 3, dtype=torch.bool)
        padded_mask = torch.cat(
            [
                valid_mask,
                torch.zeros(2, 4, dtype=torch.bool),
            ],
            dim=1,
        )

        with torch.no_grad():
            expected = teacher(actions, valid_mask)
            actual = teacher(padded_actions, padded_mask)

        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(
            actual.norm(dim=-1),
            torch.ones(2),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_pair_loss_validates_weight_shape_and_finiteness(self) -> None:
        student = torch.randn(2, 4)
        success = torch.randn(2, 4)
        failure = torch.randn(2, 4)

        with self.assertRaisesRegex(ValueError, "shape"):
            weighted_pair_loss(
                student,
                success,
                failure,
                torch.ones(2, 1),
            )
        with self.assertRaisesRegex(ValueError, "finite"):
            weighted_pair_loss(
                student,
                success,
                failure,
                torch.tensor([1.0, float("nan")]),
            )
        with self.assertRaisesRegex(ValueError, "positive"):
            weighted_pair_loss(
                student,
                success,
                failure,
                torch.zeros(2),
            )


if __name__ == "__main__":
    unittest.main()
