from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from policy_io import (  # noqa: E402
    KEY_FAILURE_CONTEXT,
    KEY_FAILURE_CONTEXT_MASK,
    KEY_NEGATIVE_CONTEXT,
    KEY_NEGATIVE_CONTEXT_MASK,
    to_inference_tensors,
    validate_policy_observation,
)


class PolicyIoCfgTest(unittest.TestCase):
    def test_cached_success_and_base_contexts_are_converted(self) -> None:
        observation = {
            "input_image": np.zeros((3, 16, 16), dtype=np.float32),
            "context": np.ones((2, 5), dtype=np.float32),
            "context_mask": np.ones(2, dtype=bool),
            KEY_NEGATIVE_CONTEXT: np.zeros((2, 5), dtype=np.float32),
            KEY_NEGATIVE_CONTEXT_MASK: np.ones(2, dtype=bool),
        }

        tensors = to_inference_tensors(
            observation,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

        self.assertEqual(tuple(tensors[KEY_NEGATIVE_CONTEXT].shape), (2, 5))
        self.assertEqual(tensors[KEY_NEGATIVE_CONTEXT_MASK].dtype, torch.bool)

    def test_partial_or_mixed_negative_conditioning_is_rejected(self) -> None:
        base = {
            "input_image": np.zeros((3, 16, 16), dtype=np.float32),
            "context": np.ones((2, 5), dtype=np.float32),
            "context_mask": np.ones(2, dtype=bool),
        }
        with self.assertRaisesRegex(ValueError, "both 'negative_context'"):
            validate_policy_observation(
                {**base, KEY_NEGATIVE_CONTEXT: np.zeros((2, 5), dtype=np.float32)}
            )
        with self.assertRaisesRegex(ValueError, "Cached context input"):
            validate_policy_observation({**base, "negative_prompt": "base"})

    def test_partial_or_mixed_failure_conditioning_is_rejected(self) -> None:
        base = {
            "input_image": np.zeros((3, 16, 16), dtype=np.float32),
            "context": np.ones((2, 5), dtype=np.float32),
            "context_mask": np.ones(2, dtype=bool),
        }
        with self.assertRaisesRegex(ValueError, "both 'failure_context'"):
            validate_policy_observation(
                {**base, KEY_FAILURE_CONTEXT: np.zeros((2, 5), dtype=np.float32)}
            )
        with self.assertRaisesRegex(ValueError, "Cached context input"):
            validate_policy_observation({**base, "failure_prompt": "fail"})

    def test_cached_failure_context_is_converted(self) -> None:
        observation = {
            "input_image": np.zeros((3, 16, 16), dtype=np.float32),
            "context": np.ones((2, 5), dtype=np.float32),
            "context_mask": np.ones(2, dtype=bool),
            KEY_NEGATIVE_CONTEXT: np.zeros((2, 5), dtype=np.float32),
            KEY_NEGATIVE_CONTEXT_MASK: np.ones(2, dtype=bool),
            KEY_FAILURE_CONTEXT: np.full((2, 5), 2.0, dtype=np.float32),
            KEY_FAILURE_CONTEXT_MASK: np.ones(2, dtype=bool),
        }
        tensors = to_inference_tensors(
            observation,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        self.assertEqual(tuple(tensors[KEY_FAILURE_CONTEXT].shape), (2, 5))
        self.assertTrue(torch.all(tensors[KEY_FAILURE_CONTEXT] == 2.0))


if __name__ == "__main__":
    unittest.main()
