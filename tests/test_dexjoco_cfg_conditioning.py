from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "dexjoco_async" / "dexjoco_fastwam_adapter.py"
spec = importlib.util.spec_from_file_location("dexjoco_cfg_adapter_under_test", MODULE_PATH)
adapter_module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = adapter_module
spec.loader.exec_module(adapter_module)


class DexJoCoCfgConditioningTest(unittest.TestCase):
    @staticmethod
    def _settings(*, use_prompt: bool) -> dict:
        return {
            "image_size_wh": (16, 16),
            "action_horizon": 2,
            "action_output_dim": 2,
            "policy_action_prefix_dim": 0,
            "policy_action_control_slice": [0, 2],
            "proprio_output_dim": 2,
            "text_embedding_cache_dir": "/tmp/cfg-cache",
            "context_len": 4,
            "load_text_encoder": use_prompt,
            "concat_multi_camera": None,
            "image_keys": ["front"],
            "image_sizes_wh": [(16, 16)],
        }

    @staticmethod
    def _env_obs() -> dict:
        return {
            "front": np.zeros((16, 16, 3), dtype=np.uint8),
            "state": np.zeros(2, dtype=np.float32),
        }

    def test_prompt_mode_sends_explicit_success_and_base_prompts(self) -> None:
        adapter = adapter_module.DexJoCoFastWAMAdapter(self._settings(use_prompt=True))
        observation = adapter.env_obs_to_policy_obs(
            self._env_obs(),
            camera_key="front",
            camera_mapping={"base": "front"},
            task_prompt="Task. Successful execution.",
            cfg_base_prompt="Task.",
        )

        self.assertTrue(observation["prompt"].endswith("Task. Successful execution."))
        self.assertTrue(observation["negative_prompt"].endswith("Task."))

    def test_cached_mode_loads_both_success_and_base_contexts(self) -> None:
        adapter = adapter_module.DexJoCoFastWAMAdapter(self._settings(use_prompt=False))

        def fake_load(instruction, **_kwargs):
            fill = 1.0 if instruction.endswith("Task. Successful execution.") else 0.0
            return (
                np.full((4, 3), fill, dtype=np.float32),
                np.ones(4, dtype=bool),
            )

        with mock.patch.object(
            adapter_module,
            "load_text_context_arrays",
            side_effect=fake_load,
        ) as loader:
            observation = adapter.env_obs_to_policy_obs(
                self._env_obs(),
                camera_key="front",
                camera_mapping={"base": "front"},
                task_prompt="Task. Successful execution.",
                cfg_base_prompt="Task.",
            )

        self.assertEqual(loader.call_count, 2)
        self.assertTrue(np.all(observation["context"] == 1.0))
        self.assertTrue(np.all(observation["negative_context"] == 0.0))

    def test_cached_mode_loads_failure_context(self) -> None:
        adapter = adapter_module.DexJoCoFastWAMAdapter(self._settings(use_prompt=False))

        def fake_load(instruction, **_kwargs):
            if instruction.endswith("Failed execution."):
                fill = 2.0
            elif instruction.endswith("Successful execution."):
                fill = 1.0
            else:
                fill = 0.0
            return (
                np.full((4, 3), fill, dtype=np.float32),
                np.ones(4, dtype=bool),
            )

        with mock.patch.object(
            adapter_module,
            "load_text_context_arrays",
            side_effect=fake_load,
        ) as loader:
            observation = adapter.env_obs_to_policy_obs(
                self._env_obs(),
                camera_key="front",
                camera_mapping={"base": "front"},
                task_prompt="Task. Successful execution.",
                cfg_base_prompt="Task.",
                cfg_failure_prompt="Task. Failed execution.",
            )

        self.assertEqual(loader.call_count, 3)
        self.assertTrue(np.all(observation["context"] == 1.0))
        self.assertTrue(np.all(observation["negative_context"] == 0.0))
        self.assertTrue(np.all(observation["failure_context"] == 2.0))


if __name__ == "__main__":
    unittest.main()
