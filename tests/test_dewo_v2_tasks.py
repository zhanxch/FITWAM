"""Tests for DEWO v2 task registry and CFG env overrides."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from dewo_v2.tasks import (  # noqa: E402
    CfgRecipe,
    eval_task_yaml,
    get_task,
    parse_cfg_recipe,
    t5_cache_name,
)


class DewoV2TaskTests(unittest.TestCase):
    def test_water_plant_prompt_and_t5_hash(self) -> None:
        task = get_task("water_plant")
        self.assertEqual(
            task.success_prompt,
            "Grasp the watering can and apply water to the plant.",
        )
        self.assertEqual(
            t5_cache_name(task.success_prompt),
            "f742556deff61d95d9f67eb3522f56d6f6c69ff9833ffa5b4beb83dc0d6a40df.t5_len128.wan22ti2v5b.pt",
        )

    def test_hammer_cfg_defaults(self) -> None:
        cfg = parse_cfg_recipe({})
        self.assertEqual(cfg.primary, (0.5, 0.0, 0.5))
        self.assertEqual(cfg.aux_success, (0.4, 0.2, 0.4))
        self.assertEqual(cfg.aux_fail, (0.0, 0.2, 0.4))
        self.assertEqual(cfg.success_suffix, " Successful execution.")
        self.assertIsNone(cfg.failure_suffix)
        self.assertTrue(cfg.fast_fail_closed)
        self.assertEqual(cfg.dropout, 0.0)

    def test_compact_cfg_override(self) -> None:
        cfg = parse_cfg_recipe(
            {
                "CFG_PRIMARY": "0.6,0.0,0.4",
                "CFG_AUX_SUCCESS": "0.3,0.3,0.4",
                "CFG_AUX_FAIL": "0.0,0.5,0.5",
                "CFG_FAILURE_SUFFIX": " Failed execution.",
                "CFG_FAST_FAIL_CLOSED": "0",
            }
        )
        self.assertEqual(cfg.primary, (0.6, 0.0, 0.4))
        self.assertEqual(cfg.aux_success, (0.3, 0.3, 0.4))
        self.assertEqual(cfg.aux_fail, (0.0, 0.5, 0.5))
        self.assertEqual(cfg.failure_suffix, " Failed execution.")
        self.assertFalse(cfg.fast_fail_closed)

    def test_primary_fast_rejected(self) -> None:
        with self.assertRaises(ValueError):
            parse_cfg_recipe({"CFG_PRIMARY": "0.4,0.2,0.4"})

    def test_eval_yaml_uses_success_suffix(self) -> None:
        task = get_task("water_plant")
        text = eval_task_yaml(task, CfgRecipe())
        self.assertIn(
            "Grasp the watering can and apply water to the plant. Successful execution.",
            text,
        )
        self.assertIn("cfg_base_prompt:", text)
        self.assertNotIn("cfg_failure_prompt:", text)

    def test_eval_yaml_includes_failure_prompt_when_suffix_set(self) -> None:
        task = get_task("water_plant")
        text = eval_task_yaml(
            task,
            CfgRecipe(failure_suffix=" Failed execution."),
        )
        self.assertIn("cfg_failure_prompt:", text)
        self.assertIn(
            "Grasp the watering can and apply water to the plant. Failed execution.",
            text,
        )

    def test_unknown_task(self) -> None:
        with self.assertRaises(KeyError):
            get_task("not_a_task")


if __name__ == "__main__":
    unittest.main()
