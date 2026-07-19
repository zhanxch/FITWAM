from __future__ import annotations

import importlib.util
import json
from argparse import Namespace
from pathlib import Path
import tempfile
import unittest


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "water_plant"
    / "compare_offline_rollouts.py"
)
SPEC = importlib.util.spec_from_file_location("compare_offline_rollouts", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _write_report(root: Path, variant: str, successful_seeds: set[int]) -> None:
    digest = "a" * 64
    report = {
        "status": "valid",
        "variant": variant,
        "final_successes": len(successful_seeds),
        "provenance": {
            "variant": variant,
            "checkpoint_step": 6500,
            "checkpoint_sha256": digest,
            "resolved_config_sha256": digest,
            "task_config_sha256": digest,
            "normalization_kind": "meta_dir",
            "normalization_sha256": digest,
            "text_cache_sha256": digest,
            "inference_seed": 314159,
            "code_files_sha256": {"runner": digest},
        },
        "settings": {
            "episodes": 50,
            "base_seed": 20261000,
            "inference_seed": 314159,
            "gpus": [0, 1, 2, 3],
            "task": "water_plant",
            "replan_steps": 25,
            "max_env_steps": 1500,
            "save_video": True,
            "save_actions": True,
            "randomize": False,
            "randomize_dynamics": False,
            "action_clip": False,
        },
        "episodes": [
            {"seed": seed, "success": seed in successful_seeds}
            for seed in range(20261000, 20261050)
        ],
    }
    output = root / f"{variant}_step006500" / "validated_summary.json"
    output.parent.mkdir(parents=True)
    output.write_text(json.dumps(report))


class CompareOfflineRolloutsTest(unittest.TestCase):
    def test_paired_comparison_and_primary_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            seeds = list(range(20261000, 20261050))
            _write_report(root, "B1", set(seeds[:25]))
            _write_report(root, "B0", set(seeds[:24]))
            _write_report(root, "C", set(seeds[:26]))
            _write_report(root, "M", set(seeds[:27]))

            result = MODULE.run(
                args := Namespace(
                    rollout_root=root,
                    output_json=root / "paired.json",
                    output_csv=root / "paired.csv",
                    output_md=root / "paired.md",
                    checkpoint_step=6500,
                    bootstrap_samples=2_000,
                    bootstrap_seed=7,
                    min_primary_delta=0.04,
                )
            )

            primary = result["comparisons"]["M_vs_B1"]
            self.assertEqual(primary["success_delta_count"], 2)
            self.assertEqual(primary["success_delta"], 0.04)
            self.assertEqual(
                primary["discordant"],
                {
                    "method_only_success": 2,
                    "baseline_only_success": 0,
                },
            )
            self.assertEqual(primary["mcnemar_exact_two_sided_p"], 0.5)
            self.assertTrue(result["primary_gate"]["passed"])
            self.assertEqual(
                result["comparisons"]["M_vs_C"]["success_delta_count"], 1
            )
            MODULE._write_outputs(args, result)
            self.assertEqual(
                json.loads(args.output_json.read_text()),
                result,
            )
            self.assertIn("| M | 27/50 | 54.0% |", args.output_md.read_text())
            self.assertIn("M_vs_C", args.output_csv.read_text())

    def test_mcnemar_is_one_without_discordant_pairs(self) -> None:
        outcomes = {seed: seed % 2 == 0 for seed in range(50)}
        comparison = MODULE._paired_comparison(
            outcomes,
            outcomes,
            bootstrap_samples=100,
            bootstrap_seed=1,
        )
        self.assertEqual(comparison["success_delta"], 0.0)
        self.assertEqual(comparison["mcnemar_exact_two_sided_p"], 1.0)

    def test_report_rejects_string_success(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            seeds = list(range(20261000, 20261050))
            _write_report(root, "B1", set(seeds[:25]))
            report_path = (
                root / "B1_step006500" / "validated_summary.json"
            )
            payload = json.loads(report_path.read_text())
            payload["episodes"][0]["success"] = "false"
            report_path.write_text(json.dumps(payload))

            with self.assertRaisesRegex(ValueError, "non-boolean success"):
                MODULE._read_report(report_path, "B1")

    def test_paired_comparison_rejects_empty_bootstrap(self) -> None:
        outcomes = {seed: seed % 2 == 0 for seed in range(50)}
        with self.assertRaisesRegex(
            ValueError, "bootstrap_samples must be positive"
        ):
            MODULE._paired_comparison(
                outcomes,
                outcomes,
                bootstrap_samples=0,
                bootstrap_seed=1,
            )

    def test_report_rejects_non_integer_seed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            seeds = list(range(20261000, 20261050))
            _write_report(root, "B1", set(seeds[:25]))
            report_path = (
                root / "B1_step006500" / "validated_summary.json"
            )
            payload = json.loads(report_path.read_text())
            payload["episodes"][0]["seed"] = 20261000.9
            report_path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "seed must be an integer"):
                MODULE._read_report(report_path, "B1")


if __name__ == "__main__":
    unittest.main()
