from __future__ import annotations

import importlib.util
import json
from argparse import Namespace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


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


def _write_generic_summary(
    path: Path,
    *,
    successful_seeds: set[int],
    seed_start: int = 100,
    episodes: int = 6,
) -> None:
    rows = [
        {"seed": seed, "success": seed in successful_seeds}
        for seed in range(seed_start, seed_start + episodes)
    ]
    payload = {
        "task": "water_plant",
        "inference_seed": 314159,
        "replan_steps": 25,
        "max_env_steps": 1500,
        "control_mode": "blocking",
        "save_video": True,
        "save_actions": True,
        "randomize": False,
        "randomize_dynamics": False,
        "action_clip": False,
        "total_episodes": episodes,
        "episodes_per_task": episodes,
        "total_successes": len(successful_seeds),
        "tasks": [
            {
                "env_name": "water_plant",
                "episodes": episodes,
                "successes": len(successful_seeds),
                "episode_results": rows,
            }
        ],
    }
    path.write_text(json.dumps(payload))


def _generic_args(root: Path) -> Namespace:
    digest_a = "a" * 64
    sidecar = root / "m_provenance.json"
    sidecar.write_text(
        json.dumps(
            {
                "variant": "M",
                "checkpoint_step": 6000,
                "checkpoint_sha256": "b" * 64,
                "resolved_config_sha256": "c" * 64,
            }
        )
    )
    return Namespace(
        summary=[
            f"S0={root / 's0_summary.json'}",
            f"M={root / 'm_summary.json'}",
        ],
        compare=["M:S0"],
        seed_start=100,
        seed_stop_exclusive=106,
        protocol=[
            'task="water_plant"',
            "inference_seed=314159",
            "replan_steps=25",
            "max_env_steps=1500",
            'control_mode="blocking"',
            "save_video=true",
            "save_actions=true",
            "randomize=false",
            "randomize_dynamics=false",
            "action_clip=false",
        ],
        provenance_sidecar=[f"M={sidecar}"],
        variant_checkpoint_step=["S0=6500"],
        checkpoint_path=[],
        checkpoint_sha256=[f"S0={digest_a}"],
        config_path=[],
        config_sha256=[f"S0={digest_a}"],
        bootstrap_samples=2_000,
        bootstrap_seed=11,
        output_dir=root / "neutral_evidence",
        output_prefix="e1_fresh_seed_comparison",
    )


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

    def test_generic_e1_mixed_steps_cli_and_sidecar_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_generic_summary(
                root / "s0_summary.json", successful_seeds={100, 101, 102}
            )
            _write_generic_summary(
                root / "m_summary.json",
                successful_seeds={100, 101, 103, 104, 105},
            )
            args = _generic_args(root)

            result = MODULE.run(args)

            self.assertEqual(
                result["schema_version"],
                "fitwam_paired_rollout_comparison_v2",
            )
            self.assertEqual(
                result["variants"]["S0"]["provenance"]["checkpoint_step"],
                6500,
            )
            self.assertEqual(
                result["variants"]["M"]["provenance"]["checkpoint_step"],
                6000,
            )
            self.assertEqual(result["success_counts"], {"S0": 3, "M": 5})
            comparison = result["comparisons"]["M_vs_S0"]
            self.assertEqual(comparison["success_delta_count"], 2)
            self.assertEqual(
                comparison["discordant"],
                {
                    "method_only_success": 3,
                    "baseline_only_success": 1,
                },
            )
            self.assertEqual(
                comparison["mcnemar_exact_two_sided_p"], 0.625
            )
            self.assertEqual(
                len(result["variants"]["S0"]["summary_sha256"]), 64
            )
            self.assertEqual(
                len(
                    result["variants"]["M"]["provenance_sidecar"]["sha256"]
                ),
                64,
            )

            paths = MODULE._write_generic_outputs(args, result)
            self.assertEqual(
                paths["json"].name, "e1_fresh_seed_comparison.json"
            )
            self.assertEqual(json.loads(paths["json"].read_text()), result)
            self.assertIn("| M | 6000 | 5/6 |", paths["md"].read_text())
            self.assertIn("M_vs_S0", paths["csv"].read_text())

    def test_generic_rejects_protocol_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_generic_summary(
                root / "s0_summary.json", successful_seeds={100}
            )
            _write_generic_summary(
                root / "m_summary.json", successful_seeds={100}
            )
            payload = json.loads((root / "m_summary.json").read_text())
            payload["replan_steps"] = 24
            (root / "m_summary.json").write_text(json.dumps(payload))

            with self.assertRaisesRegex(ValueError, "protocol replan_steps"):
                MODULE.run(_generic_args(root))

    def test_generic_rejects_seed_range_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_generic_summary(
                root / "s0_summary.json", successful_seeds={100}
            )
            _write_generic_summary(
                root / "m_summary.json", successful_seeds={100}
            )
            payload = json.loads((root / "m_summary.json").read_text())
            payload["tasks"][0]["episode_results"][-1]["seed"] = 999
            (root / "m_summary.json").write_text(json.dumps(payload))

            with self.assertRaisesRegex(ValueError, "seed range mismatch"):
                MODULE.run(_generic_args(root))

    def test_generic_rejects_conflicting_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_generic_summary(
                root / "s0_summary.json", successful_seeds={100}
            )
            _write_generic_summary(
                root / "m_summary.json", successful_seeds={100}
            )
            args = _generic_args(root)
            args.checkpoint_sha256.append(f"M={'d' * 64}")

            with self.assertRaisesRegex(
                ValueError, "checkpoint_sha256 conflicts"
            ):
                MODULE.run(args)

    def test_generic_rejects_reported_success_rate_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_generic_summary(
                root / "s0_summary.json", successful_seeds={100}
            )
            _write_generic_summary(
                root / "m_summary.json", successful_seeds={100}
            )
            payload = json.loads((root / "m_summary.json").read_text())
            payload["overall_success_rate"] = 0.9
            (root / "m_summary.json").write_text(json.dumps(payload))

            with self.assertRaisesRegex(ValueError, "overall_success_rate"):
                MODULE.run(_generic_args(root))

    def test_generic_rejects_checkpoint_file_hash_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_generic_summary(
                root / "s0_summary.json", successful_seeds={100}
            )
            _write_generic_summary(
                root / "m_summary.json", successful_seeds={100}
            )
            checkpoint = root / "s0.pt"
            checkpoint.write_bytes(b"checkpoint")
            args = _generic_args(root)
            args.checkpoint_path = [f"S0={checkpoint}"]

            with self.assertRaisesRegex(ValueError, "checkpoint_path hash"):
                MODULE.run(args)

    def test_generic_cli_parses_explicit_evidence_contract(self) -> None:
        argv = [
            str(SCRIPT),
            "--summary",
            "S0=/tmp/s0.json",
            "--summary",
            "M=/tmp/m.json",
            "--compare",
            "M:S0",
            "--seed-start",
            "20262000",
            "--seed-stop-exclusive",
            "20262200",
            "--protocol",
            'task="water_plant"',
            "--protocol",
            "inference_seed=314159",
            "--protocol",
            "replan_steps=25",
            "--protocol",
            "max_env_steps=1500",
            "--protocol",
            'control_mode="blocking"',
            "--protocol",
            "save_video=true",
            "--protocol",
            "save_actions=true",
            "--protocol",
            "randomize=false",
            "--protocol",
            "randomize_dynamics=false",
            "--protocol",
            "action_clip=false",
            "--variant-checkpoint-step",
            "S0=6500",
            "--variant-checkpoint-step",
            "M=6000",
            "--checkpoint-sha256",
            f"S0={'a' * 64}",
            "--checkpoint-sha256",
            f"M={'b' * 64}",
            "--config-sha256",
            f"S0={'c' * 64}",
            "--config-sha256",
            f"M={'d' * 64}",
            "--output-dir",
            "/tmp/e1-evidence",
            "--output-prefix",
            "e1_seed20262000_20262199",
        ]
        with patch("sys.argv", argv):
            args = MODULE.parse_args()
        self.assertEqual(args.summary, ["S0=/tmp/s0.json", "M=/tmp/m.json"])
        self.assertEqual(args.compare, ["M:S0"])
        self.assertEqual(args.seed_stop_exclusive, 20262200)


if __name__ == "__main__":
    unittest.main()
