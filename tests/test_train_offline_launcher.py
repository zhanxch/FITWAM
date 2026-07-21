from __future__ import annotations

import os
from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "water_plant" / "train_offline_self_improving.sh"
PREPARE = ROOT / "scripts" / "water_plant" / "prepare_offline_self_improving.sh"


class TrainOfflineLauncherTest(unittest.TestCase):
    def run_validation(
        self,
        *arguments: str,
        variant: str = "B0",
        preformal_mode: str | None = None,
        resume_state_dir: str | None = None,
        strict_common_init: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        environment = dict(os.environ)
        environment["FITWAM_VALIDATE_OVERRIDES_ONLY"] = "1"
        environment.setdefault("PAIR_QUALITY_GATE_STATUS", "passed")
        environment.setdefault("PAIR_QUALITY_GATE_MODE", "formal")
        if preformal_mode is None:
            environment.pop("FITWAM_PREFORMAL_MODE", None)
        else:
            environment["FITWAM_PREFORMAL_MODE"] = preformal_mode
        if resume_state_dir is None:
            environment.pop("RESUME_STATE_DIR", None)
        else:
            environment["RESUME_STATE_DIR"] = resume_state_dir
        if strict_common_init:
            environment["FITWAM_STRICT_COMMON_INIT_COMPARISON"] = "1"
        else:
            environment.pop("FITWAM_STRICT_COMMON_INIT_COMPARISON", None)
        return subprocess.run(
            ["bash", str(LAUNCHER), variant, *arguments],
            cwd=ROOT,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_only_num_workers_override_is_allowed(self) -> None:
        result = self.run_validation("num_workers=8")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("execution_mode=formal max_steps=6500", result.stdout)
        self.assertIn("save_weights_every=0", result.stdout)

    def test_pair_shuffle_is_a_separately_labeled_variant(self) -> None:
        rejected = self.run_validation(variant="M_PAIR_SHUFFLE")
        self.assertEqual(rejected.returncode, 2)
        self.assertIn(
            "requires FITWAM_STRICT_COMMON_INIT_COMPARISON=1",
            rejected.stderr,
        )
        result = self.run_validation(
            variant="M_PAIR_SHUFFLE",
            strict_common_init=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("execution_mode=formal max_steps=6500", result.stdout)
        text = LAUNCHER.read_text(encoding="utf-8")
        self.assertIn("M_PAIR_SHUFFLE_MANIFEST_PATH", text)
        self.assertIn("PAIR_SHUFFLE_TARGETS_PATH", text)
        self.assertIn("PAIR_SHUFFLE_PROOF_PATH", text)
        self.assertIn("pair_shuffle_proof_sha256", text)
        self.assertIn(
            'if [[ "${VARIANT}" == "M_PAIR_SHUFFLE" ]]; then',
            text,
        )

    def test_base_variants_do_not_require_pair_shuffle_environment(self) -> None:
        for variant in ("B0", "B1", "C", "M"):
            with self.subTest(variant=variant):
                result = self.run_validation(variant=variant)
                self.assertEqual(result.returncode, 0, result.stderr)
        strict_b0 = self.run_validation(variant="B0", strict_common_init=True)
        self.assertEqual(strict_b0.returncode, 0, strict_b0.stderr)
        text = LAUNCHER.read_text(encoding="utf-8")
        self.assertIn("PROTOCOL_VARIANTS=(B0 B1 C M)", text)
        self.assertIn(
            'if [[ "${INCLUDE_PAIR_SHUFFLE_CONTROL}" == "1" ]]; then',
            text,
        )
        self.assertIn("PAIR_SHUFFLE_PREFLIGHT_ARGS=()", text)

    def test_m_historical_and_strict_modes_have_distinct_initialization(self) -> None:
        historical = self.run_validation(variant="M")
        strict = self.run_validation(variant="M", strict_common_init=True)
        self.assertEqual(historical.returncode, 0, historical.stderr)
        self.assertEqual(strict.returncode, 0, strict.stderr)
        text = LAUNCHER.read_text(encoding="utf-8")
        self.assertIn('SELECTED_INIT_WEIGHTS="${INIT_WEIGHTS}"', text)
        self.assertIn('SELECTED_INIT_WEIGHTS="${COMMON_INIT_WEIGHTS}"', text)
        self.assertIn(
            'export FASTWAM_RESUME="${RESUME_STATE_DIR:-${SELECTED_INIT_WEIGHTS}}"',
            text,
        )
        self.assertIn("strict_common_init_pair_shuffle", text)
        self.assertIn("COMMON_INIT_PREFLIGHT_ARGS=()", text)

    def test_protocol_overrides_are_rejected(self) -> None:
        for override in (
            "model.offline_steer.enabled=true",
            "+max_steps=1",
            "~data",
            "task=other",
            "resume=/tmp/model.pt",
            "wandb.enabled=false",
            "--multirun",
        ):
            with self.subTest(override=override):
                result = self.run_validation(override)
                self.assertEqual(result.returncode, 2)
                self.assertIn(
                    "not allowed"
                    if override.startswith("--")
                    else "rejected",
                    result.stderr,
                )

    def test_preformal_modes_have_frozen_contracts_and_namespaces(self) -> None:
        expected = {
            "smoke20": (
                "max_steps=20 eval_every=10 save_weights_every=10 "
                "save_state_every=20"
            ),
            "smoke500": (
                "max_steps=500 eval_every=100 save_weights_every=100 "
                "save_state_every=500"
            ),
        }
        for mode, contract in expected.items():
            with self.subTest(mode=mode):
                result = self.run_validation(
                    preformal_mode=mode,
                    resume_state_dir=(
                        "/tmp/step_000020" if mode == "smoke500" else None
                    ),
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(f"execution_mode={mode} {contract}", result.stdout)
                self.assertIn(
                    f"preformal_smoke/{mode}",
                    result.stdout,
                )
                self.assertIn(
                    "wandb_group=water_plant_offline_self_improving_preformal_smoke",
                    result.stdout,
                )
        text = LAUNCHER.read_text(encoding="utf-8")
        self.assertEqual(text.count("+lr_scheduler_total_steps=500"), 2)

    def test_smoke500_requires_resume_state_dir(self) -> None:
        result = self.run_validation(preformal_mode="smoke500")
        self.assertEqual(result.returncode, 2)
        self.assertIn(
            "requires non-empty RESUME_STATE_DIR",
            result.stderr,
        )

    def test_invalid_preformal_mode_is_rejected(self) -> None:
        result = self.run_validation(preformal_mode="smoke100")
        self.assertEqual(result.returncode, 2)
        self.assertIn(
            "must be formal, smoke20, or smoke500",
            result.stderr,
        )

    def test_failed_pair_quality_is_smoke_only(self) -> None:
        environment = dict(os.environ)
        environment.update(
            {
                "FITWAM_VALIDATE_OVERRIDES_ONLY": "1",
                "PAIR_QUALITY_GATE_STATUS": "failed",
                "PAIR_QUALITY_GATE_MODE": "preformal",
            }
        )
        formal = subprocess.run(
            ["bash", str(LAUNCHER), "M"],
            cwd=ROOT,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(formal.returncode, 2)
        self.assertIn("Formal training is blocked", formal.stderr)

        environment["FITWAM_PREFORMAL_MODE"] = "smoke20"
        smoke = subprocess.run(
            ["bash", str(LAUNCHER), "M"],
            cwd=ROOT,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(smoke.returncode, 0, smoke.stderr)
        self.assertIn("execution_mode=smoke20", smoke.stdout)

    def test_launcher_exports_one_code_snapshot_for_all_variants(self) -> None:
        text = LAUNCHER.read_text(encoding="utf-8")
        self.assertIn("export FITWAM_CODE_SNAPSHOT_SHA256=", text)
        self.assertEqual(text.count("build_code_snapshot()"), 1)
        self.assertLess(
            text.index("export FITWAM_CODE_SNAPSHOT_SHA256="),
            text.index("PROTOCOL_VARIANTS=(B0 B1 C M)"),
        )
        self.assertIn(
            "+experiment_provenance.run_mode=preformal_smoke",
            text,
        )
        self.assertIn("--expected-resume-step", text)
        self.assertIn(
            'export PROTOCOL_BUNDLE_PATH="${PROTOCOL_BUNDLE_BASE_PATH}.${VARIANT}"',
            text,
        )

    def test_prepare_freezes_training_inputs_and_writes_reusable_env(self) -> None:
        text = PREPARE.read_text(encoding="utf-8")
        self.assertIn('export ROLLOUT_RAW="${ROLLOUT_DATASET}"', text)
        self.assertIn("WATER_PLANT_TEXT_CACHE_BASENAME=", text)
        self.assertIn(".t5_len128.wan22ti2v5b.pt", text)
        self.assertIn("NORM_STATS_BUNDLE_SHA256=", text)
        self.assertIn(
            "python scripts/water_plant/validate_s0_formal_outputs.py",
            text,
        )
        self.assertIn('--dataset "${ROLLOUT_DATASET}"', text)
        self.assertIn('--report "${FORMAL_VALIDATION_REPORT}"', text)
        self.assertIn('EXECUTION_ENV="${EVE_ROOT}/protocol/offline_v1.env"', text)
        self.assertIn("FORMAL_VALIDATION_REPORT \\", text)
        launcher_text = LAUNCHER.read_text(encoding="utf-8")
        self.assertIn("referenced_task_texts", launcher_text)
        self.assertIn("build_text_cache_contract", launcher_text)
        self.assertIn('cmp -s "${EXECUTION_ENV_TMP}" "${EXECUTION_ENV}"', text)
        self.assertIn('PREFLIGHT_GPUS="${PREFLIGHT_GPUS:-0,1,2,3}"', text)
        self.assertIn('CUDA_VISIBLE_DEVICES="${PREFLIGHT_GPUS}"', text)
        self.assertIn('PREFLIGHT_EXECUTION_MODE=smoke20', text)
        self.assertIn(
            'FITWAM_PREFORMAL_MODE="${PREFLIGHT_EXECUTION_MODE}"',
            text,
        )
        self.assertIn("--matching bounded", text)
        self.assertIn("--max-success-uses 2", text)
        self.assertIn("--max-failure-uses 1", text)
        self.assertIn("--min-pair-weight 0.05", text)
        self.assertIn("--max-candidates-per-episode 10", text)
        self.assertIn('--fit-calibration "${PAIR_CALIBRATION}"', text)
        self.assertIn('--diagnostics-output "${PAIR_DIAGNOSTICS}"', text)
        self.assertIn("--success-auxiliary-dataset-ids", text)
        self.assertIn("--success-sample-mode event_only", text)
        self.assertIn("match_auxiliary_manifest_budget.py", text)
        self.assertIn("MANIFEST_B0_MATCH_DIAGNOSTICS", text)
        self.assertIn("render_state_line_audit.py", text)
        self.assertIn("--num-episodes 20", text)
        self.assertIn("build_pair_shuffle_control.py", text)
        self.assertIn(
            'FITWAM_ENABLE_PAIR_SHUFFLE_CONTROL="${FITWAM_ENABLE_PAIR_SHUFFLE_CONTROL:-0}"',
            text,
        )
        self.assertLess(
            text.index('"${FITWAM_ENABLE_PAIR_SHUFFLE_CONTROL}" == "1"'),
            text.index("PAIR_SHUFFLE_OUTPUTS=("),
        )
        self.assertIn('--output-manifest "${MANIFEST_M_PAIR_SHUFFLE}"', text)
        self.assertIn('--output-pair-targets "${PAIR_SHUFFLE_TARGETS}"', text)
        self.assertIn('--proof-output "${PAIR_SHUFFLE_PROOF}"', text)
        self.assertIn('--shuffle-seed "${PAIR_SHUFFLE_SEED}"', text)
        self.assertIn("PAIR_SHUFFLE_TARGETS_PATH \\", text)
        self.assertIn("PAIR_SHUFFLE_PROOF_PATH \\", text)
        self.assertIn("PAIR_SHUFFLE_SEED \\", text)
        self.assertIn("build_common_init_checkpoint.py", text)
        self.assertIn(
            'FITWAM_STRICT_COMMON_INIT_COMPARISON="${FITWAM_STRICT_COMMON_INIT_COMPARISON:-0}"',
            text,
        )
        self.assertIn(
            '--expected-config-sha256 "${COMMON_INIT_CONFIG_SHA256}"',
            text,
        )
        self.assertIn(
            '--expected-baseline-sha256 "${SOURCE_CHECKPOINT_SHA256}"',
            text,
        )
        for variable in (
            "COMMON_INIT_WEIGHTS",
            "COMMON_INIT_PROOF",
            "COMMON_INIT_CONFIG",
            "COMMON_INIT_SEED",
            "COMMON_INIT_WEIGHTS_SHA256",
            "COMMON_INIT_PROOF_SHA256",
            "COMMON_INIT_BASELINE_SHA256",
            "COMMON_INIT_CONFIG_SHA256",
        ):
            self.assertIn(f"{variable} \\", text)


if __name__ == "__main__":
    unittest.main()
