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
        preformal_mode: str | None = None,
        resume_state_dir: str | None = None,
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
        return subprocess.run(
            ["bash", str(LAUNCHER), "B0", *arguments],
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
            text.index("for protocol_variant in B0 B1 C M"),
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


if __name__ == "__main__":
    unittest.main()
