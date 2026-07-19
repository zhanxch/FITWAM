#!/usr/bin/env python3
"""Build paired statistics for the frozen FITWAM Water Plant evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from pathlib import Path
from typing import Any


VARIANTS = ("B1", "B0", "C", "M")
PRIMARY_BASELINE = "B1"
PRIMARY_METHOD = "M"
HEX_SHA256_LENGTH = 64


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--checkpoint-step", type=int, default=6500)
    parser.add_argument("--bootstrap-samples", type=int, default=100_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260719)
    parser.add_argument("--min-primary-delta", type=float, default=0.04)
    return parser.parse_args()


def _read_report(path: Path, expected_variant: str) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if payload.get("status") != "valid":
        raise ValueError(f"{path}: report status is not valid")
    if payload.get("variant") != expected_variant:
        raise ValueError(f"{path}: variant mismatch")
    settings = payload.get("settings") or {}
    expected_settings = {
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
    }
    for key, value in expected_settings.items():
        if settings.get(key) != value:
            raise ValueError(
                f"{path}: settings.{key}={settings.get(key)!r}, expected {value!r}"
            )
    episodes = payload.get("episodes")
    if not isinstance(episodes, list) or len(episodes) != 50:
        raise ValueError(f"{path}: expected 50 episode rows")
    by_seed: dict[int, bool] = {}
    for row in episodes:
        seed = row.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError(f"{path}: episode seed must be an integer")
        if seed in by_seed:
            raise ValueError(f"{path}: duplicate seed {seed}")
        success = row.get("success")
        if not isinstance(success, bool):
            raise ValueError(f"{path}: seed {seed} has non-boolean success")
        by_seed[seed] = success
    expected_seeds = set(range(20261000, 20261050))
    if set(by_seed) != expected_seeds:
        raise ValueError(f"{path}: seed set does not match the frozen protocol")
    successes = sum(by_seed.values())
    if payload.get("final_successes") != successes:
        raise ValueError(f"{path}: final_successes disagrees with episode outcomes")
    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError(f"{path}: provenance must be a JSON object")
    expected_provenance = {
        "variant": expected_variant,
        "checkpoint_step": 6500,
        "inference_seed": 314159,
    }
    for key, value in expected_provenance.items():
        if provenance.get(key) != value:
            raise ValueError(
                f"{path}: provenance.{key}={provenance.get(key)!r}, "
                f"expected {value!r}"
            )
    for key in (
        "checkpoint_sha256",
        "resolved_config_sha256",
        "task_config_sha256",
        "normalization_sha256",
        "text_cache_sha256",
    ):
        digest = provenance.get(key)
        if not isinstance(digest, str) or len(digest) != HEX_SHA256_LENGTH:
            raise ValueError(f"{path}: provenance.{key} is not a SHA-256 digest")
    if provenance.get("normalization_kind") != "meta_dir":
        raise ValueError(f"{path}: expected meta_dir normalization")
    code_files = provenance.get("code_files_sha256")
    if not isinstance(code_files, dict) or not code_files:
        raise ValueError(f"{path}: code_files_sha256 must be a nonempty object")
    for label, digest in code_files.items():
        if (
            not isinstance(label, str)
            or not isinstance(digest, str)
            or len(digest) != HEX_SHA256_LENGTH
        ):
            raise ValueError(f"{path}: invalid code_files_sha256 entry")
    return {"payload": payload, "by_seed": by_seed, "provenance": provenance}


def _verify_shared_provenance(reports: dict[str, dict[str, Any]]) -> None:
    shared_keys = (
        "task_config_sha256",
        "normalization_kind",
        "normalization_sha256",
        "text_cache_sha256",
        "inference_seed",
        "code_files_sha256",
    )
    baseline = reports[PRIMARY_BASELINE]["provenance"]
    for variant, report in reports.items():
        provenance = report["provenance"]
        for key in shared_keys:
            if provenance.get(key) != baseline.get(key):
                raise ValueError(
                    f"{variant}: provenance.{key} differs from "
                    f"{PRIMARY_BASELINE}"
                )


def _exact_mcnemar(discordant_a: int, discordant_b: int) -> float:
    total = discordant_a + discordant_b
    if total == 0:
        return 1.0
    lower = min(discordant_a, discordant_b)
    cdf = sum(math.comb(total, k) for k in range(lower + 1)) / (2**total)
    return min(1.0, 2.0 * cdf)


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _paired_comparison(
    baseline: dict[int, bool],
    method: dict[int, bool],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    seeds = sorted(baseline)
    if seeds != sorted(method):
        raise ValueError("paired comparison requires identical seeds")
    deltas = [int(method[seed]) - int(baseline[seed]) for seed in seeds]
    rng = random.Random(bootstrap_seed)
    bootstrap = [
        sum(deltas[rng.randrange(len(deltas))] for _ in deltas) / len(deltas)
        for _ in range(bootstrap_samples)
    ]
    method_only = sum(method[seed] and not baseline[seed] for seed in seeds)
    baseline_only = sum(baseline[seed] and not method[seed] for seed in seeds)
    return {
        "episodes": len(seeds),
        "baseline_successes": sum(baseline.values()),
        "method_successes": sum(method.values()),
        "success_delta": sum(deltas) / len(deltas),
        "success_delta_count": sum(deltas),
        "bootstrap_ci_95": [
            _percentile(bootstrap, 0.025),
            _percentile(bootstrap, 0.975),
        ],
        "discordant": {
            "method_only_success": method_only,
            "baseline_only_success": baseline_only,
        },
        "mcnemar_exact_two_sided_p": _exact_mcnemar(
            method_only, baseline_only
        ),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = args.rollout_root.expanduser().resolve()
    reports = {
        variant: _read_report(
            root
            / f"{variant}_step{args.checkpoint_step:06d}"
            / "validated_summary.json",
            variant,
        )
        for variant in VARIANTS
    }
    _verify_shared_provenance(reports)
    rates = {
        variant: sum(report["by_seed"].values()) / 50
        for variant, report in reports.items()
    }
    success_counts = {
        variant: sum(report["by_seed"].values())
        for variant, report in reports.items()
    }
    comparisons = {
        f"{variant}_vs_{PRIMARY_BASELINE}": _paired_comparison(
            reports[PRIMARY_BASELINE]["by_seed"],
            reports[variant]["by_seed"],
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed,
        )
        for variant in ("B0", "C", "M")
    }
    comparisons["M_vs_C"] = _paired_comparison(
        reports["C"]["by_seed"],
        reports["M"]["by_seed"],
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    primary = comparisons[f"{PRIMARY_METHOD}_vs_{PRIMARY_BASELINE}"]
    gate_passed = primary["success_delta"] >= args.min_primary_delta
    result = {
        "schema_version": "fitwam_offline_paired_comparison_v1",
        "status": "valid",
        "evaluation_scope": "checkpoint_screening_50_paired_episodes",
        "claim_limit": (
            "This report screens one checkpoint per variant. Publication-level "
            "claims require the planned larger evaluation and training-seed "
            "replication."
        ),
        "protocol": {
            "task": "water_plant",
            "checkpoint_step": args.checkpoint_step,
            "episodes": 50,
            "base_seed": 20261000,
            "bootstrap_samples": args.bootstrap_samples,
            "bootstrap_seed": args.bootstrap_seed,
        },
        "success_counts": success_counts,
        "success_rates": rates,
        "comparisons": comparisons,
        "primary_gate": {
            "role": "checkpoint_screening",
            "comparison": f"{PRIMARY_METHOD}_vs_{PRIMARY_BASELINE}",
            "minimum_delta": args.min_primary_delta,
            "observed_delta": primary["success_delta"],
            "passed": gate_passed,
        },
    }
    return result


def _write_outputs(args: argparse.Namespace, result: dict[str, Any]) -> None:
    for path in (args.output_json, args.output_csv, args.output_md):
        path.parent.mkdir(parents=True, exist_ok=True)

    json_text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    _atomic_write_text(args.output_json, json_text)

    csv_path = args.output_csv.with_name(
        f".{args.output_csv.name}.tmp-{os.getpid()}"
    )
    try:
        with csv_path.open("w", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=(
                    "comparison",
                    "baseline_successes",
                    "method_successes",
                    "success_delta_count",
                    "success_delta",
                    "ci95_low",
                    "ci95_high",
                    "method_only_success",
                    "baseline_only_success",
                    "mcnemar_p",
                ),
            )
            writer.writeheader()
            for name, row in result["comparisons"].items():
                writer.writerow(
                    {
                        "comparison": name,
                        "baseline_successes": row["baseline_successes"],
                        "method_successes": row["method_successes"],
                        "success_delta_count": row["success_delta_count"],
                        "success_delta": row["success_delta"],
                        "ci95_low": row["bootstrap_ci_95"][0],
                        "ci95_high": row["bootstrap_ci_95"][1],
                        "method_only_success": row["discordant"][
                            "method_only_success"
                        ],
                        "baseline_only_success": row["discordant"][
                            "baseline_only_success"
                        ],
                        "mcnemar_p": row["mcnemar_exact_two_sided_p"],
                    }
                )
        os.replace(csv_path, args.output_csv)
    finally:
        csv_path.unlink(missing_ok=True)

    lines = [
        "# Water Plant Paired Offline Checkpoint Screening",
        "",
        "| Variant | Successes | Success rate |",
        "| --- | ---: | ---: |",
    ]
    for variant in VARIANTS:
        rate = result["success_rates"][variant]
        successes = result["success_counts"][variant]
        lines.append(f"| {variant} | {successes}/50 | {rate:.1%} |")
    lines.extend(
        [
            "",
            "| Comparison | Delta | 95% paired bootstrap CI | McNemar p |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for name, row in result["comparisons"].items():
        low, high = row["bootstrap_ci_95"]
        lines.append(
            f"| {name} | {row['success_delta']:+.1%} | "
            f"[{low:+.1%}, {high:+.1%}] | "
            f"{row['mcnemar_exact_two_sided_p']:.4f} |"
        )
    gate = result["primary_gate"]
    lines.extend(
        [
            "",
            f"Primary gate (`{gate['comparison']} >= "
            f"{gate['minimum_delta']:.1%}`): "
            f"**{'PASS' if gate['passed'] else 'FAIL'}** "
            f"({gate['observed_delta']:+.1%}).",
            "",
        ]
    )
    _atomic_write_text(args.output_md, "\n".join(lines))


def _atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(text)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    args = parse_args()
    result = run(args)
    _write_outputs(args, result)
    print(json.dumps(result["primary_gate"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
