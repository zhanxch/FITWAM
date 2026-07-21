#!/usr/bin/env python3
"""Build provenance-bound paired rollout statistics.

The legacy entry point preserves the frozen E0 Water Plant report. Repeated
``--summary`` arguments select the generic, evidence-driven comparison mode.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
from pathlib import Path
from typing import Any


VARIANTS = ("B1", "B0", "C", "M")
PRIMARY_BASELINE = "B1"
PRIMARY_METHOD = "M"
HEX_SHA256_LENGTH = 64
VARIANT_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
GENERIC_REQUIRED_PROTOCOL = (
    "task",
    "inference_seed",
    "replan_steps",
    "max_env_steps",
    "control_mode",
    "save_video",
    "save_actions",
    "randomize",
    "randomize_dynamics",
    "action_clip",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout-root", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--checkpoint-step", type=int, default=6500)
    parser.add_argument("--bootstrap-samples", type=int, default=100_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260719)
    parser.add_argument("--min-primary-delta", type=float, default=0.04)
    parser.add_argument(
        "--summary",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Enable generic mode and add a named rollout summary.",
    )
    parser.add_argument(
        "--compare",
        action="append",
        default=[],
        metavar="METHOD:BASELINE",
        help="Add an explicitly directed paired comparison.",
    )
    parser.add_argument("--seed-start", type=int)
    parser.add_argument("--seed-stop-exclusive", type=int)
    parser.add_argument(
        "--protocol",
        action="append",
        default=[],
        metavar="KEY=JSON_VALUE",
        help="Require an exact protocol value in every named summary.",
    )
    parser.add_argument(
        "--provenance-sidecar",
        action="append",
        default=[],
        metavar="NAME=PATH",
    )
    parser.add_argument(
        "--variant-checkpoint-step",
        action="append",
        default=[],
        metavar="NAME=STEP",
    )
    parser.add_argument(
        "--checkpoint-path",
        action="append",
        default=[],
        metavar="NAME=PATH",
    )
    parser.add_argument(
        "--checkpoint-sha256",
        action="append",
        default=[],
        metavar="NAME=HEX",
    )
    parser.add_argument(
        "--config-path",
        action="append",
        default=[],
        metavar="NAME=PATH",
    )
    parser.add_argument(
        "--config-sha256",
        action="append",
        default=[],
        metavar="NAME=HEX",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--output-prefix", default="paired_comparison")
    args = parser.parse_args()
    _validate_cli_args(parser, args)
    return args


def _validate_cli_args(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    if args.summary:
        missing = []
        if args.seed_start is None:
            missing.append("--seed-start")
        if args.seed_stop_exclusive is None:
            missing.append("--seed-stop-exclusive")
        if not args.protocol:
            missing.append("--protocol")
        if not args.compare:
            missing.append("--compare")
        if args.output_dir is None:
            missing.append("--output-dir")
        if missing:
            parser.error(
                "generic mode requires " + ", ".join(missing)
            )
        if args.seed_stop_exclusive <= args.seed_start:
            parser.error("--seed-stop-exclusive must be greater than --seed-start")
        if not VARIANT_NAME_PATTERN.fullmatch(args.output_prefix):
            parser.error("--output-prefix must be a neutral filename stem")
        return

    missing = [
        option
        for option, value in (
            ("--rollout-root", args.rollout_root),
            ("--output-json", args.output_json),
            ("--output-csv", args.output_csv),
            ("--output-md", args.output_md),
        )
        if value is None
    ]
    if missing:
        parser.error("legacy E0 mode requires " + ", ".join(missing))


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


def _split_assignment(spec: str, *, label: str) -> tuple[str, str]:
    if "=" not in spec:
        raise ValueError(f"{label} must use NAME=VALUE syntax: {spec!r}")
    name, value = spec.split("=", 1)
    if not VARIANT_NAME_PATTERN.fullmatch(name):
        raise ValueError(f"{label} has an invalid name: {name!r}")
    if not value:
        raise ValueError(f"{label} has an empty value for {name}")
    return name, value


def _named_values(
    specs: list[str],
    *,
    label: str,
    convert: Any = str,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for spec in specs:
        name, raw = _split_assignment(spec, label=label)
        if name in result:
            raise ValueError(f"{label} repeats name {name!r}")
        try:
            result[name] = convert(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{label} has an invalid value for {name}: {raw!r}"
            ) from exc
    return result


def _protocol_values(specs: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"--protocol must use KEY=JSON_VALUE syntax: {spec!r}")
        key, raw = spec.split("=", 1)
        if not key or key in result:
            raise ValueError(f"invalid or repeated protocol key: {key!r}")
        try:
            result[key] = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"protocol value for {key!r} must be valid JSON; "
                "quote string values"
            ) from exc
    return result


def _strict_equal(actual: Any, expected: Any) -> bool:
    return type(actual) is type(expected) and actual == expected


def _protocol_candidates(payload: dict[str, Any], key: str) -> list[Any]:
    candidates: list[Any] = []
    if key in payload:
        candidates.append(payload[key])
    settings = payload.get("settings")
    if isinstance(settings, dict) and key in settings:
        candidates.append(settings[key])
    if key == "task":
        tasks = payload.get("tasks")
        if isinstance(tasks, list) and len(tasks) == 1:
            task = tasks[0]
            if isinstance(task, dict) and "env_name" in task:
                candidates.append(task["env_name"])
    return candidates


def _validate_protocol(
    payload: dict[str, Any],
    *,
    path: Path,
    expected_protocol: dict[str, Any],
) -> None:
    for key, expected in expected_protocol.items():
        candidates = _protocol_candidates(payload, key)
        if not candidates:
            raise ValueError(f"{path}: protocol field {key!r} is missing")
        for actual in candidates:
            if not _strict_equal(actual, expected):
                raise ValueError(
                    f"{path}: protocol {key}={actual!r}, expected {expected!r}"
                )


def _extract_episode_rows(payload: dict[str, Any], path: Path) -> list[Any]:
    rows = payload.get("episodes")
    if isinstance(rows, list):
        return rows
    tasks = payload.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != 1:
        raise ValueError(f"{path}: expected exactly one task with episode results")
    task = tasks[0]
    if not isinstance(task, dict) or not isinstance(
        task.get("episode_results"), list
    ):
        raise ValueError(f"{path}: task episode_results must be a list")
    return task["episode_results"]


def _validated_outcomes(
    payload: dict[str, Any],
    *,
    path: Path,
    variant: str,
    seed_start: int,
    seed_stop_exclusive: int,
) -> dict[int, bool]:
    embedded_variant = payload.get("variant")
    if embedded_variant is not None and embedded_variant != variant:
        raise ValueError(
            f"{path}: embedded variant {embedded_variant!r} != {variant!r}"
        )
    rows = _extract_episode_rows(payload, path)
    expected_seeds = set(range(seed_start, seed_stop_exclusive))
    if len(rows) != len(expected_seeds):
        raise ValueError(
            f"{path}: found {len(rows)} episodes, expected {len(expected_seeds)}"
        )
    by_seed: dict[int, bool] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"{path}: every episode row must be an object")
        seed = row.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError(f"{path}: episode seed must be an integer")
        if seed in by_seed:
            raise ValueError(f"{path}: duplicate seed {seed}")
        success = row.get("success")
        if not isinstance(success, bool):
            raise ValueError(f"{path}: seed {seed} has non-boolean success")
        by_seed[seed] = success
    if set(by_seed) != expected_seeds:
        missing = sorted(expected_seeds - set(by_seed))
        extra = sorted(set(by_seed) - expected_seeds)
        raise ValueError(
            f"{path}: seed range mismatch; missing={missing}, extra={extra}"
        )

    successes = sum(by_seed.values())
    count_fields = {
        "total_episodes": len(rows),
        "episodes_per_task": len(rows),
        "final_successes": successes,
        "total_successes": successes,
    }
    for key, expected in count_fields.items():
        if key in payload and not _strict_equal(payload[key], expected):
            raise ValueError(
                f"{path}: {key}={payload[key]!r}, expected {expected!r}"
            )
    tasks = payload.get("tasks")
    if isinstance(tasks, list) and len(tasks) == 1 and isinstance(tasks[0], dict):
        for key, expected in (("episodes", len(rows)), ("successes", successes)):
            if key in tasks[0] and not _strict_equal(tasks[0][key], expected):
                raise ValueError(
                    f"{path}: tasks[0].{key}={tasks[0][key]!r}, "
                    f"expected {expected!r}"
                )
    expected_rate = successes / len(rows)
    for owner, key, value in (
        ("summary", "overall_success_rate", payload.get("overall_success_rate")),
        (
            "tasks[0]",
            "success_rate",
            tasks[0].get("success_rate")
            if isinstance(tasks, list)
            and len(tasks) == 1
            and isinstance(tasks[0], dict)
            else None,
        ),
    ):
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isclose(value, expected_rate, rel_tol=0.0, abs_tol=1e-12)
        ):
            raise ValueError(
                f"{path}: {owner}.{key}={value!r}, expected {expected_rate!r}"
            )
    return by_seed


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_provenance(raw: dict[str, Any], *, label: str) -> dict[str, Any]:
    aliases = {
        "checkpoint_step": ("checkpoint_step", "step"),
        "checkpoint_path": ("checkpoint_path", "checkpoint"),
        "checkpoint_sha256": ("checkpoint_sha256",),
        "config_path": ("config_path", "resolved_config_path", "config"),
        "config_sha256": ("config_sha256", "resolved_config_sha256"),
    }
    result: dict[str, Any] = {}
    for canonical, names in aliases.items():
        values = [raw[name] for name in names if raw.get(name) is not None]
        if not values:
            continue
        first = values[0]
        if any(not _strict_equal(value, first) for value in values[1:]):
            raise ValueError(f"{label}: conflicting aliases for {canonical}")
        result[canonical] = first
    return result


def _merge_provenance(
    variant: str, sources: list[tuple[str, dict[str, Any]]]
) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    origins: dict[str, str] = {}
    for label, source in sources:
        canonical = _canonical_provenance(source, label=label)
        for key, value in canonical.items():
            if key in merged and not _strict_equal(merged[key], value):
                raise ValueError(
                    f"{variant}: {key} conflicts between {origins[key]} and {label}"
                )
            merged[key] = value
            origins[key] = label

    required = ("checkpoint_step", "checkpoint_sha256", "config_sha256")
    missing = [key for key in required if key not in merged]
    if missing:
        raise ValueError(f"{variant}: missing provenance fields {missing}")
    step = merged["checkpoint_step"]
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError(f"{variant}: checkpoint_step must be a nonnegative integer")
    for key in ("checkpoint_sha256", "config_sha256"):
        digest = merged[key]
        if (
            not isinstance(digest, str)
            or not re.fullmatch(r"[0-9a-fA-F]{64}", digest)
        ):
            raise ValueError(f"{variant}: {key} is not a SHA-256 digest")
        merged[key] = digest.lower()
    for path_key, digest_key in (
        ("checkpoint_path", "checkpoint_sha256"),
        ("config_path", "config_sha256"),
    ):
        if path_key not in merged:
            continue
        path = Path(merged[path_key]).expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"{variant}: {path_key} does not exist: {path}")
        actual = _sha256_file(path)
        if actual != merged[digest_key]:
            raise ValueError(
                f"{variant}: {path_key} hash {actual} != {merged[digest_key]}"
            )
        merged[path_key] = str(path)
    return merged


def _sidecar_payload(path: Path, variant: str) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: provenance sidecar must be a JSON object")
    if variant in payload and isinstance(payload[variant], dict):
        payload = payload[variant]
    if isinstance(payload.get("provenance"), dict):
        payload = payload["provenance"]
    sidecar_variant = payload.get("variant")
    if sidecar_variant is not None and sidecar_variant != variant:
        raise ValueError(
            f"{path}: sidecar variant {sidecar_variant!r} != {variant!r}"
        )
    return payload


def _parse_comparisons(
    specs: list[str], variants: set[str]
) -> list[tuple[str, str]]:
    result: list[tuple[str, str]] = []
    seen: set[str] = set()
    for spec in specs:
        if spec.count(":") != 1:
            raise ValueError(
                f"--compare must use METHOD:BASELINE syntax: {spec!r}"
            )
        method, baseline = spec.split(":")
        if method not in variants or baseline not in variants:
            raise ValueError(f"comparison references an unknown variant: {spec}")
        if method == baseline:
            raise ValueError(f"comparison cannot use one variant twice: {spec}")
        name = f"{method}_vs_{baseline}"
        if name in seen:
            raise ValueError(f"duplicate comparison: {name}")
        seen.add(name)
        result.append((method, baseline))
    if not result:
        raise ValueError("generic comparison requires at least one --compare")
    return result


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


def _run_legacy(args: argparse.Namespace) -> dict[str, Any]:
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


def _reject_unknown_names(
    label: str, values: dict[str, Any], variants: set[str]
) -> None:
    unknown = sorted(set(values) - variants)
    if unknown:
        raise ValueError(f"{label} references unknown variants: {unknown}")


def _run_generic(args: argparse.Namespace) -> dict[str, Any]:
    summaries = _named_values(
        list(getattr(args, "summary", [])),
        label="--summary",
        convert=lambda raw: Path(raw).expanduser().resolve(),
    )
    if len(summaries) < 2:
        raise ValueError("generic comparison requires at least two summaries")
    for name, path in summaries.items():
        if not path.is_file():
            raise ValueError(f"{name}: summary does not exist: {path}")
    if len(set(summaries.values())) != len(summaries):
        raise ValueError("each variant must use a distinct summary path")

    variants = set(summaries)
    sidecars = _named_values(
        list(getattr(args, "provenance_sidecar", [])),
        label="--provenance-sidecar",
        convert=lambda raw: Path(raw).expanduser().resolve(),
    )
    steps = _named_values(
        list(getattr(args, "variant_checkpoint_step", [])),
        label="--variant-checkpoint-step",
        convert=int,
    )
    checkpoint_paths = _named_values(
        list(getattr(args, "checkpoint_path", [])),
        label="--checkpoint-path",
    )
    checkpoint_hashes = _named_values(
        list(getattr(args, "checkpoint_sha256", [])),
        label="--checkpoint-sha256",
    )
    config_paths = _named_values(
        list(getattr(args, "config_path", [])),
        label="--config-path",
    )
    config_hashes = _named_values(
        list(getattr(args, "config_sha256", [])),
        label="--config-sha256",
    )
    named_inputs = {
        "--provenance-sidecar": sidecars,
        "--variant-checkpoint-step": steps,
        "--checkpoint-path": checkpoint_paths,
        "--checkpoint-sha256": checkpoint_hashes,
        "--config-path": config_paths,
        "--config-sha256": config_hashes,
    }
    for label, values in named_inputs.items():
        _reject_unknown_names(label, values, variants)

    seed_start = getattr(args, "seed_start", None)
    seed_stop = getattr(args, "seed_stop_exclusive", None)
    if (
        isinstance(seed_start, bool)
        or not isinstance(seed_start, int)
        or isinstance(seed_stop, bool)
        or not isinstance(seed_stop, int)
        or seed_stop <= seed_start
    ):
        raise ValueError("generic mode requires a valid explicit seed range")
    expected_protocol = _protocol_values(list(getattr(args, "protocol", [])))
    if not expected_protocol:
        raise ValueError("generic mode requires explicit protocol constraints")
    missing_protocol = [
        key for key in GENERIC_REQUIRED_PROTOCOL if key not in expected_protocol
    ]
    if missing_protocol:
        raise ValueError(
            f"generic mode is missing required protocol fields: {missing_protocol}"
        )
    comparisons_to_run = _parse_comparisons(
        list(getattr(args, "compare", [])), variants
    )

    reports: dict[str, dict[str, Any]] = {}
    for variant, summary_path in summaries.items():
        payload = json.loads(summary_path.read_text())
        if not isinstance(payload, dict):
            raise ValueError(f"{summary_path}: summary must be a JSON object")
        _validate_protocol(
            payload,
            path=summary_path,
            expected_protocol=expected_protocol,
        )
        outcomes = _validated_outcomes(
            payload,
            path=summary_path,
            variant=variant,
            seed_start=seed_start,
            seed_stop_exclusive=seed_stop,
        )
        provenance_sources: list[tuple[str, dict[str, Any]]] = []
        embedded = payload.get("provenance")
        if isinstance(embedded, dict):
            provenance_sources.append((f"{variant} summary", embedded))
        sidecar_record: dict[str, Any] | None = None
        if variant in sidecars:
            sidecar_path = sidecars[variant]
            if not sidecar_path.is_file():
                raise ValueError(
                    f"{variant}: provenance sidecar does not exist: {sidecar_path}"
                )
            sidecar_record = {
                "path": str(sidecar_path),
                "sha256": _sha256_file(sidecar_path),
            }
            provenance_sources.append(
                (
                    f"{variant} sidecar {sidecar_path}",
                    _sidecar_payload(sidecar_path, variant),
                )
            )
        cli_provenance = {
            key: values[variant]
            for key, values in (
                ("checkpoint_step", steps),
                ("checkpoint_path", checkpoint_paths),
                ("checkpoint_sha256", checkpoint_hashes),
                ("config_path", config_paths),
                ("config_sha256", config_hashes),
            )
            if variant in values
        }
        if cli_provenance:
            provenance_sources.append((f"{variant} CLI", cli_provenance))
        provenance = _merge_provenance(variant, provenance_sources)
        reports[variant] = {
            "outcomes": outcomes,
            "summary_path": str(summary_path),
            "summary_sha256": _sha256_file(summary_path),
            "provenance": provenance,
            "provenance_sidecar": sidecar_record,
        }

    bootstrap_samples = getattr(args, "bootstrap_samples", 100_000)
    bootstrap_seed = getattr(args, "bootstrap_seed", 20260719)
    comparisons = {
        f"{method}_vs_{baseline}": _paired_comparison(
            reports[baseline]["outcomes"],
            reports[method]["outcomes"],
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
        )
        for method, baseline in comparisons_to_run
    }
    success_counts = {
        variant: sum(report["outcomes"].values())
        for variant, report in reports.items()
    }
    episodes = seed_stop - seed_start
    return {
        "schema_version": "fitwam_paired_rollout_comparison_v2",
        "status": "valid",
        "evaluation_scope": "explicit_seed_range_paired_rollout_evaluation",
        "protocol": {
            "expected": expected_protocol,
            "seed_start": seed_start,
            "seed_stop_exclusive": seed_stop,
            "episodes": episodes,
            "bootstrap_samples": bootstrap_samples,
            "bootstrap_seed": bootstrap_seed,
            "mcnemar": "exact_two_sided",
        },
        "variants": {
            variant: {
                key: value
                for key, value in report.items()
                if key != "outcomes"
            }
            for variant, report in reports.items()
        },
        "success_counts": success_counts,
        "success_rates": {
            variant: count / episodes
            for variant, count in success_counts.items()
        },
        "comparisons": comparisons,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if getattr(args, "summary", None):
        return _run_generic(args)
    return _run_legacy(args)


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


def _generic_output_paths(args: argparse.Namespace) -> dict[str, Path]:
    output_dir = args.output_dir.expanduser().resolve()
    prefix = args.output_prefix
    if not VARIANT_NAME_PATTERN.fullmatch(prefix):
        raise ValueError("output_prefix must be a filename stem")
    return {
        suffix: output_dir / f"{prefix}.{suffix}"
        for suffix in ("json", "csv", "md")
    }


def _write_generic_outputs(
    args: argparse.Namespace, result: dict[str, Any]
) -> dict[str, Path]:
    paths = _generic_output_paths(args)
    paths["json"].parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(
        paths["json"], json.dumps(result, indent=2, sort_keys=True) + "\n"
    )

    csv_path = paths["csv"].with_name(f".{paths['csv'].name}.tmp-{os.getpid()}")
    try:
        with csv_path.open("w", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=(
                    "comparison",
                    "episodes",
                    "baseline_successes",
                    "method_successes",
                    "success_delta_count",
                    "success_delta",
                    "ci95_low",
                    "ci95_high",
                    "method_only_success",
                    "baseline_only_success",
                    "mcnemar_exact_two_sided_p",
                ),
            )
            writer.writeheader()
            for name, row in result["comparisons"].items():
                writer.writerow(
                    {
                        "comparison": name,
                        "episodes": row["episodes"],
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
                        "mcnemar_exact_two_sided_p": row[
                            "mcnemar_exact_two_sided_p"
                        ],
                    }
                )
        os.replace(csv_path, paths["csv"])
    finally:
        csv_path.unlink(missing_ok=True)

    episodes = result["protocol"]["episodes"]
    lines = [
        "# Paired Rollout Comparison",
        "",
        "| Variant | Checkpoint step | Successes | Success rate |",
        "| --- | ---: | ---: | ---: |",
    ]
    for variant, row in result["variants"].items():
        count = result["success_counts"][variant]
        rate = result["success_rates"][variant]
        step = row["provenance"]["checkpoint_step"]
        lines.append(f"| {variant} | {step} | {count}/{episodes} | {rate:.1%} |")
    lines.extend(
        [
            "",
            "| Comparison | Delta | 95% paired bootstrap CI | Exact McNemar p |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for name, row in result["comparisons"].items():
        low, high = row["bootstrap_ci_95"]
        lines.append(
            f"| {name} | {row['success_delta']:+.1%} | "
            f"[{low:+.1%}, {high:+.1%}] | "
            f"{row['mcnemar_exact_two_sided_p']:.6g} |"
        )
    lines.append("")
    _atomic_write_text(paths["md"], "\n".join(lines))
    return paths


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
    if args.summary:
        paths = _write_generic_outputs(args, result)
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "comparisons": sorted(result["comparisons"]),
                    "outputs": {key: str(value) for key, value in paths.items()},
                },
                sort_keys=True,
            )
        )
        return 0
    _write_outputs(args, result)
    print(json.dumps(result["primary_gate"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
