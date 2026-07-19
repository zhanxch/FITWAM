#!/usr/bin/env python3
"""Attach event-pair references to an EveRobot v0.2 training manifest."""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
for source_root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from fastwam.everobot_schema import (  # noqa: E402
    SCHEMA_VERSION,
    compute_manifest_hash,
    validate_manifest,
)

PAIR_TARGET_MODULE_PATH = (
    PROJECT_ROOT / "src" / "fastwam" / "datasets" / "eve" / "pair_targets.py"
)
PAIR_TARGET_MODULE_NAME = "_fastwam_pair_targets_standalone"
pair_target_spec = importlib.util.spec_from_file_location(
    PAIR_TARGET_MODULE_NAME, PAIR_TARGET_MODULE_PATH
)
if pair_target_spec is None or pair_target_spec.loader is None:
    raise ImportError(f"Cannot load pair-target module: {PAIR_TARGET_MODULE_PATH}")
pair_target_module = importlib.util.module_from_spec(pair_target_spec)
sys.modules[PAIR_TARGET_MODULE_NAME] = pair_target_module
pair_target_spec.loader.exec_module(pair_target_module)
PairTargetStore = pair_target_module.PairTargetStore


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} must contain a JSON object")
            rows.append(row)
    return rows


def write_json_atomic_new(path: Path, payload: Mapping[str, Any]) -> None:
    """Create ``path`` atomically without replacing an existing output."""

    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {path}")
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing output: {path}")
        os.link(temporary_path, path)
        temporary_path.unlink()
    finally:
        temporary_path.unlink(missing_ok=True)


def _nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _pair_weight(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number in [0, 1]")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{label} must be a finite number in [0, 1]")
    return result


def _sample_split(sample: Mapping[str, Any]) -> str:
    return _nonempty_string(sample.get("split", "train"), "sample.split")


def attach_pairs(
    manifest: Mapping[str, Any],
    pair_rows: Sequence[Mapping[str, Any]],
    targets: PairTargetStore,
    *,
    attach_side: str = "both",
) -> dict[str, Any]:
    """Return a validated copy with external target references.

    ``failure`` attaches pair supervision only to the failure event. This is
    the controlled Offline Steer protocol: the success event supplies the
    frozen Teacher target but is not duplicated in the Fast-WAM manifest.
    ``both`` retains the stricter diagnostic mode that requires and annotates
    both event samples.
    """

    validate_manifest(manifest, strict=True)
    if str(manifest.get("schema_version")) != SCHEMA_VERSION:
        raise ValueError(f"Only EveRobot v{SCHEMA_VERSION} manifests are supported")
    if attach_side not in {"failure", "both"}:
        raise ValueError("attach_side must be `failure` or `both`")

    result = copy.deepcopy(dict(manifest))
    samples = result["samples"]
    event_samples: dict[str, dict[str, Any]] = {}
    for index, sample in enumerate(samples):
        if sample.get("sample_type") != "event":
            continue
        event_id = _nonempty_string(
            sample.get("event_id"), f"samples[{index}].event_id"
        )
        if event_id in event_samples:
            raise ValueError(f"Manifest contains duplicate event_id: {event_id}")
        if sample.get("pair_id") is not None or float(
            sample.get("pair_weight", 0.0)
        ) > 0.0:
            raise ValueError(
                f"Event {event_id} already has pair metadata; attach to a clean manifest"
            )
        event_samples[event_id] = sample

    seen_pair_ids: set[str] = set()
    pair_by_event: dict[str, str] = {}
    selected: list[tuple[Mapping[str, Any], Any, tuple[str, ...]]] = []
    for row_index, row in enumerate(pair_rows):
        pair_id = _nonempty_string(row.get("pair_id"), f"pairs[{row_index}].pair_id")
        if pair_id in seen_pair_ids:
            raise ValueError(f"Duplicate pair_id in pair ledger: {pair_id}")
        seen_pair_ids.add(pair_id)
        success_event_id = _nonempty_string(
            row.get("success_event_id"), f"{pair_id}.success_event_id"
        )
        failure_event_id = _nonempty_string(
            row.get("failure_event_id"), f"{pair_id}.failure_event_id"
        )
        success_present = success_event_id in event_samples
        failure_present = failure_event_id in event_samples
        if attach_side == "both":
            if not success_present and not failure_present:
                continue
            if not success_present or not failure_present:
                missing = [
                    event_id
                    for event_id, exists in (
                        (success_event_id, success_present),
                        (failure_event_id, failure_present),
                    )
                    if not exists
                ]
                raise ValueError(
                    f"Pair {pair_id} is only partially represented in the manifest; "
                    f"missing event samples: {missing}"
                )
            event_ids_to_attach = (success_event_id, failure_event_id)
        else:
            if not failure_present:
                continue
            event_ids_to_attach = (failure_event_id,)

        for event_id in event_ids_to_attach:
            previous = pair_by_event.get(event_id)
            if previous is not None:
                raise ValueError(
                    f"Event {event_id} is reused by pairs {previous} and {pair_id}; "
                    "first-round attachment requires one-to-one event pairs"
                )
            pair_by_event[event_id] = pair_id

        if pair_id not in targets:
            raise ValueError(f"Pair {pair_id} has no exported target")
        target = targets.get(pair_id)
        if (
            target.success_event_id != success_event_id
            or target.failure_event_id != failure_event_id
        ):
            raise ValueError(f"Pair {pair_id} target event IDs do not match its ledger")
        ledger_weight = _pair_weight(
            row.get("pair_weight", 1.0), f"{pair_id}.pair_weight"
        )
        if not math.isclose(
            ledger_weight, target.pair_weight, rel_tol=1e-6, abs_tol=1e-7
        ):
            raise ValueError(
                f"Pair {pair_id} target weight does not match its pair ledger"
            )

        failure_sample = event_samples[failure_event_id]
        if failure_sample.get("episode_outcome") != "failure":
            raise ValueError(
                f"Pair {pair_id} failure event sample is not from a failure episode"
            )
        failure_split = _sample_split(failure_sample)
        if attach_side == "both":
            success_sample = event_samples[success_event_id]
            if success_sample.get("episode_outcome") != "success":
                raise ValueError(
                    f"Pair {pair_id} success event sample is not from a success episode"
                )
            success_split = _sample_split(success_sample)
            if success_split != failure_split:
                raise ValueError(
                    f"Pair {pair_id} event samples cross split boundaries"
                )
        if target.split != failure_split:
            raise ValueError(
                f"Pair {pair_id} target split {target.split!r} does not match "
                f"manifest split {failure_split!r}"
            )
        selected.append((row, target, event_ids_to_attach))

    if not selected:
        raise ValueError("No complete manifest pair matched the pair ledger")

    for row, target, event_ids_to_attach in selected:
        pair_id = str(row["pair_id"])
        for event_id in event_ids_to_attach:
            sample = event_samples[event_id]
            sample["pair_id"] = pair_id
            sample["pair_weight"] = float(target.pair_weight)

    result["manifest_hash"] = compute_manifest_hash(result)
    validate_manifest(result, strict=True)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pair-ledger", type=Path, required=True)
    parser.add_argument("--pair-targets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-teacher-sha256")
    parser.add_argument(
        "--attach-side",
        choices=["failure", "both"],
        default="failure",
        help=(
            "Attach targets only to failure observations for the controlled "
            "main method, or to both events for diagnostics."
        ),
    )
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> tuple[Path, int, str]:
    manifest_path = args.manifest.expanduser().resolve()
    pair_ledger_path = args.pair_ledger.expanduser().resolve()
    target_path = args.pair_targets.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_path}")
    manifest = read_json(manifest_path)
    pair_rows = read_jsonl(pair_ledger_path)
    with PairTargetStore(
        target_path,
        expected_teacher_sha256=args.expected_teacher_sha256,
    ) as targets:
        attached = attach_pairs(
            manifest,
            pair_rows,
            targets,
            attach_side=args.attach_side,
        )
        teacher_hash = targets.teacher_sha256
    write_json_atomic_new(output_path, attached)
    attached_count = sum(
        1
        for sample in attached["samples"]
        if sample.get("pair_id") is not None
    )
    return output_path, attached_count, teacher_hash


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_path, attached_count, teacher_hash = run(args)
    print(
        json.dumps(
            {
                "output": str(output_path),
                "attached_event_samples": attached_count,
                "teacher_sha256": teacher_hash,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
