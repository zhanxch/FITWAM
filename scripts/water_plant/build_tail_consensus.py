#!/usr/bin/env python3
"""Build conservative cutoff consensus from independent tail audits.

This utility reads comparator ``episodes.csv`` and ``summary.json`` pairs. It
does not read, build, or modify an EveRobot manifest.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


REPORT_NAME = "tail_consensus_report.json"
CUTOFFS_NAME = "tail_consensus_cutoffs.jsonl"
REQUIRED_EPISODE_FIELDS = {
    "dataset_id",
    "episode_index",
    "num_frames",
    "cutoff_frame",
    "should_trim",
    "material",
}


@dataclass(frozen=True)
class InputSpec:
    label: str
    expected_periodic_window_frames: int
    episodes_csv: Path
    summary_json: Path


@dataclass(frozen=True)
class EpisodeDecision:
    num_frames: int
    cutoff_frame: int
    should_trim: bool
    material: bool


@dataclass(frozen=True)
class LoadedInput:
    spec: InputSpec
    manifest_sha256: str
    episodes_sha256: str
    summary_sha256: str
    summary_materiality: dict[str, Any]
    episodes: dict[tuple[str, int], EpisodeDecision]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _strict_int(value: Any, *, field: str, context: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{context}: {field} must be an integer")
    text = str(value).strip()
    try:
        parsed = int(text)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{context}: {field} must be an integer, got {value!r}"
        ) from error
    if str(parsed) != text and text != f"+{parsed}":
        raise ValueError(f"{context}: {field} must use integer syntax, got {value!r}")
    return parsed


def _strict_bool(value: Any, *, field: str, context: str) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"{context}: {field} must be true or false, got {value!r}")


def _require_sha256(value: Any, *, field: str, context: str) -> str:
    digest = str(value).strip().lower()
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError(f"{context}: {field} must be a 64-character SHA256 digest")
    return digest


def _load_json_object(path: Path, *, context: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"{context}: file does not exist: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"{context}: invalid JSON: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{context}: JSON root must be an object")
    return payload


def _load_episodes(path: Path, *, label: str) -> dict[tuple[str, int], EpisodeDecision]:
    context = f"input {label!r} episodes.csv"
    try:
        stream = path.open(newline="", encoding="utf-8")
    except FileNotFoundError as error:
        raise ValueError(f"{context}: file does not exist: {path}") from error
    with stream:
        reader = csv.DictReader(stream)
        fields = set(reader.fieldnames or ())
        missing = sorted(REQUIRED_EPISODE_FIELDS - fields)
        if missing:
            raise ValueError(f"{context}: missing required columns: {missing}")
        episodes: dict[tuple[str, int], EpisodeDecision] = {}
        for line_number, row in enumerate(reader, start=2):
            row_context = f"{context} line {line_number}"
            dataset_id = str(row["dataset_id"]).strip()
            if not dataset_id:
                raise ValueError(f"{row_context}: dataset_id must be non-empty")
            episode_index = _strict_int(
                row["episode_index"], field="episode_index", context=row_context
            )
            num_frames = _strict_int(
                row["num_frames"], field="num_frames", context=row_context
            )
            cutoff_frame = _strict_int(
                row["cutoff_frame"], field="cutoff_frame", context=row_context
            )
            should_trim = _strict_bool(
                row["should_trim"], field="should_trim", context=row_context
            )
            material = _strict_bool(
                row["material"], field="material", context=row_context
            )
            if episode_index < 0:
                raise ValueError(f"{row_context}: episode_index must be non-negative")
            if num_frames <= 0:
                raise ValueError(f"{row_context}: num_frames must be positive")
            if not 0 <= cutoff_frame <= num_frames:
                raise ValueError(
                    f"{row_context}: cutoff_frame must be in [0, num_frames]"
                )
            if should_trim and cutoff_frame >= num_frames:
                raise ValueError(
                    f"{row_context}: should_trim=true requires cutoff_frame < num_frames"
                )
            if not should_trim and cutoff_frame != num_frames:
                raise ValueError(
                    f"{row_context}: should_trim=false requires cutoff_frame == num_frames"
                )
            key = (dataset_id, episode_index)
            if key in episodes:
                raise ValueError(f"{row_context}: duplicate episode key {key!r}")
            episodes[key] = EpisodeDecision(
                num_frames=num_frames,
                cutoff_frame=cutoff_frame,
                should_trim=should_trim,
                material=material,
            )
    if not episodes:
        raise ValueError(f"{context}: no episode rows")
    return episodes


def _load_input(spec: InputSpec) -> LoadedInput:
    context = f"input {spec.label!r} summary.json"
    if spec.expected_periodic_window_frames <= 0:
        raise ValueError(
            f"input {spec.label!r}: expected_periodic_window_frames must be positive"
        )
    summary = _load_json_object(spec.summary_json, context=context)
    if summary.get("status") != "ok":
        raise ValueError(f"{context}: status must be 'ok'")
    manifest_sha256 = _require_sha256(
        summary.get("manifest_sha256"), field="manifest_sha256", context=context
    )
    tail_config = summary.get("tail_config")
    if not isinstance(tail_config, dict):
        raise ValueError(f"{context}: tail_config must be an object")
    observed_window = _strict_int(
        tail_config.get("periodic_window_frames"),
        field="tail_config.periodic_window_frames",
        context=context,
    )
    if observed_window != spec.expected_periodic_window_frames:
        raise ValueError(
            f"{context}: periodic_window_frames={observed_window}, expected "
            f"{spec.expected_periodic_window_frames}"
        )
    episodes = _load_episodes(spec.episodes_csv, label=spec.label)
    reported_count = _strict_int(
        summary.get("failure_episodes"), field="failure_episodes", context=context
    )
    if reported_count != len(episodes):
        raise ValueError(
            f"{context}: failure_episodes={reported_count}, but episodes.csv has "
            f"{len(episodes)} rows"
        )
    materiality = summary.get("materiality")
    if not isinstance(materiality, dict):
        raise ValueError(f"{context}: materiality must be an object")
    reported_trimmed = _strict_int(
        materiality.get("episodes_trimmed"),
        field="materiality.episodes_trimmed",
        context=context,
    )
    reported_material = _strict_int(
        materiality.get("material_episodes"),
        field="materiality.material_episodes",
        context=context,
    )
    actual_trimmed = sum(decision.should_trim for decision in episodes.values())
    actual_material = sum(decision.material for decision in episodes.values())
    if reported_trimmed != actual_trimmed:
        raise ValueError(
            f"{context}: episodes_trimmed={reported_trimmed}, expected {actual_trimmed}"
        )
    if reported_material != actual_material:
        raise ValueError(
            f"{context}: material_episodes={reported_material}, expected {actual_material}"
        )
    return LoadedInput(
        spec=spec,
        manifest_sha256=manifest_sha256,
        episodes_sha256=_sha256_file(spec.episodes_csv),
        summary_sha256=_sha256_file(spec.summary_json),
        summary_materiality=materiality,
        episodes=episodes,
    )


def _validate_inputs(specs: Sequence[InputSpec]) -> list[LoadedInput]:
    if len(specs) < 3:
        raise ValueError(f"at least three inputs are required, got {len(specs)}")
    labels = [spec.label for spec in specs]
    if any(not label.strip() for label in labels):
        raise ValueError("input labels must be non-empty")
    if len(set(labels)) != len(labels):
        raise ValueError("input labels must be unique")

    loaded = sorted(
        (_load_input(spec) for spec in specs),
        key=lambda item: (
            item.spec.expected_periodic_window_frames,
            item.spec.label,
        ),
    )
    manifest_hashes = {item.manifest_sha256 for item in loaded}
    if len(manifest_hashes) != 1:
        details = ", ".join(
            f"{item.spec.label}={item.manifest_sha256}" for item in loaded
        )
        raise ValueError(f"manifest SHA mismatch: {details}")

    reference = loaded[0]
    reference_keys = set(reference.episodes)
    for item in loaded[1:]:
        keys = set(item.episodes)
        if keys != reference_keys:
            missing = sorted(reference_keys - keys)
            extra = sorted(keys - reference_keys)
            raise ValueError(
                f"episode set mismatch for {item.spec.label!r}: "
                f"missing={missing[:5]!r}, extra={extra[:5]!r}"
            )
        for key in sorted(reference_keys):
            expected_frames = reference.episodes[key].num_frames
            observed_frames = item.episodes[key].num_frames
            if observed_frames != expected_frames:
                raise ValueError(
                    f"num_frames mismatch for episode {key!r}: "
                    f"{reference.spec.label}={expected_frames}, "
                    f"{item.spec.label}={observed_frames}"
                )
    return loaded


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def build_consensus(
    specs: Sequence[InputSpec],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    loaded = _validate_inputs(specs)
    manifest_sha256 = loaded[0].manifest_sha256
    keys = sorted(loaded[0].episodes)
    records: list[dict[str, Any]] = []

    for dataset_id, episode_index in keys:
        key = (dataset_id, episode_index)
        num_frames = loaded[0].episodes[key].num_frames
        decisions = [item.episodes[key] for item in loaded]
        should_trim = all(decision.should_trim for decision in decisions)
        cutoff_frame = (
            max(decision.cutoff_frame for decision in decisions)
            if should_trim
            else num_frames
        )
        decision_disagreement = (
            len({decision.should_trim for decision in decisions}) > 1
        )
        cutoff_disagreement = (
            len({decision.cutoff_frame for decision in decisions}) > 1
        )
        materiality_disagreement = (
            len({decision.material for decision in decisions}) > 1
        )
        instability_flags = []
        if decision_disagreement:
            instability_flags.append("decision_disagreement")
        if cutoff_disagreement:
            instability_flags.append("cutoff_disagreement")
        if materiality_disagreement:
            instability_flags.append("materiality_disagreement")
        per_input = [
            {
                "label": item.spec.label,
                "periodic_window_frames": item.spec.expected_periodic_window_frames,
                "should_trim": item.episodes[key].should_trim,
                "cutoff_frame": item.episodes[key].cutoff_frame,
                "material": item.episodes[key].material,
            }
            for item in loaded
        ]
        records.append(
            {
                "dataset_id": dataset_id,
                "episode_index": episode_index,
                "num_frames": num_frames,
                "should_trim": should_trim,
                "cutoff_frame": cutoff_frame,
                "dropped_frames": num_frames - cutoff_frame,
                "trimmed_fraction": (num_frames - cutoff_frame) / num_frames,
                "consensus_material": should_trim
                and all(decision.material for decision in decisions),
                "all_inputs_material": all(decision.material for decision in decisions),
                "any_input_material": any(decision.material for decision in decisions),
                "cutoff_spread_frames": max(
                    decision.cutoff_frame for decision in decisions
                )
                - min(decision.cutoff_frame for decision in decisions),
                "instability_flags": instability_flags,
                "per_input": per_input,
            }
        )

    episode_count = len(records)
    consensus_trimmed = sum(record["should_trim"] for record in records)
    unanimous_no_trim = sum(
        all(
            not item.episodes[
                (record["dataset_id"], record["episode_index"])
            ].should_trim
            for item in loaded
        )
        for record in records
    )
    decision_disagreements = sum(
        "decision_disagreement" in record["instability_flags"] for record in records
    )
    cutoff_disagreements = sum(
        "cutoff_disagreement" in record["instability_flags"] for record in records
    )
    materiality_disagreements = sum(
        "materiality_disagreement" in record["instability_flags"] for record in records
    )
    total_frames = sum(record["num_frames"] for record in records)
    dropped_frames = sum(record["dropped_frames"] for record in records)
    aggregate = {
        "inputs": len(loaded),
        "episodes": episode_count,
        "agreement": {
            "all_decisions_agree_episodes": episode_count - decision_disagreements,
            "all_decisions_agree_fraction": _ratio(
                episode_count - decision_disagreements, episode_count
            ),
            "unanimous_trim_episodes": consensus_trimmed,
            "unanimous_no_trim_episodes": unanimous_no_trim,
            "decision_disagreement_episodes": decision_disagreements,
            "exact_cutoff_agreement_episodes": episode_count - cutoff_disagreements,
            "exact_cutoff_agreement_fraction": _ratio(
                episode_count - cutoff_disagreements, episode_count
            ),
            "materiality_disagreement_episodes": materiality_disagreements,
            "unstable_episodes": sum(
                bool(record["instability_flags"]) for record in records
            ),
            "max_cutoff_spread_frames": max(
                (record["cutoff_spread_frames"] for record in records), default=0
            ),
        },
        "materiality": {
            "consensus_trim_episodes": consensus_trimmed,
            "consensus_trim_fraction": _ratio(consensus_trimmed, episode_count),
            "consensus_material_episodes": sum(
                record["consensus_material"] for record in records
            ),
            "all_inputs_material_episodes": sum(
                record["all_inputs_material"] for record in records
            ),
            "any_input_material_episodes": sum(
                record["any_input_material"] for record in records
            ),
            "total_frames": total_frames,
            "dropped_frames": dropped_frames,
            "dropped_frame_fraction": _ratio(dropped_frames, total_frames),
        },
    }
    report = {
        "format": "FITWAMTailConsensusReport",
        "schema_version": "1.0",
        "status": "ok",
        "rule": {
            "trim_condition": "all_inputs_should_trim",
            "trim_cutoff": "max_input_cutoff_frame",
            "no_trim_cutoff": "num_frames",
        },
        "manifest_sha256": manifest_sha256,
        "expected_periodic_window_frames": [
            item.spec.expected_periodic_window_frames for item in loaded
        ],
        "inputs": [
            {
                "label": item.spec.label,
                "expected_periodic_window_frames": item.spec.expected_periodic_window_frames,
                "observed_periodic_window_frames": item.spec.expected_periodic_window_frames,
                "episodes_csv_name": item.spec.episodes_csv.name,
                "episodes_csv_sha256": item.episodes_sha256,
                "summary_json_name": item.spec.summary_json.name,
                "summary_json_sha256": item.summary_sha256,
                "reported_materiality": item.summary_materiality,
            }
            for item in loaded
        ],
        "generator_sha256": _sha256_file(Path(__file__).resolve()),
        "aggregate": aggregate,
    }
    return report, records


def _json_bytes(payload: Any, *, indent: int | None = None) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=True,
            indent=indent,
            sort_keys=True,
            separators=(",", ":") if indent is None else None,
        )
        + "\n"
    ).encode("utf-8")


def write_outputs(
    output_dir: Path, report: dict[str, Any], records: Sequence[dict[str, Any]]
) -> tuple[Path, Path]:
    cutoff_bytes = b"".join(_json_bytes(record) for record in records)
    final_report = dict(report)
    final_report["outputs"] = {
        "cutoff_records_file": CUTOFFS_NAME,
        "cutoff_records_sha256": _sha256_bytes(cutoff_bytes),
        "cutoff_records_count": len(records),
    }
    report_bytes = _json_bytes(final_report, indent=2)

    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cutoffs_path = output_dir / CUTOFFS_NAME
    report_path = output_dir / REPORT_NAME
    cutoffs_temp = output_dir / f".{CUTOFFS_NAME}.tmp"
    report_temp = output_dir / f".{REPORT_NAME}.tmp"
    cutoffs_temp.write_bytes(cutoff_bytes)
    report_temp.write_bytes(report_bytes)
    cutoffs_temp.replace(cutoffs_path)
    report_temp.replace(report_path)
    return report_path, cutoffs_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        action="append",
        nargs=4,
        required=True,
        metavar=("LABEL", "EXPECTED_WINDOW", "EPISODES_CSV", "SUMMARY_JSON"),
        help=(
            "Comparator input. Repeat at least three times. EXPECTED_WINDOW is "
            "validated against summary.tail_config.periodic_window_frames."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help=f"Writes {REPORT_NAME} and {CUTOFFS_NAME} here.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        specs = [
            InputSpec(
                label=values[0],
                expected_periodic_window_frames=_strict_int(
                    values[1], field="EXPECTED_WINDOW", context=f"input {values[0]!r}"
                ),
                episodes_csv=Path(values[2]).expanduser().resolve(),
                summary_json=Path(values[3]).expanduser().resolve(),
            )
            for values in args.input
        ]
        report, records = build_consensus(specs)
        report_path, cutoffs_path = write_outputs(args.output_dir, report, records)
    except (OSError, ValueError) as error:
        raise SystemExit(f"tail consensus failed: {error}") from error
    print(f"report={report_path}")
    print(f"cutoffs={cutoffs_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
