#!/usr/bin/env python3
"""Match a success-control auxiliary pool to a reference manifest budget."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
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
    sha256_json,
    validate_manifest,
)


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
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            if not isinstance(row, dict):
                raise ValueError(
                    f"{path}:{line_number} must contain a JSON object"
                )
            rows.append(row)
    return rows


def _nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _sample_id(sample: Mapping[str, Any], label: str) -> str:
    return _nonempty_string(sample.get("sample_id"), f"{label}.sample_id")


def _sample_split(sample: Mapping[str, Any], label: str) -> str:
    return _nonempty_string(sample.get("split"), f"{label}.split")


def _sample_outcome(sample: Mapping[str, Any]) -> str:
    event_outcome = sample.get("event_outcome")
    if event_outcome in {"success", "failure"}:
        return str(event_outcome)
    return str(sample.get("episode_outcome") or "")


def _partition_manifest(
    manifest: Mapping[str, Any],
    *,
    label: str,
    expected_auxiliary_outcome: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    validate_manifest(manifest, strict=True, verify_hash=True)
    if str(manifest.get("schema_version")) != SCHEMA_VERSION:
        raise ValueError(
            f"{label} must use EveRobot manifest schema {SCHEMA_VERSION}"
        )

    primary: list[dict[str, Any]] = []
    auxiliary: list[dict[str, Any]] = []
    for index, raw_sample in enumerate(manifest["samples"]):
        sample = dict(raw_sample)
        role = sample.get("batch_role")
        if role == "primary":
            primary.append(sample)
            continue
        if role != "auxiliary":
            raise ValueError(
                f"{label}.samples[{index}].batch_role must be primary or auxiliary"
            )
        if sample.get("sample_type") != "event":
            raise ValueError(
                f"{label} auxiliary sample {_sample_id(sample, label)!r} "
                "must be an event"
            )
        if sample.get("window_selection") != "core_start_anchor":
            raise ValueError(
                f"{label} auxiliary sample {_sample_id(sample, label)!r} "
                "must use window_selection=core_start_anchor"
            )
        if _sample_outcome(sample) != expected_auxiliary_outcome:
            raise ValueError(
                f"{label} auxiliary sample {_sample_id(sample, label)!r} "
                f"must have {expected_auxiliary_outcome} outcome"
            )
        _sample_split(sample, f"{label}.samples[{index}]")
        auxiliary.append(sample)

    if not auxiliary:
        raise ValueError(f"{label} has no auxiliary event samples")
    return primary, auxiliary


def _episode_key(row: Mapping[str, Any], label: str) -> tuple[str, int]:
    dataset_id = _nonempty_string(row.get("dataset_id"), f"{label}.dataset_id")
    episode_index = row.get("episode_index")
    if (
        isinstance(episode_index, bool)
        or not isinstance(episode_index, int)
        or episode_index < 0
    ):
        raise ValueError(f"{label}.episode_index must be a non-negative integer")
    return dataset_id, episode_index


def _index_episode_lengths(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, int], tuple[int, str | None]]:
    indexed: dict[tuple[str, int], tuple[int, str | None]] = {}
    for index, row in enumerate(rows):
        label = f"episode_meta[{index}]"
        key = _episode_key(row, label)
        length = row.get("length")
        if (
            isinstance(length, bool)
            or not isinstance(length, int)
            or length <= 0
        ):
            raise ValueError(f"{label}.length must be a positive integer")
        split_value = row.get("split")
        split = (
            None
            if split_value is None
            else _nonempty_string(split_value, f"{label}.split")
        )
        value = (length, split)
        previous = indexed.get(key)
        if previous is not None and previous != value:
            raise ValueError(f"Conflicting episode metadata for {key}")
        if previous is not None:
            raise ValueError(f"Duplicate episode metadata for {key}")
        indexed[key] = value
    return indexed


def _progress_record(
    sample: Mapping[str, Any],
    *,
    label: str,
    episodes: Mapping[tuple[str, int], tuple[int, str | None]],
) -> dict[str, Any]:
    sample_id = _sample_id(sample, label)
    split = _sample_split(sample, label)
    key = _episode_key(sample, label)
    if key not in episodes:
        raise ValueError(
            f"{label} sample {sample_id!r} has no matching episode_meta row"
        )
    episode_length, episode_split = episodes[key]
    if episode_split is not None and episode_split != split:
        raise ValueError(
            f"{label} sample {sample_id!r} split {split!r} disagrees with "
            f"episode_meta split {episode_split!r}"
        )
    core_start = sample.get("core_start_frame")
    if (
        isinstance(core_start, bool)
        or not isinstance(core_start, int)
        or not 0 <= core_start < episode_length
    ):
        raise ValueError(
            f"{label} sample {sample_id!r} requires core_start_frame inside "
            f"its episode length {episode_length}"
        )
    progress = core_start / episode_length
    return {
        "sample": dict(sample),
        "sample_id": sample_id,
        "split": split,
        "episode_length": episode_length,
        "core_start_frame": core_start,
        "progress": progress,
    }


def _seeded_tie_key(seed: int, namespace: str, sample_id: str) -> str:
    return hashlib.sha256(
        f"{seed}:{namespace}:{sample_id}".encode("utf-8")
    ).hexdigest()


def _minimum_l1_subsequence_match(
    references: Sequence[Mapping[str, Any]],
    controls: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    split: str,
) -> list[dict[str, Any]]:
    if len(controls) < len(references):
        raise ValueError(
            f"Control auxiliary pool is insufficient for split {split!r}: "
            f"control={len(controls)}, reference={len(references)}"
        )

    ordered_references = sorted(
        references,
        key=lambda row: (
            float(row["progress"]),
            _seeded_tie_key(seed, f"reference:{split}", str(row["sample_id"])),
            str(row["sample_id"]),
        ),
    )
    ordered_controls = sorted(
        controls,
        key=lambda row: (
            float(row["progress"]),
            _seeded_tie_key(seed, f"control:{split}", str(row["sample_id"])),
            str(row["sample_id"]),
        ),
    )
    reference_count = len(ordered_references)
    control_count = len(ordered_controls)
    infinity = float("inf")
    costs = [
        [infinity] * (control_count + 1)
        for _ in range(reference_count + 1)
    ]
    matched = [
        [False] * (control_count + 1)
        for _ in range(reference_count + 1)
    ]
    for control_index in range(control_count + 1):
        costs[0][control_index] = 0.0

    tolerance = 1e-15
    for reference_index in range(1, reference_count + 1):
        for control_index in range(1, control_count + 1):
            skip_cost = costs[reference_index][control_index - 1]
            match_cost = (
                costs[reference_index - 1][control_index - 1]
                + abs(
                    float(ordered_references[reference_index - 1]["progress"])
                    - float(ordered_controls[control_index - 1]["progress"])
                )
            )
            if match_cost < skip_cost - tolerance:
                costs[reference_index][control_index] = match_cost
                matched[reference_index][control_index] = True
            else:
                # On an exact tie, retaining the earlier control subsequence is
                # the stable tie-break after seeded ordering.
                costs[reference_index][control_index] = skip_cost

    if not math.isfinite(costs[reference_count][control_count]):
        raise ValueError(f"Unable to match auxiliary samples for split {split!r}")

    pairs: list[dict[str, Any]] = []
    reference_index = reference_count
    control_index = control_count
    while reference_index:
        if control_index <= 0:
            raise RuntimeError("Auxiliary matching reconstruction failed")
        if matched[reference_index][control_index]:
            reference = ordered_references[reference_index - 1]
            control = ordered_controls[control_index - 1]
            pairs.append(
                {
                    "split": split,
                    "reference_sample_id": str(reference["sample_id"]),
                    "reference_progress": float(reference["progress"]),
                    "control_sample_id": str(control["sample_id"]),
                    "control_progress": float(control["progress"]),
                    "absolute_progress_distance": abs(
                        float(reference["progress"])
                        - float(control["progress"])
                    ),
                }
            )
            reference_index -= 1
            control_index -= 1
        else:
            control_index -= 1
    pairs.reverse()
    return pairs


def _decile_counts(progress_values: Sequence[float]) -> list[int]:
    counts = [0] * 10
    for progress in progress_values:
        if not 0.0 <= progress < 1.0:
            raise ValueError(f"Normalized progress must be in [0, 1), got {progress}")
        counts[min(int(progress * 10.0), 9)] += 1
    return counts


def _progress_diagnostics(matches: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    reference_progress = [
        float(match["reference_progress"]) for match in matches
    ]
    control_progress = [float(match["control_progress"]) for match in matches]
    distances = [
        float(match["absolute_progress_distance"]) for match in matches
    ]
    reference_deciles = _decile_counts(reference_progress)
    control_deciles = _decile_counts(control_progress)
    denominator = len(matches)
    max_fraction = max(
        abs(control - reference) / denominator
        for control, reference in zip(control_deciles, reference_deciles)
    )
    return {
        "decile_edges": [index / 10.0 for index in range(11)],
        "reference_counts": reference_deciles,
        "selected_control_counts": control_deciles,
        "max_abs_decile_fraction": max_fraction,
        "max_abs_decile_percentage_points": max_fraction * 100.0,
        "mean_absolute_distance": statistics.fmean(distances),
        "median_absolute_distance": statistics.median(distances),
    }


def match_auxiliary_budget(
    control_manifest: Mapping[str, Any],
    reference_manifest: Mapping[str, Any],
    episode_rows: Sequence[Mapping[str, Any]],
    *,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    primary, control_auxiliary = _partition_manifest(
        control_manifest,
        label="control_manifest",
        expected_auxiliary_outcome="success",
    )
    _, reference_auxiliary = _partition_manifest(
        reference_manifest,
        label="reference_manifest",
        expected_auxiliary_outcome="failure",
    )
    episodes = _index_episode_lengths(episode_rows)
    control_records = [
        _progress_record(
            sample,
            label=f"control_manifest.samples[{index}]",
            episodes=episodes,
        )
        for index, sample in enumerate(control_auxiliary)
    ]
    reference_records = [
        _progress_record(
            sample,
            label=f"reference_manifest.samples[{index}]",
            episodes=episodes,
        )
        for index, sample in enumerate(reference_auxiliary)
    ]

    control_by_split: dict[str, list[dict[str, Any]]] = {}
    reference_by_split: dict[str, list[dict[str, Any]]] = {}
    for record in control_records:
        control_by_split.setdefault(str(record["split"]), []).append(record)
    for record in reference_records:
        reference_by_split.setdefault(str(record["split"]), []).append(record)
    if set(control_by_split) != set(reference_by_split):
        raise ValueError(
            "Control and reference auxiliary split sets differ; refusing split "
            f"mixing: control={sorted(control_by_split)}, "
            f"reference={sorted(reference_by_split)}"
        )

    matches: list[dict[str, Any]] = []
    for split in sorted(reference_by_split):
        matches.extend(
            _minimum_l1_subsequence_match(
                reference_by_split[split],
                control_by_split[split],
                seed=seed,
                split=split,
            )
        )
    selected_ids = {str(match["control_sample_id"]) for match in matches}
    if len(selected_ids) != len(matches):
        raise RuntimeError("Auxiliary matching reused a control sample")
    if len(matches) != len(reference_auxiliary):
        raise RuntimeError("Auxiliary matching did not preserve reference budget")

    selected_samples = [
        copy.deepcopy(sample)
        for sample in control_auxiliary
        if str(sample["sample_id"]) in selected_ids
    ]
    if len(selected_samples) != len(selected_ids):
        raise RuntimeError("Selected auxiliary IDs do not map one-to-one to samples")

    selected_ids_ordered = sorted(selected_ids)
    selected_ids_hash = sha256_json(selected_ids_ordered)
    result = copy.deepcopy(dict(control_manifest))
    result["samples"] = [
        *[copy.deepcopy(sample) for sample in primary],
        *selected_samples,
    ]
    result["num_samples"] = len(result["samples"])
    result["source_round_ids"] = sorted(
        {str(sample["round_id"]) for sample in result["samples"]}
    )
    selection = copy.deepcopy(dict(result.get("selection", {})))
    selection["auxiliary_budget_match"] = {
        "method": "splitwise_ordered_minimum_l1",
        "control_manifest_hash": str(control_manifest["manifest_hash"]),
        "reference_manifest_hash": str(reference_manifest["manifest_hash"]),
        "seed": int(seed),
        "selected_auxiliary_count": len(selected_samples),
        "selected_sample_ids_sha256": selected_ids_hash,
    }
    result["selection"] = selection
    result["manifest_hash"] = compute_manifest_hash(result)
    validate_manifest(result, strict=True, verify_hash=True)

    counts_by_split = {
        split: {
            "control_available": len(control_by_split[split]),
            "reference_auxiliary": len(reference_by_split[split]),
            "selected_control_auxiliary": sum(
                match["split"] == split for match in matches
            ),
        }
        for split in sorted(reference_by_split)
    }
    diagnostics = {
        "format": "EveRobotAuxiliaryBudgetMatchDiagnostics",
        "schema_version": "0.1",
        "method": "splitwise_ordered_minimum_l1",
        "seed": int(seed),
        "control_manifest_hash": str(control_manifest["manifest_hash"]),
        "reference_manifest_hash": str(reference_manifest["manifest_hash"]),
        "output_manifest_hash": str(result["manifest_hash"]),
        "counts": {
            "primary_preserved": len(primary),
            "control_auxiliary_available": len(control_auxiliary),
            "reference_auxiliary": len(reference_auxiliary),
            "selected_control_auxiliary": len(selected_samples),
            "output_samples": len(result["samples"]),
            "by_split": counts_by_split,
        },
        "progress": _progress_diagnostics(matches),
        "selected_sample_ids_sha256": selected_ids_hash,
        "selected_sample_ids": selected_ids_ordered,
        "matches": sorted(
            matches,
            key=lambda match: (
                str(match["split"]),
                str(match["reference_sample_id"]),
                str(match["control_sample_id"]),
            ),
        ),
    }
    return result, diagnostics


def _serialize_json(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _prepare_temporary(path: Path, content: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    return temporary_path


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_outputs_atomically_new(
    output_path: Path,
    output: Mapping[str, Any],
    diagnostics_path: Path,
    diagnostics: Mapping[str, Any],
) -> None:
    output_path = output_path.expanduser().resolve()
    diagnostics_path = diagnostics_path.expanduser().resolve()
    if output_path == diagnostics_path:
        raise ValueError("output and diagnostics-output must be different paths")
    for path in (output_path, diagnostics_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing output: {path}")

    temporary_output: Path | None = None
    temporary_diagnostics: Path | None = None
    created: list[Path] = []
    try:
        temporary_output = _prepare_temporary(
            output_path, _serialize_json(output)
        )
        temporary_diagnostics = _prepare_temporary(
            diagnostics_path, _serialize_json(diagnostics)
        )
        for temporary, destination in (
            (temporary_output, output_path),
            (temporary_diagnostics, diagnostics_path),
        ):
            if destination.exists():
                raise FileExistsError(
                    f"Refusing to overwrite existing output: {destination}"
                )
            os.link(temporary, destination)
            created.append(destination)
            _fsync_directory(destination.parent)
    except Exception:
        for destination in reversed(created):
            destination.unlink(missing_ok=True)
            _fsync_directory(destination.parent)
        raise
    finally:
        if temporary_output is not None:
            temporary_output.unlink(missing_ok=True)
        if temporary_diagnostics is not None:
            temporary_diagnostics.unlink(missing_ok=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-manifest", type=Path, required=True)
    parser.add_argument("--reference-manifest", type=Path, required=True)
    parser.add_argument("--eve-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--diagnostics-output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> tuple[Path, Path, dict[str, Any]]:
    control_path = args.control_manifest.expanduser().resolve()
    reference_path = args.reference_manifest.expanduser().resolve()
    eve_root = args.eve_root.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    diagnostics_path = args.diagnostics_output.expanduser().resolve()
    if not eve_root.is_dir():
        raise FileNotFoundError(f"EveRobot root does not exist: {eve_root}")
    episode_meta_path = eve_root / "episode_meta.jsonl"
    if not episode_meta_path.is_file():
        raise FileNotFoundError(f"Missing EveRobot episode ledger: {episode_meta_path}")

    result, diagnostics = match_auxiliary_budget(
        read_json(control_path),
        read_json(reference_path),
        read_jsonl(episode_meta_path),
        seed=int(args.seed),
    )
    write_outputs_atomically_new(
        output_path,
        result,
        diagnostics_path,
        diagnostics,
    )
    return output_path, diagnostics_path, diagnostics


def main(argv: Sequence[str] | None = None) -> int:
    output_path, diagnostics_path, diagnostics = run(parse_args(argv))
    print(
        json.dumps(
            {
                "output": str(output_path),
                "diagnostics_output": str(diagnostics_path),
                "selected_auxiliary": diagnostics["counts"][
                    "selected_control_auxiliary"
                ],
                "selected_sample_ids_sha256": diagnostics[
                    "selected_sample_ids_sha256"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
