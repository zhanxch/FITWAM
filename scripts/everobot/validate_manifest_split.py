#!/usr/bin/env python3
"""Audit EveRobot manifests and detect episode leakage across data splits.

Example:
    python scripts/everobot/validate_manifest_split.py \
        train=eve/manifests/train.json \
        val=eve/manifests/val.json \
        --output eve/reports/train_val_audit.json
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import itertools
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fastwam.everobot_schema import (  # noqa: E402
    compute_manifest_hash,
    resolve_dataset_roots,
    validate_manifest,
)


def load_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path).expanduser().resolve()
    with manifest_path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"Manifest must be a JSON object: {manifest_path}")
    return payload


def effective_outcome(sample: Mapping[str, Any]) -> str:
    event_outcome = sample.get("event_outcome")
    if event_outcome in {None, "", "unknown"}:
        return str(sample.get("episode_outcome") or "success")
    return str(event_outcome)


def infer_batch_role(sample: Mapping[str, Any]) -> str:
    explicit_role = sample.get("batch_role")
    if explicit_role in {"primary", "auxiliary"}:
        return str(explicit_role)
    if explicit_role not in {None, ""}:
        return "invalid"
    if effective_outcome(sample) == "failure":
        return "auxiliary"
    return (
        "primary"
        if sample.get("action_loss", "enabled") == "enabled"
        else "auxiliary"
    )


def _counter_dict(values: Iterable[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def _episode_id(sample: Mapping[str, Any]) -> str | None:
    value = sample.get("episode_id")
    return value if isinstance(value, str) and value else None


def _dataset_episode(sample: Mapping[str, Any]) -> tuple[str, int] | None:
    dataset_id = sample.get("dataset_id")
    episode_index = sample.get("episode_index")
    if (
        not isinstance(dataset_id, str)
        or not dataset_id
        or isinstance(episode_index, bool)
        or not isinstance(episode_index, int)
        or episode_index < 0
    ):
        return None
    return dataset_id, episode_index


def audit_pair_fields(
    samples: Sequence[Mapping[str, Any]],
    *,
    require_complete_pairs: bool = True,
) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    groups: dict[str, list[tuple[int, Mapping[str, Any], float]]] = defaultdict(list)
    unpaired_samples = 0

    for index, sample in enumerate(samples):
        pair_id = sample.get("pair_id")
        pair_weight = sample.get("pair_weight")
        valid_id = isinstance(pair_id, str) and bool(pair_id.strip())
        valid_weight = (
            not isinstance(pair_weight, bool)
            and isinstance(pair_weight, (int, float))
            and math.isfinite(float(pair_weight))
            and 0.0 <= float(pair_weight) <= 1.0
        )

        if pair_id is None and pair_weight is None:
            unpaired_samples += 1
            continue
        if pair_id is not None and not valid_id:
            errors.append(f"samples[{index}] has an invalid pair_id")
        if pair_weight is not None and not valid_weight:
            errors.append(f"samples[{index}] has an invalid pair_weight")
        if not valid_id or not valid_weight:
            continue
        weight = float(pair_weight)
        if weight <= 0.0:
            errors.append(
                f"samples[{index}] pair_id={pair_id!r} requires positive pair_weight"
            )
            continue
        groups[str(pair_id)].append((index, sample, weight))

    complete = 0
    singleton: list[str] = []
    oversized: list[str] = []
    weight_mismatch: list[str] = []
    outcome_mismatch: list[str] = []
    for pair_id, members in sorted(groups.items()):
        if len(members) == 1:
            singleton.append(pair_id)
        elif len(members) > 2:
            oversized.append(pair_id)
        else:
            complete += 1
        weights = {member[2] for member in members}
        if len(weights) > 1:
            weight_mismatch.append(pair_id)
        outcomes = {effective_outcome(member[1]) for member in members}
        if len(members) == 2 and outcomes != {"success", "failure"}:
            outcome_mismatch.append(pair_id)

    if require_complete_pairs:
        for pair_id in singleton:
            errors.append(
                f"pair_id={pair_id!r} has one manifest member; expected two"
            )
        for pair_id in oversized:
            errors.append(f"pair_id={pair_id!r} has more than two manifest members")
    for pair_id in weight_mismatch:
        errors.append(f"pair_id={pair_id!r} has inconsistent pair_weight values")
    for pair_id in outcome_mismatch:
        errors.append(
            f"pair_id={pair_id!r} does not contain one success and one failure sample"
        )

    return (
        {
            "unpaired_samples": unpaired_samples,
            "paired_samples": sum(len(members) for members in groups.values()),
            "distinct_pair_ids": len(groups),
            "complete_pair_ids": complete,
            "singleton_pair_ids": singleton,
            "oversized_pair_ids": oversized,
            "weight_mismatch_pair_ids": weight_mismatch,
            "outcome_mismatch_pair_ids": outcome_mismatch,
        },
        errors,
    )


def audit_manifest(
    path: str | Path,
    *,
    split: str,
    check_dataset_roots: bool = False,
) -> tuple[dict[str, Any], list[str], list[str]]:
    manifest_path = Path(path).expanduser().resolve()
    errors: list[str] = []
    warnings: list[str] = []

    try:
        manifest = load_manifest(manifest_path)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return (
            {
                "split": split,
                "path": str(manifest_path),
                "loaded": False,
                "errors": [str(error)],
            },
            [f"{manifest_path}: {error}"],
            warnings,
        )

    samples_value = manifest.get("samples")
    samples = samples_value if isinstance(samples_value, list) else []
    declared_count = manifest.get("num_samples")
    count_matches = declared_count == len(samples)
    if not count_matches:
        errors.append(
            f"{manifest_path}: num_samples={declared_count!r} "
            f"does not equal len(samples)={len(samples)}"
        )

    stored_hash = manifest.get("manifest_hash")
    computed_hash: str | None = None
    hash_error: str | None = None
    try:
        computed_hash = compute_manifest_hash(manifest)
    except (TypeError, ValueError) as error:
        hash_error = str(error)
        errors.append(
            f"{manifest_path}: canonical manifest hash cannot be computed: {error}"
        )
    hash_matches = (
        computed_hash is not None
        and isinstance(stored_hash, str)
        and stored_hash.lower() == computed_hash.lower()
    )
    if computed_hash is not None and not hash_matches:
        errors.append(
            f"{manifest_path}: manifest_hash is missing or does not match "
            f"{computed_hash}"
        )

    schema_valid = True
    schema_error: str | None = None
    try:
        validate_manifest(manifest, strict=True, verify_hash=True)
    except (TypeError, ValueError) as error:
        schema_valid = False
        schema_error = str(error)
        errors.append(f"{manifest_path}: schema validation failed: {error}")

    action_counts = _counter_dict(
        str(sample.get("action_loss", "missing"))
        if isinstance(sample, Mapping)
        else "invalid_sample"
        for sample in samples
    )
    raw_role_counts = _counter_dict(
        str(sample.get("sample_role") or "missing")
        if isinstance(sample, Mapping)
        else "invalid_sample"
        for sample in samples
    )
    explicit_batch_role_counts = _counter_dict(
        str(sample.get("batch_role") or "missing")
        if isinstance(sample, Mapping)
        else "invalid_sample"
        for sample in samples
    )
    normalized_role_counts = _counter_dict(
        infer_batch_role(sample) if isinstance(sample, Mapping) else "invalid"
        for sample in samples
    )
    if normalized_role_counts.get("invalid", 0):
        errors.append(f"{manifest_path}: invalid batch_role value found")

    mapping_samples = [sample for sample in samples if isinstance(sample, Mapping)]
    pair_report, pair_errors = audit_pair_fields(
        mapping_samples,
        require_complete_pairs=False,
    )
    errors.extend(f"{manifest_path}: {error}" for error in pair_errors)

    missing_episode_id = sum(_episode_id(sample) is None for sample in mapping_samples)
    missing_dataset_episode = sum(
        _dataset_episode(sample) is None for sample in mapping_samples
    )
    if missing_episode_id:
        errors.append(
            f"{manifest_path}: {missing_episode_id} samples lack a valid episode_id"
        )
    if missing_dataset_episode:
        errors.append(
            f"{manifest_path}: {missing_dataset_episode} samples lack a valid "
            "(dataset_id, episode_index)"
        )

    dataset_root_report: dict[str, Any] = {
        "checked": bool(check_dataset_roots),
        "roots": {},
    }
    if check_dataset_roots:
        try:
            roots = resolve_dataset_roots(manifest)
            for dataset_id, root in roots.items():
                root_path = Path(root)
                exists = root_path.exists() and root_path.is_dir()
                dataset_root_report["roots"][dataset_id] = {
                    "path": str(root_path),
                    "exists": exists,
                }
                if not exists:
                    errors.append(
                        f"{manifest_path}: dataset root does not exist: "
                        f"{dataset_id}={root_path}"
                    )
        except ValueError as error:
            errors.append(f"{manifest_path}: dataset root resolution failed: {error}")

    report = {
        "split": split,
        "path": str(manifest_path),
        "loaded": True,
        "manifest_name": manifest.get("manifest_name"),
        "schema_version": manifest.get("schema_version"),
        "schema_valid": schema_valid,
        "schema_error": schema_error,
        "manifest_hash": {
            "stored": stored_hash,
            "computed": computed_hash,
            "matches": hash_matches,
            "error": hash_error,
        },
        "sample_count": {
            "declared": declared_count,
            "actual": len(samples),
            "matches": count_matches,
        },
        "action_loss_counts": action_counts,
        "sample_role_counts": raw_role_counts,
        "explicit_batch_role_counts": explicit_batch_role_counts,
        "normalized_batch_role_counts": normalized_role_counts,
        "pair_consistency": pair_report,
        "identity_counts": {
            "episode_id": len(
                {_episode_id(sample) for sample in mapping_samples}
                - {None}
            ),
            "dataset_episode": len(
                {_dataset_episode(sample) for sample in mapping_samples}
                - {None}
            ),
            "missing_episode_id_samples": missing_episode_id,
            "missing_dataset_episode_samples": missing_dataset_episode,
        },
        "dataset_roots": dataset_root_report,
        "errors": errors,
        "warnings": warnings,
    }
    return report, errors, warnings


def _identity_sort_key(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True)


def _serialize_identity(identity: Any) -> Any:
    return list(identity) if isinstance(identity, tuple) else identity


def _cross_split_overlaps(
    identities_by_split: Mapping[str, set[Any]],
    *,
    max_examples: int,
) -> list[dict[str, Any]]:
    overlaps: list[dict[str, Any]] = []
    for left, right in itertools.combinations(sorted(identities_by_split), 2):
        shared = sorted(
            identities_by_split[left] & identities_by_split[right],
            key=_identity_sort_key,
        )
        if shared:
            overlaps.append(
                {
                    "split_a": left,
                    "split_b": right,
                    "count": len(shared),
                    "examples": [
                        _serialize_identity(value)
                        for value in shared[:max_examples]
                    ],
                    "examples_truncated": len(shared) > max_examples,
                }
            )
    return overlaps


def audit_manifest_splits(
    manifests_by_split: Mapping[str, Sequence[str | Path]],
    *,
    check_dataset_roots: bool = False,
    max_overlap_examples: int = 100,
) -> dict[str, Any]:
    if len(manifests_by_split) < 2:
        raise ValueError("At least two distinct split names are required")
    if max_overlap_examples < 0:
        raise ValueError("max_overlap_examples must be non-negative")

    manifest_reports: list[dict[str, Any]] = []
    errors: list[str] = []
    warnings: list[str] = []
    episode_ids: dict[str, set[str]] = defaultdict(set)
    dataset_episodes: dict[str, set[tuple[str, int]]] = defaultdict(set)
    pair_ids: dict[str, set[str]] = defaultdict(set)
    pair_samples: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    within_split_counts: dict[str, dict[str, Counter[Any]]] = {}

    for split in sorted(manifests_by_split):
        paths = manifests_by_split[split]
        if not paths:
            errors.append(f"Split {split!r} has no manifests")
            continue
        id_counter: Counter[str] = Counter()
        key_counter: Counter[tuple[str, int]] = Counter()
        for path in paths:
            report, manifest_errors, manifest_warnings = audit_manifest(
                path,
                split=split,
                check_dataset_roots=check_dataset_roots,
            )
            manifest_reports.append(report)
            errors.extend(manifest_errors)
            warnings.extend(manifest_warnings)
            if not report.get("loaded"):
                continue
            manifest = load_manifest(path)
            for sample in manifest.get("samples", []):
                if not isinstance(sample, Mapping):
                    continue
                episode_id = _episode_id(sample)
                dataset_episode = _dataset_episode(sample)
                if episode_id is not None:
                    episode_ids[split].add(episode_id)
                    id_counter[episode_id] += 1
                if dataset_episode is not None:
                    dataset_episodes[split].add(dataset_episode)
                    key_counter[dataset_episode] += 1
                pair_id = sample.get("pair_id")
                if isinstance(pair_id, str) and pair_id:
                    pair_ids[split].add(pair_id)
                pair_samples[split].append(sample)
        within_split_counts[split] = {
            "episode_id": id_counter,
            "dataset_episode": key_counter,
        }

    episode_overlaps = _cross_split_overlaps(
        episode_ids, max_examples=max_overlap_examples
    )
    dataset_episode_overlaps = _cross_split_overlaps(
        dataset_episodes, max_examples=max_overlap_examples
    )
    pair_overlaps = _cross_split_overlaps(
        pair_ids, max_examples=max_overlap_examples
    )
    overlap_count = sum(item["count"] for item in episode_overlaps) + sum(
        item["count"] for item in dataset_episode_overlaps
    )
    if overlap_count:
        errors.append(
            "Cross-split episode leakage detected by episode_id and/or "
            "(dataset_id, episode_index)"
        )
    if pair_overlaps:
        errors.append("pair_id values cross split boundaries")

    split_pair_consistency: dict[str, dict[str, Any]] = {}
    for split in sorted(manifests_by_split):
        pair_report, pair_errors = audit_pair_fields(
            pair_samples[split],
            require_complete_pairs=True,
        )
        split_pair_consistency[split] = pair_report
        errors.extend(f"Split {split!r}: {error}" for error in pair_errors)

    within_split_duplicates: dict[str, Any] = {}
    for split, counters in within_split_counts.items():
        duplicate_ids = sorted(
            (identity, count)
            for identity, count in counters["episode_id"].items()
            if count > 1
        )
        duplicate_keys = sorted(
            (
                (_serialize_identity(identity), count)
                for identity, count in counters["dataset_episode"].items()
                if count > 1
            ),
            key=lambda item: _identity_sort_key(item[0]),
        )
        within_split_duplicates[split] = {
            "episode_id": [
                {"identity": identity, "sample_count": count}
                for identity, count in duplicate_ids[:max_overlap_examples]
            ],
            "dataset_episode": [
                {"identity": identity, "sample_count": count}
                for identity, count in duplicate_keys[:max_overlap_examples]
            ],
        }
        if duplicate_ids or duplicate_keys:
            warnings.append(
                f"Split {split!r} contains repeated episode identities; this may "
                "be intentional when multiple event windows share an episode"
            )

    split_summary = {
        split: {
            "manifest_count": len(manifests_by_split[split]),
            "episode_id_count": len(episode_ids[split]),
            "dataset_episode_count": len(dataset_episodes[split]),
            "pair_id_count": len(pair_ids[split]),
        }
        for split in sorted(manifests_by_split)
    }
    return {
        "status": "error" if errors else "ok",
        "manifest_count": len(manifest_reports),
        "splits": split_summary,
        "manifests": manifest_reports,
        "cross_split_overlap": {
            "episode_id": episode_overlaps,
            "dataset_episode": dataset_episode_overlaps,
            "pair_id": pair_overlaps,
            "has_episode_leakage": bool(
                episode_overlaps or dataset_episode_overlaps
            ),
        },
        "split_pair_consistency": split_pair_consistency,
        "within_split_duplicates": within_split_duplicates,
        "errors": errors,
        "warnings": warnings,
    }


def parse_manifest_specs(specs: Sequence[str]) -> dict[str, list[Path]]:
    manifests_by_split: dict[str, list[Path]] = defaultdict(list)
    for spec in specs:
        if "=" not in spec:
            raise ValueError(
                f"Manifest input must use SPLIT=PATH syntax, got {spec!r}"
            )
        split, path = spec.split("=", 1)
        split = split.strip()
        path = path.strip()
        if not split or not path:
            raise ValueError(
                f"Manifest input must use non-empty SPLIT=PATH syntax, got {spec!r}"
            )
        manifests_by_split[split].append(Path(path).expanduser())
    if len(manifests_by_split) < 2:
        raise ValueError("At least two distinct split names are required")
    return dict(manifests_by_split)


def write_report(path: str | Path, report: Mapping[str, Any]) -> None:
    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(output_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "manifests",
        nargs="+",
        metavar="SPLIT=PATH",
        help="Manifest path tagged with its split; repeat for two or more splits.",
    )
    parser.add_argument("--output", type=Path, help="Optional JSON report path.")
    parser.add_argument(
        "--check-dataset-roots",
        action="store_true",
        help="Resolve each dataset_id and require its directory to exist.",
    )
    parser.add_argument(
        "--max-overlap-examples",
        type=int,
        default=100,
        help="Maximum serialized identities per overlap or duplicate group.",
    )
    parser.add_argument(
        "--allow-overlap",
        action="store_true",
        help="Report cross-split overlap without making overlap alone fail the CLI.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        manifests_by_split = parse_manifest_specs(args.manifests)
        report = audit_manifest_splits(
            manifests_by_split,
            check_dataset_roots=bool(args.check_dataset_roots),
            max_overlap_examples=int(args.max_overlap_examples),
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        report = {
            "status": "error",
            "manifest_count": 0,
            "splits": {},
            "manifests": [],
            "cross_split_overlap": {
                "episode_id": [],
                "dataset_episode": [],
                "pair_id": [],
                "has_episode_leakage": False,
            },
            "split_pair_consistency": {},
            "within_split_duplicates": {},
            "errors": [str(error)],
            "warnings": [],
        }

    if args.output is not None:
        write_report(args.output, report)
    print(json.dumps(report, indent=2, ensure_ascii=True, sort_keys=True))

    non_overlap_errors = [
        error
        for error in report["errors"]
        if error
        not in {
            "Cross-split episode leakage detected by episode_id and/or "
            "(dataset_id, episode_index)",
            "pair_id values cross split boundaries",
        }
    ]
    has_overlap = bool(
        report["cross_split_overlap"]["has_episode_leakage"]
        or report["cross_split_overlap"]["pair_id"]
    )
    if non_overlap_errors:
        return 2
    if has_overlap and not args.allow_overlap:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
