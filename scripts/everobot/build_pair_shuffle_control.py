#!/usr/bin/env python3
"""Build a deterministic M pair-shuffle control from frozen artifacts.

The control preserves each failure-side training sample and target while
deranging the success identity and its matching ``z_plus`` target inside the
same task/split group.  It accepts the formal M protocol only: positive pair
supervision must be attached to one auxiliary failure event per target row.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping, Sequence
import zipfile

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
for source_root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from fastwam.datasets.eve.pair_targets import PairTargetStore  # noqa: E402
from fastwam.everobot_schema import (  # noqa: E402
    compute_manifest_hash,
    validate_manifest,
)


CONTROL_FORMAT = "FastWAMPairShuffleControl"
CONTROL_VERSION = "1.0"
PAIR_ID_PREFIX = "pair:shuffle-v1:"
ARRAY_ORDER = (
    "pair_id",
    "success_event_id",
    "failure_event_id",
    "split",
    "pair_weight",
    "z_plus",
    "z_minus",
    "teacher_sha256",
)


@dataclass(frozen=True)
class SourcePair:
    row_index: int
    sample_index: int
    pair_id: str
    success_event_id: str
    failure_event_id: str
    split: str
    task: str

    @property
    def group(self) -> tuple[str, str]:
        return self.task, self.split

    @property
    def success_episode_id(self) -> str:
        return _event_episode_identity(self.success_event_id)


_EPISODE_PATTERNS = (
    re.compile(r"^(?P<episode>.*?_ep(?:isode_)?\d+)(?:_|$)"),
    re.compile(r"^(?P<episode>.*?:episode:\d+)(?::|_|$)"),
)


def _event_episode_identity(event_id: str) -> str:
    """Return the trajectory identity represented by an event identifier."""

    value = _nonempty_string(event_id, "success_event_id")
    for pattern in _EPISODE_PATTERNS:
        match = pattern.match(value)
        if match is not None:
            return match.group("episode")
    # Synthetic and legacy IDs may only distinguish windows by a candidate suffix.
    # Removing it still prevents two candidates from one trajectory being exchanged.
    without_candidate = re.sub(r"(?:_|:)candidate(?:_|:)\d+$", "", value)
    return without_candidate


def _canonical_json_bytes(payload: Any, *, pretty: bool = False) -> bytes:
    options: dict[str, Any] = {
        "ensure_ascii": True,
        "allow_nan": False,
        "sort_keys": True,
    }
    if pretty:
        options["indent"] = 2
        return (json.dumps(payload, **options) + "\n").encode("utf-8")
    options["separators"] = (",", ":")
    return json.dumps(payload, **options).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_json(payload: Any) -> str:
    return _sha256_bytes(_canonical_json_bytes(payload))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


def _sample_role(sample: Mapping[str, Any]) -> str:
    explicit = sample.get("batch_role")
    if explicit in {"primary", "auxiliary"}:
        return str(explicit)
    is_failure = (
        sample.get("episode_outcome") == "failure"
        or sample.get("event_outcome") == "failure"
    )
    action_enabled = sample.get("action_loss") != "disabled"
    return "primary" if not is_failure and action_enabled else "auxiliary"


def _task_identity(sample: Mapping[str, Any], label: str) -> str:
    value = sample.get("task_name") or sample.get("task")
    return _nonempty_string(value, f"{label}.task_name/task")


def _positive_pair_weight(sample: Mapping[str, Any], label: str) -> float:
    value = sample.get("pair_weight", 0.0)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label}.pair_weight must be numeric")
    weight = float(value)
    if not math.isfinite(weight) or not 0.0 <= weight <= 1.0:
        raise ValueError(f"{label}.pair_weight must be finite and in [0, 1]")
    return weight


def _load_target_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.array(archive[name], copy=True) for name in ARRAY_ORDER}


def _collect_source_pairs(
    manifest: Mapping[str, Any], arrays: Mapping[str, np.ndarray]
) -> list[SourcePair]:
    target_pair_ids = [str(value) for value in arrays["pair_id"]]
    target_index = {pair_id: index for index, pair_id in enumerate(target_pair_ids)}
    if len(target_index) != len(target_pair_ids):
        raise ValueError("pair-target pair_id values must be unique")

    pairs: list[SourcePair] = []
    seen_manifest_pairs: set[str] = set()
    for sample_index, sample in enumerate(manifest.get("samples", [])):
        label = f"samples[{sample_index}]"
        if _positive_pair_weight(sample, label) <= 0.0:
            continue
        if sample.get("sample_type") != "event":
            raise ValueError(
                f"{label} has positive pair supervision but is not an event"
            )
        if _sample_role(sample) != "auxiliary":
            raise ValueError(
                f"{label} has positive pair supervision but is not auxiliary"
            )
        if sample.get("episode_outcome") != "failure":
            raise ValueError(
                f"{label} has positive pair supervision but is not a failure event"
            )

        pair_id = _nonempty_string(sample.get("pair_id"), f"{label}.pair_id")
        if pair_id in seen_manifest_pairs:
            raise ValueError(
                f"Formal M manifest must reference each target once; duplicate {pair_id}"
            )
        seen_manifest_pairs.add(pair_id)
        if pair_id not in target_index:
            raise ValueError(f"Manifest pair {pair_id!r} has no target row")

        row_index = target_index[pair_id]
        event_id = _nonempty_string(sample.get("event_id"), f"{label}.event_id")
        failure_event_id = str(arrays["failure_event_id"][row_index])
        if event_id != failure_event_id:
            raise ValueError(
                f"Manifest pair {pair_id!r} event {event_id!r} does not match "
                f"target failure {failure_event_id!r}"
            )
        split = _nonempty_string(sample.get("split", "train"), f"{label}.split")
        target_split = str(arrays["split"][row_index])
        if split != target_split:
            raise ValueError(
                f"Manifest pair {pair_id!r} split {split!r} does not match "
                f"target split {target_split!r}"
            )
        sample_weight = float(sample["pair_weight"])
        target_weight = float(arrays["pair_weight"][row_index])
        if not math.isclose(sample_weight, target_weight, rel_tol=1e-6, abs_tol=1e-7):
            raise ValueError(
                f"Manifest pair {pair_id!r} weight does not match its target row"
            )
        pairs.append(
            SourcePair(
                row_index=row_index,
                sample_index=sample_index,
                pair_id=pair_id,
                success_event_id=str(arrays["success_event_id"][row_index]),
                failure_event_id=failure_event_id,
                split=split,
                task=_task_identity(sample, label),
            )
        )

    if not pairs:
        raise ValueError("Manifest contains no positive formal-M pair supervision")
    missing_target_ids = sorted(set(target_pair_ids) - seen_manifest_pairs)
    unreferenced_train_ids = [
        pair_id
        for pair_id in missing_target_ids
        if str(arrays["split"][target_index[pair_id]]) == "train"
    ]
    if unreferenced_train_ids:
        raise ValueError(
            "Formal M manifest leaves train target rows unreferenced: "
            f"{unreferenced_train_ids}"
        )
    return pairs


def _seeded_rank(seed: int, group: tuple[str, str], pair_id: str) -> str:
    return _sha256_json(
        {
            "shuffle_seed": seed,
            "task": group[0],
            "split": group[1],
            "pair_id": pair_id,
        }
    )


def _derange_group(pairs: Sequence[SourcePair], *, seed: int) -> dict[int, int]:
    """Map destination target rows to donors from different success episodes."""

    if not pairs:
        return {}
    group = pairs[0].group
    if any(pair.group != group for pair in pairs):
        raise ValueError("Internal error: mixed task/split group")
    if len(pairs) < 2:
        raise ValueError(
            f"Cannot derange task={group[0]!r}, split={group[1]!r}: only one pair"
        )

    counts = Counter(pair.success_episode_id for pair in pairs)
    maximum = max(counts.values())
    if maximum * 2 > len(pairs):
        raise ValueError(
            f"Cannot derange task={group[0]!r}, split={group[1]!r}: "
            f"success episode multiplicity {maximum}/{len(pairs)} violates Hall's condition"
        )

    ordered = sorted(
        pairs,
        key=lambda pair: (
            pair.success_episode_id,
            pair.success_event_id,
            _seeded_rank(seed, group, pair.pair_id),
            pair.pair_id,
        ),
    )
    direction_digest = _sha256_json(
        {"shuffle_seed": seed, "task": group[0], "split": group[1], "direction": 1}
    )
    shift = (
        maximum if int(direction_digest[-1], 16) % 2 == 0 else len(ordered) - maximum
    )
    donors = ordered[shift:] + ordered[:shift]
    assignment = {
        destination.row_index: donor.row_index
        for destination, donor in zip(ordered, donors, strict=True)
    }
    if any(
        destination.success_episode_id
        == next(pair.success_episode_id for pair in pairs if pair.row_index == donor_row)
        for destination, donor_row in (
            (pair, assignment[pair.row_index]) for pair in pairs
        )
    ):
        raise ValueError(
            "Internal error: failed to cross-episode derange "
            f"task={group[0]!r}, split={group[1]!r}"
        )
    return assignment


def _array_item_digest(event_id: str, embedding: np.ndarray) -> str:
    return _sha256_json(
        {
            "success_event_id": event_id,
            "dtype": embedding.dtype.str,
            "shape": list(embedding.shape),
            "bytes_sha256": _sha256_bytes(embedding.tobytes(order="C")),
        }
    )


def _make_pair_id(
    *,
    seed: int,
    source_manifest_sha256: str,
    source_targets_sha256: str,
    destination: SourcePair,
    donor: SourcePair,
) -> str:
    digest = _sha256_json(
        {
            "control_version": CONTROL_VERSION,
            "shuffle_seed": seed,
            "source_manifest_sha256": source_manifest_sha256,
            "source_pair_targets_sha256": source_targets_sha256,
            "task": destination.task,
            "split": destination.split,
            "source_pair_id": destination.pair_id,
            "donor_pair_id": donor.pair_id,
            "failure_event_id": destination.failure_event_id,
            "success_event_id": donor.success_event_id,
        }
    )
    return f"{PAIR_ID_PREFIX}{digest[:24]}"


def build_control(
    manifest: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    *,
    shuffle_seed: int,
    source_manifest_sha256: str,
    source_targets_sha256: str,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    """Build shuffled artifacts and an auditable proof without writing files."""

    validate_manifest(manifest, strict=True, verify_hash=True)
    pairs = _collect_source_pairs(manifest, arrays)
    by_row = {pair.row_index: pair for pair in pairs}
    assignments: dict[int, int] = {}
    groups: dict[tuple[str, str], list[SourcePair]] = defaultdict(list)
    for pair in pairs:
        groups[pair.group].append(pair)
    for group in sorted(groups):
        assignments.update(_derange_group(groups[group], seed=shuffle_seed))

    output_arrays = {name: np.array(arrays[name], copy=True) for name in ARRAY_ORDER}
    new_pair_ids: dict[str, str] = {}
    mapping: list[dict[str, Any]] = []
    for destination in sorted(pairs, key=lambda pair: pair.row_index):
        donor = by_row[assignments[destination.row_index]]
        new_pair_id = _make_pair_id(
            seed=shuffle_seed,
            source_manifest_sha256=source_manifest_sha256,
            source_targets_sha256=source_targets_sha256,
            destination=destination,
            donor=donor,
        )
        if destination.pair_id in new_pair_ids:
            raise ValueError(f"Duplicate source pair id: {destination.pair_id}")
        new_pair_ids[destination.pair_id] = new_pair_id
        row = destination.row_index
        donor_row = donor.row_index
        output_arrays["success_event_id"][row] = arrays["success_event_id"][donor_row]
        output_arrays["z_plus"][row] = arrays["z_plus"][donor_row]
        mapping.append(
            {
                "source_pair_id": destination.pair_id,
                "shuffled_pair_id": new_pair_id,
                "failure_event_id": destination.failure_event_id,
                "original_success_event_id": destination.success_event_id,
                "original_success_episode_id": destination.success_episode_id,
                "shuffled_success_event_id": donor.success_event_id,
                "shuffled_success_episode_id": donor.success_episode_id,
                "donor_source_pair_id": donor.pair_id,
                "task": destination.task,
                "split": destination.split,
            }
        )

    pair_id_width = max(len(value) for value in new_pair_ids.values())
    output_arrays["pair_id"] = np.asarray(
        [new_pair_ids.get(str(value), str(value)) for value in arrays["pair_id"]],
        dtype=f"<U{max(pair_id_width, arrays['pair_id'].dtype.itemsize // 4)}",
    )

    output_manifest = copy.deepcopy(dict(manifest))
    for pair in pairs:
        output_manifest["samples"][pair.sample_index]["pair_id"] = new_pair_ids[
            pair.pair_id
        ]
    output_manifest["manifest_hash"] = compute_manifest_hash(output_manifest)
    validate_manifest(output_manifest, strict=True, verify_hash=True)

    source_success_items = sorted(
        _array_item_digest(str(event_id), arrays["z_plus"][index])
        for index, event_id in enumerate(arrays["success_event_id"])
    )
    output_success_items = sorted(
        _array_item_digest(str(event_id), output_arrays["z_plus"][index])
        for index, event_id in enumerate(output_arrays["success_event_id"])
    )
    source_samples_without_pair = copy.deepcopy(manifest["samples"])
    output_samples_without_pair = copy.deepcopy(output_manifest["samples"])
    for sample in source_samples_without_pair:
        sample.pop("pair_id", None)
    for sample in output_samples_without_pair:
        sample.pop("pair_id", None)

    invariant_checks = {
        "sample_order_and_non_pair_fields_preserved": (
            source_samples_without_pair == output_samples_without_pair
        ),
        "roles_windows_and_weights_preserved": (
            source_samples_without_pair == output_samples_without_pair
            and np.array_equal(arrays["pair_weight"], output_arrays["pair_weight"])
        ),
        "failure_event_ids_preserved_rowwise": np.array_equal(
            arrays["failure_event_id"], output_arrays["failure_event_id"]
        ),
        "z_minus_preserved_rowwise": np.array_equal(
            arrays["z_minus"], output_arrays["z_minus"]
        ),
        "split_preserved_rowwise": np.array_equal(
            arrays["split"], output_arrays["split"]
        ),
        "teacher_hash_preserved_rowwise": np.array_equal(
            arrays["teacher_sha256"], output_arrays["teacher_sha256"]
        ),
        "success_event_id_z_plus_multiset_preserved": (
            source_success_items == output_success_items
        ),
        "all_referenced_success_identities_deranged": all(
            str(arrays["success_event_id"][index])
            != str(output_arrays["success_event_id"][index])
            for index in assignments
        ),
        "all_referenced_success_episodes_deranged": all(
            _event_episode_identity(str(arrays["success_event_id"][index]))
            != _event_episode_identity(str(output_arrays["success_event_id"][index]))
            for index in assignments
        ),
        "pair_ids_unique": len(set(str(value) for value in output_arrays["pair_id"]))
        == len(output_arrays["pair_id"]),
    }
    failed = sorted(name for name, passed in invariant_checks.items() if not passed)
    if failed:
        raise ValueError(f"Pair-shuffle invariant failure: {failed}")

    proof: dict[str, Any] = {
        "format": CONTROL_FORMAT,
        "version": CONTROL_VERSION,
        "shuffle_seed": shuffle_seed,
        "source": {
            "manifest_file_sha256": source_manifest_sha256,
            "manifest_hash": str(manifest["manifest_hash"]),
            "pair_targets_file_sha256": source_targets_sha256,
            "teacher_sha256": str(arrays["teacher_sha256"][0]),
            "target_row_count": len(arrays["pair_id"]),
            "referenced_pair_count": len(pairs),
            "passthrough_unreferenced_target_count": (
                len(arrays["pair_id"]) - len(pairs)
            ),
        },
        "groups": [
            {"task": task, "split": split, "pair_count": len(groups[(task, split)])}
            for task, split in sorted(groups)
        ],
        "mapping": mapping,
        "invariant_checks": invariant_checks,
        "invariant_hashes": {
            "source_success_event_id_z_plus_multiset_sha256": _sha256_json(
                source_success_items
            ),
            "output_success_event_id_z_plus_multiset_sha256": _sha256_json(
                output_success_items
            ),
            "source_failure_event_id_z_minus_rowwise_sha256": _sha256_json(
                [
                    {
                        "failure_event_id": str(event_id),
                        "z_minus_sha256": _sha256_bytes(
                            arrays["z_minus"][index].tobytes(order="C")
                        ),
                    }
                    for index, event_id in enumerate(arrays["failure_event_id"])
                ]
            ),
        },
    }
    proof["proof_sha256"] = _sha256_json(proof)
    return output_manifest, output_arrays, proof


def _npy_bytes(array: np.ndarray) -> bytes:
    stream = io.BytesIO()
    np.lib.format.write_array(stream, array, allow_pickle=False)
    return stream.getvalue()


def _write_deterministic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    with zipfile.ZipFile(
        path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for name in ARRAY_ORDER:
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o600 << 16
            archive.writestr(
                info,
                _npy_bytes(np.asarray(arrays[name])),
                compress_type=zipfile.ZIP_DEFLATED,
                compresslevel=9,
            )


def _temporary_path(final_path: Path, suffix: str) -> Path:
    final_path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(
        prefix=f".{final_path.name}.", suffix=suffix, dir=final_path.parent
    )
    os.close(fd)
    return Path(name)


def _publish_new(temporary: Path, final_path: Path) -> None:
    if final_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {final_path}")
    os.link(temporary, final_path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = args.manifest.expanduser().resolve()
    targets_path = args.pair_targets.expanduser().resolve()
    output_manifest_path = args.output_manifest.expanduser().resolve()
    output_targets_path = args.output_pair_targets.expanduser().resolve()
    proof_path = args.proof_output.expanduser().resolve()
    source_paths = {manifest_path, targets_path}
    output_paths = {output_manifest_path, output_targets_path, proof_path}
    if len(output_paths) != 3:
        raise ValueError("The three output paths must be distinct")
    if source_paths & output_paths:
        raise ValueError("Outputs must not replace source artifacts")
    for path in output_paths:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing output: {path}")

    manifest_sha256 = _sha256_file(manifest_path)
    targets_sha256 = _sha256_file(targets_path)
    manifest = _read_json(manifest_path)
    with PairTargetStore(
        targets_path, expected_teacher_sha256=args.expected_teacher_sha256
    ):
        pass
    arrays = _load_target_arrays(targets_path)
    output_manifest, output_arrays, proof = build_control(
        manifest,
        arrays,
        shuffle_seed=args.shuffle_seed,
        source_manifest_sha256=manifest_sha256,
        source_targets_sha256=targets_sha256,
    )

    manifest_bytes = _canonical_json_bytes(output_manifest, pretty=True)
    temp_manifest = _temporary_path(output_manifest_path, ".json.tmp")
    temp_targets = _temporary_path(output_targets_path, ".tmp.npz")
    temp_proof = _temporary_path(proof_path, ".json.tmp")
    created: list[Path] = []
    try:
        temp_manifest.write_bytes(manifest_bytes)
        _write_deterministic_npz(temp_targets, output_arrays)
        proof["output"] = {
            "manifest_file_sha256": _sha256_file(temp_manifest),
            "manifest_hash": str(output_manifest["manifest_hash"]),
            "pair_targets_file_sha256": _sha256_file(temp_targets),
        }
        proof["proof_sha256"] = _sha256_json(
            {key: value for key, value in proof.items() if key != "proof_sha256"}
        )
        temp_proof.write_bytes(_canonical_json_bytes(proof, pretty=True))

        validate_manifest(_read_json(temp_manifest), strict=True, verify_hash=True)
        with PairTargetStore(
            temp_targets, expected_teacher_sha256=args.expected_teacher_sha256
        ) as targets:
            for sample in output_manifest["samples"]:
                if float(sample.get("pair_weight", 0.0)) <= 0.0:
                    continue
                target = targets.get(str(sample["pair_id"]))
                if target.failure_event_id != str(sample["event_id"]):
                    raise ValueError(
                        f"Published pair {sample['pair_id']} does not match manifest failure"
                    )

        for temporary, final_path in (
            (temp_manifest, output_manifest_path),
            (temp_targets, output_targets_path),
            (temp_proof, proof_path),
        ):
            _publish_new(temporary, final_path)
            created.append(final_path)
    except Exception:
        for path in reversed(created):
            path.unlink(missing_ok=True)
        raise
    finally:
        temp_manifest.unlink(missing_ok=True)
        temp_targets.unlink(missing_ok=True)
        temp_proof.unlink(missing_ok=True)

    return {
        "output_manifest": str(output_manifest_path),
        "output_pair_targets": str(output_targets_path),
        "proof_output": str(proof_path),
        "pair_count": len(output_arrays["pair_id"]),
        "group_count": len(proof["groups"]),
        "shuffle_seed": args.shuffle_seed,
        "proof_sha256": proof["proof_sha256"],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pair-targets", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-pair-targets", type=Path, required=True)
    parser.add_argument("--proof-output", type=Path, required=True)
    parser.add_argument("--shuffle-seed", type=int, required=True)
    parser.add_argument("--expected-teacher-sha256")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    report = run(parse_args(argv))
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
