#!/usr/bin/env python3
"""Bind a formal S0 rollout dataset to its frozen seeds and protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any


EXPECTED_EPISODES = 200
HASH_CHUNK_BYTES = 8 * 1024 * 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate formal S0 seeds, protocol, and dataset fingerprint."
    )
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--outcome-validation", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def _signature(metadata: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_size,
        metadata.st_mtime_ns,
    )


def _open_regular_file(path: Path) -> tuple[Any, os.stat_result]:
    path_metadata = os.lstat(path)
    if not stat.S_ISREG(path_metadata.st_mode):
        raise ValueError(f"Expected a regular file: {path}")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    stream = os.fdopen(descriptor, "rb")
    metadata = os.fstat(stream.fileno())
    if not stat.S_ISREG(metadata.st_mode):
        stream.close()
        raise ValueError(f"Expected a regular file: {path}")
    if (path_metadata.st_dev, path_metadata.st_ino) != (
        metadata.st_dev,
        metadata.st_ino,
    ):
        stream.close()
        raise RuntimeError(f"File was replaced while opening: {path}")
    return stream, metadata


def hash_regular_file(path: Path) -> tuple[str, int, tuple[int, int, int, int, int]]:
    digest = hashlib.sha256()
    stream, before = _open_regular_file(path)
    with stream:
        while True:
            chunk = stream.read(HASH_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
        after = os.fstat(stream.fileno())
    if _signature(before) != _signature(after):
        raise RuntimeError(f"File changed while hashing: {path}")
    current = os.lstat(path)
    if _signature(after) != _signature(current):
        raise RuntimeError(f"File was replaced while hashing: {path}")
    return digest.hexdigest(), after.st_size, _signature(after)


def read_stable_bytes(
    path: Path,
) -> tuple[bytes, str, tuple[int, int, int, int, int]]:
    digest = hashlib.sha256()
    chunks = []
    stream, before = _open_regular_file(path)
    with stream:
        while True:
            chunk = stream.read(HASH_CHUNK_BYTES)
            if not chunk:
                break
            chunks.append(chunk)
            digest.update(chunk)
        after = os.fstat(stream.fileno())
    if _signature(before) != _signature(after):
        raise RuntimeError(f"File changed while reading: {path}")
    current = os.lstat(path)
    if _signature(after) != _signature(current):
        raise RuntimeError(f"File was replaced while reading: {path}")
    return b"".join(chunks), digest.hexdigest(), _signature(after)


def load_json_bytes(payload: bytes, *, path: Path) -> dict[str, Any]:
    value = json.loads(payload.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def load_jsonl_bytes(payload: bytes, *, path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, raw_line in enumerate(payload.decode("utf-8").splitlines(), 1):
        if not raw_line.strip():
            continue
        row = json.loads(raw_line)
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number} must contain a JSON object")
        rows.append(row)
    return rows


def require_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer, got {value!r}")
    return value


def validate_episode_index_rows(
    rows: list[dict[str, Any]],
    *,
    path: Path,
) -> list[int]:
    if len(rows) != EXPECTED_EPISODES:
        raise ValueError(
            f"{path} must contain exactly {EXPECTED_EPISODES} rows, got {len(rows)}"
        )
    indexes = [
        require_int(row.get("episode_index"), label=f"{path}:{index}:episode_index")
        for index, row in enumerate(rows, 1)
    ]
    if len(set(indexes)) != EXPECTED_EPISODES:
        raise ValueError(f"{path} contains duplicate episode_index values")
    expected = list(range(EXPECTED_EPISODES))
    if sorted(indexes) != expected:
        raise ValueError(
            f"{path} must cover episode_index 0..{EXPECTED_EPISODES - 1}"
        )
    return indexes


def validate_ledger(
    rows: list[dict[str, Any]],
    *,
    path: Path,
    base_seed: int,
) -> dict[str, Any]:
    indexes = validate_episode_index_rows(rows, path=path)
    seeds = []
    successes = 0
    failures = 0
    for row_number, row in enumerate(rows, 1):
        seed = require_int(row.get("seed"), label=f"{path}:{row_number}:seed")
        seeds.append(seed)
        success = row.get("success")
        if not isinstance(success, bool):
            raise ValueError(f"{path}:{row_number}:success must be boolean")
        expected_outcome = "success" if success else "failure"
        if row.get("outcome") != expected_outcome:
            raise ValueError(
                f"{path}:{row_number}:outcome disagrees with success={success}"
            )
        successes += int(success)
        failures += int(not success)

    if len(set(seeds)) != EXPECTED_EPISODES:
        raise ValueError(f"{path} contains duplicate seed values")
    expected_seeds = list(range(base_seed, base_seed + EXPECTED_EPISODES))
    if sorted(seeds) != expected_seeds:
        raise ValueError(
            f"{path} seeds must be exactly {base_seed}.."
            f"{base_seed + EXPECTED_EPISODES - 1}"
        )
    return {
        "episode_indexes": indexes,
        "seed_start": base_seed,
        "seed_stop_exclusive": base_seed + EXPECTED_EPISODES,
        "successes": successes,
        "failures": failures,
    }


def collect_dataset_fingerprint(dataset_root: Path) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    signatures: dict[Path, tuple[int, int, int, int, int]] = {}
    for directory, directory_names, file_names in os.walk(
        dataset_root, topdown=True, followlinks=False
    ):
        directory_path = Path(directory)
        directory_names.sort()
        file_names.sort()
        for name in directory_names:
            candidate = directory_path / name
            if candidate.is_symlink():
                raise ValueError(f"Dataset contains a symlinked directory: {candidate}")
        for name in file_names:
            path = directory_path / name
            if path.is_symlink():
                raise ValueError(f"Dataset contains a symlinked file: {path}")
            relative_path = path.relative_to(dataset_root).as_posix()
            digest, size_bytes, signature = hash_regular_file(path)
            records.append(
                {
                    "path": relative_path,
                    "size_bytes": size_bytes,
                    "sha256": digest,
                }
            )
            signatures[path] = signature

    records.sort(key=lambda record: record["path"])
    paths = {record["path"] for record in records}
    required_meta = {
        "meta/info.json",
        "meta/episodes.jsonl",
        "meta/episode_outcomes.jsonl",
    }
    missing_meta = sorted(required_meta.difference(paths))
    if missing_meta:
        raise FileNotFoundError(f"Dataset fingerprint is missing {missing_meta}")
    if not any(path.endswith(".parquet") for path in paths):
        raise FileNotFoundError("Dataset fingerprint contains no parquet files")
    if not any(path.endswith(".mp4") for path in paths):
        raise FileNotFoundError("Dataset fingerprint contains no video files")

    for path, expected_signature in signatures.items():
        if _signature(os.lstat(path)) != expected_signature:
            raise RuntimeError(f"Dataset changed during fingerprinting: {path}")

    canonical = json.dumps(
        records,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "algorithm": "sha256(canonical-json-v1[path,size_bytes,sha256])",
        "sha256": hashlib.sha256(canonical).hexdigest(),
        "file_count": len(records),
        "total_size_bytes": sum(record["size_bytes"] for record in records),
        "files": records,
    }


def _record_by_path(
    fingerprint: dict[str, Any], relative_path: str
) -> dict[str, Any]:
    matches = [
        record
        for record in fingerprint["files"]
        if record["path"] == relative_path
    ]
    if len(matches) != 1:
        raise ValueError(f"Fingerprint must contain exactly one {relative_path}")
    return matches[0]


def validate_formal_outputs(
    protocol_path: Path,
    dataset_root: Path,
    outcome_validation_path: Path,
) -> dict[str, Any]:
    raw_dataset_root = dataset_root.expanduser()
    if raw_dataset_root.is_symlink():
        raise ValueError(f"Dataset root must not be a symlink: {raw_dataset_root}")
    dataset_root = raw_dataset_root.resolve(strict=True)
    if not dataset_root.is_dir():
        raise NotADirectoryError(dataset_root)

    protocol_path = protocol_path.expanduser().resolve(strict=True)
    outcome_validation_path = outcome_validation_path.expanduser().resolve(strict=True)
    ledger_path = dataset_root / "meta" / "episode_outcomes.jsonl"
    episodes_path = dataset_root / "meta" / "episodes.jsonl"

    protocol_bytes, protocol_sha256, protocol_signature = read_stable_bytes(
        protocol_path
    )
    outcome_bytes, outcome_sha256, outcome_signature = read_stable_bytes(
        outcome_validation_path
    )
    ledger_bytes, ledger_sha256, ledger_signature = read_stable_bytes(ledger_path)
    episodes_bytes, episodes_sha256, episodes_signature = read_stable_bytes(
        episodes_path
    )
    protocol = load_json_bytes(protocol_bytes, path=protocol_path)
    outcome_validation = load_json_bytes(outcome_bytes, path=outcome_validation_path)
    ledger_rows = load_jsonl_bytes(ledger_bytes, path=ledger_path)
    episode_rows = load_jsonl_bytes(episodes_bytes, path=episodes_path)

    collection = protocol.get("collection")
    if not isinstance(collection, dict):
        raise ValueError(f"{protocol_path}: collection must be an object")
    if collection.get("kind") != "formal":
        raise ValueError(f"{protocol_path}: collection.kind must be 'formal'")
    protocol_episodes = require_int(
        collection.get("episodes"), label="collection.episodes"
    )
    if protocol_episodes != EXPECTED_EPISODES:
        raise ValueError(
            f"collection.episodes must be {EXPECTED_EPISODES}, got {protocol_episodes}"
        )
    base_seed = require_int(collection.get("base_seed"), label="collection.base_seed")
    seed_stop_exclusive = require_int(
        collection.get("seed_stop_exclusive"),
        label="collection.seed_stop_exclusive",
    )
    if seed_stop_exclusive != base_seed + EXPECTED_EPISODES:
        raise ValueError(
            "collection.seed_stop_exclusive does not match base_seed + 200"
        )

    episode_indexes = validate_episode_index_rows(episode_rows, path=episodes_path)
    ledger = validate_ledger(ledger_rows, path=ledger_path, base_seed=base_seed)
    if set(episode_indexes) != set(ledger["episode_indexes"]):
        raise ValueError("Outcome ledger does not cover the dataset episode metadata")

    if outcome_validation.get("status") != "valid":
        raise ValueError("outcome_validation.json status must be 'valid'")
    if outcome_validation.get("check_media") is not True:
        raise ValueError("outcome_validation.json must come from --check-media")
    outcome_episodes = require_int(
        outcome_validation.get("episodes"), label="outcome_validation.episodes"
    )
    if outcome_episodes != EXPECTED_EPISODES:
        raise ValueError(
            f"outcome_validation.episodes must be {EXPECTED_EPISODES}"
        )
    reported_root = Path(str(outcome_validation.get("dataset_root", ""))).expanduser()
    if reported_root.resolve(strict=True) != dataset_root:
        raise ValueError("outcome_validation dataset_root does not match dataset")
    reported_ledger = Path(
        str(outcome_validation.get("outcome_ledger", ""))
    ).expanduser()
    if reported_ledger.resolve(strict=True) != ledger_path.resolve(strict=True):
        raise ValueError("outcome_validation outcome_ledger does not match dataset")
    if outcome_validation.get("outcome_ledger_sha256") != ledger_sha256:
        raise ValueError("outcome_validation ledger SHA256 is stale or incorrect")
    if require_int(
        outcome_validation.get("successes"), label="outcome_validation.successes"
    ) != ledger["successes"]:
        raise ValueError("outcome_validation successes disagree with ledger")
    if require_int(
        outcome_validation.get("failures"), label="outcome_validation.failures"
    ) != ledger["failures"]:
        raise ValueError("outcome_validation failures disagree with ledger")
    physical = outcome_validation.get("physical_validation")
    if not isinstance(physical, dict):
        raise ValueError("outcome_validation is missing physical_validation")
    if require_int(
        physical.get("episodes"), label="physical_validation.episodes"
    ) != EXPECTED_EPISODES:
        raise ValueError("physical_validation.episodes must be 200")

    fingerprint = collect_dataset_fingerprint(dataset_root)
    ledger_record = _record_by_path(fingerprint, "meta/episode_outcomes.jsonl")
    episodes_record = _record_by_path(fingerprint, "meta/episodes.jsonl")
    if ledger_record["sha256"] != ledger_sha256:
        raise RuntimeError("Outcome ledger changed before dataset fingerprint completed")
    if episodes_record["sha256"] != episodes_sha256:
        raise RuntimeError("Episode metadata changed before dataset fingerprint completed")

    for path, expected_signature in (
        (protocol_path, protocol_signature),
        (outcome_validation_path, outcome_signature),
        (ledger_path, ledger_signature),
        (episodes_path, episodes_signature),
    ):
        if _signature(os.lstat(path)) != expected_signature:
            raise RuntimeError(f"Validation input changed during execution: {path}")

    return {
        "status": "valid",
        "expected_episodes": EXPECTED_EPISODES,
        "protocol": {
            "path": str(protocol_path),
            "sha256": protocol_sha256,
            "base_seed": base_seed,
            "seed_stop_exclusive": seed_stop_exclusive,
        },
        "dataset": {
            "root": str(dataset_root),
            "episode_outcomes_sha256": ledger_sha256,
            "episodes_sha256": episodes_sha256,
            "successes": ledger["successes"],
            "failures": ledger["failures"],
            "fingerprint": fingerprint,
        },
        "outcome_validation": {
            "path": str(outcome_validation_path),
            "sha256": outcome_sha256,
        },
    }


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.incomplete-{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def validate_and_write(
    protocol_path: Path,
    dataset_root: Path,
    outcome_validation_path: Path,
    report_path: Path,
) -> dict[str, Any]:
    raw_dataset_root = dataset_root.expanduser().resolve(strict=True)
    report_path = report_path.expanduser().resolve()
    try:
        report_path.relative_to(raw_dataset_root)
    except ValueError:
        pass
    else:
        raise ValueError("Formal validation report must be outside the dataset root")

    report_path.unlink(missing_ok=True)
    try:
        report = validate_formal_outputs(
            protocol_path,
            dataset_root,
            outcome_validation_path,
        )
        atomic_write_json(report_path, report)
        return report
    except BaseException:
        report_path.unlink(missing_ok=True)
        raise


def main() -> None:
    args = parse_args()
    report = validate_and_write(
        args.protocol,
        args.dataset,
        args.outcome_validation,
        args.report,
    )
    print(f"[s0-formal-validation] status={report['status']}")
    print(
        "[s0-formal-validation] dataset_sha256="
        f"{report['dataset']['fingerprint']['sha256']}"
    )
    print(f"[s0-formal-validation] report={args.report.expanduser().resolve()}")


if __name__ == "__main__":
    main()
