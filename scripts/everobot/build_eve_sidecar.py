#!/usr/bin/env python3
"""Build EveRobot sidecar metadata for LeRobot-compatible datasets.

EveRobot v0.2 keeps LeRobot data untouched and writes all self-evolution
metadata under an ``eve/`` directory.  The sidecar records episode provenance,
failure event windows, and round-specific training manifests.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fastwam.everobot_schema import (  # noqa: E402
    SCHEMA_VERSION,
    compute_manifest_hash,
    sha256_json,
    validate_manifest,
)


FAILURE_PHRASE = "Failed to finish the whole process."


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    text = json.dumps(payload, indent=2, ensure_ascii=True) + "\n"
    write_text_atomic(path, text)


def write_text_atomic(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_episode_outcome_ledger(
    dataset_root: Path,
    *,
    required: bool = False,
    expected_episode_indices: Iterable[int] | None = None,
) -> dict[int, dict[str, Any]]:
    """Load the structured rollout outcome ledger and validate full coverage."""

    path = dataset_root / "meta" / "episode_outcomes.jsonl"
    if not path.exists():
        if required:
            raise FileNotFoundError(
                "Formal rollout data requires a structured outcome ledger: "
                f"{path}"
            )
        return {}

    rows = load_jsonl(path)
    if not rows:
        raise ValueError(f"Episode outcome ledger is empty: {path}")

    mapping: dict[int, dict[str, Any]] = {}
    for row_index, raw_row in enumerate(rows):
        row = dict(raw_row)
        episode_index = row.get("episode_index")
        if isinstance(episode_index, bool) or not isinstance(episode_index, int):
            raise ValueError(
                f"Invalid episode_index at {path} row {row_index}: {episode_index!r}"
            )
        episode_index = int(episode_index)
        if episode_index < 0:
            raise ValueError(
                f"Negative episode_index at {path} row {row_index}: {episode_index}"
            )
        if episode_index in mapping:
            raise ValueError(
                f"Duplicate episode_index {episode_index} in outcome ledger {path}"
            )

        outcome = row.get("outcome", row.get("episode_outcome"))
        if outcome not in {"success", "failure"}:
            raise ValueError(
                f"Invalid outcome at {path} row {row_index}: {outcome!r}"
            )
        success = row.get("success")
        if success is not None:
            if not isinstance(success, bool):
                raise ValueError(
                    f"success must be boolean at {path} row {row_index}"
                )
            if success != (outcome == "success"):
                raise ValueError(
                    f"Conflicting success/outcome at {path} row {row_index}"
                )
        row["episode_index"] = episode_index
        row["outcome"] = str(outcome)
        mapping[episode_index] = row

    if expected_episode_indices is not None:
        expected = {int(index) for index in expected_episode_indices}
        observed = set(mapping)
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        if missing or extra:
            raise ValueError(
                f"Episode outcome ledger coverage mismatch at {path}: "
                f"missing={missing}, extra={extra}"
            )
    return mapping


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    payload = "".join(
        json.dumps(row, ensure_ascii=True) + "\n" for row in rows
    )
    write_text_atomic(path, payload)


def load_episode_split_map(
    path: Path | None,
) -> dict[tuple[str, int], str] | None:
    if path is None:
        return None
    resolved = path.expanduser().resolve()
    rows = load_jsonl(resolved)
    if not rows:
        raise ValueError(f"Episode split map is empty: {resolved}")
    mapping: dict[tuple[str, int], str] = {}
    for index, row in enumerate(rows):
        dataset_id = str(row.get("dataset_id") or "")
        episode_index = row.get("episode_index")
        split = str(row.get("split") or "")
        if not dataset_id or isinstance(episode_index, bool) or not isinstance(
            episode_index, int
        ):
            raise ValueError(
                f"Invalid split-map identity at row {index}: {row}"
            )
        if split not in {"train", "val", "test"}:
            raise ValueError(
                f"Invalid split {split!r} at split-map row {index}"
            )
        key = (dataset_id, int(episode_index))
        previous = mapping.get(key)
        if previous is not None and previous != split:
            raise ValueError(
                f"Conflicting split assignments for {key}: "
                f"{previous!r} vs {split!r}"
            )
        mapping[key] = split
    return mapping


def resolve_episode_split(
    *,
    split_map: dict[tuple[str, int], str] | None,
    dataset_id: str,
    episode_index: int,
    default_split: str,
) -> str:
    if split_map is None:
        return default_split
    key = (str(dataset_id), int(episode_index))
    if key not in split_map:
        raise ValueError(
            f"Episode split map has no assignment for dataset_id={dataset_id!r}, "
            f"episode_index={episode_index}."
        )
    return split_map[key]


@contextmanager
def sidecar_write_lock(eve_root: Path):
    eve_root.mkdir(parents=True, exist_ok=True)
    lock_path = eve_root / ".write.lock"
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _without_fields(row: dict[str, Any], fields: set[str]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key not in fields}


def prepare_immutable_jsonl(
    path: Path,
    new_rows: list[dict[str, Any]],
    *,
    key_fields: tuple[str, ...],
    compare_ignore_fields: tuple[str, ...] = ("created_at",),
) -> tuple[list[dict[str, Any]], int]:
    """Plan an append without writing, so a multi-ledger update can preflight."""

    rows = load_jsonl(path)
    keyed: dict[tuple[Any, ...], dict[str, Any]] = {}
    ignored = set(compare_ignore_fields)
    for row in rows:
        key = tuple(row.get(field) for field in key_fields)
        if None in key:
            raise ValueError(f"Missing identity field in {path}: {key_fields} row={row}")
        if key in keyed and _without_fields(keyed[key], ignored) != _without_fields(row, ignored):
            raise ValueError(f"Conflicting duplicate identity {key} already exists in {path}")
        keyed[key] = row

    appended = 0
    for row in new_rows:
        key = tuple(row.get(field) for field in key_fields)
        if None in key:
            raise ValueError(f"Missing identity field in new row for {path}: {key_fields} row={row}")
        if key in keyed:
            if _without_fields(keyed[key], ignored) != _without_fields(row, ignored):
                raise ValueError(
                    f"Immutable EveRobot identity collision in {path}: {key}. "
                    "Use a new dataset_id/round_id instead of overwriting a prior round."
                )
            continue
        keyed[key] = row
        rows.append(row)
        appended += 1

    return rows, appended


def append_immutable_jsonl_group(
    updates: list[
        tuple[
            Path,
            list[dict[str, Any]],
            tuple[str, ...],
            tuple[str, ...],
        ]
    ],
) -> list[int]:
    """Preflight every ledger before atomically replacing individual files."""

    plans: list[tuple[Path, list[dict[str, Any]], int]] = []
    for path, rows, key_fields, compare_ignore_fields in updates:
        merged, appended = prepare_immutable_jsonl(
            path,
            rows,
            key_fields=key_fields,
            compare_ignore_fields=compare_ignore_fields,
        )
        plans.append((path, merged, appended))

    for path, rows, appended in plans:
        if appended:
            write_jsonl(path, rows)
    return [appended for _, _, appended in plans]


def validate_round_parent_refs(
    eve_root: Path, new_round_rows: list[dict[str, Any]]
) -> None:
    existing_round_ids = {
        str(row["round_id"]) for row in load_jsonl(eve_root / "round_meta.jsonl")
    }
    new_round_ids = {str(row["round_id"]) for row in new_round_rows}
    available_round_ids = existing_round_ids | new_round_ids
    for row in new_round_rows:
        round_id = str(row["round_id"])
        parent_round_ids = {str(item) for item in row.get("parent_round_ids", [])}
        if round_id in parent_round_ids:
            raise ValueError(f"Round {round_id} cannot be its own parent")
        missing = parent_round_ids - available_round_ids
        if missing:
            raise ValueError(
                f"Round {round_id} references missing parents: {sorted(missing)}"
            )


def file_sha256(path: Path | None) -> str | None:
    if path is None or not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_content_fingerprint(dataset_root: Path) -> str:
    """Hash LeRobot data, videos, and metadata without including Eve sidecars."""

    digest = hashlib.sha256(b"EveRobotDatasetContentV1\0")
    files: list[Path] = []
    for directory_name in ("data", "videos", "meta"):
        directory = dataset_root / directory_name
        if directory.exists():
            files.extend(path for path in directory.rglob("*") if path.is_file())

    included = 0
    for path in sorted(files, key=lambda item: item.relative_to(dataset_root).as_posix()):
        relative = path.relative_to(dataset_root)
        if relative.parts[:2] == ("meta", "eve"):
            continue
        file_digest = file_sha256(path)
        if file_digest is None:
            raise FileNotFoundError(f"Could not hash dataset file: {path}")
        digest.update(relative.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(file_digest))
        included += 1
    if included == 0:
        raise ValueError(f"No LeRobot content files found under {dataset_root}")
    return digest.hexdigest()


def current_git_provenance(
    explicit_commit: str | None = None,
) -> tuple[str | None, bool | None, str | None]:
    if explicit_commit:
        return explicit_commit, False, None
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            stderr=subprocess.DEVNULL,
        ).decode("utf-8").strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain=v1", "-z", "--untracked-files=all"],
            cwd=PROJECT_ROOT,
            stderr=subprocess.DEVNULL,
        )
        if not status:
            return commit, False, None

        digest = hashlib.sha256(b"EveRobotCodeDiffV1\0")
        digest.update(status)
        digest.update(
            subprocess.check_output(
                ["git", "diff", "--binary", "HEAD"],
                cwd=PROJECT_ROOT,
                stderr=subprocess.DEVNULL,
            )
        )
        untracked = subprocess.check_output(
            ["git", "ls-files", "--others", "--exclude-standard", "-z"],
            cwd=PROJECT_ROOT,
            stderr=subprocess.DEVNULL,
        )
        for relative_bytes in sorted(item for item in untracked.split(b"\0") if item):
            relative = relative_bytes.decode("utf-8")
            path = PROJECT_ROOT / relative
            content_hash = file_sha256(path)
            digest.update(relative_bytes)
            digest.update(b"\0")
            if content_hash is not None:
                digest.update(bytes.fromhex(content_hash))
        return commit, True, digest.hexdigest()
    except (OSError, subprocess.CalledProcessError):
        return None, None, None


def utc_now(explicit: str | None = None) -> str:
    return explicit or datetime.now(timezone.utc).isoformat()


def make_round_id(dataset_id: str, collection_round: int) -> str:
    return f"{dataset_id}:round:{int(collection_round)}"


def make_episode_id(dataset_id: str, episode_index: int) -> str:
    return f"{dataset_id}:episode:{int(episode_index):06d}"


def make_round_row(
    *,
    dataset_root: Path,
    dataset_id: str,
    task_name: str,
    source_type: str,
    source_policy: str,
    collection_round: int,
    source_checkpoint: str | None,
    source_checkpoint_sha256: str | None,
    dataset_fingerprint_sha256: str | None,
    parent_round_ids: list[str] | None,
    config_path: Path | None,
    code_commit: str | None,
    created_at: str | None,
    dataset_uri: str | None,
) -> dict[str, Any]:
    collection_summary_path = dataset_root / "collection_summary.json"
    checkpoint_path = Path(source_checkpoint).expanduser() if source_checkpoint else None
    actual_checkpoint_hash = file_sha256(checkpoint_path)
    if (
        source_checkpoint_sha256 is not None
        and actual_checkpoint_hash is not None
        and source_checkpoint_sha256.lower() != actual_checkpoint_hash
    ):
        raise ValueError("source_checkpoint_sha256 does not match source_checkpoint")
    checkpoint_hash = source_checkpoint_sha256 or actual_checkpoint_hash
    if source_checkpoint and checkpoint_hash is None:
        raise FileNotFoundError(
            f"Source checkpoint does not exist and no hash was provided: {source_checkpoint}"
        )
    if checkpoint_hash is not None:
        if len(checkpoint_hash) != 64:
            raise ValueError("source_checkpoint_sha256 must be a SHA-256 hex digest")
        try:
            bytes.fromhex(checkpoint_hash)
        except ValueError as error:
            raise ValueError(
                "source_checkpoint_sha256 must be a SHA-256 hex digest"
            ) from error
    dataset_hash = dataset_fingerprint_sha256 or dataset_content_fingerprint(dataset_root)
    if len(dataset_hash) != 64:
        raise ValueError("dataset_fingerprint_sha256 must be a SHA-256 hex digest")
    try:
        bytes.fromhex(dataset_hash)
    except ValueError as error:
        raise ValueError(
            "dataset_fingerprint_sha256 must be a SHA-256 hex digest"
        ) from error
    config_hash = file_sha256(config_path)
    if config_path is not None and config_hash is None:
        raise FileNotFoundError(f"Config path does not exist: {config_path}")
    code_commit_value, code_dirty, code_diff_hash = current_git_provenance(
        code_commit
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "round_id": make_round_id(dataset_id, collection_round),
        "collection_round": int(collection_round),
        "dataset_id": dataset_id,
        "dataset_uri": dataset_uri or f"dataset://{dataset_id}",
        "dataset_fingerprint_sha256": dataset_hash,
        "task_name": task_name,
        "source_type": source_type,
        "source_policy": source_policy,
        "source_checkpoint": source_checkpoint,
        "source_checkpoint_id": checkpoint_path.name if checkpoint_path else None,
        "source_checkpoint_sha256": checkpoint_hash,
        "parent_round_ids": sorted(parent_round_ids or []),
        "config_sha256": config_hash,
        "code_commit": code_commit_value,
        "code_dirty": code_dirty,
        "code_diff_sha256": code_diff_hash,
        "collection_summary_sha256": file_sha256(collection_summary_path),
        "created_at": utc_now(created_at),
    }


def strip_failure_phrase(task: str, failure_phrase: str) -> str:
    return " ".join(str(task).replace(failure_phrase, "").split()).strip()


def first_task(ep_row: dict[str, Any]) -> str:
    tasks = ep_row.get("tasks", [])
    if isinstance(tasks, list) and tasks:
        return str(tasks[0])
    return str(ep_row.get("task", ""))


def is_failure_task(task: str, failure_phrase: str) -> bool:
    return failure_phrase in str(task)


def load_lerobot_episodes(dataset_root: Path) -> list[dict[str, Any]]:
    episodes_path = dataset_root / "meta" / "episodes.jsonl"
    if not episodes_path.exists():
        raise FileNotFoundError(f"Missing LeRobot episodes file: {episodes_path}")
    return load_jsonl(episodes_path)


def load_lerobot_info(dataset_root: Path) -> dict[str, Any]:
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Missing LeRobot info file: {info_path}")
    return read_json(info_path)


def assert_schema_compatible(eve_root: Path) -> None:
    path = eve_root / "schema_version.json"
    if path.exists():
        existing_version = str(read_json(path).get("schema_version", ""))
        if existing_version != SCHEMA_VERSION:
            raise ValueError(
                f"EveRobot sidecar {eve_root} uses schema {existing_version}; "
                f"v{SCHEMA_VERSION} requires a fresh sidecar root. Legacy manifests remain readable."
            )


def write_schema(eve_root: Path) -> None:
    assert_schema_compatible(eve_root)
    path = eve_root / "schema_version.json"
    write_json(
        path,
        {
            "format": "EveRobot",
            "schema_version": SCHEMA_VERSION,
            "compatible_base_format": "LeRobot",
            "frame_interval": "half_open",
            "allowed_outcomes": ["success", "failure", "unknown"],
            "allowed_action_loss": ["enabled", "disabled"],
            "allowed_effectors": ["left", "right", "bimanual", "global"],
            "hash_algorithms": {
                "dataset": "EveRobotDatasetContentV1",
                "manifest": "canonical_json_sha256",
            },
            "description": (
                "LeRobot-compatible sidecar metadata for failure-aware "
                "self-evolution training."
            ),
            "files": {
                "task_schema": "task_schema.json",
                "round_meta": "round_meta.jsonl",
                "episode_meta": "episode_meta.jsonl",
                "event_meta": "event_meta.jsonl",
                "soft_annotations": "annotations/*.parquet",
                "manifests": "manifests/*.json",
            },
        },
    )


def attempt_log_by_episode(
    summary: dict[str, Any],
    *,
    episode_count: int | None = None,
) -> dict[int, dict[str, Any]]:
    logs = [dict(item) for item in summary.get("attempt_log", [])]
    if episode_count is not None and len(logs) == int(episode_count):
        saved = [
            item.get("saved_episode_index", item.get("saved_failure_index"))
            for item in logs
        ]
        saved_int = [int(item) for item in saved if item is not None]
        # Merged shard summaries may keep shard-local saved_episode_index
        # values.  When every attempt was saved, the merged episode order is
        # the reliable global mapping.
        if len(saved_int) != len(set(saved_int)) or (
            saved_int and max(saved_int) < int(episode_count) - 1
        ):
            return {idx: item for idx, item in enumerate(logs)}

    out: dict[int, dict[str, Any]] = {}
    for item in logs:
        ep_idx = item.get("saved_episode_index")
        if ep_idx is None:
            ep_idx = item.get("saved_failure_index")
        if ep_idx is None:
            continue
        out[int(ep_idx)] = dict(item)
    return out


def load_collection_summary(dataset_root: Path) -> dict[str, Any]:
    path = dataset_root / "collection_summary.json"
    if path.exists():
        return read_json(path)
    return {}


def load_trim_report(trimmed_root: Path | None) -> dict[int, dict[str, Any]]:
    if trimmed_root is None:
        return {}

    collection_summary = trimmed_root / "collection_summary.json"
    if collection_summary.exists():
        summary = read_json(collection_summary)
        if "trim_report" in summary:
            return {int(row["episode_index"]): dict(row) for row in summary["trim_report"]}

    trim_summary = trimmed_root / "trim_summary.json"
    if trim_summary.exists():
        summary = read_json(trim_summary)
        if "episodes" in summary:
            return {int(row["episode_index"]): dict(row) for row in summary["episodes"]}

    episodes_path = trimmed_root / "meta" / "episodes.jsonl"
    if episodes_path.exists():
        return {
            int(row["episode_index"]): {
                "episode_index": int(row["episode_index"]),
                "trimmed_length": int(row["length"]),
            }
            for row in load_jsonl(episodes_path)
        }

    raise FileNotFoundError(f"Could not find trim metadata under {trimmed_root}")


def trim_frame_interval(
    *,
    episode_index: int,
    raw_length: int,
    trim_report: dict[int, dict[str, Any]],
) -> tuple[int, int, str]:
    if raw_length <= 0:
        raise ValueError(f"Episode {episode_index} has invalid length {raw_length}")
    row = trim_report.get(int(episode_index))
    if row is None:
        return 0, raw_length, "full_failure_episode"
    if "trimmed" in row and not bool(row["trimmed"]):
        return 0, raw_length, "full_failure_episode"
    if "trim_end_frame" in row and "trim_start_frame" in row:
        start = int(row["trim_start_frame"])
        end = int(row["trim_end_frame"])
        if start < 0 or start >= end or end > raw_length:
            raise ValueError(
                f"Invalid trim interval for episode {episode_index}: "
                f"[{start}, {end}) with length {raw_length}"
            )
        return start, end, "trimmed_failure_window"
    if "trimmed_length" in row:
        trimmed_length = int(row["trimmed_length"])
        if not 0 < trimmed_length <= raw_length:
            raise ValueError(
                f"Invalid trimmed_length {trimmed_length} for episode {episode_index} "
                f"with length {raw_length}"
            )
        return 0, trimmed_length, "trimmed_failure_window"
    if "new_length" in row:
        new_length = int(row["new_length"])
        if not 0 < new_length <= raw_length:
            raise ValueError(
                f"Invalid new_length {new_length} for episode {episode_index} "
                f"with length {raw_length}"
            )
        return 0, new_length, "trimmed_failure_window"
    return 0, raw_length, "full_failure_episode"


def init_base(args: argparse.Namespace) -> None:
    dataset_root = args.dataset_root.expanduser().resolve()
    dataset_id = args.dataset_id
    eve_root = args.eve_root.expanduser().resolve() if args.eve_root else dataset_root / "eve"
    assert_schema_compatible(eve_root)
    episodes = load_lerobot_episodes(dataset_root)
    info = load_lerobot_info(dataset_root)
    round_id = make_round_id(dataset_id, args.collection_round)
    round_row = make_round_row(
        dataset_root=dataset_root,
        dataset_id=dataset_id,
        task_name=args.task_name,
        source_type=args.source_type,
        source_policy=args.source_policy,
        collection_round=args.collection_round,
        source_checkpoint=None,
        source_checkpoint_sha256=None,
        dataset_fingerprint_sha256=args.dataset_fingerprint_sha256,
        parent_round_ids=args.parent_round_ids,
        config_path=args.config_path,
        code_commit=args.code_commit,
        created_at=args.created_at,
        dataset_uri=args.dataset_uri,
    )
    split_map_path = getattr(args, "split_map", None)
    split_map = load_episode_split_map(split_map_path)

    rows: list[dict[str, Any]] = []
    for ep in episodes:
        ep_idx = int(ep["episode_index"])
        task = first_task(ep)
        outcome = "failure" if is_failure_task(task, args.failure_phrase) else "success"
        if args.force_success:
            outcome = "success"
        episode_split = resolve_episode_split(
            split_map=split_map,
            dataset_id=dataset_id,
            episode_index=ep_idx,
            default_split=args.split,
        )
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "episode_id": make_episode_id(dataset_id, ep_idx),
                "round_id": round_id,
                "dataset_id": dataset_id,
                "dataset_root": str(dataset_root),
                "episode_index": ep_idx,
                "task_name": args.task_name,
                "task": strip_failure_phrase(task, args.failure_phrase),
                "source_type": args.source_type,
                "source_policy": args.source_policy,
                "collection_round": int(args.collection_round),
                "episode_outcome": outcome,
                "outcome_source": "forced_success" if args.force_success else "task_marker",
                "failure_type": None if outcome == "success" else args.default_failure_type,
                "seed": None,
                "length": int(ep["length"]),
                "fps": int(info["fps"]),
                "split": episode_split,
            }
        )

    with sidecar_write_lock(eve_root):
        write_schema(eve_root)
        validate_round_parent_refs(eve_root, [round_row])
        append_immutable_jsonl_group(
            [
                (
                    eve_root / "round_meta.jsonl",
                    [round_row],
                    ("round_id",),
                    ("created_at", "source_checkpoint"),
                ),
                (
                    eve_root / "episode_meta.jsonl",
                    rows,
                    ("episode_id",),
                    ("dataset_root",),
                ),
            ]
        )
    write_json(
        eve_root / "reports" / f"init_base_{dataset_id}_r{int(args.collection_round)}.json",
        {
            "schema_version": SCHEMA_VERSION,
            "round_id": round_id,
            "dataset_id": dataset_id,
            "dataset_root": str(dataset_root),
            "dataset_fingerprint_sha256": round_row["dataset_fingerprint_sha256"],
            "episodes": len(rows),
            "successes": sum(1 for row in rows if row["episode_outcome"] == "success"),
            "failures": sum(1 for row in rows if row["episode_outcome"] == "failure"),
            "source_type": args.source_type,
            "collection_round": int(args.collection_round),
            "split_map": (
                str(split_map_path.expanduser().resolve())
                if split_map_path is not None
                else None
            ),
        },
    )
    print(f"[eve] initialized {eve_root} with {len(rows)} episode rows from {dataset_root}")


def append_rollout(args: argparse.Namespace) -> None:
    eve_root = args.base_eve_root.expanduser().resolve()
    rollout_root = args.rollout_root.expanduser().resolve()
    trimmed_root = args.trimmed_event_root.expanduser().resolve() if args.trimmed_event_root else None
    dataset_id = args.dataset_id
    assert_schema_compatible(eve_root)
    if (
        args.annotation_confidence is not None
        and not 0.0 <= args.annotation_confidence <= 1.0
    ):
        raise ValueError("annotation_confidence must be in [0, 1]")

    episodes = load_lerobot_episodes(rollout_root)
    info = load_lerobot_info(rollout_root)
    summary = load_collection_summary(rollout_root)
    attempt_by_ep = attempt_log_by_episode(summary, episode_count=len(episodes))
    episode_indices = [int(episode["episode_index"]) for episode in episodes]
    require_explicit_outcomes = bool(
        getattr(args, "require_explicit_outcomes", False)
    )
    structured_outcomes = load_episode_outcome_ledger(
        rollout_root,
        required=require_explicit_outcomes,
        expected_episode_indices=episode_indices,
    )
    trim_report = load_trim_report(trimmed_root)
    round_id = make_round_id(dataset_id, args.collection_round)
    round_row = make_round_row(
        dataset_root=rollout_root,
        dataset_id=dataset_id,
        task_name=args.task_name,
        source_type="policy_rollout",
        source_policy=args.source_policy,
        collection_round=args.collection_round,
        source_checkpoint=args.source_checkpoint,
        source_checkpoint_sha256=args.source_checkpoint_sha256,
        dataset_fingerprint_sha256=args.dataset_fingerprint_sha256,
        parent_round_ids=args.parent_round_ids,
        config_path=args.config_path,
        code_commit=args.code_commit,
        created_at=args.created_at,
        dataset_uri=args.dataset_uri,
    )
    split_map_path = getattr(args, "split_map", None)
    split_map = load_episode_split_map(split_map_path)

    episode_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    for ep in episodes:
        ep_idx = int(ep["episode_index"])
        task_raw = first_task(ep)
        task = strip_failure_phrase(task_raw, args.failure_phrase)
        attempt = attempt_by_ep.get(ep_idx, {})
        structured = structured_outcomes.get(ep_idx)
        task_marks_failure = is_failure_task(task_raw, args.failure_phrase)
        if structured is not None:
            outcome = str(structured["outcome"])
            outcome_source = "structured_outcome_ledger"
            if task_marks_failure and outcome != "failure":
                raise ValueError(
                    f"Episode {ep_idx} task marker says failure but structured "
                    f"outcome says {outcome!r}"
                )
            if "success" in attempt and bool(attempt["success"]) != (
                outcome == "success"
            ):
                raise ValueError(
                    f"Episode {ep_idx} collection summary disagrees with the "
                    "structured outcome ledger"
                )
        elif task_marks_failure:
            outcome = "failure"
            outcome_source = "task_marker"
        elif "success" in attempt:
            outcome = "success" if bool(attempt["success"]) else "failure"
            outcome_source = "collection_summary"
        else:
            outcome = "success"
            outcome_source = "assumed_success"

        length = int(ep["length"])
        failure_type = None if outcome == "success" else args.default_failure_type
        episode_split = resolve_episode_split(
            split_map=split_map,
            dataset_id=dataset_id,
            episode_index=ep_idx,
            default_split=args.split,
        )
        episode_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "episode_id": make_episode_id(dataset_id, ep_idx),
                "round_id": round_id,
                "dataset_id": dataset_id,
                "dataset_root": str(rollout_root),
                "episode_index": ep_idx,
                "task_name": args.task_name,
                "task": task,
                "source_type": "policy_rollout",
                "source_policy": args.source_policy,
                "collection_round": int(args.collection_round),
                "episode_outcome": outcome,
                "outcome_source": outcome_source,
                "failure_type": failure_type,
                "seed": (
                    structured.get("seed", attempt.get("seed"))
                    if structured is not None
                    else attempt.get("seed")
                ),
                "attempt_index": (
                    structured.get(
                        "attempt_index", attempt.get("attempt_index")
                    )
                    if structured is not None
                    else attempt.get("attempt_index")
                ),
                "length": length,
                "fps": int(info["fps"]),
                "split": episode_split,
            }
        )

        if outcome != "failure":
            continue

        start_frame, end_frame, window_rule = trim_frame_interval(
            episode_index=ep_idx,
            raw_length=length,
            trim_report=trim_report,
        )
        if end_frame <= start_frame:
            raise ValueError(
                f"Failure event for episode {ep_idx} has empty interval "
                f"[{start_frame}, {end_frame})"
            )
        event_id = f"{dataset_id}_ep{ep_idx:06d}_failure_event"
        annotation_confidence = args.annotation_confidence
        if annotation_confidence is None:
            if args.annotation_source == "manual":
                annotation_confidence = 1.0
            else:
                annotation_confidence = (
                    0.25 if window_rule != "full_failure_episode" else 0.1
                )
        trim_row = trim_report.get(int(ep_idx), {})
        event_row: dict[str, Any] = {
                "schema_version": SCHEMA_VERSION,
                "event_id": event_id,
                "episode_id": make_episode_id(dataset_id, ep_idx),
                "round_id": round_id,
                "dataset_id": dataset_id,
                "dataset_root": str(rollout_root),
                "episode_index": ep_idx,
                "task_name": args.task_name,
                "task": task,
                "event_type": "failure_event",
                "event_level": "failure",
                "event_label": failure_type,
                "effector": "global",
                "event_outcome": "failure",
                "failure_type": failure_type,
                "source_policy": args.source_policy,
                "collection_round": int(args.collection_round),
                "start_frame": start_frame,
                "end_frame": int(end_frame),
                "failure_frame": None,
                "source_window_rule": window_rule,
                "action_loss": args.failure_action_loss,
                "sample_role": "failure_context",
                "annotation": {
                    "source": args.annotation_source,
                    "method": args.annotation_method or window_rule,
                    "version": args.annotation_version,
                    "confidence": float(annotation_confidence),
                },
                "split": episode_split,
            }
        for field in (
            "core_start_frame",
            "core_end_frame",
            "event_center_frame",
            "jump_ratio",
            "width",
            "baseline_median",
            "detection",
        ):
            if field in trim_row and trim_row[field] is not None:
                event_row[field] = trim_row[field]
        if "core_start_frame" not in event_row and (end_frame - start_frame) > 0:
            # Single-window events: core anchor = interval start.
            event_row["core_start_frame"] = int(start_frame)
            event_row["core_end_frame"] = int(end_frame)
        event_rows.append(event_row)

    with sidecar_write_lock(eve_root):
        write_schema(eve_root)
        validate_round_parent_refs(eve_root, [round_row])
        append_immutable_jsonl_group(
            [
                (
                    eve_root / "round_meta.jsonl",
                    [round_row],
                    ("round_id",),
                    ("created_at", "source_checkpoint"),
                ),
                (
                    eve_root / "episode_meta.jsonl",
                    episode_rows,
                    ("episode_id",),
                    ("dataset_root",),
                ),
                (
                    eve_root / "event_meta.jsonl",
                    event_rows,
                    ("event_id",),
                    ("dataset_root",),
                ),
            ]
        )
    write_json(
        eve_root / "reports" / f"append_rollout_{dataset_id}_r{int(args.collection_round)}.json",
        {
            "schema_version": SCHEMA_VERSION,
            "round_id": round_id,
            "dataset_id": dataset_id,
            "rollout_root": str(rollout_root),
            "trimmed_event_root": str(trimmed_root) if trimmed_root is not None else None,
            "episodes": len(episode_rows),
            "successes": sum(1 for row in episode_rows if row["episode_outcome"] == "success"),
            "failures": sum(1 for row in episode_rows if row["episode_outcome"] == "failure"),
            "failure_events": len(event_rows),
            "source_policy": args.source_policy,
            "collection_round": int(args.collection_round),
            "dataset_fingerprint_sha256": round_row["dataset_fingerprint_sha256"],
            "outcome_ledger": (
                str(rollout_root / "meta" / "episode_outcomes.jsonl")
                if structured_outcomes
                else None
            ),
            "outcome_ledger_sha256": file_sha256(
                rollout_root / "meta" / "episode_outcomes.jsonl"
            ),
            "require_explicit_outcomes": require_explicit_outcomes,
            "split_map": (
                str(split_map_path.expanduser().resolve())
                if split_map_path is not None
                else None
            ),
        },
    )
    print(
        f"[eve] appended rollout {dataset_id}: episodes={len(episode_rows)} "
        f"failure_events={len(event_rows)} into {eve_root}"
    )


def selected_dataset(row: dict[str, Any], dataset_ids: set[str] | None) -> bool:
    return dataset_ids is None or str(row.get("dataset_id")) in dataset_ids


def parse_optional_set(values: list[str] | None) -> set[str] | None:
    if values is None:
        return None
    return {str(value) for value in values}


def row_matches_manifest_selection(
    row: dict[str, Any],
    *,
    collection_rounds: set[int] | None,
    splits: set[str] | None,
) -> bool:
    if collection_rounds is not None:
        round_value = row.get("collection_round", row.get("collection_iter"))
        if round_value is None or int(round_value) not in collection_rounds:
            return False
    if splits is not None and str(row.get("split", "train")) not in splits:
        return False
    return True


def canonical_ledger_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = [
        {
            key: value
            for key, value in row.items()
            if key not in {"dataset_root", "source_checkpoint"}
        }
        for row in rows
    ]
    return sorted(
        normalized,
        key=lambda row: json.dumps(row, ensure_ascii=True, sort_keys=True),
    )


def event_episode_outcome(event: dict[str, Any]) -> str:
    """Return the linked episode outcome declared by an event row."""

    outcome = event.get("episode_outcome")
    if outcome is None and event.get("event_outcome") in {"success", "failure"}:
        # v0.2 failure events predate the explicit episode_outcome field.
        outcome = event["event_outcome"]
    if outcome not in {"success", "failure"}:
        event_id = str(event.get("event_id", "<unknown>"))
        raise ValueError(
            f"Event {event_id} must declare episode_outcome when "
            "event_outcome is not success or failure"
        )
    return str(outcome)


def validate_event_episode_link(
    event: dict[str, Any], episode: dict[str, Any] | None
) -> dict[str, Any]:
    event_id = str(event.get("event_id", "<unknown>"))
    if episode is None:
        raise ValueError(f"Event {event_id} references a missing episode")

    dataset_id = str(event["dataset_id"])
    episode_index = int(event["episode_index"])
    collection_round = int(event.get("collection_round", -1))
    expected_episode_id = event.get(
        "episode_id", make_episode_id(dataset_id, episode_index)
    )
    expected_round_id = event.get(
        "round_id", make_round_id(dataset_id, collection_round)
    )
    episode_values = {
        "episode_id": episode.get(
            "episode_id",
            make_episode_id(
                str(episode["dataset_id"]), int(episode["episode_index"])
            ),
        ),
        "dataset_id": episode.get("dataset_id"),
        "episode_index": int(episode["episode_index"]),
        "round_id": episode.get(
            "round_id",
            make_round_id(
                str(episode["dataset_id"]),
                int(episode.get("collection_round", -1)),
            ),
        ),
        "collection_round": int(episode.get("collection_round", -1)),
        "split": str(episode.get("split", "train")),
    }
    expected_values = {
        "episode_id": expected_episode_id,
        "dataset_id": dataset_id,
        "episode_index": episode_index,
        "round_id": expected_round_id,
        "collection_round": collection_round,
        "split": str(event.get("split", "train")),
    }
    mismatches = {
        field: (expected_values[field], episode_values[field])
        for field in expected_values
        if expected_values[field] != episode_values[field]
    }
    if mismatches:
        raise ValueError(f"Event {event_id} does not match its episode: {mismatches}")

    expected_episode_outcome = event_episode_outcome(event)
    observed_episode_outcome = episode.get("episode_outcome")
    if observed_episode_outcome != expected_episode_outcome:
        raise ValueError(
            f"Event {event_id} episode_outcome does not match its episode: "
            f"expected={expected_episode_outcome!r}, "
            f"observed={observed_episode_outcome!r}"
        )
    event_outcome = event.get("event_outcome", expected_episode_outcome)
    if event_outcome not in {"success", "failure", "unknown"}:
        raise ValueError(
            f"Event {event_id} has unsupported event_outcome {event_outcome!r}"
        )
    if (
        event_outcome in {"success", "failure"}
        and event_outcome != expected_episode_outcome
    ):
        raise ValueError(
            f"Event {event_id} event_outcome does not match episode_outcome: "
            f"event_outcome={event_outcome!r}, "
            f"episode_outcome={expected_episode_outcome!r}"
        )

    start_frame = int(event["start_frame"])
    end_frame = int(event["end_frame"])
    episode_length = int(episode["length"])
    if start_frame < 0 or start_frame >= end_frame or end_frame > episode_length:
        raise ValueError(
            f"Event {event_id} interval [{start_frame}, {end_frame}) exceeds "
            f"episode length {episode_length}"
        )

    core_start = event.get("core_start_frame")
    core_end = event.get("core_end_frame")
    if (core_start is None) != (core_end is None):
        raise ValueError(
            f"Event {event_id} must provide both core_start_frame and "
            "core_end_frame"
        )
    if core_start is not None:
        core_start = int(core_start)
        core_end = int(core_end)
        if (
            core_start < start_frame
            or core_start >= core_end
            or core_end > end_frame
        ):
            raise ValueError(
                f"Event {event_id} core interval [{core_start}, {core_end}) "
                f"must lie inside [{start_frame}, {end_frame})"
            )

    event_weight = event.get("event_weight")
    if event_weight is not None and (
        isinstance(event_weight, bool)
        or not isinstance(event_weight, (int, float))
        or not math.isfinite(float(event_weight))
        or not 0.0 <= float(event_weight) <= 1.0
    ):
        raise ValueError(f"Event {event_id} event_weight must be in [0, 1]")
    for field in ("absolute_confidence", "episode_sampling_weight"):
        value = event.get(field)
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0.0 <= float(value) <= 1.0
        ):
            raise ValueError(f"Event {event_id} {field} must be in [0, 1]")
    return episode


def event_matches_type(
    event: dict[str, Any], event_types: set[str] | None
) -> bool:
    return event_types is None or str(event.get("event_type", "")) in event_types


def event_manifest_sample(
    row: dict[str, Any],
    matching_episode: dict[str, Any],
    *,
    episode_outcome: str,
    success_sample_stride: int,
    failure_sample_stride: int,
    failure_action_loss: str,
    batch_role: str | None = None,
    auxiliary_window_selection: str | None = "core_start_anchor",
) -> dict[str, Any]:
    dataset_id = str(row["dataset_id"])
    episode_index = int(row["episode_index"])
    collection_round = int(row.get("collection_round", -1))
    episode_id = row.get(
        "episode_id", make_episode_id(dataset_id, episode_index)
    )
    event_outcome = str(row.get("event_outcome", episode_outcome))
    event_type = str(
        row.get(
            "event_type",
            "interaction_candidate"
            if event_outcome == "unknown"
            else f"{episode_outcome}_event",
        )
    )
    is_candidate = (
        event_outcome == "unknown" or event_type == "interaction_candidate"
    )
    if episode_outcome == "success":
        action_loss = "disabled" if batch_role == "auxiliary" else "enabled"
        if batch_role == "auxiliary":
            default_role = "success_auxiliary"
        elif is_candidate:
            default_role = "success_candidate"
        else:
            default_role = "success_context"
        sample_stride = success_sample_stride
    else:
        action_loss = (
            "disabled"
            if is_candidate
            else row.get("action_loss", failure_action_loss)
        )
        default_role = (
            "failure_candidate" if is_candidate else "failure_context"
        )
        sample_stride = failure_sample_stride

    sample = {
        "sample_type": "event",
        "sample_id": row["event_id"],
        "event_id": row["event_id"],
        "dataset_id": dataset_id,
        "dataset_root": row.get(
            "dataset_root", matching_episode["dataset_root"]
        ),
        "episode_id": episode_id,
        "episode_index": episode_index,
        "round_id": row.get(
            "round_id", make_round_id(dataset_id, collection_round)
        ),
        "collection_round": collection_round,
        "task": row.get("task", matching_episode.get("task", "")),
        "episode_outcome": episode_outcome,
        "event_outcome": event_outcome,
        "event_type": event_type,
        "event_level": row.get(
            "event_level",
            "candidate" if is_candidate else episode_outcome,
        ),
        "event_label": row.get("event_label", row.get("failure_type")),
        "effector": row.get("effector", "global"),
        "failure_type": row.get("failure_type"),
        "start_frame": int(row["start_frame"]),
        "end_frame": int(row["end_frame"]),
        "failure_frame": row.get("failure_frame"),
        "action_loss": action_loss,
        "sample_role": row.get("sample_role", default_role),
        "sample_stride": int(sample_stride),
        "annotation": row.get("annotation"),
        "split": row.get("split", "train"),
    }
    if batch_role is not None:
        sample["batch_role"] = batch_role
    if batch_role == "auxiliary" and auxiliary_window_selection is not None:
        sample["window_selection"] = auxiliary_window_selection
    for field in (
        "absolute_confidence",
        "episode_sampling_weight",
        "event_weight",
        "source_window_rule",
        "core_start_frame",
        "core_end_frame",
        "core_interval",
    ):
        if field in row:
            sample[field] = row[field]
    return sample


def _build_manifest_locked(args: argparse.Namespace, eve_root: Path) -> None:
    round_rows = load_jsonl(eve_root / "round_meta.jsonl")
    episode_rows = load_jsonl(eve_root / "episode_meta.jsonl")
    event_rows = load_jsonl(eve_root / "event_meta.jsonl")
    episode_by_id = {
        str(
            row.get(
                "episode_id",
                make_episode_id(str(row["dataset_id"]), int(row["episode_index"])),
            )
        ): row
        for row in episode_rows
    }
    include_outcomes = {str(item) for item in args.include_outcomes}
    invalid_outcomes = include_outcomes - {"success", "failure"}
    if invalid_outcomes:
        raise ValueError(f"Unsupported outcomes: {sorted(invalid_outcomes)}")
    success_dataset_ids = parse_optional_set(args.success_dataset_ids)
    success_auxiliary_dataset_ids = parse_optional_set(
        getattr(args, "success_auxiliary_dataset_ids", None)
    )
    failure_dataset_ids = parse_optional_set(args.failure_dataset_ids)
    if success_auxiliary_dataset_ids is not None:
        if success_dataset_ids is None:
            raise ValueError(
                "`success-dataset-ids` must be explicit when "
                "`success-auxiliary-dataset-ids` is used."
            )
        overlap = success_dataset_ids & success_auxiliary_dataset_ids
        if overlap:
            raise ValueError(
                "Primary and auxiliary success dataset IDs must be disjoint: "
                f"{sorted(overlap)}"
            )
    collection_rounds = (
        None
        if args.collection_rounds is None
        else {int(item) for item in args.collection_rounds}
    )
    splits = parse_optional_set(args.splits)
    event_types = parse_optional_set(getattr(args, "event_types", None))
    success_sample_mode = getattr(args, "success_sample_mode", "episode_only")
    if success_sample_mode not in {"episode_only", "event_only", "both"}:
        raise ValueError(
            f"Unsupported success_sample_mode: {success_sample_mode!r}"
        )
    failure_window_selection = getattr(
        args, "failure_window_selection", "core_start_anchor"
    )
    if failure_window_selection not in {"core_start_anchor", "sliding"}:
        raise ValueError(
            "Unsupported failure_window_selection: "
            f"{failure_window_selection!r}"
        )
    failure_source_window_rules = parse_optional_set(
        getattr(args, "failure_source_window_rules", None)
    )
    include_sample_ids = parse_optional_set(args.include_sample_ids)
    exclude_sample_ids = parse_optional_set(args.exclude_sample_ids) or set()
    if include_sample_ids is not None:
        overlap = include_sample_ids & exclude_sample_ids
        if overlap:
            raise ValueError(
                f"Sample IDs cannot be both included and excluded: {sorted(overlap)}"
            )

    samples: list[dict[str, Any]] = []
    used_episode_rows: dict[str, dict[str, Any]] = {}
    used_event_rows: dict[str, dict[str, Any]] = {}

    if "success" in include_outcomes:
        if (
            success_sample_mode in {"episode_only", "both"}
            or success_auxiliary_dataset_ids is not None
        ):
            for row in episode_rows:
                if row.get("episode_outcome") != "success":
                    continue
                is_primary = selected_dataset(row, success_dataset_ids)
                is_auxiliary = (
                    success_auxiliary_dataset_ids is not None
                    and selected_dataset(row, success_auxiliary_dataset_ids)
                )
                include_primary = (
                    is_primary
                    and (
                        success_auxiliary_dataset_ids is not None
                        or success_sample_mode in {"episode_only", "both"}
                    )
                )
                include_auxiliary = (
                    is_auxiliary
                    and success_sample_mode in {"episode_only", "both"}
                )
                if not include_primary and not include_auxiliary:
                    continue
                if not row_matches_manifest_selection(
                    row, collection_rounds=collection_rounds, splits=splits
                ):
                    continue
                dataset_id = str(row["dataset_id"])
                episode_index = int(row["episode_index"])
                collection_round = int(row.get("collection_round", -1))
                episode_id = row.get(
                    "episode_id", make_episode_id(dataset_id, episode_index)
                )
                samples.append(
                    {
                        "sample_type": "episode",
                        "sample_id": (
                            f"{dataset_id}_ep{episode_index:06d}_success_aux"
                            if include_auxiliary
                            else f"{dataset_id}_ep{episode_index:06d}"
                        ),
                        "dataset_id": dataset_id,
                        "dataset_root": row["dataset_root"],
                        "episode_id": episode_id,
                        "episode_index": episode_index,
                        "round_id": row.get(
                            "round_id",
                            make_round_id(dataset_id, collection_round),
                        ),
                        "collection_round": collection_round,
                        "task": row.get("task", ""),
                        "episode_outcome": "success",
                        "event_outcome": "success",
                        "start_frame": 0,
                        "end_frame": int(row["length"]),
                        "action_loss": (
                            "disabled" if include_auxiliary else "enabled"
                        ),
                        "sample_role": (
                            "success_auxiliary"
                            if include_auxiliary
                            else "success_episode"
                        ),
                        "batch_role": (
                            "auxiliary" if include_auxiliary else "primary"
                        ),
                        "sample_stride": int(args.success_sample_stride),
                        "split": row.get("split", "train"),
                    }
                )
                used_episode_rows[str(episode_id)] = row
        if success_sample_mode in {"event_only", "both"}:
            for row in event_rows:
                if not event_matches_type(row, event_types):
                    continue
                if event_episode_outcome(row) != "success":
                    continue
                is_primary = selected_dataset(row, success_dataset_ids)
                is_auxiliary = (
                    success_auxiliary_dataset_ids is not None
                    and selected_dataset(row, success_auxiliary_dataset_ids)
                )
                include_primary = (
                    is_primary and success_auxiliary_dataset_ids is None
                )
                if not include_primary and not is_auxiliary:
                    continue
                if not row_matches_manifest_selection(
                    row, collection_rounds=collection_rounds, splits=splits
                ):
                    continue
                dataset_id = str(row["dataset_id"])
                episode_index = int(row["episode_index"])
                episode_id = row.get(
                    "episode_id", make_episode_id(dataset_id, episode_index)
                )
                matching_episode = validate_event_episode_link(
                    row, episode_by_id.get(str(episode_id))
                )
                samples.append(
                    event_manifest_sample(
                        row,
                        matching_episode,
                        episode_outcome="success",
                        success_sample_stride=int(args.success_sample_stride),
                        failure_sample_stride=int(args.failure_sample_stride),
                        failure_action_loss=args.failure_action_loss,
                        batch_role=(
                            "auxiliary" if is_auxiliary else "primary"
                        ),
                    )
                )
                used_event_rows[str(row["event_id"])] = row
                used_episode_rows[str(episode_id)] = matching_episode

    if "failure" in include_outcomes:
        if args.failure_sample_mode in {"event_only", "both"}:
            for row in event_rows:
                if not event_matches_type(row, event_types):
                    continue
                if event_episode_outcome(row) != "failure":
                    continue
                if not selected_dataset(row, failure_dataset_ids):
                    continue
                if (
                    failure_source_window_rules is not None
                    and str(row.get("source_window_rule", ""))
                    not in failure_source_window_rules
                ):
                    continue
                if not row_matches_manifest_selection(
                    row, collection_rounds=collection_rounds, splits=splits
                ):
                    continue
                dataset_id = str(row["dataset_id"])
                episode_index = int(row["episode_index"])
                collection_round = int(row.get("collection_round", -1))
                episode_id = row.get(
                    "episode_id", make_episode_id(dataset_id, episode_index)
                )
                matching_episode = validate_event_episode_link(
                    row, episode_by_id.get(str(episode_id))
                )
                samples.append(
                    event_manifest_sample(
                        row,
                        matching_episode,
                        episode_outcome="failure",
                        success_sample_stride=int(args.success_sample_stride),
                        failure_sample_stride=int(args.failure_sample_stride),
                        failure_action_loss=args.failure_action_loss,
                        batch_role="auxiliary",
                        auxiliary_window_selection=(
                            None
                            if failure_window_selection == "sliding"
                            else "core_start_anchor"
                        ),
                    )
                )
                used_event_rows[str(row["event_id"])] = row
                used_episode_rows[str(episode_id)] = matching_episode
        if args.failure_sample_mode in {"full_episode", "both"}:
            for row in episode_rows:
                if row.get("episode_outcome") != "failure":
                    continue
                if not selected_dataset(row, failure_dataset_ids):
                    continue
                if not row_matches_manifest_selection(
                    row, collection_rounds=collection_rounds, splits=splits
                ):
                    continue
                dataset_id = str(row["dataset_id"])
                episode_index = int(row["episode_index"])
                collection_round = int(row.get("collection_round", -1))
                episode_id = row.get(
                    "episode_id", make_episode_id(dataset_id, episode_index)
                )
                samples.append(
                    {
                        "sample_type": "episode",
                        "sample_id": f"{dataset_id}_ep{episode_index:06d}_failure_full",
                        "dataset_id": dataset_id,
                        "dataset_root": row["dataset_root"],
                        "episode_id": episode_id,
                        "episode_index": episode_index,
                        "round_id": row.get(
                            "round_id", make_round_id(dataset_id, collection_round)
                        ),
                        "collection_round": collection_round,
                        "task": row.get("task", ""),
                        "episode_outcome": "failure",
                        "event_outcome": "failure",
                        "failure_type": row.get("failure_type"),
                        "start_frame": 0,
                        "end_frame": int(row["length"]),
                        "action_loss": args.failure_action_loss,
                        "sample_role": "failure_episode",
                        "sample_stride": int(args.failure_sample_stride),
                        "split": row.get("split", "train"),
                    }
                )
                used_episode_rows[str(episode_id)] = row

    available_sample_ids = {str(sample["sample_id"]) for sample in samples}
    if include_sample_ids is not None:
        missing_ids = include_sample_ids - available_sample_ids
        if missing_ids:
            raise ValueError(
                "Requested sample IDs are absent after dataset/outcome filters: "
                f"{sorted(missing_ids)}"
            )
        samples = [
            sample for sample in samples if str(sample["sample_id"]) in include_sample_ids
        ]
    samples = [
        sample for sample in samples if str(sample["sample_id"]) not in exclude_sample_ids
    ]
    samples.sort(key=lambda sample: str(sample["sample_id"]))

    sample_ids = [str(sample["sample_id"]) for sample in samples]
    if len(sample_ids) != len(set(sample_ids)):
        duplicates = sorted(
            sample_id for sample_id in set(sample_ids) if sample_ids.count(sample_id) > 1
        )
        raise ValueError(f"Duplicate sample IDs in manifest selection: {duplicates}")

    selected_episode_ids = {str(sample["episode_id"]) for sample in samples}
    selected_event_ids = {
        str(sample["event_id"])
        for sample in samples
        if sample.get("event_id") is not None
    }
    selected_episode_rows = [
        row
        for episode_id, row in used_episode_rows.items()
        if episode_id in selected_episode_ids
    ]
    selected_event_rows = [
        row for event_id, row in used_event_rows.items() if event_id in selected_event_ids
    ]
    selected_round_ids = {str(sample["round_id"]) for sample in samples}
    selected_round_rows = [
        row for row in round_rows if str(row.get("round_id")) in selected_round_ids
    ]
    missing_round_ids = selected_round_ids - {
        str(row.get("round_id")) for row in selected_round_rows
    }
    if missing_round_ids:
        raise ValueError(f"Manifest samples reference missing rounds: {sorted(missing_round_ids)}")
    round_by_id = {str(row["round_id"]): row for row in selected_round_rows}
    for sample in samples:
        round_row = round_by_id[str(sample["round_id"])]
        expected = (
            str(sample["dataset_id"]),
            int(sample["collection_round"]),
        )
        observed = (
            str(round_row.get("dataset_id")),
            int(round_row.get("collection_round", -1)),
        )
        if expected != observed:
            raise ValueError(
                f"Sample {sample['sample_id']} does not match round "
                f"{sample['round_id']}: expected={expected}, observed={observed}"
            )

    dataset_roots = {
        str(sample["dataset_id"]): str(Path(sample["dataset_root"]).expanduser().resolve())
        for sample in samples
    }
    manifest = {
        "format": "EveRobotTrainManifest",
        "schema_version": SCHEMA_VERSION,
        "manifest_name": args.manifest_name,
        "eve_root": str(eve_root),
        "frame_interval": "half_open",
        "selection": {
            "include_outcomes": sorted(include_outcomes),
            "success_dataset_ids": sorted(success_dataset_ids) if success_dataset_ids else None,
            "failure_dataset_ids": sorted(failure_dataset_ids) if failure_dataset_ids else None,
            "collection_rounds": sorted(collection_rounds) if collection_rounds else None,
            "splits": sorted(splits) if splits else None,
            "include_sample_ids": sorted(include_sample_ids) if include_sample_ids else None,
            "exclude_sample_ids": sorted(exclude_sample_ids) if exclude_sample_ids else None,
            "success_sample_mode": success_sample_mode,
            "success_auxiliary_dataset_ids": (
                sorted(success_auxiliary_dataset_ids)
                if success_auxiliary_dataset_ids
                else None
            ),
            "failure_sample_mode": args.failure_sample_mode,
            "failure_window_selection": failure_window_selection,
            "failure_source_window_rules": (
                sorted(failure_source_window_rules)
                if failure_source_window_rules is not None
                else None
            ),
            "event_types": sorted(event_types) if event_types is not None else None,
        },
        "dataset_roots": dataset_roots,
        "source_round_ids": sorted(selected_round_ids),
        "source_hashes": {
            "round_meta_sha256": sha256_json(canonical_ledger_rows(selected_round_rows)),
            "episode_meta_sha256": sha256_json(canonical_ledger_rows(selected_episode_rows)),
            "event_meta_sha256": sha256_json(canonical_ledger_rows(selected_event_rows)),
        },
        "num_samples": len(samples),
        "samples": samples,
    }
    manifest["manifest_hash"] = compute_manifest_hash(manifest)
    validate_manifest(manifest, strict=True)
    out_path = eve_root / "manifests" / f"{args.manifest_name}.json"
    write_json(out_path, manifest)
    print(f"[eve] wrote manifest {out_path} samples={len(samples)}")


def build_manifest(args: argparse.Namespace) -> None:
    eve_root = args.eve_root.expanduser().resolve()
    assert_schema_compatible(eve_root)
    if not (eve_root / "schema_version.json").exists():
        raise FileNotFoundError(f"EveRobot sidecar is not initialized: {eve_root}")
    with sidecar_write_lock(eve_root):
        _build_manifest_locked(args, eve_root)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    init = subparsers.add_parser("init-base", help="Create Eve episode metadata for an initial LeRobot dataset.")
    init.add_argument("--dataset-root", type=Path, required=True)
    init.add_argument("--dataset-id", type=str, required=True)
    init.add_argument("--eve-root", type=Path, default=None)
    init.add_argument("--task-name", type=str, required=True)
    init.add_argument("--source-type", type=str, default="expert_success")
    init.add_argument("--source-policy", type=str, default="human_or_expert")
    init.add_argument("--collection-round", type=int, default=-1)
    init.add_argument("--split", type=str, default="train")
    init.add_argument(
        "--split-map",
        type=Path,
        default=None,
        help="Frozen JSONL mapping of dataset_id/episode_index to train/val/test.",
    )
    init.add_argument("--failure-phrase", type=str, default=FAILURE_PHRASE)
    init.add_argument("--default-failure-type", type=str, default="unknown_failure")
    init.add_argument("--force-success", action="store_true", default=False)
    init.add_argument("--dataset-uri", type=str, default=None)
    init.add_argument("--dataset-fingerprint-sha256", type=str, default=None)
    init.add_argument("--parent-round-ids", nargs="+", default=None)
    init.add_argument("--config-path", type=Path, default=None)
    init.add_argument("--code-commit", type=str, default=None)
    init.add_argument("--created-at", type=str, default=None)
    init.set_defaults(func=init_base)

    append = subparsers.add_parser("append-rollout", help="Append rollout provenance and failure events.")
    append.add_argument("--base-eve-root", type=Path, required=True)
    append.add_argument("--rollout-root", type=Path, required=True)
    append.add_argument("--trimmed-event-root", type=Path, default=None)
    append.add_argument("--dataset-id", type=str, required=True)
    append.add_argument("--task-name", type=str, required=True)
    append.add_argument("--source-policy", type=str, required=True)
    append.add_argument("--source-checkpoint", type=str, default=None)
    append.add_argument("--source-checkpoint-sha256", type=str, default=None)
    append.add_argument("--collection-round", type=int, required=True)
    append.add_argument("--split", type=str, default="train")
    append.add_argument(
        "--split-map",
        type=Path,
        default=None,
        help="Frozen JSONL mapping of dataset_id/episode_index to train/val/test.",
    )
    append.add_argument("--failure-phrase", type=str, default=FAILURE_PHRASE)
    append.add_argument(
        "--require-explicit-outcomes",
        action="store_true",
        help=(
            "Require meta/episode_outcomes.jsonl with exactly one validated "
            "success/failure row for every rollout episode."
        ),
    )
    append.add_argument("--default-failure-type", type=str, default="unknown_failure")
    append.add_argument("--failure-action-loss", choices=["enabled", "disabled"], default="disabled")
    append.add_argument("--annotation-source", choices=["auto", "manual"], default="auto")
    append.add_argument("--annotation-method", type=str, default=None)
    append.add_argument("--annotation-version", type=str, default="event_window_v1")
    append.add_argument("--annotation-confidence", type=float, default=None)
    append.add_argument("--dataset-uri", type=str, default=None)
    append.add_argument("--dataset-fingerprint-sha256", type=str, default=None)
    append.add_argument("--parent-round-ids", nargs="+", default=None)
    append.add_argument("--config-path", type=Path, default=None)
    append.add_argument("--code-commit", type=str, default=None)
    append.add_argument("--created-at", type=str, default=None)
    append.set_defaults(func=append_rollout)

    manifest = subparsers.add_parser("build-manifest", help="Build a round-specific Eve training manifest.")
    manifest.add_argument("--eve-root", type=Path, required=True)
    manifest.add_argument("--manifest-name", type=str, required=True)
    manifest.add_argument("--include-outcomes", nargs="+", default=["success", "failure"])
    manifest.add_argument("--success-dataset-ids", nargs="+", default=None)
    manifest.add_argument(
        "--success-auxiliary-dataset-ids",
        nargs="+",
        default=None,
        help=(
            "Success datasets sampled as action-disabled auxiliary video. "
            "Requires explicit, disjoint --success-dataset-ids."
        ),
    )
    manifest.add_argument("--failure-dataset-ids", nargs="+", default=None)
    manifest.add_argument(
        "--success-sample-mode",
        choices=["episode_only", "event_only", "both"],
        default="episode_only",
    )
    manifest.add_argument("--failure-sample-mode", choices=["event_only", "full_episode", "both"], default="event_only")
    manifest.add_argument(
        "--failure-window-selection",
        choices=["core_start_anchor", "sliding"],
        default="core_start_anchor",
        help=(
            "Select one anchored window per failure event or expand every legal "
            "training window inside the event interval."
        ),
    )
    manifest.add_argument(
        "--failure-source-window-rules",
        nargs="+",
        default=None,
        help=(
            "Include failure events only when source_window_rule is listed, "
            "for example trimmed_failure_window to exclude full-episode fallbacks."
        ),
    )
    manifest.add_argument(
        "--event-types",
        nargs="+",
        default=None,
        help="Include only event rows whose event_type is listed.",
    )
    manifest.add_argument("--collection-rounds", nargs="+", type=int, default=None)
    manifest.add_argument("--splits", nargs="+", default=None)
    manifest.add_argument("--include-sample-ids", nargs="+", default=None)
    manifest.add_argument("--exclude-sample-ids", nargs="+", default=None)
    manifest.add_argument("--success-sample-stride", type=int, default=1)
    manifest.add_argument("--failure-sample-stride", type=int, default=1)
    manifest.add_argument("--failure-action-loss", choices=["enabled", "disabled"], default="disabled")
    manifest.set_defaults(func=build_manifest)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
