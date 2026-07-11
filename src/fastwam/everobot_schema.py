"""Pure-Python helpers for EveRobot v0.2 training manifests."""

from __future__ import annotations

import copy
import hashlib
import hmac
import json
import os
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = "0.2"
MANIFEST_FORMAT = "EveRobotTrainManifest"
ALLOWED_ACTION_LOSS = frozenset({"enabled", "disabled"})

_RUNTIME_MANIFEST_FIELDS = frozenset(
    {"manifest_hash", "eve_root", "dataset_roots"}
)


def canonical_manifest_payload(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Return the path-independent payload used to identify a manifest.

    Runtime locations are deliberately omitted. The returned object is a deep
    copy, so callers can safely serialize or modify it.
    """

    if not isinstance(manifest, Mapping):
        raise TypeError("manifest must be a mapping")

    payload = copy.deepcopy(dict(manifest))
    for field in _RUNTIME_MANIFEST_FIELDS:
        payload.pop(field, None)

    samples = payload.get("samples")
    if isinstance(samples, list):
        for sample in samples:
            if isinstance(sample, dict):
                sample.pop("dataset_root", None)
    return payload


def canonical_manifest_json(manifest: Mapping[str, Any]) -> str:
    """Serialize a manifest deterministically for hashing."""

    return json.dumps(
        canonical_manifest_payload(manifest),
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_json(payload: Any) -> str:
    """Return a deterministic SHA-256 digest for a JSON-compatible value."""

    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def compute_manifest_hash(manifest: Mapping[str, Any]) -> str:
    """Return the SHA-256 hex digest of a canonical manifest."""

    return sha256_json(canonical_manifest_payload(manifest))


def canonical_manifest_hash(manifest: Mapping[str, Any]) -> str:
    """Compatibility name for :func:`compute_manifest_hash`."""

    return compute_manifest_hash(manifest)


def with_manifest_hash(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Return a deep copy with its canonical ``manifest_hash`` populated."""

    result = copy.deepcopy(dict(manifest))
    result["manifest_hash"] = compute_manifest_hash(result)
    return result


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _validate_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{label} must be a SHA-256 hex digest")
    try:
        bytes.fromhex(value)
    except ValueError as error:
        raise ValueError(f"{label} must be a SHA-256 hex digest") from error
    return value.lower()


def _require_nonempty_id(sample: Mapping[str, Any], field: str, index: int) -> Any:
    value = sample.get(field)
    if value is None or isinstance(value, bool):
        raise ValueError(f"samples[{index}].{field} is required")
    if isinstance(value, str) and not value.strip():
        raise ValueError(f"samples[{index}].{field} must not be empty")
    if not isinstance(value, (str, int)):
        raise ValueError(f"samples[{index}].{field} must be a string or integer")
    return value


def _validate_interval(
    value: Any,
    *,
    label: str,
    lower_bound: int | None = None,
    upper_bound: int | None = None,
) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{label} must be a [start, end] interval")
    start, end = value
    if not _is_int(start) or not _is_int(end):
        raise ValueError(f"{label} bounds must be integers")
    if start < 0 or start >= end:
        raise ValueError(f"{label} must be a non-empty half-open interval")
    if lower_bound is not None and start < lower_bound:
        raise ValueError(f"{label} starts before the sample interval")
    if upper_bound is not None and end > upper_bound:
        raise ValueError(f"{label} ends after the sample interval")
    return start, end


def _validate_sample(sample: Any, index: int) -> str:
    label = f"samples[{index}]"
    if not isinstance(sample, Mapping):
        raise ValueError(f"{label} must be an object")

    sample_id = _require_nonempty_id(sample, "sample_id", index)
    dataset_id = _require_nonempty_id(sample, "dataset_id", index)
    if not isinstance(sample_id, str) or not isinstance(dataset_id, str):
        raise ValueError(f"{label} sample_id and dataset_id must be strings")

    episode_id = _require_nonempty_id(sample, "episode_id", index)
    round_id = _require_nonempty_id(sample, "round_id", index)
    if not isinstance(episode_id, str) or not isinstance(round_id, str):
        raise ValueError(f"{label} episode_id and round_id must be strings")

    sample_type = sample.get("sample_type")
    if sample_type not in {"episode", "event"}:
        raise ValueError(f"{label}.sample_type must be 'episode' or 'event'")
    if sample_type == "event" and not isinstance(sample.get("event_id"), str):
        raise ValueError(f"{label}.event_id is required for event samples")
    if sample_type == "event" and sample.get("effector") not in {
        "left",
        "right",
        "bimanual",
        "global",
    }:
        raise ValueError(f"{label}.effector is invalid")

    episode_index = sample.get("episode_index")
    if not _is_int(episode_index) or episode_index < 0:
        raise ValueError(f"{label}.episode_index must be a non-negative integer")

    collection_round = sample.get("collection_round")
    if not _is_int(collection_round):
        raise ValueError(f"{label}.collection_round must be an integer")

    if "start_frame" not in sample or "end_frame" not in sample:
        raise ValueError(f"{label} requires start_frame and end_frame")
    start, end = _validate_interval(
        [sample["start_frame"], sample["end_frame"]], label=f"{label} frame interval"
    )

    stride = sample.get("sample_stride")
    if not _is_int(stride) or stride <= 0:
        raise ValueError(f"{label}.sample_stride must be a positive integer")

    action_loss = sample.get("action_loss")
    if action_loss not in ALLOWED_ACTION_LOSS:
        allowed = ", ".join(sorted(ALLOWED_ACTION_LOSS))
        raise ValueError(f"{label}.action_loss must be one of: {allowed}")

    if sample.get("action_loss_window") is not None:
        _validate_interval(
            sample["action_loss_window"],
            label=f"{label}.action_loss_window",
            lower_bound=start,
            upper_bound=end,
        )

    failure_frame = sample.get("failure_frame")
    if failure_frame is not None:
        if not _is_int(failure_frame) or not start <= failure_frame < end:
            raise ValueError(f"{label}.failure_frame must lie inside the sample interval")

    annotation = sample.get("annotation")
    if annotation is not None:
        if not isinstance(annotation, Mapping):
            raise ValueError(f"{label}.annotation must be an object")
        confidence = annotation.get("confidence")
        if confidence is not None and (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not 0.0 <= float(confidence) <= 1.0
        ):
            raise ValueError(f"{label}.annotation.confidence must be in [0, 1]")

    valid_intervals = sample.get("valid_intervals")
    if valid_intervals is not None:
        if not isinstance(valid_intervals, list) or not valid_intervals:
            raise ValueError(f"{label}.valid_intervals must be a non-empty list")
        for interval_index, interval in enumerate(valid_intervals):
            _validate_interval(
                interval,
                label=f"{label}.valid_intervals[{interval_index}]",
                lower_bound=start,
                upper_bound=end,
            )

    return sample_id


def _validate_legacy_manifest(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    """Validate the fields required to keep existing v0.1 manifests readable."""

    samples = manifest.get("samples")
    if not isinstance(samples, list):
        raise ValueError("samples must be a list")
    if "num_samples" in manifest:
        num_samples = manifest["num_samples"]
        if not _is_int(num_samples) or num_samples != len(samples):
            raise ValueError("num_samples must equal len(samples)")

    sample_ids: set[str] = set()
    for index, sample in enumerate(samples):
        label = f"samples[{index}]"
        if not isinstance(sample, Mapping):
            raise ValueError(f"{label} must be an object")
        dataset_id = sample.get("dataset_id")
        if not isinstance(dataset_id, str) or not dataset_id:
            raise ValueError(f"{label}.dataset_id is required")
        episode_index = sample.get("episode_index")
        if not _is_int(episode_index) or episode_index < 0:
            raise ValueError(f"{label}.episode_index must be a non-negative integer")
        if "start_frame" in sample and "end_frame" in sample:
            _validate_interval(
                [sample["start_frame"], sample["end_frame"]],
                label=f"{label} frame interval",
            )
        if sample.get("sample_stride") is not None:
            stride = sample["sample_stride"]
            if not _is_int(stride) or stride <= 0:
                raise ValueError(f"{label}.sample_stride must be a positive integer")
        if sample.get("action_loss") is not None:
            if sample["action_loss"] not in ALLOWED_ACTION_LOSS:
                raise ValueError(f"{label}.action_loss is invalid")
        sample_id = sample.get("sample_id")
        if sample_id is not None:
            if not isinstance(sample_id, str) or not sample_id:
                raise ValueError(f"{label}.sample_id must be a non-empty string")
            if sample_id in sample_ids:
                raise ValueError(f"duplicate sample_id: {sample_id!r}")
            sample_ids.add(sample_id)

    dataset_roots = manifest.get("dataset_roots")
    if dataset_roots is not None and not isinstance(dataset_roots, Mapping):
        raise ValueError("dataset_roots must be an object")
    return manifest


def validate_manifest(
    manifest: Mapping[str, Any],
    *,
    strict: bool = True,
    verify_hash: bool | None = None,
) -> Mapping[str, Any]:
    """Validate an EveRobot v0.2 training manifest.

    ``ValueError`` is raised on the first schema violation. Runtime paths are
    intentionally not required because callers can provide explicit roots.
    """

    if verify_hash is None:
        verify_hash = strict

    if not isinstance(manifest, Mapping):
        raise ValueError("manifest must be an object")
    if manifest.get("format") != MANIFEST_FORMAT:
        raise ValueError(f"format must be {MANIFEST_FORMAT!r}")
    if "schema_version" not in manifest:
        raise ValueError("schema_version is required")
    schema_version = str(manifest["schema_version"])
    if schema_version == "0.1":
        return _validate_legacy_manifest(manifest)
    if schema_version != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION!r}")
    if manifest.get("frame_interval") != "half_open":
        raise ValueError("frame_interval must be 'half_open'")
    if not isinstance(manifest.get("manifest_name"), str) or not manifest["manifest_name"]:
        raise ValueError("manifest_name must be a non-empty string")
    if not isinstance(manifest.get("selection"), Mapping):
        raise ValueError("selection must be an object")

    samples = manifest.get("samples")
    if not isinstance(samples, list):
        raise ValueError("samples must be a list")
    if "num_samples" in manifest:
        num_samples = manifest["num_samples"]
        if not _is_int(num_samples) or num_samples != len(samples):
            raise ValueError("num_samples must equal len(samples)")

    sample_ids: set[str] = set()
    for index, sample in enumerate(samples):
        sample_id = _validate_sample(sample, index)
        if sample_id in sample_ids:
            raise ValueError(f"duplicate sample_id: {sample_id!r}")
        sample_ids.add(sample_id)

    dataset_roots = manifest.get("dataset_roots")
    if dataset_roots is not None and not isinstance(dataset_roots, Mapping):
        raise ValueError("dataset_roots must be an object")
    if isinstance(dataset_roots, Mapping):
        missing_roots = {
            str(sample["dataset_id"])
            for sample in samples
            if str(sample["dataset_id"]) not in dataset_roots
            and "dataset_root" not in sample
        }
        if missing_roots:
            raise ValueError(f"dataset roots are missing for: {sorted(missing_roots)}")

    source_round_ids = manifest.get("source_round_ids")
    if not isinstance(source_round_ids, list) or not all(
        isinstance(round_id, str) and round_id for round_id in source_round_ids
    ):
        raise ValueError("source_round_ids must be a list of non-empty strings")
    sample_round_ids = {str(sample["round_id"]) for sample in samples}
    if set(source_round_ids) != sample_round_ids:
        raise ValueError("source_round_ids must match sample round_id values")

    source_hashes = manifest.get("source_hashes")
    if not isinstance(source_hashes, Mapping):
        raise ValueError("source_hashes must be an object")
    for field in (
        "round_meta_sha256",
        "episode_meta_sha256",
        "event_meta_sha256",
    ):
        _validate_sha256(source_hashes.get(field), f"source_hashes.{field}")

    if verify_hash:
        stored_hash = _validate_sha256(manifest.get("manifest_hash"), "manifest_hash")
        expected_hash = compute_manifest_hash(manifest)
        if not hmac.compare_digest(stored_hash.lower(), expected_hash):
            raise ValueError(
                f"manifest_hash mismatch: expected {expected_hash}, got {stored_hash}"
            )

    return manifest


def _absolute_path(value: str | os.PathLike[str]) -> str:
    if not isinstance(value, (str, os.PathLike)):
        raise ValueError(f"dataset root must be path-like, got {type(value).__name__}")
    if not os.fspath(value):
        raise ValueError("dataset root must not be empty")
    return str(Path(value).expanduser().resolve())


def resolve_dataset_roots(
    manifest: Mapping[str, Any],
    overrides: Mapping[str, str | os.PathLike[str]] | None = None,
) -> dict[str, str]:
    """Resolve dataset IDs to absolute roots, applying explicit overrides.

    Resolution precedence is ``overrides``, top-level ``dataset_roots``, then
    legacy per-sample ``dataset_root``. Every referenced dataset must resolve.
    """

    if not isinstance(manifest, Mapping):
        raise ValueError("manifest must be an object")
    stored_roots = manifest.get("dataset_roots", {})
    if not isinstance(stored_roots, Mapping):
        raise ValueError("dataset_roots must be an object")
    if overrides is None:
        overrides = {}
    if not isinstance(overrides, Mapping):
        raise ValueError("overrides must be an object")

    dataset_ids: set[str] = {str(key) for key in stored_roots}
    sample_roots: dict[str, str | os.PathLike[str]] = {}
    samples = manifest.get("samples", [])
    if not isinstance(samples, list):
        raise ValueError("samples must be a list")
    for index, sample in enumerate(samples):
        if not isinstance(sample, Mapping):
            raise ValueError(f"samples[{index}] must be an object")
        dataset_id = sample.get("dataset_id")
        if not isinstance(dataset_id, str) or not dataset_id:
            raise ValueError(f"samples[{index}].dataset_id is required")
        dataset_ids.add(dataset_id)
        if (
            "dataset_root" in sample
            and dataset_id not in overrides
            and dataset_id not in stored_roots
        ):
            previous = sample_roots.get(dataset_id)
            current = sample["dataset_root"]
            if previous is not None and _absolute_path(previous) != _absolute_path(current):
                raise ValueError(
                    f"conflicting per-sample dataset_root values for {dataset_id!r}"
                )
            sample_roots[dataset_id] = current

    unknown_overrides = set(overrides) - dataset_ids
    if unknown_overrides:
        unknown = ", ".join(sorted(str(key) for key in unknown_overrides))
        raise ValueError(f"overrides contain unknown dataset IDs: {unknown}")

    resolved: dict[str, str] = {}
    for dataset_id in sorted(dataset_ids):
        if dataset_id in overrides:
            root = overrides[dataset_id]
        elif dataset_id in stored_roots:
            root = stored_roots[dataset_id]
        elif dataset_id in sample_roots:
            root = sample_roots[dataset_id]
        else:
            raise ValueError(f"no dataset root for dataset_id {dataset_id!r}")
        resolved[dataset_id] = _absolute_path(root)
    return resolved


def resolve_dataset_root(
    manifest: Mapping[str, Any],
    sample: Mapping[str, Any],
    overrides: Mapping[str, str | os.PathLike[str]] | None = None,
) -> str:
    """Resolve the root for one sample using manifest-level precedence."""

    dataset_id = sample.get("dataset_id")
    if not isinstance(dataset_id, str) or not dataset_id:
        raise ValueError("sample.dataset_id is required")
    roots = resolve_dataset_roots(manifest, overrides)
    if dataset_id not in roots:
        raise ValueError(f"no dataset root for dataset_id {dataset_id!r}")
    return roots[dataset_id]


def resolve_manifest_dataset_root(
    manifest: Mapping[str, Any],
    sample: Mapping[str, Any],
    overrides: Mapping[str, str | os.PathLike[str]] | None = None,
) -> str:
    """Resolve one sample without requiring roots for unrelated samples."""

    dataset_id = sample.get("dataset_id")
    if not isinstance(dataset_id, str) or not dataset_id:
        raise ValueError("sample.dataset_id is required")
    if overrides is None:
        overrides = {}
    if not isinstance(overrides, Mapping):
        raise ValueError("overrides must be an object")
    if dataset_id in overrides:
        return _absolute_path(overrides[dataset_id])

    stored_roots = manifest.get("dataset_roots", {})
    if not isinstance(stored_roots, Mapping):
        raise ValueError("dataset_roots must be an object")
    if dataset_id in stored_roots:
        return _absolute_path(stored_roots[dataset_id])
    if "dataset_root" in sample:
        return _absolute_path(sample["dataset_root"])
    raise ValueError(f"no dataset root for dataset_id {dataset_id!r}")
