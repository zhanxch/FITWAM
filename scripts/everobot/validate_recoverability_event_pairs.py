#!/usr/bin/env python3
"""Validate Fold Glasses recoverability event-pair artifacts.

The recoverability scanner writes sidecar artifacts (``pair.json`` plus one
descriptor/NPZ for each branch).  They are useful for audit, but are not
automatically valid EVE training samples.  This validator is deliberately
independent of MuJoCo, LeRobot, and the training stack so it can run as a
preflight gate on CPU.

The structural contract checked here is:

* ``t`` is a recoverable prefix (at least one of M continuations succeeds);
* the next aligned scan point ``t+24`` is the first 0/M failure frame;
* the event is exactly ``[t+24-33, t+24)`` and has 33 frames;
* success/failure arrays have matching frame IDs and dimensions;
* the factual prefix before ``t`` is identical in both arrays;
* failure actions are never marked as action-imitation targets.
* 4/4 then 0/4 is a valid pair; all-failure seeds are eligible.

The scanner version in the wild predates explicit action-loss/hash fields.  In
that compatibility mode outcome determines the safe default (failure is
disabled, success is enabled) and the missing provenance is surfaced as a
warning.  ``--require-explicit-action-loss`` and ``--require-hashes`` turn
those warnings into hard pre-training errors for a newer materializer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


PAIR_FORMAT = "FoldGlassesRecoverabilityEventPair"
DESCRIPTOR_FORMATS = {
    "success": "FoldGlassesCounterfactualSuccessEvent",
    "failure": "FoldGlassesFactualFailureEvent",
}
PASS_M = 4
REPLAN_STEPS = 24
EVENT_PRE_FRAMES = 9
EVENT_POST_FRAMES = 24
EVENT_NUM_FRAMES = EVENT_PRE_FRAMES + EVENT_POST_FRAMES
ACTION_DIM = 22
STATE_DIM = 23


class PairValidationError(ValueError):
    """Raised by :func:`validate_pair` when an artifact is not trainable."""


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except json.JSONDecodeError as error:
        raise ValueError(f"{path}: invalid JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _resolve_path(value: Any, *, base: Path, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _int(value: Any, label: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    result = int(value)
    if minimum is not None and result < minimum:
        raise ValueError(f"{label} must be >= {minimum}")
    return result


def _finite_array(array: np.ndarray, label: str) -> None:
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"{label} must be numeric, got dtype={array.dtype}")
    if not np.isfinite(array).all():
        raise ValueError(f"{label} contains non-finite values")


def _validate_optional_hash(value: Any, label: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{label} must be a 64-character SHA-256 hex digest")
    try:
        bytes.fromhex(value)
    except ValueError as error:
        raise ValueError(f"{label} must be a 64-character SHA-256 hex digest") from error
    return value.lower()


def _load_event_arrays(path: Path, label: str) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} arrays missing: {path}")
    try:
        with np.load(path, allow_pickle=False) as loaded:
            required = {"frame_indices", "actions", "states"}
            missing = sorted(required - set(loaded.files))
            if missing:
                raise ValueError(f"{label} arrays missing keys: {missing}")
            arrays = {key: np.asarray(loaded[key]) for key in required}
    except (OSError, ValueError) as error:
        if isinstance(error, ValueError) and str(error).startswith(label):
            raise
        raise ValueError(f"Unable to read {label} arrays {path}: {error}") from error

    frame_indices = arrays["frame_indices"]
    actions = arrays["actions"]
    states = arrays["states"]
    if frame_indices.ndim != 1:
        raise ValueError(f"{label}.frame_indices must be 1-D")
    if not np.issubdtype(frame_indices.dtype, np.integer):
        raise ValueError(f"{label}.frame_indices must be integer")
    if actions.ndim != 2 or actions.shape[1] != ACTION_DIM:
        raise ValueError(
            f"{label}.actions must have shape [T,{ACTION_DIM}], got {actions.shape}"
        )
    if states.ndim != 2 or states.shape[1] != STATE_DIM:
        raise ValueError(
            f"{label}.states must have shape [T,{STATE_DIM}], got {states.shape}"
        )
    lengths = {len(frame_indices), len(actions), len(states)}
    if len(lengths) != 1:
        raise ValueError(
            f"{label} arrays have inconsistent lengths: frames={len(frame_indices)} "
            f"actions={len(actions)} states={len(states)}"
        )
    if len(frame_indices) != EVENT_NUM_FRAMES:
        raise ValueError(
            f"{label} must contain exactly {EVENT_NUM_FRAMES} frames, "
            f"got {len(frame_indices)}"
        )
    if len(frame_indices) and not np.array_equal(
        frame_indices, np.arange(int(frame_indices[0]), int(frame_indices[0]) + len(frame_indices))
    ):
        raise ValueError(f"{label}.frame_indices must be contiguous")
    _finite_array(actions, f"{label}.actions")
    _finite_array(states, f"{label}.states")
    return arrays


def _descriptor_action_loss(
    descriptor: Mapping[str, Any], outcome: str, *, label: str
) -> tuple[str, bool]:
    """Return effective action-loss mode and whether it was explicit."""

    value = descriptor.get("action_loss")
    if value is None:
        # Legacy scanner descriptors carry the outcome but not this field.
        return ("disabled" if outcome == "failure" else "enabled"), False
    if value not in {"enabled", "disabled"}:
        raise ValueError(f"{label}.action_loss must be 'enabled' or 'disabled'")
    expected = "disabled" if outcome == "failure" else "enabled"
    if value != expected:
        raise ValueError(
            f"{label}.action_loss={value!r} conflicts with outcome={outcome!r}; "
            f"expected {expected!r}"
        )
    return str(value), True


def _validate_action_loss_window(
    descriptor: Mapping[str, Any],
    outcome: str,
    *,
    t: int,
    label: str,
    require_success_window: bool = False,
) -> None:
    """Validate optional global-frame action supervision metadata.

    The EVE loader's 33-frame convention exposes 32 action tokens.  A success
    event may therefore narrow supervision to the first 24-frame core, which
    corresponds to action labels in ``[t,t+24)``.  A failure branch must have
    no action window because its entire action loss is disabled.
    """

    window = descriptor.get("action_loss_window")
    if window is None:
        if outcome == "success" and require_success_window:
            raise ValueError(
                f"{label}.action_loss_window is required and must be "
                f"[{t}, {t + REPLAN_STEPS})"
            )
        return
    if outcome == "failure":
        raise ValueError(
            f"{label}.action_loss_window must be absent for a failure branch"
        )
    if not isinstance(window, (list, tuple)) or len(window) != 2:
        raise ValueError(f"{label}.action_loss_window must be [t, t+24]")
    start, end = window
    if start != t or end != t + REPLAN_STEPS:
        raise ValueError(
            f"{label}.action_loss_window={window!r} must equal "
            f"[{t}, {t + REPLAN_STEPS})"
        )


def _descriptor_window(
    descriptor: Mapping[str, Any], *, label: str
) -> tuple[int, int]:
    start = _int(descriptor.get("frame_start"), f"{label}.frame_start", minimum=0)
    end = _int(
        descriptor.get("frame_end_exclusive"),
        f"{label}.frame_end_exclusive",
        minimum=1,
    )
    if end <= start:
        raise ValueError(f"{label} has an empty frame interval")
    return start, end


def _compare_optional_identity(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    fields: Iterable[str],
    *,
    label: str,
) -> None:
    for field in fields:
        lvalue = left.get(field)
        rvalue = right.get(field)
        if lvalue is not None and rvalue is not None and lvalue != rvalue:
            raise ValueError(
                f"{label} {field} mismatch: {lvalue!r} != {rvalue!r}"
            )


def _trajectory_metadata(
    path_value: Any,
    *,
    base: Path,
    label: str,
    expected_prefix: int,
    expected_replicate: int,
    expected_success: bool,
    strict: bool,
) -> tuple[dict[str, Any] | None, list[str]]:
    """Check an optional trajectory ledger referenced by a descriptor."""

    warnings: list[str] = []
    if path_value is None:
        warnings.append(f"{label} trajectory ledger is not recorded")
        return None, warnings
    path = _resolve_path(path_value, base=base, label=f"{label}.trajectory_ledger")
    try:
        ledger = _read_json(path)
    except FileNotFoundError:
        if strict:
            raise ValueError(f"{label} trajectory ledger missing: {path}")
        warnings.append(f"{label} trajectory ledger missing: {path}")
        return None, warnings
    prefix = _int(ledger.get("prefix_frame"), f"{label}.trajectory.prefix_frame", minimum=0)
    replicate = _int(
        ledger.get("replicate_index"), f"{label}.trajectory.replicate_index", minimum=0
    )
    if prefix != expected_prefix:
        raise ValueError(
            f"{label} trajectory prefix_frame={prefix} != t={expected_prefix}"
        )
    if replicate != expected_replicate:
        raise ValueError(
            f"{label} trajectory replicate_index={replicate} != {expected_replicate}"
        )
    if bool(ledger.get("success")) != expected_success:
        raise ValueError(
            f"{label} trajectory success={ledger.get('success')!r} does not match "
            f"expected {expected_success}"
        )
    return ledger, warnings


def inspect_pair(
    pair_path: str | Path,
    *,
    require_explicit_action_loss: bool = False,
    require_action_loss_window: bool = False,
    require_hashes: bool = False,
    require_trajectory_ledgers: bool = False,
    require_videos: bool = False,
) -> dict[str, Any]:
    """Inspect one pair and return a serializable validation report.

    ``status`` is ``valid`` for a trainable pair, ``rejected`` for a readable
    artifact that violates the contract, and ``error`` for malformed/unreadable
    input.  No input is modified.
    """

    pair_file = Path(pair_path).expanduser().resolve()
    report: dict[str, Any] = {
        "pair_path": str(pair_file),
        "status": "error",
        "trainable": False,
        "errors": [],
        "warnings": [],
    }
    try:
        pair = _read_json(pair_file)
        report["pair_id"] = pair.get("pair_id")
        if pair.get("format") != PAIR_FORMAT:
            raise ValueError(f"pair.format must be {PAIR_FORMAT!r}")
        if pair.get("status") != "complete":
            raise ValueError(
                f"pair.status must be 'complete', got {pair.get('status')!r}"
            )
        classification = pair.get("seed_classification")
        if classification not in {"mixed", "all_failure"}:
            raise ValueError(
                "a training pair requires seed_classification in "
                "{'mixed','all_failure'}; "
                f"got seed_classification={classification!r}"
            )
        if pair.get("evaluation_only") is not False or pair.get("training_eligible") is not True:
            raise ValueError(
                "pair must explicitly set evaluation_only=false and "
                "training_eligible=true"
            )
        seed = _int(pair.get("seed"), "pair.seed")
        episode = _int(
            pair.get("source_failure_episode_index"),
            "pair.source_failure_episode_index",
            minimum=0,
        )
        frontier = pair.get("frontier")
        if not isinstance(frontier, Mapping):
            raise ValueError("pair.frontier must be an object")

        t = _int(frontier.get("t_frame"), "frontier.t_frame", minimum=0)
        expected_start = max(0, t - EVENT_PRE_FRAMES)
        expected_end = t + EVENT_POST_FRAMES
        expected_zero = t + REPLAN_STEPS
        pass_m = _int(frontier.get("pass_m"), "frontier.pass_m", minimum=1)
        success_count = _int(
            frontier.get("last_recoverable_success_count"),
            "frontier.last_recoverable_success_count",
            minimum=0,
        )
        if pass_m != PASS_M:
            raise ValueError(f"frontier.pass_m must be {PASS_M}, got {pass_m}")
        if success_count < 1:
            raise ValueError(
                "frontier.t must be recoverable: expected >=1 success for Pass@4, "
                f"got {success_count}/{pass_m}"
            )
        if frontier.get("event_pre_frames") not in {None, EVENT_PRE_FRAMES}:
            raise ValueError(
                f"frontier.event_pre_frames must be {EVENT_PRE_FRAMES}"
            )
        if frontier.get("event_post_frames") not in {None, EVENT_POST_FRAMES}:
            raise ValueError(
                f"frontier.event_post_frames must be {EVENT_POST_FRAMES}"
            )
        for field, expected in (
            ("last_recoverable_frame", t),
            ("t_plus_24_frame", expected_zero),
            ("first_zero_frame", expected_zero),
            ("failure_frame", expected_zero),
        ):
            if frontier.get(field) is None and field == "failure_frame":
                continue
            actual = _int(frontier.get(field), f"frontier.{field}", minimum=0)
            if actual != expected:
                raise ValueError(
                    f"frontier.{field}={actual} != required {expected}"
                )
        if frontier.get("event_window") != [expected_start, expected_end]:
            raise ValueError(
                "frontier.event_window must equal "
                f"[{expected_start}, {expected_end})"
            )
        if _int(frontier.get("event_start"), "frontier.event_start", minimum=0) != expected_start:
            raise ValueError("frontier.event_start does not match event_window")
        if _int(
            frontier.get("event_end_exclusive"),
            "frontier.event_end_exclusive",
            minimum=1,
        ) != expected_end:
            raise ValueError("frontier.event_end_exclusive does not match t+24")
        for field, expected in (
            ("core_event_start", t),
            ("snapshot_frame", t),
        ):
            if frontier.get(field) is not None and _int(
                frontier.get(field), f"frontier.{field}", minimum=0
            ) != expected:
                raise ValueError(f"frontier.{field} must equal t={t}")
        if frontier.get("core_event_end") is not None and _int(
            frontier.get("core_event_end"), "frontier.core_event_end", minimum=1
        ) != expected_zero:
            raise ValueError("frontier.core_event_end must equal t+24")
        if expected_end - expected_start != EVENT_NUM_FRAMES:
            raise ValueError(
                "event interval is shorter than 33 frames after left-edge clamp; "
                f"got [{expected_start}, {expected_end})"
            )

        descriptors: dict[str, dict[str, Any]] = {}
        descriptor_paths: dict[str, Path] = {}
        arrays: dict[str, dict[str, np.ndarray]] = {}
        explicit_action_loss: dict[str, bool] = {}
        for outcome, pair_field in (("failure", "factual_failure_event"), ("success", "counterfactual_success_event")):
            descriptor_path = _resolve_path(
                pair.get(pair_field), base=pair_file.parent, label=f"pair.{pair_field}"
            )
            descriptor = _read_json(descriptor_path)
            label = f"{outcome} descriptor"
            expected_format = DESCRIPTOR_FORMATS[outcome]
            if descriptor.get("format") != expected_format:
                raise ValueError(
                    f"{label}.format must be {expected_format!r}"
                )
            if descriptor.get("outcome") != outcome:
                raise ValueError(
                    f"{label}.outcome must be {outcome!r}, got {descriptor.get('outcome')!r}"
                )
            if outcome == "success" and descriptor.get("deterministic_rerun_succeeded") is False:
                raise ValueError(
                    f"{label}.deterministic_rerun_succeeded is false"
                )
            start, end = _descriptor_window(descriptor, label=label)
            if (start, end) != (expected_start, expected_end):
                raise ValueError(
                    f"{label} interval [{start}, {end}) != required "
                    f"[{expected_start}, {expected_end})"
                )
            if _int(descriptor.get("exact_counterfactual_prefix_frame"), f"{label}.exact_counterfactual_prefix_frame", minimum=0) != t:
                raise ValueError(f"{label} prefix frame does not equal t={t}")
            if _int(descriptor.get("source_failure_episode_index"), f"{label}.source_failure_episode_index", minimum=0) != episode:
                raise ValueError(f"{label} source episode mismatch")
            if _int(descriptor.get("seed"), f"{label}.seed") != seed:
                raise ValueError(f"{label} seed mismatch")
            if _int(descriptor.get("num_frames"), f"{label}.num_frames", minimum=1) != EVENT_NUM_FRAMES:
                raise ValueError(f"{label}.num_frames must be {EVENT_NUM_FRAMES}")
            action_loss, explicit = _descriptor_action_loss(descriptor, outcome, label=label)
            if require_explicit_action_loss and not explicit:
                raise ValueError(
                    f"explicit action_loss is required for {label} during training"
                )
            _validate_action_loss_window(
                descriptor,
                outcome,
                t=t,
                label=label,
                require_success_window=require_action_loss_window,
            )
            explicit_action_loss[outcome] = explicit
            if not explicit:
                report["warnings"].append(f"{label} lacks explicit action_loss; inferred {action_loss}")
            arrays_path = _resolve_path(
                descriptor.get("arrays"), base=descriptor_path.parent, label=f"{label}.arrays"
            )
            descriptors[outcome] = descriptor
            descriptor_paths[outcome] = descriptor_path
            for hash_field in ("snapshot_hash", "prefix_hash"):
                _validate_optional_hash(
                    descriptor.get(hash_field), f"{label}.{hash_field}"
                )
            arrays[outcome] = _load_event_arrays(arrays_path, label)
            report.setdefault("artifacts", {})[outcome] = {
                "descriptor": str(descriptor_path),
                "arrays": str(arrays_path),
                "action_loss": action_loss,
                "action_loss_explicit": explicit,
            }
            if require_videos:
                for video_field in ("front_video", "wrist_video"):
                    video_path = _resolve_path(
                        descriptor.get(video_field),
                        base=descriptor_path.parent,
                        label=f"{label}.{video_field}",
                    )
                    if not video_path.is_file():
                        raise ValueError(f"{label}.{video_field} missing: {video_path}")

        if require_explicit_action_loss and not all(explicit_action_loss.values()):
            missing = [outcome for outcome, present in explicit_action_loss.items() if not present]
            raise ValueError(
                "explicit action_loss metadata is required for both branches; "
                f"missing={missing}"
            )

        _compare_optional_identity(
            pair,
            descriptors["success"],
            ("seed", "source_failure_episode_index", "source_repeat"),
            label="pair/success",
        )
        _compare_optional_identity(
            pair,
            descriptors["failure"],
            ("seed", "source_failure_episode_index", "source_repeat"),
            label="pair/failure",
        )
        _compare_optional_identity(
            descriptors["success"],
            descriptors["failure"],
            ("seed", "source_failure_episode_index", "source_repeat", "exact_counterfactual_prefix_frame", "frontier_first_zero_frame"),
            label="success/failure descriptor",
        )
        # A materializer may persist hashes on either descriptor (or on the
        # pair).  When both branches provide one, equality is mandatory: a
        # pair with two different snapshot identities is not a same-state
        # counterfactual even if its frame arrays happen to have the same
        # length.
        _compare_optional_identity(
            descriptors["success"],
            descriptors["failure"],
            ("snapshot_hash", "prefix_hash"),
            label="success/failure descriptor",
        )
        for field in ("successful_replicate_index",):
            if pair.get(field) is None:
                raise ValueError(f"pair.{field} is required")
            _int(pair[field], f"pair.{field}", minimum=0)
        if _int(pair["successful_replicate_index"], "pair.successful_replicate_index", minimum=0) != _int(descriptors["success"].get("successful_replicate_index"), "success descriptor.successful_replicate_index", minimum=0):
            raise ValueError("success replicate index mismatch")

        frame_indices = arrays["success"]["frame_indices"]
        if not np.array_equal(frame_indices, arrays["failure"]["frame_indices"]):
            raise ValueError("success/failure frame_indices differ")
        expected_frames = np.arange(expected_start, expected_end, dtype=frame_indices.dtype)
        if not np.array_equal(frame_indices, expected_frames):
            raise ValueError(
                "event frame_indices do not equal the exact [t-9,t+24) interval"
            )
        prefix_mask = frame_indices < t
        if int(prefix_mask.sum()) != EVENT_PRE_FRAMES:
            raise ValueError(
                f"expected {EVENT_PRE_FRAMES} pre-t frames, got {int(prefix_mask.sum())}"
            )
        for key in ("actions", "states"):
            if not np.array_equal(
                arrays["success"][key][prefix_mask], arrays["failure"][key][prefix_mask]
            ):
                raise ValueError(f"success/failure factual prefix {key} differs before t")

        _compare_optional_identity(
            pair,
            descriptors["success"],
            ("snapshot_hash", "prefix_hash"),
            label="pair/success",
        )
        _compare_optional_identity(
            pair,
            descriptors["failure"],
            ("snapshot_hash", "prefix_hash"),
            label="pair/failure",
        )
        hash_fields = ("snapshot_hash", "prefix_hash")
        missing_hashes = [
            field
            for field in hash_fields
            if pair.get(field) is None
            and descriptors["success"].get(field) is None
            and descriptors["failure"].get(field) is None
        ]
        if missing_hashes:
            message = f"pair provenance hashes are absent: {missing_hashes}"
            if require_hashes:
                raise ValueError(message)
            report["warnings"].append(message)
        computed_prefix_hash = _sha256_bytes(
            b"".join(
                [
                    np.ascontiguousarray(frame_indices[prefix_mask]).tobytes(),
                    np.ascontiguousarray(arrays["success"]["actions"][prefix_mask]).tobytes(),
                    np.ascontiguousarray(arrays["success"]["states"][prefix_mask]).tobytes(),
                ]
            )
        )
        report["computed_prefix_hash"] = computed_prefix_hash

        success_ledger, success_warnings = _trajectory_metadata(
            descriptors["success"].get("successful_continuation_ledger"),
            base=descriptor_paths["success"].parent,
            label="success",
            expected_prefix=t,
            expected_replicate=_int(pair["successful_replicate_index"], "pair.successful_replicate_index", minimum=0),
            expected_success=True,
            strict=require_trajectory_ledgers,
        )
        report["warnings"].extend(success_warnings)
        failure_ledger_path = descriptors["failure"].get("trajectory_ledger")
        if failure_ledger_path is not None or require_trajectory_ledgers:
            _failure_ledger, failure_warnings = _trajectory_metadata(
                failure_ledger_path,
                base=descriptor_paths["failure"].parent,
                label="failure",
                expected_prefix=t,
                expected_replicate=0,
                expected_success=False,
                strict=require_trajectory_ledgers,
            )
            report["warnings"].extend(failure_warnings)
        else:
            report["warnings"].append(
                "failure branch is the factual GT window; no continuation ledger"
            )
        report["seed"] = seed
        report["source_failure_episode_index"] = episode
        report["t_frame"] = t
        report["event_interval"] = [expected_start, expected_end]
        report["num_frames"] = EVENT_NUM_FRAMES
        report["success_count_at_t"] = success_count
        report["trainable"] = True
        report["status"] = "valid"
        return report
    except FileNotFoundError as error:
        report["status"] = "error"
        report["errors"] = [str(error)]
        return report
    except (TypeError, ValueError, OSError) as error:
        report["status"] = "rejected"
        report["errors"] = [str(error)]
        return report


def validate_pair(
    pair_path: str | Path,
    *,
    require_explicit_action_loss: bool = True,
    require_action_loss_window: bool = True,
    **kwargs: Any,
) -> dict[str, Any]:
    """Validate one training-facing pair or raise ``PairValidationError``.

    Unlike :func:`inspect_pair`, this API defaults to the post-materialization
    contract: both branches explicitly declare action-loss behavior and the
    success branch names the core ``[t,t+24)`` supervision interval.
    """

    report = inspect_pair(
        pair_path,
        require_explicit_action_loss=require_explicit_action_loss,
        require_action_loss_window=require_action_loss_window,
        **kwargs,
    )
    if report["status"] != "valid":
        raise PairValidationError(
            f"{pair_path}: {report['status']}: "
            + "; ".join(str(error) for error in report.get("errors", []))
        )
    return report


def discover_pairs(root: str | Path) -> list[Path]:
    root_path = Path(root).expanduser().resolve()
    if root_path.is_file():
        return [root_path]
    return sorted(root_path.glob("event_pairs/**/pair.json"))


def _write_report(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(_canonical_json(dict(payload)) + "\n", encoding="utf-8")
    temporary.replace(path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", dest="pairs", action="append", default=[])
    parser.add_argument("--pairs-root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--require-explicit-action-loss",
        dest="require_explicit_action_loss",
        action="store_true",
        default=True,
        help="Require explicit branch action_loss metadata (default).",
    )
    parser.add_argument(
        "--allow-legacy-action-loss",
        dest="require_explicit_action_loss",
        action="store_false",
        help="Audit legacy descriptors that omit action_loss metadata.",
    )
    parser.add_argument(
        "--require-action-loss-window",
        dest="require_action_loss_window",
        action="store_true",
        default=True,
        help="Require success action_loss_window=[t,t+24) (default).",
    )
    parser.add_argument(
        "--allow-legacy-action-window",
        dest="require_action_loss_window",
        action="store_false",
        help="Audit descriptors that omit the success core action window.",
    )
    parser.add_argument(
        "--audit-legacy",
        action="store_true",
        help="Disable both action metadata gates for compatibility auditing.",
    )
    parser.add_argument("--require-hashes", action="store_true")
    parser.add_argument("--require-trajectory-ledgers", action="store_true")
    parser.add_argument("--require-videos", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.audit_legacy:
        args.require_explicit_action_loss = False
        args.require_action_loss_window = False
    paths = [Path(value) for value in args.pairs]
    if args.pairs_root is not None:
        paths.extend(discover_pairs(args.pairs_root))
    paths = sorted({path.expanduser().resolve() for path in paths})
    if not paths:
        raise SystemExit("Provide at least one --pair or --pairs-root")
    reports = [
        inspect_pair(
            path,
            require_explicit_action_loss=args.require_explicit_action_loss,
            require_action_loss_window=args.require_action_loss_window,
            require_hashes=args.require_hashes,
            require_trajectory_ledgers=args.require_trajectory_ledgers,
            require_videos=args.require_videos,
        )
        for path in paths
    ]
    valid = [row for row in reports if row["status"] == "valid"]
    error = [row for row in reports if row["status"] == "error"]
    payload = {
        "format": "FoldGlassesRecoverabilityEventPairValidation",
        "schema_version": "0.1",
        "status": "passed" if len(valid) == len(reports) else "failed",
        "num_pairs": len(reports),
        "num_valid": len(valid),
        "num_rejected": sum(row["status"] == "rejected" for row in reports),
        "num_errors": len(error),
        "options": {
            "require_explicit_action_loss": bool(args.require_explicit_action_loss),
            "require_action_loss_window": bool(args.require_action_loss_window),
            "require_hashes": bool(args.require_hashes),
            "require_trajectory_ledgers": bool(args.require_trajectory_ledgers),
            "require_videos": bool(args.require_videos),
        },
        "pairs": reports,
    }
    if args.output is not None:
        _write_report(args.output.expanduser().resolve(), payload)
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if len(valid) == len(reports) else (2 if error else 1)


if __name__ == "__main__":
    raise SystemExit(main())
