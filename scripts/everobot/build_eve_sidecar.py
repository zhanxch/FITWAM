#!/usr/bin/env python3
"""Build EveRobot sidecar metadata for LeRobot-compatible datasets.

EveRobot v0.1 keeps LeRobot data untouched and writes all self-evolution
metadata under an ``eve/`` directory.  The sidecar records episode provenance,
failure event windows, and round-specific training manifests.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "0.1"
FAILURE_PHRASE = "Failed to finish the whole process."


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)
        f.write("\n")


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


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def upsert_jsonl(
    path: Path,
    new_rows: list[dict[str, Any]],
    *,
    key_fields: tuple[str, ...],
) -> None:
    rows = load_jsonl(path)
    keyed: dict[tuple[Any, ...], dict[str, Any]] = {}
    order: list[tuple[Any, ...]] = []
    for row in rows:
        key = tuple(row.get(field) for field in key_fields)
        keyed[key] = row
        order.append(key)
    for row in new_rows:
        key = tuple(row.get(field) for field in key_fields)
        if key not in keyed:
            order.append(key)
        keyed[key] = row
    deduped_order = []
    seen = set()
    for key in order:
        if key not in seen:
            seen.add(key)
            deduped_order.append(key)
    write_jsonl(path, [keyed[key] for key in deduped_order])


def replace_jsonl_rows(
    path: Path,
    new_rows: list[dict[str, Any]],
    *,
    drop_field: str,
    drop_value: Any,
) -> None:
    existing = [
        row
        for row in load_jsonl(path)
        if row.get(drop_field) != drop_value
    ]
    write_jsonl(path, existing + new_rows)


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


def write_schema(eve_root: Path) -> None:
    write_json(
        eve_root / "schema_version.json",
        {
            "format": "EveRobot",
            "schema_version": SCHEMA_VERSION,
            "compatible_base_format": "LeRobot",
            "description": (
                "LeRobot-compatible sidecar metadata for failure-aware "
                "self-evolution training."
            ),
            "files": {
                "episode_meta": "episode_meta.jsonl",
                "event_meta": "event_meta.jsonl",
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


def trim_end_frame(
    *,
    episode_index: int,
    raw_length: int,
    trim_report: dict[int, dict[str, Any]],
) -> tuple[int, str]:
    row = trim_report.get(int(episode_index))
    if row is None:
        return raw_length, "full_failure_episode"
    if "trimmed" in row and not bool(row["trimmed"]):
        return raw_length, "full_failure_episode"
    if "trimmed_length" in row:
        return min(int(row["trimmed_length"]), raw_length), "trimmed_failure_window"
    if "new_length" in row:
        return min(int(row["new_length"]), raw_length), "trimmed_failure_window"
    if "trim_end_frame" in row and "trim_start_frame" in row:
        start = int(row.get("trim_start_frame", 0))
        end = int(row["trim_end_frame"])
        if start != 0:
            return min(max(end - start, 1), raw_length), "trimmed_failure_window_rebased"
        return min(end, raw_length), "trimmed_failure_window"
    return raw_length, "full_failure_episode"


def init_base(args: argparse.Namespace) -> None:
    dataset_root = args.dataset_root.expanduser().resolve()
    dataset_id = args.dataset_id
    eve_root = args.eve_root.expanduser().resolve() if args.eve_root else dataset_root / "eve"
    episodes = load_lerobot_episodes(dataset_root)
    info = load_lerobot_info(dataset_root)

    rows: list[dict[str, Any]] = []
    for ep in episodes:
        ep_idx = int(ep["episode_index"])
        task = first_task(ep)
        outcome = "failure" if is_failure_task(task, args.failure_phrase) else "success"
        if args.force_success:
            outcome = "success"
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "dataset_id": dataset_id,
                "dataset_root": str(dataset_root),
                "episode_index": ep_idx,
                "task_name": args.task_name,
                "task": strip_failure_phrase(task, args.failure_phrase),
                "source_type": args.source_type,
                "source_policy": args.source_policy,
                "collection_round": int(args.collection_round),
                "episode_outcome": outcome,
                "failure_type": None if outcome == "success" else args.default_failure_type,
                "seed": None,
                "length": int(ep["length"]),
                "fps": int(info["fps"]),
                "split": args.split,
            }
        )

    write_schema(eve_root)
    upsert_jsonl(eve_root / "episode_meta.jsonl", rows, key_fields=("dataset_id", "episode_index"))
    write_json(
        eve_root / "reports" / f"init_base_{dataset_id}.json",
        {
            "dataset_id": dataset_id,
            "dataset_root": str(dataset_root),
            "episodes": len(rows),
            "successes": sum(1 for row in rows if row["episode_outcome"] == "success"),
            "failures": sum(1 for row in rows if row["episode_outcome"] == "failure"),
            "source_type": args.source_type,
            "collection_round": int(args.collection_round),
        },
    )
    print(f"[eve] initialized {eve_root} with {len(rows)} episode rows from {dataset_root}")


def append_rollout(args: argparse.Namespace) -> None:
    eve_root = args.base_eve_root.expanduser().resolve()
    rollout_root = args.rollout_root.expanduser().resolve()
    trimmed_root = args.trimmed_event_root.expanduser().resolve() if args.trimmed_event_root else None
    dataset_id = args.dataset_id

    episodes = load_lerobot_episodes(rollout_root)
    info = load_lerobot_info(rollout_root)
    summary = load_collection_summary(rollout_root)
    attempt_by_ep = attempt_log_by_episode(summary, episode_count=len(episodes))
    trim_report = load_trim_report(trimmed_root)

    episode_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    for ep in episodes:
        ep_idx = int(ep["episode_index"])
        task_raw = first_task(ep)
        task = strip_failure_phrase(task_raw, args.failure_phrase)
        attempt = attempt_by_ep.get(ep_idx, {})
        # Prefer the LeRobot task marker because old merged summaries kept
        # shard-local saved_episode_index values.  The attempt log is still
        # useful for seed/provenance after the remapping above.
        task_marks_failure = is_failure_task(task_raw, args.failure_phrase)
        if task_marks_failure:
            outcome = "failure"
        elif "success" in attempt:
            outcome = "success" if bool(attempt["success"]) else "failure"
        else:
            outcome = "success"

        length = int(ep["length"])
        failure_type = None if outcome == "success" else args.default_failure_type
        episode_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "dataset_id": dataset_id,
                "dataset_root": str(rollout_root),
                "episode_index": ep_idx,
                "task_name": args.task_name,
                "task": task,
                "source_type": "policy_rollout",
                "source_policy": args.source_policy,
                "source_checkpoint": args.source_checkpoint,
                "collection_round": int(args.collection_round),
                "episode_outcome": outcome,
                "failure_type": failure_type,
                "seed": attempt.get("seed"),
                "attempt_index": attempt.get("attempt_index"),
                "length": length,
                "fps": int(info["fps"]),
                "split": args.split,
            }
        )

        if outcome != "failure":
            continue

        end_frame, window_rule = trim_end_frame(
            episode_index=ep_idx,
            raw_length=length,
            trim_report=trim_report,
        )
        start_frame = 0
        if end_frame <= start_frame:
            end_frame = min(length, start_frame + 1)
        event_id = f"{dataset_id}_ep{ep_idx:06d}_failure_event"
        event_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "event_id": event_id,
                "dataset_id": dataset_id,
                "dataset_root": str(rollout_root),
                "episode_index": ep_idx,
                "task_name": args.task_name,
                "task": task,
                "event_type": "failure_event",
                "event_outcome": "failure",
                "failure_type": failure_type,
                "source_policy": args.source_policy,
                "source_checkpoint": args.source_checkpoint,
                "collection_round": int(args.collection_round),
                "start_frame": start_frame,
                "end_frame": int(end_frame),
                "failure_frame": length - 1,
                "source_window_rule": window_rule,
                "action_loss": args.failure_action_loss,
                "sample_role": "failure_context",
                "steer_token": None,
                "paired_success_event_id": None,
                "paired_trimmed_dataset": str(trimmed_root) if trimmed_root is not None else None,
                "split": args.split,
            }
        )

    write_schema(eve_root)
    upsert_jsonl(eve_root / "episode_meta.jsonl", episode_rows, key_fields=("dataset_id", "episode_index"))
    replace_jsonl_rows(
        eve_root / "event_meta.jsonl",
        event_rows,
        drop_field="dataset_id",
        drop_value=dataset_id,
    )
    write_json(
        eve_root / "reports" / f"append_rollout_{dataset_id}.json",
        {
            "dataset_id": dataset_id,
            "rollout_root": str(rollout_root),
            "trimmed_event_root": str(trimmed_root) if trimmed_root is not None else None,
            "episodes": len(episode_rows),
            "successes": sum(1 for row in episode_rows if row["episode_outcome"] == "success"),
            "failures": sum(1 for row in episode_rows if row["episode_outcome"] == "failure"),
            "failure_events": len(event_rows),
            "source_policy": args.source_policy,
            "collection_round": int(args.collection_round),
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


def build_manifest(args: argparse.Namespace) -> None:
    eve_root = args.eve_root.expanduser().resolve()
    episode_rows = load_jsonl(eve_root / "episode_meta.jsonl")
    event_rows = load_jsonl(eve_root / "event_meta.jsonl")
    include_outcomes = {str(item) for item in args.include_outcomes}
    success_dataset_ids = parse_optional_set(args.success_dataset_ids)
    failure_dataset_ids = parse_optional_set(args.failure_dataset_ids)

    samples: list[dict[str, Any]] = []

    if "success" in include_outcomes:
        for row in episode_rows:
            if row.get("episode_outcome") != "success":
                continue
            if not selected_dataset(row, success_dataset_ids):
                continue
            samples.append(
                {
                    "sample_type": "episode",
                    "sample_id": f"{row['dataset_id']}_ep{int(row['episode_index']):06d}",
                    "dataset_id": row["dataset_id"],
                    "dataset_root": row["dataset_root"],
                    "episode_index": int(row["episode_index"]),
                    "task": row.get("task", ""),
                    "episode_outcome": "success",
                    "event_outcome": "success",
                    "start_frame": 0,
                    "end_frame": int(row["length"]),
                    "action_loss": "enabled",
                    "sample_role": "success_episode",
                    "sample_stride": int(args.success_sample_stride),
                    "split": row.get("split", "train"),
                }
            )

    if "failure" in include_outcomes:
        if args.failure_sample_mode in {"event_only", "both"}:
            for row in event_rows:
                if row.get("event_outcome") != "failure":
                    continue
                if not selected_dataset(row, failure_dataset_ids):
                    continue
                sample = dict(row)
                sample.update(
                    {
                        "sample_type": "event",
                        "sample_id": row["event_id"],
                        "episode_outcome": "failure",
                        "sample_stride": int(args.failure_sample_stride),
                    }
                )
                samples.append(sample)
        if args.failure_sample_mode in {"full_episode", "both"}:
            for row in episode_rows:
                if row.get("episode_outcome") != "failure":
                    continue
                if not selected_dataset(row, failure_dataset_ids):
                    continue
                samples.append(
                    {
                        "sample_type": "episode",
                        "sample_id": f"{row['dataset_id']}_ep{int(row['episode_index']):06d}_failure_full",
                        "dataset_id": row["dataset_id"],
                        "dataset_root": row["dataset_root"],
                        "episode_index": int(row["episode_index"]),
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

    dataset_roots = {
        str(sample["dataset_id"]): str(Path(sample["dataset_root"]).expanduser().resolve())
        for sample in samples
    }
    manifest = {
        "format": "EveRobotTrainManifest",
        "schema_version": SCHEMA_VERSION,
        "manifest_name": args.manifest_name,
        "eve_root": str(eve_root),
        "include_outcomes": sorted(include_outcomes),
        "failure_sample_mode": args.failure_sample_mode,
        "dataset_roots": dataset_roots,
        "num_samples": len(samples),
        "samples": samples,
    }
    out_path = eve_root / "manifests" / f"{args.manifest_name}.json"
    write_json(out_path, manifest)
    print(f"[eve] wrote manifest {out_path} samples={len(samples)}")


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
    init.add_argument("--failure-phrase", type=str, default=FAILURE_PHRASE)
    init.add_argument("--default-failure-type", type=str, default="unknown_failure")
    init.add_argument("--force-success", action="store_true", default=False)
    init.set_defaults(func=init_base)

    append = subparsers.add_parser("append-rollout", help="Append rollout provenance and failure events.")
    append.add_argument("--base-eve-root", type=Path, required=True)
    append.add_argument("--rollout-root", type=Path, required=True)
    append.add_argument("--trimmed-event-root", type=Path, default=None)
    append.add_argument("--dataset-id", type=str, required=True)
    append.add_argument("--task-name", type=str, required=True)
    append.add_argument("--source-policy", type=str, required=True)
    append.add_argument("--source-checkpoint", type=str, default=None)
    append.add_argument("--collection-round", type=int, required=True)
    append.add_argument("--split", type=str, default="train")
    append.add_argument("--failure-phrase", type=str, default=FAILURE_PHRASE)
    append.add_argument("--default-failure-type", type=str, default="unknown_failure")
    append.add_argument("--failure-action-loss", choices=["enabled", "disabled"], default="disabled")
    append.set_defaults(func=append_rollout)

    manifest = subparsers.add_parser("build-manifest", help="Build a round-specific Eve training manifest.")
    manifest.add_argument("--eve-root", type=Path, required=True)
    manifest.add_argument("--manifest-name", type=str, required=True)
    manifest.add_argument("--include-outcomes", nargs="+", default=["success", "failure"])
    manifest.add_argument("--success-dataset-ids", nargs="+", default=None)
    manifest.add_argument("--failure-dataset-ids", nargs="+", default=None)
    manifest.add_argument("--failure-sample-mode", choices=["event_only", "full_episode", "both"], default="event_only")
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
