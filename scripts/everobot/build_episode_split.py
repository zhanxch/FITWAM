#!/usr/bin/env python3
"""Create a frozen, outcome-stratified EveRobot episode split map."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
for source_root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from fastwam.everobot_schema import sha256_json  # noqa: E402
from scripts.everobot import build_eve_sidecar  # noqa: E402


def parse_dataset(value: str) -> tuple[str, Path]:
    dataset_id, separator, raw_path = value.partition("=")
    if not separator or not dataset_id.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError(
            "Dataset must use DATASET_ID=/absolute/or/relative/path"
        )
    return dataset_id.strip(), Path(raw_path).expanduser().resolve()


def classify_episodes(
    dataset_id: str,
    dataset_root: Path,
    *,
    force_success: bool,
    require_explicit_outcomes: bool,
    failure_phrase: str,
) -> list[dict[str, Any]]:
    episodes = build_eve_sidecar.load_lerobot_episodes(dataset_root)
    summary = build_eve_sidecar.load_collection_summary(dataset_root)
    attempts = build_eve_sidecar.attempt_log_by_episode(
        summary, episode_count=len(episodes)
    )
    structured_outcomes = build_eve_sidecar.load_episode_outcome_ledger(
        dataset_root,
        required=require_explicit_outcomes,
        expected_episode_indices=(
            int(episode["episode_index"]) for episode in episodes
        ),
    )
    rows: list[dict[str, Any]] = []
    for episode in episodes:
        episode_index = int(episode["episode_index"])
        task = build_eve_sidecar.first_task(episode)
        attempt = attempts.get(episode_index, {})
        structured = structured_outcomes.get(episode_index)
        if force_success:
            outcome = "success"
            outcome_source = "forced_success"
        elif structured is not None:
            outcome = str(structured["outcome"])
            outcome_source = "structured_outcome_ledger"
            if build_eve_sidecar.is_failure_task(
                task, failure_phrase
            ) and outcome != "failure":
                raise ValueError(
                    f"{dataset_id} episode {episode_index} task marker "
                    "disagrees with its structured outcome"
                )
            if "success" in attempt and bool(attempt["success"]) != (
                outcome == "success"
            ):
                raise ValueError(
                    f"{dataset_id} episode {episode_index} collection summary "
                    "disagrees with its structured outcome"
                )
        elif build_eve_sidecar.is_failure_task(task, failure_phrase):
            outcome = "failure"
            outcome_source = "task_marker"
        elif "success" in attempt:
            outcome = "success" if bool(attempt["success"]) else "failure"
            outcome_source = "collection_summary"
        else:
            outcome = "success"
            outcome_source = "assumed_success"
        rows.append(
            {
                "dataset_id": dataset_id,
                "episode_index": episode_index,
                "episode_id": build_eve_sidecar.make_episode_id(
                    dataset_id, episode_index
                ),
                "episode_outcome": outcome,
                "outcome_source": outcome_source,
            }
        )
    return rows


def assign_splits(
    rows: Sequence[Mapping[str, Any]],
    *,
    val_fraction: float,
    seed: int,
) -> list[dict[str, Any]]:
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("val_fraction must lie in [0, 1)")
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for raw_row in rows:
        row = dict(raw_row)
        key = (str(row["dataset_id"]), str(row["episode_outcome"]))
        grouped.setdefault(key, []).append(row)

    assigned: list[dict[str, Any]] = []
    for (dataset_id, outcome), group in sorted(grouped.items()):
        ordered = sorted(group, key=lambda row: int(row["episode_index"]))
        group_seed = int.from_bytes(
            hashlib.sha256(
                f"{seed}:{dataset_id}:{outcome}".encode("utf-8")
            ).digest()[:8],
            byteorder="big",
        )
        random.Random(group_seed).shuffle(ordered)
        val_count = int(round(len(ordered) * val_fraction))
        if val_fraction > 0.0 and len(ordered) >= 2:
            val_count = min(max(val_count, 1), len(ordered) - 1)
        else:
            val_count = 0
        val_ids = {
            int(row["episode_index"]) for row in ordered[:val_count]
        }
        for row in group:
            episode_index = int(row["episode_index"])
            assigned.append(
                {
                    **row,
                    "split": "val" if episode_index in val_ids else "train",
                    "split_method": "outcome_stratified_v1",
                    "split_seed": int(seed),
                    "val_fraction": float(val_fraction),
                }
            )
    assigned.sort(
        key=lambda row: (str(row["dataset_id"]), int(row["episode_index"]))
    )
    return assigned


def build_split_map(
    datasets: Sequence[tuple[str, Path]],
    *,
    force_success_dataset_ids: set[str],
    failure_phrase: str,
    val_fraction: float,
    seed: int,
    require_explicit_outcome_dataset_ids: set[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    require_explicit_outcome_dataset_ids = (
        set(require_explicit_outcome_dataset_ids or set())
    )
    dataset_ids = [dataset_id for dataset_id, _ in datasets]
    if len(dataset_ids) != len(set(dataset_ids)):
        raise ValueError("Dataset IDs must be unique")
    unknown_force_success = force_success_dataset_ids - set(dataset_ids)
    if unknown_force_success:
        raise ValueError(
            "force-success dataset IDs were not provided: "
            f"{sorted(unknown_force_success)}"
        )
    unknown_explicit_outcomes = (
        require_explicit_outcome_dataset_ids - set(dataset_ids)
    )
    if unknown_explicit_outcomes:
        raise ValueError(
            "require-explicit-outcome dataset IDs were not provided: "
            f"{sorted(unknown_explicit_outcomes)}"
        )

    source_rows: list[dict[str, Any]] = []
    roots: dict[str, str] = {}
    for dataset_id, dataset_root in datasets:
        if not dataset_root.exists():
            raise FileNotFoundError(dataset_root)
        roots[dataset_id] = str(dataset_root)
        source_rows.extend(
            classify_episodes(
                dataset_id,
                dataset_root,
                force_success=dataset_id in force_success_dataset_ids,
                require_explicit_outcomes=(
                    dataset_id in require_explicit_outcome_dataset_ids
                ),
                failure_phrase=failure_phrase,
            )
        )
    rows = assign_splits(
        source_rows, val_fraction=val_fraction, seed=seed
    )
    counts: dict[str, dict[str, int]] = {}
    for row in rows:
        key = f"{row['dataset_id']}:{row['episode_outcome']}"
        split_counts = counts.setdefault(key, {"train": 0, "val": 0})
        split_counts[str(row["split"])] += 1
    report = {
        "format": "EveRobotEpisodeSplitMap",
        "version": "1.0",
        "split_method": "outcome_stratified_v1",
        "seed": int(seed),
        "val_fraction": float(val_fraction),
        "dataset_roots": roots,
        "force_success_dataset_ids": sorted(force_success_dataset_ids),
        "require_explicit_outcome_dataset_ids": sorted(
            require_explicit_outcome_dataset_ids
        ),
        "num_episodes": len(rows),
        "counts": counts,
        "split_map_sha256": sha256_json(rows),
    }
    return rows, report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        action="append",
        type=parse_dataset,
        required=True,
        help="Repeat DATASET_ID=PATH for every source dataset.",
    )
    parser.add_argument(
        "--force-success-dataset-id", action="append", default=[]
    )
    parser.add_argument(
        "--require-explicit-outcome-dataset-id",
        action="append",
        default=[],
        help=(
            "Require a complete meta/episode_outcomes.jsonl ledger for this "
            "dataset ID. Repeat for multiple rollout datasets."
        ),
    )
    parser.add_argument(
        "--failure-phrase", default=build_eve_sidecar.FAILURE_PHRASE
    )
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output = args.output.expanduser().resolve()
    report_path = (
        args.report.expanduser().resolve()
        if args.report is not None
        else output.with_suffix(".report.json")
    )
    if output.exists() or report_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite split artifacts: {output}, {report_path}"
        )
    rows, report = build_split_map(
        args.dataset,
        force_success_dataset_ids=set(args.force_success_dataset_id),
        failure_phrase=args.failure_phrase,
        val_fraction=args.val_fraction,
        seed=args.seed,
        require_explicit_outcome_dataset_ids=set(
            args.require_explicit_outcome_dataset_id
        ),
    )
    build_eve_sidecar.write_jsonl(output, rows)
    build_eve_sidecar.write_json(report_path, report)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
