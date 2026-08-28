#!/usr/bin/env python3
"""Select N complete S0 success rollouts as DEWO v2 primary (not expert).

Writes:
  - primary_success_episodes.json  (the N chosen train episodes)
  - episode_splits.jsonl covering every episode in the LeRobot dataset
      selected successes -> train
      leftover successes -> val
      failures           -> test  (never sampled as primary)
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

MIN_COMPLETE_LENGTH = 33  # pair events are 33 frames; primary must be full episodes


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def episode_id(dataset_id: str, episode_index: int) -> str:
    return f"{dataset_id}:episode:{int(episode_index):06d}"


def is_complete_success(outcome: dict[str, Any], length: int) -> bool:
    success = bool(outcome.get("success")) or str(outcome.get("outcome", "")).lower() == "success"
    return success and int(length) > MIN_COMPLETE_LENGTH


def select_primary(
    *,
    outcomes: list[dict[str, Any]],
    lengths: dict[int, int],
    n: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    successes: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for row in outcomes:
        ep = int(row["episode_index"])
        length = int(lengths[ep])
        item = {**row, "length": length}
        if is_complete_success(row, length):
            successes.append(item)
        else:
            failures.append(item)
    if len(successes) < n:
        raise SystemExit(
            f"Need {n} complete success rollouts, found {len(successes)} "
            f"(success and length>{MIN_COMPLETE_LENGTH})."
        )
    ordered = sorted(successes, key=lambda r: int(r["episode_index"]))
    rng = random.Random(int(seed))
    primary = rng.sample(ordered, n)
    primary.sort(key=lambda r: int(r["episode_index"]))
    primary_ids = {int(r["episode_index"]) for r in primary}
    leftover = [r for r in ordered if int(r["episode_index"]) not in primary_ids]
    return primary, leftover, failures


def select_one_per_all_success_seed(
    *,
    outcomes: list[dict[str, Any]],
    lengths: dict[int, int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """D0 = one complete success episode per 4/4 all-success seed.

    Deterministic pick: min ``attempt_index``, then min ``episode_index``.
    Other successes from those seeds, and mixed-seed successes, go to leftover.
    """

    by_seed: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in outcomes:
        ep = int(row["episode_index"])
        item = {**row, "length": int(lengths[ep])}
        by_seed[int(row["seed"])].append(item)

    primary: list[dict[str, Any]] = []
    leftover: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for seed in sorted(by_seed):
        rows = by_seed[seed]
        complete = [r for r in rows if is_complete_success(r, int(r["length"]))]
        rest = [r for r in rows if not is_complete_success(r, int(r["length"]))]
        failures.extend(rest)
        if complete and len(complete) == len(rows):
            ordered = sorted(
                complete,
                key=lambda r: (
                    int(r.get("attempt_index", r.get("repeat", 10**9))),
                    int(r["episode_index"]),
                ),
            )
            primary.append(ordered[0])
            leftover.extend(ordered[1:])
        else:
            leftover.extend(complete)
    primary.sort(key=lambda r: int(r["episode_index"]))
    leftover.sort(key=lambda r: int(r["episode_index"]))
    failures.sort(key=lambda r: int(r["episode_index"]))
    if not primary:
        raise SystemExit("No 4/4 all-success seeds found for DEWO v9 D0.")
    return primary, leftover, failures


def build_split_rows(
    *,
    dataset_id: str,
    primary: list[dict[str, Any]],
    leftover: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    seed: int,
    split_method: str = "success_rollout_primary_v1",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split, group in (
        ("train", primary),
        ("val", leftover),
        ("test", failures),
    ):
        for row in group:
            ep = int(row["episode_index"])
            rows.append(
                {
                    "dataset_id": dataset_id,
                    "episode_index": ep,
                    "episode_id": episode_id(dataset_id, ep),
                    "episode_outcome": "success" if split != "test" else "failure",
                    "outcome_source": "structured_outcome_ledger",
                    "split": split,
                    "split_method": str(split_method),
                    "split_seed": int(seed),
                    "length": int(row["length"]),
                }
            )
    rows.sort(key=lambda r: int(r["episode_index"]))
    return rows


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True, help="LeRobot rollout_raw root")
    p.add_argument("--dataset-id", required=True)
    p.add_argument("--n", type=int, default=15)
    p.add_argument(
        "--mode",
        choices=("random_n", "one_per_all_success_seed"),
        default="random_n",
        help=(
            "random_n: sample --n complete successes (v2/v7). "
            "one_per_all_success_seed: locked v9 D0 (one episode per 4/4 seed)."
        ),
    )
    p.add_argument("--seed", type=int, default=20260820)
    p.add_argument("--output-json", type=Path, required=True)
    p.add_argument("--output-splits", type=Path, required=True)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.dataset.expanduser().resolve()
    outcomes = load_jsonl(root / "meta" / "episode_outcomes.jsonl")
    episodes = load_jsonl(root / "meta" / "episodes.jsonl")
    lengths = {int(r["episode_index"]): int(r["length"]) for r in episodes}
    missing = [int(r["episode_index"]) for r in outcomes if int(r["episode_index"]) not in lengths]
    if missing:
        raise SystemExit(f"Outcomes without episode length: {missing[:8]}")
    if set(lengths) != {int(r["episode_index"]) for r in outcomes}:
        raise SystemExit("episode_outcomes.jsonl and episodes.jsonl index sets differ")

    split_method = "success_rollout_primary_v1"
    if args.mode == "one_per_all_success_seed":
        primary, leftover, failures = select_one_per_all_success_seed(
            outcomes=outcomes, lengths=lengths
        )
        split_method = "all_success_seed_primary_v8"
    else:
        primary, leftover, failures = select_primary(
            outcomes=outcomes, lengths=lengths, n=int(args.n), seed=int(args.seed)
        )
    splits = build_split_rows(
        dataset_id=str(args.dataset_id),
        primary=primary,
        leftover=leftover,
        failures=failures,
        seed=int(args.seed),
        split_method=split_method,
    )
    if len(splits) != len(episodes):
        raise SystemExit(
            f"split map size {len(splits)} != n_episodes {len(episodes)}"
        )

    payload = {
        "dataset_root": str(root),
        "dataset_id": str(args.dataset_id),
        "n_primary": len(primary),
        "selection_mode": str(args.mode),
        "split_method": split_method,
        "seed": int(args.seed),
        "min_complete_length": MIN_COMPLETE_LENGTH,
        "n_complete_success": len(primary) + len(leftover),
        "n_failure_or_short": len(failures),
        "primary_episode_indices": [int(r["episode_index"]) for r in primary],
        "primary": primary,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    args.output_splits.parent.mkdir(parents=True, exist_ok=True)
    args.output_splits.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in splits)
    )
    print(
        f"wrote {args.output_json} mode={args.mode} primary={len(primary)} "
        f"complete_success={len(primary)+len(leftover)} val={len(leftover)} "
        f"test={len(failures)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
