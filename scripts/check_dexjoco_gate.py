#!/usr/bin/env python3
"""Check a DexJoCo rollout summary against the current benchmark gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


BENCHMARKS = {
    "hammer_nail": {"pi0_5": 84.7, "groot_n1_5": 67.3},
    "click_mouse": {"pi0_5": 64.7, "groot_n1_5": 85.3},
    "pick_bucket": {"pi0_5": 84.0, "groot_n1_5": 72.0},
    "pinch_tongs": {"pi0_5": 24.0, "groot_n1_5": 12.7},
    "fold_glasses": {"pi0_5": 72.0, "groot_n1_5": 27.3},
    "water_plant": {"pi0_5": 88.7, "groot_n1_5": 72.7},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path, help="Combined DexJoCo eval summary.json")
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument(
        "--tolerance-pp",
        type=float,
        default=1.0,
        help="Pass if success rate is within this many percentage points of the lower benchmark.",
    )
    parser.add_argument("--json-out", type=Path, default=None)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_task_result(summary: dict[str, Any], task: str | None) -> tuple[str, int, int, float]:
    tasks = summary.get("tasks")
    if isinstance(tasks, list) and tasks:
        if task is None:
            if len(tasks) != 1:
                names = [item.get("env_name") for item in tasks]
                raise ValueError(f"Summary has multiple tasks; pass --task. tasks={names}")
            row = tasks[0]
        else:
            matches = [item for item in tasks if item.get("env_name") == task]
            if not matches:
                names = [item.get("env_name") for item in tasks]
                raise ValueError(f"Task {task!r} not found in summary. tasks={names}")
            row = matches[0]
        name = str(row.get("env_name") or task)
        episodes = int(row["episodes"])
        successes = int(row["successes"])
        rate = float(row.get("success_rate", successes / max(episodes, 1)))
        return name, episodes, successes, rate

    if task is None:
        task = str(summary.get("env_name") or summary.get("task") or "")
    if not task:
        raise ValueError("Cannot infer task name from summary; pass --task.")
    episodes = int(summary.get("total_episodes", summary.get("episodes")))
    successes = int(summary.get("total_successes", summary.get("successes")))
    rate = float(summary.get("overall_success_rate", summary.get("success_rate", successes / max(episodes, 1))))
    return task, episodes, successes, rate


def main() -> None:
    args = parse_args()
    summary = load_json(args.summary.expanduser())
    task, episodes, successes, rate = extract_task_result(summary, args.task)
    if task not in BENCHMARKS:
        raise KeyError(f"No benchmark gate configured for task={task!r}")

    pi0 = BENCHMARKS[task]["pi0_5"]
    groot = BENCHMARKS[task]["groot_n1_5"]
    lower = min(pi0, groot)
    threshold = lower - float(args.tolerance_pp)
    rate_pp = 100.0 * rate
    passed = rate_pp >= threshold
    payload = {
        "task": task,
        "episodes": episodes,
        "successes": successes,
        "success_rate": rate,
        "success_rate_pp": rate_pp,
        "pi0_5_pp": pi0,
        "groot_n1_5_pp": groot,
        "lower_benchmark_pp": lower,
        "tolerance_pp": float(args.tolerance_pp),
        "pass_threshold_pp": threshold,
        "passed": passed,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
