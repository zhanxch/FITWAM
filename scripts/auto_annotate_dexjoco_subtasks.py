#!/usr/bin/env python3
"""Auto-annotate DexJoCo microwave-cook episodes with dual-hand subtasks.

Uses observation.state trajectories (TCP pose + gripper joints) to infer the
five canonical subtask segments, then writes annotations/dual_hand_subtasks.jsonl.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from dexjoco_subtask_annotator import (  # noqa: E402
    ANNOTATION_FILENAME,
    DEFAULT_TASK,
    LEFT_SUBTASKS,
    RIGHT_SUBTASKS,
)

SEGMENT_LABELS: list[tuple[str, str]] = [
    (LEFT_SUBTASKS[0], RIGHT_SUBTASKS[0]),  # open door + pick food
    (LEFT_SUBTASKS[1], RIGHT_SUBTASKS[1]),  # nothing + place food
    (LEFT_SUBTASKS[1], RIGHT_SUBTASKS[2]),  # nothing + move out
    (LEFT_SUBTASKS[2], RIGHT_SUBTASKS[3]),  # close door + nothing
    (LEFT_SUBTASKS[1], RIGHT_SUBTASKS[4]),  # nothing + press button
]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _first_sustained(mask: np.ndarray, n: int, start: int = 0) -> int | None:
    run = 0
    for i in range(start, len(mask)):
        run = run + 1 if mask[i] else 0
        if run >= n:
            return i - n + 1
    return None


def _last_sustained(mask: np.ndarray, n: int, end: int | None = None) -> int | None:
    end = len(mask) if end is None else end
    run = 0
    last: int | None = None
    for i in range(end):
        run = run + 1 if mask[i] else 0
        if run >= n:
            last = i
    return last


def _active_regions(speed: np.ndarray, percentile: float = 55.0, min_len: int = 20) -> list[tuple[int, int]]:
    thr = float(np.percentile(speed, percentile))
    active = speed > thr
    regions: list[tuple[int, int]] = []
    i = 0
    while i < len(active):
        if not active[i]:
            i += 1
            continue
        j = i
        while j < len(active) and active[j]:
            j += 1
        if j - i >= min_len:
            regions.append((i, j - 1))
        i = j
    return regions


def _clamp_split(value: int, lo: int, hi: int) -> int:
    return int(max(lo, min(hi, value)))


def detect_splits(state: np.ndarray) -> list[int]:
    """Return four split frame indices that define five segments."""
    total = len(state)
    if total < 40:
        return [max(1, total // 5), max(2, 2 * total // 5), max(3, 3 * total // 5), max(4, 4 * total // 5)]

    right_pos = state[:, 0:3]
    left_pos = state[:, 7:10]
    right_thumb = state[:, 26]
    right_speed = np.linalg.norm(np.diff(right_pos, axis=0, prepend=right_pos[:1]), axis=1)
    left_speed = np.linalg.norm(np.diff(left_pos, axis=0, prepend=left_pos[:1]), axis=1)

    left_regions = _active_regions(left_speed, percentile=55, min_len=25)
    door_open_end = left_regions[0][1] if left_regions else int(total * 0.18)

    grasp_ready = _first_sustained(right_thumb > 0.62, n=25, start=35)
    if grasp_ready is None:
        grasp_ready = _first_sustained(right_thumb > 0.55, n=15, start=35) or int(total * 0.15)
    split_open_pick = max(door_open_end, grasp_ready)

    in_cavity = (right_pos[:, 1] > 0.03) & (right_pos[:, 2] < 1.28)
    place_start = _first_sustained(in_cavity, n=12, start=split_open_pick + 10)
    if place_start is None:
        place_start = int(total * 0.45)
    place_end = _last_sustained(in_cavity, n=12, end=total)
    if place_end is None:
        place_end = int(total * 0.62)

    outside_cavity = ~in_cavity
    exit_cavity = _first_sustained(outside_cavity, n=18, start=place_end + 3)
    if exit_cavity is not None:
        move_out_end = exit_cavity + 25
    else:
        move_out_end = place_end + 35

    close_end = int(total * 0.85)
    for start, end in left_regions:
        if start > move_out_end + 10:
            close_end = end
            break

    tail = np.arange(max(int(total * 0.68), close_end - 3), total)
    button_start = int(tail[np.argmin(right_pos[tail, 1])]) if len(tail) else total - 1

    raw_splits = [
        split_open_pick,
        place_start,
        move_out_end,
        max(close_end, button_start - 5),
    ]

    # Enforce monotonic order with minimum segment length.
    min_seg = max(8, total // 80)
    splits: list[int] = []
    prev = 0
    for idx, candidate in enumerate(raw_splits):
        lo = prev + min_seg
        hi = total - min_seg * (4 - idx)
        value = _clamp_split(candidate, lo, hi)
        splits.append(value)
        prev = value

    return splits


def build_segments(total: int, splits: list[int]) -> list[dict[str, Any]]:
    points = [0, *splits, total - 1]
    segments: list[dict[str, Any]] = []
    for i in range(len(points) - 1):
        left_label, right_label = SEGMENT_LABELS[min(i, len(SEGMENT_LABELS) - 1)]
        segments.append(
            {
                "start_frame": int(points[i]),
                "end_frame": int(points[i + 1] - 1) if i < len(points) - 2 else int(total - 1),
                "left_subtask": left_label,
                "right_subtask": right_label,
            }
        )
    return segments


def annotate_episode(
    episode_index: int,
    length: int,
    task: str,
    parquet_path: Path,
) -> dict[str, Any]:
    table = pq.read_table(parquet_path)
    state = np.stack(table.column("observation.state").to_pylist())
    if len(state) != length:
        length = len(state)
    splits = detect_splits(state)
    return {
        "episode_index": episode_index,
        "task": task,
        "length": length,
        "splits": splits,
        "segments": build_segments(length, splits),
        "auto_annotated": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/dexjoco_microwave_cook"),
        help="DexJoCo LeRobot dataset root.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output jsonl path (default: <dataset>/annotations/dual_hand_subtasks.jsonl).",
    )
    parser.add_argument(
        "--episodes",
        type=str,
        default=None,
        help="Optional comma-separated episode indices, e.g. 0,1,5 or 0-10.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace existing annotations.")
    return parser.parse_args()


def _parse_episode_spec(spec: str, total: int) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted({i for i in out if 0 <= i < total})


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset.resolve()
    meta_dir = dataset_root / "meta"
    episodes = _read_jsonl(meta_dir / "episodes.jsonl")
    tasks = {row["task_index"]: row["task"] for row in _read_jsonl(meta_dir / "tasks.jsonl")}
    info = json.loads((meta_dir / "info.json").read_text(encoding="utf-8"))
    chunk_size = int(info.get("chunks_size", 1000))
    data_template = info["data_path"]

    output_path = args.output or (dataset_root / "annotations" / ANNOTATION_FILENAME)
    existing = {row["episode_index"]: row for row in _read_jsonl(output_path)}

    if args.episodes:
        episode_indices = _parse_episode_spec(args.episodes, len(episodes))
    else:
        episode_indices = list(range(len(episodes)))

    written = 0
    for episode_index in episode_indices:
        if episode_index in existing and not args.overwrite:
            continue
        ep = episodes[episode_index]
        chunk = episode_index // chunk_size
        parquet_path = dataset_root / data_template.format(
            episode_chunk=chunk,
            episode_index=episode_index,
        )
        task = ep["tasks"][0] if ep.get("tasks") else tasks.get(ep.get("task_index", 0), DEFAULT_TASK)
        row = annotate_episode(episode_index, int(ep["length"]), task, parquet_path)
        existing[episode_index] = row
        written += 1
        seg_summary = ", ".join(
            f"[{s['start_frame']}-{s['end_frame']}] L={s['left_subtask'][:12]}.. R={s['right_subtask'][:12]}.."
            for s in row["segments"]
        )
        print(f"episode {episode_index:03d}: splits={row['splits']} | {seg_summary}")

    ordered = [existing[i] for i in sorted(existing)]
    _write_jsonl(output_path, ordered)
    print(f"\nWrote {written} episode(s) to {output_path} (total {len(ordered)}).")


if __name__ == "__main__":
    main()
