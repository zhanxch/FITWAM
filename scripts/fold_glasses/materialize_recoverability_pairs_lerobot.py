#!/usr/bin/env python3
"""Convert recoverability pair sidecars into a 33-frame LeRobot dataset."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.collect_dexjoco_rollouts import (  # noqa: E402
    aggregate_stats,
    compute_episode_stats,
    prepare_dataset,
    serialize_dict,
    update_info,
    write_episode_parquet,
    write_json,
)
from scripts.fold_glasses.collect_opensource_4x50 import SUCCESS_PROMPT  # noqa: E402
from scripts.fold_glasses.validate_factual_replay import read_json  # noqa: E402


EVENT_NUM_FRAMES = 33
FAILURE_PHRASE = "Failed to finish the whole process."


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def copy_pair_videos(
    descriptor: dict[str, Any],
    output_dataset: Path,
    info: dict[str, Any],
    episode_index: int,
) -> None:
    chunk = episode_index // int(info["chunks_size"])
    mapping = {
        "observation.images.front": Path(descriptor["front_video"]),
        "observation.images.wrist": Path(descriptor["wrist_video"]),
    }
    for video_key, src in mapping.items():
        dst = output_dataset / info["video_path"].format(
            episode_chunk=chunk,
            video_key=video_key,
            episode_index=episode_index,
        )
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def save_pair_episode(
    *,
    output_dataset: Path,
    info: dict[str, Any],
    stats_list: list[dict[str, dict]],
    episode_index: int,
    global_start_index: int,
    descriptor: dict[str, Any],
    task_text: str,
    fps: int,
) -> int:
    arrays = np.load(descriptor["arrays"])
    actions = np.asarray(arrays["actions"], dtype=np.float32)
    states = np.asarray(arrays["states"], dtype=np.float32)
    length = int(actions.shape[0])
    if length != EVENT_NUM_FRAMES:
        raise ValueError(
            f"Pair event must have {EVENT_NUM_FRAMES} frames, got {length}"
        )
    copy_pair_videos(descriptor, output_dataset, info, episode_index)
    timestamps = np.arange(length, dtype=np.float32) / float(fps)
    write_episode_parquet(
        output_dataset
        / info["data_path"].format(
            episode_chunk=episode_index // int(info["chunks_size"]),
            episode_index=episode_index,
        ),
        actions=actions,
        states=states,
        timestamps=timestamps,
        frame_indices=np.arange(length, dtype=np.int64),
        episode_indices=np.full((length,), episode_index, dtype=np.int64),
        global_indices=np.arange(
            global_start_index, global_start_index + length, dtype=np.int64
        ),
        task_indices=np.zeros((length,), dtype=np.int64),
    )
    append_jsonl(
        output_dataset / "meta" / "episodes.jsonl",
        {"episode_index": episode_index, "tasks": [task_text], "length": length},
    )
    ep_stats = compute_episode_stats(
        {
            "action": actions,
            "observation.state": states,
            "timestamp": timestamps,
            "frame_index": np.arange(length, dtype=np.int64),
            "episode_index": np.full((length,), episode_index, dtype=np.int64),
            "index": np.arange(
                global_start_index, global_start_index + length, dtype=np.int64
            ),
            "task_index": np.zeros((length,), dtype=np.int64),
        },
        info["features"],
        is_compute_episode_stats_image=False,
    )
    stats_list.append(ep_stats)
    append_jsonl(
        output_dataset / "meta" / "episodes_stats.jsonl",
        {"episode_index": episode_index, "stats": serialize_dict(ep_stats)},
    )
    return length


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan-root", type=Path, required=True)
    parser.add_argument("--source-dataset", type=Path, required=True)
    parser.add_argument("--output-dataset", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    scan_root = args.scan_root.expanduser().resolve()
    pairs = [
        row
        for row in load_jsonl(scan_root / "event_pair_manifest.jsonl")
        if row.get("status") == "complete" and bool(row.get("training_eligible", True))
    ]
    if not pairs:
        raise SystemExit(f"No complete training pairs under {scan_root}")

    info, n_ep, global_i, stats_list, _attempts = prepare_dataset(
        args.source_dataset.expanduser().resolve(),
        args.output_dataset.expanduser().resolve(),
        SUCCESS_PROMPT,
        f"{SUCCESS_PROMPT} {FAILURE_PHRASE}",
        overwrite=bool(args.overwrite),
        resume=False,
        save_all_trajectories=True,
        outcome_task_mode="clean",
    )
    fps = int(info.get("fps", 30))
    output = args.output_dataset.expanduser().resolve()
    pair_index: list[dict[str, Any]] = []
    for pair in pairs:
        success_desc = read_json(Path(pair["counterfactual_success_event"]))
        failure_desc = read_json(Path(pair["factual_failure_event"]))
        success_ep = n_ep
        length = save_pair_episode(
            output_dataset=output,
            info=info,
            stats_list=stats_list,
            episode_index=success_ep,
            global_start_index=global_i,
            descriptor=success_desc,
            task_text=SUCCESS_PROMPT,
            fps=fps,
        )
        append_jsonl(
            output / "meta" / "episode_outcomes.jsonl",
            {
                "episode_index": success_ep,
                "success": True,
                "outcome": "success",
                "pair_id": pair["pair_id"],
                "event_role": "success_event",
                "seed": pair["seed"],
                "source_failure_episode_index": pair["source_failure_episode_index"],
            },
        )
        global_i += length
        n_ep += 1
        failure_ep = n_ep
        length = save_pair_episode(
            output_dataset=output,
            info=info,
            stats_list=stats_list,
            episode_index=failure_ep,
            global_start_index=global_i,
            descriptor=failure_desc,
            task_text=SUCCESS_PROMPT,
            fps=fps,
        )
        append_jsonl(
            output / "meta" / "episode_outcomes.jsonl",
            {
                "episode_index": failure_ep,
                "success": False,
                "outcome": "failure",
                "pair_id": pair["pair_id"],
                "event_role": "failure_event",
                "seed": pair["seed"],
                "source_failure_episode_index": pair["source_failure_episode_index"],
            },
        )
        global_i += length
        n_ep += 1
        pair_index.append(
            {
                "pair_id": pair["pair_id"],
                "seed": pair["seed"],
                "success_episode_index": success_ep,
                "failure_episode_index": failure_ep,
                "t_frame": pair["frontier"]["t_frame"],
                "failure_frame": pair["frontier"].get(
                    "failure_frame", pair["frontier"]["t_plus_24_frame"]
                ),
            }
        )
        update_info(
            output,
            info,
            num_episodes=n_ep,
            total_frames=global_i,
            total_tasks=1,
        )

    write_json(output / "meta" / "stats.json", serialize_dict(aggregate_stats(stats_list)))
    write_json(output / "pair_index.json", {"pairs": pair_index, "num_pairs": len(pair_index)})
    print(
        f"wrote {n_ep} episodes ({len(pair_index)} pairs) to {output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
