#!/usr/bin/env python3
"""Merge collected DexJoCo failure LeRobot shards into one dataset directory."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from fastwam.datasets.lerobot.lerobot.datasets.compute_stats import aggregate_stats
from fastwam.datasets.lerobot.lerobot.datasets.utils import cast_stats_to_numpy, serialize_dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-episodes", type=int, default=None)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def chunk_for_episode(index: int, chunk_size: int) -> int:
    return index // chunk_size


def parquet_path(root: Path, episode_index: int, chunk_size: int) -> Path:
    chunk = chunk_for_episode(episode_index, chunk_size)
    return root / "data" / f"chunk-{chunk:03d}" / f"episode_{episode_index:06d}.parquet"


def video_path(root: Path, video_key: str, episode_index: int, chunk_size: int) -> Path:
    chunk = chunk_for_episode(episode_index, chunk_size)
    return root / "videos" / f"chunk-{chunk:03d}" / video_key / f"episode_{episode_index:06d}.mp4"


def video_keys_from_info(info: dict[str, Any]) -> list[str]:
    return [
        key
        for key, feature in info.get("features", {}).items()
        if feature.get("dtype") in {"video", "image"} and key.startswith("observation.images")
    ]


def validate_task_text(existing: str | None, root: Path) -> str:
    rows = read_jsonl(root / "meta" / "tasks.jsonl")
    if not rows:
        raise ValueError(f"Missing task metadata: {root / 'meta/tasks.jsonl'}")
    task = rows[0]["task"]
    if existing is not None and task != existing:
        raise ValueError(f"Task text mismatch in {root}: {task!r} != {existing!r}")
    return task


def main() -> None:
    args = parse_args()
    output: Path = args.output
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output exists: {output}")
        shutil.rmtree(output)

    inputs = [p.resolve() for p in args.inputs if p.exists()]
    if not inputs:
        raise ValueError("No existing input shard was provided")

    first_info = read_json(inputs[0] / "meta" / "info.json")
    chunk_size = int(first_info.get("chunks_size", 1000))
    video_keys = video_keys_from_info(first_info)
    task_text: str | None = None
    total_frames = 0
    total_episodes = 0
    episode_stats: list[dict[str, dict]] = []

    output.mkdir(parents=True, exist_ok=True)
    if (inputs[0] / "meta" / "modality.json").exists():
        shutil.copy2(inputs[0] / "meta" / "modality.json", output / "meta" / "modality.json")

    for root in inputs:
        info = read_json(root / "meta" / "info.json")
        if int(info.get("chunks_size", chunk_size)) != chunk_size:
            raise ValueError(f"Chunk size mismatch in {root}")
        if video_keys_from_info(info) != video_keys:
            raise ValueError(f"Video key mismatch in {root}")
        task_text = validate_task_text(task_text, root)

        episodes = sorted(read_jsonl(root / "meta" / "episodes.jsonl"), key=lambda x: int(x["episode_index"]))
        stats_by_episode = {
            int(row["episode_index"]): row.get("stats", {})
            for row in read_jsonl(root / "meta" / "episodes_stats.jsonl")
        }

        for episode in episodes:
            if args.max_episodes is not None and total_episodes >= args.max_episodes:
                break
            old_index = int(episode["episode_index"])
            old_parquet = parquet_path(root, old_index, chunk_size)
            if not old_parquet.exists():
                raise FileNotFoundError(old_parquet)

            for video_key in video_keys:
                old_video = video_path(root, video_key, old_index, chunk_size)
                if not old_video.exists():
                    raise FileNotFoundError(old_video)

            df = pd.read_parquet(old_parquet)
            length = int(len(df))
            df["episode_index"] = total_episodes
            df["index"] = range(total_frames, total_frames + length)

            new_parquet = parquet_path(output, total_episodes, chunk_size)
            new_parquet.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(new_parquet, index=False)

            for video_key in video_keys:
                src = video_path(root, video_key, old_index, chunk_size)
                dst = video_path(output, video_key, total_episodes, chunk_size)
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)

            new_episode = dict(episode)
            new_episode["episode_index"] = total_episodes
            new_episode["length"] = length
            append_jsonl(output / "meta" / "episodes.jsonl", new_episode)

            if old_index in stats_by_episode:
                stat_payload = stats_by_episode[old_index]
                append_jsonl(
                    output / "meta" / "episodes_stats.jsonl",
                    {"episode_index": total_episodes, "stats": stat_payload},
                )
                episode_stats.append(cast_stats_to_numpy(stat_payload))

            total_frames += length
            total_episodes += 1

        if args.max_episodes is not None and total_episodes >= args.max_episodes:
            break

    if task_text is None:
        raise ValueError("No episodes were merged")

    info = dict(first_info)
    info["total_episodes"] = total_episodes
    info["total_frames"] = total_frames
    info["total_tasks"] = 1
    info["total_chunks"] = chunk_for_episode(max(total_episodes - 1, 0), chunk_size) + 1
    info["total_videos"] = total_episodes * len(video_keys)
    info["splits"] = {"train": f"0:{total_episodes}"}

    append_jsonl(output / "meta" / "tasks.jsonl", {"task_index": 0, "task": task_text})
    write_json(output / "meta" / "info.json", info)
    if episode_stats:
        write_json(output / "meta" / "stats.json", serialize_dict(aggregate_stats(episode_stats)))

    summary = {
        "status": "complete",
        "inputs": [str(p) for p in inputs],
        "output": str(output),
        "episodes": total_episodes,
        "frames": total_frames,
        "video_keys": video_keys,
        "task": task_text,
    }
    write_json(output / "merge_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
