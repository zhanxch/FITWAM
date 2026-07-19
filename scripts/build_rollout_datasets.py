#!/usr/bin/env python3
"""Build raw and failure-tail-trimmed LeRobot rollout datasets from shards."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


FAILURE_PHRASE = "Failed to finish the whole process."
OUTCOME_LEDGER_NAME = "episode_outcomes.jsonl"
OUTCOME_SOURCE = "dexjoco_env"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    merge = subparsers.add_parser("merge-shards")
    merge.add_argument("--shard-datasets", type=Path, nargs="+", required=True)
    merge.add_argument("--output-dataset", type=Path, required=True)
    merge.add_argument("--failure-phrase", type=str, default=FAILURE_PHRASE)
    merge.add_argument("--overwrite", action="store_true")

    trim = subparsers.add_parser("trim-failures")
    trim.add_argument("--source-dataset", type=Path, required=True)
    trim.add_argument("--output-dataset", type=Path, required=True)
    trim.add_argument("--trim-failure-seconds", type=float, default=8.0)
    trim.add_argument(
        "--trim-only-length",
        type=int,
        default=600,
        help="Only trim failed episodes with this original frame length. Use <=0 to trim all failures.",
    )
    trim.add_argument("--failure-phrase", type=str, default=FAILURE_PHRASE)
    trim.add_argument("--overwrite", action="store_true")

    validate = subparsers.add_parser("validate-outcomes")
    validate.add_argument("--dataset", type=Path, required=True)
    validate.add_argument("--expected-episodes", type=int, default=None)
    validate.add_argument("--failure-phrase", type=str, default=FAILURE_PHRASE)
    validate.add_argument(
        "--check-media",
        action="store_true",
        help="Also decode every video and validate parquet, stats, and frame indexes.",
    )
    validate.add_argument("--report", type=Path, default=None)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_episode_rows(episodes: list[dict[str, Any]], *, dataset_root: Path) -> None:
    indices = []
    for row_number, episode in enumerate(episodes, start=1):
        episode_index = episode.get("episode_index")
        if isinstance(episode_index, bool) or not isinstance(episode_index, int):
            raise ValueError(
                f"{dataset_root}: episode row {row_number} has invalid episode_index={episode_index!r}"
            )
        indices.append(episode_index)
    if len(indices) != len(set(indices)):
        raise ValueError(f"{dataset_root}: duplicate episode_index in meta/episodes.jsonl")
    expected = list(range(len(indices)))
    if sorted(indices) != expected:
        raise ValueError(
            f"{dataset_root}: episode indexes must be contiguous 0..{len(indices) - 1}; "
            f"got {sorted(indices)}"
        )


def validate_outcome_rows(
    rows: list[dict[str, Any]],
    episodes: list[dict[str, Any]],
    *,
    dataset_root: Path,
    failure_phrase: str,
) -> dict[int, dict[str, Any]]:
    required = {
        "episode_index",
        "outcome",
        "success",
        "attempt_index",
        "seed",
        "source",
    }
    by_episode: dict[int, dict[str, Any]] = {}
    episodes_by_index = {int(ep["episode_index"]): ep for ep in episodes}
    for row_number, row in enumerate(rows, start=1):
        missing = required.difference(row)
        if missing:
            raise ValueError(
                f"{dataset_root}: outcome row {row_number} is missing fields {sorted(missing)}"
            )
        episode_index = row["episode_index"]
        success = row["success"]
        attempt_index = row["attempt_index"]
        seed = row["seed"]
        if isinstance(episode_index, bool) or not isinstance(episode_index, int) or episode_index < 0:
            raise ValueError(
                f"{dataset_root}: outcome row {row_number} has invalid "
                f"episode_index={episode_index!r}"
            )
        if episode_index in by_episode:
            raise ValueError(
                f"{dataset_root}: duplicate outcome row for episode_index={episode_index}"
            )
        if isinstance(success, bool) is False:
            raise ValueError(
                f"{dataset_root}: outcome row {row_number} has non-boolean success={success!r}"
            )
        expected_outcome = "success" if success else "failure"
        if row["outcome"] != expected_outcome:
            raise ValueError(
                f"{dataset_root}: outcome row {row_number} is inconsistent: "
                f"outcome={row['outcome']!r}, success={success!r}"
            )
        if isinstance(attempt_index, bool) or not isinstance(attempt_index, int) or attempt_index < 0:
            raise ValueError(
                f"{dataset_root}: outcome row {row_number} has invalid "
                f"attempt_index={attempt_index!r}"
            )
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError(
                f"{dataset_root}: outcome row {row_number} has invalid seed={seed!r}"
            )
        if row["source"] != OUTCOME_SOURCE:
            raise ValueError(
                f"{dataset_root}: outcome row {row_number} has source={row['source']!r}; "
                f"expected {OUTCOME_SOURCE!r}"
            )
        by_episode[episode_index] = row

    episode_indexes = set(episodes_by_index)
    ledger_indexes = set(by_episode)
    if ledger_indexes != episode_indexes:
        raise ValueError(
            f"{dataset_root}: outcome ledger must contain exactly one row per episode; "
            f"missing={sorted(episode_indexes - ledger_indexes)} "
            f"extra={sorted(ledger_indexes - episode_indexes)}"
        )

    summary_path = dataset_root / "collection_summary.json"
    summary = read_json(summary_path) if summary_path.exists() else {}
    task_mode = summary.get("outcome_task_mode")
    for episode_index, episode in episodes_by_index.items():
        has_failure_marker = any(
            failure_phrase in str(task) for task in episode.get("tasks", [])
        )
        outcome = by_episode[episode_index]["outcome"]
        if has_failure_marker and outcome != "failure":
            raise ValueError(
                f"{dataset_root}: episode {episode_index} has a failure task marker "
                f"but ledger outcome={outcome!r}"
            )
        if task_mode == "task-marker" and has_failure_marker != (outcome == "failure"):
            raise ValueError(
                f"{dataset_root}: episode {episode_index} task marker and ledger outcome disagree"
            )
        if task_mode == "clean" and has_failure_marker:
            raise ValueError(
                f"{dataset_root}: clean task mode contains a failure task marker "
                f"for episode {episode_index}"
            )

    attempt_log = summary.get("attempt_log")
    if attempt_log is not None:
        attempts_by_episode: dict[int, dict[str, Any]] = {}
        for attempt in attempt_log:
            saved_episode_index = attempt.get("saved_episode_index")
            if saved_episode_index is None:
                continue
            saved_episode_index = int(saved_episode_index)
            if saved_episode_index in attempts_by_episode:
                raise ValueError(
                    f"{dataset_root}: attempt log maps multiple attempts to episode "
                    f"{saved_episode_index}"
                )
            attempts_by_episode[saved_episode_index] = attempt
        if set(attempts_by_episode) != episode_indexes:
            raise ValueError(
                f"{dataset_root}: attempt log and outcome ledger cover different episodes"
            )
        for episode_index, outcome_row in by_episode.items():
            attempt = attempts_by_episode[episode_index]
            if (
                int(attempt["attempt_index"]) != outcome_row["attempt_index"]
                or int(attempt["seed"]) != outcome_row["seed"]
                or bool(attempt["success"]) != outcome_row["success"]
            ):
                raise ValueError(
                    f"{dataset_root}: outcome ledger row for episode {episode_index} "
                    "disagrees with attempt log"
                )
    return by_episode


def load_outcome_ledger(
    dataset_root: Path,
    episodes: list[dict[str, Any]],
    *,
    failure_phrase: str,
    required: bool,
) -> list[dict[str, Any]] | None:
    validate_episode_rows(episodes, dataset_root=dataset_root)
    path = dataset_root / "meta" / OUTCOME_LEDGER_NAME
    if not path.exists():
        if required:
            raise FileNotFoundError(
                f"{dataset_root}: missing required meta/{OUTCOME_LEDGER_NAME}"
            )
        return None
    rows = load_jsonl(path)
    validate_outcome_rows(
        rows,
        episodes,
        dataset_root=dataset_root,
        failure_phrase=failure_phrase,
    )
    return rows


def validate_outcome_dataset(
    dataset_root: Path,
    *,
    failure_phrase: str,
    expected_episodes: int | None = None,
    check_media: bool = False,
) -> dict[str, Any]:
    dataset_root = dataset_root.expanduser().resolve()
    info = read_json(dataset_root / "meta" / "info.json")
    episodes = load_jsonl(dataset_root / "meta" / "episodes.jsonl")
    outcomes = load_outcome_ledger(
        dataset_root,
        episodes,
        failure_phrase=failure_phrase,
        required=True,
    )
    if outcomes is None:
        raise AssertionError("Required outcome ledger unexpectedly resolved to None")

    observed_episodes = len(episodes)
    if expected_episodes is not None and observed_episodes != int(
        expected_episodes
    ):
        raise ValueError(
            f"{dataset_root}: expected {expected_episodes} episodes, "
            f"found {observed_episodes}"
        )
    if int(info.get("total_episodes", -1)) != observed_episodes:
        raise ValueError(
            f"{dataset_root}: meta/info.json total_episodes disagrees with "
            "meta/episodes.jsonl"
        )

    summary_path = dataset_root / "collection_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    summary = read_json(summary_path)
    if summary.get("status") != "complete":
        raise ValueError(
            f"{dataset_root}: collection status is {summary.get('status')!r}, "
            "expected 'complete'"
        )
    if int(summary.get("episodes", -1)) != observed_episodes:
        raise ValueError(
            f"{dataset_root}: collection summary episode count disagrees with "
            "meta/episodes.jsonl"
        )

    successes = sum(row["outcome"] == "success" for row in outcomes)
    failures = sum(row["outcome"] == "failure" for row in outcomes)
    if int(summary.get("successes_saved", -1)) != successes:
        raise ValueError(
            f"{dataset_root}: collection summary successes_saved disagrees "
            "with the outcome ledger"
        )
    if int(summary.get("failures", -1)) != failures:
        raise ValueError(
            f"{dataset_root}: collection summary failures disagrees with the "
            "outcome ledger"
        )

    physical_report = (
        validate_dataset_files(dataset_root, info, episodes)
        if check_media
        else None
    )
    ledger_path = dataset_root / "meta" / OUTCOME_LEDGER_NAME
    report = {
        "status": "valid",
        "dataset_root": str(dataset_root),
        "episodes": observed_episodes,
        "successes": successes,
        "failures": failures,
        "check_media": bool(check_media),
        "outcome_task_mode": summary.get("outcome_task_mode"),
        "outcome_ledger": str(ledger_path),
        "outcome_ledger_sha256": file_sha256(ledger_path),
    }
    if physical_report is not None:
        report["physical_validation"] = physical_report
    return report


def video_keys(info: dict[str, Any]) -> list[str]:
    return [key for key, spec in info["features"].items() if spec.get("dtype") == "video"]


def _int_column(table: pa.Table, name: str, *, path: Path) -> list[int]:
    if name not in table.column_names:
        raise ValueError(f"{path}: missing required parquet column {name!r}")
    return [
        int(value)
        for value in table[name].combine_chunks().to_pylist()
    ]


def count_video_frames(path: Path) -> int:
    try:
        container = av.open(str(path), mode="r")
    except Exception as exc:
        raise ValueError(f"{path}: cannot open video: {exc}") from exc
    try:
        if not container.streams.video:
            raise ValueError(f"{path}: contains no video stream")
        stream = container.streams.video[0]
        return sum(1 for _ in container.decode(stream))
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"{path}: video decode failed: {exc}") from exc
    finally:
        container.close()


def validate_dataset_files(
    dataset_root: Path,
    info: dict[str, Any],
    episodes: list[dict[str, Any]],
) -> dict[str, Any]:
    stats_path = dataset_root / "meta" / "episodes_stats.jsonl"
    if not stats_path.exists():
        raise FileNotFoundError(stats_path)
    stats_rows = load_jsonl(stats_path)
    expected_episode_indexes = list(range(len(episodes)))
    stats_indexes = [row.get("episode_index") for row in stats_rows]
    if stats_indexes != expected_episode_indexes:
        raise ValueError(
            f"{dataset_root}: meta/episodes_stats.jsonl episode indexes must be "
            f"{expected_episode_indexes}, got {stats_indexes}"
        )

    episode_indexes = [row.get("episode_index") for row in episodes]
    if episode_indexes != expected_episode_indexes:
        raise ValueError(
            f"{dataset_root}: meta/episodes.jsonl rows must be ordered by contiguous "
            f"episode_index; got {episode_indexes}"
        )

    global_start = 0
    checked_videos = 0
    keys = video_keys(info)
    stats_feature_keys = [
        key
        for key, spec in info["features"].items()
        if spec.get("dtype") not in {"image", "video", "string"}
    ]
    for episode in episodes:
        episode_index = int(episode["episode_index"])
        length = episode.get("length")
        if isinstance(length, bool) or not isinstance(length, int) or length <= 0:
            raise ValueError(
                f"{dataset_root}: episode {episode_index} has invalid length={length!r}"
            )
        episode_stats = stats_rows[episode_index].get("stats")
        if not isinstance(episode_stats, dict):
            raise ValueError(
                f"{stats_path}: episode {episode_index} stats must be an object"
            )
        missing_stats = set(stats_feature_keys).difference(episode_stats)
        if missing_stats:
            raise ValueError(
                f"{stats_path}: episode {episode_index} is missing stats for "
                f"{sorted(missing_stats)}"
            )
        for key in stats_feature_keys:
            count = episode_stats[key].get("count")
            if (
                not isinstance(count, list)
                or len(count) != 1
                or isinstance(count[0], bool)
                or not isinstance(count[0], (int, float))
                or int(count[0]) != length
            ):
                raise ValueError(
                    f"{stats_path}: episode {episode_index} feature {key!r} "
                    f"has count={count!r}, expected [{length}]"
                )
        chunk = episode_index // int(info["chunks_size"])
        parquet_path = dataset_root / info["data_path"].format(
            episode_chunk=chunk,
            episode_index=episode_index,
        )
        if not parquet_path.exists():
            raise FileNotFoundError(parquet_path)
        try:
            table = pq.read_table(parquet_path)
        except Exception as exc:
            raise ValueError(f"{parquet_path}: cannot read parquet: {exc}") from exc
        if int(table.num_rows) != length:
            raise ValueError(
                f"{parquet_path}: row count {table.num_rows} != episode length {length}"
            )

        frame_indexes = _int_column(table, "frame_index", path=parquet_path)
        if frame_indexes != list(range(length)):
            raise ValueError(f"{parquet_path}: frame_index is not contiguous 0..{length - 1}")
        parquet_episode_indexes = _int_column(
            table, "episode_index", path=parquet_path
        )
        if parquet_episode_indexes != [episode_index] * length:
            raise ValueError(
                f"{parquet_path}: episode_index column does not match {episode_index}"
            )
        global_indexes = _int_column(table, "index", path=parquet_path)
        expected_global_indexes = list(range(global_start, global_start + length))
        if global_indexes != expected_global_indexes:
            raise ValueError(
                f"{parquet_path}: global index is not contiguous "
                f"{global_start}..{global_start + length - 1}"
            )

        for key in keys:
            video_path = dataset_root / info["video_path"].format(
                episode_chunk=chunk,
                video_key=key,
                episode_index=episode_index,
            )
            if not video_path.exists():
                raise FileNotFoundError(video_path)
            frame_count = count_video_frames(video_path)
            if frame_count != length:
                raise ValueError(
                    f"{video_path}: decoded frame count {frame_count} "
                    f"!= episode length {length}"
                )
            checked_videos += 1
        global_start += length

    if int(info.get("total_frames", -1)) != global_start:
        raise ValueError(
            f"{dataset_root}: meta/info.json total_frames={info.get('total_frames')!r} "
            f"!= physical frame count {global_start}"
        )
    if "total_videos" in info and int(info["total_videos"]) != checked_videos:
        raise ValueError(
            f"{dataset_root}: meta/info.json total_videos={info['total_videos']!r} "
            f"!= physical video count {checked_videos}"
        )
    return {
        "episodes": len(episodes),
        "frames": global_start,
        "video_keys": keys,
        "videos": checked_videos,
    }


def fixed_size_float_array(values: np.ndarray, dim: int) -> pa.FixedSizeListArray:
    values = np.asarray(values, dtype=np.float32)
    flat = pa.array(values.reshape(-1), type=pa.float32())
    return pa.FixedSizeListArray.from_arrays(flat, dim)


def table_to_numpy_dict(table: pa.Table) -> dict[str, np.ndarray]:
    def col_to_2d(name: str, dim: int) -> np.ndarray:
        col = table[name].combine_chunks()
        flat = col.flatten().to_numpy(zero_copy_only=False)
        return flat.reshape(-1, dim)

    return {
        "action": col_to_2d("action", 22),
        "observation.state": col_to_2d("observation.state", 23),
        "timestamp": table["timestamp"].combine_chunks().to_numpy(zero_copy_only=False),
        "frame_index": table["frame_index"].combine_chunks().to_numpy(zero_copy_only=False),
        "episode_index": table["episode_index"].combine_chunks().to_numpy(zero_copy_only=False),
        "index": table["index"].combine_chunks().to_numpy(zero_copy_only=False),
        "task_index": table["task_index"].combine_chunks().to_numpy(zero_copy_only=False),
    }


def write_episode_parquet(path: Path, data: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "action": fixed_size_float_array(data["action"], 22),
            "observation.state": fixed_size_float_array(data["observation.state"], 23),
            "timestamp": pa.array(data["timestamp"].astype(np.float32), type=pa.float32()),
            "frame_index": pa.array(data["frame_index"].astype(np.int64), type=pa.int64()),
            "episode_index": pa.array(data["episode_index"].astype(np.int64), type=pa.int64()),
            "index": pa.array(data["index"].astype(np.int64), type=pa.int64()),
            "task_index": pa.array(data["task_index"].astype(np.int64), type=pa.int64()),
        }
    )
    pq.write_table(table.replace_schema_metadata(), path)


def feature_stats(array: np.ndarray, keepdims: bool) -> dict[str, np.ndarray]:
    return {
        "min": np.min(array, axis=0, keepdims=keepdims),
        "max": np.max(array, axis=0, keepdims=keepdims),
        "mean": np.mean(array, axis=0, keepdims=keepdims),
        "std": np.std(array, axis=0, keepdims=keepdims),
        "count": np.array([len(array)]),
    }


def compute_episode_stats(data: dict[str, np.ndarray], features: dict[str, Any]) -> dict[str, Any]:
    stats = {}
    for key in ("action", "observation.state", "timestamp", "frame_index", "episode_index", "index", "task_index"):
        if key not in features:
            continue
        arr = data[key]
        stats[key] = feature_stats(arr, keepdims=arr.ndim == 1)
    return stats


def aggregate_feature_stats(items: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    counts = np.stack([item["count"] for item in items], axis=0).squeeze(-1)
    total = counts.sum(axis=0, keepdims=True)
    mean = sum(item["mean"] * item["count"] for item in items) / total
    var = sum((item["std"] ** 2 + (item["mean"] - mean) ** 2) * item["count"] for item in items) / total
    return {
        "min": np.min(np.stack([item["min"] for item in items], axis=0), axis=0, keepdims=True),
        "max": np.max(np.stack([item["max"] for item in items], axis=0), axis=0, keepdims=True),
        "mean": mean,
        "std": np.sqrt(var),
        "count": total,
    }


def aggregate_stats(stats_list: list[dict[str, Any]]) -> dict[str, Any]:
    keys = sorted({key for stats in stats_list for key in stats.keys()})
    return {key: aggregate_feature_stats([stats[key] for stats in stats_list if key in stats]) for key in keys}


def serialize(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: serialize(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def copy_video(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def read_video_frames(path: Path) -> list[np.ndarray]:
    frames = []
    container = av.open(str(path), mode="r")
    try:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            frames.append(frame.to_ndarray(format="rgb24"))
    finally:
        container.close()
    return frames


def save_video_frames(frames: list[np.ndarray], path: Path, fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not frames:
        raise ValueError(f"Cannot write empty video: {path}")
    height, width = frames[0].shape[:2]
    container = av.open(str(path), mode="w")
    try:
        stream = container.add_stream("libx264", rate=fps)
        stream.width = int(width)
        stream.height = int(height)
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "21"}
        for frame in frames:
            video_frame = av.VideoFrame.from_ndarray(np.asarray(frame, dtype=np.uint8), format="rgb24")
            for packet in stream.encode(video_frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()


def prepare_output(source_info: dict[str, Any], output_root: Path, overwrite: bool) -> dict[str, Any]:
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(output_root)
        shutil.rmtree(output_root)
    (output_root / "meta").mkdir(parents=True)
    (output_root / "data" / "chunk-000").mkdir(parents=True)
    for key in video_keys(source_info):
        (output_root / "videos" / "chunk-000" / key).mkdir(parents=True)
    return copy.deepcopy(source_info)


def finalize_dataset(
    output_root: Path,
    info: dict[str, Any],
    episodes: list[dict[str, Any]],
    episode_stats: list[dict[str, Any]],
    *,
    total_frames: int,
    extra_summary: dict[str, Any],
    episode_outcomes: list[dict[str, Any]] | None = None,
) -> None:
    info["total_episodes"] = len(episodes)
    info["total_frames"] = int(total_frames)
    info["total_videos"] = len(episodes) * len(video_keys(info))
    info["total_chunks"] = 1 if episodes else 0
    info["splits"] = {"train": f"0:{len(episodes)}"}
    for key in video_keys(info):
        if episodes:
            path = output_root / info["video_path"].format(episode_chunk=0, video_key=key, episode_index=0)
            try:
                container = av.open(str(path), mode="r")
                try:
                    stream = container.streams.video[0]
                    info["features"][key]["info"] = {
                        "video.fps": float(stream.average_rate),
                        "video.height": int(stream.height),
                        "video.width": int(stream.width),
                    }
                finally:
                    container.close()
            except Exception:
                pass
    write_json(output_root / "meta" / "info.json", info)
    write_jsonl(output_root / "meta" / "episodes.jsonl", episodes)
    write_jsonl(
        output_root / "meta" / "episodes_stats.jsonl",
        [{"episode_index": i, "stats": serialize(stats)} for i, stats in enumerate(episode_stats)],
    )
    if episode_outcomes is not None:
        write_jsonl(output_root / "meta" / OUTCOME_LEDGER_NAME, episode_outcomes)
    write_json(output_root / "meta" / "stats.json", serialize(aggregate_stats(episode_stats)))
    write_json(output_root / "collection_summary.json", extra_summary)


def validate_merge_shard_summary(
    shard_root: Path,
    info: dict[str, Any],
    episodes: list[dict[str, Any]],
    outcomes: list[dict[str, Any]],
) -> dict[str, Any]:
    summary_path = shard_root / "collection_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"{shard_root}: missing required collection_summary.json"
        )
    summary = read_json(summary_path)
    if summary.get("status") != "complete":
        raise ValueError(
            f"{shard_root}: collection status is {summary.get('status')!r}, "
            "expected 'complete'"
        )
    if summary.get("mode") != "save_all":
        raise ValueError(
            f"{shard_root}: merge requires mode='save_all', "
            f"got {summary.get('mode')!r}"
        )
    if summary.get("outcome_task_mode") not in {"clean", "task-marker"}:
        raise ValueError(
            f"{shard_root}: outcome_task_mode must be 'clean' or 'task-marker', "
            f"got {summary.get('outcome_task_mode')!r}"
        )
    target = summary.get("target_episodes")
    if isinstance(target, bool) or not isinstance(target, int) or target < 1:
        raise ValueError(
            f"{shard_root}: target_episodes must be a positive integer, got {target!r}"
        )
    attempt_log = summary.get("attempt_log")
    if not isinstance(attempt_log, list):
        raise ValueError(f"{shard_root}: collection summary attempt_log must be a list")

    counts = {
        "info.total_episodes": info.get("total_episodes"),
        "summary.episodes": summary.get("episodes"),
        "episodes": len(episodes),
        "outcomes": len(outcomes),
        "attempt_log": len(attempt_log),
        "target_episodes": target,
    }
    normalized_counts: dict[str, int] = {}
    for name, value in counts.items():
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{shard_root}: {name} must be an integer, got {value!r}")
        normalized_counts[name] = int(value)
    if len(set(normalized_counts.values())) != 1:
        raise ValueError(
            f"{shard_root}: shard episode counts disagree: {normalized_counts}"
        )
    if int(summary.get("attempts", -1)) != len(attempt_log):
        raise ValueError(
            f"{shard_root}: summary.attempts disagrees with attempt_log length"
        )
    successes = sum(bool(row["success"]) for row in outcomes)
    failures = len(outcomes) - successes
    if int(summary.get("successes_saved", -1)) != successes:
        raise ValueError(
            f"{shard_root}: summary.successes_saved disagrees with outcome ledger"
        )
    if int(summary.get("failures", -1)) != failures:
        raise ValueError(
            f"{shard_root}: summary.failures disagrees with outcome ledger"
        )
    return summary


def merge_shards(shard_datasets: list[Path], output_dataset: Path, overwrite: bool, failure_phrase: str) -> None:
    validated_shards = []
    reference_tasks = None
    task_modes = set()
    for shard_root in shard_datasets:
        shard_info = read_json(shard_root / "meta" / "info.json")
        shard_episodes = load_jsonl(shard_root / "meta" / "episodes.jsonl")
        shard_outcomes = load_outcome_ledger(
            shard_root,
            shard_episodes,
            failure_phrase=failure_phrase,
            required=True,
        )
        if shard_outcomes is None:
            raise AssertionError("Required shard outcome ledger resolved to None")
        summary = validate_merge_shard_summary(
            shard_root,
            shard_info,
            shard_episodes,
            shard_outcomes,
        )
        task_rows = load_jsonl(shard_root / "meta" / "tasks.jsonl")
        if reference_tasks is None:
            reference_tasks = task_rows
        elif task_rows != reference_tasks:
            raise ValueError(f"{shard_root}: meta/tasks.jsonl differs from the first shard")
        task_mode = summary.get("outcome_task_mode")
        if task_mode is not None:
            task_modes.add(task_mode)
        validated_shards.append(
            (shard_root, shard_info, shard_episodes, shard_outcomes, summary)
        )
    if len(task_modes) > 1:
        raise ValueError(f"Shard outcome_task_mode values disagree: {sorted(task_modes)}")
    merged_task_mode = next(iter(task_modes), None)

    first_info = read_json(shard_datasets[0] / "meta" / "info.json")
    info = prepare_output(first_info, output_dataset, overwrite)
    shutil.copy2(shard_datasets[0] / "meta" / "tasks.jsonl", output_dataset / "meta" / "tasks.jsonl")
    modality = shard_datasets[0] / "meta" / "modality.json"
    if modality.exists():
        shutil.copy2(modality, output_dataset / "meta" / "modality.json")

    out_episodes = []
    out_stats = []
    global_index = 0
    new_ep_idx = 0
    attempt_log = []
    out_outcomes = []
    for shard_id, (
        shard_root,
        shard_info,
        shard_episodes,
        shard_outcomes,
        summary,
    ) in enumerate(
        validated_shards
    ):
        shard_outcomes_by_episode = {
            int(row["episode_index"]): row for row in shard_outcomes
        }
        shard_attempts_by_episode = {
            int(row["saved_episode_index"]): row
            for row in summary["attempt_log"]
        }
        for ep in sorted(shard_episodes, key=lambda row: int(row["episode_index"])):
            old_ep_idx = int(ep["episode_index"])
            old_chunk = old_ep_idx // int(shard_info["chunks_size"])
            src_parquet = shard_root / shard_info["data_path"].format(
                episode_chunk=old_chunk,
                episode_index=old_ep_idx,
            )
            data = table_to_numpy_dict(pq.read_table(src_parquet))
            length = len(data["action"])
            data["frame_index"] = np.arange(length, dtype=np.int64)
            data["timestamp"] = (data["frame_index"] / float(info["fps"])).astype(np.float32)
            data["episode_index"] = np.full(length, new_ep_idx, dtype=np.int64)
            data["index"] = np.arange(global_index, global_index + length, dtype=np.int64)

            chunk = new_ep_idx // int(info["chunks_size"])
            dst_parquet = output_dataset / info["data_path"].format(episode_chunk=chunk, episode_index=new_ep_idx)
            write_episode_parquet(dst_parquet, data)
            for key in video_keys(info):
                src_video = shard_root / shard_info["video_path"].format(
                    episode_chunk=old_chunk,
                    video_key=key,
                    episode_index=old_ep_idx,
                )
                dst_video = output_dataset / info["video_path"].format(
                    episode_chunk=chunk,
                    video_key=key,
                    episode_index=new_ep_idx,
                )
                copy_video(src_video, dst_video)

            out_episodes.append({"episode_index": new_ep_idx, "tasks": ep["tasks"], "length": length})
            remapped_outcome = dict(shard_outcomes_by_episode[old_ep_idx])
            source_attempt_index = int(remapped_outcome["attempt_index"])
            remapped_outcome["source_shard_id"] = shard_id
            remapped_outcome["source_episode_index"] = old_ep_idx
            remapped_outcome["source_attempt_index"] = source_attempt_index
            remapped_outcome["episode_index"] = new_ep_idx
            remapped_outcome["attempt_index"] = new_ep_idx
            out_outcomes.append(remapped_outcome)

            remapped_attempt = dict(shard_attempts_by_episode[old_ep_idx])
            if int(remapped_attempt["attempt_index"]) != source_attempt_index:
                raise ValueError(
                    f"{shard_root}: attempt log and ledger source attempt disagree "
                    f"for episode {old_ep_idx}"
                )
            remapped_attempt["source_shard_id"] = shard_id
            remapped_attempt["source_episode_index"] = old_ep_idx
            remapped_attempt["source_attempt_index"] = source_attempt_index
            remapped_attempt["shard_id"] = shard_id
            remapped_attempt["saved_episode_index"] = new_ep_idx
            remapped_attempt["attempt_index"] = new_ep_idx
            attempt_log.append(remapped_attempt)

            out_stats.append(compute_episode_stats(data, info["features"]))
            global_index += length
            new_ep_idx += 1

    validate_outcome_rows(
        out_outcomes,
        out_episodes,
        dataset_root=output_dataset,
        failure_phrase=failure_phrase,
    )
    finalize_dataset(
        output_dataset,
        info,
        out_episodes,
        out_stats,
        total_frames=global_index,
        extra_summary={
            "status": "complete",
            "mode": "raw_merged_save_all",
            "episodes": len(out_episodes),
            "failures": sum(1 for row in out_outcomes if row["outcome"] == "failure"),
            "successes_saved": sum(1 for row in out_outcomes if row["outcome"] == "success"),
            "outcome_source": OUTCOME_LEDGER_NAME,
            "outcome_task_mode": merged_task_mode,
            "shard_datasets": [str(path) for path in shard_datasets],
            "attempt_log": attempt_log,
        },
        episode_outcomes=out_outcomes,
    )


def trim_failures(
    source_dataset: Path,
    output_dataset: Path,
    trim_seconds: float,
    trim_only_length: int,
    failure_phrase: str,
    overwrite: bool,
) -> None:
    source_info = read_json(source_dataset / "meta" / "info.json")
    source_summary_path = source_dataset / "collection_summary.json"
    source_summary = read_json(source_summary_path) if source_summary_path.exists() else {}
    source_episodes = load_jsonl(source_dataset / "meta" / "episodes.jsonl")
    source_outcomes = load_outcome_ledger(
        source_dataset,
        source_episodes,
        failure_phrase=failure_phrase,
        required=False,
    )
    outcomes_by_episode = (
        {int(row["episode_index"]): row for row in source_outcomes}
        if source_outcomes is not None
        else None
    )
    info = prepare_output(source_info, output_dataset, overwrite)
    shutil.copy2(source_dataset / "meta" / "tasks.jsonl", output_dataset / "meta" / "tasks.jsonl")
    modality = source_dataset / "meta" / "modality.json"
    if modality.exists():
        shutil.copy2(modality, output_dataset / "meta" / "modality.json")

    fps = int(info["fps"])
    trim_steps = int(round(trim_seconds * fps))
    out_episodes = []
    out_stats = []
    trim_report = []
    global_index = 0
    for ep in sorted(source_episodes, key=lambda row: int(row["episode_index"])):
        ep_idx = int(ep["episode_index"])
        chunk = ep_idx // int(source_info["chunks_size"])
        if outcomes_by_episode is not None:
            is_failure = outcomes_by_episode[ep_idx]["outcome"] == "failure"
        else:
            is_failure = any(failure_phrase in str(task) for task in ep["tasks"])
        data = table_to_numpy_dict(
            pq.read_table(
                source_dataset / source_info["data_path"].format(episode_chunk=chunk, episode_index=ep_idx)
            )
        )
        orig_len = len(data["action"])
        should_trim = is_failure and (trim_only_length <= 0 or orig_len == trim_only_length)
        keep = max(1, orig_len - trim_steps) if should_trim else orig_len
        trimmed = {key: value[:keep] for key, value in data.items()}
        trimmed["frame_index"] = np.arange(keep, dtype=np.int64)
        trimmed["timestamp"] = (trimmed["frame_index"] / float(fps)).astype(np.float32)
        trimmed["episode_index"] = np.full(keep, ep_idx, dtype=np.int64)
        trimmed["index"] = np.arange(global_index, global_index + keep, dtype=np.int64)

        dst_parquet = output_dataset / info["data_path"].format(episode_chunk=chunk, episode_index=ep_idx)
        write_episode_parquet(dst_parquet, trimmed)
        for key in video_keys(info):
            src_video = source_dataset / source_info["video_path"].format(
                episode_chunk=chunk,
                video_key=key,
                episode_index=ep_idx,
            )
            dst_video = output_dataset / info["video_path"].format(
                episode_chunk=chunk,
                video_key=key,
                episode_index=ep_idx,
            )
            if should_trim:
                save_video_frames(read_video_frames(src_video)[:keep], dst_video, fps)
            else:
                copy_video(src_video, dst_video)

        out_episodes.append({"episode_index": ep_idx, "tasks": ep["tasks"], "length": keep})
        out_stats.append(compute_episode_stats(trimmed, info["features"]))
        trim_report.append(
            {
                "episode_index": ep_idx,
                "failure": is_failure,
                "trimmed": should_trim,
                "original_length": orig_len,
                "trimmed_length": keep,
                "trimmed_tail_steps": orig_len - keep,
            }
        )
        global_index += keep

    finalize_dataset(
        output_dataset,
        info,
        out_episodes,
        out_stats,
        total_frames=global_index,
        extra_summary={
            "status": "complete",
            "mode": "trimmed_failures",
            "source_dataset": str(source_dataset),
            "trim_failure_seconds": trim_seconds,
            "trim_only_length": int(trim_only_length),
            "episodes": len(out_episodes),
            "failures": sum(1 for item in trim_report if item["failure"]),
            "successes_saved": sum(1 for item in trim_report if not item["failure"]),
            "trimmed_failures": sum(1 for item in trim_report if item["trimmed"]),
            "outcome_source": (
                OUTCOME_LEDGER_NAME if source_outcomes is not None else "legacy_task_marker"
            ),
            "outcome_task_mode": source_summary.get("outcome_task_mode"),
            "trim_report": trim_report,
        },
        episode_outcomes=source_outcomes,
    )


def main() -> None:
    args = parse_args()
    if args.command == "merge-shards":
        merge_shards(
            [path.expanduser().resolve() for path in args.shard_datasets],
            args.output_dataset.expanduser().resolve(),
            args.overwrite,
            args.failure_phrase,
        )
    elif args.command == "trim-failures":
        trim_failures(
            args.source_dataset.expanduser().resolve(),
            args.output_dataset.expanduser().resolve(),
            args.trim_failure_seconds,
            args.trim_only_length,
            args.failure_phrase,
            args.overwrite,
        )
    elif args.command == "validate-outcomes":
        report = validate_outcome_dataset(
            args.dataset,
            failure_phrase=args.failure_phrase,
            expected_episodes=args.expected_episodes,
            check_media=args.check_media,
        )
        if args.report is not None:
            write_json(args.report.expanduser().resolve(), report)
        print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
