import math
from typing import Any

import numpy as np

from .constants import CHUNKS_SIZE, VIDEO_KEY


def feature_names(prefix: str, dim: int) -> list[str]:
    return [f"{prefix}_{i}" for i in range(dim)]


def stats(values: np.ndarray) -> dict[str, list[float] | list[int]]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    elif arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)
    return {
        "mean": arr.mean(axis=0).astype(float).tolist(),
        "std": arr.std(axis=0).astype(float).tolist(),
        "min": arr.min(axis=0).astype(float).tolist(),
        "max": arr.max(axis=0).astype(float).tolist(),
        "q01": np.quantile(arr, 0.01, axis=0).astype(float).tolist(),
        "q99": np.quantile(arr, 0.99, axis=0).astype(float).tolist(),
        "count": [int(arr.shape[0])],
    }


def combine_stats(stats_with_counts: list[tuple[dict[str, Any], int]]) -> dict[str, Any]:
    total = sum(count for _, count in stats_with_counts)
    combined: dict[str, Any] = {}

    for key in stats_with_counts[0][0]:
        combined[key] = {}
        for stat_name in ("mean", "std", "min", "max", "q01", "q99"):
            values = [
                np.asarray(cur_stats[key][stat_name], dtype=np.float64)
                for cur_stats, _ in stats_with_counts
            ]
            counts = np.asarray([count for _, count in stats_with_counts], dtype=np.float64)

            if stat_name == "mean":
                value = sum(v * c for v, c in zip(values, counts, strict=True)) / total
            elif stat_name == "std":
                means = [
                    np.asarray(cur_stats[key]["mean"], dtype=np.float64)
                    for cur_stats, _ in stats_with_counts
                ]
                global_mean = sum(m * c for m, c in zip(means, counts, strict=True)) / total
                variance = sum(
                    c * (np.square(s) + np.square(m - global_mean))
                    for s, m, c in zip(values, means, counts, strict=True)
                ) / total
                value = np.sqrt(np.maximum(variance, 0.0))
            elif stat_name in {"min", "q01"}:
                value = np.minimum.reduce(values)
            else:
                value = np.maximum.reduce(values)
            combined[key][stat_name] = value.astype(float).tolist()
        combined[key]["count"] = [total]

    if "task_index" in combined:
        task_mean = sum(i * count for i, (_, count) in enumerate(stats_with_counts)) / total
        combined["task_index"] = {
            "mean": [task_mean],
            "std": [
                math.sqrt(
                    sum(
                        count * (i - task_mean) ** 2
                        for i, (_, count) in enumerate(stats_with_counts)
                    )
                    / total
                )
            ],
            "min": [0.0],
            "max": [float(len(stats_with_counts) - 1)],
            "q01": [0.0],
            "q99": [float(len(stats_with_counts) - 1)],
            "count": [total],
        }

    return combined


def build_info(
    *,
    action_dim: int,
    state_dim: int,
    image_shape: tuple[int, int, int],
    fps: int,
    total_episodes: int,
    total_frames: int,
    total_tasks: int,
    include_hand_pose: bool,
) -> dict[str, Any]:
    height, width, channels = image_shape
    if channels != 3:
        raise ValueError(f"Expected RGB images with 3 channels, got {image_shape}")

    features: dict[str, Any] = {
        "observation.state": {
            "dtype": "float32",
            "shape": [state_dim],
            "names": feature_names("qpos", state_dim),
        },
        "action": {
            "dtype": "float32",
            "shape": [action_dim],
            "names": feature_names("action", action_dim),
        },
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
        VIDEO_KEY: {
            "dtype": "video",
            "shape": [height, width, channels],
            "names": ["height", "width", "rgb"],
            "info": {
                "video.fps": fps,
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "has_audio": False,
            },
        },
    }

    if include_hand_pose:
        features["observation.state.left_ee_pose"] = {
            "dtype": "float32",
            "shape": [7],
            "names": ["x", "y", "z", "qx", "qy", "qz", "qw"],
        }
        features["observation.state.right_ee_pose"] = {
            "dtype": "float32",
            "shape": [7],
            "names": ["x", "y", "z", "qx", "qy", "qz", "qw"],
        }
        features["observation.state.left_finger_tip_pos"] = {
            "dtype": "float32",
            "shape": [15],
            "names": [f"left_tip_{tip}_{axis}" for tip in range(5) for axis in ("x", "y", "z")],
        }
        features["observation.state.right_finger_tip_pos"] = {
            "dtype": "float32",
            "shape": [15],
            "names": [f"right_tip_{tip}_{axis}" for tip in range(5) for axis in ("x", "y", "z")],
        }
    return {
        "codebase_version": "v2.0",
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "fps": fps,
        "chunks_size": CHUNKS_SIZE,
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": total_tasks,
        "total_videos": total_episodes,
        "total_chunks": math.ceil(total_episodes / CHUNKS_SIZE),
        "splits": {"train": f"0:{total_episodes}"},
        "features": features,
        "robot_type": "EgoVLA_SIM",
    }

