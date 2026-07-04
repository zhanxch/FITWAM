#!/usr/bin/env python3
"""Compute FastWAM action/state normalization stats for DexJoCo success datasets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from fastwam.datasets.lerobot.base_lerobot_dataset import BaseLerobotDataset
from fastwam.datasets.lerobot.processors.fastwam_processor import FastWAMProcessor
from fastwam.datasets.lerobot.transforms.action_state_merger import ConcatLeftAlign
from fastwam.datasets.lerobot.utils.normalizer import save_dataset_stats_to_json


DEFAULT_TASKS = [
    "hammer_nail",
    "click_mouse",
    "pick_bucket",
    "pinch_tongs",
    "fold_glasses",
]
DEFAULT_DATASET_ROOT = Path("/data_all/share/datasets/dexjoco/dexjoco_lerobot_datasets")
DEFAULT_STATS_DIR = Path("/data_all/zhaoyc/Summer2/FastWAM_zhaoyc_failure_moved_from_share_20260703/artifacts/dataset_stats")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--stats-dir", type=Path, default=DEFAULT_STATS_DIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def image_keys_from_info(dataset_dir: Path) -> list[str]:
    info = read_json(dataset_dir / "meta" / "info.json")
    keys = [
        str(key).split("observation.images.", 1)[1]
        for key in info.get("features", {})
        if str(key).startswith("observation.images.")
    ]
    if not keys:
        raise ValueError(f"No image features found in {dataset_dir / 'meta' / 'info.json'}")
    return keys


def make_shape_meta(image_keys: list[str]) -> dict[str, Any]:
    return {
        "images": [
            {"key": key, "raw_shape": [3, 640, 640], "shape": [3, 384, 384]}
            for key in image_keys
        ],
        "action": [{"key": "default", "raw_shape": 22, "shape": 22}],
        "state": [{"key": "default", "raw_shape": 23, "shape": 23}],
    }


def make_processor(shape_meta: dict[str, Any]) -> FastWAMProcessor:
    return FastWAMProcessor(
        shape_meta=shape_meta,
        num_obs_steps=33,
        num_output_cameras=len(shape_meta["images"]),
        action_output_dim=22,
        proprio_output_dim=23,
        action_state_transforms=None,
        use_stepwise_action_norm=False,
        norm_default_mode="min/max",
        norm_exception_mode=None,
        action_state_merger=ConcatLeftAlign(),
        train_transforms=None,
        val_transforms=None,
    )


def compute_task_stats(task: str, dataset_root: Path, stats_dir: Path, *, overwrite: bool) -> Path:
    dataset_dir = dataset_root / task
    if not dataset_dir.exists():
        raise FileNotFoundError(dataset_dir)

    output_path = stats_dir / f"dexjoco_{task}_success_action_state.json"
    if output_path.exists() and not overwrite:
        print(f"[stats] skip existing {output_path}", flush=True)
        return output_path

    image_keys = image_keys_from_info(dataset_dir)
    shape_meta = make_shape_meta(image_keys)
    processor = make_processor(shape_meta)
    dataset = BaseLerobotDataset(
        dataset_dirs=[str(dataset_dir)],
        shape_meta=shape_meta,
        obs_size=33,
        action_size=32,
        val_set_proportion=0.01,
        is_training_set=True,
        global_sample_stride=1,
        tolerance_s=1e-4,
        video_backend="pyav",
    )
    stats = dataset.get_dataset_stats(processor)
    save_dataset_stats_to_json(stats, str(output_path))
    print(f"[stats] wrote {output_path} image_keys={image_keys}", flush=True)
    return output_path


def main() -> None:
    args = parse_args()
    args.stats_dir.mkdir(parents=True, exist_ok=True)
    for task in args.tasks:
        compute_task_stats(
            task,
            args.dataset_root.expanduser(),
            args.stats_dir.expanduser(),
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
