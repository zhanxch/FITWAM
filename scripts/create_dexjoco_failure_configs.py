#!/usr/bin/env python3
"""Create DexJoCo two-camera failure-ablation configs for one or more tasks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


DEFAULT_TASKS = [
    "hammer_nail",
    "click_mouse",
    "pick_bucket",
    "pinch_tongs",
    "fold_glasses",
]
DEFAULT_SUCCESS_ROOT = Path("/data_all/share/datasets/dexjoco/dexjoco_lerobot_datasets")
DEFAULT_FAILURE_ROOT = Path("/data_all/share/dexjoco_failure_datasets")
DEFAULT_STATS_DIR = Path("/data_all/zhaoyc/Summer2/FastWAM_zhaoyc_failure_moved_from_share_20260703/artifacts/dataset_stats")
DEFAULT_TEXT_CACHE_ROOT = Path("/data_all/zhaoyc/Summer2/FastWAM_zhaoyc_failure_moved_from_share_20260703/artifacts/text_embeds_cache")
FAILURE_PHRASE = "Failed to finish the whole process."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate FastWAM YAML configs for DexJoCo failure embedding/text ablations."
    )
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    parser.add_argument("--success-root", type=Path, default=DEFAULT_SUCCESS_ROOT)
    parser.add_argument("--failure-root", type=Path, default=DEFAULT_FAILURE_ROOT)
    parser.add_argument("--stats-dir", type=Path, default=DEFAULT_STATS_DIR)
    parser.add_argument("--text-cache-root", type=Path, default=DEFAULT_TEXT_CACHE_ROOT)
    parser.add_argument("--config-root", type=Path, default=Path("configs"))
    parser.add_argument("--resume-map-json", type=Path, default=None)
    parser.add_argument("--max-steps", type=int, default=6000)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_resume_map(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    payload = read_json(path.expanduser())
    if not isinstance(payload, dict):
        raise ValueError(f"Resume map must be a JSON object: {path}")
    return {str(k): str(v) for k, v in payload.items()}


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


def shape_meta(image_keys: list[str]) -> dict[str, Any]:
    return {
        "images": [
            {"key": key, "raw_shape": [3, 640, 640], "shape": [3, 384, 384]}
            for key in image_keys
        ],
        "action": [{"key": "default", "raw_shape": 22, "shape": 22}],
        "state": [{"key": "default", "raw_shape": 23, "shape": 23}],
    }


def processor_cfg() -> dict[str, Any]:
    return {
        "_target_": "fastwam.datasets.lerobot.processors.fastwam_processor.FastWAMProcessor",
        "shape_meta": "${data.train.shape_meta}",
        "num_obs_steps": "${data.train.num_frames}",
        "num_output_cameras": 2,
        "action_output_dim": 22,
        "proprio_output_dim": 23,
        "action_state_transforms": None,
        "use_stepwise_action_norm": False,
        "norm_default_mode": "min/max",
        "norm_exception_mode": None,
        "action_state_merger": {
            "_target_": "fastwam.datasets.lerobot.transforms.action_state_merger.ConcatLeftAlign"
        },
        "train_transforms": [
            {"_target_": "fastwam.datasets.lerobot.transforms.image.ToTensor"},
            {"_target_": "torchvision.transforms.Resize", "size": [384, 384]},
        ],
        "val_transforms": [
            {"_target_": "fastwam.datasets.lerobot.transforms.image.ToTensor"},
            {"_target_": "torchvision.transforms.Resize", "size": [384, 384]},
        ],
    }


def data_cfg(
    task: str,
    *,
    variant: str,
    image_keys: list[str],
    success_root: Path,
    failure_root: Path,
    stats_dir: Path,
    text_cache_root: Path,
) -> dict[str, Any]:
    success_dir = success_root / task
    failure_dir = failure_root / f"{task}_failure_fastwam_2cam_text"
    if variant == "success":
        cache_dir = text_cache_root / f"dexjoco_{task}_2cam_success"
        dataset_dirs = [str(success_dir)]
        extra_train = {}
    elif variant == "failure_embedding":
        cache_dir = text_cache_root / f"dexjoco_{task}_2cam_success"
        dataset_dirs = [str(success_dir), str(failure_dir)]
        extra_train = {
            "action_loss_zero_if_instruction_contains": FAILURE_PHRASE,
            "outcome_flag_if_instruction_contains": FAILURE_PHRASE,
            "strip_instruction_suffix_if_contains": FAILURE_PHRASE,
        }
    elif variant == "text_failure":
        cache_dir = text_cache_root / f"dexjoco_{task}_2cam_text_failure"
        dataset_dirs = [str(success_dir), str(failure_dir)]
        extra_train = {
            "action_loss_zero_if_instruction_contains": FAILURE_PHRASE,
        }
    else:
        raise ValueError(f"Unknown variant: {variant}")

    train = {
        "_target_": "fastwam.datasets.lerobot.robot_video_dataset.RobotVideoDataset",
        "dataset_dirs": dataset_dirs,
        "shape_meta": shape_meta(image_keys),
        "num_frames": 33,
        "global_sample_stride": 1,
        "action_video_freq_ratio": 4,
        "video_size": [384, 768],
        "camera_key": None,
        "val_set_proportion": 0.01,
        "is_training_set": True,
        "pretrained_norm_stats": str(stats_dir / f"dexjoco_{task}_success_action_state.json"),
        "skip_padding_as_possible": False,
        "concat_multi_camera": "horizontal",
        **extra_train,
        "processor": processor_cfg(),
        "text_embedding_cache_dir": str(cache_dir),
        "context_len": 128,
    }

    val = {
        "_target_": "fastwam.datasets.lerobot.robot_video_dataset.RobotVideoDataset",
        "dataset_dirs": [str(success_dir)],
        "shape_meta": "${data.train.shape_meta}",
        "num_frames": "${data.train.num_frames}",
        "global_sample_stride": "${data.train.global_sample_stride}",
        "action_video_freq_ratio": "${data.train.action_video_freq_ratio}",
        "video_size": "${data.train.video_size}",
        "camera_key": None,
        "val_set_proportion": "${data.train.val_set_proportion}",
        "is_training_set": False,
        "pretrained_norm_stats": "${data.train.pretrained_norm_stats}",
        "skip_padding_as_possible": False,
        "concat_multi_camera": "${data.train.concat_multi_camera}",
        "processor": processor_cfg(),
        "text_embedding_cache_dir": "${data.train.text_embedding_cache_dir}",
        "context_len": "${data.train.context_len}",
    }
    if "action_loss_zero_if_instruction_contains" in train:
        val["action_loss_zero_if_instruction_contains"] = "${data.train.action_loss_zero_if_instruction_contains}"
    else:
        val["action_loss_zero_if_instruction_contains"] = None
    if variant == "failure_embedding":
        val["outcome_flag_if_instruction_contains"] = "${data.train.outcome_flag_if_instruction_contains}"
        val["strip_instruction_suffix_if_contains"] = "${data.train.strip_instruction_suffix_if_contains}"

    return {"train": train, "val": val}


def task_cfg(
    task: str,
    *,
    variant: str,
    resume: str | None,
    max_steps: int,
    save_every: int,
    eval_every: int,
    batch_size: int,
    gradient_accumulation_steps: int,
) -> dict[str, Any]:
    data_name = f"dexjoco_{task}_2cam_{variant}"
    run_name = f"dexjoco_{task}_{variant}_2cam_proprio_1e-4"
    model_cfg = {
        "proprio_dim": "${data.train.processor.proprio_output_dim}",
        "state_dit_config": None,
    }
    if variant != "success":
        model_cfg["skip_dit_load_from_pretrain"] = True
        model_cfg["action_dit_pretrained_path"] = None
    if variant == "failure_embedding":
        model_cfg["outcome_num_classes"] = 2

    return {
        "# @package _global_": None,
        "defaults": [
            {"override /data": data_name},
            {"override /model": "fastwam"},
            "_self_",
        ],
        "output_dir": "/data_all/zhaoyc/Summer2/dexjoco_fastwam_results_moved_from_share_20260703/${wandb.name}/${now:%Y-%m-%d_%H-%M-%S}",
        "batch_size": batch_size,
        "num_workers": 4,
        "model": model_cfg,
        "lr_scheduler_type": "cosine",
        "learning_rate": 0.0001,
        "num_epochs": 5,
        "max_steps": max_steps,
        "log_every": 10,
        "save_every": save_every,
        "eval_every": eval_every,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "weight_decay": 0.01,
        "resume": resume,
        "wandb": {
            "enabled": True,
            "workspace": None,
            "project": "fast-wam",
            "name": run_name,
            "group": f"dexjoco_{task}_failure_ablation",
            "mode": "online",
        },
    }


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)
    text = text.replace("'# @package _global_': null\n", "# @package _global_\n")
    path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    resume_map = read_resume_map(args.resume_map_json)
    data_dir = args.config_root / "data"
    task_dir = args.config_root / "task" / "dexjoco"

    for task in args.tasks:
        success_dir = args.success_root / task
        image_keys = image_keys_from_info(success_dir)
        for variant in ("success", "failure_embedding", "text_failure"):
            write_yaml(
                data_dir / f"dexjoco_{task}_2cam_{variant}.yaml",
                data_cfg(
                    task,
                    variant=variant,
                    image_keys=image_keys,
                    success_root=args.success_root,
                    failure_root=args.failure_root,
                    stats_dir=args.stats_dir,
                    text_cache_root=args.text_cache_root,
                ),
            )
            write_yaml(
                task_dir / f"dexjoco_{task}_{variant}_2cam_proprio_1e-4.yaml",
                task_cfg(
                    task,
                    variant=variant,
                    resume=resume_map.get(task),
                    max_steps=args.max_steps,
                    save_every=args.save_every,
                    eval_every=args.eval_every,
                    batch_size=args.batch_size,
                    gradient_accumulation_steps=args.gradient_accumulation_steps,
                ),
            )
        print(f"[create-configs] wrote configs for {task}: image_keys={image_keys}", flush=True)


if __name__ == "__main__":
    main()
