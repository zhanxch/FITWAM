#!/usr/bin/env python3
"""Build FastWAM-infer-in-DexJoco artifacts (wrapped T5 + z-score stats).

Works for any registered DEWO v2 task:
  python scripts/dewo_v2/export_opensource_artifacts.py --task water_plant
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from omegaconf import OmegaConf
from hydra.utils import instantiate

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from dewo_v2.tasks import (  # noqa: E402
    DEFAULT_OPEN_REPO,
    get_task,
    resolve_expert,
    t5_cache_name,
)
from fastwam.datasets.lerobot.utils.normalizer import save_dataset_stats_to_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", default="water_plant")
    p.add_argument("--expert-dataset", type=Path, default=None)
    p.add_argument("--text-src", type=Path, default=None)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--force-stats", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    task = get_task(args.task)
    expert = (
        args.expert_dataset.expanduser().resolve()
        if args.expert_dataset is not None
        else resolve_expert(task)
    )
    t5_name = t5_cache_name(task.success_prompt)
    candidates = []
    if args.text_src is not None:
        candidates.append(args.text_src.expanduser().resolve())
    else:
        cache_root = ROOT / "data" / "text_embeds_cache"
        candidates.extend(
            [
                cache_root / task.name / t5_name,
                cache_root / f"{task.name}_dewo" / t5_name,
                cache_root / f"{task.name}_dewo_v2_pair" / t5_name,
                cache_root / f"{task.name}_dewo_v2_opensource" / t5_name,
            ]
        )
    src_t5 = next((p for p in candidates if p.is_file()), None)
    if src_t5 is None:
        raise FileNotFoundError(
            f"Missing wrapped-prompt T5 cache {t5_name}. Tried: "
            + ", ".join(str(p) for p in candidates)
        )
    out = (
        args.output.expanduser().resolve()
        if args.output is not None
        else (DEFAULT_OPEN_REPO / "artifacts" / task.name)
    )
    out.mkdir(parents=True, exist_ok=True)
    dst_t5 = out / t5_name
    if not dst_t5.exists() or dst_t5.stat().st_size != src_t5.stat().st_size:
        shutil.copy2(src_t5, dst_t5)
    print(f"[artifacts] task={task.name} t5={dst_t5}", flush=True)

    stats_path = out / "dataset_stats.json"
    if stats_path.is_file() and not args.force_stats:
        print(f"[artifacts] reuse stats={stats_path}", flush=True)
        print(stats_path)
        print(dst_t5)
        return 0

    h, w = task.image_raw
    processor_cfg = OmegaConf.create(
        {
            "_target_": "fastwam.datasets.lerobot.processors.fastwam_processor.FastWAMProcessor",
            "shape_meta": {
                "images": [
                    {"key": cam, "raw_shape": [3, h, w], "shape": [3, 224, 224]}
                    for cam in task.cameras
                ],
                "action": [{"key": "default", "raw_shape": task.action_dim, "shape": task.action_dim}],
                "state": [{"key": "default", "raw_shape": task.state_dim, "shape": task.state_dim}],
            },
            "num_obs_steps": 33,
            "num_output_cameras": len(task.cameras),
            "action_output_dim": task.action_dim,
            "proprio_output_dim": task.state_dim,
            "action_state_transforms": None,
            "use_stepwise_action_norm": False,
            "norm_default_mode": "z-score",
            "norm_exception_mode": None,
            "norm_stats_source": "compute",
            "norm_stats_meta_dir": None,
            "action_state_merger": {
                "_target_": "fastwam.datasets.lerobot.transforms.action_state_merger.ConcatLeftAlign"
            },
            "train_transforms": [
                {"_target_": "fastwam.datasets.lerobot.transforms.image.ToTensor"},
                {"_target_": "torchvision.transforms.Resize", "size": [224, 224]},
            ],
            "val_transforms": [
                {"_target_": "fastwam.datasets.lerobot.transforms.image.ToTensor"},
                {"_target_": "torchvision.transforms.Resize", "size": [224, 224]},
            ],
        }
    )
    ds_cfg = OmegaConf.create(
        {
            "_target_": "fastwam.datasets.lerobot.robot_video_dataset.RobotVideoDataset",
            "dataset_dirs": [str(expert)],
            "shape_meta": processor_cfg.shape_meta,
            "num_frames": 33,
            "global_sample_stride": 1,
            "action_video_freq_ratio": 4,
            "video_size": [224, 224 * len(task.cameras)],
            "camera_key": None,
            "val_set_proportion": 0.0,
            "is_training_set": True,
            "pretrained_norm_stats": None,
            "skip_padding_as_possible": False,
            "concat_multi_camera": "horizontal",
            "video_backend": "pyav",
            "processor": processor_cfg,
            "text_embedding_cache_dir": str(out),
            "context_len": 128,
        }
    )
    print(f"[artifacts] computing z-score stats from {expert}", flush=True)
    dataset = instantiate(ds_cfg)
    processor = dataset.lerobot_dataset.processor
    stats = dataset.lerobot_dataset.get_dataset_stats(processor)
    save_dataset_stats_to_json(stats, str(stats_path))
    print(f"[artifacts] stats={stats_path}", flush=True)
    print(stats_path)
    print(dst_t5)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
