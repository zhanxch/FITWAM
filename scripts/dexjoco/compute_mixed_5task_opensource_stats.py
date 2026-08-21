#!/usr/bin/env python3
"""Compute OPEN-stack (224 / z-score) dataset_stats over the 5 mixed tasks.

Uses the same FastWAM get_dataset_stats path as scripts/dewo_v2/export_opensource_artifacts.py,
but concatenates the five official expert LeRobot datasets. Does not use the local
384 / uncond multi-task training recipe.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from hydra.utils import instantiate
from omegaconf import OmegaConf

ROOT = Path("/data_all/xiangchengzhan/FastWAM")
OPEN = Path("/data_all/xiangchengzhan/FastWAM-infer-in-DexJoco")
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from fastwam.datasets.lerobot.utils.normalizer import save_dataset_stats_to_json  # noqa: E402

TASKS = (
    "fold_glasses",
    "hammer_nail",
    "pick_bucket",
    "pinch_tongs",
    "water_plant",
)
EXPERT_ROOT = ROOT / "data" / "dexjoco" / "dexjoco_lerobot_datasets"
OUT = ROOT / "artifacts" / "mixed_5task" / "dataset_stats.json"


def _processor_cfg() -> OmegaConf:
    return OmegaConf.create(
        {
            "_target_": "fastwam.datasets.lerobot.processors.fastwam_processor.FastWAMProcessor",
            "shape_meta": {
                "images": [
                    {"key": "front", "raw_shape": [3, 640, 640], "shape": [3, 224, 224]},
                    {"key": "wrist", "raw_shape": [3, 640, 640], "shape": [3, 224, 224]},
                ],
                "action": [{"key": "default", "raw_shape": 22, "shape": 22}],
                "state": [{"key": "default", "raw_shape": 23, "shape": 23}],
            },
            "num_obs_steps": 33,
            "num_output_cameras": 2,
            "action_output_dim": 22,
            "proprio_output_dim": 23,
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


def _to_np(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


def _compare(mixed: dict) -> None:
    mixed_a = _to_np(mixed["action"]["default"]["global_mean"])
    mixed_s = _to_np(mixed["action"]["default"]["global_std"])
    print("[compare] mixed action global_mean[:6]", np.round(mixed_a[:6], 4).tolist())
    print("[compare] mixed action global_std[:6]", np.round(mixed_s[:6], 4).tolist())
    for task in TASKS:
        path = OPEN / "artifacts" / task / "dataset_stats.json"
        per = json.loads(path.read_text())
        a = np.asarray(per["action"]["default"]["global_mean"], dtype=np.float64)
        s = np.asarray(per["action"]["default"]["global_std"], dtype=np.float64)
        dmean = float(np.linalg.norm(mixed_a - a))
        dstd = float(np.linalg.norm(mixed_s - s))
        print(
            f"[compare] vs {task:13s}  ||Δmean||={dmean:.4f}  ||Δstd||={dstd:.4f}"
        )


def main() -> int:
    dirs = [str(EXPERT_ROOT / t) for t in TASKS]
    for d in dirs:
        if not Path(d).is_dir():
            raise FileNotFoundError(d)

    processor_cfg = _processor_cfg()
    ds_cfg = OmegaConf.create(
        {
            "_target_": "fastwam.datasets.lerobot.base_lerobot_dataset.BaseLerobotDataset",
            "dataset_dirs": dirs,
            "shape_meta": processor_cfg.shape_meta,
            "obs_size": 33,
            "action_size": 32,
            "val_set_proportion": 0.0,
            "is_training_set": True,
            "global_sample_stride": 1,
            "tolerance_s": 1e-4,
            "video_backend": "pyav",
        }
    )
    print("[mixed-stats] datasets:", dirs, flush=True)
    dataset = instantiate(ds_cfg)
    processor = instantiate(processor_cfg)
    processor.train()
    dataset.set_processor(processor)
    dataset._set_return_images(False)
    print(
        f"[mixed-stats] episodes={dataset.multi_dataset.num_episodes} "
        f"frames={dataset.multi_dataset.num_frames}",
        flush=True,
    )
    stats = dataset.get_dataset_stats(processor)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    save_dataset_stats_to_json(stats, str(OUT))
    print(f"[mixed-stats] wrote {OUT}", flush=True)
    _compare(stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
