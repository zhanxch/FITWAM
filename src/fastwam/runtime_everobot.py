"""EveRobot training runtime — parallel to the standard FastWAM training path.

Reuses ``Wan22Trainer`` and ``FastWAM.training_loss()`` unchanged; only the
dataset construction differs (``EveRobotDataset`` with episode-level sampling
and first-frame dropout instead of ``RobotVideoDataset`` with frame windows).
"""

import logging
from pathlib import Path

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from fastwam.runtime import (
    _mixed_precision_to_model_dtype,
    _normalize_mixed_precision,
    _resolve_train_device,
)
from fastwam.trainer import Wan22Trainer
from fastwam.utils.logging_config import get_logger, setup_logging

logger = get_logger(__name__)


def run_everobot_training(cfg: DictConfig):
    setup_logging(
        log_level=logging.INFO,
        is_main_process=torch.distributed.get_rank() == 0 if torch.distributed.is_initialized() else True,
    )

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "config.yaml", "w", encoding="utf-8") as f:
        OmegaConf.save(OmegaConf.to_container(cfg, resolve=True), f)

    model_device = _resolve_train_device()
    mixed_precision = _normalize_mixed_precision(cfg.mixed_precision)
    model_dtype = _mixed_precision_to_model_dtype(mixed_precision)
    model = instantiate(cfg.model, model_dtype=model_dtype, device=model_device)

    train_ds = instantiate(cfg.data.train)
    if cfg.data.get("val") is None:
        val_ds = train_ds
    else:
        val_ds = instantiate(cfg.data.val)
    logger.info(
        "EveRobot train dataset: %d samples, val dataset: %d samples",
        len(train_ds),
        len(val_ds),
    )

    trainer = Wan22Trainer(
        cfg=cfg,
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
    )
    trainer.train()
