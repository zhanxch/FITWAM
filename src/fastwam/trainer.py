from __future__ import annotations

import logging
import csv
import hashlib
import json
import os
import re
import shutil
from collections.abc import Sequence
from math import ceil
from pathlib import Path
import time
import uuid

import numpy as np
import torch
from accelerate import Accelerator
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torch.optim.lr_scheduler import ConstantLR, CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from .datasets.vae_latent_cache import collate_robot_video_batch
from .utils.fs import ensure_dir
from .utils.logging_config import get_logger, setup_logging
from .utils.pytorch_utils import set_global_seed
from .utils.samplers import ResumableEpochSampler
from .utils.video_io import save_mp4
from .utils.video_metrics import pil_frames_to_video_tensor, video_psnr, video_ssim

logger = get_logger(__name__)


class NonFiniteTrainingError(RuntimeError):
    """Raised when any rank observes a non-finite training value."""


class TrainingDataError(RuntimeError):
    """Raised for failures while loading or preparing a training batch."""


def _is_cuda_oom(error: BaseException) -> bool:
    out_of_memory_error = getattr(torch, "OutOfMemoryError", None)
    if out_of_memory_error is not None and isinstance(error, out_of_memory_error):
        return True
    message = str(error).lower()
    return (
        "cuda" in message
        and (
            "out of memory" in message
            or "cublas_status_alloc_failed" in message
        )
    )


def _classify_stop_reason(error: BaseException) -> str:
    if isinstance(error, NonFiniteTrainingError):
        return "nan"
    if isinstance(error, KeyboardInterrupt):
        return "manual"
    if _is_cuda_oom(error):
        return "oom"
    if isinstance(error, TrainingDataError):
        return "data_error"
    return "exception"


def _optional_interval(cfg: DictConfig, key: str, fallback: int) -> int:
    value = cfg.get(key, None)
    interval = int(fallback if value is None else value)
    if interval < 0:
        raise ValueError(f"`{key}` must be >= 0, got {interval}.")
    return interval


def _optional_positive_step_list(cfg: DictConfig, key: str) -> tuple[int, ...]:
    value = cfg.get(key, None)
    if value is None:
        return ()
    if getattr(OmegaConf, "is_config", lambda _: False)(value):
        value = OmegaConf.to_container(value, resolve=True)
    if not isinstance(value, list):
        raise ValueError(
            f"`{key}` must be null or a strictly increasing list of positive integers."
        )

    steps = []
    previous = 0
    for index, step in enumerate(value):
        if type(step) is not int or step <= 0:
            raise ValueError(
                f"`{key}[{index}]` must be a positive integer, got {step!r}."
            )
        if step <= previous:
            raise ValueError(
                f"`{key}` must be strictly increasing, got {value!r}."
            )
        steps.append(step)
        previous = step
    return tuple(steps)


def _should_save_weights_at_step(
    step: int,
    *,
    save_weights_every: int,
    save_weight_steps: tuple[int, ...],
    is_final_step: bool,
) -> bool:
    periodic = save_weights_every > 0 and step % save_weights_every == 0
    return periodic or step in save_weight_steps or is_final_step


def _deterministic_eval_indices(
    dataset_size: int,
    *,
    eval_seed: int,
    process_index: int,
    num_processes: int,
    num_samples: int,
) -> list[int]:
    if dataset_size <= 0:
        raise ValueError("Validation dataset must contain at least one sample.")
    if num_samples <= 0:
        raise ValueError(f"`eval_num_samples` must be >= 1, got {num_samples}.")

    if num_processes < 1:
        raise ValueError(f"`num_processes` must be >= 1, got {num_processes}.")
    if not 0 <= process_index < num_processes:
        raise ValueError(
            f"`process_index` must be in [0, {num_processes}), got {process_index}."
        )

    generator = torch.Generator(device="cpu").manual_seed(int(eval_seed))
    total_samples = num_samples * num_processes
    indices = []
    while len(indices) < total_samples:
        indices.extend(
            torch.randperm(dataset_size, generator=generator).tolist()
        )
    return indices[:total_samples][process_index::num_processes]


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    with temp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp_path, path)


def _sha256_json(payload: dict) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _resolved_experiment_provenance(cfg: DictConfig) -> dict | None:
    provenance = cfg.get("experiment_provenance", None)
    if provenance is None:
        return None
    payload = OmegaConf.to_container(provenance, resolve=True)
    if not isinstance(payload, dict):
        raise ValueError("`experiment_provenance` must resolve to a mapping.")
    return payload


def _canonical_output_dir(path: str | os.PathLike[str]) -> str:
    return str(Path(path).expanduser().resolve())


def _stable_output_dir_hash(path: str | os.PathLike[str]) -> int:
    digest = hashlib.sha256(
        _canonical_output_dir(path).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) & (
        (1 << 63) - 1
    )


def _send_batch_to_device(value, device):
    """Recursively move tensors while preserving strings and metadata."""

    if torch.is_tensor(value):
        return value.to(device=device, non_blocking=True)
    if isinstance(value, dict):
        return {
            key: _send_batch_to_device(item, device)
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return tuple(_send_batch_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_send_batch_to_device(item, device) for item in value]
    return value


def _infer_auxiliary_roles(
    roles: Sequence[str],
    *,
    primary_role: str = "primary",
) -> tuple[str, ...]:
    """Prefer pair-shaped aux roles, then keep any remaining non-primary roles."""

    present = {role for role in roles if role and role != primary_role}
    if not present:
        raise ValueError(
            "Role-balanced sampling requires at least one auxiliary role in "
            "`dataset.sampling_roles`."
        )
    ordered: list[str] = []
    for role in ("auxiliary_success", "auxiliary"):
        if role in present:
            ordered.append(role)
            present.remove(role)
    ordered.extend(sorted(present))
    return tuple(ordered)


class Wan22Trainer:
    def __init__(self, model, train_dataset, val_dataset=None, *, cfg: DictConfig):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.cfg = cfg
        self.output_dir = str(cfg.output_dir)
        self.learning_rate = float(cfg.learning_rate)
        self.weight_decay = float(cfg.weight_decay)
        self.batch_size = int(cfg.batch_size)
        self.num_workers = int(cfg.num_workers)
        self.num_epochs = int(cfg.num_epochs)
        max_steps = cfg.max_steps
        self.max_steps = int(max_steps) if max_steps is not None else None
        self.log_every = int(cfg.log_every)
        self.save_every = int(cfg.save_every)
        self.save_weights_every = _optional_interval(
            cfg, "save_weights_every", self.save_every
        )
        self.save_weight_steps = _optional_positive_step_list(
            cfg, "save_weight_steps"
        )
        self.save_state_every = _optional_interval(
            cfg, "save_state_every", self.save_every
        )
        state_keep_last = cfg.get("state_keep_last", None)
        self.state_keep_last = (
            None if state_keep_last is None else int(state_keep_last)
        )
        if self.state_keep_last is not None and self.state_keep_last < 1:
            raise ValueError(
                f"`state_keep_last` must be null or >= 1, got {self.state_keep_last}."
            )
        self.eval_every = int(cfg.eval_every)
        self.eval_num_inference_steps = int(cfg.eval_num_inference_steps)
        self.eval_num_samples = int(cfg.get("eval_num_samples", 1))
        if self.eval_num_samples < 1:
            raise ValueError(
                f"`eval_num_samples` must be >= 1, got {self.eval_num_samples}."
            )
        self.best_metric = str(cfg.get("best_metric", "val_loss"))
        self.best_metric_mode = str(cfg.get("best_metric_mode", "min")).lower()
        if self.best_metric_mode not in {"min", "max"}:
            raise ValueError(
                f"`best_metric_mode` must be 'min' or 'max', got {self.best_metric_mode}."
            )
        self.gradient_accumulation_steps = int(cfg.gradient_accumulation_steps)
        if self.gradient_accumulation_steps < 1:
            raise ValueError("`gradient_accumulation_steps` must be >= 1.")
        self.max_grad_norm = float(cfg.max_grad_norm)
        self.seed = int(cfg.seed)
        self.eval_seed = int(cfg.get("eval_seed", self.seed))
        self.experiment_provenance = _resolved_experiment_provenance(cfg)
        role_cfg = cfg.get("role_balanced_sampling", None)
        self.role_balanced_sampling_enabled = bool(
            role_cfg is not None and role_cfg.get("enabled", False)
        )
        self.role_balanced_primary_per_batch = (
            2
            if role_cfg is None
            else int(role_cfg.get("primary_per_batch", 2))
        )
        configured_role_seed = (
            None if role_cfg is None else role_cfg.get("seed", None)
        )
        self.role_balanced_seed = (
            self.seed
            if configured_role_seed is None
            else int(configured_role_seed)
        )
        if self.role_balanced_sampling_enabled:
            if self.batch_size < 2 or self.batch_size % 2 != 0:
                raise ValueError(
                    "Role-balanced sampling requires an even per-rank "
                    f"`batch_size>=2`, got {self.batch_size}."
                )
            expected_primary = self.batch_size // 2
            if self.role_balanced_primary_per_batch != expected_primary:
                raise ValueError(
                    "Role-balanced sampling requires "
                    f"`primary_per_batch={expected_primary}` for "
                    f"`batch_size={self.batch_size}` "
                    f"(got {self.role_balanced_primary_per_batch})."
                )
        
        self.resume = cfg.resume
        resume_experts = cfg.get("resume_experts", None)
        if resume_experts is None:
            self.resume_experts = None
        else:
            self.resume_experts = list(OmegaConf.to_container(resume_experts, resolve=True))
        self.mixed_precision = str(cfg.mixed_precision).strip().lower()
        if self.mixed_precision not in {"no", "fp16", "bf16"}:
            raise ValueError(
                f"Unsupported mixed_precision: {cfg.mixed_precision}. "
                "Expected one of: ['no', 'fp16', 'bf16']."
            )
        self.wandb_enabled = bool(cfg.wandb.enabled)

        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            mixed_precision=self.mixed_precision,
            step_scheduler_with_optimizer=False,
        )
        self._configure_role_balanced_deepspeed_batch_contract()
        self._assert_output_dir_consistent()
        
        logger.info(
            "Accelerate training: distributed_type=%s zero_stage=%s world_size=%d process_index=%d cfg_mixed_precision=%s accelerator_mixed_precision=%s grad_accum=%d grad_clip=%.4f",
            self.accelerator.distributed_type,
            self.accelerator.state.deepspeed_plugin.deepspeed_config.get("zero_optimization", {}).get("stage", "unknown"),
            self.accelerator.num_processes,
            self.accelerator.process_index,
            self.mixed_precision,
            self.accelerator.mixed_precision,
            self.gradient_accumulation_steps,
            self.max_grad_norm,
        )
        logger.info("using accelerator.device=%s", self.accelerator.device)
        worker_init_fn = set_global_seed(self.seed, get_worker_init_fn=True)
        self._assert_dataset_length_consistent(self.train_dataset, "train_dataset")
        if self.val_dataset is not None:
            self._assert_dataset_length_consistent(self.val_dataset, "val_dataset")

        # Freeze non-trainable modules before optimizer/deepspeed initialization.
        # This keeps DiT (+ optional proprio encoder) as trainable when ZeRO builds optimizer state.
        self._apply_dit_only_train_mode(self.model)
        trainable_params = [param for param in self.model.parameters() if param.requires_grad]
        if not trainable_params:
            raise ValueError("No trainable parameters found after applying training mode.")
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.95),
        )
        self.train_loader = self._build_loader(self.train_dataset, worker_init_fn=worker_init_fn)
        total_train_steps = self._estimate_total_train_steps()
        self.max_steps = total_train_steps
        configured_scheduler_steps = cfg.get("lr_scheduler_total_steps", None)
        scheduler_total_steps = (
            total_train_steps
            if configured_scheduler_steps is None
            else int(configured_scheduler_steps)
        )
        if scheduler_total_steps < total_train_steps:
            raise ValueError(
                "`lr_scheduler_total_steps` must be >= the run's max training "
                f"steps, got {scheduler_total_steps} < {total_train_steps}."
            )
        self.lr_scheduler_total_steps = scheduler_total_steps
        warmup_steps = int(scheduler_total_steps * 0.05)
        self.scheduler = self._build_scheduler(
            scheduler_type=cfg.lr_scheduler_type,
            total_train_steps=scheduler_total_steps,
            warmup_steps=warmup_steps,
        )
        self.global_step = 0
        self.epoch = 0
        self.batch_in_epoch = 0

        self.checkpoint_root = os.path.join(self.output_dir, "checkpoints")
        self.weights_dir = os.path.join(self.checkpoint_root, "weights")
        self.state_dir = os.path.join(self.checkpoint_root, "state")
        self.eval_dir = os.path.join(self.output_dir, "eval")

        ensure_dir(self.output_dir)
        ensure_dir(self.checkpoint_root)
        ensure_dir(self.weights_dir)
        ensure_dir(self.state_dir)
        ensure_dir(self.eval_dir)
        self.checkpoint_owner_id = self._load_or_create_checkpoint_owner()
        self.best_checkpoint_path = Path(self.output_dir) / "best_checkpoint.json"
        self.stop_reason_path = Path(self.output_dir) / "stop_reason.json"
        self.best_checkpoint = self._load_best_checkpoint_index()

        self._prepare_training_components()
        self.optimizer.zero_grad(set_to_none=True)
        self.wandb_run = None
        self._init_wandb()
        self._resume_or_load_checkpoint()

        val_size = len(self.val_dataset) if self.val_dataset is not None else len(self.train_dataset)
        logger.info("Train/val dataset size: %d/%d", len(self.train_dataset), val_size)

    def _init_wandb(self):
        if not self.wandb_enabled or not self.accelerator.is_main_process:
            return
        try:
            import wandb
        except ImportError as e:
            raise ImportError(
                "wandb logging is enabled in config (`wandb.enabled=true`) but wandb is not installed."
            ) from e

        wandb_config = OmegaConf.to_container(self.cfg, resolve=True)
        workspace = self.cfg.wandb.workspace
        configured_entity = None if workspace in (None, "null", "") else str(workspace)
        wandb_entity = os.getenv("WANDB_ENTITY") or configured_entity
        wandb_project = os.getenv("WANDB_PROJECT") or str(self.cfg.wandb.project)

        init_kwargs = {
            "project": wandb_project,
            "name": self.cfg.wandb.name,
            "group": None if self.cfg.wandb.group in (None, "null", "") else str(self.cfg.wandb.group),
            "mode": self.cfg.wandb.mode,
            "dir": self.output_dir,
            "config": wandb_config,
        }
        if wandb_entity is not None:
            init_kwargs["entity"] = wandb_entity

        self.wandb_run = wandb.init(**init_kwargs)

        logger.info(
            "Initialized wandb run: workspace=%s project=%s name=%s",
            wandb_entity or "(default)",
            wandb_project,
            self.cfg.wandb.name,
        )

    def _assert_output_dir_consistent(self) -> None:
        canonical_path = _canonical_output_dir(self.output_dir)
        local_hash = _stable_output_dir_hash(canonical_path)
        gathered = self.accelerator.gather(
            torch.tensor(
                [local_hash],
                device=self.accelerator.device,
                dtype=torch.int64,
            )
        ).reshape(-1)
        gathered_hashes = [
            int(value) for value in gathered.detach().cpu().tolist()
        ]
        if len(set(gathered_hashes)) == 1:
            return
        self.accelerator.wait_for_everyone()
        raise RuntimeError(
            "Inconsistent `output_dir` across ranks. Generate one RUN_ID before "
            "launch and pass the same resolved path to every process. "
            f"rank={self.accelerator.process_index} "
            f"local_output_dir={canonical_path!r} "
            f"gathered_path_hashes={gathered_hashes}"
        )

    def _wandb_log(self, payload: dict):
        if self.wandb_run is None:
            return
        self.wandb_run.log(payload, step=self.global_step)

    def _local_metrics_log(self, payload: dict) -> None:
        if not self.accelerator.is_main_process:
            return
        record = {"step": int(self.global_step), **payload}
        jsonl_path = Path(self.output_dir) / "metrics.jsonl"
        with jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")

        csv_path = Path(self.output_dir) / "metrics.csv"
        write_header = not csv_path.exists()
        with csv_path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["step", "metric", "value"])
            if write_header:
                writer.writeheader()
            for metric, value in sorted(payload.items()):
                writer.writerow(
                    {"step": int(self.global_step), "metric": metric, "value": value}
                )

    def _finish_wandb(self):
        if self.wandb_run is None:
            return
        self.wandb_run.finish()
        self.wandb_run = None

    def _load_or_create_checkpoint_owner(self) -> str:
        owner_path = Path(self.output_dir) / ".checkpoint_owner.json"
        if self.accelerator.is_main_process and not owner_path.exists():
            _atomic_write_json(
                owner_path,
                {
                    "owner_id": uuid.uuid4().hex,
                    "output_dir": str(Path(self.output_dir).resolve()),
                },
            )
        self.accelerator.wait_for_everyone()
        with owner_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        owner_id = str(payload.get("owner_id", "")).strip()
        if not owner_id:
            raise ValueError(f"Checkpoint owner file is invalid: {owner_path}")
        return owner_id

    def _load_best_checkpoint_index(self) -> dict | None:
        if not self.best_checkpoint_path.exists():
            return None
        try:
            with self.best_checkpoint_path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
        except (OSError, ValueError) as exc:
            logger.warning(
                "Ignoring unreadable best-checkpoint index %s: %s",
                self.best_checkpoint_path,
                exc,
            )
            return None
        if (
            payload.get("metric") != self.best_metric
            or payload.get("mode") != self.best_metric_mode
        ):
            logger.info(
                "Best-checkpoint metric changed from %s/%s to %s/%s; starting a new index.",
                payload.get("metric"),
                payload.get("mode"),
                self.best_metric,
                self.best_metric_mode,
            )
            return None
        return payload

    def _is_better_metric(self, value: float) -> bool:
        if not np.isfinite(value):
            return False
        if self.best_checkpoint is None:
            return True
        best_value = float(self.best_checkpoint["value"])
        if self.best_metric_mode == "min":
            return value < best_value
        return value > best_value

    def _update_best_checkpoint_index(self, metrics: dict) -> None:
        if not self.accelerator.is_main_process:
            return
        if self.best_metric not in metrics:
            logger.warning(
                "Best metric `%s` is missing from evaluation metrics; index not updated.",
                self.best_metric,
            )
            return
        value = float(metrics[self.best_metric])
        if not self._is_better_metric(value):
            return

        step_weights_path = Path(self.weights_dir) / f"step_{self.global_step:06d}.pt"
        if step_weights_path.is_file():
            weights_path = step_weights_path
        else:
            weights_path = Path(self.weights_dir) / "best.pt"
            temp_path = weights_path.with_name(
                f".{weights_path.name}.incomplete-{self.checkpoint_owner_id}"
            )
            if temp_path.exists():
                temp_path.unlink()
            model = self.accelerator.unwrap_model(self.model)
            model.save_checkpoint(
                str(temp_path),
                optimizer=None,
                step=self.global_step,
            )
            os.replace(temp_path, weights_path)
        payload = {
            "metric": self.best_metric,
            "mode": self.best_metric_mode,
            "value": value,
            "step": int(self.global_step),
            "weights_path": str(weights_path),
            "weights_available": True,
        }
        _atomic_write_json(self.best_checkpoint_path, payload)
        self.best_checkpoint = payload
        logger.info(
            "[best] metric=%s value=%.6f step=%d weights_available=%s",
            self.best_metric,
            value,
            self.global_step,
            payload["weights_available"],
        )

    def _refresh_best_checkpoint_availability(self) -> None:
        if not self.accelerator.is_main_process or self.best_checkpoint is None:
            return
        if int(self.best_checkpoint.get("step", -1)) != self.global_step:
            return
        weights_path = Path(self.best_checkpoint["weights_path"])
        available = weights_path.is_file()
        if bool(self.best_checkpoint.get("weights_available")) == available:
            return
        payload = {**self.best_checkpoint, "weights_available": available}
        _atomic_write_json(self.best_checkpoint_path, payload)
        self.best_checkpoint = payload

    def _write_stop_reason(
        self,
        *,
        status: str,
        reason: str,
        error: BaseException | None = None,
    ) -> None:
        if not self.accelerator.is_main_process:
            return
        payload = {
            "status": str(status),
            "reason": str(reason),
            "global_step": int(self.global_step),
            "max_steps": int(self.max_steps) if self.max_steps is not None else None,
            "epoch": int(self.epoch),
            "batch_in_epoch": int(self.batch_in_epoch),
        }
        if error is not None:
            payload["error_type"] = type(error).__name__
            payload["error_message"] = str(error)
        _atomic_write_json(self.stop_reason_path, payload)

    def _raise_if_non_finite(self, value, *, value_name: str, device) -> None:
        value_tensor = torch.as_tensor(value, device=device)
        local_non_finite = (~torch.isfinite(value_tensor.detach()).all()).to(
            dtype=torch.int64
        )
        any_non_finite = self.accelerator.reduce(
            local_non_finite,
            reduction="max",
        )
        if not bool(any_non_finite.item()):
            return
        self.optimizer.zero_grad(set_to_none=True)
        raise NonFiniteTrainingError(
            f"Non-finite {value_name} detected on at least one training rank "
            f"at global_step={self.global_step}."
        )

    def _next_train_sample(self, data_iter):
        try:
            sample = next(data_iter)
            self.batch_in_epoch += 1
            return self._prepare_train_sample(sample)
        except StopIteration:
            raise
        except Exception as exc:
            if _is_cuda_oom(exc):
                raise
            raise TrainingDataError(
                "Training data/input pipeline failed while loading or "
                f"preparing batch_in_epoch={self.batch_in_epoch}."
            ) from exc

    def _build_loader(self, dataset, worker_init_fn=None):
        if getattr(self, "role_balanced_sampling_enabled", False):
            from .utils.role_balanced_sampler import RoleBalancedBatchSampler

            roles = getattr(dataset, "sampling_roles", None)
            if roles is None:
                raise TypeError(
                    "Role-balanced sampling requires a dataset with a stable "
                    "`sampling_roles` sequence."
                )
            if len(roles) != len(dataset):
                raise ValueError(
                    "`dataset.sampling_roles` must align one-to-one with dataset "
                    f"windows: roles={len(roles)} dataset={len(dataset)}."
                )
            self.train_sampler = RoleBalancedBatchSampler(
                roles=roles,
                batch_size=self.batch_size,
                primary_per_batch=self.role_balanced_primary_per_batch,
                num_replicas=self.accelerator.num_processes,
                rank=self.accelerator.process_index,
                seed=self.role_balanced_seed,
                primary_role="primary",
                auxiliary_roles=_infer_auxiliary_roles(roles, primary_role="primary"),
            )
            return DataLoader(
                dataset,
                batch_sampler=self.train_sampler,
                num_workers=self.num_workers,
                pin_memory=torch.cuda.is_available(),
                worker_init_fn=worker_init_fn,
                collate_fn=collate_robot_video_batch,
            )

        self.train_sampler = ResumableEpochSampler(
            dataset=dataset,
            seed=self.seed,
            batch_size=self.batch_size,
            num_processes=self.accelerator.num_processes,
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            sampler=self.train_sampler,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=worker_init_fn,
            collate_fn=collate_robot_video_batch,
        )

    def _prepare_training_components(self) -> None:
        if getattr(self, "role_balanced_sampling_enabled", False):
            # RoleBalancedBatchSampler already partitions indices by rank.
            # Passing this DataLoader to Accelerate.prepare would wrap its
            # batch_sampler in BatchSamplerShard and shard it a second time.
            # Keep the local loader untouched and move its batches explicitly
            # in `_prepare_train_sample`.
            self.model, self.optimizer, self.scheduler = self.accelerator.prepare(
                self.model,
                self.optimizer,
                self.scheduler,
            )
            return

        (
            self.model,
            self.optimizer,
            self.train_loader,
            self.scheduler,
        ) = self.accelerator.prepare(
            self.model,
            self.optimizer,
            self.train_loader,
            self.scheduler,
        )

    def _configure_role_balanced_deepspeed_batch_contract(self) -> None:
        if not getattr(self, "role_balanced_sampling_enabled", False):
            return
        state = getattr(self.accelerator, "state", None)
        plugin = getattr(state, "deepspeed_plugin", None)
        if plugin is None:
            return

        world_size = int(self.accelerator.num_processes)
        expected = {
            "train_micro_batch_size_per_gpu": int(self.batch_size),
            "gradient_accumulation_steps": int(self.gradient_accumulation_steps),
            "train_batch_size": int(
                self.batch_size
                * self.gradient_accumulation_steps
                * world_size
            ),
        }
        deepspeed_config = plugin.deepspeed_config
        for key, value in expected.items():
            configured = deepspeed_config.get(key, "auto")
            if configured not in {"auto", value}:
                raise ValueError(
                    "DeepSpeed batch contract disagrees with the role-balanced "
                    f"loader: {key}={configured!r}, expected {value}."
                )
            deepspeed_config[key] = value

    def _prepare_train_sample(self, sample):
        if not getattr(self, "role_balanced_sampling_enabled", False):
            return sample
        return _send_batch_to_device(sample, self.accelerator.device)

    def _assert_dataset_length_consistent(self, dataset, dataset_name: str):
        if not hasattr(dataset, "__len__"):
            raise TypeError(f"`{dataset_name}` must implement __len__ for rank consistency checks.")

        local_length = len(dataset)
        gathered_lengths = self.accelerator.gather(
            torch.tensor([local_length], device=self.accelerator.device, dtype=torch.int64)
        ).reshape(-1)
        if torch.all(gathered_lengths == gathered_lengths[0]):
            return

        if self.accelerator.is_main_process:
            print(f"[dataset-check] {dataset_name} length mismatch across ranks after initialization:")
            for rank, rank_length in enumerate(gathered_lengths.cpu().tolist()):
                print(f"rank {rank}: {rank_length}")
        self.accelerator.wait_for_everyone()
        raise RuntimeError(
            f"{dataset_name} length mismatch across ranks: {gathered_lengths.cpu().tolist()}"
        )

    def _estimate_total_train_steps(self) -> int:
        if self.max_steps is not None:
            return max(int(self.max_steps), 1)

        if not hasattr(self.train_dataset, "__len__"):
            raise TypeError("`train_dataset` must implement __len__ when `max_steps` is None.")

        if getattr(self, "role_balanced_sampling_enabled", False):
            micro_steps_per_epoch = max(
                int(self.train_sampler.num_batches_per_epoch),
                1,
            )
        else:
            num_processes = max(int(self.accelerator.num_processes), 1)
            global_batch_size = max(self.batch_size * num_processes, 1)
            micro_steps_per_epoch = max(
                ceil(len(self.train_dataset) / global_batch_size),
                1,
            )
        opt_steps_per_epoch = max(
            ceil(micro_steps_per_epoch / self.gradient_accumulation_steps),
            1,
        )
        return max(opt_steps_per_epoch * self.num_epochs, 1)

    def _build_scheduler(self, scheduler_type, total_train_steps: int, warmup_steps: int = 0):
        scheduler_type = str(scheduler_type).strip().lower()
        total_train_steps = max(int(total_train_steps), 1)
        warmup_steps = min(max(int(warmup_steps), 0), total_train_steps - 1)

        remaining_steps = max(total_train_steps - warmup_steps, 1)
        if scheduler_type == "cosine":
            main_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=remaining_steps,
                eta_min=self.learning_rate * 0.01,
            )
        elif scheduler_type == "constant":
            main_scheduler = ConstantLR(self.optimizer, factor=1.0, total_iters=remaining_steps)
        else:
            raise ValueError(
                f"Unsupported lr_scheduler_type: {scheduler_type}. "
                "Expected one of: ['cosine', 'constant']."
            )

        if warmup_steps <= 0:
            return main_scheduler

        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=1.0 / warmup_steps,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        return SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_steps],
        )
    
    def _estimate_eta(self):
        elapsed = max(time.perf_counter() - self.run_start_time, 1e-6)
        done_steps = max(self.global_step - self.run_start_step, 1)
        steps_per_sec = done_steps / elapsed
        remaining_steps = max(self.max_steps - self.global_step, 0)
        eta_seconds = int(remaining_steps / max(steps_per_sec, 1e-9))
        eta_h, eta_rem = divmod(eta_seconds, 3600)
        eta_m, eta_s = divmod(eta_rem, 60)
        return f"{eta_h:02d}:{eta_m:02d}:{eta_s:02d}", steps_per_sec

    def _resume_or_load_checkpoint(self):
        resume = self.resume
        if not resume:
            return
        resume_path = Path(str(resume))
        if resume_path.is_dir():
            logger.info("Resuming full training state from directory: %s", resume)
            self.load_training_state(str(resume_path))
            return
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume}")
        logger.info("Loading weight checkpoint only: %s", resume)
        self.accelerator.unwrap_model(self.model).load_checkpoint(
            str(resume_path),
            optimizer=None,
            experts=self.resume_experts,
        )
        logger.warning("Loaded .pt weights only; optimizer/scheduler/step were not restored under ZeRO2.")

    def _set_dit_only_train_mode(self):
        # Match DiffSynth's freeze_except("dit"): only DiT stays trainable/in-train-mode.
        logger.info("Setting DiT to train mode and freezing other model components.")
        model = self.accelerator.unwrap_model(self.model)
        self._apply_dit_only_train_mode(model)

    @staticmethod
    def _apply_dit_only_train_mode(model):
        if getattr(model, "video_lora_enabled", False):
            from fastwam.models.wan22.video_lora import apply_video_lora_training_mode

            apply_video_lora_training_mode(model)
            return

        model.eval()
        model.requires_grad_(False)
        model.dit.train()
        model.dit.requires_grad_(True)
        proprio_encoder = getattr(model, "proprio_encoder", None)
        if proprio_encoder is not None:
            proprio_encoder.train()
            proprio_encoder.requires_grad_(True)
        outcome_encoder = getattr(model, "outcome_encoder", None)
        if outcome_encoder is not None:
            outcome_encoder.train()
            outcome_encoder.requires_grad_(True)

    @staticmethod
    def _to_batched_eval_sample(sample):
        video = sample.get("video", None)
        prompt = sample["prompt"]
        action = sample.get("action", None)
        proprio = sample.get("proprio", None)
        context = sample.get("context", None)
        context_mask = sample.get("context_mask", None)
        outcome_flag = sample.get("outcome_flag", None)
        input_latents = sample.get("input_latents", None)

        if input_latents is not None:
            if not isinstance(input_latents, torch.Tensor):
                raise TypeError(
                    f"Expected tensor input_latents for evaluation, got {type(input_latents)}"
                )
            if input_latents.ndim == 4:
                input_latents = input_latents.unsqueeze(0)
            if input_latents.ndim != 5:
                raise ValueError(
                    f"Expected input_latents shape [C,T,H,W] or [B,C,T,H,W], got {tuple(input_latents.shape)}"
                )
            latent_t = int(input_latents.shape[2])
            num_video_frames = 1 + (latent_t - 1) * 4
            batch_size = int(input_latents.shape[0])
        else:
            if not isinstance(video, torch.Tensor):
                raise TypeError(
                    f"Expected tensor video for evaluation, got {type(video)}. "
                    "Evaluation now expects `video` with shape [3,T,H,W] or [B,3,T,H,W]."
                )
            if video.ndim == 4:
                video = video.unsqueeze(0)
            if video.ndim != 5:
                raise ValueError(f"Expected video shape [3,T,H,W] or [B,3,T,H,W], got {tuple(video.shape)}")
            num_video_frames = video.shape[2]
            batch_size = int(video.shape[0])

        if video is not None:
            if not isinstance(video, torch.Tensor):
                raise TypeError(
                    f"Expected tensor video for evaluation, got {type(video)}."
                )
            if video.ndim == 4:
                video = video.unsqueeze(0)
            if video.ndim != 5:
                raise ValueError(f"Expected video shape [3,T,H,W] or [B,3,T,H,W], got {tuple(video.shape)}")

        if num_video_frames <= 1:
            raise ValueError(
                f"Eval sample must have at least 2 video frames, got T={num_video_frames}"
            )

        if isinstance(prompt, str):
            prompt = [prompt]
        elif isinstance(prompt, tuple):
            prompt = list(prompt)
        elif not isinstance(prompt, list):
            raise TypeError(f"Expected prompt type str/list[str], got {type(prompt)}")
        if len(prompt) != batch_size:
            raise ValueError(f"Prompt batch mismatch: len(prompt)={len(prompt)} vs batch={batch_size}")
        
        action_horizon = None
        action = None
        if "action" in sample:
            action = sample["action"]
            if not isinstance(action, torch.Tensor):
                raise TypeError(
                    f"`sample['action']` must be a torch.Tensor, got {type(action)}"
                )
            if action.ndim == 2:
                action = action.unsqueeze(0)
            if action.ndim != 3:
                raise ValueError(f"`sample['action']` must be 3D [B, T, a_dim], got shape {tuple(action.shape)}")
            if action.shape[1] % (num_video_frames - 1) != 0:
                raise ValueError(f"`sample['action']` temporal dimension must be divisible by video frames-1={num_video_frames - 1}, got {action.shape[1]}")
            action_horizon = int(action.shape[1])

        proprio = None
        if "proprio" in sample:
            proprio = sample["proprio"]
            if not isinstance(proprio, torch.Tensor):
                raise TypeError(f"`sample['proprio']` must be a torch.Tensor, got {type(proprio)}")
            if proprio.ndim == 2:
                proprio = proprio.unsqueeze(0)
            if proprio.ndim != 3:
                raise ValueError(f"`sample['proprio']` must be 3D [B, T, d], got shape {tuple(proprio.shape)}")

        if context is not None or context_mask is not None:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must both exist in eval sample.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )

        if outcome_flag is not None:
            if not isinstance(outcome_flag, torch.Tensor):
                outcome_flag = torch.as_tensor(outcome_flag, dtype=torch.long)
            if outcome_flag.ndim == 0:
                outcome_flag = outcome_flag.view(1)
            elif outcome_flag.ndim == 2 and outcome_flag.shape[1] == 1:
                outcome_flag = outcome_flag.view(-1)
            elif outcome_flag.ndim != 1:
                raise ValueError(f"`sample['outcome_flag']` must be scalar, [B], or [B,1], got {tuple(outcome_flag.shape)}")
            if outcome_flag.shape[0] != batch_size:
                raise ValueError(f"Outcome batch mismatch: {outcome_flag.shape[0]} vs batch={batch_size}")

        out = {
            "video": video,
            "prompt": prompt,
            "action": action,
            "proprio": proprio,
            "context": context,
            "context_mask": context_mask,
            "outcome_flag": outcome_flag,
            "action_horizon": action_horizon,
        }
        if input_latents is not None:
            out["input_latents"] = input_latents
        return out

    @torch.no_grad()
    def _evaluate_one(self, model, *, eval_index: int, sample_number: int):
        eval_device = torch.device(self.accelerator.device)
        rng_devices = []
        if eval_device.type == "cuda":
            rng_devices = [
                eval_device.index
                if eval_device.index is not None
                else torch.cuda.current_device()
            ]
        sample_seed = (
            self.eval_seed
            + self.accelerator.process_index * 1_000_003
            + int(eval_index)
        )
        with torch.random.fork_rng(devices=rng_devices):
            torch.manual_seed(sample_seed)
            if rng_devices:
                torch.cuda.manual_seed_all(sample_seed)
            raw_sample = self.val_dataset[eval_index]
            sample_id = str(
                raw_sample.get(
                    "eve_sample_id",
                    f"dataset-index-{eval_index:08d}",
                )
            )
            sample = self._to_batched_eval_sample(raw_sample)
            with self.accelerator.autocast():
                val_loss, val_loss_dict = model.training_loss(sample)
                val_loss = val_loss.float().item()

        prompt = sample["prompt"][0]
        video0 = sample.get("video")
        if video0 is not None:
            video0 = video0[0]
        has_real_video = (
            video0 is not None
            and torch.is_tensor(video0)
            and video0.ndim == 4
            and int(video0.shape[-1]) >= 64
            and int(video0.shape[-2]) >= 64
        )
        action = sample["action"][0] if "action" in sample and sample["action"] is not None else None
        proprio = sample["proprio"][0, 0] if "proprio" in sample and sample["proprio"] is not None else None

        # Latent-only / no-VAE mode: keep selection metric (val_base_loss) but skip
        # pixel-space rollout / VAE recon visualization.
        if (not has_real_video) or getattr(model, "vae", None) is None:
            return {
                "metrics": [
                    float(val_loss),
                    float("nan"),
                    float("nan"),
                    float("nan"),
                    float("nan"),
                    float("nan"),
                    float("nan"),
                    float("nan"),
                    float("nan"),
                ],
                "video_path": None,
                "selection": {
                    "eval_index": int(eval_index),
                    "sample_id": sample_id,
                    "noise_seed": int(sample_seed),
                },
            }

        input_image = video0[:, 0].unsqueeze(0)
        _, num_frames, _, _ = video0.shape

        infer_kwargs = {
            "input_image": input_image,
            "num_frames": num_frames,
            "action": action,
            "action_horizon": sample['action_horizon'],
            "proprio": proprio,
            "outcome_flag": sample.get("outcome_flag", None),
            "text_cfg_scale": 1.0,
            "action_cfg_scale": 1.0,
            "num_inference_steps": self.eval_num_inference_steps,
            "seed": sample_seed,
            "tiled": False,
        }
        if sample["context"] is not None:
            infer_kwargs["prompt"] = None
            infer_kwargs["context"] = sample["context"][0]
            infer_kwargs["context_mask"] = sample["context_mask"][0]
        else:
            infer_kwargs["prompt"] = prompt

        pred = model.infer(
            **infer_kwargs,
        )

        pred_video = pred["video"]
        pred_action = pred.get("action", None)

        pred_video_tensor = pil_frames_to_video_tensor(pred_video)
        gt_video_tensor = ((video0.detach().float().cpu().clamp(-1.0, 1.0) + 1.0) * 0.5).contiguous()

        assert pred_video_tensor.shape == gt_video_tensor.shape, (
            "Eval infer prediction/GT shape mismatch: "
            f"pred={tuple(pred_video_tensor.shape)} vs gt={tuple(gt_video_tensor.shape)}"
        )

        psnr_rollout_vs_gt = video_psnr(pred=pred_video_tensor, target=gt_video_tensor)
        ssim_rollout_vs_gt = video_ssim(pred=pred_video_tensor, target=gt_video_tensor)

        action_l1 = None
        action_l2 = None
        if action is not None and pred_action is not None:
            if sample["proprio"] is None:
                raise ValueError("Eval sample must contain `proprio` for action denormalization.")
            proprio = sample["proprio"].detach().to(device="cpu", dtype=torch.float32)
            
            processor = self.val_dataset.lerobot_dataset.processor

            denorm_actions = {}
            action_meta = processor.shape_meta["action"]
            state_meta = processor.shape_meta["state"]
            for action_name, raw_action in (("pred", pred_action), ("gt", action)):
                if not isinstance(raw_action, torch.Tensor):
                    raise TypeError(f"{action_name} action must be a torch.Tensor, got {type(raw_action)}")
                if raw_action.ndim == 2:
                    action_btd = raw_action.unsqueeze(0)
                elif raw_action.ndim == 3 and raw_action.shape[0] == 1:
                    action_btd = raw_action
                else:
                    raise ValueError(
                        f"{action_name} action must have shape [T, D] or [1, T, D], got {tuple(raw_action.shape)}"
                    )
                action_btd = action_btd.detach().to(device="cpu", dtype=torch.float32)

                batch = {
                    "action": action_btd,
                    "state": proprio,
                }
                batch = processor.action_state_merger.backward(batch)
                batch = processor.normalizer.backward(batch)
                merged_batch = {
                    "action": {meta["key"]: batch["action"][meta["key"]].squeeze(0) for meta in action_meta},
                    "state": {meta["key"]: batch["state"][meta["key"]].squeeze(0) for meta in state_meta},
                }
                merged_batch = processor.action_state_merger.forward(merged_batch)
                denorm_action = merged_batch["action"].unsqueeze(0)
                if denorm_action.ndim != 3 or denorm_action.shape[0] != 1:
                    raise ValueError(
                        f"Denormalized {action_name} action must have shape [1, T, D], got {tuple(denorm_action.shape)}"
                    )
                denorm_actions[action_name] = denorm_action

            pred_action_denorm = denorm_actions["pred"]
            gt_action_denorm = denorm_actions["gt"]

            if pred_action_denorm.shape != gt_action_denorm.shape:
                raise ValueError(
                    "Predicted action/GT action shape mismatch after denormalization: "
                    f"pred={tuple(pred_action_denorm.shape)} vs gt={tuple(gt_action_denorm.shape)}"
                )
            action_diff = pred_action_denorm - gt_action_denorm
            action_l1 = action_diff.abs().mean().item()
            action_l2 = action_diff.pow(2).mean().item()

        gt_video_batch = video0.unsqueeze(0).to(device=model.device, dtype=model.torch_dtype)
        vae_latents = model._encode_video_latents(gt_video_batch, tiled=False)
        vae_recon_video = model._decode_latents(vae_latents, tiled=False)
        vae_video_tensor = pil_frames_to_video_tensor(vae_recon_video)

        assert vae_video_tensor.shape == gt_video_tensor.shape, (
            "Eval VAE reconstruction/GT shape mismatch: "
            f"vae={tuple(vae_video_tensor.shape)} vs gt={tuple(gt_video_tensor.shape)}"
        )

        psnr_decode_vs_gt = video_psnr(pred=vae_video_tensor, target=gt_video_tensor)
        ssim_decode_vs_gt = video_ssim(pred=vae_video_tensor, target=gt_video_tensor)

        psnr_rollout_vs_decode = video_psnr(pred=pred_video_tensor, target=vae_video_tensor)
        ssim_rollout_vs_decode = video_ssim(pred=pred_video_tensor, target=vae_video_tensor)

        stitched_video_tensor = torch.cat(
            [pred_video_tensor, vae_video_tensor, gt_video_tensor],
            dim=2,
        ).contiguous()
        stitched_frames = []
        for t in range(stitched_video_tensor.shape[1]):
            frame = (stitched_video_tensor[:, t].permute(1, 2, 0).clamp(0.0, 1.0).numpy() * 255.0).astype(np.uint8)
            stitched_frames.append(Image.fromarray(frame))

        sample_suffix = ""
        if self.eval_num_samples > 1:
            sample_suffix = f"_sample_{sample_number:03d}_index_{eval_index:06d}"
        video_path = os.path.join(
            self.eval_dir,
            f"step_{self.global_step:06d}_rank_{self.accelerator.process_index:03d}{sample_suffix}.mp4",
        )
        save_mp4(stitched_frames, video_path, fps=8)

        return {
            "metrics": [
                float(val_loss),
                float(psnr_rollout_vs_gt),
                float(ssim_rollout_vs_gt),
                float(psnr_rollout_vs_decode),
                float(ssim_rollout_vs_decode),
                float(psnr_decode_vs_gt),
                float(ssim_decode_vs_gt),
                float(action_l2) if action_l2 is not None else float("nan"),
                float(action_l1) if action_l1 is not None else float("nan"),
            ],
            "video_path": video_path,
            "selection": {
                "eval_index": int(eval_index),
                "sample_id": sample_id,
                "noise_seed": int(sample_seed),
            },
        }

    @torch.no_grad()
    def evaluate(self):
        if self.val_dataset is None:
            return None

        model = self.accelerator.unwrap_model(self.model)
        was_dit_training = model.dit.training
        model.eval()
        eval_indices = _deterministic_eval_indices(
            len(self.val_dataset),
            eval_seed=self.eval_seed,
            process_index=self.accelerator.process_index,
            num_processes=self.accelerator.num_processes,
            num_samples=self.eval_num_samples,
        )

        try:
            local_rows = []
            video_paths = []
            selection_rows = []
            for sample_number, eval_index in enumerate(eval_indices):
                sample_result = self._evaluate_one(
                    model,
                    eval_index=eval_index,
                    sample_number=sample_number,
                )
                local_rows.append(sample_result["metrics"])
                video_paths.append(sample_result["video_path"])
                selection_rows.append(sample_result["selection"])

            selection_payload = {
                "eval_seed": self.eval_seed,
                "process_index": self.accelerator.process_index,
                "num_processes": self.accelerator.num_processes,
                "samples": selection_rows,
            }
            selection_path = (
                Path(self.output_dir)
                / f"eval_selection_rank_{self.accelerator.process_index:03d}.json"
            )
            if selection_path.exists():
                with selection_path.open("r", encoding="utf-8") as stream:
                    existing_selection = json.load(stream)
                if existing_selection != selection_payload:
                    raise RuntimeError(
                        "Frozen validation selection changed within one run: "
                        f"{selection_path}"
                    )
            else:
                _atomic_write_json(selection_path, selection_payload)

            local_metrics = torch.tensor(
                local_rows,
                device=self.accelerator.device,
                dtype=torch.float32,
            )
            gathered_metrics = self.accelerator.gather_for_metrics(local_metrics)
            mean_metrics = gathered_metrics[:, :7].mean(dim=0)

            action_l2_values = gathered_metrics[:, 7]
            action_l1_values = gathered_metrics[:, 8]
            action_l2_valid = action_l2_values[torch.isfinite(action_l2_values)]
            action_l1_valid = action_l1_values[torch.isfinite(action_l1_values)]

            result = {
                "val_loss": float(mean_metrics[0].item()),
                "val_base_loss": float(mean_metrics[0].item()),
                "psnr_rg": float(mean_metrics[1].item()),
                "ssim_rg": float(mean_metrics[2].item()),
                "psnr_rd": float(mean_metrics[3].item()),
                "ssim_rd": float(mean_metrics[4].item()),
                "psnr_dg": float(mean_metrics[5].item()),
                "ssim_dg": float(mean_metrics[6].item()),
                "video_path": video_paths[0],
                "video_paths": video_paths,
                "num_samples_per_process": len(eval_indices),
                "num_samples_total": int(gathered_metrics.shape[0]),
            }
            if action_l2_valid.numel() > 0:
                result["action_l2"] = float(action_l2_valid.mean().item())
            if action_l1_valid.numel() > 0:
                result["action_l1"] = float(action_l1_valid.mean().item())
            return result
        finally:
            if was_dit_training:
                self._set_dit_only_train_mode()

    def _save_weights_checkpoint(self, step_tag: str):
        model = self.accelerator.unwrap_model(self.model)
        ckpt_path = Path(self.weights_dir) / f"{step_tag}.pt"
        if ckpt_path.is_file():
            return str(ckpt_path)

        temp_path = ckpt_path.with_name(
            f".{ckpt_path.name}.incomplete-{self.checkpoint_owner_id}"
        )
        if temp_path.exists():
            temp_path.unlink()
        model.save_checkpoint(str(temp_path), optimizer=None, step=self.global_step)
        os.replace(temp_path, ckpt_path)
        return str(ckpt_path)

    def _save_trainer_state(self, state_path: str):
        payload = {
            "global_step": int(self.global_step),
            "epoch": int(self.epoch),
            "batch_in_epoch": int(self.batch_in_epoch),
        }
        if getattr(self, "role_balanced_sampling_enabled", False):
            payload["train_sampler_state"] = {
                "epoch": int(self.epoch),
                "batch_offset": int(self.batch_in_epoch),
            }
        _atomic_write_json(Path(state_path) / "trainer_state.json", payload)

    def _checkpoint_provenance_payload(self) -> dict:
        provenance = getattr(self, "experiment_provenance", None)
        if provenance is None:
            return {}
        return {
            "experiment_provenance": provenance,
            "experiment_provenance_sha256": _sha256_json(provenance),
        }

    def _validate_state_provenance(self, state_path: Path) -> None:
        expected = getattr(self, "experiment_provenance", None)
        if expected is None:
            return
        meta = self._read_checkpoint_meta(state_path)
        if meta is None:
            raise ValueError(
                f"Resume state lacks readable checkpoint_meta.json: {state_path}"
            )
        actual = meta.get("experiment_provenance")
        actual_hash = meta.get("experiment_provenance_sha256")
        if not isinstance(actual, dict) or not isinstance(actual_hash, str):
            raise ValueError(
                "Resume state is not bound to an experiment provenance contract."
            )
        if actual_hash != _sha256_json(actual):
            raise ValueError("Resume state experiment provenance hash is invalid.")
        if actual != expected:
            raise ValueError(
                "Resume state experiment provenance does not match the current "
                "run; refusing cross-variant or cross-dataset resume."
            )

    def _restore_train_sampler_progress(
        self,
        *,
        epoch: int,
        batch_in_epoch: int,
        sampler_state: dict | None = None,
    ) -> None:
        if getattr(self, "role_balanced_sampling_enabled", False):
            if sampler_state is None:
                sampler_state = {
                    "epoch": int(epoch),
                    "batch_offset": int(batch_in_epoch),
                }
            self.train_sampler.load_state_dict(sampler_state)
            return
        self.train_sampler.set_epoch_offset(epoch)
        self.train_sampler.set_resume_batch_offset(batch_in_epoch)

    @staticmethod
    def _read_checkpoint_meta(state_path: Path) -> dict | None:
        meta_path = state_path / "checkpoint_meta.json"
        if not meta_path.is_file():
            return None
        try:
            with meta_path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError):
            return None

    def _state_is_completed_and_owned(self, state_path: Path) -> bool:
        meta = self._read_checkpoint_meta(state_path)
        return bool(
            meta
            and meta.get("complete") is True
            and meta.get("owner_id") == self.checkpoint_owner_id
        )

    def _retain_state_checkpoints(self) -> None:
        if self.state_keep_last is None or not self.accelerator.is_main_process:
            return

        completed = []
        for path in Path(self.state_dir).iterdir():
            if not path.is_dir():
                continue
            match = re.fullmatch(r"step_(\d+)", path.name)
            if match is None or not self._state_is_completed_and_owned(path):
                continue
            completed.append((int(match.group(1)), path))
        completed.sort(key=lambda item: item[0], reverse=True)

        for _, path in completed[self.state_keep_last :]:
            deleting_path = path.with_name(
                f".{path.name}.deleting-{self.checkpoint_owner_id}"
            )
            if deleting_path.exists():
                shutil.rmtree(deleting_path)
            os.replace(path, deleting_path)
            shutil.rmtree(deleting_path)
            logger.info("[ckpt] removed retained state directory: %s", path)

    def _save_state_checkpoint(self, step_tag: str) -> str:
        state_path = Path(self.state_dir) / step_tag
        self.accelerator.wait_for_everyone()
        if self._state_is_completed_and_owned(state_path):
            return str(state_path)
        if state_path.exists():
            raise FileExistsError(
                f"Refusing to replace incomplete or foreign state directory: {state_path}"
            )

        temp_path = Path(self.state_dir) / (
            f".{step_tag}.incomplete-{self.checkpoint_owner_id}"
        )
        if self.accelerator.is_main_process:
            if temp_path.exists():
                shutil.rmtree(temp_path)
            temp_path.mkdir(parents=True)
        self.accelerator.wait_for_everyone()

        self.accelerator.save_state(output_dir=str(temp_path))
        if self.accelerator.is_main_process:
            self._save_trainer_state(str(temp_path))
            _atomic_write_json(
                temp_path / "checkpoint_meta.json",
                {
                    "complete": True,
                    "owner_id": self.checkpoint_owner_id,
                    "global_step": int(self.global_step),
                    **self._checkpoint_provenance_payload(),
                },
            )
        self.accelerator.wait_for_everyone()

        if self.accelerator.is_main_process:
            os.replace(temp_path, state_path)
            self._retain_state_checkpoints()
        self.accelerator.wait_for_everyone()
        return str(state_path)

    def save_checkpoint(self, *, save_weights: bool = True, save_state: bool = True):
        step_tag = f"step_{self.global_step:06d}"

        self.accelerator.wait_for_everyone()
        ckpt_path = None
        if save_weights and self.accelerator.is_main_process:
            ckpt_path = self._save_weights_checkpoint(step_tag=step_tag)
        self.accelerator.wait_for_everyone()

        state_path = None
        if save_state:
            state_path = self._save_state_checkpoint(step_tag=step_tag)

        self._refresh_best_checkpoint_availability()
        return {"weights_path": ckpt_path, "state_path": state_path}

    def load_training_state(self, state_dir: str):
        self._validate_state_provenance(Path(state_dir))
        self.accelerator.load_state(input_dir=state_dir)
        state_file = Path(state_dir) / "trainer_state.json"
        if state_file.exists():
            with open(state_file, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self.global_step = int(payload["global_step"])

            if "epoch" in payload and "batch_in_epoch" in payload:
                self.epoch = int(payload["epoch"])
                self.batch_in_epoch = int(payload["batch_in_epoch"])
                self._restore_train_sampler_progress(
                    epoch=self.epoch,
                    batch_in_epoch=self.batch_in_epoch,
                    sampler_state=payload.get("train_sampler_state"),
                )
                logger.info(
                    "Restored dataloader progress: epoch=%d batch_in_epoch=%d sample_offset=%d",
                    self.epoch,
                    self.batch_in_epoch,
                    self.batch_in_epoch * self.batch_size * self.accelerator.num_processes,
                )
            else:
                self.epoch = 0
                self.batch_in_epoch = 0
                self.train_sampler.clear_resume_batch_offset()
                logger.warning(
                    "State file does not contain `epoch`/`batch_in_epoch`; "
                    "optimizer/scheduler were restored, but dataloader progress resume is skipped."
                )
            self.accelerator.wait_for_everyone()
            return

        match = re.search(r"step[_-](\d+)$", str(state_dir).rstrip("/"))
        if match:
            self.global_step = int(match.group(1))
        else:
            self.global_step = 0
        self.epoch = 0
        self.batch_in_epoch = 0
        self.train_sampler.clear_resume_batch_offset()
        self.accelerator.wait_for_everyone()
        logger.info("Loaded accelerate training state from %s at step=%d", state_dir, self.global_step)
        logger.warning(
            "State file `%s` is missing; dataloader progress resume is skipped.",
            state_file,
        )

    def train(self):
        try:
            result = self._train_impl()
        except BaseException as exc:
            reason = _classify_stop_reason(exc)
            self._write_stop_reason(
                status="interrupted" if reason == "manual" else "failed",
                reason=reason,
                error=exc,
            )
            raise
        else:
            reason = "max_steps" if self.global_step >= self.max_steps else "completed"
            self._write_stop_reason(status="completed", reason=reason)
            return result
        finally:
            self._finish_wandb()

    def _train_impl(self):
        self._set_dit_only_train_mode()

        if self.max_steps is None:
            raise ValueError("`max_steps` must be set before entering the while-step training loop.")

        logger.info("Starting training with max_steps=%d.", self.max_steps)
        data_iter = iter(self.train_loader)
        self.run_start_step = self.global_step
        self.run_start_time = time.perf_counter()

        while self.global_step < self.max_steps:
            try:
                sample = self._next_train_sample(data_iter)
            except StopIteration:
                self.epoch += 1
                self.batch_in_epoch = 0
                self.train_sampler.clear_resume_batch_offset()
                if getattr(self, "role_balanced_sampling_enabled", False):
                    self.train_sampler.set_epoch(self.epoch)
                data_iter = iter(self.train_loader)
                continue

            with self.accelerator.accumulate(self.model):
                train_model = self.model if hasattr(self.model, "training_loss") else self.accelerator.unwrap_model(self.model)

                with self.accelerator.autocast():
                    loss, loss_dict = train_model.training_loss(sample)
                self._raise_if_non_finite(
                    loss,
                    value_name="loss",
                    device=loss.device,
                )
                self.accelerator.backward(loss)

                if self.accelerator.sync_gradients:
                    grad_norm = self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self._raise_if_non_finite(
                        grad_norm,
                        value_name="gradient norm",
                        device=loss.device,
                    )
                    self.optimizer.step()
                    if not self.accelerator.optimizer_step_was_skipped:
                        self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.global_step += 1
                    global_loss = float(
                        self.accelerator.gather(loss.detach().float().reshape(1)).mean().item()
                    )
                    global_loss_metrics = {}
                    for key, value in loss_dict.items():
                        metric_tensor = torch.tensor(float(value), device=loss.device, dtype=torch.float32).reshape(1)
                        global_loss_metrics[key] = float(
                            self.accelerator.gather(metric_tensor).mean().item()
                        )
                    grad_norm_tensor = torch.tensor(grad_norm, device=loss.device, dtype=torch.float32)
                    global_grad_norm = float(self.accelerator.gather(grad_norm_tensor).mean().item())

                    current_lr = float(self.optimizer.param_groups[0]["lr"])

                    if self.log_every > 0 and self.global_step % self.log_every == 0 and self.accelerator.is_main_process:
                        eta_str, steps_per_sec = self._estimate_eta()
                        description = "[train] epoch=%d step=%d/%d loss=%.4f " % (
                            self.epoch,
                            self.global_step,
                            self.max_steps,
                            global_loss,
                        )
                        if global_loss_metrics:
                            detail_str = " ".join([f"{k}={v:.4f}" for k, v in sorted(global_loss_metrics.items())])
                            description += detail_str + " "
                        description += "lr=%.2e speed=%.2f step/s, %.2f samples/s eta=%s" % (
                            current_lr,
                            steps_per_sec,
                            steps_per_sec * self.batch_size * self.accelerator.num_processes,
                            eta_str,
                        )
                        logger.info(description)

                        wandb_payload = {
                            "train/loss": global_loss,
                            "train/grad_norm": global_grad_norm,
                            "train/lr": current_lr,
                            "train/epoch": self.epoch,
                            "performance/steps_per_sec": steps_per_sec,
                            "performance/samples_per_sec": steps_per_sec * self.batch_size * self.accelerator.num_processes,
                        }
                        for key, value in global_loss_metrics.items():
                            wandb_payload[f"train/{key}"] = value
                        self._local_metrics_log(wandb_payload)
                        self._wandb_log(wandb_payload)

                    eval_metrics = None
                    if (
                        self.eval_every > 0
                        and self.val_dataset is not None
                        and self.global_step % self.eval_every == 0
                    ):
                        eval_metrics = self.evaluate()
                        self.accelerator.wait_for_everyone()
                        if eval_metrics is not None and self.accelerator.is_main_process:
                            description = "[eval] step=%d val_loss=%.4f infer_psnr=%.4f infer_ssim=%.4f" % (
                                self.global_step,
                                eval_metrics["val_loss"],
                                eval_metrics["psnr_rd"],
                                eval_metrics["ssim_rd"],
                            )
                            if "action_l2" in eval_metrics:
                                description += " action_l2=%.4f" % eval_metrics["action_l2"]
                            if "action_l1" in eval_metrics:
                                description += " action_l1=%.4f" % eval_metrics["action_l1"]
                            logger.info(description)
                            eval_payload = {
                                "eval/val_loss": float(eval_metrics["val_loss"]),
                                "eval/val_base_loss": float(
                                    eval_metrics["val_base_loss"]
                                ),
                                "eval/psnr_rg": float(eval_metrics["psnr_rg"]),
                                "eval/ssim_rg": float(eval_metrics["ssim_rg"]),
                                "eval/psnr_rd": float(eval_metrics["psnr_rd"]),
                                "eval/ssim_rd": float(eval_metrics["ssim_rd"]),
                                "eval/psnr_dg": float(eval_metrics["psnr_dg"]),
                                "eval/ssim_dg": float(eval_metrics["ssim_dg"]),
                                "eval/num_samples_total": int(
                                    eval_metrics["num_samples_total"]
                                ),
                            }
                            if "action_l2" in eval_metrics:
                                eval_payload["eval/action_l2"] = float(eval_metrics["action_l2"])
                            if "action_l1" in eval_metrics:
                                eval_payload["eval/action_l1"] = float(eval_metrics["action_l1"])
                            self._local_metrics_log(eval_payload)
                            self._wandb_log(eval_payload)

                    save_weights = _should_save_weights_at_step(
                        self.global_step,
                        save_weights_every=self.save_weights_every,
                        save_weight_steps=self.save_weight_steps,
                        is_final_step=self.global_step >= self.max_steps,
                    )
                    save_state = (
                        self.save_state_every > 0
                        and self.global_step % self.save_state_every == 0
                    )
                    if save_weights or save_state:
                        ckpt_info = self.save_checkpoint(
                            save_weights=save_weights,
                            save_state=save_state,
                        )
                        if self.accelerator.is_main_process:
                            logger.info(
                                "[ckpt] step=%d weights=%s state=%s",
                                self.global_step,
                                ckpt_info["weights_path"],
                                ckpt_info["state_path"],
                            )

                    if eval_metrics is not None:
                        self._update_best_checkpoint_index(eval_metrics)

                    if self.global_step >= self.max_steps:
                        step_tag = f"step_{self.global_step:06d}"
                        weights_path = Path(self.weights_dir) / f"{step_tag}.pt"
                        state_path = Path(self.state_dir) / step_tag
                        final_weights_needed = not weights_path.is_file()
                        final_state_needed = not self._state_is_completed_and_owned(
                            state_path
                        )
                        if final_weights_needed or final_state_needed:
                            ckpt_info = self.save_checkpoint(
                                save_weights=final_weights_needed,
                                save_state=final_state_needed,
                            )
                        else:
                            ckpt_info = {
                                "weights_path": str(weights_path),
                                "state_path": str(state_path),
                            }
                        if self.accelerator.is_main_process:
                            logger.info(
                                "[done] max_steps reached step=%d weights=%s state=%s",
                                self.global_step,
                                ckpt_info["weights_path"],
                                ckpt_info["state_path"],
                            )
                        return

        raise RuntimeError("Training loop exited before reaching max_steps.")
        
