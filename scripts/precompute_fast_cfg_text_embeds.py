#!/usr/bin/env python3
"""Precompute T5 embeds for FAST(action) CFG prompts over an Eve manifest.

Walks the training Eve dataset once, builds FAST suffixes from each window's
action chunk, and writes missing ``*.t5_len{L}.wan22ti2v5b.pt`` caches.

Example:
  EVE_MANIFEST_PATH=... BASE_DATASET=... ROLLOUT_RAW=... \\
  python scripts/precompute_fast_cfg_text_embeds.py \\
    task=dexjoco/dexjoco_fold_glasses_offline_b1_jump_fast_lora_3e-5
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import uuid
from pathlib import Path

import hydra
import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from fastwam.datasets.cfg_text import format_fast_action_suffix
from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from fastwam.models.wan22.helpers.loader import _load_registered_model, _resolve_configs
from fastwam.models.wan22.wan_video_text_encoder import HuggingfaceTokenizer
from fastwam.utils.config_resolvers import register_default_resolvers
from fastwam.utils.logging_config import get_logger, setup_logging
from hydra.utils import instantiate

register_default_resolvers()
logger = get_logger(__name__)

DEFAULT_MODEL_ID = "Wan-AI/Wan2.2-TI2V-5B"
DEFAULT_TOKENIZER_MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B"
DEFAULT_BATCH_SIZE = 16


def _atomic_torch_save(payload: dict[str, torch.Tensor], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.parent / f".{output_path.name}.tmp.{uuid.uuid4().hex}"
    torch.save(payload, str(tmp_path))
    os.replace(tmp_path, output_path)


def _init_distributed() -> tuple[bool, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return False, 0, 1, 0
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo",
            init_method="env://",
        )
    return True, dist.get_rank(), dist.get_world_size(), local_rank


def _prompt_list_path(
    *,
    cache_dir: Path,
    manifest_path: str,
    fast_model_id: str,
    fast_max_tokens: int,
    max_windows: int,
) -> Path:
    digest = hashlib.sha256()
    digest.update(Path(manifest_path).expanduser().read_bytes())
    digest.update(fast_model_id.encode("utf-8"))
    digest.update(str(fast_max_tokens).encode("ascii"))
    digest.update(str(max_windows).encode("ascii"))
    return cache_dir / f"fast_cfg_prompts_{digest.hexdigest()[:16]}.jsonl"


def _write_prompt_list(path: Path, prompts: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.parent / f".{path.name}.tmp.{uuid.uuid4().hex}"
    with tmp_path.open("w", encoding="utf-8") as stream:
        for prompt in prompts:
            stream.write(json.dumps(prompt, ensure_ascii=True) + "\n")
    os.replace(tmp_path, path)


def _read_prompt_list(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _strip_cfg_for_action_scan(node: DictConfig) -> DictConfig:
    """Instantiate dataset without CFG so we can read bare task + actions."""

    cfg = OmegaConf.create(OmegaConf.to_container(node, resolve=True))
    cfg["is_training_set"] = False
    cfg["outcome_text_success_suffix"] = None
    cfg["outcome_text_failure_suffix"] = None
    cfg["outcome_text_dropout_prob"] = 0.0
    cfg["cfg_channel_probs"] = None
    cfg["failure_cfg_channel_probs"] = None
    # FAST prompt scan only needs actions/text; VAE caches are built later in train.
    cfg["require_vae_latent_cache"] = False
    cfg["vae_latent_cache_dir"] = None
    cfg["drop_video_when_latents_cached"] = False
    return cfg


def _scan_prompts(
    *,
    train_node: DictConfig,
    fast_model_id: str,
    fast_max_tokens: int,
    fast_fail_closed: bool,
    max_windows: int,
    rank: int,
    world_size: int,
) -> list[str]:
    scan_cfg = _strip_cfg_for_action_scan(train_node)
    dataset = instantiate(scan_cfg)
    dataset.force_skip_video = True
    logger.info(
        "Scanning Eve windows for FAST CFG prompts without video: "
        "total=%d rank=%d/%d.",
        len(dataset),
        rank,
        world_size,
    )

    prompts: list[str] = []
    seen: set[str] = set()
    errors = 0
    stop = min(len(dataset), max_windows) if max_windows > 0 else len(dataset)
    indices = range(rank, stop, world_size)
    for idx in tqdm(
        indices,
        desc=f"FAST prompt scan rank {rank}/{world_size}",
        dynamic_ncols=True,
        disable=world_size > 1 and rank != 0,
    ):
        try:
            sample = dataset[idx]
        except Exception as exc:  # noqa: BLE001
            errors += 1
            if fast_fail_closed:
                raise RuntimeError(f"FAST dataset scan failed at idx={idx}.") from exc
            if errors <= 5:
                logger.warning("skip idx=%d: %s", idx, exc)
            continue
        action = sample.get("action")
        if action is None:
            if fast_fail_closed:
                raise ValueError(f"FAST dataset scan found no action at idx={idx}.")
            continue
        raw_task = str(sample.get("prompt", ""))
        prefix = "A video recorded from a robot's point of view executing the following instruction: "
        task = raw_task[len(prefix) :] if raw_task.startswith(prefix) else raw_task
        try:
            import numpy as np

            act_np = (
                action.detach().cpu().numpy()
                if torch.is_tensor(action)
                else np.asarray(action)
            )
            fast_suffix = format_fast_action_suffix(
                act_np,
                model_id=fast_model_id,
                max_tokens=fast_max_tokens,
            )
        except Exception as exc:  # noqa: BLE001
            errors += 1
            if fast_fail_closed:
                raise RuntimeError(f"FAST encode failed at idx={idx}.") from exc
            if errors <= 5:
                logger.warning("FAST encode failed idx=%d: %s", idx, exc)
            continue
        prompt = DEFAULT_PROMPT.format(task=f"{task}{fast_suffix}")
        if prompt not in seen:
            seen.add(prompt)
            prompts.append(prompt)

    logger.info(
        "FAST scan rank=%d/%d unique_prompts=%d errors=%d",
        rank,
        world_size,
        len(prompts),
        errors,
    )
    return prompts


def _prompt_shard_path(path: Path, rank: int, world_size: int) -> Path:
    return path.with_name(f".{path.name}.rank{rank:03d}-of-{world_size:03d}")


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig) -> None:
    setup_logging(log_level=logging.INFO)
    is_distributed, rank, world_size, local_rank = _init_distributed()
    model_cfg = cfg.get("model")
    data_cfg = cfg.get("data")
    if model_cfg is None or data_cfg is None:
        raise ValueError("Need task overrides that set both `model` and `data`.")

    train_node = data_cfg.get("train")
    if train_node is None:
        raise ValueError("`data.train` is required")
    cache_dir = Path(str(train_node.text_embedding_cache_dir)).expanduser()
    context_len = int(train_node.context_len)
    fast_model_id = str(
        train_node.get("fast_tokenizer_model_id", "physical-intelligence/fast")
    )
    fast_max_tokens = int(train_node.get("fast_max_tokens", 32))
    fast_fail_closed = bool(train_node.get("fast_fail_closed", False))
    max_windows = int(cfg.get("fast_cfg_max_windows", 0))
    prompt_list_path = _prompt_list_path(
        cache_dir=cache_dir,
        manifest_path=str(train_node.manifest_path),
        fast_model_id=fast_model_id,
        fast_max_tokens=fast_max_tokens,
        max_windows=max_windows,
    )
    if not prompt_list_path.exists():
        prompts = _scan_prompts(
            train_node=train_node,
            fast_model_id=fast_model_id,
            fast_max_tokens=fast_max_tokens,
            fast_fail_closed=fast_fail_closed,
            max_windows=max_windows,
            rank=rank,
            world_size=world_size,
        )
        _write_prompt_list(
            _prompt_shard_path(prompt_list_path, rank, world_size), prompts
        )
    if is_distributed:
        dist.barrier()
    if rank == 0 and not prompt_list_path.exists():
        prompts = []
        seen: set[str] = set()
        for shard_rank in range(world_size):
            shard_path = _prompt_shard_path(
                prompt_list_path, shard_rank, world_size
            )
            for prompt in _read_prompt_list(shard_path):
                if prompt not in seen:
                    seen.add(prompt)
                    prompts.append(prompt)
        _write_prompt_list(prompt_list_path, prompts)
        logger.info("Wrote FAST prompt list: %s", prompt_list_path)
    if is_distributed:
        dist.barrier()
    prompts = _read_prompt_list(prompt_list_path)
    if not prompts:
        if fast_fail_closed:
            raise RuntimeError("No FAST prompts collected for fail-closed formal recipe.")
        logger.warning("No FAST prompts collected; nothing to encode.")
        return

    device = (
        f"cuda:{local_rank}"
        if torch.cuda.is_available() and is_distributed
        else "cuda" if torch.cuda.is_available() else "cpu"
    )
    torch_dtype = torch.bfloat16
    model_id = str(model_cfg.get("model_id", DEFAULT_MODEL_ID))
    tokenizer_model_id = str(
        model_cfg.get("tokenizer_model_id", DEFAULT_TOKENIZER_MODEL_ID)
    )
    redirect_common_files = bool(model_cfg.get("redirect_common_files", True))
    _, text_config, _, tokenizer_config = _resolve_configs(
        model_id=model_id,
        tokenizer_model_id=tokenizer_model_id,
        redirect_common_files=redirect_common_files,
    )
    text_config.download_if_necessary()
    tokenizer_config.download_if_necessary()
    text_encoder = _load_registered_model(
        text_config.path,
        "wan_video_text_encoder",
        torch_dtype=torch_dtype,
        device=device,
    ).eval()
    tokenizer = HuggingfaceTokenizer(
        name=tokenizer_config.path,
        seq_len=context_len,
        clean="whitespace",
    )

    enc_id = "wan22ti2v5b"
    prompts = prompts[rank::world_size]
    batch_size = int(cfg.get("fast_cfg_batch_size", DEFAULT_BATCH_SIZE))
    if batch_size <= 0:
        raise ValueError(f"fast_cfg_batch_size must be positive, got {batch_size}")
    new_count = 0
    skip_count = 0
    with torch.no_grad():
        for start in tqdm(
            range(0, len(prompts), batch_size),
            desc=f"Encode FAST prompts rank {rank}/{world_size}",
            dynamic_ncols=True,
            disable=is_distributed and rank != 0,
        ):
            batch = prompts[start : start + batch_size]
            missing = []
            for prompt in batch:
                hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
                path = cache_dir / f"{hashed}.t5_len{context_len}.{enc_id}.pt"
                if path.exists():
                    skip_count += 1
                else:
                    missing.append(prompt)
            if not missing:
                continue
            ids, mask = tokenizer(missing, return_mask=True, add_special_tokens=True)
            ids = ids.to(device)
            mask = mask.to(device=device, dtype=torch.bool)
            context = text_encoder(ids, mask)
            for i, prompt in enumerate(missing):
                hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
                path = cache_dir / f"{hashed}.t5_len{context_len}.{enc_id}.pt"
                payload = {
                    "context": context[i].detach().to("cpu", dtype=torch.bfloat16).contiguous(),
                    "mask": mask[i].detach().to("cpu", dtype=torch.bool).contiguous(),
                }
                _atomic_torch_save(payload, path)
                new_count += 1

    counts = torch.tensor([new_count, skip_count], device=device, dtype=torch.long)
    if is_distributed:
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)
    if rank == 0:
        logger.info(
            "FAST CFG cache done: new=%d skip=%d dir=%s prompt_list=%s",
            int(counts[0].item()),
            int(counts[1].item()),
            cache_dir,
            prompt_list_path,
        )
    if is_distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
