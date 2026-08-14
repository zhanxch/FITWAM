import hashlib
import os
from typing import Optional, Any
import time
import numpy as np
import traceback
import torch
import torchvision.transforms.functional as transforms_F
from contextlib import contextmanager

from omegaconf import DictConfig, OmegaConf

from hydra.utils import instantiate
from .base_lerobot_dataset import BaseLerobotDataset
from .utils.normalizer import save_dataset_stats_to_json, load_dataset_stats_from_json
from ..dataset_utils import ResizeSmallestSideAspectPreserving, CenterCrop, Normalize
from fastwam.datasets.cfg_text import apply_ternary_cfg_suffix, normalize_cfg_channel_probs
from fastwam.datasets.vae_latent_cache import (
    resolve_optional_path,
    vae_latent_cache_path,
)
from fastwam.utils.logging_config import get_logger
from fastwam.utils import misc, pytorch_utils
from accelerate import PartialState
logger = get_logger(__name__)


def _as_bool(value: Any, *, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "y"}:
            return True
        if text in {"0", "false", "no", "n", ""}:
            return False
    raise ValueError(f"{name} must be bool-like, got {value!r}")


DEFAULT_PROMPT = "A video recorded from a robot's point of view executing the following instruction: {task}"

class RobotVideoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_dirs,
        shape_meta,
        num_frames=33,
        video_size=[384, 640],
        camera_key=None,
        processor=None,
        text_embedding_cache_dir=None,
        context_len=128,
        pretrained_norm_stats=None,
        val_set_proportion=0.05,
        is_training_set=False,
        global_sample_stride=1,
        action_video_freq_ratio: int = 1,
        skip_padding_as_possible: bool = False,
        max_padding_retry: int = 3,
        concat_multi_camera: str = "horizontal", # "horizontal", "vertical", "robotwin", or None
        override_instruction: Optional[str] = None, # whether to hardcode a specific instruction for all samples, for debugging
        tolerance_s: float = 1e-4,
        video_backend: Optional[str] = "pyav",
        action_loss_zero_if_instruction_contains: Optional[str] = None,
        outcome_flag_if_instruction_contains: Optional[str] = None,
        strip_instruction_suffix_if_contains: Optional[str] = None,
        outcome_text_success_suffix: Optional[str] = None,
        outcome_text_failure_suffix: Optional[str] = None,
        outcome_text_dropout_prob: float = 0.0,
        cfg_channel_probs=None,
        failure_cfg_channel_probs=None,
        fast_tokenizer_model_id: str = "physical-intelligence/fast",
        fast_max_tokens: int = 32,
        fast_fail_closed: bool = False,
        vae_latent_cache_dir=None,
        require_vae_latent_cache: bool = False,
        drop_video_when_latents_cached: bool = False,
    ):
        self.lerobot_dataset = BaseLerobotDataset(
            dataset_dirs=dataset_dirs,
            shape_meta=OmegaConf.to_container(shape_meta, resolve=True),
            obs_size=num_frames,
            action_size=num_frames - 1,
            val_set_proportion=val_set_proportion,
            is_training_set=is_training_set,
            global_sample_stride=global_sample_stride,
            tolerance_s=tolerance_s,
            video_backend=video_backend,
        )
    
        self.num_frames = num_frames
        self.action_video_freq_ratio = action_video_freq_ratio
        self.is_training_set = bool(is_training_set)
        
        assert (num_frames - 1) % self.action_video_freq_ratio == 0, \
            f"num_frames-1 must be divisible by action_video_freq_ratio, got {num_frames - 1} and {self.action_video_freq_ratio}"
        assert ((num_frames - 1) // self.action_video_freq_ratio) % 4 == 0, \
            f"video frames must be divisible by 4 for tokenization, got {(num_frames - 1) // self.action_video_freq_ratio}"
        self.video_sample_indices = list(range(0, num_frames, self.action_video_freq_ratio))

        self.camera_key = camera_key
        self.lerobot_dataset._set_return_images(True)

        self.video_size = video_size
        self.text_embedding_cache_dir = text_embedding_cache_dir
        self.context_len = context_len
        self.skip_padding_as_possible = skip_padding_as_possible
        self.max_padding_retry = max_padding_retry
        self.concat_multi_camera = concat_multi_camera
        self.override_instruction = override_instruction
        self.action_loss_zero_if_instruction_contains = action_loss_zero_if_instruction_contains
        self.outcome_flag_if_instruction_contains = outcome_flag_if_instruction_contains
        self.strip_instruction_suffix_if_contains = strip_instruction_suffix_if_contains
        self.outcome_text_success_suffix = outcome_text_success_suffix
        self.outcome_text_failure_suffix = outcome_text_failure_suffix
        dropout_prob = float(outcome_text_dropout_prob)
        if not np.isfinite(dropout_prob) or dropout_prob < 0.0 or dropout_prob > 1.0:
            raise ValueError(
                "outcome_text_dropout_prob must be finite and in [0, 1], "
                f"got {outcome_text_dropout_prob!r}"
            )
        self.outcome_text_dropout_prob = dropout_prob
        self.cfg_channel_probs = normalize_cfg_channel_probs(cfg_channel_probs)
        self.failure_cfg_channel_probs = normalize_cfg_channel_probs(
            failure_cfg_channel_probs
        )
        self.fast_tokenizer_model_id = str(fast_tokenizer_model_id)
        self.fast_max_tokens = int(fast_max_tokens)
        self.fast_fail_closed = _as_bool(
            fast_fail_closed,
            name="fast_fail_closed",
        )
        if self.fast_max_tokens < 1:
            raise ValueError(f"fast_max_tokens must be >= 1, got {fast_max_tokens}")
        if (
            self.cfg_channel_probs is not None
            and self.cfg_channel_probs["outcome"] > 0.0
            and self.outcome_text_success_suffix is None
        ):
            raise ValueError(
                "Success outcome channel has positive probability but "
                "outcome_text_success_suffix is null."
            )
        effective_failure_probs = (
            self.failure_cfg_channel_probs
            if self.failure_cfg_channel_probs is not None
            else self.cfg_channel_probs
        )
        if (
            effective_failure_probs is not None
            and effective_failure_probs["outcome"] > 0.0
            and self.outcome_text_failure_suffix is None
        ):
            raise ValueError(
                "Failure outcome channel has positive probability but "
                "outcome_text_failure_suffix is null."
            )
        self.vae_latent_cache_dir = resolve_optional_path(vae_latent_cache_dir)
        self.require_vae_latent_cache = _as_bool(
            require_vae_latent_cache, name="require_vae_latent_cache"
        )
        self.drop_video_when_latents_cached = _as_bool(
            drop_video_when_latents_cached, name="drop_video_when_latents_cached"
        )
        if self.require_vae_latent_cache and self.vae_latent_cache_dir is None:
            raise ValueError(
                "require_vae_latent_cache=True requires vae_latent_cache_dir / "
                "VAE_LATENT_CACHE_DIR to be set."
            )

        self.resize_transform = ResizeSmallestSideAspectPreserving(
            args={"img_w": self.video_size[1], "img_h": self.video_size[0]},
        )
        self.crop_transform = CenterCrop(
            args={"img_w": self.video_size[1], "img_h": self.video_size[0]},
        )
        self.normalize_transform = Normalize(
            args={"mean": 0.5, "std": 0.5},
        )
        if processor is not None:
            if isinstance(processor, DictConfig):
                processor = instantiate(processor)
            if processor.wants_modality_stats:
                if PartialState().is_main_process:
                    logger.info(
                        "Loading modality stats from %s (GR00T-style).",
                        processor.norm_stats_meta_dir,
                    )
                processor.set_normalizer_from_modality_stats()
            elif not pretrained_norm_stats:
                if not is_training_set:
                    raise ValueError("pretrained_norm_stats must be provided for validation/test sets since we don't want to calculate stats on them.")
                if PartialState().is_main_process:
                    logger.info("Calculating dataset stats for normalization...")
                    dataset_stats = self.lerobot_dataset.get_dataset_stats(processor)
                    work_dir = misc.get_work_dir()
                    save_dataset_stats_to_json(dataset_stats, os.path.join(work_dir, "dataset_stats.json"))
                else:
                    dataset_stats = None
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    obj_list = [dataset_stats]
                    torch.distributed.broadcast_object_list(obj_list, src=0)
                    dataset_stats = obj_list[0]
                processor.set_normalizer_from_stats(dataset_stats)
            else:
                dataset_stats = load_dataset_stats_from_json(pretrained_norm_stats)
                logger.info(f"Using dataset stats: {pretrained_norm_stats}")
                work_dir = misc.get_work_dir()
                dest_path = os.path.join(work_dir, "dataset_stats.json")
                if (
                    PartialState().is_main_process
                    and os.path.abspath(pretrained_norm_stats) != os.path.abspath(dest_path)
                ):
                    save_dataset_stats_to_json(dataset_stats, dest_path)
                processor.set_normalizer_from_stats(dataset_stats)
            self.lerobot_dataset.set_processor(processor)
        
    def __len__(self):
        return len(self.lerobot_dataset)

    def _apply_outcome_text_cfg(
        self,
        task_text: str,
        outcome_flag: int,
        *,
        actions=None,
    ) -> tuple[str, str]:
        """Append ternary CFG suffix: outcome / FAST(action) / base."""

        return apply_ternary_cfg_suffix(
            task_text,
            outcome_flag=int(outcome_flag),
            success_suffix=self.outcome_text_success_suffix,
            failure_suffix=self.outcome_text_failure_suffix,
            channel_probs=self.cfg_channel_probs,
            failure_channel_probs=self.failure_cfg_channel_probs,
            actions=actions,
            is_training=self.is_training_set,
            fast_model_id=self.fast_tokenizer_model_id,
            fast_max_tokens=self.fast_max_tokens,
            fast_fail_closed=self.fast_fail_closed,
            legacy_dropout_prob=self.outcome_text_dropout_prob,
        )

    def _get(self, idx, *, outcome_flag_override: Optional[int] = None, skip_video: bool = False):
        sample_idx = idx
        sample = None
        if skip_video:
            self.lerobot_dataset._set_return_images(False)
        try:
            for attempt in range(self.max_padding_retry + 1):
                sample = self.lerobot_dataset[sample_idx]

                if not self.skip_padding_as_possible:
                    break

                action_is_pad = sample["action_is_pad"]
                image_is_pad = sample["image_is_pad"]
                proprio_is_pad = sample["proprio_is_pad"]
                has_pad = False
                if bool(action_is_pad.any().item()):
                    has_pad = True
                if bool(image_is_pad.any().item()):
                    has_pad = True
                if bool(proprio_is_pad.any().item()):
                    has_pad = True

                if not has_pad or attempt >= self.max_padding_retry:
                    break

                sample_idx = np.random.randint(len(self.lerobot_dataset))
        finally:
            if skip_video:
                self.lerobot_dataset._set_return_images(True)

        image_is_pad = sample["image_is_pad"]
        video = sample["pixel_values"]  # [T, C, H, W] / [num_cameras, T, C, H, W] / None
        num_cameras = 1
        if video is None:
            T_video = len(self.video_sample_indices)
            image_is_pad = image_is_pad[self.video_sample_indices]
            # Placeholder unused when input_latents are provided.
            video = torch.zeros(3, T_video, 16, 16, dtype=torch.float32)
        else:
            if video.ndim == 5:
                video = video[:, self.video_sample_indices, :, :, :] # [num_cameras, T_video, C, H, W]
                num_cameras, T_video, C, H, W = video.shape
            else:
                assert video.ndim == 4, f"Expected video to have shape [T, C, H, W], but got {video.shape}"
                video = video[self.video_sample_indices, :, :, :] # [T_video, C, H, W]
                T_video, C, H, W = video.shape
            image_is_pad = image_is_pad[self.video_sample_indices]

            video = video.view(num_cameras, T_video, C, H, W)  # [num_cameras, T_video, C, H, W]
            if self.concat_multi_camera == "robotwin":
                if num_cameras != 3:
                    raise ValueError(
                        f"`concat_multi_camera='robotwin'` requires exactly 3 cameras, got {num_cameras}"
                    )
                cam_top = transforms_F.resize(
                    video[0],
                    size=[256, 320],
                    interpolation=transforms_F.InterpolationMode.BILINEAR,
                    antialias=True,
                )  # [T_video, C, 256, 320]
                cam_left = transforms_F.resize(
                    video[1],
                    size=[128, 160],
                    interpolation=transforms_F.InterpolationMode.BILINEAR,
                    antialias=True,
                )  # [T_video, C, 128, 160]
                cam_right = transforms_F.resize(
                    video[2],
                    size=[128, 160],
                    interpolation=transforms_F.InterpolationMode.BILINEAR,
                    antialias=True,
                )  # [T_video, C, 128, 160]
                bottom = torch.cat([cam_left, cam_right], dim=-1)  # [T_video, C, 128, 320]
                video = torch.cat([cam_top, bottom], dim=-2)  # [T_video, C, 384, 320]
            elif num_cameras > 1:
                if self.concat_multi_camera == "horizontal":
                    video = torch.cat([video[i] for i in range(num_cameras)], dim=-1)  # [T_video, C, H, num_cameras*W]
                elif self.concat_multi_camera == "vertical":
                    video = torch.cat([video[i] for i in range(num_cameras)], dim=-2)  # [T_video, C, num_cameras*H, W]
                else:
                    raise ValueError(
                        f"Invalid concat_multi_camera: {self.concat_multi_camera}. "
                        "Expected one of: horizontal, vertical, robotwin."
                    )
            else:
                video = video.squeeze(0)  # [T_video, C, H, W]

            # final resize and normalization
            video = self.resize_transform(video)
            video = self.crop_transform(video)
            video = self.normalize_transform(video)  # [T_video, C, H, W]

            video = video.permute(1, 0, 2, 3) # [C, T_video, H, W], range [-1, 1]

        # Proxy (from lerobot):
        #   action: [num_frames-1, action_dim] # start from t0, except the last frame
        #   proprio: [num_frames, proprio_dim] # start from t0 to the last frame, aligned with video frames
        action = sample["action"] # [T-1, action_dim]
        proprio = sample["proprio"] # [T, state_dim], first state is current-condition
        proprio_is_pad = sample["proprio_is_pad"]
        if video.shape[1] <= 1:
            raise ValueError(f"`video` must have at least 2 frames, got shape {tuple(video.shape)}")
        if action.shape[0] % (video.shape[1] - 1) != 0:
            raise ValueError(
                f"`action` horizon must be divisible by `video` transitions, got {action.shape[0]} and {video.shape[1] - 1}"
            )

        task = sample["instruction"]
        raw_task_text = str(task)
        task_text = raw_task_text

        # FIXME
        if self.override_instruction is not None:
            task_text = str(self.override_instruction)
            raw_task_text = task_text

        action_loss_weight = 1.0
        marker = self.action_loss_zero_if_instruction_contains
        if marker is not None and marker in raw_task_text:
            action_loss_weight = 0.0

        outcome_flag = 0
        outcome_marker = self.outcome_flag_if_instruction_contains
        if outcome_marker is not None and outcome_marker in raw_task_text:
            outcome_flag = 1
        if outcome_flag_override is not None:
            outcome_flag = int(outcome_flag_override)

        strip_marker = self.strip_instruction_suffix_if_contains
        if strip_marker is not None and strip_marker in task_text:
            task_text = task_text.replace(strip_marker, "").strip()
            task_text = " ".join(task_text.split())

        # Use raw action chunk for FAST conditioning (before any further mutation).
        action_for_cfg = action.detach().cpu().numpy() if torch.is_tensor(action) else action
        base_task_text = task_text
        video_task_text, cfg_channel = self._apply_outcome_text_cfg(
            base_task_text,
            outcome_flag,
            actions=action_for_cfg,
        )
        instruction = DEFAULT_PROMPT.format(task=video_task_text)
        action_instruction = (
            DEFAULT_PROMPT.format(task=base_task_text)
            if cfg_channel == "fast"
            else instruction
        )

        context, context_mask = self._get_cached_text_context(instruction)
        # NOTE: to keep consistent with wan2.2's behavior
        context[~context_mask] = 0.0
        context_mask = torch.ones_like(context_mask)
        if cfg_channel == "fast":
            action_context, action_context_mask = self._get_cached_text_context(
                action_instruction
            )
            action_context[~action_context_mask] = 0.0
            action_context_mask = torch.ones_like(action_context_mask)
        else:
            action_context = context
            action_context_mask = context_mask

        data = {
            "video": video,
            "action": action,
            "proprio": proprio,
            "prompt": instruction,
            "context": context,
            "context_mask": context_mask,
            "action_prompt": action_instruction,
            "action_context": action_context,
            "action_context_mask": action_context_mask,
            "cfg_channel": cfg_channel,
            "image_is_pad": image_is_pad,
            "action_is_pad": sample["action_is_pad"],
            "proprio_is_pad": proprio_is_pad,
            "action_loss_weight": torch.tensor(action_loss_weight, dtype=torch.float32),
            "outcome_flag": torch.tensor(outcome_flag, dtype=torch.long),
        }
        return data

    def _maybe_attach_vae_latents(
        self,
        data: dict,
        *,
        sample_id: str,
        window_start: int,
    ) -> dict:
        if self.vae_latent_cache_dir is None:
            if self.require_vae_latent_cache:
                raise ValueError("require_vae_latent_cache=True but vae_latent_cache_dir is unset.")
            return data
        cache_path = vae_latent_cache_path(
            self.vae_latent_cache_dir,
            sample_id=str(sample_id),
            window_start=int(window_start),
        )
        if not cache_path.exists():
            if self.require_vae_latent_cache:
                raise FileNotFoundError(
                    f"Missing VAE latent cache for sample_id={sample_id!r} "
                    f"window_start={window_start}: {cache_path}"
                )
            return data
        payload = torch.load(cache_path, map_location="cpu")
        latents = payload["input_latents"] if isinstance(payload, dict) else payload
        if not torch.is_tensor(latents):
            raise TypeError(f"Cached VAE latent must be a tensor, got {type(latents)} in {cache_path}")
        if latents.ndim != 4:
            raise ValueError(
                f"Cached VAE latent must be [C,T,H,W], got {tuple(latents.shape)} in {cache_path}"
            )
        data["input_latents"] = latents.contiguous()
        if self.drop_video_when_latents_cached:
            t_video = len(self.video_sample_indices)
            data["video"] = torch.zeros(3, t_video, 16, 16, dtype=torch.float32)
        return data

    def _get_cached_text_context(self, prompt: str):
        if self.text_embedding_cache_dir is None:
            raise ValueError("text_embedding_cache_dir is not set.")
        cache_dir = self.text_embedding_cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        cache_path = os.path.join(cache_dir, f"{hashed}.t5_len{self.context_len}.wan22ti2v5b.pt")
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"Missing text embedding cache: {cache_path}. "
                "Run scripts/precompute_text_embeds.py first."
            )
        payload = torch.load(cache_path, map_location="cpu")
        context = payload["context"]
        context_mask = payload["mask"].bool()
        if context.ndim != 2:
            raise ValueError(
                f"Cached `context` must be 2D [L, D], got shape {tuple(context.shape)} in {cache_path}"
            )
        if context_mask.ndim != 1:
            raise ValueError(
                f"Cached `mask` must be 1D [L], got shape {tuple(context_mask.shape)} in {cache_path}"
            )
        if context.shape[0] != self.context_len:
            raise ValueError(
                f"Cached context_len mismatch: expected {self.context_len}, got {context.shape[0]} in {cache_path}"
            )
        if context_mask.shape[0] != self.context_len:
            raise ValueError(
                f"Cached mask_len mismatch: expected {self.context_len}, got {context_mask.shape[0]} in {cache_path}"
            )

        return context, context_mask

    def __getitem__(self, idx):
        try:
            data = self._get(idx)
        except Exception as e:
            print(f"Error processing sample idx {idx}: {e}. Returning a random sample instead.")
            # trace back
            print(traceback.format_exc())
            random_idx = np.random.randint(len(self))
            data = self._get(random_idx)
        return data
