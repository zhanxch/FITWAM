import hashlib
import os
import traceback
from typing import Optional

import numpy as np
import torch
import torchvision.transforms.functional as transforms_F
from accelerate import PartialState
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from fastwam.utils import misc
from fastwam.utils.logging_config import get_logger
from ..dataset_utils import CenterCrop, Normalize, ResizeSmallestSideAspectPreserving
from .base_lerobot_episode_dataset import BaseLerobotEpisodeDataset
from .utils.normalizer import load_dataset_stats_from_json, save_dataset_stats_to_json

logger = get_logger(__name__)

DEFAULT_PROMPT = "A video recorded from a robot's point of view executing the following instruction: {task}"


class RobotVideoEpisodeDataset(torch.utils.data.Dataset):
    """Episode-level dataset: one mp4/episode -> one padded training sample."""

    def __init__(
        self,
        dataset_dirs,
        shape_meta,
        num_frames=33,
        action_stride=10,
        video_size=(384, 384),
        camera_key=None,
        processor=None,
        text_embedding_cache_dir=None,
        context_len=128,
        pretrained_norm_stats=None,
        val_set_proportion=0.05,
        is_training_set=False,
        action_video_freq_ratio: int = 1,
        left_pad: bool = True,
        concat_multi_camera: str | None = None,
        override_instruction: Optional[str] = None,
    ):
        self.lerobot_dataset = BaseLerobotEpisodeDataset(
            dataset_dirs=dataset_dirs,
            shape_meta=OmegaConf.to_container(shape_meta, resolve=True),
            num_frames=num_frames,
            action_stride=action_stride,
            val_set_proportion=val_set_proportion,
            is_training_set=is_training_set,
            left_pad=left_pad,
        )

        self.num_frames = num_frames
        self.action_stride = action_stride
        self.action_horizon = (num_frames - 1) * action_stride
        self.action_video_freq_ratio = action_video_freq_ratio

        assert (num_frames - 1) % self.action_video_freq_ratio == 0, (
            f"num_frames-1 must be divisible by action_video_freq_ratio, got {num_frames - 1} "
            f"and {self.action_video_freq_ratio}"
        )
        assert ((num_frames - 1) // self.action_video_freq_ratio) % 4 == 0, (
            "video frames must be divisible by 4 for tokenization, got "
            f"{(num_frames - 1) // self.action_video_freq_ratio}"
        )
        self.video_sample_indices = list(range(0, num_frames, self.action_video_freq_ratio))

        self.camera_key = camera_key
        self.video_size = video_size
        self.text_embedding_cache_dir = text_embedding_cache_dir
        self.context_len = context_len
        self.left_pad = left_pad
        self.concat_multi_camera = concat_multi_camera
        self.override_instruction = override_instruction

        self.resize_transform = ResizeSmallestSideAspectPreserving(
            args={"img_w": self.video_size[1], "img_h": self.video_size[0]},
        )
        self.crop_transform = CenterCrop(
            args={"img_w": self.video_size[1], "img_h": self.video_size[0]},
        )
        self.normalize_transform = Normalize(args={"mean": 0.5, "std": 0.5})

        if processor is not None:
            if isinstance(processor, DictConfig):
                processor = instantiate(processor)
            if not pretrained_norm_stats:
                if not is_training_set:
                    raise ValueError(
                        "pretrained_norm_stats must be provided for validation/test sets."
                    )
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
            else:
                dataset_stats = load_dataset_stats_from_json(pretrained_norm_stats)
                logger.info(f"Using dataset stats: {pretrained_norm_stats}")
                if PartialState().is_main_process:
                    work_dir = misc.get_work_dir()
                    save_dataset_stats_to_json(dataset_stats, os.path.join(work_dir, "dataset_stats.json"))

            processor.set_normalizer_from_stats(dataset_stats)
            self.lerobot_dataset.set_processor(processor)

    def __len__(self):
        return len(self.lerobot_dataset)

    def _get(self, idx):
        sample = self.lerobot_dataset[idx]
        image_is_pad = sample["image_is_pad"]

        video = sample["pixel_values"]
        num_cameras = 1
        if video.ndim == 5:
            video = video[:, self.video_sample_indices, :, :, :]
            num_cameras, t_video, c, h, w = video.shape
        else:
            assert video.ndim == 4, f"Expected video shape [T, C, H, W], got {video.shape}"
            video = video[self.video_sample_indices, :, :, :]
            t_video, c, h, w = video.shape
        image_is_pad = image_is_pad[self.video_sample_indices]

        video = video.view(num_cameras, t_video, c, h, w)
        if num_cameras > 1:
            if self.concat_multi_camera == "horizontal":
                video = torch.cat([video[i] for i in range(num_cameras)], dim=-1)
            elif self.concat_multi_camera == "vertical":
                video = torch.cat([video[i] for i in range(num_cameras)], dim=-2)
            else:
                raise ValueError(
                    f"Invalid concat_multi_camera: {self.concat_multi_camera}. "
                    "Expected one of: horizontal, vertical."
                )
        else:
            video = video.squeeze(0)

        video = self.resize_transform(video)
        video = self.crop_transform(video)
        video = self.normalize_transform(video)
        video = video.permute(1, 0, 2, 3)

        action = sample["action"]
        proprio = sample["proprio"][:-1, :]
        if video.shape[1] <= 1:
            raise ValueError(f"`video` must have at least 2 frames, got shape {tuple(video.shape)}")
        if action.shape[0] % (video.shape[1] - 1) != 0:
            raise ValueError(
                f"`action` horizon must be divisible by video transitions, got {action.shape[0]} "
                f"and {video.shape[1] - 1}"
            )

        task = sample["instruction"]
        if self.override_instruction is not None:
            task = self.override_instruction
        instruction = DEFAULT_PROMPT.format(task=task)
        context, context_mask = self._get_cached_text_context(instruction)
        context[~context_mask] = 0.0
        context_mask = torch.ones_like(context_mask)

        return {
            "video": video,
            "action": action,
            "proprio": proprio,
            "prompt": instruction,
            "context": context,
            "context_mask": context_mask,
            "image_is_pad": image_is_pad,
            "action_is_pad": sample["action_is_pad"],
            "proprio_is_pad": sample["proprio_is_pad"],
        }

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
        if context.shape[0] != self.context_len or context_mask.shape[0] != self.context_len:
            raise ValueError(f"Cached context length mismatch in {cache_path}")
        return context, context_mask

    def __getitem__(self, idx):
        try:
            return self._get(idx)
        except Exception as err:
            print(f"Error processing episode idx {idx}: {err}. Returning a random episode instead.")
            print(traceback.format_exc())
            random_idx = np.random.randint(len(self))
            return self._get(random_idx)
