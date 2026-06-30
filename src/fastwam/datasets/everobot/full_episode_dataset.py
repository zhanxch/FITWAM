"""EveRobotFullEpisodeDataset: one mp4 episode = one training sample.

Unlike ``EveRobotDataset`` (fixed ``num_frames=33`` windows), this loader keeps
the **entire** episode (after optional leading dropout and tail trim) as a
single sample — matching DiffSynth-Studio WAN video fine-tuning where each
video is one dataset item.

Variable ``T_video`` per episode; action horizon is ``4 * (T_video - 1)``.
"""

from __future__ import annotations

import hashlib
import os
import random
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torchvision.transforms.functional as transforms_F
from accelerate import PartialState
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from fastwam.datasets.dataset_utils import CenterCrop, Normalize, ResizeSmallestSideAspectPreserving
from fastwam.datasets.lerobot.lerobot.datasets.video_utils import decode_video_frames
from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
from fastwam.utils.logging_config import get_logger

from .episode_length import EpisodeTrim, compute_episode_trim
from .manifest import load_manifest

logger = get_logger(__name__)

DEFAULT_PROMPT = "A video recorded from a robot's point of view executing the following instruction: {task}"


class EveRobotFullEpisodeDataset(torch.utils.data.Dataset):
    """Load whole episodes from EveRobot manifest + per-episode npz + mp4."""

    def __init__(
        self,
        dataset_dirs: List[str],
        shape_meta: Dict[str, Any],
        video_size: List[int],
        processor=None,
        text_embedding_cache_dir: Optional[str] = None,
        context_len: int = 128,
        pretrained_norm_stats: Optional[str] = None,
        val_set_proportion: float = 0.05,
        is_training_set: bool = False,
        seed: int = 42,
        action_video_freq_ratio: int = 4,
        concat_multi_camera: str = "horizontal",
        override_instruction: Optional[str] = None,
        tolerance_s: float = 1e-4,
        video_backend: Optional[str] = "pyav",
        manifest_path: Optional[str] = None,
        drop_first_frames: int = 0,
        drop_first_frames_random: bool = False,
        samples_per_episode: int = 1,
        min_t_video: int = 2,
    ):
        self.shape_meta = OmegaConf.to_container(shape_meta, resolve=True)
        self.video_size = list(video_size)
        self.action_video_freq_ratio = int(action_video_freq_ratio)
        self.concat_multi_camera = concat_multi_camera
        self.override_instruction = override_instruction
        self.tolerance_s = float(tolerance_s)
        self.video_backend = video_backend
        self.text_embedding_cache_dir = text_embedding_cache_dir
        self.context_len = int(context_len)
        self.is_training_set = bool(is_training_set)
        self.drop_first_frames = int(drop_first_frames)
        self.drop_first_frames_random = bool(drop_first_frames_random)
        self.samples_per_episode = int(samples_per_episode)
        self.min_t_video = int(min_t_video)

        if self.drop_first_frames < 0:
            raise ValueError(f"`drop_first_frames` must be >= 0, got {self.drop_first_frames}")
        if self.samples_per_episode < 1:
            raise ValueError(f"`samples_per_episode` must be >= 1, got {self.samples_per_episode}")
        if (
            self.samples_per_episode > 1
            and not (self.is_training_set and self.drop_first_frames > 0 and self.drop_first_frames_random)
        ):
            raise ValueError(
                "`samples_per_episode` > 1 requires training-time random drop augmentation: "
                "set `drop_first_frames` > 0 and `drop_first_frames_random=true`."
            )

        self.action_meta = self.shape_meta["action"]
        self.state_meta = self.shape_meta["state"]
        self.image_meta = self.shape_meta["images"]
        self.video_keys = [m["key"] for m in self.image_meta]

        dataset_dir = Path(dataset_dirs[0]).resolve()
        if manifest_path is None:
            manifest_path = str(dataset_dir / "everobot" / "everobot_manifest.json")
        manifest = load_manifest(manifest_path)
        if manifest.get("format") != "everobot":
            raise ValueError(f"Unsupported manifest format: {manifest.get('format')}")

        self.fps = int(manifest["fps"])
        self._manifest = manifest
        self._dataset_dir = dataset_dir

        episode_records = list(manifest["episodes"])
        rng = np.random.default_rng(seed)
        episode_indices = list(range(len(episode_records)))
        rng.shuffle(episode_indices)

        if val_set_proportion < 1e-6:
            selected = episode_indices
        else:
            split_idx = int(len(episode_indices) * (1 - val_set_proportion))
            if is_training_set:
                selected = episode_indices[:split_idx]
            else:
                selected = episode_indices[split_idx:]

        self._episodes: List[Dict[str, Any]] = []
        self._trims: List[EpisodeTrim] = []
        for idx in selected:
            ep = episode_records[idx]
            length = int(ep["length"])
            drop_start = 0 if self.drop_first_frames <= 0 else min(
                self.drop_first_frames, max(length - 5, 0)
            )
            if self.drop_first_frames_random and is_training_set:
                drop_start = 0
            trim = compute_episode_trim(
                length,
                drop_start=drop_start,
                action_video_freq_ratio=self.action_video_freq_ratio,
                min_t_video=self.min_t_video,
            )
            if trim is None:
                logger.warning(
                    "Episode %d length=%d too short after drop_start=%d; skipping.",
                    ep.get("episode_index", idx),
                    length,
                    drop_start,
                )
                continue
            if "npz_path" not in ep:
                raise ValueError(
                    f"Episode {ep.get('episode_index')} missing npz_path. "
                    "Run scripts/convert_lerobot_to_everobot.py with --extract-arrays."
                )
            self._episodes.append(ep)
            self._trims.append(trim)

        if not self._episodes:
            raise ValueError("No valid full-episode samples found in manifest.")

        self.resize_transform = ResizeSmallestSideAspectPreserving(
            args={"img_w": self.video_size[1], "img_h": self.video_size[0]},
        )
        self.crop_transform = CenterCrop(
            args={"img_w": self.video_size[1], "img_h": self.video_size[0]},
        )
        self.normalize_transform = Normalize(args={"mean": 0.5, "std": 0.5})

        self.processor = None
        if processor is not None:
            if isinstance(processor, DictConfig):
                processor = instantiate(processor)
            self.processor = processor
            self._init_processor(pretrained_norm_stats)

        aug_enabled = self._drop_augmentation_enabled()
        logger.info(
            "EveRobotFullEpisodeDataset: %d episodes, samples_per_episode=%d, "
            "total_samples=%d, drop_augmentation=%s, drop_first_frames=%d (random=%s), "
            "action_video_freq_ratio=%d, t_video range [%d, %d]",
            len(self._episodes),
            self.samples_per_episode,
            len(self),
            aug_enabled,
            self.drop_first_frames,
            self.drop_first_frames_random,
            self.action_video_freq_ratio,
            min(t.t_video for t in self._trims),
            max(t.t_video for t in self._trims),
        )

    def _init_processor(self, pretrained_norm_stats: Optional[str]):
        processor = self.processor
        assert processor is not None

        if processor.wants_modality_stats:
            if PartialState().is_main_process:
                logger.info(
                    "Loading modality stats from %s (GR00T-style).",
                    processor.norm_stats_meta_dir,
                )
            processor.set_normalizer_from_modality_stats()
        elif not pretrained_norm_stats:
            if not self.is_training_set:
                raise ValueError(
                    "pretrained_norm_stats must be provided for validation/test sets."
                )
            raise ValueError(
                "EveRobotFullEpisodeDataset requires norm_stats_source=meta or pretrained_norm_stats."
            )
        else:
            dataset_stats = load_dataset_stats_from_json(pretrained_norm_stats)
            processor.set_normalizer_from_stats(dataset_stats)

        if self.is_training_set:
            processor.train()
        else:
            processor.eval()

    def __len__(self) -> int:
        return len(self._episodes) * self.samples_per_episode

    def _drop_augmentation_enabled(self) -> bool:
        return (
            self.is_training_set
            and self.drop_first_frames > 0
            and self.drop_first_frames_random
        )

    def _pick_drop_start(self, ep_length: int) -> int:
        upper = min(self.drop_first_frames, max(ep_length - 5, 0))
        return random.randint(0, upper)

    def _resolve_trim(self, ep_idx: int) -> EpisodeTrim:
        ep = self._episodes[ep_idx]
        length = int(ep["length"])
        if self._drop_augmentation_enabled():
            drop_start = self._pick_drop_start(length)
        elif self.drop_first_frames > 0:
            drop_start = min(self.drop_first_frames, max(length - 5, 0))
        else:
            drop_start = 0

        if drop_start == self._trims[ep_idx].drop_start and not self._drop_augmentation_enabled():
            return self._trims[ep_idx]

        trim = compute_episode_trim(
            length,
            drop_start=drop_start,
            action_video_freq_ratio=self.action_video_freq_ratio,
            min_t_video=self.min_t_video,
        )
        if trim is None:
            return self._trims[ep_idx]
        return trim

    def _load_npz_arrays(self, npz_path: str, trim: EpisodeTrim):
        payload = np.load(npz_path)
        action = payload["action"]
        state = payload["state"]
        timestamp = payload["timestamp"]

        start = trim.drop_start
        end = start + trim.num_raw_steps
        action_end = start + trim.action_horizon

        action_slice = action[start:action_end]
        state_slice = state[start:end]
        ts_slice = timestamp[start:end]

        if action_slice.shape[0] != trim.action_horizon:
            raise ValueError(
                f"Action slice length mismatch: expected {trim.action_horizon}, "
                f"got {action_slice.shape[0]} in {npz_path}"
            )
        if state_slice.shape[0] != trim.num_raw_steps:
            raise ValueError(
                f"State slice length mismatch: expected {trim.num_raw_steps}, "
                f"got {state_slice.shape[0]} in {npz_path}"
            )

        video_ts = ts_slice[trim.video_sample_indices].astype(np.float32).tolist()
        return action_slice, state_slice, video_ts

    def _normalize_action_state(
        self,
        action_np: np.ndarray,
        state_np: np.ndarray,
        task: str,
        idx: int,
    ):
        assert self.processor is not None
        action = torch.from_numpy(action_np.astype(np.float32))
        state = torch.from_numpy(state_np.astype(np.float32))

        data: Dict[str, Any] = {
            "task": task,
            "idx": idx,
            "action": {},
            "state": {},
            "action_is_pad": torch.zeros(action.shape[0], dtype=torch.bool),
            "state_is_pad": torch.zeros(state.shape[0], dtype=torch.bool),
        }
        for meta in self.action_meta:
            data["action"][meta["key"]] = action
        for meta in self.state_meta:
            data["state"][meta["key"]] = state

        if self.processor.action_state_transforms is not None:
            data = self.processor.action_state_transform(data)
        data = self.processor.normalizer.forward(data)
        data = self.processor.action_state_merger.forward(data)
        return data["action"], data["state"]

    def _decode_camera_video(self, video_path: str, timestamps: List[float]) -> torch.Tensor:
        frames = decode_video_frames(
            video_path,
            timestamps,
            self.tolerance_s,
            self.video_backend,
        )
        return frames

    def _concat_and_process_video(self, camera_videos: List[torch.Tensor]) -> torch.Tensor:
        video = torch.stack(camera_videos, dim=0)
        num_cameras, t_video, c, h, w = video.shape

        if self.concat_multi_camera == "robotwin":
            if num_cameras != 3:
                raise ValueError(
                    f"`concat_multi_camera='robotwin'` requires exactly 3 cameras, got {num_cameras}"
                )
            cam_top = transforms_F.resize(
                video[0], size=[256, 320], interpolation=transforms_F.InterpolationMode.BILINEAR, antialias=True
            )
            cam_left = transforms_F.resize(
                video[1], size=[128, 160], interpolation=transforms_F.InterpolationMode.BILINEAR, antialias=True
            )
            cam_right = transforms_F.resize(
                video[2], size=[128, 160], interpolation=transforms_F.InterpolationMode.BILINEAR, antialias=True
            )
            bottom = torch.cat([cam_left, cam_right], dim=-1)
            video = torch.cat([cam_top, bottom], dim=-2)
        elif num_cameras > 1:
            if self.concat_multi_camera == "horizontal":
                video = torch.cat([video[i] for i in range(num_cameras)], dim=-1)
            elif self.concat_multi_camera == "vertical":
                video = torch.cat([video[i] for i in range(num_cameras)], dim=-2)
            else:
                raise ValueError(f"Invalid concat_multi_camera: {self.concat_multi_camera}")
        else:
            video = video.squeeze(0)

        video = self.resize_transform(video)
        video = self.crop_transform(video)
        video = self.normalize_transform(video)
        video = video.permute(1, 0, 2, 3)
        return video

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

    def _get(self, idx: int) -> Dict[str, Any]:
        ep_local = idx // self.samples_per_episode
        ep = self._episodes[ep_local]
        trim = self._resolve_trim(ep_local)
        action_np, state_np, video_ts = self._load_npz_arrays(ep["npz_path"], trim)

        camera_videos = []
        for key in self.video_keys:
            video_path = ep["video_paths"][key]
            frames = self._decode_camera_video(video_path, video_ts)
            camera_videos.append(frames)

        video = self._concat_and_process_video(camera_videos)
        action, state = self._normalize_action_state(
            action_np,
            state_np,
            task=ep.get("task", ""),
            idx=idx,
        )
        proprio = state[:-1, :]

        if video.shape[1] <= 1:
            raise ValueError(f"`video` must have at least 2 frames, got {tuple(video.shape)}")
        if action.shape[0] != video.shape[1] - 1:
            if action.shape[0] % (video.shape[1] - 1) != 0:
                raise ValueError(
                    f"Action horizon {action.shape[0]} not aligned with video transitions "
                    f"{video.shape[1] - 1}"
                )

        task = ep.get("task", "")
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
            "image_is_pad": torch.zeros(trim.t_video, dtype=torch.bool),
            "action_is_pad": torch.zeros(trim.action_horizon, dtype=torch.bool),
            "proprio_is_pad": torch.zeros(trim.action_horizon, dtype=torch.bool),
            "everobot_episode_index": ep.get("episode_index", ep_local),
            "everobot_sample_in_episode": idx % self.samples_per_episode,
            "everobot_drop_start": trim.drop_start,
            "everobot_t_video": trim.t_video,
            "everobot_action_horizon": trim.action_horizon,
        }

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds {len(self)}.")
        try:
            return self._get(idx)
        except Exception as e:
            ep_local = idx // self.samples_per_episode
            logger.warning(
                "EveRobotFullEpisode: error loading idx=%d episode=%d: %s. Retrying episode 0.",
                idx,
                ep_local,
                e,
            )
            print(traceback.format_exc())
            return self._get(0)
