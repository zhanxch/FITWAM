import traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, DefaultDict, Dict, List

import numpy as np
import torch
from tqdm import tqdm

from fastwam.utils.logging_config import get_logger
from .lerobot.lerobot_dataset import LeRobotDatasetMetadata, MultiLeRobotDataset
from .lerobot.datasets.video_utils import decode_video_frames
from .processors.base_processor import BaseProcessor

logger = get_logger(__name__)


class BaseLerobotEpisodeDataset(torch.utils.data.Dataset):
    """One training sample per episode with optional left padding."""

    def __init__(
        self,
        dataset_dirs: List[str],
        shape_meta: Dict[str, Any],
        num_frames: int = 33,
        action_stride: int = 10,
        val_set_proportion: float = 0.05,
        is_training_set: bool = False,
        seed: int = 42,
        left_pad: bool = True,
    ):
        if num_frames <= 1:
            raise ValueError(f"`num_frames` must be > 1, got {num_frames}")
        if action_stride <= 0:
            raise ValueError(f"`action_stride` must be positive, got {action_stride}")
        if (num_frames - 1) * action_stride <= 0:
            raise ValueError("Invalid combination of `num_frames` and `action_stride`.")

        self.dataset_dirs = dataset_dirs
        self.shape_meta = shape_meta
        self.num_frames = num_frames
        self.action_stride = action_stride
        self.action_horizon = (num_frames - 1) * action_stride
        self.left_pad = left_pad
        self.val_set_proportion = val_set_proportion
        self.is_training_set = is_training_set
        self.processor = None
        self.return_images = False

        self.image_meta = shape_meta["images"]
        self.state_meta = shape_meta["state"]
        self.action_meta = shape_meta["action"]
        for meta in self.image_meta:
            key = meta["key"]
            meta["lerobot_key"] = (
                f"observation.images.{key}" if key != "default" else "observation.images"
            )
        for meta in self.state_meta:
            key = meta["key"]
            meta["lerobot_key"] = (
                f"observation.state.{key}" if key != "default" else "observation.state"
            )
        for meta in self.action_meta:
            key = meta["key"]
            meta["lerobot_key"] = f"action.{key}" if key != "default" else "action"

        metas = []
        for ds_dir in dataset_dirs:
            ds_root = Path(ds_dir)
            metas.append(LeRobotDatasetMetadata(repo_id=ds_dir, root=ds_root))

        fps_list = [m.fps for m in metas]
        if len(set(fps_list)) != 1:
            raise ValueError(f"All dataset_dirs must have the same fps, got {fps_list}")
        self.fps = fps_list[0]

        self.video_subsampled = bool(metas[0].info.get("allinone", {}).get("video_subsampled", False))
        source_fps = metas[0].info.get("allinone", {}).get("source_fps")
        self.source_fps = int(source_fps) if source_fps is not None else self.fps

        episodes: dict[str, list[int]] = {}
        if val_set_proportion < 1e-6:
            for meta in metas:
                episodes.update({meta.repo_id: list(range(meta.total_episodes))})
        else:
            for meta in metas:
                split_idx = int(meta.total_episodes * (1 - val_set_proportion))
                episode_indices = list(range(meta.total_episodes))
                rng = np.random.default_rng(seed)
                rng.shuffle(episode_indices)
                if self.is_training_set:
                    selected = [episode_indices[i] for i in range(split_idx)]
                else:
                    selected = [episode_indices[i] for i in range(split_idx, meta.total_episodes)]
                episodes.update({meta.repo_id: selected})

        self.episode_indices = episodes
        self.multi_dataset = MultiLeRobotDataset(
            dataset_dirs=self.dataset_dirs,
            episodes=episodes,
            delta_timestamps=None,
        )

    def _resolve_episode(self, episode_idx: int) -> tuple[Any, int, int]:
        for dataset in self.multi_dataset._datasets:
            if episode_idx < dataset.num_episodes:
                global_ep_idx = dataset.episodes[episode_idx]
                return dataset, episode_idx, global_ep_idx
            episode_idx -= dataset.num_episodes
        raise IndexError(f"Episode index {episode_idx} out of bounds.")

    def _subsample_indices(self, length: int) -> list[int]:
        return list(range(0, length, self.action_stride))

    def _left_pad_with_reference(
        self,
        values: torch.Tensor,
        target_len: int,
        reference: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if values.shape[0] > target_len:
            values = values[-target_len:]
            is_pad = torch.zeros(target_len, dtype=torch.bool, device=values.device)
            return values, is_pad

        pad_len = target_len - values.shape[0]
        if pad_len <= 0:
            is_pad = torch.zeros(target_len, dtype=torch.bool, device=values.device)
            return values, is_pad

        if reference.shape[0] != 1:
            raise ValueError(f"`reference` must be a single-frame tensor, got shape {reference.shape}")
        pad_chunk = reference.expand(pad_len, *reference.shape[1:])
        padded = torch.cat([pad_chunk, values], dim=0)
        is_pad = torch.cat(
            [
                torch.ones(pad_len, dtype=torch.bool, device=values.device),
                torch.zeros(values.shape[0], dtype=torch.bool, device=values.device),
            ],
            dim=0,
        )
        return padded, is_pad

    def _left_pad_with_zeros(
        self,
        values: torch.Tensor,
        target_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if values.shape[0] > target_len:
            values = values[-target_len:]
            is_pad = torch.zeros(target_len, dtype=torch.bool, device=values.device)
            return values, is_pad

        pad_len = target_len - values.shape[0]
        if pad_len <= 0:
            is_pad = torch.zeros(target_len, dtype=torch.bool, device=values.device)
            return values, is_pad

        pad_chunk = torch.zeros(
            (pad_len, *values.shape[1:]),
            dtype=values.dtype,
            device=values.device,
        )
        padded = torch.cat([pad_chunk, values], dim=0)
        is_pad = torch.cat(
            [
                torch.ones(pad_len, dtype=torch.bool, device=values.device),
                torch.zeros(values.shape[0], dtype=torch.bool, device=values.device),
            ],
            dim=0,
        )
        return padded, is_pad

    def _left_pad_sequence(
        self,
        values: torch.Tensor,
        target_len: int,
        *,
        pad_value: float = 0.0,
        replicate_edge: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if replicate_edge:
            if values.shape[0] == 0:
                raise ValueError("Cannot replicate edge padding on an empty sequence.")
            return self._left_pad_with_reference(values, target_len, values[0:1])

        return self._left_pad_with_zeros(values, target_len) if pad_value == 0.0 else self._right_align_sequence(
            values, target_len, pad_value=pad_value
        )

    def _right_align_sequence(
        self,
        values: torch.Tensor,
        target_len: int,
        *,
        pad_value: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if values.shape[0] > target_len:
            values = values[:target_len]
            is_pad = torch.zeros(target_len, dtype=torch.bool, device=values.device)
            return values, is_pad

        pad_len = target_len - values.shape[0]
        if pad_len <= 0:
            is_pad = torch.zeros(target_len, dtype=torch.bool, device=values.device)
            return values, is_pad

        pad_chunk = torch.full(
            (pad_len, *values.shape[1:]),
            pad_value,
            dtype=values.dtype,
            device=values.device,
        )
        padded = torch.cat([pad_chunk, values], dim=0)
        is_pad = torch.cat(
            [
                torch.ones(pad_len, dtype=torch.bool, device=values.device),
                torch.zeros(values.shape[0], dtype=torch.bool, device=values.device),
            ],
            dim=0,
        )
        return padded, is_pad

    def _decode_episode_video(
        self,
        dataset,
        global_ep_idx: int,
        subsample_indices: list[int],
    ) -> torch.Tensor:
        video_key = self.image_meta[0]["lerobot_key"]
        video_path = dataset.root / dataset.meta.get_video_file_path(global_ep_idx, video_key)
        tolerance_s = dataset.tolerance_s

        if self.video_subsampled:
            num_video_frames = len(subsample_indices)
            timestamps = [i / float(self.fps) for i in range(num_video_frames)]
        else:
            timestamps = [idx / float(self.source_fps) for idx in subsample_indices]

        frames = decode_video_frames(video_path, timestamps, tolerance_s, dataset.video_backend)
        frames = (frames * 255).to(torch.uint8)
        return frames

    def _build_episode_sample(self, episode_idx: int) -> Dict[str, Any]:
        dataset, _, global_ep_idx = self._resolve_episode(episode_idx)
        episode_data = self.multi_dataset.get_episode_data(episode_idx)

        action_key = self.action_meta[0]["lerobot_key"]
        image_key = self.image_meta[0]["key"]

        states_by_key: dict[str, torch.Tensor] = {}
        for meta in self.state_meta:
            states = episode_data[meta["lerobot_key"]].float()
            if states.ndim == 1:
                states = states.unsqueeze(-1)
            states_by_key[meta["key"]] = states

        actions = episode_data[action_key].float()
        if actions.ndim == 1:
            actions = actions.unsqueeze(-1)

        first_state_key = self.state_meta[0]["key"]
        length = int(states_by_key[first_state_key].shape[0])
        for key, states in states_by_key.items():
            if int(states.shape[0]) != length:
                raise ValueError(
                    f"Episode {global_ep_idx}: state key {key} length {states.shape[0]} "
                    f"does not match {first_state_key} length {length}."
                )

        subsample_indices = self._subsample_indices(length)
        if len(subsample_indices) > self.num_frames:
            subsample_indices = subsample_indices[: self.num_frames]

        states_sub_by_key = {key: states[subsample_indices] for key, states in states_by_key.items()}
        images = self._decode_episode_video(dataset, global_ep_idx, subsample_indices)
        if images.ndim != 4:
            raise ValueError(
                f"Episode {global_ep_idx}: expected decoded video [T, C, H, W], got {tuple(images.shape)}."
            )
        if images.shape[0] != states_sub_by_key[first_state_key].shape[0]:
            raise ValueError(
                f"Episode {global_ep_idx}: decoded video frames ({images.shape[0]}) != "
                f"subsampled states ({states_sub_by_key[first_state_key].shape[0]})."
            )

        first_state_by_key = {key: states[0:1] for key, states in states_by_key.items()}
        first_frame = images[0:1]
        episode_actions = actions[: min(actions.shape[0], self.action_horizon)]

        if self.left_pad:
            # Left padding: replicate episode first frame for vision/state, zero actions for empty commands.
            states_padded_by_key = {}
            state_is_pad = None
            for key, states_sub in states_sub_by_key.items():
                states_padded, cur_state_is_pad = self._left_pad_with_reference(
                    states_sub, self.num_frames, first_state_by_key[key]
                )
                states_padded_by_key[key] = states_padded
                if state_is_pad is None:
                    state_is_pad = cur_state_is_pad
                elif not torch.equal(state_is_pad, cur_state_is_pad):
                    raise ValueError(f"Episode {global_ep_idx}: inconsistent state padding for key {key}.")
            images_padded, image_is_pad = self._left_pad_with_reference(
                images, self.num_frames, first_frame
            )
            actions_padded, action_is_pad = self._left_pad_with_zeros(
                episode_actions, self.action_horizon
            )
        else:
            states_padded_by_key = {}
            state_is_pad = None
            for key, states_sub in states_sub_by_key.items():
                states_padded, cur_state_is_pad = self._right_align_sequence(states_sub, self.num_frames)
                states_padded_by_key[key] = states_padded
                if state_is_pad is None:
                    state_is_pad = cur_state_is_pad
                elif not torch.equal(state_is_pad, cur_state_is_pad):
                    raise ValueError(f"Episode {global_ep_idx}: inconsistent state padding for key {key}.")
            images_padded, image_is_pad = self._right_align_sequence(images, self.num_frames)
            actions_padded, action_is_pad = self._right_align_sequence(
                episode_actions, self.action_horizon
            )
        if state_is_pad is None:
            raise ValueError(f"Episode {global_ep_idx}: no state fields were loaded.")

        task_idx = int(episode_data["task_index"][0].item())
        task = dataset.meta.tasks[task_idx]

        sample = {
            "idx": episode_idx,
            "task": task,
            "action": {self.action_meta[0]["key"]: actions_padded},
            "state": states_padded_by_key,
            "images": {image_key: images_padded},
            "action_is_pad": action_is_pad,
            "state_is_pad": state_is_pad,
            "image_is_pad": image_is_pad,
        }
        return sample

    def _set_return_images(self, flag: bool):
        self.return_images = flag

    def __len__(self):
        return self.multi_dataset.num_episodes

    def __getitem__(self, episode_idx: int):
        if episode_idx >= len(self):
            raise IndexError(f"Episode index {episode_idx} out of bounds {len(self)}.")

        try:
            sample = self._build_episode_sample(episode_idx)
        except Exception as err:
            logger.warning(f"Error loading episode {episode_idx}: {err}")
            print(traceback.format_exc())
            raise

        if self.processor is not None:
            sample = self.processor.preprocess(sample)
        return sample

    def set_processor(self, processor: BaseProcessor):
        self.processor = processor
        if self.is_training_set:
            self.processor.train()
        else:
            self.processor.eval()
        return self

    def get_dataset_stats(self, preprocessor: BaseProcessor):
        state_min: DefaultDict[str, list] = defaultdict(list)
        state_max: DefaultDict[str, list] = defaultdict(list)
        state_mean: DefaultDict[str, list] = defaultdict(list)
        state_var: DefaultDict[str, list] = defaultdict(list)
        state_q01: DefaultDict[str, list] = defaultdict(list)
        state_q99: DefaultDict[str, list] = defaultdict(list)

        action_min: DefaultDict[str, list] = defaultdict(list)
        action_max: DefaultDict[str, list] = defaultdict(list)
        action_mean: DefaultDict[str, list] = defaultdict(list)
        action_var: DefaultDict[str, list] = defaultdict(list)
        action_q01: DefaultDict[str, list] = defaultdict(list)
        action_q99: DefaultDict[str, list] = defaultdict(list)

        episodes_num = self.multi_dataset.num_episodes

        def process_episode(episode_idx: int):
            episode_data = self.multi_dataset.get_episode_data(episode_idx)
            batch = {
                "state": {
                    meta["key"]: episode_data[meta["lerobot_key"]].float()
                    for meta in self.state_meta
                },
                "action": {
                    self.action_meta[0]["key"]: episode_data[self.action_meta[0]["lerobot_key"]].float()
                },
            }
            return preprocessor.action_state_transform(batch)

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_episode, idx) for idx in range(episodes_num)]
            for future in tqdm(as_completed(futures), total=episodes_num, desc="Iterating episodes for normalization"):
                batch = future.result()
                for meta in self.state_meta:
                    key = meta["key"]
                    cur_state = batch["state"][key]
                    if cur_state.ndim == 1:
                        cur_state = cur_state.unsqueeze(-1)
                    state_min[key].append(cur_state.amin(0))
                    state_max[key].append(cur_state.amax(0))
                    state_mean[key].append(cur_state.mean(0))
                    state_var[key].append(cur_state.var(0))
                    state_q01[key].append(torch.quantile(cur_state, 0.01, dim=0, keepdim=False))
                    state_q99[key].append(torch.quantile(cur_state, 0.99, dim=0, keepdim=False))
                for meta in self.action_meta:
                    key = meta["key"]
                    cur_action = batch["action"][key]
                    if cur_action.ndim == 1:
                        cur_action = cur_action.unsqueeze(-1)
                    action_min[key].append(cur_action.amin(0))
                    action_max[key].append(cur_action.amax(0))
                    action_mean[key].append(cur_action.mean(0))
                    action_var[key].append(cur_action.var(0))
                    action_q01[key].append(torch.quantile(cur_action, 0.01, dim=0, keepdim=False))
                    action_q99[key].append(torch.quantile(cur_action, 0.99, dim=0, keepdim=False))

        def get_mean_std(means, vars_):
            means = torch.stack(means)
            vars_ = torch.stack(vars_)
            stepwise_mean = means.mean(0)
            stepwise_std = (vars_ + (means - stepwise_mean) ** 2).mean(0).sqrt()
            global_mean = means.mean((0, 1))
            global_std = (vars_ + (means - global_mean) ** 2).mean((0, 1)).sqrt()
            return stepwise_mean, stepwise_std, global_mean, global_std

        stats = {
            "state": defaultdict(dict),
            "action": defaultdict(dict),
            "num_episodes": episodes_num,
            "num_transition": episodes_num,
        }
        for meta in self.state_meta:
            key = meta["key"]
            stats["state"][key]["stepwise_min"] = torch.stack(state_min[key]).amin(0)
            stats["state"][key]["stepwise_max"] = torch.stack(state_max[key]).amax(0)
            stats["state"][key]["global_min"] = stats["state"][key]["stepwise_min"].amin(0)
            stats["state"][key]["global_max"] = stats["state"][key]["stepwise_max"].amax(0)
            stats["state"][key]["stepwise_q01"] = torch.stack(state_q01[key]).amin(0)
            stats["state"][key]["stepwise_q99"] = torch.stack(state_q99[key]).amax(0)
            stats["state"][key]["global_q01"] = stats["state"][key]["stepwise_q01"].amin(0)
            stats["state"][key]["global_q99"] = stats["state"][key]["stepwise_q99"].amax(0)
            (
                stats["state"][key]["stepwise_mean"],
                stats["state"][key]["stepwise_std"],
                stats["state"][key]["global_mean"],
                stats["state"][key]["global_std"],
            ) = get_mean_std(state_mean[key], state_var[key])

        for meta in self.action_meta:
            key = meta["key"]
            stats["action"][key]["stepwise_min"] = torch.stack(action_min[key]).amin(0)
            stats["action"][key]["stepwise_max"] = torch.stack(action_max[key]).amax(0)
            stats["action"][key]["global_min"] = stats["action"][key]["stepwise_min"].amin(0)
            stats["action"][key]["global_max"] = stats["action"][key]["stepwise_max"].amax(0)
            stats["action"][key]["stepwise_q01"] = torch.stack(action_q01[key]).amin(0)
            stats["action"][key]["stepwise_q99"] = torch.stack(action_q99[key]).amax(0)
            stats["action"][key]["global_q01"] = stats["action"][key]["stepwise_q01"].amin(0)
            stats["action"][key]["global_q99"] = stats["action"][key]["stepwise_q99"].amax(0)
            (
                stats["action"][key]["stepwise_mean"],
                stats["action"][key]["stepwise_std"],
                stats["action"][key]["global_mean"],
                stats["action"][key]["global_std"],
            ) = get_mean_std(action_mean[key], action_var[key])

        return stats
