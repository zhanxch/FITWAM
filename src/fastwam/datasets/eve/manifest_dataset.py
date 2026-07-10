"""Manifest-driven EveRobot dataset adapter.

This adapter keeps the existing FastWAM sample contract unchanged while using
EveRobot sidecar manifests to decide which LeRobot episode windows are sampled.
"""

from __future__ import annotations

import json
import random
import traceback
from pathlib import Path
from typing import Any, Optional

import torch

from fastwam.datasets.lerobot.robot_video_dataset import RobotVideoDataset
from fastwam.utils.logging_config import get_logger


logger = get_logger(__name__)

def _load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).expanduser().open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_optional_json(path: str | Path) -> dict[str, Any] | None:
    path = Path(path).expanduser()
    if not path.exists():
        return None
    return _load_json(path)


def _resolve_root(path: str | Path) -> str:
    return str(Path(path).expanduser().resolve())


def _load_eve_action_schema(dataset_dirs: list[str]) -> dict[str, Any] | None:
    schemas: list[dict[str, Any]] = []
    for dataset_dir in dataset_dirs:
        schema = _load_optional_json(Path(dataset_dir) / "meta" / "eve" / "action_schema.json")
        if schema is not None:
            schemas.append(schema)
    if not schemas:
        return None
    first = schemas[0]
    for schema in schemas[1:]:
        if int(schema.get("policy_action_prefix_dim", 0)) != int(first.get("policy_action_prefix_dim", 0)):
            raise ValueError(f"Inconsistent EveRobot action schemas across dataset_dirs: {dataset_dirs}")
    return first


class EveManifestRobotVideoDataset(RobotVideoDataset):
    """FastWAM-compatible dataset driven by an EveRobot training manifest.

    The manifest contains episode/event units.  This dataset expands each unit
    into valid fixed-length windows and delegates actual frame loading,
    preprocessing, text-cache lookup, and normalization to ``RobotVideoDataset``.
    """

    def __init__(
        self,
        *args,
        manifest_path: str,
        dataset_dirs: Optional[list[str]] = None,
        manifest_splits: Optional[list[str]] = None,
        manifest_collection_iters: Optional[list[int]] = None,
        policy_action_prefix_dim: Optional[int] = None,
        event_sample_stride: Optional[int] = None,
        episode_sample_stride: Optional[int] = None,
        max_load_retry: int = 3,
        **kwargs,
    ):
        self.manifest_path = str(Path(manifest_path).expanduser().resolve())
        self.manifest = _load_json(self.manifest_path)
        if self.manifest.get("format") != "EveRobotTrainManifest":
            raise ValueError(
                f"Unsupported Eve manifest format: {self.manifest.get('format')}"
            )

        if dataset_dirs is None:
            dataset_dirs = list(self.manifest.get("dataset_roots", {}).values())
        if not dataset_dirs:
            raise ValueError("Eve manifest dataset requires at least one dataset root.")
        dataset_dirs = [_resolve_root(path) for path in dataset_dirs]
        self.eve_action_schema = _load_eve_action_schema(dataset_dirs)
        self.policy_action_prefix_dim = int(
            (self.eve_action_schema or {}).get("policy_action_prefix_dim", 0)
        )
        if policy_action_prefix_dim is not None and int(policy_action_prefix_dim) != self.policy_action_prefix_dim:
            raise ValueError(
                "policy_action_prefix_dim must come from meta/eve/action_schema.json; "
                f"config={policy_action_prefix_dim}, meta={self.policy_action_prefix_dim}"
            )

        # The manifest is the split/subset authority.  The underlying LeRobot
        # loader must include all referenced episodes so event windows can be
        # resolved exactly.
        self.requested_val_set_proportion = kwargs.pop("val_set_proportion", 0.0)
        kwargs["val_set_proportion"] = 0.0

        super().__init__(*args, dataset_dirs=dataset_dirs, **kwargs)

        self.manifest_splits = None if manifest_splits is None else set(manifest_splits)
        self.manifest_collection_iters = (
            None if manifest_collection_iters is None else {int(item) for item in manifest_collection_iters}
        )
        self.event_sample_stride = event_sample_stride
        self.episode_sample_stride = episode_sample_stride
        self.max_load_retry = int(max_load_retry)
        if self.max_load_retry < 0:
            raise ValueError(f"`max_load_retry` must be >= 0, got {max_load_retry}")

        self._episode_index = self._build_episode_index()
        self._samples = self._expand_manifest_samples()
        if not self._samples:
            raise ValueError(
                f"No trainable Eve windows found in manifest {self.manifest_path}. "
                f"Check event lengths and num_frames={self.num_frames}."
            )

        logger.info(
            "EveManifestRobotVideoDataset: manifest=%s units=%d windows=%d dataset_dirs=%d",
            self.manifest_path,
            len(self.manifest.get("samples", [])),
            len(self._samples),
            len(dataset_dirs),
        )

    def _build_episode_index(self) -> dict[tuple[str, int], tuple[int, int]]:
        episode_index: dict[tuple[str, int], tuple[int, int]] = {}
        frame_offset = 0
        for dataset in self.lerobot_dataset.multi_dataset._datasets:
            root = _resolve_root(dataset.root)
            ep_from = dataset.episode_data_index["from"].tolist()
            ep_to = dataset.episode_data_index["to"].tolist()
            for local_ep_pos, global_ep_idx in enumerate(dataset.episodes):
                start = int(ep_from[local_ep_pos]) + frame_offset
                end = int(ep_to[local_ep_pos]) + frame_offset
                episode_index[(root, int(global_ep_idx))] = (start, end - start)
            frame_offset += int(dataset.num_frames)
        return episode_index

    def _unit_stride(self, unit: dict[str, Any]) -> int:
        if "sample_stride" in unit and unit["sample_stride"] is not None:
            return max(int(unit["sample_stride"]), 1)
        if unit.get("sample_type") == "event" and self.event_sample_stride is not None:
            return max(int(self.event_sample_stride), 1)
        if unit.get("sample_type") == "episode" and self.episode_sample_stride is not None:
            return max(int(self.episode_sample_stride), 1)
        return 1

    def _include_unit(self, unit: dict[str, Any]) -> bool:
        if self.manifest_splits is not None and str(unit.get("split", "train")) not in self.manifest_splits:
            return False
        if self.manifest_collection_iters is not None:
            collection_iter = unit.get("collection_iter")
            if collection_iter is None:
                collection_iter = unit.get("collection_round")
            if collection_iter is None or int(collection_iter) not in self.manifest_collection_iters:
                return False
        return True

    def _expand_manifest_samples(self) -> list[dict[str, Any]]:
        expanded: list[dict[str, Any]] = []
        skipped_short = 0
        skipped_missing = 0
        for unit in self.manifest.get("samples", []):
            if not self._include_unit(unit):
                continue
            dataset_root = _resolve_root(unit["dataset_root"])
            episode_index = int(unit["episode_index"])
            ep_key = (dataset_root, episode_index)
            if ep_key not in self._episode_index:
                skipped_missing += 1
                logger.warning("Skipping missing Eve episode reference: %s", ep_key)
                continue

            global_ep_start, ep_length = self._episode_index[ep_key]
            start_frame = max(int(unit.get("start_frame", 0)), 0)
            end_frame = min(int(unit.get("end_frame", ep_length)), ep_length)
            if end_frame - start_frame < self.num_frames:
                skipped_short += 1
                continue

            max_start = end_frame - self.num_frames
            stride = self._unit_stride(unit)
            for window_start in range(start_frame, max_start + 1, stride):
                expanded.append(
                    {
                        "unit": unit,
                        "dataset_root": dataset_root,
                        "episode_index": episode_index,
                        "global_frame_idx": global_ep_start + window_start,
                        "window_start": window_start,
                        "window_end": window_start + self.num_frames,
                    }
                )

        if skipped_short:
            logger.warning(
                "Skipped %d Eve sample units shorter than num_frames=%d.",
                skipped_short,
                self.num_frames,
            )
        if skipped_missing:
            logger.warning("Skipped %d Eve sample units with missing episodes.", skipped_missing)
        return expanded

    def __len__(self) -> int:
        return len(self._samples)

    @staticmethod
    def _action_loss_weight(unit: dict[str, Any]) -> float:
        return 0.0 if unit.get("action_loss") == "disabled" else 1.0

    @staticmethod
    def _outcome_flag(unit: dict[str, Any]) -> int:
        outcome = unit.get("event_outcome", unit.get("episode_outcome", "success"))
        return 1 if outcome == "failure" else 0

    def _apply_action_loss_window(
        self,
        data: dict[str, Any],
        unit: dict[str, Any],
        window_start: int,
    ) -> None:
        action_loss_window = unit.get("action_loss_window")
        if action_loss_window is None:
            return
        if not isinstance(action_loss_window, list) or len(action_loss_window) != 2:
            raise ValueError(
                f"`action_loss_window` must be [start, end], got {action_loss_window}"
            )
        action_is_pad = data.get("action_is_pad")
        action = data.get("action")
        if action_is_pad is None or action is None:
            return
        action_len = int(action.shape[0])
        valid_start = int(action_loss_window[0])
        valid_end = int(action_loss_window[1])
        frame_ids = torch.arange(window_start, window_start + action_len)
        invalid = (frame_ids < valid_start) | (frame_ids >= valid_end)
        data["action_is_pad"] = action_is_pad.to(dtype=torch.bool).clone() | invalid.to(
            device=action_is_pad.device
        )

    def _get_eve(self, idx: int) -> dict[str, Any]:
        sample_ref = self._samples[idx]
        unit = sample_ref["unit"]
        data = self._get(sample_ref["global_frame_idx"])

        data["action_loss_weight"] = torch.tensor(
            self._action_loss_weight(unit), dtype=torch.float32
        )
        data["outcome_flag"] = torch.tensor(self._outcome_flag(unit), dtype=torch.long)
        self._apply_action_loss_window(data, unit, int(sample_ref["window_start"]))

        data["eve_manifest_path"] = self.manifest_path
        data["eve_sample_id"] = unit.get("sample_id", unit.get("event_id", ""))
        data["eve_sample_type"] = unit.get("sample_type", "")
        data["eve_dataset_id"] = unit.get("dataset_id", "")
        data["eve_collection_iter"] = int(unit.get("collection_iter", unit.get("collection_round", -1)))
        data["eve_episode_index"] = int(sample_ref["episode_index"])
        data["eve_window_start"] = int(sample_ref["window_start"])
        data["eve_window_end"] = int(sample_ref["window_end"])
        data["eve_event_outcome"] = unit.get("event_outcome", unit.get("episode_outcome", ""))
        data["eve_sample_role"] = unit.get("sample_role", "")
        return data

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds {len(self)}.")

        sample_idx = idx
        last_error: Exception | None = None
        for attempt in range(self.max_load_retry + 1):
            try:
                return self._get_eve(sample_idx)
            except Exception as err:
                last_error = err
                logger.warning(
                    "EveManifest failed to load idx=%d attempt=%d/%d: %s",
                    sample_idx,
                    attempt + 1,
                    self.max_load_retry + 1,
                    err,
                )
                print(traceback.format_exc())
                sample_idx = random.randrange(len(self._samples))
        raise RuntimeError(f"Failed to load Eve sample {idx}") from last_error
