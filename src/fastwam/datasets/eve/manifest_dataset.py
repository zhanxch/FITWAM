"""Manifest-driven EveRobot dataset adapter.

This adapter keeps the existing FastWAM sample contract unchanged while using
EveRobot sidecar manifests to decide which LeRobot episode windows are sampled.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import torch

from fastwam.datasets.lerobot.robot_video_dataset import RobotVideoDataset
from fastwam.everobot_schema import (
    resolve_manifest_dataset_root,
    validate_manifest,
)
from fastwam.utils.logging_config import get_logger


logger = get_logger(__name__)


_VIDEO_DECODE_RUNTIME_MARKERS = (
    "decode video",
    "decoding video",
    "failed to decode",
    "video decode",
    "video decoder",
    "video stream",
)


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
        dataset_root_overrides: dict[str, str] | None = None,
        strict_manifest_references: bool = True,
        verify_manifest_hash: bool = True,
        policy_action_prefix_dim: Optional[int] = None,
        global_sample_stride: int = 1,
        event_sample_stride: Optional[int] = None,
        episode_sample_stride: Optional[int] = None,
        max_load_retry: int = 3,
        **kwargs,
    ):
        self.manifest_path = str(Path(manifest_path).expanduser().resolve())
        self.manifest = _load_json(self.manifest_path)
        self.dataset_root_overrides = dict(dataset_root_overrides or {})
        self.strict_manifest_references = bool(strict_manifest_references)
        self.manifest_splits = None if manifest_splits is None else set(manifest_splits)
        self.manifest_collection_iters = (
            None
            if manifest_collection_iters is None
            else {int(item) for item in manifest_collection_iters}
        )
        validate_manifest(
            self.manifest,
            strict=True,
            verify_hash=bool(verify_manifest_hash),
        )
        configured_dataset_dirs = dataset_dirs
        dataset_dirs = []
        seen_roots: set[str] = set()
        for unit in self.manifest.get("samples", []):
            if not self._include_unit(unit):
                continue
            root = self._resolve_unit_dataset_root(unit)
            if root not in seen_roots:
                dataset_dirs.append(root)
                seen_roots.add(root)
        if configured_dataset_dirs is not None:
            configured_roots = {_resolve_root(path) for path in configured_dataset_dirs}
            if configured_roots != seen_roots:
                logger.warning(
                    "Ignoring dataset_dirs that disagree with selected manifest roots; "
                    "use dataset_root_overrides to relocate EveRobot datasets."
                )
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
        self.global_sample_stride = int(global_sample_stride)

        super().__init__(
            *args,
            dataset_dirs=dataset_dirs,
            global_sample_stride=self.global_sample_stride,
            **kwargs,
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
        self.sampling_roles = tuple(
            self._sampling_role(sample["unit"]) for sample in self._samples
        )
        role_names = tuple(dict.fromkeys(self.sampling_roles))
        self._sampling_role_indices = {
            role: tuple(
                index
                for index, sample_role in enumerate(self.sampling_roles)
                if sample_role == role
            )
            for role in role_names
        }

        logger.info(
            "EveManifestRobotVideoDataset: manifest=%s units=%d windows=%d "
            "roles=%s dataset_dirs=%d",
            self.manifest_path,
            len(self.manifest.get("samples", [])),
            len(self._samples),
            {
                role: len(indices)
                for role, indices in self._sampling_role_indices.items()
            },
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

    def _resolve_unit_dataset_root(self, unit: dict[str, Any]) -> str:
        root = resolve_manifest_dataset_root(
            self.manifest,
            unit,
            self.dataset_root_overrides,
        )
        return _resolve_root(root)

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

    @staticmethod
    def _core_start_anchor(
        unit: dict[str, Any],
        window_starts: set[int],
    ) -> int:
        anchor_frame = int(
            unit.get("core_start_frame", unit.get("start_frame", 0))
        )
        return min(window_starts, key=lambda start: (abs(start - anchor_frame), start))

    def _expand_manifest_samples(self) -> list[dict[str, Any]]:
        expanded: list[dict[str, Any]] = []
        skipped_short = 0
        skipped_missing = 0
        for unit in self.manifest.get("samples", []):
            if not self._include_unit(unit):
                continue
            dataset_root = self._resolve_unit_dataset_root(unit)
            episode_index = int(unit["episode_index"])
            ep_key = (dataset_root, episode_index)
            if ep_key not in self._episode_index:
                if self.strict_manifest_references:
                    raise ValueError(
                        "Eve manifest references an episode absent from the loaded "
                        f"datasets: dataset_root={dataset_root!r}, "
                        f"episode_index={episode_index}. Set "
                        "strict_manifest_references=False to warn and skip it."
                    )
                skipped_missing += 1
                logger.warning("Skipping missing Eve episode reference: %s", ep_key)
                continue

            global_ep_start, ep_length = self._episode_index[ep_key]
            source_span = (self.num_frames - 1) * int(self.global_sample_stride) + 1
            unit_start = max(int(unit.get("start_frame", 0)), 0)
            unit_end = min(int(unit.get("end_frame", ep_length)), ep_length)
            intervals = unit.get("valid_intervals") or [[unit_start, unit_end]]
            stride = self._unit_stride(unit)
            unit_window_starts: set[int] = set()
            for interval_start, interval_end in intervals:
                start_frame = max(int(interval_start), unit_start)
                end_frame = min(int(interval_end), unit_end)
                if end_frame - start_frame < source_span:
                    continue
                max_start = end_frame - source_span
                for window_start in range(start_frame, max_start + 1, stride):
                    unit_window_starts.add(window_start)

            if not unit_window_starts:
                skipped_short += 1
                continue
            window_selection = unit.get("window_selection")
            if window_selection not in {None, "", "core_start_anchor"}:
                raise ValueError(
                    "Unsupported Eve manifest window_selection "
                    f"{window_selection!r} for sample "
                    f"{unit.get('sample_id', unit.get('event_id', '<unknown>'))!r}."
                )
            if window_selection == "core_start_anchor":
                window_starts = [
                    self._core_start_anchor(unit, unit_window_starts)
                ]
            else:
                window_starts = sorted(unit_window_starts)
            for window_start in window_starts:
                expanded.append(
                    {
                        "unit": unit,
                        "dataset_root": dataset_root,
                        "episode_index": episode_index,
                        "global_frame_idx": global_ep_start + window_start,
                        "window_start": window_start,
                        "window_end": window_start + source_span,
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
    def _effective_outcome(unit: dict[str, Any]) -> str:
        event_outcome = unit.get("event_outcome")
        if event_outcome in {None, "", "unknown"}:
            return str(unit.get("episode_outcome") or "success")
        return str(event_outcome)

    @classmethod
    def _outcome_flag(cls, unit: dict[str, Any]) -> int:
        return 1 if cls._effective_outcome(unit) == "failure" else 0

    @classmethod
    def _sampling_role(cls, unit: dict[str, Any]) -> str:
        """Return the stable binary sampling role for every expanded window."""

        outcome = cls._effective_outcome(unit)
        is_failure = (
            unit.get("episode_outcome") == "failure"
            or unit.get("event_outcome") == "failure"
        )
        explicit_role = unit.get("batch_role")
        if explicit_role not in {None, "", "primary", "auxiliary"}:
            raise ValueError(
                "`batch_role` must be `primary` or `auxiliary`, got "
                f"{explicit_role!r}."
            )
        if is_failure:
            if explicit_role == "primary":
                raise ValueError(
                    "Failure manifest units cannot use `batch_role=primary`."
                )
            return "auxiliary"
        if explicit_role == "auxiliary" or (
            explicit_role in {None, ""} and cls._action_loss_weight(unit) <= 0.0
        ):
            return "auxiliary_success"
        if explicit_role == "primary":
            return "primary"
        if outcome == "success" and cls._action_loss_weight(unit) > 0.0:
            return "primary"
        return "auxiliary_success"

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
        frame_ids = window_start + torch.arange(action_len) * int(
            self.global_sample_stride
        )
        invalid = (frame_ids < valid_start) | (frame_ids >= valid_end)
        data["action_is_pad"] = action_is_pad.to(dtype=torch.bool).clone() | invalid.to(
            device=action_is_pad.device
        )

    def _get_eve(self, idx: int) -> dict[str, Any]:
        sample_ref = self._samples[idx]
        unit = sample_ref["unit"]
        outcome_flag = self._outcome_flag(unit)
        sample_id = str(unit.get("sample_id", unit.get("event_id", "")))
        window_start = int(sample_ref["window_start"])
        skip_video = bool(getattr(self, "force_skip_video", False))
        if (
            not skip_video
            and getattr(self, "vae_latent_cache_dir", None) is not None
            and getattr(self, "drop_video_when_latents_cached", False)
        ):
            from fastwam.datasets.vae_latent_cache import vae_latent_cache_path

            skip_video = vae_latent_cache_path(
                self.vae_latent_cache_dir,
                sample_id=sample_id,
                window_start=window_start,
            ).exists()
        data = self._get(
            sample_ref["global_frame_idx"],
            outcome_flag_override=outcome_flag,
            skip_video=skip_video,
        )

        data["action_loss_weight"] = torch.tensor(
            self._action_loss_weight(unit), dtype=torch.float32
        )
        data["outcome_flag"] = torch.tensor(outcome_flag, dtype=torch.long)
        event_weight = unit.get("event_weight")
        pair_weight = unit.get("pair_weight")
        data["event_weight"] = torch.tensor(
            1.0 if event_weight is None else float(event_weight), dtype=torch.float32
        )
        data["pair_weight"] = torch.tensor(
            0.0 if pair_weight is None else float(pair_weight), dtype=torch.float32
        )
        data["pair_id"] = str(unit.get("pair_id") or "")
        self._apply_action_loss_window(data, unit, window_start)

        data["eve_manifest_path"] = self.manifest_path
        data["eve_sample_id"] = sample_id
        data["eve_sample_type"] = unit.get("sample_type", "")
        data["eve_dataset_id"] = unit.get("dataset_id", "")
        data["eve_collection_iter"] = int(unit.get("collection_iter", unit.get("collection_round", -1)))
        data["eve_episode_index"] = int(sample_ref["episode_index"])
        data["eve_window_start"] = window_start
        data["eve_window_end"] = int(sample_ref["window_end"])
        data["eve_event_outcome"] = self._effective_outcome(unit)
        data["eve_sample_role"] = unit.get("sample_role", "")
        sampling_roles = getattr(self, "sampling_roles", None)
        data["eve_batch_role"] = (
            sampling_roles[idx]
            if sampling_roles is not None
            else self._sampling_role(unit)
        )
        data = self._maybe_attach_vae_latents(
            data,
            sample_id=sample_id,
            window_start=window_start,
        )
        return data

    @staticmethod
    def _is_video_decode_runtime_error(error: RuntimeError) -> bool:
        module = type(error).__module__.lower()
        if module.startswith(("av.", "torchcodec.", "torchvision.io")):
            return True
        message = str(error).lower()
        return any(marker in message for marker in _VIDEO_DECODE_RUNTIME_MARKERS)

    def _deterministic_retry_index(
        self,
        requested_idx: int,
        retry_number: int,
    ) -> int:
        requested_role = self.sampling_roles[requested_idx]
        role_indices = self._sampling_role_indices[requested_role]
        try:
            position = role_indices.index(requested_idx)
        except ValueError as error:
            raise RuntimeError(
                f"Sample {requested_idx} is absent from role index "
                f"{requested_role!r}."
            ) from error
        return role_indices[(position + retry_number) % len(role_indices)]

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds {len(self)}.")

        sample_idx = idx
        last_error: Exception | None = None
        for attempt in range(self.max_load_retry + 1):
            try:
                return self._get_eve(sample_idx)
            except (OSError, EOFError) as err:
                last_error = err
            except RuntimeError as err:
                if not self._is_video_decode_runtime_error(err):
                    raise
                last_error = err

            if attempt >= self.max_load_retry:
                break
            next_idx = self._deterministic_retry_index(idx, attempt + 1)
            logger.warning(
                "EveManifest I/O/decode failure at idx=%d attempt=%d/%d; "
                "retrying deterministic same-role idx=%d: %s",
                sample_idx,
                attempt + 1,
                self.max_load_retry + 1,
                next_idx,
                last_error,
            )
            sample_idx = next_idx
        raise RuntimeError(f"Failed to load Eve sample {idx}") from last_error
