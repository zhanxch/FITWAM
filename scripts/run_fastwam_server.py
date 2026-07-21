#!/usr/bin/env python3
"""Launch a FastWAM inference policy server (ZMQ API, training-aligned I/O)."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import signal
import sys
import threading
from pathlib import Path
from typing import Any

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_ROOT, SCRIPTS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from fastwam_policy_server import DEFAULT_SERVER_PORT, PolicyServer
from policy_io import KEY_ACTION, KEY_CONTEXT, KEY_CONTEXT_MASK, KEY_INPUT_IMAGE, KEY_PROMPT, KEY_PROPRIO
from policy_io import to_inference_tensors, validate_policy_observation


STEER_INFERENCE_MODES = ("learned", "bypass", "cached")
STEER_CACHE_SCHEMA_VERSION = 2
STEER_PROTOCOL_SCHEMA_VERSION = 1
STEER_CACHE_COVERAGE_OBSERVED = "observed_contiguous"
STEER_CACHE_COVERAGE_FULL = "full_horizon"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_sha256(value: str, *, label: str) -> str:
    normalized = str(value).strip().lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{label} must be a 64-character hexadecimal SHA256.")
    return normalized


def _embedding_sha256(embedding: torch.Tensor) -> str:
    canonical = embedding.detach().to(device="cpu", dtype=torch.float32).contiguous()
    return hashlib.sha256(canonical.numpy().tobytes(order="C")).hexdigest()


def _canonical_json(payload: Any) -> str:
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )


def _json_sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _normalize_steer_protocol(protocol: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """Validate and canonicalize the single-task, single-client cache protocol."""

    if not isinstance(protocol, dict):
        raise ValueError("Steer protocol must be a JSON object.")
    canonical = json.loads(_canonical_json(protocol))
    expected_scalars = {
        "schema": "fastwam.steer_protocol",
        "schema_version": STEER_PROTOCOL_SCHEMA_VERSION,
    }
    for key, expected in expected_scalars.items():
        if canonical.get(key) != expected:
            raise ValueError(
                f"Steer protocol {key} must be {expected!r}, got {canonical.get(key)!r}."
            )
    task = canonical.get("task")
    if not isinstance(task, str) or not task.strip():
        raise ValueError("Steer protocol requires one non-empty task name.")

    seeds = canonical.get("environment_seeds")
    episodes = canonical.get("episodes")
    inference = canonical.get("inference")
    options = canonical.get("environment_options")
    model = canonical.get("model")
    for name, value in (
        ("environment_seeds", seeds),
        ("episodes", episodes),
        ("inference", inference),
        ("environment_options", options),
        ("model", model),
    ):
        if not isinstance(value, dict):
            raise ValueError(f"Steer protocol {name} must be an object.")

    integer_fields = (
        (seeds, "global_base"),
        (seeds, "global_end_exclusive"),
        (seeds, "shard_base"),
        (seeds, "shard_end_exclusive"),
        (episodes, "global_start"),
        (episodes, "global_end_exclusive"),
        (episodes, "shard_global_start"),
        (episodes, "shard_global_end_exclusive"),
        (episodes, "local_start"),
        (episodes, "local_end_exclusive"),
        (episodes, "shard_id"),
        (inference, "replan_steps"),
        (inference, "max_env_steps"),
        (inference, "max_requests_per_episode"),
    )
    for container, key in integer_fields:
        value = container.get(key)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"Steer protocol field {key} must be an integer.")
    inference_seed = inference.get("seed")
    if inference_seed is not None and (
        isinstance(inference_seed, bool) or not isinstance(inference_seed, int)
    ):
        raise ValueError("Steer protocol inference.seed must be an integer or null.")
    if inference.get("control_mode") != "blocking":
        raise ValueError(
            "Steer cache replay/recording currently requires blocking control mode."
        )
    if inference.get("async_fallback") not in ("wait", "hold_last"):
        raise ValueError("Steer protocol inference.async_fallback is invalid.")
    for key in ("action_horizon_override", "num_inference_steps_override"):
        value = inference.get(key)
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
        ):
            raise ValueError(f"Steer protocol inference.{key} must be positive or null.")

    if episodes["global_start"] != 0 or episodes["local_start"] != 0:
        raise ValueError("Steer protocol global/local episode ranges must start at zero.")
    local_count = episodes["local_end_exclusive"]
    shard_count = episodes["shard_global_end_exclusive"] - episodes["shard_global_start"]
    seed_count = seeds["shard_end_exclusive"] - seeds["shard_base"]
    if local_count <= 0 or local_count != shard_count or local_count != seed_count:
        raise ValueError("Steer protocol shard episode and seed ranges must have equal positive size.")
    if seeds["global_end_exclusive"] - seeds["global_base"] != episodes["global_end_exclusive"]:
        raise ValueError("Steer protocol global seed and episode ranges must have equal size.")
    if seeds["shard_base"] != seeds["global_base"] + episodes["shard_global_start"]:
        raise ValueError("Steer protocol shard seed base does not match its global episode offset.")
    if inference["replan_steps"] <= 0 or inference["max_env_steps"] <= 0:
        raise ValueError("Steer protocol replan_steps and max_env_steps must be positive.")
    expected_requests = math.ceil(
        inference["max_env_steps"] / inference["replan_steps"]
    )
    if inference["max_requests_per_episode"] != expected_requests:
        raise ValueError(
            "Steer protocol max_requests_per_episode must equal "
            "ceil(max_env_steps / replan_steps)."
        )
    for key in ("randomize", "randomize_dynamics", "action_clip"):
        if not isinstance(options.get(key), bool):
            raise ValueError(f"Steer protocol environment_options.{key} must be boolean.")
    for key in ("clip_max_xyz_step", "clip_max_dz_down"):
        value = options.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"Steer protocol environment_options.{key} must be finite.")
    if not isinstance(options.get("task_config_dir"), str) or not options["task_config_dir"]:
        raise ValueError("Steer protocol environment_options.task_config_dir is required.")
    for key in ("checkpoint_path", "config_path"):
        if not isinstance(model.get(key), str) or not model[key]:
            raise ValueError(f"Steer protocol model.{key} must be a non-empty path.")
    for key in ("checkpoint_sha256", "config_sha256"):
        model[key] = _normalize_sha256(model.get(key, ""), label=f"protocol model.{key}")
    return canonical, _json_sha256(canonical)


def _parse_steer_protocol_json(value: str | None) -> tuple[dict[str, Any], str]:
    if value is None:
        raise ValueError(
            "Steer cache replay/recording requires --steer-protocol-json from the eval orchestrator."
        )
    try:
        payload = json.loads(value)
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid --steer-protocol-json: {error}") from error
    return _normalize_steer_protocol(payload)


def _cache_keyset_sha256(keys: set[tuple[int, int]]) -> str:
    return _json_sha256(
        [{"episode": episode, "request": request} for episode, request in sorted(keys)]
    )


def _validate_cache_coverage(
    keys: set[tuple[int, int]],
    *,
    protocol: dict[str, Any],
    coverage_policy: str,
) -> dict[str, int]:
    episodes = protocol["episodes"]
    inference = protocol["inference"]
    local_episode_ids = range(
        int(episodes["local_start"]),
        int(episodes["local_end_exclusive"]),
    )
    request_counts: dict[str, int] = {}
    expected_keys: set[tuple[int, int]] = set()
    for episode in local_episode_ids:
        requests = sorted(request for ep, request in keys if ep == episode)
        if not requests:
            raise ValueError(f"Steer cache is missing episode {episode}.")
        if requests != list(range(requests[-1] + 1)):
            raise ValueError(f"Steer cache episode {episode} has non-contiguous requests.")
        request_counts[str(episode)] = len(requests)
        if coverage_policy == STEER_CACHE_COVERAGE_FULL:
            expected_keys.update(
                (episode, request)
                for request in range(int(inference["max_requests_per_episode"]))
            )
    valid_episode_ids = set(local_episode_ids)
    unexpected_episodes = sorted({episode for episode, _ in keys} - valid_episode_ids)
    if unexpected_episodes:
        raise ValueError(f"Steer cache contains out-of-range episodes {unexpected_episodes}.")
    if coverage_policy == STEER_CACHE_COVERAGE_FULL and keys != expected_keys:
        missing = len(expected_keys - keys)
        extra = len(keys - expected_keys)
        raise ValueError(
            "Steer cache full_horizon coverage mismatch: "
            f"missing={missing}, extra={extra}."
        )
    if coverage_policy not in (
        STEER_CACHE_COVERAGE_OBSERVED,
        STEER_CACHE_COVERAGE_FULL,
    ):
        raise ValueError(f"Unknown steer cache coverage_policy {coverage_policy!r}.")
    return request_counts


class SteerEmbeddingCache:
    """Complete protocol-bound JSONL cache for deterministic steer replay."""

    def __init__(
        self,
        path: Path,
        *,
        expected_file_sha256: str,
        checkpoint_sha256: str,
        config_sha256: str,
        embedding_dim: int,
        protocol: dict[str, Any],
        required_coverage_policy: str = STEER_CACHE_COVERAGE_FULL,
    ) -> None:
        expected_file_sha256 = _normalize_sha256(
            expected_file_sha256,
            label="expected_file_sha256",
        )
        actual_file_sha256 = _sha256_file(path)
        if actual_file_sha256 != expected_file_sha256:
            raise ValueError(
                f"Steer cache SHA256 mismatch: expected {expected_file_sha256}, "
                f"got {actual_file_sha256}."
            )
        self.path = path
        self.file_sha256 = actual_file_sha256
        self.entries: dict[tuple[int, int], torch.Tensor] = {}
        expected_protocol, expected_protocol_sha256 = _normalize_steer_protocol(protocol)
        with path.open("r", encoding="utf-8") as stream:
            lines = [line for line in stream if line.strip()]
        if not lines:
            raise ValueError(f"Steer cache is empty: {path}")
        header = json.loads(lines[0])
        expected_header = {
            "type": "header",
            "schema_version": STEER_CACHE_SCHEMA_VERSION,
            "checkpoint_sha256": checkpoint_sha256,
            "config_sha256": config_sha256,
            "embedding_dim": int(embedding_dim),
            "protocol": expected_protocol,
            "protocol_sha256": expected_protocol_sha256,
            "coverage_policy": required_coverage_policy,
        }
        for key, expected in expected_header.items():
            if header.get(key) != expected:
                raise ValueError(
                    f"Steer cache header mismatch for {key}: expected {expected!r}, "
                    f"got {header.get(key)!r}."
                )
        footer = json.loads(lines[-1])
        if footer.get("type") != "footer":
            raise ValueError("Steer cache is incomplete: missing completion footer.")
        if footer.get("complete") is not True:
            raise ValueError("Steer cache completion footer is not complete=true.")
        for line_number, line in enumerate(lines[1:-1], start=2):
            entry = json.loads(line)
            if entry.get("type") != "entry":
                raise ValueError(
                    f"Steer cache line {line_number} must have type='entry'."
                )
            episode = int(entry.get("episode", -1))
            request = int(entry.get("request", -1))
            if episode < 0 or request < 0:
                raise ValueError(
                    f"Steer cache line {line_number} has a negative episode/request key."
                )
            key = (episode, request)
            if key in self.entries:
                raise ValueError(f"Duplicate steer cache key {key} at line {line_number}.")
            embedding = torch.tensor(entry.get("embedding"), dtype=torch.float32)
            if embedding.ndim != 1 or embedding.shape[0] != embedding_dim:
                raise ValueError(
                    f"Steer cache key {key} must have shape [{embedding_dim}], "
                    f"got {tuple(embedding.shape)}."
                )
            if not torch.isfinite(embedding).all():
                raise ValueError(f"Steer cache key {key} contains non-finite values.")
            actual_embedding_sha256 = _embedding_sha256(embedding)
            if entry.get("embedding_sha256") != actual_embedding_sha256:
                raise ValueError(
                    f"Steer cache key {key} embedding SHA256 mismatch."
                )
            self.entries[key] = embedding
        if not self.entries:
            raise ValueError(f"Steer cache contains no entries: {path}")
        keys = set(self.entries)
        request_counts = _validate_cache_coverage(
            keys,
            protocol=expected_protocol,
            coverage_policy=required_coverage_policy,
        )
        expected_footer = {
            "schema_version": STEER_CACHE_SCHEMA_VERSION,
            "protocol_sha256": expected_protocol_sha256,
            "coverage_policy": required_coverage_policy,
            "entry_count": len(keys),
            "episode_request_counts": request_counts,
            "keyset_sha256": _cache_keyset_sha256(keys),
        }
        for key, expected in expected_footer.items():
            if footer.get(key) != expected:
                raise ValueError(
                    f"Steer cache footer mismatch for {key}: expected {expected!r}, "
                    f"got {footer.get(key)!r}."
                )
        self.protocol = expected_protocol
        self.protocol_sha256 = expected_protocol_sha256
        self.coverage_policy = required_coverage_policy

    def get(self, episode: int, request: int) -> torch.Tensor:
        key = (int(episode), int(request))
        if key not in self.entries:
            raise KeyError(f"Missing steer cache entry for episode/request {key}.")
        return self.entries[key].unsqueeze(0).clone()


class SteerEmbeddingRecorder:
    """Record complete observed request keys and seal them with a footer."""

    def __init__(
        self,
        path: Path,
        *,
        checkpoint_sha256: str,
        config_sha256: str,
        embedding_dim: int,
        protocol: dict[str, Any],
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.embedding_dim = int(embedding_dim)
        self.protocol, self.protocol_sha256 = _normalize_steer_protocol(protocol)
        self._stream = path.open("x", encoding="utf-8")
        self._keys: set[tuple[int, int]] = set()
        self._finalized = False
        header = {
            "type": "header",
            "schema_version": STEER_CACHE_SCHEMA_VERSION,
            "checkpoint_sha256": checkpoint_sha256,
            "config_sha256": config_sha256,
            "embedding_dim": self.embedding_dim,
            "protocol": self.protocol,
            "protocol_sha256": self.protocol_sha256,
            "coverage_policy": STEER_CACHE_COVERAGE_OBSERVED,
        }
        self._stream.write(_canonical_json(header) + "\n")
        self._stream.flush()

    def record(self, episode: int, request: int, embedding: torch.Tensor) -> None:
        key = (int(episode), int(request))
        if key in self._keys:
            raise ValueError(f"Duplicate steer cache recording key {key}.")
        canonical = embedding.detach().to(device="cpu", dtype=torch.float32).contiguous()
        if canonical.ndim == 2 and canonical.shape[0] == 1:
            canonical = canonical[0]
        if canonical.ndim != 1 or canonical.shape[0] != self.embedding_dim:
            raise ValueError(
                f"Recorded steer embedding must have shape [{self.embedding_dim}], "
                f"got {tuple(canonical.shape)}."
            )
        if not torch.isfinite(canonical).all():
            raise ValueError("Recorded steer embedding contains non-finite values.")
        payload = {
            "type": "entry",
            "episode": key[0],
            "request": key[1],
            "embedding": canonical.tolist(),
            "embedding_sha256": _embedding_sha256(canonical),
        }
        self._stream.write(_canonical_json(payload) + "\n")
        self._stream.flush()
        self._keys.add(key)

    def close(self) -> None:
        if self._stream.closed:
            return
        complete = True
        error_message = None
        request_counts: dict[str, int] = {}
        try:
            request_counts = _validate_cache_coverage(
                self._keys,
                protocol=self.protocol,
                coverage_policy=STEER_CACHE_COVERAGE_OBSERVED,
            )
        except ValueError as error:
            complete = False
            error_message = str(error)
        footer = {
            "type": "footer",
            "schema_version": STEER_CACHE_SCHEMA_VERSION,
            "complete": complete,
            "protocol_sha256": self.protocol_sha256,
            "coverage_policy": STEER_CACHE_COVERAGE_OBSERVED,
            "entry_count": len(self._keys),
            "episode_request_counts": request_counts,
            "keyset_sha256": _cache_keyset_sha256(self._keys),
            "error": error_message,
        }
        self._stream.write(_canonical_json(footer) + "\n")
        self._stream.flush()
        self._stream.close()
        self._finalized = True
        if not complete:
            raise ValueError(f"Cannot finalize incomplete steer cache: {error_message}")


class MockFastWAMPolicy:
    def __init__(self) -> None:
        self._reset_count = 0

    def get_action(self, observation: dict, options: dict | None = None) -> dict:
        del observation, options
        import numpy as np

        return {"action": np.zeros(7, dtype=np.float32)}, {"mock": True}

    def reset(self, options: dict | None = None) -> dict:
        del options
        self._reset_count += 1
        return {"reset_count": self._reset_count}

    def get_modality_config(self) -> dict:
        return {}


class FastWAMPolicy:
    """Training-aligned policy: observation keys fixed in policy_io.py."""

    def __init__(
        self,
        model: Any,
        processor: Any,
        *,
        device: str,
        action_horizon: int,
        num_inference_steps: int,
        num_video_frames: int | None = None,
        text_cfg_scale: float = 1.0,
        negative_prompt: str = "",
        sigma_shift: float | None = None,
        seed: int | None = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        steer_inference_mode: str = "learned",
        steer_cache: SteerEmbeddingCache | None = None,
        steer_cache_recorder: SteerEmbeddingRecorder | None = None,
        steer_protocol: dict[str, Any] | None = None,
        model_provenance: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.processor = processor
        self.device = device
        self.action_horizon = int(action_horizon)
        self.num_inference_steps = int(num_inference_steps)
        self.num_video_frames = None if num_video_frames is None else int(num_video_frames)
        self.text_cfg_scale = float(text_cfg_scale)
        self.negative_prompt = str(negative_prompt)
        self.sigma_shift = sigma_shift
        self.seed = seed
        self.rand_device = str(rand_device)
        self.tiled = bool(tiled)
        self.steer_inference_mode = str(steer_inference_mode).strip().lower()
        if self.steer_inference_mode not in STEER_INFERENCE_MODES:
            raise ValueError(
                f"steer_inference_mode must be one of {STEER_INFERENCE_MODES}, "
                f"got {steer_inference_mode!r}."
            )
        if self.steer_inference_mode == "cached" and steer_cache is None:
            raise ValueError("cached steer mode requires a loaded steer cache.")
        if self.steer_inference_mode != "cached" and steer_cache is not None:
            raise ValueError("A steer cache may only be supplied in cached mode.")
        if steer_cache_recorder is not None and self.steer_inference_mode != "learned":
            raise ValueError("Steer cache recording is only supported in learned mode.")
        self.steer_cache = steer_cache
        self.steer_cache_recorder = steer_cache_recorder
        self.steer_protocol = steer_protocol
        self.model_provenance = dict(model_provenance or {})
        self._steer_lock = threading.Lock()
        self._steer_episode_index = -1
        self._steer_request_index = 0
        self._episode = 0

    def get_modality_config(self) -> dict:
        return {
            "shape_meta": OmegaConf.to_container(self.processor.shape_meta, resolve=True),
            "steer_inference": {
                "mode": self.steer_inference_mode,
                "cache_path": (
                    None if self.steer_cache is None else str(self.steer_cache.path)
                ),
                "cache_sha256": (
                    None if self.steer_cache is None else self.steer_cache.file_sha256
                ),
                "record_path": (
                    None
                    if self.steer_cache_recorder is None
                    else str(self.steer_cache_recorder.path)
                ),
                "protocol": self.steer_protocol,
                "protocol_sha256": (
                    None
                    if self.steer_protocol is None
                    else _json_sha256(self.steer_protocol)
                ),
                "client_semantics": "single_client_per_server",
            },
            "model_provenance": self.model_provenance,
        }

    def close(self) -> None:
        if self.steer_cache_recorder is not None:
            self.steer_cache_recorder.close()

    def reset(self, options: dict | None = None) -> dict:
        del options
        with self._steer_lock:
            self._episode += 1
            self._steer_episode_index += 1
            self._steer_request_index = 0
            return {"episode": self._episode}

    def _select_modality_value(self, value: Any, meta_name: str, field_name: str) -> Any:
        if not isinstance(value, dict):
            return value

        meta = self.processor.shape_meta.get(meta_name, [])
        preferred_keys = [str(item["key"]) for item in meta if "key" in item]
        for key in preferred_keys:
            if key in value:
                return value[key]

        if len(value) == 1:
            return next(iter(value.values()))

        raise ValueError(
            f"observation['{field_name}'] is a nested dict with keys {sorted(value.keys())}; "
            f"expected one of training keys {preferred_keys}."
        )

    def _slice_proprio_to_state_keys(self, proprio: torch.Tensor) -> dict:
        """Split a merged proprio tensor into per-state-key sub-tensors.

        When the dataset uses a single 'default' state key, the proprio is
        returned as-is under that key. When multiple state keys exist (GR00T
        modality.json alignment), proprio is sliced per key using the
        modality_slice stored on each state meta entry by BaseLerobotDataset.
        """
        state_meta = self.processor.shape_meta["state"]
        state_batch = {}
        for meta in state_meta:
            key = meta["key"]
            sl = meta.get("modality_slice")
            if sl is not None:
                state_batch[key] = proprio[..., sl[0]:sl[1]]
            else:
                state_batch[key] = proprio
        return state_batch

    def _merge_state_keys_to_proprio(self, state_batch: dict) -> torch.Tensor:
        """Concatenate per-state-key tensors back into a merged proprio (left-aligned)."""
        merger = self.processor.action_state_merger
        # merger.forward expects {"state": {key: [T, D]}} and returns {"state": [T, D_total]}.
        out = merger.forward({"state": state_batch})
        return out["state"]

    def _normalize_proprio(self, proprio: torch.Tensor) -> torch.Tensor:
        """Apply state transforms + normalization, mirroring the training pipeline.

        Handles both single-key (default) and multi-key (modality.json) layouts.
        """
        state_batch = {"state": self._slice_proprio_to_state_keys(proprio)}
        state_batch = self.processor.action_state_transform(state_batch)
        state_batch = self.processor.normalizer.forward(state_batch)
        # Re-merge into the single proprio tensor the model expects.
        return self._merge_state_keys_to_proprio(state_batch["state"])

    def _normalize_observation(self, observation: dict) -> dict:
        """Accept both direct tensors and shape_meta-keyed modality dicts."""
        normalized = dict(observation)
        if KEY_INPUT_IMAGE in normalized:
            normalized[KEY_INPUT_IMAGE] = self._select_modality_value(
                normalized[KEY_INPUT_IMAGE],
                "images",
                KEY_INPUT_IMAGE,
            )
        if KEY_PROPRIO in normalized:
            normalized[KEY_PROPRIO] = self._select_modality_value(
                normalized[KEY_PROPRIO],
                "state",
                KEY_PROPRIO,
            )
        return normalized

    def get_action(self, observation: dict, options: dict | None = None) -> tuple[dict, dict]:
        if self.steer_cache is not None or self.steer_cache_recorder is not None:
            with self._steer_lock:
                return self._get_action_impl(observation, options)
        return self._get_action_impl(observation, options)

    def _get_action_impl(
        self,
        observation: dict,
        options: dict | None = None,
    ) -> tuple[dict, dict]:
        del options
        observation = self._normalize_observation(observation)
        validate_policy_observation(observation)
        tensors = to_inference_tensors(
            observation,
            device=self.model.device,
            dtype=self.model.torch_dtype,
        )

        proprio = tensors.get(KEY_PROPRIO)
        if proprio is not None:
            proprio = self._normalize_proprio(proprio)
            # Cache the normalized proprio on CPU so the action denormalizer can
            # use the last obs frame as the relative->absolute reference state.
            self._last_normalized_proprio = proprio.detach().to(dtype=torch.float32, device="cpu")

        infer_kwargs: dict[str, Any] = {
            KEY_INPUT_IMAGE: tensors[KEY_INPUT_IMAGE],
            "action_horizon": self.action_horizon,
            KEY_PROPRIO: proprio,
            "negative_prompt": self.negative_prompt,
            "text_cfg_scale": self.text_cfg_scale,
            "num_inference_steps": self.num_inference_steps,
            "sigma_shift": self.sigma_shift,
            "seed": self.seed,
            "rand_device": self.rand_device,
            "tiled": self.tiled,
        }
        infer_parameters = inspect.signature(self.model.infer_action).parameters
        if "num_video_frames" in infer_parameters:
            if self.num_video_frames is None:
                raise ValueError("Model infer_action requires num_video_frames, but config did not provide it.")
            infer_kwargs["num_video_frames"] = self.num_video_frames

        cache_key: tuple[int, int] | None = None
        explicit_steer_embedding: torch.Tensor | None = None
        if self.steer_cache is not None or self.steer_cache_recorder is not None:
            if self._steer_episode_index < 0:
                raise RuntimeError(
                    "Steer cache replay/recording requires reset() before get_action()."
                )
            cache_key = (self._steer_episode_index, self._steer_request_index)
        if self.steer_cache is not None:
            explicit_steer_embedding = self.steer_cache.get(*cache_key)

        if self.steer_inference_mode != "learned" or explicit_steer_embedding is not None:
            if "steer_inference_mode" not in infer_parameters:
                raise RuntimeError(
                    "The loaded model does not support inference-only steer interventions."
                )
            infer_kwargs["steer_inference_mode"] = (
                "explicit"
                if self.steer_inference_mode == "cached"
                else self.steer_inference_mode
            )
        if explicit_steer_embedding is not None:
            if "steer_embedding" not in infer_parameters:
                raise RuntimeError("The loaded model does not accept explicit steer embeddings.")
            infer_kwargs["steer_embedding"] = explicit_steer_embedding
        if self.steer_cache_recorder is not None:
            if "return_steer_embedding" not in infer_parameters:
                raise RuntimeError("The loaded model cannot return learned steer embeddings.")
            infer_kwargs["return_steer_embedding"] = True

        if KEY_CONTEXT in tensors:
            infer_kwargs[KEY_CONTEXT] = tensors[KEY_CONTEXT]
            infer_kwargs[KEY_CONTEXT_MASK] = tensors[KEY_CONTEXT_MASK]
            infer_kwargs[KEY_PROMPT] = None
        else:
            infer_kwargs[KEY_PROMPT] = tensors[KEY_PROMPT]

        try:
            with torch.no_grad():
                pred = self.model.infer_action(**infer_kwargs)
        except Exception:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise

        if self.steer_cache_recorder is not None:
            learned_embedding = pred.get("steer_embedding")
            if learned_embedding is None:
                raise RuntimeError("Learned steer cache recording produced no embedding.")
            self.steer_cache_recorder.record(*cache_key, learned_embedding)
        if cache_key is not None:
            self._steer_request_index += 1

        action_tensor = pred[KEY_ACTION]
        if action_tensor.ndim == 2:
            action_tensor = action_tensor.unsqueeze(0)
        action_np = self._denormalize_action(action_tensor)
        return {KEY_ACTION: action_np, "action_horizon": self.action_horizon}

    def _denormalize_action(self, action_tensor: torch.Tensor) -> np.ndarray:
        """Reverse normalization + relative->absolute transform for a predicted action.

        Uses the full processor.postprocess() chain (merger.backward ->
        normalizer.backward -> transforms.backward) so that multi-key modality
        layouts and relative-action transforms are correctly inverted, matching
        the training pipeline. Falls back to the legacy single-key normalizer
        path when there are no action_state_transforms.
        """
        action_meta = self.processor.shape_meta["action"]
        has_transforms = self.processor.action_state_transforms is not None
        action_tensor = action_tensor.to(dtype=torch.float32, device="cpu")
        if len(action_meta) == 1 and not has_transforms:
            action_key = action_meta[0]["key"]
            normalizer = self.processor.normalizer.normalizers["action"][action_key]
            return normalizer.backward(action_tensor).numpy()[0]

        # Multi-key / relative-transform path: build a postprocess input with the
        # predicted action and the LAST proprio frame as the reference state.
        # proprio was already normalized for the model; for postprocess we need the
        # normalized state at the last obs step as the relative reference frame.
        # The processor.postprocess expects {"action": (B, T, D), "proprio": (B, T_obs, D)}.
        # We reconstruct a minimal proprio from the cached last normalized proprio.
        proprio = getattr(self, "_last_normalized_proprio", None)
        if proprio is None:
            # No proprio available (e.g. proprio-less policy); skip state-dependent
            # transforms by providing a zero reference of the right dim.
            proprio = torch.zeros(
                1, self.processor.num_obs_steps, action_tensor.shape[-1],
                dtype=action_tensor.dtype, device=action_tensor.device,
            )
        if proprio.ndim == 2:
            proprio = proprio.unsqueeze(0)
        data = {"action": action_tensor, "proprio": proprio}
        # Run the reverse chain manually (mirrors processor.postprocess but WITHOUT
        # the obs-overlap slice, since deploy predictions contain only future
        # action steps, not the observation-aligned prefix).
        data["state"] = data.pop("proprio")
        data = self.processor.action_state_merger.backward(data)
        data = self.processor.normalizer.backward(data)
        if self.processor.action_state_transforms is not None:
            for trans in reversed(self.processor.action_state_transforms):
                data = trans.backward(data)
        # data["action"] is now a per-key dict of 3D tensors; re-merge to flat.
        action_flat = torch.cat(
            [data["action"][m["key"]] for m in action_meta], dim=-1
        )
        return action_flat[0].numpy()


def _resolve_stats_path(run_dir: Path, dataset_stats_path: str | None) -> Path:
    if dataset_stats_path:
        stats = Path(dataset_stats_path).expanduser().resolve()
        if not stats.is_file() or stats.stat().st_size <= 0:
            raise FileNotFoundError(
                f"dataset_stats_path must be a non-empty file: {stats}"
            )
        return stats
    default = run_dir / "dataset_stats.json"
    if not default.is_file() or default.stat().st_size <= 0:
        raise FileNotFoundError(
            f"Non-empty dataset_stats.json not found under run dir {run_dir}. "
            "Pass --dataset-stats-path explicitly."
        )
    return default


def _resolve_meta_stats_dir(
    run_dir: Path,
    configured_meta_dir: Any,
    norm_stats_meta_dir: str | None,
) -> Path:
    if norm_stats_meta_dir:
        meta_dir = Path(norm_stats_meta_dir).expanduser().resolve()
    else:
        if configured_meta_dir is None or not str(configured_meta_dir).strip():
            raise ValueError(
                "norm_stats_source=meta requires processor.norm_stats_meta_dir "
                "or --norm-stats-meta-dir."
            )
        raw = Path(str(configured_meta_dir)).expanduser()
        meta_dir = raw.resolve() if raw.is_absolute() else (run_dir / raw).resolve()
    for name in ("stats.json", "modality.json"):
        path = meta_dir / name
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(
                f"norm_stats_source=meta requires a non-empty {path}. "
                "Pass --norm-stats-meta-dir when the frozen artifacts were relocated."
            )
    return meta_dir


def _resolve_normalization_binding(
    processor_cfg: Any,
    *,
    run_dir: Path,
    dataset_stats_path: str | None,
    norm_stats_meta_dir: str | None,
) -> tuple[str, Path]:
    norm_stats_source = str(
        processor_cfg.get("norm_stats_source", "compute")
    ).strip().lower()
    if dataset_stats_path is not None and norm_stats_meta_dir is not None:
        raise ValueError(
            "--dataset-stats-path and --norm-stats-meta-dir are mutually exclusive"
        )
    if norm_stats_source == "meta":
        if dataset_stats_path is not None:
            raise ValueError(
                "The resolved config selects norm_stats_source=meta; "
                "--dataset-stats-path is not allowed."
            )
        meta_dir = _resolve_meta_stats_dir(
            run_dir,
            processor_cfg.get("norm_stats_meta_dir"),
            norm_stats_meta_dir,
        )
        processor_cfg["norm_stats_meta_dir"] = str(meta_dir)
        return "meta", meta_dir
    if norm_stats_meta_dir is not None:
        raise ValueError(
            f"The resolved config selects norm_stats_source={norm_stats_source!r}; "
            "--norm-stats-meta-dir is not allowed."
        )
    return "dataset_stats", _resolve_stats_path(run_dir, dataset_stats_path)


def _resolve_run_dir(run_dir: Path) -> Path:
    """Resolve to the training run directory that owns config.yaml."""
    if (run_dir / "config.yaml").exists():
        return run_dir
    for parent in run_dir.parents:
        if (parent / "config.yaml").exists():
            print(f"  Resolved run dir from {run_dir} to {parent}", flush=True)
            return parent
    raise FileNotFoundError(
        f"Training config not found under {run_dir} or its parents. "
        "Pass --run-dir as the training run directory containing config.yaml."
    )


DEFAULT_INFER_NUM_FRAMES = 33


def _resolve_inference_horizons(
    train_data: Any,
    *,
    action_horizon: int | None,
) -> tuple[int, int]:
    """Resolve closed-loop (action_horizon, num_video_frames) from run config."""
    num_frames = train_data.get("num_frames") if hasattr(train_data, "get") else None
    if num_frames is None and hasattr(train_data, "num_frames"):
        num_frames = train_data.num_frames
    if num_frames is not None:
        num_video_frames = int(num_frames)
        resolved_action_horizon = (
            int(action_horizon) if action_horizon is not None else num_video_frames - 1
        )
        return resolved_action_horizon, num_video_frames
    if action_horizon is not None:
        resolved_action_horizon = int(action_horizon)
        return resolved_action_horizon, resolved_action_horizon + 1
    target = ""
    if hasattr(train_data, "get"):
        target = str(train_data.get("_target_", ""))
    elif hasattr(train_data, "_target_"):
        target = str(train_data._target_)
    if "EveRobotFullEpisodeDataset" in target:
        raise ValueError(
            "EveRobot full-episode run config has no fixed `num_frames`. "
            "Pass --action-horizon when starting the policy server."
        )
    num_video_frames = DEFAULT_INFER_NUM_FRAMES
    return num_video_frames - 1, num_video_frames


def _resolve_checkpoint_path(run_dir: Path, checkpoint: str) -> Path:
    """Resolve checkpoint from absolute/relative paths or run-dir checkpoint layout."""
    raw = Path(checkpoint).expanduser()
    candidates: list[Path] = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.extend(
            [
                (Path.cwd() / raw),
                (run_dir / raw),
                (run_dir / "checkpoints" / "weights" / raw.name),
            ]
        )

    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.exists():
            return resolved

    tried = "\n  - ".join(str(candidate.resolve()) for candidate in candidates)
    raise FileNotFoundError(
        f"Checkpoint not found for --checkpoint {checkpoint!r}. Tried:\n  - {tried}"
    )


def _build_policy_from_run(
    run_dir: Path,
    checkpoint: str,
    dataset_stats_path: str | None,
    norm_stats_meta_dir: str | None,
    device: str,
    action_horizon: int | None,
    num_inference_steps: int | None,
    load_text_encoder: bool,
    inference_seed: int | None = None,
    steer_inference_mode: str = "learned",
    steer_cache_path: str | None = None,
    steer_cache_sha256: str | None = None,
    steer_cache_record_path: str | None = None,
    steer_protocol_json: str | None = None,
) -> FastWAMPolicy:
    from fastwam.datasets.lerobot.processors.fastwam_processor import FastWAMProcessor
    from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
    from fastwam.runtime import _mixed_precision_to_model_dtype, _normalize_mixed_precision

    run_dir = _resolve_run_dir(run_dir)
    checkpoint_path = _resolve_checkpoint_path(run_dir, checkpoint)
    config_path = run_dir / "config.yaml"

    cfg = OmegaConf.load(config_path)
    model_cfg = OmegaConf.create(OmegaConf.to_container(cfg.model, resolve=True))
    model_cfg.load_text_encoder = bool(load_text_encoder)
    mixed_precision = _normalize_mixed_precision(str(cfg.get("mixed_precision", "bf16")))
    model_dtype = _mixed_precision_to_model_dtype(mixed_precision)

    processor_cfg = OmegaConf.create(OmegaConf.to_container(cfg.data.train.processor, resolve=True))
    normalization_kind, normalization_path = _resolve_normalization_binding(
        processor_cfg,
        run_dir=run_dir,
        dataset_stats_path=dataset_stats_path,
        norm_stats_meta_dir=norm_stats_meta_dir,
    )

    if device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA unavailable; falling back to CPU.", flush=True)
        device = "cpu"

    print(f"  Config: {config_path}", flush=True)
    print(f"  Checkpoint: {checkpoint_path}", flush=True)
    print(f"  Load text encoder: {model_cfg.load_text_encoder}", flush=True)
    print(f"Loading FastWAM model on {device} ...", flush=True)
    model = instantiate(model_cfg, model_dtype=model_dtype, device=device)
    model.load_checkpoint(str(checkpoint_path))
    model.eval()

    steer_inference_mode = str(steer_inference_mode).strip().lower()
    if steer_inference_mode not in STEER_INFERENCE_MODES:
        raise ValueError(
            f"steer_inference_mode must be one of {STEER_INFERENCE_MODES}, "
            f"got {steer_inference_mode!r}."
        )
    if steer_inference_mode == "cached":
        if steer_cache_path is None or steer_cache_sha256 is None:
            raise ValueError(
                "cached steer mode requires --steer-cache-path and "
                "--steer-cache-sha256."
            )
    elif steer_cache_path is not None or steer_cache_sha256 is not None:
        raise ValueError(
            "--steer-cache-path/--steer-cache-sha256 require cached steer mode."
        )
    if steer_cache_record_path is not None and steer_inference_mode != "learned":
        raise ValueError("--steer-cache-record-path requires learned steer mode.")
    uses_steer_cache_io = steer_cache_path is not None or steer_cache_record_path is not None
    if uses_steer_cache_io and steer_protocol_json is None:
        raise ValueError(
            "Steer cache replay/recording requires --steer-protocol-json."
        )
    if not uses_steer_cache_io and steer_protocol_json is not None:
        raise ValueError("--steer-protocol-json requires cache replay or recording.")

    steer_cache = None
    steer_cache_recorder = None
    steer_protocol = None
    checkpoint_sha256 = None
    config_sha256 = None
    if uses_steer_cache_io:
        if not bool(getattr(model, "offline_steer_enabled", False)):
            raise ValueError("Steer cache replay/recording requires offline_steer.enabled=true.")
        embedding_dim = int(model.offline_steer_config["embedding_dim"])
        checkpoint_sha256 = _sha256_file(checkpoint_path)
        config_sha256 = _sha256_file(config_path)
        steer_protocol, _ = _parse_steer_protocol_json(steer_protocol_json)
        expected_model = {
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_sha256,
            "config_path": str(config_path.resolve()),
            "config_sha256": config_sha256,
        }
        if steer_protocol["model"] != expected_model:
            raise ValueError(
                "Steer protocol model provenance does not match the resolved server model: "
                f"expected {expected_model!r}, got {steer_protocol['model']!r}."
            )
        if steer_cache_path is not None:
            expected_cache_sha256 = _normalize_sha256(
                steer_cache_sha256,
                label="--steer-cache-sha256",
            )
            steer_cache = SteerEmbeddingCache(
                Path(steer_cache_path).expanduser().resolve(),
                expected_file_sha256=expected_cache_sha256,
                checkpoint_sha256=checkpoint_sha256,
                config_sha256=config_sha256,
                embedding_dim=embedding_dim,
                protocol=steer_protocol,
                required_coverage_policy=STEER_CACHE_COVERAGE_FULL,
            )
        if steer_cache_record_path is not None:
            steer_cache_recorder = SteerEmbeddingRecorder(
                Path(steer_cache_record_path).expanduser().resolve(),
                checkpoint_sha256=checkpoint_sha256,
                config_sha256=config_sha256,
                embedding_dim=embedding_dim,
                protocol=steer_protocol,
            )

    processor: FastWAMProcessor = instantiate(processor_cfg)
    processor.eval()
    if normalization_kind == "meta":
        # GR00T/meta path: rebuild normalizer from meta/stats.json + modality.json.
        processor.set_normalizer_from_modality_stats()
        print(f"  Modality stats (GR00T-style) from: {normalization_path}", flush=True)
    else:
        processor.set_normalizer_from_stats(
            load_dataset_stats_from_json(str(normalization_path))
        )
        print(f"  Dataset stats: {normalization_path}", flush=True)

    if action_horizon is None:
        action_horizon, num_video_frames = _resolve_inference_horizons(
            cfg.data.train, action_horizon=None
        )
    else:
        action_horizon, num_video_frames = _resolve_inference_horizons(
            cfg.data.train, action_horizon=action_horizon
        )
    if num_inference_steps is None:
        num_inference_steps = int(cfg.get("eval_num_inference_steps", 10))
    evaluation_cfg = cfg.get("EVALUATION", {})
    resolved_inference_seed = (
        int(inference_seed)
        if inference_seed is not None
        else evaluation_cfg.get("seed")
    )
    print(f"  Action horizon: {action_horizon}", flush=True)
    print(f"  Num video frames: {num_video_frames}", flush=True)
    print(f"  Num inference steps: {num_inference_steps}", flush=True)
    print(f"  Inference seed: {resolved_inference_seed}", flush=True)
    print(f"  Steer inference mode: {steer_inference_mode}", flush=True)
    if steer_cache is not None:
        print(f"  Steer cache: {steer_cache.path}", flush=True)
        print(f"  Steer cache SHA256: {steer_cache.file_sha256}", flush=True)
    if steer_cache_recorder is not None:
        print(f"  Steer cache record path: {steer_cache_recorder.path}", flush=True)

    return FastWAMPolicy(
        model=model,
        processor=processor,
        device=device,
        action_horizon=action_horizon,
        num_inference_steps=num_inference_steps,
        num_video_frames=num_video_frames,
        text_cfg_scale=float(evaluation_cfg.get("text_cfg_scale", 1.0)),
        negative_prompt=str(evaluation_cfg.get("negative_prompt", "")),
        sigma_shift=evaluation_cfg.get("sigma_shift"),
        seed=resolved_inference_seed,
        rand_device=str(evaluation_cfg.get("rand_device", "cpu")),
        tiled=bool(evaluation_cfg.get("tiled", False)),
        steer_inference_mode=steer_inference_mode,
        steer_cache=steer_cache,
        steer_cache_recorder=steer_cache_recorder,
        steer_protocol=steer_protocol,
        model_provenance={
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_sha256,
            "config_path": str(config_path.resolve()),
            "config_sha256": config_sha256,
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FastWAM ZMQ policy server.")
    parser.add_argument("--mock", action="store_true", help="Start mock policy (no model load).")
    parser.add_argument("--run-dir", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    normalization = parser.add_mutually_exclusive_group()
    normalization.add_argument("--dataset-stats-path", type=str, default=None)
    normalization.add_argument("--norm-stats-meta-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--action-horizon", type=int, default=None)
    parser.add_argument("--num-inference-steps", type=int, default=None)
    parser.add_argument(
        "--inference-seed",
        type=int,
        default=None,
        help="Override EVALUATION.seed for deterministic diffusion sampling.",
    )
    parser.add_argument(
        "--steer-inference-mode",
        choices=STEER_INFERENCE_MODES,
        default="learned",
        help="Inference-only steer intervention. cached replays explicit embeddings.",
    )
    parser.add_argument("--steer-cache-path", type=str, default=None)
    parser.add_argument("--steer-cache-sha256", type=str, default=None)
    parser.add_argument(
        "--steer-cache-record-path",
        type=str,
        default=None,
        help="Record learned steer embeddings as strict JSONL for later replay.",
    )
    parser.add_argument(
        "--steer-protocol-json",
        type=str,
        default=None,
        help="Canonical per-shard rollout protocol bound into steer caches.",
    )
    parser.add_argument(
        "--load-text-encoder",
        dest="load_text_encoder",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load text encoder/tokenizer so get_action can accept prompt strings.",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=DEFAULT_SERVER_PORT)
    parser.add_argument("--api-token", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("Starting FastWAM inference server...", flush=True)
    print(f"  Host: {args.host}", flush=True)
    print(f"  Port: {args.port}", flush=True)

    if args.mock:
        policy = MockFastWAMPolicy()
        print("  Policy: mock (no checkpoint)", flush=True)
    else:
        if not args.run_dir or not args.checkpoint:
            raise ValueError("--run-dir and --checkpoint are required unless --mock is set.")
        run_dir = Path(args.run_dir).expanduser().resolve()
        policy = _build_policy_from_run(
            run_dir=run_dir,
            checkpoint=args.checkpoint,
            dataset_stats_path=args.dataset_stats_path,
            norm_stats_meta_dir=args.norm_stats_meta_dir,
            device=args.device,
            action_horizon=args.action_horizon,
            num_inference_steps=args.num_inference_steps,
            load_text_encoder=args.load_text_encoder,
            inference_seed=args.inference_seed,
            steer_inference_mode=args.steer_inference_mode,
            steer_cache_path=args.steer_cache_path,
            steer_cache_sha256=args.steer_cache_sha256,
            steer_cache_record_path=args.steer_cache_record_path,
            steer_protocol_json=args.steer_protocol_json,
        )
        print(f"  Run dir: {_resolve_run_dir(run_dir)}", flush=True)
        print(f"  Device: {args.device}", flush=True)

    server = PolicyServer(policy=policy, host=args.host, port=args.port, api_token=args.api_token)
    print(f"\n✓ Server ready — listening on {args.host}:{args.port}\n", flush=True)
    def _terminate(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _terminate)
    try:
        server.run()
    except KeyboardInterrupt:
        print("\nShutting down server...", flush=True)
    finally:
        close_policy = getattr(policy, "close", None)
        if close_policy is not None:
            close_policy()


if __name__ == "__main__":
    main()
