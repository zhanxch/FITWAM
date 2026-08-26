#!/usr/bin/env python3
"""Launch a FastWAM inference policy server (ZMQ API, training-aligned I/O)."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import signal
import sys
from pathlib import Path
from typing import Any

import numpy as np
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
from policy_io import (
    KEY_ACTION,
    KEY_CONTEXT,
    KEY_CONTEXT_MASK,
    KEY_INPUT_IMAGE,
    KEY_NEGATIVE_CONTEXT,
    KEY_NEGATIVE_CONTEXT_MASK,
    KEY_NEGATIVE_PROMPT,
    KEY_FAILURE_CONTEXT,
    KEY_FAILURE_CONTEXT_MASK,
    KEY_FAILURE_PROMPT,
    KEY_PROMPT,
    KEY_PROPRIO,
)
from policy_io import to_inference_tensors, validate_policy_observation


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        negative_prompt: str | None = None,
        failure_prompt: str | None = None,
        sigma_shift: float | None = None,
        seed: int | None = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        adaptive_cfg_tau: float | None = None,
        cfg_exec_horizon: int = 24,
        cfg_epsilon_l: float | None = None,
        cfg_residual_clip_mode: str = "rms",
        model_provenance: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.processor = processor
        self.device = device
        self.action_horizon = int(action_horizon)
        self.num_inference_steps = int(num_inference_steps)
        self.num_video_frames = None if num_video_frames is None else int(num_video_frames)
        self.text_cfg_scale = float(text_cfg_scale)
        self.negative_prompt = (
            None if negative_prompt is None or str(negative_prompt) == "" else str(negative_prompt)
        )
        self.failure_prompt = (
            None if failure_prompt is None or str(failure_prompt) == "" else str(failure_prompt)
        )
        self.sigma_shift = sigma_shift
        self.seed = seed
        self.rand_device = str(rand_device)
        self.tiled = bool(tiled)
        self.adaptive_cfg_tau = None if adaptive_cfg_tau is None else float(adaptive_cfg_tau)
        self.cfg_exec_horizon = int(cfg_exec_horizon)
        self.cfg_epsilon_l = None if cfg_epsilon_l is None else float(cfg_epsilon_l)
        self.cfg_residual_clip_mode = str(cfg_residual_clip_mode)
        self.model_provenance = dict(model_provenance or {})
        self._episode = 0

    def get_modality_config(self) -> dict:
        return {
            "shape_meta": OmegaConf.to_container(self.processor.shape_meta, resolve=True),
            "model_provenance": self.model_provenance,
        }

    def reset(self, options: dict | None = None) -> dict:
        del options
        self._episode += 1
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

        infer_seed = self.seed
        return_cfg_residual = False
        cfg_exec_horizon = int(self.cfg_exec_horizon)
        adaptive_cfg_tau = self.adaptive_cfg_tau
        cfg_epsilon_l = self.cfg_epsilon_l
        cfg_residual_clip_mode = self.cfg_residual_clip_mode
        if options:
            raw_seed = options.get("seed", options.get("inference_seed"))
            if raw_seed is not None:
                infer_seed = int(raw_seed)
            return_cfg_residual = bool(options.get("return_cfg_residual", False))
            if options.get("cfg_exec_horizon") is not None:
                cfg_exec_horizon = int(options["cfg_exec_horizon"])
            if options.get("adaptive_cfg_tau") is not None:
                adaptive_cfg_tau = float(options["adaptive_cfg_tau"])
            for key in ("cfg_epsilon_l", "epsilon_l", "cfg_residual_epsilon"):
                if options.get(key) is not None:
                    cfg_epsilon_l = float(options[key])
                    break
            if options.get("cfg_residual_clip_mode") is not None:
                cfg_residual_clip_mode = str(options["cfg_residual_clip_mode"])

        infer_kwargs: dict[str, Any] = {
            KEY_INPUT_IMAGE: tensors[KEY_INPUT_IMAGE],
            "action_horizon": self.action_horizon,
            KEY_PROPRIO: proprio,
            "text_cfg_scale": self.text_cfg_scale,
            "num_inference_steps": self.num_inference_steps,
            "sigma_shift": self.sigma_shift,
            "seed": infer_seed,
            "rand_device": self.rand_device,
            "tiled": self.tiled,
        }
        infer_parameters = inspect.signature(self.model.infer_action).parameters
        if "num_video_frames" in infer_parameters:
            if self.num_video_frames is None:
                raise ValueError("Model infer_action requires num_video_frames, but config did not provide it.")
            infer_kwargs["num_video_frames"] = self.num_video_frames
        if return_cfg_residual and "return_cfg_residual" in infer_parameters:
            infer_kwargs["return_cfg_residual"] = True
        if adaptive_cfg_tau is not None and "adaptive_cfg_tau" in infer_parameters:
            infer_kwargs["adaptive_cfg_tau"] = float(adaptive_cfg_tau)
        if cfg_epsilon_l is not None and "cfg_epsilon_l" in infer_parameters:
            infer_kwargs["cfg_epsilon_l"] = float(cfg_epsilon_l)
        if (
            cfg_epsilon_l is not None
            and "cfg_residual_clip_mode" in infer_parameters
        ):
            infer_kwargs["cfg_residual_clip_mode"] = str(cfg_residual_clip_mode)
        if (
            (return_cfg_residual or adaptive_cfg_tau is not None)
            and "cfg_exec_horizon" in infer_parameters
        ):
            infer_kwargs["cfg_exec_horizon"] = int(cfg_exec_horizon)
        if options:
            if options.get("cfg_gate_mode") is not None and "cfg_gate_mode" in infer_parameters:
                infer_kwargs["cfg_gate_mode"] = str(options["cfg_gate_mode"])
            if options.get("cfg_value_prev") is not None and "cfg_value_prev" in infer_parameters:
                infer_kwargs["cfg_value_prev"] = float(options["cfg_value_prev"])
            if options.get("cfg_gate_fired") is not None and "cfg_gate_fired" in infer_parameters:
                infer_kwargs["cfg_gate_fired"] = bool(options["cfg_gate_fired"])
            if options.get("cfg_v_high") is not None and "cfg_v_high" in infer_parameters:
                infer_kwargs["cfg_v_high"] = float(options["cfg_v_high"])
            if options.get("cfg_drop_delta") is not None and "cfg_drop_delta" in infer_parameters:
                infer_kwargs["cfg_drop_delta"] = float(options["cfg_drop_delta"])
            if options.get("cfg_replan_index") is not None and "cfg_replan_index" in infer_parameters:
                infer_kwargs["cfg_replan_index"] = int(options["cfg_replan_index"])
            if options.get("cfg_growth_tau") is not None and "cfg_growth_tau" in infer_parameters:
                infer_kwargs["cfg_growth_tau"] = float(options["cfg_growth_tau"])
            if (
                options.get("cfg_growth_start_replan") is not None
                and "cfg_growth_start_replan" in infer_parameters
            ):
                infer_kwargs["cfg_growth_start_replan"] = int(
                    options["cfg_growth_start_replan"]
                )

        if KEY_CONTEXT in tensors:
            infer_kwargs[KEY_CONTEXT] = tensors[KEY_CONTEXT]
            infer_kwargs[KEY_CONTEXT_MASK] = tensors[KEY_CONTEXT_MASK]
            infer_kwargs[KEY_PROMPT] = None
            if KEY_NEGATIVE_CONTEXT in tensors:
                infer_kwargs[KEY_NEGATIVE_CONTEXT] = tensors[KEY_NEGATIVE_CONTEXT]
                infer_kwargs[KEY_NEGATIVE_CONTEXT_MASK] = tensors[KEY_NEGATIVE_CONTEXT_MASK]
            if KEY_FAILURE_CONTEXT in tensors:
                infer_kwargs[KEY_FAILURE_CONTEXT] = tensors[KEY_FAILURE_CONTEXT]
                infer_kwargs[KEY_FAILURE_CONTEXT_MASK] = tensors[KEY_FAILURE_CONTEXT_MASK]
        else:
            infer_kwargs[KEY_PROMPT] = tensors[KEY_PROMPT]
            infer_kwargs[KEY_NEGATIVE_PROMPT] = tensors.get(
                KEY_NEGATIVE_PROMPT,
                self.negative_prompt,
            )
            infer_kwargs["failure_prompt"] = tensors.get(
                KEY_FAILURE_PROMPT,
                self.failure_prompt,
            )

        try:
            with torch.no_grad():
                pred = self.model.infer_action(**infer_kwargs)
        except Exception:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise

        action_tensor = pred[KEY_ACTION]
        if action_tensor.ndim == 2:
            action_tensor = action_tensor.unsqueeze(0)
        action_np = self._denormalize_action(action_tensor)
        payload: dict[str, Any] = {KEY_ACTION: action_np, "action_horizon": self.action_horizon}
        extra_keys = []
        if return_cfg_residual:
            extra_keys.extend(
                (
                    "cfg_token_rms_nfe",
                    "cfg_chunk_rms_nfe",
                    "cfg_token_rms",
                    "cfg_chunk_rms",
                    "cfg_exec_rms",
                )
            )
        if adaptive_cfg_tau is not None:
            extra_keys.extend(("cfg_mix_weight", "cfg_gate_exec_rms"))
        extra_keys.extend(("cfg_value", "cfg_value_rel", "cfg_gate_g", "cfg_mix_weight"))
        if cfg_epsilon_l is not None:
            extra_keys.extend(("cfg_epsilon_l", "cfg_residual_clip_mode"))
        for key in extra_keys:
            if key not in pred:
                continue
            value = pred[key]
            if isinstance(value, torch.Tensor):
                payload[key] = value.detach().to(device="cpu", dtype=torch.float32).numpy()
            else:
                payload[key] = np.asarray(value, dtype=np.float32)
        return payload

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
    text_cfg_scale: float | None = None,
    negative_prompt: str | None = None,
    failure_prompt: str | None = None,
    backbone_checkpoint: str | None = None,
    uncond_adapter: str | None = None,
    adaptive_cfg_tau: float | None = None,
    cfg_epsilon_l: float | None = None,
    cfg_residual_clip_mode: str | None = None,
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
    # Training may set load_vae=false when using a VAE latent cache. Live policy
    # serving always encodes camera frames, so VAE must be loaded for inference.
    model_cfg.load_vae = True
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
    print(f"  Load VAE: {model_cfg.load_vae} (forced true for live image encode)", flush=True)
    print(f"Loading FastWAM model on {device} ...", flush=True)
    model = instantiate(model_cfg, model_dtype=model_dtype, device=device)
    if getattr(model, "uncond_adapter_injected", False):
        from fastwam.models.wan22.uncond_adapter import (
            load_uncond_adapter_state_dict,
            resolve_backbone_and_adapter_paths,
            set_uncond_adapter_enabled,
        )

        resume_raw = cfg.get("resume", None)
        config_resume = None if resume_raw in (None, "null", "") else str(resume_raw)
        backbone_arg = None
        if backbone_checkpoint:
            backbone_arg = str(
                _resolve_checkpoint_path(run_dir, backbone_checkpoint)
            )
        adapter_arg = None
        if uncond_adapter:
            adapter_arg = str(_resolve_checkpoint_path(run_dir, uncond_adapter))
        config_resume_resolved = None
        if config_resume:
            try:
                config_resume_resolved = str(
                    _resolve_checkpoint_path(run_dir, config_resume)
                )
            except FileNotFoundError:
                config_resume_resolved = config_resume
        backbone_path, adapter_path = resolve_backbone_and_adapter_paths(
            str(checkpoint_path),
            backbone_checkpoint=backbone_arg,
            adapter_checkpoint=adapter_arg,
            config_resume=config_resume_resolved,
        )
        backbone_path = str(_resolve_checkpoint_path(run_dir, backbone_path))
        adapter_path = str(_resolve_checkpoint_path(run_dir, adapter_path))
        print(f"  DEWO v5 backbone: {backbone_path}", flush=True)
        print(f"  DEWO v5 adapter: {adapter_path}", flush=True)
        model.load_checkpoint(backbone_path)
        load_uncond_adapter_state_dict(model, adapter_path)
        set_uncond_adapter_enabled(model, False)
        checkpoint_sha256 = _sha256_file(Path(adapter_path))
    else:
        model.load_checkpoint(str(checkpoint_path))
        checkpoint_sha256 = _sha256_file(checkpoint_path)
    model.eval()
    config_sha256 = _sha256_file(config_path)

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
    resolved_text_cfg_scale = (
        float(text_cfg_scale)
        if text_cfg_scale is not None
        else float(evaluation_cfg.get("text_cfg_scale", 1.0))
    )
    resolved_negative_prompt = (
        negative_prompt
        if negative_prompt is not None
        else evaluation_cfg.get("negative_prompt")
    )
    resolved_failure_prompt = (
        failure_prompt
        if failure_prompt is not None
        else evaluation_cfg.get("failure_prompt")
    )
    print(f"  Text CFG scale: {resolved_text_cfg_scale} (1.0 = 本体 remap, not mix w=1)", flush=True)
    resolved_adaptive_tau = None if adaptive_cfg_tau is None else float(adaptive_cfg_tau)
    if resolved_adaptive_tau is not None:
        print(
            f"  Adaptive CFG tau: {resolved_adaptive_tau} "
            f"(E>tau mix w={resolved_text_cfg_scale}; else mix w=0 本体)",
            flush=True,
        )
    resolved_epsilon_l = (
        None
        if cfg_epsilon_l is None and evaluation_cfg.get("cfg_epsilon_l") is None
        else float(
            cfg_epsilon_l
            if cfg_epsilon_l is not None
            else evaluation_cfg.get("cfg_epsilon_l")
        )
    )
    raw_clip_mode = (
        cfg_residual_clip_mode
        if cfg_residual_clip_mode is not None
        else evaluation_cfg.get("cfg_residual_clip_mode", "rms")
    )
    resolved_clip_mode = "rms" if raw_clip_mode in (None, "") else str(raw_clip_mode)
    if resolved_epsilon_l is not None:
        print(
            f"  CFG residual epsilon_l: {resolved_epsilon_l} "
            f"(clip mode={resolved_clip_mode})",
            flush=True,
        )

    return FastWAMPolicy(
        model=model,
        processor=processor,
        device=device,
        action_horizon=action_horizon,
        num_inference_steps=num_inference_steps,
        num_video_frames=num_video_frames,
        text_cfg_scale=resolved_text_cfg_scale,
        negative_prompt=resolved_negative_prompt,
        failure_prompt=resolved_failure_prompt,
        sigma_shift=evaluation_cfg.get("sigma_shift"),
        seed=resolved_inference_seed,
        rand_device=str(evaluation_cfg.get("rand_device", "cpu")),
        tiled=bool(evaluation_cfg.get("tiled", False)),
        adaptive_cfg_tau=resolved_adaptive_tau,
        cfg_epsilon_l=resolved_epsilon_l,
        cfg_residual_clip_mode=resolved_clip_mode,
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
        "--text-cfg-scale",
        type=float,
        default=None,
        help=(
            "Action CFG knob. 1.0 = 本体 bypass (adapter off + cfg_base_prompt). "
            "Values != 1 run the guided mix. v5/v6 mix w=1 is ε_posi; "
            "v7 mix w=1 is ε_base+(ε_posi-ε_fail), not ε_posi."
        ),
    )
    parser.add_argument(
        "--adaptive-cfg-tau",
        type=float,
        default=None,
        help=(
            "If set, freeze mix from NFE0 exec RMS: E>tau uses --text-cfg-scale, "
            "else mix w=0 (本体). Requires text_cfg_scale != 1."
        ),
    )
    parser.add_argument(
        "--cfg-epsilon-l",
        "--epsilon-l",
        "--cfg-residual-epsilon",
        dest="cfg_epsilon_l",
        type=float,
        default=None,
        help=(
            "Bound the per-token action CFG residual before text-cfg scaling. "
            "None keeps legacy unbounded guidance; 0 is the base branch."
        ),
    )
    parser.add_argument(
        "--cfg-residual-clip-mode",
        choices=("rms", "elementwise"),
        default=None,
        help="How --cfg-epsilon-l bounds the residual (default: rms).",
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default=None,
        help="Base prompt for prompt-mode CFG. Cached-context clients send their base context per request.",
    )
    parser.add_argument(
        "--failure-prompt",
        type=str,
        default=None,
        help="Failure-conditioned prompt for DEWO v7 CFG. Cached-context clients send failure context per request.",
    )
    parser.add_argument(
        "--load-text-encoder",
        dest="load_text_encoder",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load text encoder/tokenizer so get_action can accept prompt strings.",
    )
    parser.add_argument(
        "--backbone-checkpoint",
        type=str,
        default=None,
        help="Frozen base MoT for DEWO v5 CFG (adapter-off branch).",
    )
    parser.add_argument(
        "--uncond-adapter",
        type=str,
        default=None,
        help="DEWO v5 uncond-adapter weights. If omitted, --checkpoint may be the adapter file.",
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
            text_cfg_scale=args.text_cfg_scale,
            negative_prompt=args.negative_prompt,
            failure_prompt=getattr(args, "failure_prompt", None),
            backbone_checkpoint=args.backbone_checkpoint,
            uncond_adapter=args.uncond_adapter,
            adaptive_cfg_tau=args.adaptive_cfg_tau,
            cfg_epsilon_l=args.cfg_epsilon_l,
            cfg_residual_clip_mode=args.cfg_residual_clip_mode,
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


if __name__ == "__main__":
    main()
