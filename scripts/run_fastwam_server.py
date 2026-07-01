#!/usr/bin/env python3
"""Launch a FastWAM inference policy server (ZMQ API, training-aligned I/O)."""

from __future__ import annotations

import argparse
import inspect
import sys
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
        self._episode = 0

    def get_modality_config(self) -> dict:
        return {"shape_meta": OmegaConf.to_container(self.processor.shape_meta, resolve=True)}

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
        if "num_video_frames" in inspect.signature(self.model.infer_action).parameters:
            if self.num_video_frames is None:
                raise ValueError("Model infer_action requires num_video_frames, but config did not provide it.")
            infer_kwargs["num_video_frames"] = self.num_video_frames

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
        if not stats.exists():
            raise FileNotFoundError(f"dataset_stats_path not found: {stats}")
        return stats
    default = run_dir / "dataset_stats.json"
    if not default.exists():
        raise FileNotFoundError(
            f"dataset_stats.json not found under run dir {run_dir}. "
            "Pass --dataset-stats-path explicitly."
        )
    return default


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


def _maybe_resolve_norm_stats_meta_dir(processor_cfg: Any, cfg: Any, run_dir: Path) -> None:
    """Repair training-machine relative meta paths for deployed DexJoCo runs."""
    if not hasattr(processor_cfg, "get"):
        return
    raw_meta_dir = processor_cfg.get("norm_stats_meta_dir")
    if raw_meta_dir in (None, "", "null"):
        return

    raw_path = Path(str(raw_meta_dir)).expanduser()
    candidates: list[Path] = []
    if raw_path.is_absolute():
        candidates.append(raw_path)
    else:
        candidates.extend([Path.cwd() / raw_path, run_dir / raw_path])

    train_data = cfg.get("data", {}).get("train", {}) if hasattr(cfg, "get") else {}
    dataset_dirs = train_data.get("dataset_dirs", []) if hasattr(train_data, "get") else []
    if isinstance(dataset_dirs, (str, Path)):
        dataset_dirs = [dataset_dirs]
    for dataset_dir in dataset_dirs:
        dataset_path = Path(str(dataset_dir)).expanduser()
        if dataset_path.is_absolute():
            candidates.append(dataset_path / "meta")
        else:
            candidates.extend([Path.cwd() / dataset_path / "meta", run_dir / dataset_path / "meta"])

        name = dataset_path.name
        task = name[:-8] if name.endswith("_fastwam") else name
        if task:
            candidates.append(
                Path("/data_all/share/datasets/dexjoco/dexjoco_lerobot_datasets") / task / "meta"
            )

    for candidate in candidates:
        if candidate.exists():
            processor_cfg.norm_stats_meta_dir = str(candidate.resolve())
            if str(raw_meta_dir) != processor_cfg.norm_stats_meta_dir:
                print(
                    f"  Resolved norm_stats_meta_dir from {raw_meta_dir} "
                    f"to {processor_cfg.norm_stats_meta_dir}",
                    flush=True,
                )
            return


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
    device: str,
    action_horizon: int | None,
    num_inference_steps: int | None,
    load_text_encoder: bool,
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

    processor_cfg = cfg.data.train.processor
    _maybe_resolve_norm_stats_meta_dir(processor_cfg, cfg, run_dir)
    processor: FastWAMProcessor = instantiate(processor_cfg)
    processor.eval()
    if processor.wants_modality_stats:
        # GR00T/meta path: rebuild normalizer from meta/stats.json + modality.json.
        processor.set_normalizer_from_modality_stats()
        print(f"  Modality stats (GR00T-style) from: {processor.norm_stats_meta_dir}", flush=True)
    else:
        stats_path = _resolve_stats_path(run_dir, dataset_stats_path)
        processor.set_normalizer_from_stats(load_dataset_stats_from_json(str(stats_path)))
        print(f"  Dataset stats: {stats_path}", flush=True)

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
    print(f"  Action horizon: {action_horizon}", flush=True)
    print(f"  Num video frames: {num_video_frames}", flush=True)
    print(f"  Num inference steps: {num_inference_steps}", flush=True)

    return FastWAMPolicy(
        model=model,
        processor=processor,
        device=device,
        action_horizon=action_horizon,
        num_inference_steps=num_inference_steps,
        num_video_frames=num_video_frames,
        text_cfg_scale=float(cfg.get("EVALUATION", {}).get("text_cfg_scale", 1.0)),
        negative_prompt=str(cfg.get("EVALUATION", {}).get("negative_prompt", "")),
        sigma_shift=cfg.get("EVALUATION", {}).get("sigma_shift"),
        seed=cfg.get("EVALUATION", {}).get("seed"),
        rand_device=str(cfg.get("EVALUATION", {}).get("rand_device", "cpu")),
        tiled=bool(cfg.get("EVALUATION", {}).get("tiled", False)),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FastWAM ZMQ policy server.")
    parser.add_argument("--mock", action="store_true", help="Start mock policy (no model load).")
    parser.add_argument("--run-dir", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--dataset-stats-path", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--action-horizon", type=int, default=None)
    parser.add_argument("--num-inference-steps", type=int, default=None)
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
            device=args.device,
            action_horizon=args.action_horizon,
            num_inference_steps=args.num_inference_steps,
            load_text_encoder=args.load_text_encoder,
        )
        print(f"  Run dir: {_resolve_run_dir(run_dir)}", flush=True)
        print(f"  Device: {args.device}", flush=True)

    server = PolicyServer(policy=policy, host=args.host, port=args.port, api_token=args.api_token)
    print(f"\n✓ Server ready — listening on {args.host}:{args.port}\n", flush=True)
    try:
        server.run()
    except KeyboardInterrupt:
        print("\nShutting down server...", flush=True)


if __name__ == "__main__":
    main()
