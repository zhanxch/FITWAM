#!/usr/bin/env python3
"""Launch a FastWAM policy server for Wuji/Astribot real-robot deployment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
SRC_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_ROOT, SCRIPTS_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dexjoco_fastwam_adapter import DEFAULT_PROMPT
from fastwam_policy_server import DEFAULT_SERVER_PORT, PolicyServer
from policy_io import KEY_ACTION, KEY_CONTEXT, KEY_CONTEXT_MASK, KEY_INPUT_IMAGE, KEY_PROMPT, KEY_PROPRIO
from policy_io import to_inference_tensors, validate_policy_observation
from wuji_fastwam_adapter import (
    WUJI_ACTION_DIM,
    WUJI_ACTION_KEYS,
    gr00t_obs_to_policy_obs,
    is_gr00t_observation,
    split_wuji_action,
)


class MockFastWAMPolicy:
    def __init__(self, action_horizon: int = 16) -> None:
        self.action_horizon = int(action_horizon)
        self._reset_count = 0

    def get_action(self, observation: dict, options: dict | None = None) -> tuple[dict, dict]:
        del observation, options
        action = split_wuji_action([[0.0] * WUJI_ACTION_DIM] * self.action_horizon)
        return action, {"mock": True, "action_horizon": self.action_horizon}

    def reset(self, options: dict | None = None) -> dict:
        del options
        self._reset_count += 1
        return {"reset_count": self._reset_count}

    def get_modality_config(self) -> dict:
        return {"action_dim": WUJI_ACTION_DIM}


class FastWAMPolicy:
    def __init__(
        self,
        model: Any,
        processor: Any,
        *,
        action_horizon: int,
        num_inference_steps: int,
        use_proprio: bool,
        text_cfg_scale: float = 1.0,
        negative_prompt: str = "",
        sigma_shift: float | None = None,
        seed: int | None = None,
        rand_device: str = "cpu",
        tiled: bool = False,
    ) -> None:
        self.model = model
        self.processor = processor
        self.action_horizon = int(action_horizon)
        self.num_inference_steps = int(num_inference_steps)
        self.use_proprio = bool(use_proprio)
        self.text_cfg_scale = float(text_cfg_scale)
        self.negative_prompt = str(negative_prompt)
        self.sigma_shift = sigma_shift
        self.seed = seed
        self.rand_device = str(rand_device)
        self.tiled = bool(tiled)
        self._episode = 0

    def get_modality_config(self) -> dict:
        return {
            "action_dim": WUJI_ACTION_DIM,
            "action_keys": list(WUJI_ACTION_KEYS),
            "shape_meta": OmegaConf.to_container(self.processor.shape_meta, resolve=True),
        }

    def reset(self, options: dict | None = None) -> dict:
        del options
        self._episode += 1
        return {"episode": self._episode}

    def _to_policy_observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        if is_gr00t_observation(observation):
            return gr00t_obs_to_policy_obs(observation, include_proprio=self.use_proprio)
        return dict(observation)

    def _normalized_proprio(self, proprio: Any) -> Any:
        if not self.use_proprio or proprio is None:
            return None

        state_meta = self.processor.shape_meta["state"]
        if len(state_meta) != 1:
            raise ValueError("Wuji FastWAM server expects one merged state key.")
        state_key = state_meta[0]["key"]
        state_batch = {"state": {state_key: proprio.to(dtype=torch.float32)}}
        state_batch = self.processor.action_state_transform(state_batch)
        state_batch = self.processor.normalizer.forward(state_batch)
        return state_batch["state"][state_key]

    def _denormalize_action(self, action: Any) -> np.ndarray:
        action_tensor = action
        if not isinstance(action_tensor, torch.Tensor):
            action_tensor = torch.as_tensor(action_tensor)
        if action_tensor.ndim == 2:
            pass
        elif action_tensor.ndim == 3 and action_tensor.shape[0] == 1:
            action_tensor = action_tensor[0]
        else:
            raise ValueError(f"Expected action tensor [T,58] or [1,T,58], got {tuple(action_tensor.shape)}")
        if action_tensor.shape[-1] != WUJI_ACTION_DIM:
            raise ValueError(f"FastWAM action dim must be {WUJI_ACTION_DIM}, got {action_tensor.shape[-1]}")

        action_meta = self.processor.shape_meta["action"]
        if len(action_meta) != 1:
            raise ValueError("Wuji FastWAM server expects one merged action key.")
        action_key = action_meta[0]["key"]
        normalizer = self.processor.normalizer.normalizers["action"][action_key]
        denorm = normalizer.backward(action_tensor.to(dtype=torch.float32, device="cpu"))
        return denorm.numpy()

    def get_action(self, observation: dict, options: dict | None = None) -> tuple[dict, dict]:
        del options
        policy_obs = self._to_policy_observation(observation)
        validate_policy_observation(policy_obs)
        tensors = to_inference_tensors(
            policy_obs,
            device=self.model.device,
            dtype=self.model.torch_dtype,
        )

        proprio = self._normalized_proprio(tensors.get(KEY_PROPRIO))
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

        if KEY_CONTEXT in tensors:
            infer_kwargs[KEY_CONTEXT] = tensors[KEY_CONTEXT]
            infer_kwargs[KEY_CONTEXT_MASK] = tensors[KEY_CONTEXT_MASK]
            infer_kwargs[KEY_PROMPT] = None
        else:
            task = str(tensors[KEY_PROMPT])
            infer_kwargs[KEY_PROMPT] = DEFAULT_PROMPT.format(task=task)

        try:
            with torch.no_grad():
                pred = self.model.infer_action(**infer_kwargs)
        except Exception:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise

        action_np = self._denormalize_action(pred[KEY_ACTION])
        return split_wuji_action(action_np), {"action_horizon": self.action_horizon}


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


def _resolve_checkpoint_path(run_dir: Path, checkpoint: str) -> Path:
    raw = Path(checkpoint).expanduser()
    candidates: list[Path] = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.extend(
            [
                Path.cwd() / raw,
                run_dir / raw,
                run_dir / "checkpoints" / "weights" / raw.name,
            ]
        )

    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.exists():
            return resolved

    tried = "\n  - ".join(str(candidate.resolve()) for candidate in candidates)
    raise FileNotFoundError(f"Checkpoint not found for --checkpoint {checkpoint!r}. Tried:\n  - {tried}")


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

    print("Loading FastWAM Wuji policy...", flush=True)
    print(f"  Config: {config_path}", flush=True)
    print(f"  Checkpoint: {checkpoint_path}", flush=True)
    print(f"  Device: {device}", flush=True)
    print(f"  Load text encoder: {model_cfg.load_text_encoder}", flush=True)

    model = instantiate(model_cfg, model_dtype=model_dtype, device=device)
    model.load_checkpoint(str(checkpoint_path))
    model.eval()

    processor_cfg = cfg.data.train.processor
    processor: FastWAMProcessor = instantiate(processor_cfg)
    processor.eval()
    stats_path = _resolve_stats_path(run_dir, dataset_stats_path)
    processor.set_normalizer_from_stats(load_dataset_stats_from_json(str(stats_path)))

    if action_horizon is None:
        action_horizon = int(cfg.data.train.num_frames) - 1
    if num_inference_steps is None:
        num_inference_steps = int(cfg.get("eval_num_inference_steps", 10))

    use_proprio = getattr(model, "proprio_dim", None) is not None
    print(f"  Dataset stats: {stats_path}", flush=True)
    print(f"  Action horizon: {action_horizon}", flush=True)
    print(f"  Num inference steps: {num_inference_steps}", flush=True)
    print(f"  Use proprio: {use_proprio}", flush=True)

    evaluation_cfg = cfg.get("EVALUATION", {})
    return FastWAMPolicy(
        model=model,
        processor=processor,
        action_horizon=action_horizon,
        num_inference_steps=num_inference_steps,
        use_proprio=use_proprio,
        text_cfg_scale=float(evaluation_cfg.get("text_cfg_scale", 1.0)),
        negative_prompt=str(evaluation_cfg.get("negative_prompt", "")),
        sigma_shift=evaluation_cfg.get("sigma_shift"),
        seed=evaluation_cfg.get("seed"),
        rand_device=str(evaluation_cfg.get("rand_device", "cpu")),
        tiled=bool(evaluation_cfg.get("tiled", False)),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a FastWAM Wuji real-robot policy server.")
    parser.add_argument("--mock", action="store_true", help="Start mock policy without loading FastWAM.")
    parser.add_argument("--run-dir", type=str, required=False, help="Training run dir containing config.yaml")
    parser.add_argument("--checkpoint", type=str, required=False, help="Checkpoint path or filename under run dir")
    parser.add_argument("--dataset-stats-path", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--action-horizon", type=int, default=None)
    parser.add_argument("--num-inference-steps", type=int, default=None)
    parser.add_argument(
        "--load-text-encoder",
        dest="load_text_encoder",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load text encoder/tokenizer so get_action can accept task strings.",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=DEFAULT_SERVER_PORT)
    parser.add_argument("--api-token", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mock:
        policy: Any = MockFastWAMPolicy(action_horizon=args.action_horizon or 16)
        print("Starting mock FastWAM Wuji policy server...", flush=True)
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

    server = PolicyServer(policy=policy, host=args.host, port=args.port, api_token=args.api_token)
    print(f"\nServer ready: tcp://{args.host}:{args.port}\n", flush=True)
    try:
        server.run()
    except KeyboardInterrupt:
        print("\nShutting down server...", flush=True)


if __name__ == "__main__":
    main()
