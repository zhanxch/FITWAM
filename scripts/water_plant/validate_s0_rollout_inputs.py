#!/usr/bin/env python3
"""Freeze and validate the formal Water Plant S0 rollout protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import numbers
import os
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_VERSION = "fitwam.water_plant_s0_rollout.v2"
SUCCESS_PROMPT = "Grasp the watering can and apply water to the plant."
MODEL_PROMPT = (
    "A video recorded from a robot's point of view executing the following "
    "instruction: {task}"
)
CODE_FILES = (
    "scripts/collect_dexjoco_rollouts.py",
    "scripts/build_rollout_datasets.py",
    "scripts/dexjoco_async/multi_gpu_eval_utils.py",
    "scripts/dexjoco_async/run_multi_gpu_dexjoco_collect.py",
    "scripts/dexjoco_async/run_multi_gpu_dexjoco_eval.py",
    "scripts/dexjoco_async/dexjoco_fastwam_adapter.py",
    "scripts/dexjoco_async/eval_dexjoco_fastwam_control.py",
    "scripts/dexjoco_async/eval_summary_aggregator.py",
    "scripts/run_fastwam_server.py",
    "scripts/run_fastwam_server_async.py",
    "scripts/water_plant/collect_offline_s0_rollout_200.sh",
    "scripts/water_plant/sanity_s0_rollout_4.sh",
    "scripts/water_plant/validate_s0_sanity_outputs.py",
    "scripts/water_plant/validate_s0_rollout_inputs.py",
    "src/fastwam/models/wan22/action_dit.py",
    "src/fastwam/models/wan22/fastwam.py",
    "src/fastwam/models/wan22/helpers/io.py",
)
BASE_MODEL_FILES = (
    "Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model-00001-of-00003.safetensors",
    "Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model-00002-of-00003.safetensors",
    "Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model-00003-of-00003.safetensors",
    "DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors",
    "ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt",
)
STAT_NAMES = ("min", "max", "q01", "q99", "mean", "std")
S0_STAT_SHAPES = {
    "action": {
        "label": "action",
        "stepwise": (32, 22),
        "global": (22,),
    },
    "state": {
        "label": "observation.state",
        "stepwise": (1, 23),
        "global": (23,),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate S0 inputs and write an immutable collection protocol."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    normalization = parser.add_mutually_exclusive_group()
    normalization.add_argument("--dataset-stats", type=Path, default=None)
    normalization.add_argument("--norm-stats-meta-dir", type=Path, default=None)
    parser.add_argument("--text-cache-dir", type=Path, default=None)
    parser.add_argument("--source-dataset", type=Path, required=True)
    parser.add_argument(
        "--base-checkpoints-dir",
        type=Path,
        default=PROJECT_ROOT / "checkpoints",
    )
    parser.add_argument("--dexjoco-root", type=Path, required=True)
    parser.add_argument("--protocol-out", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-name", default="step_006500.pt")
    parser.add_argument(
        "--collection-kind",
        choices=("formal", "sanity"),
        default="formal",
    )
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--base-seed", type=int, default=20260718)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--replan-steps", type=int, default=25)
    parser.add_argument("--max-env-steps", type=int, default=1500)
    parser.add_argument("--video-fps", type=int, default=30)
    parser.add_argument("--outcome-task-mode", choices=("clean",), default="clean")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Require an existing protocol and verify that every immutable field matches.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.incomplete-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def require_shape(
    features: dict[str, Any],
    key: str,
    expected: list[int],
    *,
    source_dataset: Path,
) -> None:
    feature = features.get(key)
    shape = feature.get("shape") if isinstance(feature, dict) else None
    if list(shape or []) != expected:
        raise ValueError(
            f"{source_dataset}: feature {key!r} must have shape {expected}, got {shape}"
        )


def validate_source_dataset(source_dataset: Path) -> dict[str, Any]:
    info_path = source_dataset / "meta" / "info.json"
    episodes_path = source_dataset / "meta" / "episodes.jsonl"
    for path in (info_path, episodes_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    info = load_json(info_path)
    features = info.get("features")
    if not isinstance(features, dict):
        raise ValueError(f"{info_path} is missing a feature mapping")
    require_shape(features, "action", [22], source_dataset=source_dataset)
    require_shape(
        features,
        "observation.state",
        [23],
        source_dataset=source_dataset,
    )
    camera_keys = {
        key
        for key, spec in features.items()
        if key.startswith("observation.images.")
        and isinstance(spec, dict)
        and spec.get("dtype") == "video"
    }
    expected_cameras = {
        "observation.images.front",
        "observation.images.wrist",
    }
    if camera_keys != expected_cameras:
        raise ValueError(
            f"{source_dataset}: expected exactly front+wrist video features, "
            f"got {sorted(camera_keys)}"
        )
    return {
        "info_path": str(info_path),
        "info_sha256": sha256_file(info_path),
        "episodes_path": str(episodes_path),
        "episodes_sha256": sha256_file(episodes_path),
        "fps": int(info.get("fps", -1)),
        "total_episodes": int(info.get("total_episodes", -1)),
        "camera_keys": sorted(camera_keys),
        "action_dim": 22,
        "proprio_dim": 23,
    }


def numeric_shape(value: Any, *, field: str) -> tuple[int, ...]:
    if isinstance(value, list):
        if not value:
            raise ValueError(f"{field} must be a non-empty numeric array")
        child_shapes = [
            numeric_shape(item, field=f"{field}[{index}]")
            for index, item in enumerate(value)
        ]
        if any(shape != child_shapes[0] for shape in child_shapes[1:]):
            raise ValueError(f"{field} must be rectangular, got ragged nested arrays")
        return (len(value), *child_shapes[0])
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(
            f"{field} must contain only numeric values, got {type(value).__name__}"
        )
    try:
        finite = math.isfinite(float(value))
    except (OverflowError, ValueError):
        finite = False
    if not finite:
        raise ValueError(f"{field} must contain only finite numeric values, got {value!r}")
    return ()


def flatten_numeric(value: Any) -> list[float]:
    if isinstance(value, list):
        return [number for item in value for number in flatten_numeric(item)]
    return [float(value)]


def require_stat_order(
    stats: dict[str, Any],
    *,
    prefix: str,
    field: str,
) -> None:
    values = {
        name: flatten_numeric(stats[f"{prefix}_{name}"])
        for name in STAT_NAMES
    }
    for index, std in enumerate(values["std"]):
        if std < 0:
            raise ValueError(
                f"{field}.{prefix}_std[{index}] must be non-negative, got {std}"
            )
    for index, (minimum, q01, mean, q99, maximum) in enumerate(
        zip(
            values["min"],
            values["q01"],
            values["mean"],
            values["q99"],
            values["max"],
        )
    ):
        if not minimum <= q01 <= q99 <= maximum:
            raise ValueError(
                f"{field}.{prefix} statistics at flat index {index} must satisfy "
                f"min <= q01 <= q99 <= max, got "
                f"{minimum}, {q01}, {q99}, {maximum}"
            )
        if not minimum <= mean <= maximum:
            raise ValueError(
                f"{field}.{prefix}_mean[{index}]={mean} must lie within "
                f"[{minimum}, {maximum}]"
            )


def validate_stats(path: Path) -> dict[str, Any]:
    payload = load_json(path)
    if not isinstance(payload, dict) or not payload:
        raise ValueError(f"{path} must contain non-empty normalization statistics")
    for count_key in ("num_episodes", "num_transition"):
        count = payload.get(count_key)
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError(
                f"{path}: {count_key} must be a positive integer, got {count!r}"
            )

    report: dict[str, Any] = {}
    for modality, specification in S0_STAT_SHAPES.items():
        modality_stats = payload.get(modality)
        label = str(specification["label"])
        if not isinstance(modality_stats, dict):
            raise ValueError(
                f"{path}: missing top-level {modality!r} statistics for {label}"
            )
        if set(modality_stats) != {"default"}:
            raise ValueError(
                f"{path}: {modality} must contain exactly the 'default' field, "
                f"got {sorted(map(str, modality_stats))}"
            )
        field_stats = modality_stats["default"]
        if not isinstance(field_stats, dict):
            raise ValueError(f"{path}: {modality}.default must be a statistics mapping")

        field_report: dict[str, list[int]] = {}
        for prefix in ("stepwise", "global"):
            expected_shape = tuple(specification[prefix])
            for stat_name in STAT_NAMES:
                key = f"{prefix}_{stat_name}"
                if key not in field_stats:
                    raise ValueError(
                        f"{path}: {modality}.default is missing required field {key!r}"
                    )
                field = f"{path}:{modality}.default.{key}"
                actual_shape = numeric_shape(field_stats[key], field=field)
                if actual_shape != expected_shape:
                    raise ValueError(
                        f"{field} must have shape {list(expected_shape)} for the "
                        f"frozen S0 {label} statistics, got {list(actual_shape)}"
                    )
            require_stat_order(
                field_stats,
                prefix=prefix,
                field=f"{path}:{modality}.default",
            )
            field_report[f"{prefix}_shape"] = list(expected_shape)
        report[modality] = field_report

    return {
        "source": "dataset_stats",
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "num_episodes": payload["num_episodes"],
        "num_transition": payload["num_transition"],
        "fields": report,
    }


def validate_modality_group(
    modality: dict[str, Any],
    *,
    group: str,
    expected_dim: int,
    path: Path,
) -> list[dict[str, Any]]:
    slices = modality.get(group)
    if not isinstance(slices, dict) or not slices:
        raise ValueError(f"{path}: modality.json must define non-empty {group!r} slices")

    cursor = 0
    report = []
    for name, bounds in slices.items():
        if not isinstance(bounds, dict):
            raise ValueError(f"{path}: {group}.{name} must be a start/end mapping")
        start = bounds.get("start")
        end = bounds.get("end")
        if (
            isinstance(start, bool)
            or not isinstance(start, int)
            or isinstance(end, bool)
            or not isinstance(end, int)
        ):
            raise ValueError(
                f"{path}: {group}.{name} start/end must be integers, got "
                f"{start!r}/{end!r}"
            )
        if start != cursor or end <= start or end > expected_dim:
            raise ValueError(
                f"{path}: {group}.{name} must continue the ordered contiguous "
                f"[0, {expected_dim}) layout at {cursor}, got [{start}, {end})"
            )
        report.append({"name": str(name), "start": start, "end": end})
        cursor = end
    if cursor != expected_dim:
        raise ValueError(
            f"{path}: {group} slices must cover exactly [0, {expected_dim}), "
            f"got [0, {cursor})"
        )
    return report


def validate_meta_feature_stats(
    stats: dict[str, Any],
    *,
    feature_key: str,
    expected_dim: int,
    path: Path,
) -> dict[str, Any]:
    feature_stats = stats.get(feature_key)
    if not isinstance(feature_stats, dict):
        raise ValueError(f"{path}: missing {feature_key!r} statistics")

    values: dict[str, list[float]] = {}
    for stat_name in STAT_NAMES:
        if stat_name not in feature_stats:
            raise ValueError(
                f"{path}: {feature_key} is missing required field {stat_name!r}"
            )
        field = f"{path}:{feature_key}.{stat_name}"
        shape = numeric_shape(feature_stats[stat_name], field=field)
        if shape != (expected_dim,):
            raise ValueError(
                f"{field} must have shape [{expected_dim}], got {list(shape)}"
            )
        values[stat_name] = flatten_numeric(feature_stats[stat_name])

    for index, std in enumerate(values["std"]):
        if std < 0:
            raise ValueError(
                f"{path}:{feature_key}.std[{index}] must be non-negative, got {std}"
            )
    for index, (minimum, q01, mean, q99, maximum) in enumerate(
        zip(
            values["min"],
            values["q01"],
            values["mean"],
            values["q99"],
            values["max"],
        )
    ):
        if not minimum <= q01 <= q99 <= maximum:
            raise ValueError(
                f"{path}:{feature_key} statistics at index {index} must satisfy "
                f"min <= q01 <= q99 <= max, got "
                f"{minimum}, {q01}, {q99}, {maximum}"
            )
        if not minimum <= mean <= maximum:
            raise ValueError(
                f"{path}:{feature_key}.mean[{index}]={mean} must lie within "
                f"[{minimum}, {maximum}]"
            )
    return {"shape": [expected_dim], "required_stats": list(STAT_NAMES)}


def resolve_meta_dir(
    *,
    override: Path | None,
    config_path: Path,
) -> Path:
    if override is None:
        raise ValueError(
            f"{config_path}: norm_stats_source=meta requires an explicit "
            "--norm-stats-meta-dir for relocatable S0 rollout"
        )
    return override.expanduser().resolve()


def validate_meta_stats(meta_dir: Path) -> dict[str, Any]:
    stats_path = meta_dir / "stats.json"
    modality_path = meta_dir / "modality.json"
    if not meta_dir.is_dir():
        raise FileNotFoundError(meta_dir)
    for path in (stats_path, modality_path):
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(
                f"Normalization meta source requires a non-empty file: {path}"
            )
    stats = load_json(stats_path)
    modality = load_json(modality_path)
    if not isinstance(stats, dict):
        raise ValueError(f"{stats_path} must contain a JSON mapping")
    if not isinstance(modality, dict):
        raise ValueError(f"{modality_path} must contain a JSON mapping")

    modality_report = {
        "action": validate_modality_group(
            modality,
            group="action",
            expected_dim=22,
            path=modality_path,
        ),
        "state": validate_modality_group(
            modality,
            group="state",
            expected_dim=23,
            path=modality_path,
        ),
    }
    fields = {
        "action": validate_meta_feature_stats(
            stats,
            feature_key="action",
            expected_dim=22,
            path=stats_path,
        ),
        "state": validate_meta_feature_stats(
            stats,
            feature_key="observation.state",
            expected_dim=23,
            path=stats_path,
        ),
    }
    return {
        "source": "meta",
        "meta_dir": str(meta_dir),
        "stats": {
            "path": str(stats_path),
            "sha256": sha256_file(stats_path),
            "size_bytes": stats_path.stat().st_size,
        },
        "modality": {
            "path": str(modality_path),
            "sha256": sha256_file(modality_path),
            "size_bytes": modality_path.stat().st_size,
            "slices": modality_report,
        },
        "fields": fields,
    }


def resolve_text_cache(
    train: dict[str, Any],
    *,
    config_path: Path,
    override: Path | None,
) -> dict[str, Any]:
    configured = train.get("text_embedding_cache_dir")
    if override is not None:
        cache_dir = override.expanduser().resolve()
    elif configured:
        raw = Path(str(configured)).expanduser()
        cache_dir = (
            raw.resolve()
            if raw.is_absolute()
            else (config_path.parent / raw).resolve()
        )
    else:
        raise ValueError(
            f"{config_path}: load_text_encoder=false requires text_embedding_cache_dir"
        )
    context_len = int(train.get("context_len", 128))
    instruction = MODEL_PROMPT.format(task=SUCCESS_PROMPT)
    digest = hashlib.sha256(instruction.encode("utf-8")).hexdigest()
    cache_file = cache_dir / f"{digest}.t5_len{context_len}.npz"
    if not cache_file.is_file():
        raise FileNotFoundError(
            f"Missing Water Plant text embedding cache: {cache_file}"
        )
    return {
        "configured_dir": configured,
        "effective_dir": str(cache_dir),
        "context_len": context_len,
        "water_plant_npz": str(cache_file),
        "water_plant_npz_sha256": sha256_file(cache_file),
    }


def validate_run_config(
    path: Path,
    *,
    replan_steps: int,
    text_cache_override: Path | None,
) -> dict[str, Any]:
    payload = load_yaml(path)
    data = payload.get("data")
    train = data.get("train") if isinstance(data, dict) else None
    if not isinstance(train, dict):
        raise ValueError(f"{path} is missing resolved data.train")
    shape_meta = train.get("shape_meta")
    images = shape_meta.get("images") if isinstance(shape_meta, dict) else None
    image_keys = {
        str(item.get("key"))
        for item in images or []
        if isinstance(item, dict)
    }
    if image_keys != {"front", "wrist"}:
        raise ValueError(f"{path}: expected front+wrist cameras, got {sorted(image_keys)}")
    if list(train.get("video_size") or []) != [384, 768]:
        raise ValueError(f"{path}: data.train.video_size must be [384, 768]")
    if train.get("concat_multi_camera") != "horizontal":
        raise ValueError(f"{path}: concat_multi_camera must be 'horizontal'")
    processor = train.get("processor")
    if not isinstance(processor, dict):
        raise ValueError(f"{path} is missing data.train.processor")
    if int(processor.get("num_output_cameras", -1)) != 2:
        raise ValueError(f"{path}: num_output_cameras must be 2")
    if int(processor.get("action_output_dim", -1)) != 22:
        raise ValueError(f"{path}: action_output_dim must be 22")
    if int(processor.get("proprio_output_dim", -1)) != 23:
        raise ValueError(f"{path}: proprio_output_dim must be 23")
    norm_stats_source = str(processor.get("norm_stats_source", "compute")).strip().lower()
    if not norm_stats_source:
        norm_stats_source = "compute"
    model = payload.get("model")
    if not isinstance(model, dict) or int(model.get("proprio_dim", -1)) != 23:
        raise ValueError(f"{path}: model.proprio_dim must be 23")
    if bool(model.get("load_text_encoder", False)):
        raise ValueError(
            f"{path}: formal S0 protocol requires load_text_encoder=false and cached context"
        )
    num_frames = int(train.get("num_frames", -1))
    if num_frames != 33:
        raise ValueError(f"{path}: data.train.num_frames must be 33")
    action_horizon = num_frames - 1
    if replan_steps < 1 or replan_steps > action_horizon:
        raise ValueError(
            f"replan_steps={replan_steps} must be in [1, {action_horizon}]"
        )
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "camera_keys": sorted(image_keys),
        "video_size": [384, 768],
        "concat_multi_camera": "horizontal",
        "action_dim": 22,
        "proprio_dim": 23,
        "num_frames": num_frames,
        "action_horizon": action_horizon,
        "load_text_encoder": False,
        "norm_stats_source": norm_stats_source,
        "configured_norm_stats_meta_dir": processor.get("norm_stats_meta_dir"),
        "text_cache": resolve_text_cache(
            train,
            config_path=path,
            override=text_cache_override,
        ),
    }


def git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            text=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def code_fingerprints() -> dict[str, str]:
    fingerprints = {}
    for relative in CODE_FILES:
        path = PROJECT_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        fingerprints[relative] = sha256_file(path)
    return fingerprints


def validate_base_model_files(root: Path) -> dict[str, Any]:
    root = root.expanduser().resolve()
    files = {}
    for relative in BASE_MODEL_FILES:
        path = root / relative
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(f"Missing non-empty FastWAM base model file: {path}")
        files[relative] = {
            "path": str(path),
            "size_bytes": path.stat().st_size,
        }
    return {
        "root": str(root),
        "real_root": str(root.resolve()),
        "files": files,
    }


def validate_dexjoco_runtime(root: Path) -> dict[str, Any]:
    root = root.expanduser().resolve()
    runtime_files = (
        "configs/rand_obj/water_plant.yaml",
        "dexjoco/dexjoco/tasks/mappings.py",
    )
    files = {}
    for relative in runtime_files:
        path = root / relative
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(f"Missing DexJoCo runtime file: {path}")
        files[relative] = {
            "path": str(path),
            "sha256": sha256_file(path),
        }
    return {"root": str(root), "files": files}


def validate_inputs(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = args.run_dir.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    source_dataset = args.source_dataset.expanduser().resolve()
    base_checkpoints_dir = args.base_checkpoints_dir.expanduser().resolve()
    dexjoco_root = args.dexjoco_root.expanduser().resolve()
    config = run_dir / "config.yaml"
    for path in (run_dir, checkpoint, source_dataset, config):
        if not path.exists():
            raise FileNotFoundError(path)
    if checkpoint.name != args.expected_checkpoint_name:
        raise ValueError(
            f"Expected checkpoint {args.expected_checkpoint_name!r}, got {checkpoint.name!r}"
        )
    if run_dir != checkpoint.parent and run_dir not in checkpoint.parents:
        raise ValueError(f"Checkpoint {checkpoint} must be stored under run dir {run_dir}")
    expected_episodes = 200 if args.collection_kind == "formal" else 4
    if args.episodes != expected_episodes or args.gpus != "0,1,2,3":
        raise ValueError(
            f"{args.collection_kind} collection requires {expected_episodes} episodes "
            "on GPU 0,1,2,3"
        )
    if args.replan_steps != 25 or args.max_env_steps != 1500:
        raise ValueError("Formal collection requires replan=25 and max_env_steps=1500")
    if args.video_fps != 30 or args.outcome_task_mode != "clean":
        raise ValueError("Formal collection requires 30fps and clean task text")

    config_report = validate_run_config(
        config,
        replan_steps=args.replan_steps,
        text_cache_override=args.text_cache_dir,
    )
    norm_source = config_report["norm_stats_source"]
    if norm_source == "meta":
        if args.dataset_stats is not None:
            raise ValueError(
                f"{config}: norm_stats_source=meta forbids --dataset-stats; "
                "pass --norm-stats-meta-dir when relocating meta artifacts"
            )
        meta_dir = resolve_meta_dir(
            override=args.norm_stats_meta_dir,
            config_path=config,
        )
        normalization_report = validate_meta_stats(meta_dir)
    else:
        if args.norm_stats_meta_dir is not None:
            raise ValueError(
                f"{config}: norm_stats_source={norm_source!r} forbids "
                "--norm-stats-meta-dir"
            )
        if args.dataset_stats is None:
            raise ValueError(
                f"{config}: norm_stats_source={norm_source!r} requires "
                "--dataset-stats"
            )
        stats = args.dataset_stats.expanduser().resolve()
        if not stats.is_file():
            raise FileNotFoundError(stats)
        normalization_report = validate_stats(stats)
    normalization_report["configured_norm_stats_source"] = norm_source

    dataset_report = validate_source_dataset(source_dataset)
    if dataset_report["fps"] != args.video_fps:
        raise ValueError(
            f"Source dataset fps={dataset_report['fps']} does not match {args.video_fps}"
        )
    return {
        "protocol_version": PROTOCOL_VERSION,
        "model": {
            "run_dir": str(run_dir),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "checkpoint_size_bytes": checkpoint.stat().st_size,
            "config": config_report,
            "normalization": normalization_report,
            "base_model": validate_base_model_files(base_checkpoints_dir),
        },
        "environment": validate_dexjoco_runtime(dexjoco_root),
        "source_dataset": {
            "root": str(source_dataset),
            **dataset_report,
        },
        "collection": {
            "kind": args.collection_kind,
            "episodes": args.episodes,
            "base_seed": args.base_seed,
            "seed_stop_exclusive": args.base_seed + args.episodes,
            "gpus": [0, 1, 2, 3],
            "replan_steps": args.replan_steps,
            "max_env_steps": args.max_env_steps,
            "video_fps": args.video_fps,
            "outcome_task_mode": args.outcome_task_mode,
            "randomize": False,
            "randomize_dynamics": False,
            "action_clip": False,
        },
        "code": {
            "git_commit": git_commit(),
            "files_sha256": code_fingerprints(),
        },
    }


def main() -> None:
    args = parse_args()
    protocol_path = args.protocol_out.expanduser().resolve()
    immutable = validate_inputs(args)
    if args.resume:
        if not protocol_path.is_file():
            raise FileNotFoundError(
                f"Resume requires existing collection protocol: {protocol_path}"
            )
        existing = load_json(protocol_path)
        existing_immutable = dict(existing)
        existing_immutable.pop("created_at_utc", None)
        existing_immutable.pop("created_by_host", None)
        if existing_immutable != immutable:
            raise ValueError(
                "Existing collection protocol does not match the requested resume inputs"
            )
        print(f"[s0-protocol] resume verified {protocol_path}")
        return
    if protocol_path.exists():
        raise FileExistsError(f"Refusing to overwrite protocol: {protocol_path}")
    payload = {
        **immutable,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "created_by_host": socket.gethostname(),
    }
    atomic_write_json(protocol_path, payload)
    print(f"[s0-protocol] wrote {protocol_path}")
    print(f"[s0-protocol] checkpoint_sha256={payload['model']['checkpoint_sha256']}")


if __name__ == "__main__":
    main()
