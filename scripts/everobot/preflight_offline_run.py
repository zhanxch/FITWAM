#!/usr/bin/env python3
"""Block an Offline Self-Improving run unless its evidence contract is valid."""

from __future__ import annotations

import argparse
import copy
import hashlib
import hmac
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any, Mapping, Sequence

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
for source_root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from fastwam.datasets.eve.pair_targets import PairTargetStore  # noqa: E402
from fastwam.everobot_schema import validate_manifest  # noqa: E402


VARIANTS = {"B0", "B1", "C", "M"}
BUNDLE_FORMAT = "FITWAMOfflineProtocolBundle"
BUNDLE_VERSION = 3
PROTOCOL_NAME = "fitwam_offline_self_improving_v1"
WATER_PLANT_TASK = "Grasp the watering can and apply water to the plant."
WATER_PLANT_TEXT_CACHE_BASENAME = (
    hashlib.sha256(
        (
            "A video recorded from a robot's point of view executing the "
            f"following instruction: {WATER_PLANT_TASK}"
        ).encode("utf-8")
    ).hexdigest()
    + ".t5_len128.wan22ti2v5b.pt"
)
EXECUTION_MODES = {"formal", "smoke20", "smoke500"}
EXECUTION_CONTRACTS = {
    "formal": {
        "max_steps": 6500,
        "eval_every": 500,
        "save_weights_every": 0,
        "save_weight_steps": [500, 1000, 3000, 5000, 6000, 6500],
        "save_state_every": 1500,
        "lr_scheduler_total_steps": None,
        "provenance_run_mode": None,
    },
    "smoke20": {
        "max_steps": 20,
        "eval_every": 10,
        "save_weights_every": 10,
        "save_weight_steps": None,
        "save_state_every": 20,
        "lr_scheduler_total_steps": 500,
        "provenance_run_mode": "preformal_smoke",
    },
    "smoke500": {
        "max_steps": 500,
        "eval_every": 100,
        "save_weights_every": 100,
        "save_weight_steps": None,
        "save_state_every": 500,
        "lr_scheduler_total_steps": 500,
        "provenance_run_mode": "preformal_smoke",
    },
}
CODE_SNAPSHOT_FORMAT = "FITWAMCodeSnapshot"
CODE_SNAPSHOT_VERSION = 1
CODE_SNAPSHOT_STATIC_PATHS = (
    "configs/data/eve_water_plant_round1_failure_events.yaml",
    "configs/model/fastwam.yaml",
    "configs/task/dexjoco/"
    "dexjoco_water_plant_offline_self_improving_2cam_proprio_1e-4.yaml",
    "configs/train.yaml",
    "scripts/accelerate_configs/accelerate_zero1_ds.yaml",
    "scripts/ds_configs/ds_zero1_config.json",
    "scripts/everobot/attach_event_pairs_to_manifest.py",
    "scripts/everobot/build_episode_split.py",
    "scripts/everobot/build_event_pairs.py",
    "scripts/everobot/build_eve_sidecar.py",
    "scripts/everobot/extract_event_pair_features.py",
    "scripts/everobot/extract_state_line_events.py",
    "scripts/everobot/match_auxiliary_manifest_budget.py",
    "scripts/everobot/preflight_offline_run.py",
    "scripts/everobot/render_state_line_audit.py",
    "scripts/everobot/train_offline_steer_teacher.py",
    "scripts/everobot/validate_offline_event_pair_quality.py",
    "scripts/everobot/validate_manifest_split.py",
    "scripts/train.py",
    "scripts/train_zero1.sh",
    "scripts/water_plant/prepare_offline_self_improving.sh",
    "scripts/water_plant/train_offline_self_improving.sh",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def code_snapshot_relative_paths(project_root: Path) -> list[str]:
    source_root = project_root / "src" / "fastwam"
    source_paths = [
        path.relative_to(project_root).as_posix()
        for path in source_root.rglob("*.py")
        if path.is_file()
    ]
    return sorted(set(source_paths).union(CODE_SNAPSHOT_STATIC_PATHS))


def build_code_snapshot(
    project_root: Path = PROJECT_ROOT,
    *,
    relative_paths: Sequence[str] | None = None,
) -> dict[str, Any]:
    project_root = project_root.expanduser().resolve()
    selected_paths = (
        code_snapshot_relative_paths(project_root)
        if relative_paths is None
        else sorted(set(relative_paths))
    )
    files: list[dict[str, str]] = []
    for raw_relative_path in selected_paths:
        relative_path = Path(raw_relative_path)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(
                f"Code snapshot paths must be project-relative: {raw_relative_path!r}"
            )
        normalized = relative_path.as_posix()
        path = project_root / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"Code snapshot input is missing: {path}")
        files.append({"path": normalized, "sha256": sha256_file(path)})
    if not files:
        raise ValueError("Code snapshot must contain at least one file")
    payload: dict[str, Any] = {
        "format": CODE_SNAPSHOT_FORMAT,
        "version": CODE_SNAPSHOT_VERSION,
        "files": files,
    }
    payload["snapshot_sha256"] = sha256_json(payload)
    return payload


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return payload


def nested_get(payload: Mapping[str, Any], *keys: str) -> Any:
    value: Any = payload
    for key in keys:
        if not isinstance(value, Mapping) or key not in value:
            return None
        value = value[key]
    return value


def parse_variant_paths(
    specs: Sequence[str],
    *,
    label: str,
) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"{label} must use VARIANT=/path syntax: {spec!r}")
        raw_variant, raw_path = spec.split("=", 1)
        variant = raw_variant.strip().upper()
        if variant not in VARIANTS:
            raise ValueError(f"{label} has unsupported variant {variant!r}")
        if variant in parsed:
            raise ValueError(f"{label} repeats variant {variant}")
        if not raw_path.strip():
            raise ValueError(f"{label} path for {variant} must not be empty")
        parsed[variant] = Path(raw_path).expanduser().resolve()
    missing = sorted(VARIANTS - set(parsed))
    extra = sorted(set(parsed) - VARIANTS)
    if missing or extra:
        raise ValueError(
            f"{label} must define exactly B0/B1/C/M; missing={missing}, extra={extra}"
        )
    return parsed


def sample_role(sample: Mapping[str, Any]) -> str:
    explicit = sample.get("batch_role")
    if explicit in {"primary", "auxiliary"}:
        return str(explicit)
    is_failure = (
        sample.get("episode_outcome") == "failure"
        or sample.get("event_outcome") == "failure"
    )
    action_enabled = sample.get("action_loss") != "disabled"
    return "primary" if not is_failure and action_enabled else "auxiliary"


def primary_sample_identities(
    manifest: Mapping[str, Any],
) -> list[dict[str, Any]]:
    identities = []
    for sample in manifest.get("samples", []):
        if sample_role(sample) != "primary":
            continue
        identity = {
            "sample_id": str(sample["sample_id"]),
            "split": str(sample.get("split", "train")),
            "dataset_id": str(sample["dataset_id"]),
            "episode_id": str(sample["episode_id"]),
            "episode_index": int(sample["episode_index"]),
            "start_frame": int(sample["start_frame"]),
            "end_frame": int(sample["end_frame"]),
            "sample_stride": int(sample["sample_stride"]),
            "valid_intervals": copy.deepcopy(sample.get("valid_intervals")),
            "action_loss_window": copy.deepcopy(sample.get("action_loss_window")),
        }
        identities.append(identity)
    identities.sort(
        key=lambda item: json.dumps(
            item,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    if len({sha256_json(item) for item in identities}) != len(identities):
        raise ValueError("Primary sample identities must be unique within a manifest")
    return identities


def validate_source_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    if not isinstance(payload, dict):
        raise ValueError("Source config must resolve to a YAML mapping")
    train = nested_get(payload, "data", "train")
    if not isinstance(train, Mapping):
        raise ValueError("Source config is missing resolved data.train")
    images = nested_get(train, "shape_meta", "images")
    image_keys = {
        str(item.get("key"))
        for item in images or []
        if isinstance(item, Mapping)
    }
    if image_keys != {"front", "wrist"}:
        raise ValueError(
            f"Source config must use front+wrist cameras, got {sorted(image_keys)}"
        )
    if list(train.get("video_size") or []) != [384, 768]:
        raise ValueError("Source config must resolve video_size to [384, 768]")
    processor = train.get("processor")
    if not isinstance(processor, Mapping):
        raise ValueError("Source config is missing data.train.processor")
    if int(processor.get("action_output_dim", -1)) != 22:
        raise ValueError("Source config action_output_dim must be 22")
    if int(processor.get("proprio_output_dim", -1)) != 23:
        raise ValueError("Source config proprio_output_dim must be 23")
    model_proprio = nested_get(payload, "model", "proprio_dim")
    if int(model_proprio) != 23:
        raise ValueError("Source checkpoint config must have model.proprio_dim=23")
    norm_stats_source = str(
        processor.get("norm_stats_source", "compute")
    ).strip().lower()
    return {
        "camera_keys": sorted(image_keys),
        "video_size": [384, 768],
        "action_dim": 22,
        "proprio_dim": 23,
        "normalization_kind": (
            "meta" if norm_stats_source == "meta" else "compute"
        ),
    }


def validate_dataset_root(root: Path) -> dict[str, Any]:
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Missing LeRobot metadata: {info_path}")
    info = read_json(info_path)
    features = info.get("features")
    if not isinstance(features, Mapping):
        raise ValueError(f"{info_path} has no feature schema")
    required = {
        "action": 22,
        "observation.state": 23,
    }
    for key, expected_dim in required.items():
        feature = features.get(key)
        shape = feature.get("shape") if isinstance(feature, Mapping) else None
        if list(shape or []) != [expected_dim]:
            raise ValueError(
                f"{root} feature {key} must have shape [{expected_dim}], got {shape}"
            )
    for key in (
        "observation.images.front",
        "observation.images.wrist",
    ):
        if key not in features:
            raise ValueError(f"{root} is missing required camera feature {key}")
    return {
        "root": str(root),
        "fps": info.get("fps"),
        "num_episodes": info.get("total_episodes"),
    }


def validate_manifest_protocol(
    manifest: Mapping[str, Any],
    *,
    variant: str,
    targets: PairTargetStore | None,
) -> dict[str, Any]:
    validate_manifest(manifest, strict=True, verify_hash=True)
    counts: dict[str, dict[str, int]] = {}
    positive_pairs = 0
    dataset_roots: set[Path] = set()
    for index, sample in enumerate(manifest.get("samples", [])):
        split = str(sample.get("split", "train"))
        if split != "train":
            raise ValueError(
                f"Formal training manifest sample {index} must use split='train', "
                f"got {split!r}"
            )
        role = sample_role(sample)
        action_enabled = sample.get("action_loss") != "disabled"
        outcome = str(sample.get("episode_outcome"))
        bucket = counts.setdefault(
            split,
            {
                "primary": 0,
                "auxiliary": 0,
                "success": 0,
                "failure": 0,
            },
        )
        bucket[role] += 1
        bucket[outcome] += 1
        if role == "primary" and (outcome != "success" or not action_enabled):
            raise ValueError(
                f"Primary sample {sample.get('sample_id')} must be action-enabled success"
            )
        if role == "auxiliary" and action_enabled:
            raise ValueError(
                f"Auxiliary sample {sample.get('sample_id')} must disable action loss"
            )
        pair_weight = float(sample.get("pair_weight", 0.0))
        if pair_weight > 0.0:
            positive_pairs += 1
            if variant != "M":
                raise ValueError(
                    f"Variant {variant} must not carry positive pair supervision"
                )
            if outcome != "failure":
                raise ValueError(
                    "Main-method pair supervision must attach only to failure events"
                )
            pair_id = str(sample.get("pair_id") or "")
            if targets is None or pair_id not in targets:
                raise ValueError(f"Missing pair target for {pair_id!r}")
            target = targets.get(pair_id)
            if target.split != split:
                raise ValueError(f"Pair {pair_id} crosses manifest split")
            if target.failure_event_id != sample.get("event_id"):
                raise ValueError(
                    f"Pair {pair_id} does not target failure event "
                    f"{sample.get('event_id')}"
                )
        dataset_roots.add(Path(str(sample["dataset_root"])).expanduser().resolve())

    if set(counts) != {"train"}:
        raise ValueError("Formal training manifest must contain train samples")
    if counts["train"]["primary"] == 0 or counts["train"]["auxiliary"] == 0:
        raise ValueError("train must contain both primary and auxiliary samples")
    if variant == "B0":
        if any(bucket["failure"] for bucket in counts.values()):
            raise ValueError("B0 must use success auxiliary data only")
    else:
        if not all(bucket["failure"] > 0 for bucket in counts.values()):
            raise ValueError(f"{variant} requires failure auxiliary data")
    if variant == "M" and positive_pairs == 0:
        raise ValueError("M requires at least one positive failure-event pair")
    if variant != "M" and targets is not None:
        raise ValueError(f"{variant} must not load pair targets")
    return {
        "counts": counts,
        "positive_pair_samples": positive_pairs,
        "dataset_roots": sorted(str(path) for path in dataset_roots),
    }


def validate_selection_manifest(
    manifest: Mapping[str, Any],
    *,
    training_manifests: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Validate the one frozen held-out success set used by every variant."""

    validate_manifest(manifest, strict=True, verify_hash=True)
    train_episodes = {
        (str(sample["dataset_id"]), int(sample["episode_index"]))
        for training_manifest in training_manifests.values()
        for sample in training_manifest.get("samples", [])
        if str(sample.get("split", "train")) == "train"
    }
    selection_episodes: set[tuple[str, int]] = set()
    for index, sample in enumerate(manifest.get("samples", [])):
        if str(sample.get("split", "")) != "val":
            raise ValueError(
                f"Selection sample {index} must use split='val'"
            )
        if sample_role(sample) != "primary":
            raise ValueError(
                f"Selection sample {sample.get('sample_id')} must be primary"
            )
        if sample.get("episode_outcome") != "success":
            raise ValueError(
                f"Selection sample {sample.get('sample_id')} must be success"
            )
        if sample.get("action_loss") == "disabled":
            raise ValueError(
                f"Selection sample {sample.get('sample_id')} must enable action loss"
            )
        if float(sample.get("pair_weight", 0.0)) > 0.0:
            raise ValueError("Selection manifest must not carry pair supervision")
        selection_episodes.add(
            (str(sample["dataset_id"]), int(sample["episode_index"]))
        )
    if not selection_episodes:
        raise ValueError("Selection manifest must contain at least one val episode")
    overlap = train_episodes & selection_episodes
    if overlap:
        raise ValueError(
            f"Training/selection episode leakage detected: {sorted(overlap)[:5]}"
        )
    identities = primary_sample_identities(manifest)
    return (
        {
            "sample_count": len(identities),
            "episode_count": len(selection_episodes),
            "identity_sha256": sha256_json(identities),
        },
        identities,
    )


def validate_protocol_matrix(
    manifests: Mapping[str, Mapping[str, Any]],
    *,
    targets: PairTargetStore | None,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    if set(manifests) != VARIANTS:
        raise ValueError("Protocol matrix must define exactly B0/B1/C/M manifests")
    reports: dict[str, dict[str, Any]] = {}
    identities_by_variant: dict[str, list[dict[str, Any]]] = {}
    for variant in sorted(VARIANTS):
        reports[variant] = validate_manifest_protocol(
            manifests[variant],
            variant=variant,
            targets=targets if variant == "M" else None,
        )
        identities_by_variant[variant] = primary_sample_identities(
            manifests[variant]
        )

    reference = identities_by_variant["B0"]
    reference_hash = sha256_json(reference)
    for variant in ("B1", "C", "M"):
        current = identities_by_variant[variant]
        if current != reference:
            reference_rows = {
                json.dumps(row, sort_keys=True, separators=(",", ":"))
                for row in reference
            }
            current_rows = {
                json.dumps(row, sort_keys=True, separators=(",", ":"))
                for row in current
            }
            missing = sorted(reference_rows - current_rows)[:3]
            extra = sorted(current_rows - reference_rows)[:3]
            raise ValueError(
                f"Primary sample identity mismatch between B0 and {variant}; "
                f"missing={missing}, extra={extra}"
            )
        reports[variant]["primary_identity_sha256"] = reference_hash
        reports[variant]["primary_sample_count"] = len(reference)
    reports["B0"]["primary_identity_sha256"] = reference_hash
    reports["B0"]["primary_sample_count"] = len(reference)

    reference_auxiliary = reports["B1"]["counts"]["train"]["auxiliary"]
    for variant in ("B0", "C", "M"):
        current_auxiliary = reports[variant]["counts"]["train"]["auxiliary"]
        if current_auxiliary != reference_auxiliary:
            raise ValueError(
                "Auxiliary sample budget mismatch between B1 and "
                f"{variant}: B1={reference_auxiliary}, "
                f"{variant}={current_auxiliary}"
            )
    for variant in VARIANTS:
        reports[variant]["auxiliary_sample_count"] = reference_auxiliary
    return reports, reference


def _expect_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label} must be {expected!r}, got {actual!r}")


def _resolved_path(value: Any, *, label: str) -> Path:
    if value in {None, "", "null"}:
        raise ValueError(f"{label} must resolve to a path")
    path = Path(str(value)).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    return path


def _normalize_task_text(task: Any, *, strip_marker: str | None) -> str:
    text = str(task)
    if strip_marker and strip_marker in text:
        text = text.replace(strip_marker, "")
    return " ".join(text.split()).strip()


def _episode_tasks(dataset_root: Path) -> dict[int, str]:
    episodes_path = dataset_root / "meta" / "episodes.jsonl"
    if not episodes_path.is_file():
        raise FileNotFoundError(
            f"Cannot validate text-cache prompts without {episodes_path}"
        )
    tasks: dict[int, str] = {}
    with episodes_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            episode_index = int(row["episode_index"])
            raw_tasks = row.get("tasks")
            if isinstance(raw_tasks, list) and raw_tasks:
                task = str(raw_tasks[0])
            else:
                task = str(row.get("task", ""))
            if not task:
                raise ValueError(
                    f"{episodes_path}:{line_number} has no task text"
                )
            tasks[episode_index] = task
    return tasks


def referenced_task_texts(
    samples: Sequence[Mapping[str, Any]],
    *,
    strip_marker: str | None,
) -> list[str]:
    episode_tasks_by_root: dict[Path, dict[int, str]] = {}
    tasks: set[str] = set()
    for sample in samples:
        dataset_root = Path(
            str(sample["dataset_root"])
        ).expanduser().resolve()
        if dataset_root not in episode_tasks_by_root:
            episode_tasks_by_root[dataset_root] = _episode_tasks(dataset_root)
        episode_index = int(sample["episode_index"])
        raw_task = episode_tasks_by_root[dataset_root].get(episode_index)
        if raw_task is None:
            raise ValueError(
                f"Manifest references missing episode {episode_index} in "
                f"{dataset_root}"
            )
        loader_task = _normalize_task_text(
            raw_task,
            strip_marker=strip_marker,
        )
        manifest_task = _normalize_task_text(
            sample.get("task", loader_task),
            strip_marker=strip_marker,
        )
        if manifest_task != loader_task:
            raise ValueError(
                "Manifest task text disagrees with the underlying LeRobot "
                f"episode after suffix stripping: {manifest_task!r} != "
                f"{loader_task!r}"
            )
        tasks.add(loader_task)
    if not tasks:
        raise ValueError("No referenced task text was found for cache validation")
    return sorted(tasks)


def build_text_cache_contract(
    cache_dir: Path,
    *,
    tasks: Sequence[str],
    context_len: int,
) -> dict[str, Any]:
    cache_dir = cache_dir.expanduser().resolve()
    artifacts: dict[str, dict[str, str]] = {}
    for task in sorted(set(tasks)):
        prompt = (
            "A video recorded from a robot's point of view executing the "
            f"following instruction: {task}"
        )
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        basename = (
            f"{digest}.t5_len{int(context_len)}.wan22ti2v5b.pt"
        )
        artifacts[basename] = artifact_record(
            _resolved_path(
                cache_dir / basename,
                label=f"FastWAM text cache for task {task!r}",
            )
        )
    content = {
        "artifacts": {
            name: record["sha256"]
            for name, record in sorted(artifacts.items())
        }
    }
    return {
        "cache_dir": str(cache_dir),
        "artifacts": artifacts,
        "bundle_sha256": sha256_json(content),
    }


def validate_training_data_contract(
    payload: Mapping[str, Any],
    *,
    expected_dataset_roots: Sequence[str],
    expected_normalization_kind: str,
    referenced_samples: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    train = nested_get(payload, "data", "train")
    val = nested_get(payload, "data", "val")
    if not isinstance(train, Mapping) or not isinstance(val, Mapping):
        raise ValueError("Resolved config must contain data.train and data.val")

    configured_roots = {
        str(Path(str(path)).expanduser().resolve())
        for path in (train.get("dataset_dirs") or [])
    }
    expected_roots = {
        str(Path(path).expanduser().resolve())
        for path in expected_dataset_roots
    }
    if configured_roots != expected_roots:
        raise ValueError(
            "data.train.dataset_dirs must match the frozen manifest roots; "
            f"configured={sorted(configured_roots)}, expected={sorted(expected_roots)}"
        )

    train_processor = train.get("processor")
    val_processor = val.get("processor")
    if not isinstance(train_processor, Mapping) or not isinstance(
        val_processor, Mapping
    ):
        raise ValueError("Resolved config must contain train/val processors")
    train_source = str(
        train_processor.get("norm_stats_source", "compute")
    ).strip().lower()
    val_source = str(
        val_processor.get("norm_stats_source", "compute")
    ).strip().lower()
    normalization_kind = "meta" if train_source == "meta" else "compute"
    _expect_equal(
        "meta" if val_source == "meta" else "compute",
        normalization_kind,
        "train/val normalization kind",
    )
    _expect_equal(
        normalization_kind,
        expected_normalization_kind,
        "offline/source normalization kind",
    )

    normalization_artifacts: dict[str, dict[str, str]]
    if normalization_kind == "meta":
        for split_name, split in (("train", train), ("val", val)):
            _expect_equal(
                split.get("pretrained_norm_stats"),
                None,
                f"data.{split_name}.pretrained_norm_stats",
            )
        train_meta_dir = Path(
            str(train_processor.get("norm_stats_meta_dir"))
        ).expanduser().resolve()
        val_meta_dir = Path(
            str(val_processor.get("norm_stats_meta_dir"))
        ).expanduser().resolve()
        _expect_equal(val_meta_dir, train_meta_dir, "train/val norm_stats_meta_dir")
        stats_path = _resolved_path(
            train_meta_dir / "stats.json",
            label="meta normalization stats.json",
        )
        modality_path = _resolved_path(
            train_meta_dir / "modality.json",
            label="meta normalization modality.json",
        )
        normalization_artifacts = {
            "stats.json": artifact_record(stats_path),
            "modality.json": artifact_record(modality_path),
        }
    else:
        for split_name, processor in (
            ("train", train_processor),
            ("val", val_processor),
        ):
            _expect_equal(
                processor.get("norm_stats_meta_dir"),
                None,
                f"data.{split_name}.processor.norm_stats_meta_dir",
            )
        train_stats = _resolved_path(
            train.get("pretrained_norm_stats"),
            label="data.train.pretrained_norm_stats",
        )
        val_stats = _resolved_path(
            val.get("pretrained_norm_stats"),
            label="data.val.pretrained_norm_stats",
        )
        _expect_equal(val_stats, train_stats, "train/val pretrained_norm_stats")
        normalization_artifacts = {
            "dataset_stats.json": artifact_record(train_stats),
        }

    normalization_content = {
        "kind": normalization_kind,
        "artifacts": {
            name: record["sha256"]
            for name, record in sorted(normalization_artifacts.items())
        },
    }
    normalization_bundle_sha256 = sha256_json(normalization_content)

    train_cache_dir = Path(
        str(train.get("text_embedding_cache_dir"))
    ).expanduser().resolve()
    val_cache_dir = Path(
        str(val.get("text_embedding_cache_dir"))
    ).expanduser().resolve()
    _expect_equal(val_cache_dir, train_cache_dir, "train/val text cache directory")
    _expect_equal(train.get("context_len"), 128, "data.train.context_len")
    _expect_equal(val.get("context_len"), 128, "data.val.context_len")
    strip_marker = train.get("strip_instruction_suffix_if_contains")
    _expect_equal(
        val.get("strip_instruction_suffix_if_contains"),
        strip_marker,
        "train/val instruction suffix stripping",
    )
    text_cache_contract = build_text_cache_contract(
        train_cache_dir,
        tasks=referenced_task_texts(
            referenced_samples,
            strip_marker=(
                None if strip_marker is None else str(strip_marker)
            ),
        ),
        context_len=128,
    )

    return {
        "dataset_roots": sorted(expected_roots),
        "normalization_kind": normalization_kind,
        "normalization_artifacts": normalization_artifacts,
        "normalization_bundle_sha256": normalization_bundle_sha256,
        "text_embedding_cache": text_cache_contract,
    }


def validate_resolved_config(
    path: Path,
    *,
    variant: str,
    execution_mode: str = "formal",
    manifest_path: Path,
    selection_manifest_path: Path,
    init_weights: Path,
    init_weights_sha256: str,
    source_config_sha256: str,
    manifest_sha256: str,
    selection_manifest_sha256: str,
    pair_targets: Path,
    pair_targets_sha256: str,
    teacher_sha256: str,
    code_snapshot_sha256: str,
    expected_dataset_roots: Sequence[str],
    expected_normalization_kind: str,
    expected_normalization_bundle_sha256: str,
    expected_text_cache_sha256: str,
    referenced_samples: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if execution_mode not in EXECUTION_CONTRACTS:
        raise ValueError(f"Unsupported execution mode: {execution_mode!r}")
    execution_contract = EXECUTION_CONTRACTS[execution_mode]
    path = path.expanduser().resolve()
    manifest_path = manifest_path.expanduser().resolve()
    selection_manifest_path = selection_manifest_path.expanduser().resolve()
    init_weights = init_weights.expanduser().resolve()
    pair_targets = pair_targets.expanduser().resolve()
    payload = read_yaml(path)
    training_inputs = validate_training_data_contract(
        payload,
        expected_dataset_roots=expected_dataset_roots,
        expected_normalization_kind=expected_normalization_kind,
        referenced_samples=referenced_samples,
    )
    _expect_equal(
        training_inputs["normalization_bundle_sha256"],
        expected_normalization_bundle_sha256,
        f"{variant} normalization bundle SHA-256",
    )
    _expect_equal(
        training_inputs["text_embedding_cache"]["bundle_sha256"],
        expected_text_cache_sha256,
        f"{variant} text embedding cache SHA-256",
    )
    steer_enabled = variant in {"C", "M"}
    _expect_equal(payload.get("batch_size"), 4, f"{variant} batch_size")
    _expect_equal(
        nested_get(payload, "role_balanced_sampling", "enabled"),
        True,
        f"{variant} role-balanced sampling",
    )
    _expect_equal(
        nested_get(payload, "role_balanced_sampling", "primary_per_batch"),
        2,
        f"{variant} primary_per_batch",
    )
    _expect_equal(payload.get("learning_rate"), 1.0e-4, f"{variant} learning_rate")
    _expect_equal(
        payload.get("max_steps"),
        execution_contract["max_steps"],
        f"{variant} max_steps",
    )
    _expect_equal(
        payload.get("gradient_accumulation_steps"),
        1,
        f"{variant} gradient_accumulation_steps",
    )
    _expect_equal(payload.get("mixed_precision"), "bf16", f"{variant} precision")
    _expect_equal(payload.get("seed"), 42, f"{variant} seed")
    _expect_equal(payload.get("eval_seed"), 20260717, f"{variant} eval_seed")
    _expect_equal(
        payload.get("eval_every"),
        execution_contract["eval_every"],
        f"{variant} eval_every",
    )
    _expect_equal(
        payload.get("best_metric"), "val_base_loss", f"{variant} best_metric"
    )
    _expect_equal(
        payload.get("save_weights_every"),
        execution_contract["save_weights_every"],
        f"{variant} weights save",
    )
    _expect_equal(
        payload.get("save_weight_steps"),
        execution_contract["save_weight_steps"],
        f"{variant} exact weight save steps",
    )
    _expect_equal(
        payload.get("save_state_every"),
        execution_contract["save_state_every"],
        f"{variant} state save",
    )
    _expect_equal(
        payload.get("lr_scheduler_total_steps"),
        execution_contract["lr_scheduler_total_steps"],
        f"{variant} scheduler total steps",
    )
    _expect_equal(payload.get("state_keep_last"), 1, f"{variant} state retention")
    _expect_equal(
        nested_get(payload, "model", "offline_steer", "enabled"),
        steer_enabled,
        f"{variant} steer enabled",
    )
    _expect_equal(
        nested_get(payload, "model", "offline_steer", "pair_loss_weight"),
        0.1 if variant == "M" else 0.0,
        f"{variant} pair loss weight",
    )
    _expect_equal(
        nested_get(payload, "model", "offline_steer", "pair_loss_warmup_steps"),
        500 if variant == "M" else 0,
        f"{variant} pair loss warmup",
    )
    _expect_equal(
        Path(str(payload.get("resume"))).expanduser().resolve(),
        init_weights,
        f"{variant} canonical resume",
    )
    for split in ("train", "val"):
        configured_manifest = nested_get(payload, "data", split, "manifest_path")
        _expect_equal(
            Path(str(configured_manifest)).expanduser().resolve(),
            manifest_path if split == "train" else selection_manifest_path,
            f"{variant} data.{split}.manifest_path",
        )
        configured_targets = nested_get(
            payload, "data", split, "pair_targets_path"
        )
        expected_targets: Any = (
            pair_targets if variant == "M" and split == "train" else None
        )
        if configured_targets is not None:
            configured_targets = Path(str(configured_targets)).expanduser().resolve()
        _expect_equal(
            configured_targets,
            expected_targets,
            f"{variant} data.{split}.pair_targets_path",
        )
        _expect_equal(
            nested_get(payload, "data", split, "expected_teacher_sha256"),
            teacher_sha256 if variant == "M" and split == "train" else None,
            f"{variant} data.{split}.expected_teacher_sha256",
        )

    provenance = payload.get("experiment_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError(f"{variant} resolved config lacks experiment_provenance")
    expected_provenance = {
        "protocol": PROTOCOL_NAME,
        "variant": variant,
        "source_checkpoint_sha256": init_weights_sha256,
        "source_config_sha256": source_config_sha256,
        "manifest_sha256": manifest_sha256,
        "selection_manifest_sha256": selection_manifest_sha256,
        "pair_targets_sha256": (
            pair_targets_sha256 if variant == "M" else "none"
        ),
        "teacher_sha256": teacher_sha256 if variant == "M" else "none",
        "code_snapshot_sha256": code_snapshot_sha256,
        "normalization_kind": expected_normalization_kind,
        "normalization_bundle_sha256": expected_normalization_bundle_sha256,
        "text_embedding_cache_sha256": expected_text_cache_sha256,
    }
    for key, expected in expected_provenance.items():
        _expect_equal(
            provenance.get(key),
            expected,
            f"{variant} experiment_provenance.{key}",
        )
    expected_run_mode = execution_contract["provenance_run_mode"]
    if expected_run_mode is None:
        if "run_mode" in provenance:
            raise ValueError(
                f"{variant} formal experiment_provenance must not contain run_mode"
            )
    else:
        _expect_equal(
            provenance.get("run_mode"),
            expected_run_mode,
            f"{variant} experiment_provenance.run_mode",
        )
    _expect_equal(
        nested_get(payload, "wandb", "enabled"), True, f"{variant} W&B enabled"
    )
    _expect_equal(
        nested_get(payload, "wandb", "mode"), "online", f"{variant} W&B mode"
    )
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "variant": variant,
        "execution_mode": execution_mode,
        "manifest": str(manifest_path),
        "selection_manifest": str(selection_manifest_path),
        "training_inputs": training_inputs,
    }


def validate_resume_state(
    path: Path | None,
    *,
    expected_provenance: Mapping[str, Any] | None = None,
    expected_global_step: int | None = None,
) -> dict[str, Any] | None:
    if path is None:
        return None
    if not path.is_dir():
        raise ValueError(f"RESUME_STATE_DIR must be a directory: {path}")
    meta_path = path / "checkpoint_meta.json"
    trainer_state_path = path / "trainer_state.json"
    for required in (meta_path, trainer_state_path):
        if not required.is_file():
            raise ValueError(f"Resume state is missing {required.name}: {path}")
    meta = read_json(meta_path)
    trainer_state = read_json(trainer_state_path)
    if meta.get("complete") is not True:
        raise ValueError("Resume state checkpoint_meta.json is not complete")
    provenance = meta.get("experiment_provenance")
    provenance_sha256 = meta.get("experiment_provenance_sha256")
    if expected_provenance is not None:
        if not isinstance(provenance, Mapping) or not isinstance(
            provenance_sha256, str
        ):
            raise ValueError(
                "Resume state is not bound to an experiment provenance contract"
            )
        if not hmac.compare_digest(
            provenance_sha256, sha256_json(dict(provenance))
        ):
            raise ValueError("Resume state experiment provenance hash is invalid")
        if dict(provenance) != dict(expected_provenance):
            raise ValueError(
                "Resume state experiment provenance does not match the selected "
                "variant and protocol artifacts"
            )
    meta_step = int(meta.get("global_step", -1))
    trainer_step = int(trainer_state.get("global_step", -2))
    if meta_step < 0 or meta_step != trainer_step:
        raise ValueError(
            "Resume state global_step disagrees between checkpoint_meta.json "
            "and trainer_state.json"
        )
    if expected_global_step is not None and meta_step != expected_global_step:
        raise ValueError(
            f"Resume state global_step must be {expected_global_step}, got {meta_step}"
        )
    model_dir = path / "pytorch_model"
    optimizer_files = sorted(model_dir.glob("*_optim_states.pt"))
    optimizer_ranks: list[int] = []
    for optimizer_file in optimizer_files:
        match = re.search(r"zero_pp_rank_(\d+)_", optimizer_file.name)
        if match is None:
            raise ValueError(
                f"Unrecognized DeepSpeed optimizer shard: {optimizer_file.name}"
            )
        if optimizer_file.stat().st_size <= 0:
            raise ValueError(
                f"DeepSpeed optimizer shard is empty: {optimizer_file}"
            )
        optimizer_ranks.append(int(match.group(1)))
    expected_ranks = list(range(4))
    if sorted(optimizer_ranks) != expected_ranks:
        raise ValueError(
            "Resume state must contain exactly one DeepSpeed optimizer shard "
            f"for ranks 0-3; got ranks={sorted(optimizer_ranks)}"
        )
    model_state_files = sorted(model_dir.glob("*_model_states.pt"))
    if len(model_state_files) != 1 or model_state_files[0].stat().st_size <= 0:
        raise ValueError(
            "Resume state must contain one non-empty DeepSpeed model-state shard"
        )
    random_state_files = []
    for rank in expected_ranks:
        random_state = path / f"random_states_{rank}.pkl"
        if not random_state.is_file() or random_state.stat().st_size <= 0:
            raise ValueError(
                f"Resume state is missing non-empty random_states_{rank}.pkl"
            )
        random_state_files.append(random_state)
    scheduler_path = path / "scheduler.bin"
    if not scheduler_path.is_file() or scheduler_path.stat().st_size <= 0:
        raise ValueError("Resume state is missing non-empty scheduler.bin")
    payload_files = [
        *optimizer_files,
        *model_state_files,
        *random_state_files,
        scheduler_path,
    ]
    return {
        "path": str(path),
        "global_step": meta_step,
        "checkpoint_meta_sha256": sha256_file(meta_path),
        "trainer_state_sha256": sha256_file(trainer_state_path),
        "payload_file_count": len(payload_files),
        "deepspeed_optimizer_ranks": expected_ranks,
        "deepspeed_model_state": str(model_state_files[0]),
        "experiment_provenance": provenance,
        "experiment_provenance_sha256": provenance_sha256,
    }


def validate_execution_resume_request(
    *,
    execution_mode: str,
    resume_state_dir: Path | None,
    expected_resume_step: int | None,
) -> None:
    if execution_mode == "formal":
        if expected_resume_step is not None:
            raise ValueError(
                "Formal execution must not set --expected-resume-step"
            )
        return
    if execution_mode == "smoke20":
        if resume_state_dir is not None or expected_resume_step is not None:
            raise ValueError("smoke20 must start from INIT_WEIGHTS")
        return
    if execution_mode == "smoke500":
        if resume_state_dir is None:
            raise ValueError(
                "smoke500 requires a complete step-20 state via "
                "--resume-state-dir"
            )
        if expected_resume_step != 20:
            raise ValueError(
                "smoke500 resume verification must use a complete step-20 state"
            )
        return
    raise ValueError(f"Unsupported execution mode: {execution_mode!r}")


def artifact_record(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": sha256_file(path)}


def validate_source_bundle_manifest(
    path: Path,
    *,
    init_weights: Path,
    source_config: Path,
    normalization_bundle_sha256: str,
) -> dict[str, Any]:
    path = path.expanduser().resolve()
    init_weights = init_weights.expanduser().resolve()
    source_config = source_config.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    root = path.parent
    if init_weights.parent != root or source_config.parent != root:
        raise ValueError(
            "S0 checkpoint, source config, and source bundle manifest must "
            "come from the same atomic directory"
        )
    metadata: dict[str, str] = {}
    listed: dict[str, str] = {}
    in_hashes = False
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line == "sha256":
            in_hashes = True
            continue
        if not in_hashes:
            if "=" in line:
                key, value = line.split("=", 1)
                metadata[key] = value
            continue
        digest, relative = line.split(None, 1)
        relative = relative.strip()
        while relative.startswith("./"):
            relative = relative[2:]
        artifact_path = (root / relative).resolve()
        if root not in artifact_path.parents:
            raise ValueError(
                f"S0 bundle artifact escapes its root: {relative}"
            )
        if not artifact_path.is_file():
            raise FileNotFoundError(artifact_path)
        actual = sha256_file(artifact_path)
        if not hmac.compare_digest(actual, digest):
            raise ValueError(
                f"S0 bundle artifact hash mismatch: {relative}"
            )
        listed[relative] = digest
    for expected_name, artifact_path in (
        ("step_006500.pt", init_weights),
        ("config.yaml", source_config),
    ):
        expected_hash = listed.get(expected_name)
        if expected_hash is None or not hmac.compare_digest(
            expected_hash, sha256_file(artifact_path)
        ):
            raise ValueError(f"S0 bundle does not bind {expected_name}")
    normalization_kind = metadata.get("normalization_kind")
    if normalization_kind == "meta":
        required_normalization_files = {
            "norm_stats_meta/stats.json",
            "norm_stats_meta/modality.json",
        }
    elif normalization_kind == "dataset":
        required_normalization_files = {"dataset_stats.json"}
    else:
        raise ValueError(
            f"S0 bundle has invalid normalization_kind={normalization_kind!r}"
        )
    missing_normalization = required_normalization_files - set(listed)
    if missing_normalization:
        raise ValueError(
            "S0 bundle is missing normalization artifacts: "
            f"{sorted(missing_normalization)}"
        )
    if not any(
        relative.startswith("text_cache/") and relative.endswith(".npz")
        for relative in listed
    ):
        raise ValueError("S0 bundle does not contain a rollout text-cache .npz")
    _expect_equal(
        metadata.get("normalization_bundle_sha256"),
        normalization_bundle_sha256,
        "S0 bundle normalization SHA-256",
    )
    return {
        **artifact_record(path),
        "normalization_bundle_sha256": normalization_bundle_sha256,
        "artifact_count": len(listed),
    }


def build_protocol_bundle(
    *,
    init_weights: Path,
    source_config: Path,
    manifest_paths: Mapping[str, Path],
    manifests: Mapping[str, Mapping[str, Any]],
    manifest_reports: Mapping[str, Mapping[str, Any]],
    selection_manifest: Path,
    selection_manifest_payload: Mapping[str, Any],
    selection_report: Mapping[str, Any],
    resolved_config_reports: Mapping[str, Mapping[str, Any]],
    pair_targets: Path,
    teacher_checkpoint: Path,
    teacher_sha256: str,
    primary_identities: Sequence[Mapping[str, Any]],
    code_snapshot: Mapping[str, Any],
    training_inputs: Mapping[str, Any],
    source_bundle_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "format": BUNDLE_FORMAT,
        "version": BUNDLE_VERSION,
        "protocol": PROTOCOL_NAME,
        "init_weights": artifact_record(init_weights),
        "source_config": artifact_record(source_config),
        "source_bundle_manifest": copy.deepcopy(
            dict(source_bundle_manifest)
        ),
        "manifests": {},
        "selection_manifest": {
            "path": str(selection_manifest),
            "file_sha256": sha256_file(selection_manifest),
            "manifest_hash": selection_manifest_payload["manifest_hash"],
            **dict(selection_report),
        },
        "resolved_configs": {
            variant: dict(resolved_config_reports[variant])
            for variant in sorted(VARIANTS)
        },
        "pair_targets": artifact_record(pair_targets),
        "teacher_checkpoint": artifact_record(teacher_checkpoint),
        "teacher_sha256": teacher_sha256,
        "code_snapshot": copy.deepcopy(dict(code_snapshot)),
        "training_inputs": copy.deepcopy(dict(training_inputs)),
        "primary_samples": {
            "count": len(primary_identities),
            "identity_sha256": sha256_json(list(primary_identities)),
        },
    }
    for variant in sorted(VARIANTS):
        payload["manifests"][variant] = {
            "path": str(manifest_paths[variant]),
            "file_sha256": sha256_file(manifest_paths[variant]),
            "manifest_hash": manifests[variant]["manifest_hash"],
            "primary_sample_count": manifest_reports[variant][
                "primary_sample_count"
            ],
            "primary_identity_sha256": manifest_reports[variant][
                "primary_identity_sha256"
            ],
        }
    payload["bundle_sha256"] = sha256_json(payload)
    return payload


def write_or_validate_protocol_bundle(
    path: Path,
    expected: Mapping[str, Any],
) -> str:
    if path.exists():
        actual = read_json(path)
        stored_hash = actual.get("bundle_sha256")
        unhashed = dict(actual)
        unhashed.pop("bundle_sha256", None)
        computed_hash = sha256_json(unhashed)
        if not isinstance(stored_hash, str) or not hmac.compare_digest(
            stored_hash, computed_hash
        ):
            raise ValueError(f"Protocol bundle hash is invalid: {path}")
        if actual != expected:
            raise ValueError(
                "Protocol bundle does not match current artifacts; use a new "
                "PROTOCOL_BUNDLE_PATH for a changed experiment protocol"
            )
        return "validated"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(expected, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
    return "created"


def check_gpus(gpus: Sequence[int], min_free_mib: int) -> list[dict[str, int]]:
    if len(gpus) != 4 or len(set(gpus)) != 4:
        raise ValueError("Formal runs require exactly four distinct GPUs")
    command = [
        "nvidia-smi",
        "--query-gpu=index,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    output = subprocess.check_output(command, text=True)
    available: dict[int, tuple[int, int]] = {}
    for line in output.splitlines():
        index, free, utilization = [int(value.strip()) for value in line.split(",")]
        available[index] = (free, utilization)
    result = []
    for index in gpus:
        if index not in available:
            raise ValueError(f"GPU {index} is absent from nvidia-smi")
        free, utilization = available[index]
        if free < min_free_mib:
            raise ValueError(
                f"GPU {index} has only {free} MiB free; require {min_free_mib}"
            )
        result.append(
            {"index": index, "memory_free_mib": free, "utilization": utilization}
        )
    return result


def wandb_ready() -> bool:
    if importlib.util.find_spec("wandb") is None:
        return False
    if os.environ.get("WANDB_API_KEY"):
        return True
    netrc = Path.home() / ".netrc"
    return netrc.is_file() and "api.wandb.ai" in netrc.read_text(
        encoding="utf-8", errors="ignore"
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument(
        "--protocol-manifest",
        action="append",
        default=[],
        metavar="VARIANT=PATH",
        help="Repeat exactly once for B0, B1, C, and M.",
    )
    parser.add_argument(
        "--resolved-config",
        action="append",
        default=[],
        metavar="VARIANT=PATH",
        help="Canonical resolved config; repeat for B0, B1, C, and M.",
    )
    parser.add_argument("--init-weights", type=Path, required=True)
    parser.add_argument("--resume-state-dir", type=Path)
    parser.add_argument("--expected-resume-step", type=int)
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--source-bundle-manifest", type=Path, required=True)
    parser.add_argument("--pair-targets", type=Path, required=True)
    parser.add_argument("--teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-teacher-sha256")
    parser.add_argument(
        "--expected-normalization-bundle-sha256",
        required=True,
    )
    parser.add_argument("--expected-text-cache-sha256", required=True)
    parser.add_argument("--protocol-bundle", type=Path, required=True)
    parser.add_argument(
        "--execution-mode",
        choices=sorted(EXECUTION_MODES),
        default="formal",
    )
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--min-gpu-free-mib", type=int, default=75_000)
    parser.add_argument("--disk-root", type=Path, default=Path("/data_all"))
    parser.add_argument("--min-disk-free-gib", type=float, default=500.0)
    parser.add_argument("--skip-system-checks", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest_path = args.manifest.expanduser().resolve()
    selection_manifest_path = args.selection_manifest.expanduser().resolve()
    init_weights = args.init_weights.expanduser().resolve()
    resume_state_dir = (
        None
        if args.resume_state_dir is None
        else args.resume_state_dir.expanduser().resolve()
    )
    validate_execution_resume_request(
        execution_mode=args.execution_mode,
        resume_state_dir=resume_state_dir,
        expected_resume_step=args.expected_resume_step,
    )
    source_config = args.source_config.expanduser().resolve()
    source_bundle_manifest_path = (
        args.source_bundle_manifest.expanduser().resolve()
    )
    pair_path = args.pair_targets.expanduser().resolve()
    teacher_checkpoint = args.teacher_checkpoint.expanduser().resolve()
    protocol_bundle_path = args.protocol_bundle.expanduser().resolve()
    manifest_paths = parse_variant_paths(
        args.protocol_manifest, label="--protocol-manifest"
    )
    resolved_config_paths = parse_variant_paths(
        args.resolved_config, label="--resolved-config"
    )
    for path in (
        manifest_path,
        init_weights,
        source_config,
        source_bundle_manifest_path,
        pair_path,
        teacher_checkpoint,
        selection_manifest_path,
        *manifest_paths.values(),
        *resolved_config_paths.values(),
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if manifest_paths[args.variant] != manifest_path:
        raise ValueError(
            f"--manifest must match --protocol-manifest {args.variant}=..."
        )

    target_store = PairTargetStore(
        pair_path,
        expected_teacher_sha256=args.expected_teacher_sha256,
    )
    try:
        teacher_sha256 = sha256_file(teacher_checkpoint)
        if not hmac.compare_digest(
            teacher_sha256.lower(), target_store.teacher_sha256.lower()
        ):
            raise ValueError(
                "Pair-target teacher_sha256 does not match --teacher-checkpoint"
            )
        manifests = {
            variant: read_json(path)
            for variant, path in manifest_paths.items()
        }
        selection_manifest = read_json(selection_manifest_path)
        manifest_reports, primary_identities = validate_protocol_matrix(
            manifests,
            targets=target_store,
        )
        selection_report, selection_identities = validate_selection_manifest(
            selection_manifest,
            training_manifests=manifests,
        )
        manifest = manifests[args.variant]
        protocol = manifest_reports[args.variant]
        all_dataset_roots = sorted(
            {
                root
                for report in manifest_reports.values()
                for root in report["dataset_roots"]
            }
        )
        datasets = [
            validate_dataset_root(Path(root))
            for root in all_dataset_roots
        ]
        source_contract = validate_source_config(source_config)
        init_weights_sha256 = sha256_file(init_weights)
        source_config_sha256 = sha256_file(source_config)
        pair_targets_sha256 = sha256_file(pair_path)
        code_snapshot = build_code_snapshot()
        code_snapshot_sha256 = str(code_snapshot["snapshot_sha256"])
        resolved_config_reports: dict[str, dict[str, Any]] = {}
        for variant in sorted(VARIANTS):
            resolved_config_reports[variant] = validate_resolved_config(
                resolved_config_paths[variant],
                variant=variant,
                execution_mode=args.execution_mode,
                manifest_path=manifest_paths[variant],
                selection_manifest_path=selection_manifest_path,
                init_weights=init_weights,
                init_weights_sha256=init_weights_sha256,
                source_config_sha256=source_config_sha256,
                manifest_sha256=sha256_file(manifest_paths[variant]),
                selection_manifest_sha256=sha256_file(
                    selection_manifest_path
                ),
                pair_targets=pair_path,
                pair_targets_sha256=pair_targets_sha256,
                teacher_sha256=teacher_sha256,
                code_snapshot_sha256=code_snapshot_sha256,
                expected_dataset_roots=manifest_reports[variant][
                    "dataset_roots"
                ],
                expected_normalization_kind=source_contract[
                    "normalization_kind"
                ],
                expected_normalization_bundle_sha256=(
                    args.expected_normalization_bundle_sha256
                ),
                expected_text_cache_sha256=args.expected_text_cache_sha256,
                referenced_samples=[
                    sample
                    for protocol_manifest in manifests.values()
                    for sample in protocol_manifest.get("samples", [])
                ]
                + list(selection_manifest.get("samples", [])),
            )
        source_bundle_manifest = validate_source_bundle_manifest(
            source_bundle_manifest_path,
            init_weights=init_weights,
            source_config=source_config,
            normalization_bundle_sha256=(
                args.expected_normalization_bundle_sha256
            ),
        )
        training_inputs = resolved_config_reports["B0"]["training_inputs"]
        for variant in ("B1", "C", "M"):
            if (
                resolved_config_reports[variant]["training_inputs"]
                != training_inputs
            ):
                raise ValueError(
                    f"{variant} training inputs differ from B0"
                )
        bundle = build_protocol_bundle(
            init_weights=init_weights,
            source_config=source_config,
            manifest_paths=manifest_paths,
            manifests=manifests,
            manifest_reports=manifest_reports,
            selection_manifest=selection_manifest_path,
            selection_manifest_payload=selection_manifest,
            selection_report=selection_report,
            resolved_config_reports=resolved_config_reports,
            pair_targets=pair_path,
            teacher_checkpoint=teacher_checkpoint,
            teacher_sha256=teacher_sha256,
            primary_identities=primary_identities,
            code_snapshot=code_snapshot,
            training_inputs=training_inputs,
            source_bundle_manifest=source_bundle_manifest,
        )
        bundle_status = write_or_validate_protocol_bundle(
            protocol_bundle_path, bundle
        )
        selected_resolved_config = read_yaml(
            resolved_config_paths[args.variant]
        )
        expected_resume_provenance = selected_resolved_config.get(
            "experiment_provenance"
        )
        if not isinstance(expected_resume_provenance, Mapping):
            raise ValueError(
                f"{args.variant} resolved config lacks experiment_provenance"
            )
        resume_state = validate_resume_state(
            resume_state_dir,
            expected_provenance=expected_resume_provenance,
            expected_global_step=args.expected_resume_step,
        )
        system: dict[str, Any] = {"checked": not args.skip_system_checks}
        if not args.skip_system_checks:
            gpus = [int(value) for value in args.gpus.split(",") if value.strip()]
            system["gpus"] = check_gpus(gpus, args.min_gpu_free_mib)
            disk_root = args.disk_root.expanduser().resolve()
            free_gib = shutil.disk_usage(disk_root).free / (1024**3)
            if free_gib < args.min_disk_free_gib:
                raise ValueError(
                    f"{disk_root} has {free_gib:.1f} GiB free; "
                    f"require {args.min_disk_free_gib:.1f}"
                )
            system["disk_root"] = str(disk_root)
            system["disk_free_gib"] = free_gib
            if not wandb_ready():
                raise RuntimeError(
                    "W&B is not installed or no saved authentication was found"
                )
            system["wandb_ready"] = True

        report = {
            "status": "passed",
            "variant": args.variant,
            "execution_mode": args.execution_mode,
            "manifest": str(manifest_path),
            "manifest_file_sha256": sha256_file(manifest_path),
            "manifest_hash": manifest["manifest_hash"],
            "selection_manifest": str(selection_manifest_path),
            "selection_manifest_file_sha256": sha256_file(
                selection_manifest_path
            ),
            "selection_manifest_hash": selection_manifest["manifest_hash"],
            "selection": selection_report,
            "selection_identities": selection_identities,
            "init_weights": str(init_weights),
            "init_weights_sha256": init_weights_sha256,
            "resume_state": resume_state,
            "load_mode": (
                "resume_full_state" if resume_state is not None else "init_weights"
            ),
            "source_config": str(source_config),
            "source_config_sha256": source_config_sha256,
            "source_bundle_manifest": source_bundle_manifest,
            "source_contract": source_contract,
            "pair_targets": str(pair_path),
            "pair_targets_sha256": pair_targets_sha256,
            "teacher_checkpoint": str(teacher_checkpoint),
            "teacher_sha256": teacher_sha256,
            "code_snapshot": code_snapshot,
            "code_snapshot_sha256": code_snapshot_sha256,
            "training_inputs": training_inputs,
            "protocol_bundle": str(protocol_bundle_path),
            "protocol_bundle_status": bundle_status,
            "protocol_bundle_sha256": bundle["bundle_sha256"],
            "protocol_matrix": manifest_reports,
            "resolved_configs": resolved_config_reports,
            "protocol": protocol,
            "datasets": datasets,
            "system": system,
        }
    finally:
        target_store.close()

    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, output)
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
