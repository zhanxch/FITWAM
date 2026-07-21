#!/usr/bin/env python3
"""Build one immutable initialization checkpoint for paired FITWAM runs.

The builder instantiates the target offline-steer model exactly once, after
seeding Python, NumPy, and PyTorch.  It then loads a steer-free S0 checkpoint
and saves the resulting complete FastWAM checkpoint.  Original M and
M_PAIR_SHUFFLE must both load this same output file, so their model
initialization is identical by construction rather than merely reproducible in
expectation.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
import random
import sys
import tempfile
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]
for source_root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))


ARTIFACT_FORMAT = "FastWAMCommonInitialization"
ARTIFACT_VERSION = "1.0"
STEER_CHECKPOINT_KEYS = {
    "offline_steer_student",
    "offline_steer_residual",
    "offline_steer_config",
}


@dataclass(frozen=True)
class RuntimeDependencies:
    torch: Any
    numpy: Any
    omega_conf: Any
    instantiate: Any


def _load_runtime_dependencies() -> RuntimeDependencies:
    import numpy as np
    import torch
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    return RuntimeDependencies(
        torch=torch,
        numpy=np,
        omega_conf=OmegaConf,
        instantiate=instantiate,
    )


def _canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expect_sha256(actual: str, expected: str | None, label: str) -> None:
    if expected is None:
        return
    normalized = expected.strip().lower()
    if len(normalized) != 64 or any(ch not in "0123456789abcdef" for ch in normalized):
        raise ValueError(f"{label} expected SHA256 is malformed: {expected!r}")
    if actual != normalized:
        raise ValueError(f"{label} SHA256 mismatch: expected={normalized} actual={actual}")


def _load_checkpoint_payload(torch_module: Any, path: Path) -> Mapping[str, Any]:
    attempts = (
        {"map_location": "cpu", "weights_only": True, "mmap": True},
        {"map_location": "cpu", "weights_only": True},
        {"map_location": "cpu"},
    )
    last_error: Exception | None = None
    for kwargs in attempts:
        try:
            payload = torch_module.load(str(path), **kwargs)
            break
        except (TypeError, RuntimeError) as error:
            last_error = error
    else:
        assert last_error is not None
        raise last_error
    if not isinstance(payload, Mapping):
        raise ValueError(f"Checkpoint must contain a mapping: {path}")
    return payload


def _shape(value: Any) -> tuple[int, ...]:
    return tuple(int(dimension) for dimension in value.shape)


def _state_schema(state: Mapping[str, Any], label: str) -> dict[str, Any]:
    if not isinstance(state, Mapping) or not state:
        raise ValueError(f"{label} must be a non-empty state dict")
    entries: dict[str, dict[str, Any]] = {}
    total_numel = 0
    for key in sorted(state):
        value = state[key]
        if not hasattr(value, "shape") or not hasattr(value, "dtype"):
            raise ValueError(f"{label}[{key!r}] is not tensor-like")
        tensor_shape = _shape(value)
        numel = math.prod(tensor_shape)
        total_numel += numel
        entries[str(key)] = {
            "shape": list(tensor_shape),
            "dtype": str(value.dtype),
            "numel": numel,
        }
    return {
        "num_tensors": len(entries),
        "num_parameters": total_numel,
        "schema_sha256": _sha256_bytes(_canonical_json_bytes(entries)),
    }


def _require_exact_compatibility(
    target: Mapping[str, Any],
    source: Mapping[str, Any],
    label: str,
) -> None:
    if not isinstance(source, Mapping):
        raise ValueError(f"Baseline checkpoint section {label!r} is not a state dict")
    target_keys = set(target)
    source_keys = set(source)
    missing = sorted(target_keys - source_keys)
    extra = sorted(source_keys - target_keys)
    shape_mismatches = [
        {
            "key": key,
            "baseline": list(_shape(source[key])),
            "target": list(_shape(target[key])),
        }
        for key in sorted(target_keys & source_keys)
        if _shape(source[key]) != _shape(target[key])
    ]
    if missing or extra or shape_mismatches:
        raise ValueError(
            f"Baseline {label} is not exactly compatible with the target model: "
            f"missing={missing[:8]} extra={extra[:8]} "
            f"shape_mismatches={shape_mismatches[:8]}"
        )


def _clone_state(state: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): value.detach().cpu().clone()
        for key, value in state.items()
    }


def _states_equal(torch_module: Any, left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    if set(left) != set(right):
        return False
    return all(torch_module.equal(left[key], right[key].detach().cpu()) for key in left)


def _assert_finite(torch_module: Any, state: Mapping[str, Any], label: str) -> None:
    for key, value in state.items():
        if not bool(torch_module.isfinite(value).all()):
            raise ValueError(f"{label}[{key!r}] contains non-finite values")


def _assert_all_zero(torch_module: Any, state: Mapping[str, Any], label: str) -> None:
    for key, value in state.items():
        if int(torch_module.count_nonzero(value).item()) != 0:
            raise ValueError(f"{label}[{key!r}] is not zero-initialized")


def _seed_before_instantiation(dependencies: RuntimeDependencies, seed: int) -> None:
    random.seed(seed)
    dependencies.numpy.random.seed(seed)
    dependencies.torch.manual_seed(seed)
    if dependencies.torch.cuda.is_available():
        dependencies.torch.cuda.manual_seed_all(seed)


def _model_dtype(torch_module: Any, name: str) -> Any:
    mapping = {
        "bf16": torch_module.bfloat16,
        "fp16": torch_module.float16,
        "fp32": torch_module.float32,
    }
    return mapping[name]


def _load_resolved_config(dependencies: RuntimeDependencies, path: Path) -> tuple[Any, dict[str, Any]]:
    cfg = dependencies.omega_conf.load(str(path))
    dependencies.omega_conf.resolve(cfg)
    payload = dependencies.omega_conf.to_container(cfg, resolve=True)
    if not isinstance(payload, dict):
        raise ValueError("Resolved config must contain a top-level mapping")
    if "model" not in payload or not isinstance(payload["model"], dict):
        raise ValueError("Resolved config must contain a `model` mapping")
    if "${" in json.dumps(payload, ensure_ascii=True, default=str):
        raise ValueError("Config still contains unresolved interpolation")
    return cfg, payload


def _validate_config(
    config_payload: Mapping[str, Any],
    *,
    baseline_path: Path,
    baseline_sha256: str,
    requested_seed: int | None,
    requested_dtype: str,
) -> tuple[int, str]:
    model = config_payload["model"]
    steer = model.get("offline_steer")
    if not isinstance(steer, Mapping) or steer.get("enabled") is not True:
        raise ValueError("Target config must set model.offline_steer.enabled=true")
    if model.get("skip_dit_load_from_pretrain") is not True:
        raise ValueError(
            "Target config must set model.skip_dit_load_from_pretrain=true; "
            "S0 supplies the complete MoT weights"
        )
    if config_payload.get("resume_experts") not in (None, []):
        raise ValueError("Common initialization requires a full checkpoint load; resume_experts must be null")

    configured_resume = config_payload.get("resume")
    if not isinstance(configured_resume, str) or not configured_resume.strip():
        raise ValueError("Resolved config must bind top-level `resume` to the S0 checkpoint")
    if Path(configured_resume).expanduser().resolve() != baseline_path.resolve():
        raise ValueError(
            "Resolved config `resume` does not bind the supplied S0 checkpoint: "
            f"config={configured_resume} supplied={baseline_path}"
        )

    provenance = config_payload.get("experiment_provenance")
    if isinstance(provenance, Mapping):
        configured_hash = provenance.get("source_checkpoint_sha256")
        if configured_hash not in (None, "") and configured_hash != baseline_sha256:
            raise ValueError(
                "experiment_provenance.source_checkpoint_sha256 does not match S0: "
                f"config={configured_hash} actual={baseline_sha256}"
            )

    configured_seed = config_payload.get("seed")
    if configured_seed is None and requested_seed is None:
        raise ValueError("Provide --seed or a top-level config seed")
    seed = int(configured_seed if requested_seed is None else requested_seed)
    if configured_seed is not None and int(configured_seed) != seed:
        raise ValueError(
            f"Requested seed {seed} disagrees with resolved config seed {configured_seed}"
        )
    if seed < 0:
        raise ValueError("Seed must be non-negative")

    if requested_dtype == "auto":
        precision = str(config_payload.get("mixed_precision", "")).strip().lower()
        dtype_by_precision = {"bf16": "bf16", "fp16": "fp16", "no": "fp32"}
        if precision not in dtype_by_precision:
            raise ValueError(
                "--model-dtype=auto requires mixed_precision in {bf16, fp16, no}"
            )
        dtype_name = dtype_by_precision[precision]
    else:
        dtype_name = requested_dtype
    return seed, dtype_name


def _validate_baseline(model: Any, payload: Mapping[str, Any]) -> dict[str, Any]:
    present_steer = sorted(STEER_CHECKPOINT_KEYS.intersection(payload))
    if present_steer:
        raise ValueError(
            "S0 baseline must be steer-free; found checkpoint keys "
            f"{present_steer}"
        )
    if "mot" not in payload:
        raise ValueError("S0 baseline must contain complete `mot` weights")
    _require_exact_compatibility(model.mot.state_dict(), payload["mot"], "mot")

    for name in ("proprio_encoder", "outcome_encoder"):
        module = getattr(model, name, None)
        if module is None:
            if name in payload:
                raise ValueError(
                    f"S0 contains {name}, but the target model does not instantiate it"
                )
            continue
        if name not in payload:
            raise ValueError(
                f"Target model has {name}, but S0 does not contain its weights"
            )
        _require_exact_compatibility(
            module.state_dict(), payload[name], name
        )

    return {
        "step": payload.get("step"),
        "top_level_keys": sorted(str(key) for key in payload),
        "mot": _state_schema(payload["mot"], "baseline.mot"),
    }


def _validate_output_payload(model: Any, payload: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "mot",
        "step",
        "torch_dtype",
        "offline_steer_student",
        "offline_steer_residual",
        "offline_steer_config",
    }
    if getattr(model, "proprio_encoder", None) is not None:
        required.add("proprio_encoder")
    if getattr(model, "outcome_encoder", None) is not None:
        required.add("outcome_encoder")
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"Common initialization checkpoint is incomplete: missing={missing}")
    if "optimizer" in payload:
        raise ValueError("Common initialization must be weight-only and contain no optimizer")
    if payload.get("step") != 0:
        raise ValueError(f"Common initialization checkpoint step must be 0, got {payload.get('step')}")

    sections = {
        "mot": model.mot,
        "offline_steer_student": model.offline_steer_student,
        "offline_steer_residual": model.offline_steer_residual,
    }
    for name in ("proprio_encoder", "outcome_encoder"):
        module = getattr(model, name, None)
        if module is not None:
            sections[name] = module

    summaries: dict[str, Any] = {}
    for name, module in sections.items():
        _require_exact_compatibility(module.state_dict(), payload[name], name)
        summaries[name] = _state_schema(payload[name], f"output.{name}")

    expected_config = dict(model.offline_steer_config)
    if dict(payload["offline_steer_config"]) != expected_config:
        raise ValueError("Saved offline_steer_config differs from the instantiated model")
    return {
        "top_level_keys": sorted(str(key) for key in payload),
        "sections": summaries,
    }


def _link_file_no_overwrite(temporary: Path, output: Path) -> None:
    try:
        os.link(temporary, output)
    except FileExistsError as error:
        raise FileExistsError(f"Refusing to overwrite existing artifact: {output}") from error


def _write_json_no_overwrite(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp-",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(
                payload,
                stream,
                ensure_ascii=True,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        _link_file_no_overwrite(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_model_checkpoint_no_overwrite(model: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp-",
        suffix=".pt",
        dir=path.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        model.save_checkpoint(str(temporary), optimizer=None, step=0)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        _link_file_no_overwrite(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def run(
    args: argparse.Namespace,
    *,
    dependencies: RuntimeDependencies | None = None,
) -> dict[str, Any]:
    dependencies = dependencies or _load_runtime_dependencies()
    config_path = Path(args.resolved_config).expanduser().resolve()
    baseline_path = Path(args.baseline_checkpoint).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    proof_path = Path(args.proof_output).expanduser().resolve()

    for path, label in (
        (config_path, "resolved config"),
        (baseline_path, "baseline checkpoint"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {label}: {path}")
    if output_path == proof_path:
        raise ValueError("Checkpoint output and proof output must be different paths")
    if output_path in {config_path, baseline_path} or proof_path in {config_path, baseline_path}:
        raise ValueError("Output paths must not alias either input")
    for path in (output_path, proof_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing artifact: {path}")

    config_sha256 = _sha256_file(config_path)
    baseline_sha256 = _sha256_file(baseline_path)
    _expect_sha256(config_sha256, args.expected_config_sha256, "resolved config")
    _expect_sha256(baseline_sha256, args.expected_baseline_sha256, "baseline checkpoint")

    cfg, config_payload = _load_resolved_config(dependencies, config_path)
    seed, dtype_name = _validate_config(
        config_payload,
        baseline_path=baseline_path,
        baseline_sha256=baseline_sha256,
        requested_seed=args.seed,
        requested_dtype=args.model_dtype,
    )

    _seed_before_instantiation(dependencies, seed)
    model = dependencies.instantiate(
        cfg.model if hasattr(cfg, "model") else cfg["model"],
        model_dtype=_model_dtype(dependencies.torch, dtype_name),
        device=args.device,
    )
    if not getattr(model, "offline_steer_enabled", False):
        raise ValueError("Instantiated model does not have offline steer enabled")
    if getattr(model, "offline_steer_student", None) is None:
        raise ValueError("Instantiated model is missing offline_steer_student")
    if getattr(model, "offline_steer_residual", None) is None:
        raise ValueError("Instantiated model is missing offline_steer_residual")

    student_before = _clone_state(model.offline_steer_student.state_dict())
    residual_before = _clone_state(model.offline_steer_residual.state_dict())
    _assert_finite(dependencies.torch, student_before, "offline_steer_student")
    _assert_finite(dependencies.torch, residual_before, "offline_steer_residual")
    _assert_all_zero(
        dependencies.torch,
        residual_before,
        "offline_steer_residual",
    )

    baseline_payload = _load_checkpoint_payload(dependencies.torch, baseline_path)
    baseline_summary = _validate_baseline(model, baseline_payload)
    del baseline_payload
    gc.collect()

    model.load_checkpoint(str(baseline_path), optimizer=None, experts=None)
    if not _states_equal(
        dependencies.torch,
        student_before,
        model.offline_steer_student.state_dict(),
    ):
        raise RuntimeError("Loading S0 changed the seeded Student initialization")
    if not _states_equal(
        dependencies.torch,
        residual_before,
        model.offline_steer_residual.state_dict(),
    ):
        raise RuntimeError("Loading S0 changed the zero-initialized residual")

    linked_output = False
    try:
        _write_model_checkpoint_no_overwrite(model, output_path)
        linked_output = True
        output_payload = _load_checkpoint_payload(dependencies.torch, output_path)
        output_summary = _validate_output_payload(model, output_payload)
        del output_payload
        gc.collect()
        output_sha256 = _sha256_file(output_path)

        report: dict[str, Any] = {
            "format": ARTIFACT_FORMAT,
            "schema_version": ARTIFACT_VERSION,
            "seed": seed,
            "device": str(args.device),
            "model_dtype": dtype_name,
            "inputs": {
                "resolved_config": {
                    "path": str(config_path),
                    "sha256": config_sha256,
                },
                "baseline_checkpoint": {
                    "path": str(baseline_path),
                    "sha256": baseline_sha256,
                    **baseline_summary,
                },
                "builder_script": {
                    "path": str(Path(__file__).resolve()),
                    "sha256": _sha256_file(Path(__file__).resolve()),
                },
            },
            "output": {
                "checkpoint": {
                    "path": str(output_path),
                    "sha256": output_sha256,
                    "size_bytes": output_path.stat().st_size,
                    "checkpoint_format": "FastWAM.save_checkpoint",
                    "step": 0,
                    **output_summary,
                }
            },
            "model": {
                "target": config_payload["model"].get("_target_"),
                "class": f"{type(model).__module__}.{type(model).__qualname__}",
                "offline_steer_config": copy.deepcopy(model.offline_steer_config),
            },
            "invariants": {
                "seeded_before_model_instantiation": True,
                "full_s0_load": True,
                "baseline_exact_structure_match": True,
                "baseline_is_steer_free": True,
                "steer_unchanged_by_s0_load": True,
                "residual_is_zero_initialized": True,
                "complete_weight_only_checkpoint": True,
                "no_overwrite": True,
            },
        }
        report["proof_sha256"] = _sha256_bytes(_canonical_json_bytes(report))
        _write_json_no_overwrite(proof_path, report)
        return report
    except Exception:
        if linked_output and output_path.exists():
            output_path.unlink()
        raise


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolved-config", type=Path, required=True)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--proof-output", type=Path, required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--model-dtype",
        choices=("auto", "bf16", "fp16", "fp32"),
        default="auto",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--expected-config-sha256")
    parser.add_argument("--expected-baseline-sha256")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = run(args)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
