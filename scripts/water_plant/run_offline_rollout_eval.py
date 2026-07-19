#!/usr/bin/env python3
"""Run one fail-closed FITWAM Water Plant offline checkpoint evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = Path(__file__).resolve()

EPISODES = 50
REPLAN_STEPS = 25
MAX_ENV_STEPS = 1500
GPU_IDS = (0, 1, 2, 3)
TASK = "water_plant"
HASH_CHUNK_BYTES = 8 * 1024 * 1024

CODE_FILE_PATHS = {
    "wrapper": SCRIPT_PATH,
    "multi_gpu_evaluator": Path("scripts/dexjoco_async/run_multi_gpu_dexjoco_eval.py"),
    "eval_client": Path("scripts/dexjoco_async/eval_dexjoco_fastwam_control.py"),
    "eval_summary_aggregator": Path(
        "scripts/dexjoco_async/eval_summary_aggregator.py"
    ),
    "multi_gpu_eval_utils": Path("scripts/dexjoco_async/multi_gpu_eval_utils.py"),
    "policy_server_async": Path("scripts/run_fastwam_server_async.py"),
    "policy_server": Path("scripts/run_fastwam_server.py"),
    "validator": Path("scripts/water_plant/validate_offline_rollout_eval.py"),
}


class ProtocolError(RuntimeError):
    """Raised when a formal rollout input violates the frozen protocol."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one formal FITWAM Water Plant offline rollout evaluation."
    )
    parser.add_argument("--variant", choices=("B0", "B1", "C", "M"), required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-step", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--base-seed", type=int, required=True)
    parser.add_argument(
        "--inference-seed",
        type=int,
        default=None,
        help="Fixed policy-inference seed. Defaults to --base-seed.",
    )
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--env-prefix", type=Path, required=True)
    parser.add_argument("--dexjoco-root", type=Path, required=True)
    normalization = parser.add_mutually_exclusive_group()
    normalization.add_argument("--norm-stats-meta-dir", type=Path)
    normalization.add_argument("--dataset-stats-path", type=Path)
    parser.add_argument("--text-cache-dir", type=Path)
    return parser.parse_args(argv)


def _stat_signature(metadata: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_size,
        metadata.st_mtime_ns,
    )


def _open_stable_file(path: Path) -> tuple[Any, os.stat_result]:
    try:
        before = path.stat()
    except FileNotFoundError as exc:
        raise ProtocolError(f"Required file does not exist: {path}") from exc
    if not stat.S_ISREG(before.st_mode):
        raise ProtocolError(f"Expected a regular file: {path}")
    if before.st_size <= 0:
        raise ProtocolError(f"Required file is empty: {path}")

    stream = path.open("rb")
    opened = os.fstat(stream.fileno())
    if _stat_signature(opened) != _stat_signature(before):
        stream.close()
        raise ProtocolError(f"File changed while opening: {path}")
    return stream, before


def _finish_stable_file(
    path: Path, stream: Any, before: os.stat_result
) -> os.stat_result:
    after = os.fstat(stream.fileno())
    current = path.stat()
    if (
        _stat_signature(before) != _stat_signature(after)
        or _stat_signature(after) != _stat_signature(current)
    ):
        raise ProtocolError(f"File changed while hashing: {path}")
    return after


def _hash_stable_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    stream, before = _open_stable_file(path)
    with stream:
        while True:
            chunk = stream.read(HASH_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
        after = _finish_stable_file(path, stream, before)
    return digest.hexdigest(), after.st_size


def _read_stable_file(path: Path) -> tuple[bytes, str, int]:
    digest = hashlib.sha256()
    chunks: list[bytes] = []
    stream, before = _open_stable_file(path)
    with stream:
        while True:
            chunk = stream.read(HASH_CHUNK_BYTES)
            if not chunk:
                break
            chunks.append(chunk)
            digest.update(chunk)
        after = _finish_stable_file(path, stream, before)
    return b"".join(chunks), digest.hexdigest(), after.st_size


def _file_record(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    digest, size = _hash_stable_file(resolved)
    return {
        "path": str(resolved),
        "size_bytes": size,
        "sha256": digest,
    }


def _directory_record(path: Path) -> dict[str, Any]:
    root = path.expanduser().resolve()
    if not root.is_dir():
        raise ProtocolError(f"Required directory does not exist: {root}")

    files: list[dict[str, Any]] = []
    for directory, directory_names, file_names in os.walk(
        root, topdown=True, followlinks=False
    ):
        directory_path = Path(directory)
        directory_names.sort()
        file_names.sort()
        for name in directory_names:
            candidate = directory_path / name
            if candidate.is_symlink():
                raise ProtocolError(
                    f"Directory fingerprint forbids symlinked directories: {candidate}"
                )
        for name in file_names:
            candidate = directory_path / name
            if candidate.is_symlink():
                raise ProtocolError(
                    f"Directory fingerprint forbids symlinked files: {candidate}"
                )
            record = _file_record(candidate)
            record["relative_path"] = candidate.relative_to(root).as_posix()
            record.pop("path")
            files.append(record)

    if not files:
        raise ProtocolError(f"Required directory contains no files: {root}")
    files.sort(key=lambda item: item["relative_path"])
    canonical = json.dumps(
        files, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return {
        "path": str(root),
        "file_count": len(files),
        "total_size_bytes": sum(int(item["size_bytes"]) for item in files),
        "sha256": hashlib.sha256(canonical).hexdigest(),
        "files": files,
    }


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except BaseException:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass
        raise


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    ).encode("utf-8")
    _atomic_write_bytes(path, encoded)


def _parse_gpu_ids(value: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise ProtocolError(f"Invalid --gpus value: {value!r}") from exc
    if parsed != GPU_IDS:
        raise ProtocolError(
            "Formal offline rollout requires the exact GPU list 0,1,2,3"
        )
    return parsed


def _require_directory(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_dir():
        raise ProtocolError(f"{label} directory does not exist: {resolved}")
    return resolved


def _require_executable(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    _hash_stable_file(resolved)
    if not os.access(resolved, os.X_OK):
        raise ProtocolError(f"{label} is not executable: {resolved}")
    return resolved


def _require_checkpoint_under_run_dir(
    checkpoint: Path, run_dir: Path, checkpoint_step: int
) -> Path:
    if checkpoint_step < 0:
        raise ProtocolError("--checkpoint-step must be non-negative")
    resolved = checkpoint.expanduser().resolve()
    expected_name = f"step_{checkpoint_step:06d}.pt"
    if resolved.name != expected_name:
        raise ProtocolError(
            f"Checkpoint filename {resolved.name!r} does not match "
            f"--checkpoint-step {checkpoint_step}; expected {expected_name!r}"
        )
    try:
        resolved.relative_to(run_dir)
    except ValueError as exc:
        raise ProtocolError(
            f"Checkpoint must be located under --run-dir: {resolved}"
        ) from exc
    _hash_stable_file(resolved)
    return resolved


def _build_shards(base_seed: int) -> list[dict[str, Any]]:
    counts = (13, 13, 12, 12)
    shards: list[dict[str, Any]] = []
    cursor = 0
    for shard_id, (gpu, count) in enumerate(zip(GPU_IDS, counts)):
        start = cursor
        stop = start + count
        shards.append(
            {
                "shard_id": shard_id,
                "gpu": gpu,
                "global_episode_start": start,
                "episodes": count,
                "base_seed": base_seed + start,
                "seed_stop_exclusive": base_seed + stop,
            }
        )
        cursor = stop
    if cursor != EPISODES:
        raise AssertionError("Internal shard assignment does not cover 50 episodes")
    return shards


def _code_records() -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for label, path in CODE_FILE_PATHS.items():
        candidate = path if path.is_absolute() else PROJECT_ROOT / path
        records[label] = _file_record(candidate)
    return records


def _normalization_record(args: argparse.Namespace) -> dict[str, Any]:
    if args.norm_stats_meta_dir is not None:
        return {
            "kind": "meta_dir",
            "artifact": _directory_record(args.norm_stats_meta_dir),
        }
    if args.dataset_stats_path is not None:
        return {
            "kind": "dataset_stats",
            "artifact": _file_record(args.dataset_stats_path),
        }
    return {
        "kind": "run_config",
        "artifact": None,
    }


def _normalization_argv(
    normalization: dict[str, Any],
) -> list[str]:
    artifact = normalization["artifact"]
    if normalization["kind"] == "meta_dir":
        return ["--norm-stats-meta-dir", artifact["path"]]
    if normalization["kind"] == "dataset_stats":
        return ["--dataset-stats-path", artifact["path"]]
    return []


def _assert_frozen(path: Path, expected_sha256: str) -> None:
    current_sha256, _ = _hash_stable_file(path)
    if current_sha256 != expected_sha256:
        raise ProtocolError(f"Frozen file changed during evaluation: {path}")


def _create_output_root(path: Path) -> Path:
    output_root = path.expanduser().resolve()
    if os.path.lexists(output_root):
        raise FileExistsError(
            f"Refusing to use an existing output root: {output_root}"
        )
    output_root.parent.mkdir(parents=True, exist_ok=True)
    try:
        output_root.mkdir()
    except FileExistsError as exc:
        raise FileExistsError(
            f"Refusing to use an output root created concurrently: {output_root}"
        ) from exc
    return output_root


def _build_commands(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    checkpoint: Path,
    env_prefix: Path,
    env_python: Path,
    dexjoco_root: Path,
    normalization: dict[str, Any],
    output_root: Path,
    inference_seed: int,
) -> tuple[list[str], list[str]]:
    evaluator = PROJECT_ROOT / CODE_FILE_PATHS["multi_gpu_evaluator"]
    validator = PROJECT_ROOT / CODE_FILE_PATHS["validator"]
    eval_dir = output_root / "eval"
    evaluator_command = [
        str(env_python),
        str(evaluator.resolve()),
        "--gpus",
        "0,1,2,3",
        "--episodes",
        str(EPISODES),
        "--seed",
        str(args.base_seed),
        "--inference-seed",
        str(inference_seed),
        "--server-conda-env",
        str(env_prefix),
        "--client-conda-env",
        str(env_prefix),
        "--run-dir",
        str(run_dir),
        "--checkpoint",
        str(checkpoint),
        *_normalization_argv(normalization),
    ]
    if args.text_cache_dir is not None:
        evaluator_command.extend(
            [
                "--text-embedding-cache-dir",
                str(args.text_cache_dir.expanduser().resolve()),
            ]
        )
    evaluator_command.extend(
        [
            "--no-load-text-encoder",
            "--task-config-dir",
            str((dexjoco_root / "configs" / "rand_obj").resolve()),
            "--tasks",
            TASK,
            "--dexjoco-py-root",
            str((dexjoco_root / "dexjoco").resolve()),
            "--replan-steps",
            str(REPLAN_STEPS),
            "--control-mode",
            "blocking",
            "--async-fallback",
            "wait",
            "--max-env-steps",
            str(MAX_ENV_STEPS),
            "--save-video",
            "--save-actions",
            "--no-randomize",
            "--no-randomize-dynamics",
            "--no-action-clip",
            "--output-dir",
            str(eval_dir),
        ]
    )
    validator_command = [
        str(env_python),
        str(validator.resolve()),
        "--summary",
        str(eval_dir / "summary.json"),
        "--protocol",
        str(output_root / "protocol.json"),
        "--report-json",
        str(output_root / "validated_summary.json"),
        "--report-csv",
        str(output_root / "validated_summary.csv"),
        "--episodes-csv",
        str(output_root / "episodes.csv"),
    ]
    return evaluator_command, validator_command


def _prepare_protocol(
    args: argparse.Namespace,
    *,
    output_root: Path,
    wrapper_argv: list[str],
) -> tuple[dict[str, Any], list[str], list[str]]:
    _parse_gpu_ids(args.gpus)
    run_dir = _require_directory(args.run_dir, "Run")
    config_path = run_dir / "config.yaml"
    config_bytes, config_sha256, config_size = _read_stable_file(config_path)
    checkpoint = _require_checkpoint_under_run_dir(
        args.checkpoint, run_dir, args.checkpoint_step
    )
    checkpoint_record = _file_record(checkpoint)

    env_prefix = _require_directory(args.env_prefix, "Environment prefix")
    env_python = _require_executable(env_prefix / "bin" / "python", "Environment Python")
    dexjoco_root = _require_directory(args.dexjoco_root, "DexJoCo root")
    _require_directory(dexjoco_root / "dexjoco", "DexJoCo Python root")
    task_config_path = (
        dexjoco_root / "configs" / "rand_obj" / f"{TASK}.yaml"
    ).resolve()
    task_config = _file_record(task_config_path)

    normalization = _normalization_record(args)
    text_cache = (
        _directory_record(args.text_cache_dir)
        if args.text_cache_dir is not None
        else None
    )
    code_files = _code_records()
    inference_seed = (
        int(args.base_seed)
        if args.inference_seed is None
        else int(args.inference_seed)
    )

    resolved_config_path = output_root / "resolved_config.yaml"
    _atomic_write_bytes(resolved_config_path, config_bytes)
    resolved_config_record = _file_record(resolved_config_path)
    if resolved_config_record["sha256"] != config_sha256:
        raise ProtocolError("The copied resolved_config.yaml hash does not match config.yaml")

    evaluator_command, validator_command = _build_commands(
        args=args,
        run_dir=run_dir,
        checkpoint=checkpoint,
        env_prefix=env_prefix,
        env_python=env_python,
        dexjoco_root=dexjoco_root,
        normalization=normalization,
        output_root=output_root,
        inference_seed=inference_seed,
    )
    shards = _build_shards(args.base_seed)
    normalization_sha256 = (
        None
        if normalization["artifact"] is None
        else normalization["artifact"]["sha256"]
    )
    text_cache_sha256 = None if text_cache is None else text_cache["sha256"]
    provenance = {
        "variant": args.variant,
        "checkpoint_step": int(args.checkpoint_step),
        "checkpoint_path": checkpoint_record["path"],
        "checkpoint_sha256": checkpoint_record["sha256"],
        "resolved_config_path": str(resolved_config_path),
        "resolved_config_sha256": resolved_config_record["sha256"],
        "task_config_path": task_config["path"],
        "task_config_sha256": task_config["sha256"],
        "normalization_kind": normalization["kind"],
        "normalization_sha256": normalization_sha256,
        "text_cache_sha256": text_cache_sha256,
        "inference_seed": inference_seed,
        "code_files_sha256": {
            label: record["sha256"] for label, record in code_files.items()
        },
    }
    protocol = {
        "schema_version": "fitwam_offline_rollout_eval_v1",
        "status": "frozen",
        "variant": args.variant,
        "task": TASK,
        "evaluation": {
            "episodes": EPISODES,
            "base_seed": int(args.base_seed),
            "inference_seed": inference_seed,
            "expected_seeds": list(
                range(args.base_seed, args.base_seed + EPISODES)
            ),
            "gpus": list(GPU_IDS),
            "shards": shards,
            "replan_steps": REPLAN_STEPS,
            "max_env_steps": MAX_ENV_STEPS,
            "control_mode": "blocking",
            "async_fallback": "wait",
            "save_video": True,
            "save_actions": True,
            "randomize": False,
            "randomize_dynamics": False,
            "action_clip": False,
        },
        "checkpoint": {
            **checkpoint_record,
            "step": int(args.checkpoint_step),
        },
        "config": {
            "path": str(config_path.resolve()),
            "size_bytes": config_size,
            "sha256": config_sha256,
            "copied_path": str(resolved_config_path),
            "copied_size_bytes": resolved_config_record["size_bytes"],
            "copied_sha256": resolved_config_record["sha256"],
        },
        "task_config": task_config,
        "normalization": normalization,
        "text_cache": text_cache,
        "provenance": provenance,
        "environment": {
            "prefix": str(env_prefix),
            "python": _file_record(env_python),
            "dexjoco_root": str(dexjoco_root),
        },
        "code_files": code_files,
        "paths": {
            "output_root": str(output_root),
            "eval_dir": str(output_root / "eval"),
            "validated_summary_json": str(
                output_root / "validated_summary.json"
            ),
            "validated_summary_csv": str(output_root / "validated_summary.csv"),
            "episodes_csv": str(output_root / "episodes.csv"),
        },
        "argv": {
            "wrapper": wrapper_argv,
            "evaluator": evaluator_command,
            "validator": validator_command,
        },
    }
    return protocol, evaluator_command, validator_command


def run(args: argparse.Namespace, *, wrapper_argv: list[str]) -> int:
    try:
        output_root = _create_output_root(args.output_root)
    except FileExistsError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    stage = "preflight"
    active_command: list[str] | None = None
    try:
        protocol, evaluator_command, validator_command = _prepare_protocol(
            args, output_root=output_root, wrapper_argv=wrapper_argv
        )
        protocol_path = output_root / "protocol.json"
        _atomic_write_json(protocol_path, protocol)
        protocol_sha256, _ = _hash_stable_file(protocol_path)
        resolved_config_path = output_root / "resolved_config.yaml"
        resolved_config_sha256 = protocol["config"]["copied_sha256"]

        stage = "evaluator"
        active_command = evaluator_command
        subprocess.run(evaluator_command, cwd=PROJECT_ROOT, check=True)
        eval_dir = output_root / "eval"
        if not eval_dir.is_dir():
            raise ProtocolError(f"Evaluator did not create its output directory: {eval_dir}")
        _assert_frozen(protocol_path, protocol_sha256)
        _assert_frozen(resolved_config_path, resolved_config_sha256)

        stage = "validator"
        active_command = validator_command
        subprocess.run(validator_command, cwd=PROJECT_ROOT, check=True)
        _assert_frozen(protocol_path, protocol_sha256)
        _assert_frozen(resolved_config_path, resolved_config_sha256)
        for filename in (
            "validated_summary.json",
            "validated_summary.csv",
            "episodes.csv",
        ):
            _hash_stable_file(output_root / filename)
        return 0
    except BaseException as exc:
        failure: dict[str, Any] = {
            "status": "failed",
            "stage": stage,
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "wrapper_argv": wrapper_argv,
        }
        if active_command is not None:
            failure["command"] = active_command
        if isinstance(exc, subprocess.CalledProcessError):
            failure["returncode"] = exc.returncode
        try:
            _atomic_write_json(output_root / "failure.json", failure)
        except BaseException as write_exc:
            print(
                f"Failed to write {output_root / 'failure.json'}: {write_exc}",
                file=sys.stderr,
            )
        print(f"{stage} failed: {exc}", file=sys.stderr)
        return 1


def main(argv: Sequence[str] | None = None) -> int:
    parsed_argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(parsed_argv)
    wrapper_argv = (
        list(sys.argv)
        if argv is None
        else [str(SCRIPT_PATH), *parsed_argv]
    )
    return run(args, wrapper_argv=wrapper_argv)


if __name__ == "__main__":
    raise SystemExit(main())
