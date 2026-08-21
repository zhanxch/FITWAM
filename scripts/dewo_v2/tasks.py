#!/usr/bin/env python3
"""DexJoCo DEWO v2 task registry and CFG knobs.

Task identity (prompt, ckpt, expert path) lives here. CFG mixing is a separate
recipe with hammer_nail defaults, overridable via env for debugging:

  TASK=water_plant
  CFG_PRIMARY=0.5,0.0,0.5          # outcome,fast,base
  CFG_AUX_SUCCESS=0.4,0.2,0.4
  CFG_AUX_FAIL=0.0,0.2,0.4
  CFG_SUCCESS_SUFFIX=' Successful execution.'
  CFG_FAILURE_SUFFIX=null
  CFG_DROPOUT=0.0
  CFG_FAST_FAIL_CLOSED=1

Usage:
  python scripts/dewo_v2/tasks.py export-env --task water_plant
  python scripts/dewo_v2/tasks.py write-eval-yaml --task water_plant --output ...
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OPEN_REPO = Path(
    os.environ.get("OPEN_REPO", str(ROOT.parent / "FastWAM-infer-in-DexJoco"))
)

# Hammer-nail DEWO v2 CFG recipe (formal default for new tasks).
HAMMER_CFG_SUCCESS_SUFFIX = " Successful execution."
HAMMER_CFG_FAILURE_SUFFIX: Optional[str] = None
HAMMER_CFG_DROPOUT = 0.0
HAMMER_CFG_PRIMARY = (0.5, 0.0, 0.5)  # outcome, fast, base
HAMMER_CFG_AUX_SUCCESS = (0.4, 0.2, 0.4)
HAMMER_CFG_AUX_FAIL = (0.0, 0.2, 0.4)
HAMMER_CFG_FAST_MODEL = "physical-intelligence/fast"
HAMMER_CFG_FAST_MAX_TOKENS = 32
HAMMER_CFG_FAST_FAIL_CLOSED = True

DEFAULT_SEED_START = 10086
DEFAULT_SEED_END = 10135
DEFAULT_REPEATS = 4
DEFAULT_MAX_STEPS = 1000
DEFAULT_ACTION_HORIZON = 32
DEFAULT_REPLAN_STEPS = 24
DEFAULT_NFE = 10
DEFAULT_HYDRA_TASK = "dexjoco/dexjoco_dewo_v2_offline_b1_jump_fast_lora_3e-5"


@dataclass(frozen=True)
class TaskSpec:
    name: str
    success_prompt: str
    ckpt_rel: str
    open_ckpt_rel: str
    expert_rel: str
    hydra_task: str = DEFAULT_HYDRA_TASK
    max_steps: int = DEFAULT_MAX_STEPS
    seed_start: int = DEFAULT_SEED_START
    seed_end: int = DEFAULT_SEED_END
    repeats: int = DEFAULT_REPEATS
    action_horizon: int = DEFAULT_ACTION_HORIZON
    replan_steps: int = DEFAULT_REPLAN_STEPS
    nfe: int = DEFAULT_NFE
    action_dim: int = 22
    state_dim: int = 23
    image_raw: tuple[int, int] = (640, 640)
    cameras: tuple[str, ...] = ("front", "wrist")


@dataclass(frozen=True)
class CfgRecipe:
    success_suffix: str = HAMMER_CFG_SUCCESS_SUFFIX
    failure_suffix: Optional[str] = HAMMER_CFG_FAILURE_SUFFIX
    dropout: float = HAMMER_CFG_DROPOUT
    primary: tuple[float, float, float] = HAMMER_CFG_PRIMARY
    aux_success: tuple[float, float, float] = HAMMER_CFG_AUX_SUCCESS
    aux_fail: tuple[float, float, float] = HAMMER_CFG_AUX_FAIL
    fast_model_id: str = HAMMER_CFG_FAST_MODEL
    fast_max_tokens: int = HAMMER_CFG_FAST_MAX_TOKENS
    fast_fail_closed: bool = HAMMER_CFG_FAST_FAIL_CLOSED
    recipe_name: str = "hammer_nail"

    def as_json(self) -> dict[str, Any]:
        return {
            "recipe_name": self.recipe_name,
            "success_suffix": self.success_suffix,
            "failure_suffix": self.failure_suffix,
            "dropout": self.dropout,
            "primary": {"outcome": self.primary[0], "fast": self.primary[1], "base": self.primary[2]},
            "aux_success": {
                "outcome": self.aux_success[0],
                "fast": self.aux_success[1],
                "base": self.aux_success[2],
            },
            "aux_fail": {
                "outcome": self.aux_fail[0],
                "fast": self.aux_fail[1],
                "base": self.aux_fail[2],
            },
            "fast_tokenizer_model_id": self.fast_model_id,
            "fast_max_tokens": self.fast_max_tokens,
            "fast_fail_closed": self.fast_fail_closed,
        }


TASKS: dict[str, TaskSpec] = {
    "water_plant": TaskSpec(
        name="water_plant",
        success_prompt="Grasp the watering can and apply water to the plant.",
        ckpt_rel="checkpoints/dexjoco/water_plant_fastwam/weights/step_012500.pt",
        open_ckpt_rel="checkpoints/water_plant/step_012500.pt",
        expert_rel="data/dexjoco/dexjoco_lerobot_datasets/water_plant",
        hydra_task="dexjoco/dexjoco_water_plant_offline_b1_jump_fast_lora_3e-5",
    ),
    "fold_glasses": TaskSpec(
        name="fold_glasses",
        success_prompt="Fold the glasses and place them into the case.",
        ckpt_rel="checkpoints/dexjoco/fold_glasses_fastwam/weights/step_010000.pt",
        open_ckpt_rel="checkpoints/fold_glasses/step_010000.pt",
        expert_rel="data/dexjoco/dexjoco_lerobot_datasets/fold_glasses",
        hydra_task="dexjoco/dexjoco_fold_glasses_offline_b1_jump_fast_lora_3e-5",
    ),
    "hammer_nail": TaskSpec(
        name="hammer_nail",
        success_prompt="Use the hammer to drive the nail into the wooden board.",
        ckpt_rel="checkpoints/dexjoco/hammer_nail_fastwam/weights/step_002500.pt",
        open_ckpt_rel="checkpoints/hammer_nail/step_002500.pt",
        expert_rel="data/dexjoco/dexjoco_lerobot_datasets/hammer_nail",
        hydra_task="dexjoco/dexjoco_hammer_nail_offline_b1_jump_fast_lora_3e-5",
    ),
    "pick_bucket": TaskSpec(
        name="pick_bucket",
        success_prompt="Place the boxed food into the bucket and then lift the bucket.",
        ckpt_rel="checkpoints/dexjoco/pick_bucket_fastwam/weights/step_010000.pt",
        open_ckpt_rel="checkpoints/pick_bucket/step_010000.pt",
        expert_rel="data/dexjoco/dexjoco_lerobot_datasets/pick_bucket",
        hydra_task=DEFAULT_HYDRA_TASK,
    ),
    "pinch_tongs": TaskSpec(
        name="pinch_tongs",
        success_prompt="Grasp the tongs and perform three consecutive open-close motions.",
        ckpt_rel="checkpoints/dexjoco/pinch_tongs_fastwam/weights/step_010000.pt",
        open_ckpt_rel="checkpoints/pinch_tongs/step_010000.pt",
        expert_rel="data/dexjoco/dexjoco_lerobot_datasets/pinch_tongs",
        hydra_task=DEFAULT_HYDRA_TASK,
    ),
}


def get_task(name: str) -> TaskSpec:
    key = str(name).strip()
    if key not in TASKS:
        known = ", ".join(sorted(TASKS))
        raise KeyError(f"Unknown DEWO v2 task {name!r}. Known: {known}")
    return TASKS[key]


def _parse_bool(raw: str) -> bool:
    text = raw.strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Cannot parse bool: {raw!r}")


def _parse_optional_str(raw: Optional[str], default: Optional[str]) -> Optional[str]:
    if raw is None:
        return default
    text = raw
    if text.strip() == "" or text.strip().lower() in {"null", "none", "~", "nil"}:
        return None
    return text


def _parse_triple(raw: str, field: str) -> tuple[float, float, float]:
    parts = [p.strip() for p in raw.split(",")]
    if len(parts) != 3:
        raise ValueError(f"{field} must be outcome,fast,base (3 floats), got {raw!r}")
    vals = tuple(float(p) for p in parts)
    if any(v < 0.0 for v in vals):
        raise ValueError(f"{field} values must be >= 0, got {vals}")
    if sum(vals) <= 0.0:
        raise ValueError(f"{field} must sum to > 0, got {vals}")
    if vals[1] > 0.0 and field == "CFG_PRIMARY":
        raise ValueError("CFG_PRIMARY.fast must be 0; FAST cannot mix with action-loss samples")
    return vals[0], vals[1], vals[2]


def parse_cfg_recipe(env: Optional[Mapping[str, str]] = None) -> CfgRecipe:
    """Hammer defaults, then env overrides. Compact triples win over per-channel vars."""

    src = dict(os.environ if env is None else env)

    def get(key: str) -> Optional[str]:
        if key not in src:
            return None
        return src[key]

    recipe = CfgRecipe()
    success = _parse_optional_str(get("CFG_SUCCESS_SUFFIX"), recipe.success_suffix)
    if success is None:
        raise ValueError("CFG_SUCCESS_SUFFIX cannot be null; success-vs-base CFG needs it")
    failure = _parse_optional_str(
        get("CFG_FAILURE_SUFFIX") if "CFG_FAILURE_SUFFIX" in src else None,
        recipe.failure_suffix,
    )
    def get_float(key: str, default: float) -> float:
        raw = get(key)
        return default if raw is None or raw == "" else float(raw)

    dropout = get_float("CFG_DROPOUT", recipe.dropout)
    primary = recipe.primary
    aux_s = recipe.aux_success
    aux_f = recipe.aux_fail
    if get("CFG_PRIMARY"):
        primary = _parse_triple(get("CFG_PRIMARY") or "", "CFG_PRIMARY")
    else:
        primary = (
            get_float("CFG_PRIMARY_OUTCOME", primary[0]),
            get_float("CFG_PRIMARY_FAST", primary[1]),
            get_float("CFG_PRIMARY_BASE", primary[2]),
        )
        if primary[1] > 0.0:
            raise ValueError("CFG_PRIMARY_FAST must be 0")
    if get("CFG_AUX_SUCCESS"):
        aux_s = _parse_triple(get("CFG_AUX_SUCCESS") or "", "CFG_AUX_SUCCESS")
    else:
        aux_s = (
            get_float("CFG_AUX_SUCCESS_OUTCOME", aux_s[0]),
            get_float("CFG_AUX_SUCCESS_FAST", aux_s[1]),
            get_float("CFG_AUX_SUCCESS_BASE", aux_s[2]),
        )
    if get("CFG_AUX_FAIL"):
        aux_f = _parse_triple(get("CFG_AUX_FAIL") or "", "CFG_AUX_FAIL")
    else:
        aux_f = (
            get_float("CFG_AUX_FAIL_OUTCOME", aux_f[0]),
            get_float("CFG_AUX_FAIL_FAST", aux_f[1]),
            get_float("CFG_AUX_FAIL_BASE", aux_f[2]),
        )
    fast_closed = recipe.fast_fail_closed
    if get("CFG_FAST_FAIL_CLOSED") is not None:
        fast_closed = _parse_bool(get("CFG_FAST_FAIL_CLOSED") or "")
    fast_tokens = int(get_float("CFG_FAST_MAX_TOKENS", float(recipe.fast_max_tokens)))
    fast_model = get("CFG_FAST_MODEL_ID") or recipe.fast_model_id
    return replace(
        recipe,
        success_suffix=success,
        failure_suffix=failure,
        dropout=dropout,
        primary=primary,
        aux_success=aux_s,
        aux_fail=aux_f,
        fast_fail_closed=fast_closed,
        fast_max_tokens=fast_tokens,
        fast_model_id=fast_model,
    )


def resolve_ckpt(task: TaskSpec, *, root: Path = ROOT, open_repo: Path = DEFAULT_OPEN_REPO) -> Path:
    local = (root / task.ckpt_rel).resolve()
    if local.is_file():
        return local
    opened = (open_repo / task.open_ckpt_rel).resolve()
    if opened.is_file():
        return opened
    raise FileNotFoundError(
        f"Missing checkpoint for {task.name}: tried {local} and {opened}"
    )


def resolve_expert(task: TaskSpec, *, root: Path = ROOT) -> Path:
    path = (root / task.expert_rel).resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Missing expert dataset for {task.name}: {path}")
    return path


_WRAPPED_PROMPT = (
    "A video recorded from a robot's point of view executing the following instruction: {task}"
)


def wrapped_prompt_hash(success_prompt: str) -> str:
    wrapped = _WRAPPED_PROMPT.format(task=success_prompt)
    return hashlib.sha256(wrapped.encode("utf-8")).hexdigest()


def t5_cache_name(success_prompt: str) -> str:
    return f"{wrapped_prompt_hash(success_prompt)}.t5_len128.wan22ti2v5b.pt"


def eval_task_yaml(task: TaskSpec, cfg: CfgRecipe) -> str:
    pos = f"{task.success_prompt}{cfg.success_suffix}"
    return (
        "# Generated by scripts/dewo_v2/tasks.py from the active CFG recipe.\n"
        f"env_name: {task.name}\n"
        "camera_mapping:\n"
        "  base: front\n"
        "  wrist: wrist\n"
        f'prompt: {json.dumps(pos)}\n'
        f'cfg_base_prompt: {json.dumps(task.success_prompt)}\n'
        "robot_type: single_arm\n"
    )


def _shell_export(key: str, value: Any) -> str:
    if value is None:
        return f"export {key}=null"
    if isinstance(value, bool):
        return f"export {key}={'true' if value else 'false'}"
    if isinstance(value, (int, float)):
        return f"export {key}={value}"
    return f"export {key}={shlex.quote(str(value))}"


def build_exports(
    task: TaskSpec,
    cfg: CfgRecipe,
    *,
    root: Path = ROOT,
    open_repo: Path = DEFAULT_OPEN_REPO,
) -> dict[str, Any]:
    ckpt = resolve_ckpt(task, root=root, open_repo=open_repo)
    expert = resolve_expert(task, root=root)
    artifacts = (open_repo / "artifacts" / task.name).resolve()
    stats = artifacts / "dataset_stats.json"
    t5_name = t5_cache_name(task.success_prompt)
    return {
        "DEWO_TASK_NAME": task.name,
        "SUCCESS_PROMPT": task.success_prompt,
        "CKPT": str(ckpt),
        "CKPT_STEP": int(Path(task.ckpt_rel).stem.rsplit("_", 1)[-1]),
        "OPEN_CKPT_DIR": str((open_repo / "checkpoints" / task.name).resolve()),
        "SOURCE_DATASET": str(expert),
        "BASE_DATASET": str(expert),
        "OPEN_REPO": str(open_repo),
        "STATS": str(stats),
        "PRETRAINED_NORM_STATS": str(stats),
        "TEXT_EMB": str(artifacts / t5_name),
        "T5_CACHE_NAME": t5_name,
        "MAX_STEPS": task.max_steps,
        "SEED_START": task.seed_start,
        "SEED_END": task.seed_end,
        "REPEATS": task.repeats,
        "ACTION_HORIZON": task.action_horizon,
        "REPLAN_STEPS": task.replan_steps,
        "NUM_INFERENCE_STEPS": task.nfe,
        "DEWO_TASK": task.hydra_task,
        "DEWO_PROTOCOL": f"{task.name}_dewo_v2_jump_fast_lora_opensource",
        "FITWAM_WANDB_GROUP": f"{task.name}_dewo_v2_opensource",
        "DEWO_OUTPUT_DIR": f"./runs/dexjoco_{task.name}_dewo_v2",
        "CFG_TASK_CONFIG_DIR": str(root / "configs" / "eval" / "dexjoco" / f"{task.name}_dewo_v2_cfg"),
        "CFG_RECIPE_NAME": cfg.recipe_name,
        "CFG_SUCCESS_SUFFIX": cfg.success_suffix,
        "CFG_FAILURE_SUFFIX": "null" if cfg.failure_suffix is None else cfg.failure_suffix,
        "CFG_DROPOUT": cfg.dropout,
        "CFG_PRIMARY": f"{cfg.primary[0]},{cfg.primary[1]},{cfg.primary[2]}",
        "CFG_AUX_SUCCESS": f"{cfg.aux_success[0]},{cfg.aux_success[1]},{cfg.aux_success[2]}",
        "CFG_AUX_FAIL": f"{cfg.aux_fail[0]},{cfg.aux_fail[1]},{cfg.aux_fail[2]}",
        "CFG_PRIMARY_OUTCOME": cfg.primary[0],
        "CFG_PRIMARY_FAST": cfg.primary[1],
        "CFG_PRIMARY_BASE": cfg.primary[2],
        "CFG_AUX_SUCCESS_OUTCOME": cfg.aux_success[0],
        "CFG_AUX_SUCCESS_FAST": cfg.aux_success[1],
        "CFG_AUX_SUCCESS_BASE": cfg.aux_success[2],
        "CFG_AUX_FAIL_OUTCOME": cfg.aux_fail[0],
        "CFG_AUX_FAIL_FAST": cfg.aux_fail[1],
        "CFG_AUX_FAIL_BASE": cfg.aux_fail[2],
        "CFG_FAST_MODEL_ID": cfg.fast_model_id,
        "CFG_FAST_MAX_TOKENS": cfg.fast_max_tokens,
        "CFG_FAST_FAIL_CLOSED": cfg.fast_fail_closed,
    }


def render_shell_exports(values: Mapping[str, Any]) -> str:
    lines = ["# generated by scripts/dewo_v2/tasks.py", "set -a"]
    for key, value in values.items():
        lines.append(_shell_export(key, value))
    lines.append("set +a")
    return "\n".join(lines) + "\n"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    exp = sub.add_parser("export-env", help="Print bash exports for TASK + CFG")
    exp.add_argument("--task", default=os.environ.get("TASK", "water_plant"))
    dump = sub.add_parser("dump-cfg-json", help="Print resolved CFG recipe JSON")
    dump.add_argument("--task", default=os.environ.get("TASK", "water_plant"))
    ev = sub.add_parser("write-eval-yaml", help="Write CFG eval task yaml")
    ev.add_argument("--task", default=os.environ.get("TASK", "water_plant"))
    ev.add_argument("--output", type=Path, required=True)
    ls = sub.add_parser("list", help="List registered tasks")
    _ = ls
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.cmd == "list":
        for name, spec in TASKS.items():
            print(f"{name}\t{spec.success_prompt}")
        return 0
    task = get_task(args.task)
    cfg = parse_cfg_recipe()
    if args.cmd == "export-env":
        print(render_shell_exports(build_exports(task, cfg)), end="")
        return 0
    if args.cmd == "dump-cfg-json":
        payload = {"task": asdict(task), "cfg": cfg.as_json()}
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    if args.cmd == "write-eval-yaml":
        out = args.output.expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(eval_task_yaml(task, cfg), encoding="utf-8")
        print(out)
        return 0
    raise SystemExit(f"unknown cmd {args.cmd}")


if __name__ == "__main__":
    raise SystemExit(main())
