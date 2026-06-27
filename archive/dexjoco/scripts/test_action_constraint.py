#!/usr/bin/env python3
"""Test per-step action amplitude constraints on saved eval trajectories or live replay."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
for path in (PROJECT_ROOT, SCRIPTS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dexjoco_fastwam_adapter import (  # noqa: E402
    ActionConstraintConfig,
    DexJoCoFastWAMEvalEnv,
    DexJoCoTaskConfig,
    _load_action_bounds,
    constrain_rotvec_action,
    state_to_rotvec_reference,
)


def _action_delta_report(
    raw: np.ndarray,
    constrained: np.ndarray,
    state: np.ndarray,
    *,
    dual_arm: bool,
) -> dict[str, Any]:
    ref = state_to_rotvec_reference(state, dual_arm=dual_arm)
    raw_delta = raw - ref
    new_delta = constrained - ref
    return {
        "raw_xyz_norm": float(np.linalg.norm(raw_delta[:3])),
        "new_xyz_norm": float(np.linalg.norm(new_delta[:3])),
        "raw_dz": float(raw_delta[2]),
        "new_dz": float(new_delta[2]),
        "raw_rot_norm": float(np.linalg.norm(raw_delta[3:6])),
        "new_rot_norm": float(np.linalg.norm(new_delta[3:6])),
        "changed_norm": float(np.linalg.norm(constrained - raw)),
    }


def analyze_npz(
    npz_path: Path,
    *,
    config: ActionConstraintConfig,
    stats_path: Path,
    dual_arm: bool = True,
    max_steps: int | None = 20,
) -> dict[str, Any]:
    payload = np.load(npz_path)
    actions = payload["executed_actions"]
    is_stay = payload["executed_is_stay"]
    state = np.asarray(payload["initial_state"], dtype=np.float64)

    action_min, action_max = (
        _load_action_bounds(stats_path) if config.clip_to_dataset_bounds else (None, None)
    )
    policy_indices = np.where(~is_stay)[0]
    if max_steps is not None:
        policy_indices = policy_indices[:max_steps]

    per_step: list[dict[str, Any]] = []
    current_state = state.copy()
    num_changed = 0

    for step_idx in policy_indices:
        raw = np.asarray(actions[step_idx], dtype=np.float64)
        constrained = constrain_rotvec_action(
            raw,
            current_state,
            dual_arm=dual_arm,
            config=config,
            action_min=action_min,
            action_max=action_max,
        )
        report = _action_delta_report(raw, constrained, current_state, dual_arm=dual_arm)
        report["step"] = int(step_idx)
        per_step.append(report)
        if report["changed_norm"] > 1e-6:
            num_changed += 1
        # Offline replay: treat constrained target as next reference proxy.
        current_state = np.asarray(
            state_to_rotvec_reference(current_state, dual_arm=dual_arm),
            dtype=np.float64,
        )
        if dual_arm:
            current_state = np.concatenate(
                [
                    constrained[:3],
                    np.zeros(4),
                    constrained[6:22],
                    constrained[22:25],
                    np.zeros(4),
                    constrained[28:44],
                ]
            )
        else:
            current_state = np.concatenate([constrained[:3], np.zeros(4), constrained[6:22]])

    return {
        "npz": str(npz_path),
        "num_policy_steps_analyzed": len(per_step),
        "num_steps_changed": num_changed,
        "config": config.__dict__,
        "per_step": per_step,
    }


def replay_in_env(
    *,
    seed: int,
    raw_actions: np.ndarray,
    config: ActionConstraintConfig,
    stats_path: Path,
    task_yaml: Path,
    dual_arm: bool = True,
    max_steps: int = 1500,
) -> dict[str, Any]:
    import yaml

    with task_yaml.open("r", encoding="utf-8") as f:
        task_cfg = yaml.safe_load(f)
    task = DexJoCoTaskConfig.from_yaml(task_cfg)
    action_min, action_max = (
        _load_action_bounds(stats_path) if config.clip_to_dataset_bounds else (None, None)
    )

    def _run(actions: np.ndarray, *, apply_constraint: bool) -> dict[str, Any]:
        env = DexJoCoFastWAMEvalEnv(task, seed=seed)
        try:
            env.reset()
            xyz_trace: list[float] = []
            for step_idx, action in enumerate(actions):
                if step_idx >= max_steps:
                    break
                current_state = np.asarray(env._latest_obs["state"], dtype=np.float64)
                target = np.asarray(action, dtype=np.float32)
                if apply_constraint:
                    target = constrain_rotvec_action(
                        target,
                        current_state,
                        dual_arm=dual_arm,
                        config=config,
                        action_min=action_min,
                        action_max=action_max,
                    )
                env.step_rotvec(target)
                xyz_trace.append(float(target[2]))
                if env.is_done:
                    break
            return {
                "steps": step_idx + 1,
                "success": env.is_success,
                "first10_r_z": xyz_trace[:10],
                "final_r_z": xyz_trace[-1] if xyz_trace else None,
            }
        finally:
            env.close()

    raw_stats = _run(raw_actions, apply_constraint=False)
    constrained_stats = _run(raw_actions, apply_constraint=True)
    return {
        "seed": seed,
        "raw": raw_stats,
        "constrained": constrained_stats,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test per-step action amplitude constraints.")
    parser.add_argument(
        "--actions-npz",
        type=Path,
        required=True,
        help="Saved eval actions from eval_dexjoco_fastwam.py (*_actions.npz).",
    )
    parser.add_argument(
        "--stats-path",
        type=Path,
        default=PROJECT_ROOT
        / "runs/dexjoco_microwave_cook_uncond_3cam_384_1e-4/2026-06-09_16-54-35/dataset_stats.json",
        help="dataset_stats.json for global action bounds clipping.",
    )
    parser.add_argument(
        "--task-yaml",
        type=Path,
        default=PROJECT_ROOT
        / "third_party/dexjoco/configs/rand_obj/bimanual_microwave_cook.yaml",
    )
    parser.add_argument("--max-analyze-steps", type=int, default=20)
    parser.add_argument("--max-replay-steps", type=int, default=1500)
    parser.add_argument("--max-xyz-step", type=float, default=0.05)
    parser.add_argument(
        "--max-rot-step",
        type=float,
        default=0.0,
        help="Per-step rotvec delta limit (0=disabled; quat/rotvec ambiguity makes this risky).",
    )
    parser.add_argument(
        "--max-hand-step",
        type=float,
        default=0.0,
        help="Per-step hand delta limit (0=disabled).",
    )
    parser.add_argument("--max-dz-down", type=float, default=0.03)
    parser.add_argument(
        "--no-dataset-clip",
        action="store_true",
        help="Also clip to dataset_stats global_min/max (optional; default is state-relative only).",
    )
    parser.add_argument(
        "--replay-env",
        action="store_true",
        help="Replay raw vs constrained actions in DexJoCo (requires dexjoco env).",
    )
    parser.add_argument("--seed", type=int, default=0, help="Env seed for --replay-env.")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to save analysis JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ActionConstraintConfig(
        max_xyz_step=args.max_xyz_step,
        max_rot_step=args.max_rot_step,
        max_hand_step=args.max_hand_step,
        max_dz_down=args.max_dz_down,
        clip_to_dataset_bounds=not args.no_dataset_clip,
    )
    if config.clip_to_dataset_bounds:
        print(f"[test_action_constraint] also clipping to stats: {args.stats_path}")

    print("[test_action_constraint] config:", config)
    analysis = analyze_npz(
        args.actions_npz.expanduser().resolve(),
        config=config,
        stats_path=args.stats_path.expanduser().resolve(),
        max_steps=args.max_analyze_steps,
    )
    print(f"[analyze] file={analysis['npz']}")
    print(
        f"[analyze] changed {analysis['num_steps_changed']}/{analysis['num_policy_steps_analyzed']} steps"
    )
    for row in analysis["per_step"][:10]:
        print(
            f"  step={row['step']:3d}  "
            f"|xyz| {row['raw_xyz_norm']:.3f}->{row['new_xyz_norm']:.3f}  "
            f"dz {row['raw_dz']:+.3f}->{row['new_dz']:+.3f}  "
            f"changed={row['changed_norm']:.4f}"
        )

    result: dict[str, Any] = {"analysis": analysis}
    if args.replay_env:
        payload = np.load(args.actions_npz)
        raw_actions = payload["executed_actions"]
        print(f"[replay] seed={args.seed}")
        replay = replay_in_env(
            seed=args.seed,
            raw_actions=raw_actions,
            config=config,
            stats_path=args.stats_path.expanduser().resolve(),
            task_yaml=args.task_yaml.expanduser().resolve(),
            max_steps=args.max_replay_steps,
        )
        result["replay"] = replay
        print("[replay] raw:", replay["raw"])
        print("[replay] constrained:", replay["constrained"])

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"[test_action_constraint] saved {args.output_json}")


if __name__ == "__main__":
    main()
