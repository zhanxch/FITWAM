#!/usr/bin/env python3
"""v9 base CFG eval on mixed-S0 collect recoverability events (oracle-once protocol).

Official **v9 base CFG eval** for mechanism validation:
  1. Replay factual GT prefix from mixed-S0 collect up to t* (last recoverable frame).
  2. ``v9_base``: closed-loop continuation with text_cfg_scale=1 (本体).
  3. ``v9_oracle_once``: **one forced CFG replan at t*** (scale=ORACLE_CFG_SCALE, full
     Successful/Failed mix, no value gate), then 本体 for the rest of the episode.

Compare Pass@M (default M=4) per v9 pair against the scan's S0 Pass@M at the same t*.
Optional ``v9_cfg`` adds deploy-style value_growth gating (secondary, not base eval).

Task prompts / camera mapping come from ``--task`` (tasks.py) or ``--task-yaml``.
Paths for collect / pair_index resolve via ``scripts/dewo_v2/tasks.py dump-collect-replay-paths``.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("DIFFSYNTH_SKIP_DOWNLOAD", "true")
os.environ.setdefault("DIFFSYNTH_MODEL_BASE_PATH", str(ROOT / "checkpoints"))

from policy_io import KEY_ACTION  # noqa: E402
from run_fastwam_server import _build_policy_from_run  # noqa: E402
from scripts.dexjoco_async.dexjoco_fastwam_adapter import (  # noqa: E402
    DexJoCoFastWAMAdapter,
    load_dexjoco_eval_settings,
)
from scripts.fold_glasses.run_seedpair_block_interventions import (  # noqa: E402
    restore_integration_state,
)
from scripts.fold_glasses.scan_failure_recoverability_frontier import (  # noqa: E402
    continuation_noise_seed,
    prepare_factual_snapshots,
)
from scripts.fold_glasses.validate_factual_replay import (  # noqa: E402
    attempt_for_episode,
    create_environment,
    load_episode,
    read_jsonl,
    render_current_observation,
    setup_paths,
)


@dataclass(frozen=True)
class Condition:
    name: str
    text_cfg_scale: float
    cfg_gate_mode: str | None = None
    cfg_growth_tau: float | None = None
    cfg_growth_start_replan: int | None = None
    cfg_growth_stop_replan: int | None = None
    cfg_growth_once: bool = False
    cfg_low_value_threshold: float | None = None
    cfg_growth_delta: float | None = None
    oracle_once: bool = False


CONDITIONS: dict[str, Condition] = {
    "v9_base": Condition("v9_base", text_cfg_scale=1.0),
    "v9_cfg": Condition(
        "v9_cfg",
        text_cfg_scale=1.1,
        cfg_gate_mode="value_growth",
        cfg_growth_tau=0.05,
        cfg_growth_start_replan=8,
        cfg_growth_stop_replan=10,
        cfg_growth_once=True,
    ),
    "v9_oracle_once": Condition(
        "v9_oracle_once",
        text_cfg_scale=1.1,
        oracle_once=True,
    ),
    "v9_low_value_growth": Condition(
        "v9_low_value_growth", text_cfg_scale=1.1,
        cfg_gate_mode="low_value_growth", cfg_growth_once=True,
        cfg_low_value_threshold=0.10, cfg_growth_delta=0.01,
    ),
}

DEFAULT_CONDITIONS = (CONDITIONS["v9_base"], CONDITIONS["v9_oracle_once"])


def load_task_cfg(*, task: str, task_yaml: Path | None) -> dict[str, Any]:
    import yaml

    if task_yaml is not None and task_yaml.is_file():
        return yaml.safe_load(task_yaml.read_text(encoding="utf-8"))
    if not task:
        raise FileNotFoundError("Provide --task or an existing --task-yaml")
    tmp = ROOT / "data" / f".eval_task_cfg_{task}_replay.yaml"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/dewo_v2/tasks.py"),
            "write-eval-yaml",
            "--task",
            task,
            "--output",
            str(tmp),
        ],
        check=True,
    )
    return yaml.safe_load(tmp.read_text(encoding="utf-8"))


def load_pairs(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(payload["pairs"])


def load_s0_prefix_hits(path: Path) -> dict[tuple[int, int], dict[str, Any]]:
    out: dict[tuple[int, int], dict[str, Any]] = {}
    if not path.exists():
        return out
    for row in read_jsonl(path):
        ep = int(row["source_failure_episode_index"])
        frame = int(row["prefix_frame"])
        out[(ep, frame)] = row
    return out


def env_obs_from_render(rendered: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "front": np.asarray(rendered["front"], dtype=np.uint8),
        "wrist": np.asarray(rendered["wrist"], dtype=np.uint8),
        "state": np.asarray(rendered["state"], dtype=np.float32),
    }


def run_v9_continuation(
    env: Any,
    policy: Any,
    adapter: DexJoCoFastWAMAdapter,
    *,
    snapshot: tuple[np.ndarray, Mapping[str, Any]],
    episode_index: int,
    prefix_frame: int,
    replicate_index: int,
    condition: Condition,
    base_noise_seed: int,
    max_steps: int,
    replan_steps: int,
    task_prompt: str,
    cfg_base_prompt: str,
    cfg_failure_prompt: str,
    camera_key: str,
    camera_mapping: dict[str, str],
) -> dict[str, Any]:
    from fastwam_dexjoco.policy import fastwam_action_to_dexjoco

    restore_integration_state(env, snapshot[0], snapshot[1])
    env.unwrapped.image_obs = False
    pending: deque[np.ndarray] = deque()
    replan_index = 0
    value_prev: float | None = None
    value_fired = False
    cfg_fired_steps: list[int] = []
    cfg_values: list[float] = []
    cfg_gate_g: list[float] = []
    terminated = False
    truncated = False
    succeeded = False
    final_info: Mapping[str, Any] = {"succeed": False}
    prefix_replan_base = int(prefix_frame) // int(replan_steps)

    for frame in range(int(prefix_frame), int(max_steps)):
        rendered = None
        if not pending:
            rendered = render_current_observation(env)
            env_obs = env_obs_from_render(rendered)
            policy_obs = adapter.env_obs_to_policy_obs(
                env_obs,
                camera_key=camera_key,
                camera_mapping=camera_mapping,
                task_prompt=task_prompt,
                cfg_base_prompt=cfg_base_prompt,
                cfg_failure_prompt=cfg_failure_prompt,
            )
            noise_seed = continuation_noise_seed(
                base_noise_seed,
                int(episode_index),
                int(prefix_frame),
                replicate_index,
                replan_index,
            )
            options: dict[str, Any] = {"seed": noise_seed}
            if condition.oracle_once:
                step_scale = float(condition.text_cfg_scale) if replan_index == 0 else 1.0
            else:
                step_scale = float(condition.text_cfg_scale)
                if condition.cfg_gate_mode:
                    options["cfg_gate_mode"] = condition.cfg_gate_mode
                    options["cfg_replan_index"] = prefix_replan_base + replan_index
                    if condition.cfg_growth_tau is not None:
                        options["cfg_growth_tau"] = float(condition.cfg_growth_tau)
                    if condition.cfg_growth_start_replan is not None:
                        options["cfg_growth_start_replan"] = int(condition.cfg_growth_start_replan)
                    if condition.cfg_growth_stop_replan is not None:
                        options["cfg_growth_stop_replan"] = int(condition.cfg_growth_stop_replan)
                    if condition.cfg_low_value_threshold is not None:
                        options["cfg_low_value_threshold"] = float(condition.cfg_low_value_threshold)
                    if condition.cfg_growth_delta is not None:
                        options["cfg_growth_delta"] = float(condition.cfg_growth_delta)
                    if condition.cfg_growth_once:
                        options["cfg_gate_fired"] = bool(value_fired)
                    if value_prev is not None:
                        options["cfg_value_prev"] = float(value_prev)

            saved_scale = float(policy.text_cfg_scale)
            policy.text_cfg_scale = step_scale
            try:
                action_payload = policy.get_action(policy_obs, options=options)
            finally:
                policy.text_cfg_scale = saved_scale

            chunk = np.asarray(action_payload[KEY_ACTION], dtype=np.float32).reshape(
                policy.action_horizon, -1
            )[:, :22]
            pending.extend(chunk[:replan_steps])
            replan_index += 1
            if condition.oracle_once and replan_index == 1:
                cfg_fired_steps.append(int(frame))

            cfg_v = action_payload.get("cfg_value")
            if cfg_v is not None and np.asarray(cfg_v).reshape(-1)[0] == np.asarray(cfg_v).reshape(-1)[0]:
                value_prev = float(np.asarray(cfg_v).reshape(-1)[0])
                cfg_values.append(value_prev)
            g = action_payload.get("cfg_gate_g")
            if g is not None:
                g_val = float(np.asarray(g).reshape(-1)[0])
                if g_val == g_val:
                    cfg_gate_g.append(g_val)
                    if g_val >= 1.0:
                        value_fired = True
                        cfg_fired_steps.append(int(frame))

        action = np.asarray(pending.popleft(), dtype=np.float32)
        _, _, terminated, truncated, final_info = env.step(fastwam_action_to_dexjoco(action))
        succeeded = succeeded or bool(final_info.get("succeed", False))
        if terminated or truncated:
            break

    return {
        "success": bool(succeeded),
        "steps_executed": int(frame + 1 - int(prefix_frame)) if "frame" in locals() else 0,
        "cfg_fired_steps": cfg_fired_steps,
        "cfg_value_last": cfg_values[-1] if cfg_values else None,
        "cfg_gate_g_max": max(cfg_gate_g) if cfg_gate_g else None,
        "replan_count": replan_index,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--collect-dataset",
        type=Path,
        default=ROOT
        / "data/fold_glasses_mixed_s0_collect_4x50_20260823_142736/rollout_raw_200",
    )
    parser.add_argument(
        "--pair-index",
        type=Path,
        default=ROOT / "data/fold_glasses_dewo_v9_pair_full_lerobot/pair_index.json",
    )
    parser.add_argument(
        "--prefix-results",
        type=Path,
        default=ROOT
        / "data/fold_glasses_mixed_s0_collect_4x50_20260823_142736/recoverability_pairs_v2/prefix_results.jsonl",
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=str, default="step_005000.pt")
    parser.add_argument("--backbone-checkpoint", type=str, default="")
    parser.add_argument("--uncond-adapter", type=str, default="")
    parser.add_argument("--dataset-stats", type=Path, default="")
    parser.add_argument("--text-cache", type=Path, default="")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--replan-steps", type=int, default=24)
    parser.add_argument("--action-horizon", type=int, default=32)
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--pass-m", type=int, default=4)
    parser.add_argument("--base-noise-seed", type=int, default=20260813)
    parser.add_argument("--state-atol", type=float, default=2e-4)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument(
        "--task",
        type=str,
        default=os.environ.get("TASK", ""),
        help="DexJoCo task; loads eval yaml from scripts/dewo_v2/tasks.py when --task-yaml unset",
    )
    parser.add_argument("--task-yaml", type=Path, default="")
    parser.add_argument(
        "--oracle-cfg-scale",
        type=float,
        default=float(os.environ.get("ORACLE_CFG_SCALE", "1.1")),
        help="text_cfg_scale for the single forced CFG replan in v9_oracle_once",
    )
    parser.add_argument(
        "--conditions",
        type=str,
        default="v9_base,v9_oracle_once",
        help="Comma-separated: v9_base,v9_oracle_once[,v9_cfg]",
    )
    args = parser.parse_args()

    selected = []
    for name in args.conditions.split(","):
        name = name.strip()
        if not name:
            continue
        if name not in CONDITIONS:
            raise SystemExit(f"Unknown condition {name!r}; choose from {sorted(CONDITIONS)}")
        cond = CONDITIONS[name]
        if name == "v9_oracle_once":
            cond = dataclasses.replace(cond, text_cfg_scale=float(args.oracle_cfg_scale))
        selected.append(cond)
    if not selected:
        raise SystemExit("No conditions selected")

    setup_paths()
    pairs = load_pairs(args.pair_index.expanduser().resolve())
    shard_pairs = [
        p for i, p in enumerate(pairs) if i % int(args.num_shards) == int(args.shard_index)
    ]
    if not shard_pairs:
        print(f"[shard {args.shard_index}] no pairs", flush=True)
        return 0

    s0_hits = load_s0_prefix_hits(args.prefix_results.expanduser().resolve())
    run_dir = args.run_dir.expanduser().resolve()
    eval_settings = load_dexjoco_eval_settings(
        run_dir,
        action_horizon_override=int(args.action_horizon),
        text_embedding_cache_dir_override=(
            str(args.text_cache.expanduser().resolve()) if str(args.text_cache) else None
        ),
    )
    adapter = DexJoCoFastWAMAdapter(eval_settings)

    task_yaml_path = (
        args.task_yaml.expanduser().resolve() if str(args.task_yaml) else None
    )
    task_cfg = load_task_cfg(task=str(args.task).strip(), task_yaml=task_yaml_path)
    task_prompt = str(task_cfg["prompt"])
    cfg_base_prompt = str(task_cfg.get("cfg_base_prompt", task_prompt))
    cfg_failure_prompt = str(task_cfg.get("cfg_failure_prompt") or f"{cfg_base_prompt} Failed execution.")
    camera_key = str(task_cfg.get("camera_mapping", {}).get("base", "front"))
    camera_mapping = dict(task_cfg.get("camera_mapping", {"base": "front", "wrist": "wrist"}))

    stats = str(args.dataset_stats) if str(args.dataset_stats) else None
    if stats is None:
        stats = str(ROOT / "artifacts/mixed_5task/dataset_stats.json")

    print(f"[shard {args.shard_index}] loading v9 policy on {args.device}", flush=True)
    policy = _build_policy_from_run(
        run_dir,
        str(args.checkpoint),
        dataset_stats_path=stats,
        norm_stats_meta_dir=None,
        device=str(args.device),
        action_horizon=int(args.action_horizon),
        num_inference_steps=int(args.num_inference_steps),
        load_text_encoder=False,
        text_cfg_scale=1.1,
        negative_prompt=cfg_base_prompt,
        failure_prompt=cfg_failure_prompt,
        backbone_checkpoint=str(args.backbone_checkpoint) or None,
        uncond_adapter=str(args.uncond_adapter) or None,
    )

    out_dir = args.output.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / f"shard_{args.shard_index}" / "results.jsonl"
    results_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = args.collect_dataset.expanduser().resolve()
    scan_frames_cache: dict[int, list[int]] = {}
    rows: list[dict[str, Any]] = []

    for pair in shard_pairs:
        ep = int(pair["source_failure_episode_index"])
        t_star = int(pair["t_star_last_recoverable"])
        attempt = attempt_for_episode(dataset, ep)
        actions, recorded_states = load_episode(dataset, ep)
        horizon = min(len(actions), int(args.max_steps))
        if t_star >= horizon:
            print(f"[pair {pair['pair_id']}] skip t*={t_star} >= horizon={horizon}", flush=True)
            continue

        if ep not in scan_frames_cache:
            scan_frames_cache[ep] = sorted(
                {int(t_star)}
                | {int(r["prefix_frame"]) for (e, _), r in s0_hits.items() if e == ep}
            )
        scan_frames = sorted({t_star, *[f for f in scan_frames_cache[ep] if f <= horizon]})

        _, env = create_environment(int(attempt["seed"]))
        try:
            snapshots, _gate = prepare_factual_snapshots(
                env,
                actions=actions,
                recorded_states=recorded_states,
                attempt=attempt,
                scan_frames=[t_star],
                max_steps=int(args.max_steps),
                state_atol=float(args.state_atol),
            )
            snapshot = snapshots[t_star]
            s0_row = s0_hits.get((ep, t_star), {})

            for condition in selected:
                repl_results: list[dict[str, Any]] = []
                for replicate in range(int(args.pass_m)):
                    print(
                        f"[{pair['pair_id']}] t*={t_star} {condition.name} rep={replicate}",
                        flush=True,
                    )
                    t0 = time.perf_counter()
                    result = run_v9_continuation(
                        env,
                        policy,
                        adapter,
                        snapshot=snapshot,
                        episode_index=ep,
                        prefix_frame=t_star,
                        replicate_index=replicate,
                        condition=condition,
                        base_noise_seed=int(args.base_noise_seed),
                        max_steps=int(args.max_steps),
                        replan_steps=int(args.replan_steps),
                        task_prompt=task_prompt,
                        cfg_base_prompt=cfg_base_prompt,
                        cfg_failure_prompt=cfg_failure_prompt,
                        camera_key=camera_key,
                        camera_mapping=camera_mapping,
                    )
                    result["replicate_index"] = replicate
                    result["elapsed_s"] = float(time.perf_counter() - t0)
                    repl_results.append(result)

                succ = sum(bool(r["success"]) for r in repl_results)
                row = {
                    "pair_id": pair["pair_id"],
                    "seed": pair.get("seed"),
                    "source_failure_episode_index": ep,
                    "t_star": t_star,
                    "m_first_zero": int(pair["M_first_zero"]),
                    "condition": condition.name,
                    "text_cfg_scale": condition.text_cfg_scale,
                    "pass_m": int(args.pass_m),
                    "success_count": succ,
                    "pass_at_m_hit": bool(succ > 0),
                    "replicates": repl_results,
                    "s0_pass_at_m_hit": s0_row.get("pass_at_m_hit"),
                    "s0_success_count": s0_row.get("success_count"),
                    "checkpoint": str(args.checkpoint),
                    "device": str(args.device),
                }
                rows.append(row)
                with results_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
        finally:
            env.close()

    summary = {
        "shard_index": int(args.shard_index),
        "num_pairs": len(shard_pairs),
        "num_rows": len(rows),
        "results": str(results_path),
    }
    (out_dir / f"shard_{args.shard_index}" / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
