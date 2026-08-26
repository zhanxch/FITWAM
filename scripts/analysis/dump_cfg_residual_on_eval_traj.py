#!/usr/bin/env python3
"""Dump v5 CFG residual energy on saved closed-loop eval trajectories.

This is a diagnostic on the policy's *own* 0–49 inference states, not Pass@20.
Replay executed actions, re-run `infer_action` at each original replan, and
record per-token / per-chunk RMS of ``ε_posi − ε_base``.

CFG terms: ``text_cfg_scale=1`` is the 本体 bypass; mix ``w=1`` is ``ε_posi``,
not 本体. This dump always runs the two-branch mix so ``δ`` is defined.

Phases:
  replay  — DexJoCo env (dexjoco conda); writes obs npz
  dump    — FastWAM infer (fastwam conda, GPU); writes residual npz
  summarize — CPU plots + JSON (no Pass@20 alignment)

Example:
  python scripts/analysis/dump_cfg_residual_on_eval_traj.py --phase replay ...
  python scripts/analysis/dump_cfg_residual_on_eval_traj.py --phase dump --device cuda:0 ...
  python scripts/analysis/dump_cfg_residual_on_eval_traj.py --phase summarize ...
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
SRC = ROOT / "src"
DEXJOCO_ASYNC = SCRIPTS / "dexjoco_async"
for path in (ROOT, SRC, SCRIPTS, DEXJOCO_ASYNC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

DEFAULT_EVAL_RUN = (
    ROOT
    / "evaluate_results"
    / "dexjoco"
    / "water_plant_dewo_v5_mixed_s0_cfg2.0_screen1x50_20260823_154823"
    / "step_001000_cfg2.0_4x50"
    / "run1"
)
DEFAULT_TASK_YAML_DIR = (
    ROOT
    / "data"
    / "water_plant_mixed_s0_dewo_v2_pair_20260820_182236"
    / "eval_task_cfg"
)
DEFAULT_DEXJOCO_PY_ROOT = ROOT / "third_party" / "dexjoco" / "dexjoco"
DEFAULT_BACKBONE = (
    ROOT / "artifacts" / "opensource_ckpt_links" / "mixed_5task" / "step_055000.pt"
)
DEFAULT_STATS = ROOT / "artifacts" / "mixed_5task" / "dataset_stats.json"
DEFAULT_TEXT_EMBEDDING_CACHE_DIR = (
    ROOT
    / "data"
    / "water_plant_mixed_s0_dewo_v2_pair_20260820_182236"
    / "text_embeds_cache"
)


def _episodes_from_summary(eval_run: Path) -> list[dict[str, Any]]:
    summary_path = eval_run / "summary.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    tasks = payload.get("tasks") or []
    if not tasks:
        raise ValueError(f"No tasks in {summary_path}")
    rows = list(tasks[0]["episode_results"])
    rows.sort(key=lambda row: int(row["seed"]))
    return rows


def _filter_rows(
    rows: list[dict[str, Any]],
    *,
    seed_start: int | None,
    seed_end: int | None,
) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        seed = int(row["seed"])
        if seed_start is not None and seed < seed_start:
            continue
        if seed_end is not None and seed > seed_end:
            continue
        out.append(row)
    return out


def _unique_query_steps(raw: np.ndarray) -> np.ndarray:
    values = np.asarray(raw, dtype=np.int32).reshape(-1)
    seen: set[int] = set()
    ordered: list[int] = []
    for step in values.tolist():
        step_i = int(step)
        if step_i in seen:
            continue
        seen.add(step_i)
        ordered.append(step_i)
    return np.asarray(ordered, dtype=np.int32)


def phase_replay(args: argparse.Namespace) -> None:
    import os

    os.environ.setdefault("MUJOCO_GL", "egl")
    dexjoco_root = Path(args.dexjoco_py_root).expanduser().resolve()
    if str(dexjoco_root) not in sys.path:
        sys.path.insert(0, str(dexjoco_root))

    from eval_dexjoco_fastwam_control import (  # noqa: E402
        _patch_dexjoco_renderer_compat,
        opensource_diffusion_seed,
    )
    from dexjoco_fastwam_adapter import (  # noqa: E402
        DexJoCoFastWAMAdapter,
        DexJoCoFastWAMEvalEnv,
        DexJoCoTaskConfig,
        load_dexjoco_eval_settings,
        load_task_configs,
    )

    _patch_dexjoco_renderer_compat()
    eval_run = Path(args.eval_run).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve() / "obs"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _filter_rows(
        _episodes_from_summary(eval_run),
        seed_start=args.seed_start,
        seed_end=args.seed_end,
    )
    if not rows:
        raise ValueError("No episodes in the requested seed range.")

    summary = json.loads((eval_run / "summary.json").read_text(encoding="utf-8"))
    run_dir = Path(str(summary["run_dir"])).expanduser().resolve()
    settings = load_dexjoco_eval_settings(
        run_dir,
        action_horizon_override=int(args.action_horizon),
        text_embedding_cache_dir_override=args.text_embedding_cache_dir,
    )
    adapter = DexJoCoFastWAMAdapter(settings)
    task_cfgs = load_task_configs(Path(args.task_config_dir).expanduser().resolve())
    task_name = str(args.task)
    matched = [cfg for cfg in task_cfgs if str(cfg["env_name"]) == task_name]
    if not matched:
        raise ValueError(f"Task {task_name} not in {args.task_config_dir}")
    task = DexJoCoTaskConfig.from_yaml(matched[0])
    eval_repeat = int(args.eval_repeat)

    for row in rows:
        seed = int(row["seed"])
        actions_path = Path(str(row["actions_path"]))
        if not actions_path.is_file():
            raise FileNotFoundError(actions_path)
        payload = np.load(actions_path, allow_pickle=False)
        executed = np.asarray(payload["executed_actions"], dtype=np.float32)
        query_steps = _unique_query_steps(payload["policy_query_steps"])
        initial_state = np.asarray(payload["initial_state"], dtype=np.float32).reshape(-1)

        env = DexJoCoFastWAMEvalEnv(
            task,
            seed=seed,
            randomize=False,
            randomize_dynamics=False,
        )
        try:
            obs = env.reset()
            env.click_mouse_warmup()
            got = np.asarray(obs.get("state", []), dtype=np.float32).reshape(-1)
            n = min(got.size, initial_state.size)
            if n == 0 or not np.allclose(got[:n], initial_state[:n], atol=1e-4, rtol=1e-4):
                raise RuntimeError(
                    f"Replay reset mismatch seed={seed}: "
                    f"got {got[:8]} vs saved {initial_state[:8]}"
                )
            query_set = {int(s) for s in query_steps.tolist()}
            images: list[np.ndarray] = []
            proprios: list[np.ndarray] = []
            kept_steps: list[int] = []
            noise_seeds: list[int] = []
            replan_index = 0
            for t in range(int(executed.shape[0])):
                if t in query_set:
                    policy_obs = env.build_policy_obs(adapter)
                    images.append(np.asarray(policy_obs["input_image"], dtype=np.float32))
                    proprios.append(np.asarray(policy_obs["proprio"], dtype=np.float32))
                    kept_steps.append(t)
                    noise_seeds.append(
                        opensource_diffusion_seed(
                            seed, repeat=eval_repeat, replan_index=replan_index
                        )
                    )
                    replan_index += 1
                env.step_rotvec(executed[t])
        finally:
            env.close()

        if not images:
            raise RuntimeError(f"seed={seed}: no policy query steps to replay")
        dest = out_dir / f"seed_{seed:03d}.npz"
        np.savez_compressed(
            dest,
            env_seed=np.int32(seed),
            success=np.bool_(bool(row["success"])),
            episode_steps=np.int32(row["steps"]),
            query_steps=np.asarray(kept_steps, dtype=np.int32),
            noise_seeds=np.asarray(noise_seeds, dtype=np.int32),
            input_image=np.stack(images, axis=0).astype(np.float16),
            proprio=np.stack(proprios, axis=0).astype(np.float32),
            actions_path=np.asarray(str(actions_path)),
        )
        print(
            f"[replay] seed={seed} success={row['success']} "
            f"steps={row['steps']} replans={len(images)} -> {dest}",
            flush=True,
        )


def _cached_text_obs(adapter: Any, task_cfg: dict[str, Any]) -> dict[str, np.ndarray]:
    from dexjoco_fastwam_adapter import load_text_context_arrays

    cache_dir = adapter.text_embedding_cache_dir
    if not cache_dir:
        raise ValueError("text_embedding_cache_dir is required for CFG residual dump")
    posi = adapter.task_prompt(str(task_cfg["prompt"]))
    base = adapter.task_prompt(str(task_cfg["cfg_base_prompt"]))
    context, context_mask = load_text_context_arrays(
        posi,
        text_embedding_cache_dir=cache_dir,
        context_len=int(adapter.context_len),
    )
    negative_context, negative_context_mask = load_text_context_arrays(
        base,
        text_embedding_cache_dir=cache_dir,
        context_len=int(adapter.context_len),
    )
    out = {
        "context": context,
        "context_mask": context_mask,
        "negative_context": negative_context,
        "negative_context_mask": negative_context_mask,
    }
    # v7 subtractor branch (ε_-). Required for gate energy RMS(ε_+ − ε_-).
    fail_raw = task_cfg.get("cfg_failure_prompt")
    if fail_raw is not None and str(fail_raw).strip():
        fail = adapter.task_prompt(str(fail_raw))
        failure_context, failure_context_mask = load_text_context_arrays(
            fail,
            text_embedding_cache_dir=cache_dir,
            context_len=int(adapter.context_len),
        )
        out["failure_context"] = failure_context
        out["failure_context_mask"] = failure_context_mask
    return out


def phase_dump(args: argparse.Namespace) -> None:
    from dexjoco_fastwam_adapter import (  # noqa: E402
        DexJoCoFastWAMAdapter,
        load_dexjoco_eval_settings,
        load_task_configs,
    )
    from run_fastwam_server import _build_policy_from_run  # noqa: E402

    eval_run = Path(args.eval_run).expanduser().resolve()
    obs_dir = Path(args.out_dir).expanduser().resolve() / "obs"
    dump_dir = Path(args.out_dir).expanduser().resolve() / "residual"
    dump_dir.mkdir(parents=True, exist_ok=True)
    rows = _filter_rows(
        _episodes_from_summary(eval_run),
        seed_start=args.seed_start,
        seed_end=args.seed_end,
    )
    summary = json.loads((eval_run / "summary.json").read_text(encoding="utf-8"))
    run_dir = Path(str(summary["run_dir"])).expanduser().resolve()
    ckpt = Path(str(args.checkpoint or summary["model_provenance"]["checkpoint_path"]))
    print(f"Loading FastWAM model on {args.device} ...", flush=True)
    policy = _build_policy_from_run(
        run_dir=run_dir,
        checkpoint=str(ckpt),
        dataset_stats_path=str(args.dataset_stats),
        norm_stats_meta_dir=None,
        device=str(args.device),
        action_horizon=int(args.action_horizon),
        num_inference_steps=int(args.num_inference_steps),
        load_text_encoder=False,
        inference_seed=int(args.inference_seed),
        text_cfg_scale=float(args.text_cfg_scale),
        negative_prompt=None,
        backbone_checkpoint=str(args.backbone_checkpoint) if args.backbone_checkpoint else None,
        uncond_adapter=str(args.uncond_adapter) if args.uncond_adapter else None,
        adaptive_cfg_tau=(
            None if args.adaptive_cfg_tau is None else float(args.adaptive_cfg_tau)
        ),
        cfg_epsilon_l=(
            None if args.cfg_epsilon_l is None else float(args.cfg_epsilon_l)
        ),
        cfg_residual_clip_mode=str(args.cfg_residual_clip_mode),
    )
    settings = load_dexjoco_eval_settings(
        run_dir,
        action_horizon_override=int(args.action_horizon),
        text_embedding_cache_dir_override=args.text_embedding_cache_dir,
    )
    adapter = DexJoCoFastWAMAdapter(settings)
    task_cfgs = load_task_configs(Path(args.task_config_dir).expanduser().resolve())
    matched = [cfg for cfg in task_cfgs if str(cfg["env_name"]) == str(args.task)]
    if not matched:
        raise ValueError(f"Task {args.task} not in {args.task_config_dir}")
    text_obs = _cached_text_obs(adapter, matched[0])
    failures_only = bool(getattr(args, "failures_only", False))
    max_query_step = getattr(args, "max_query_step", None)
    max_query_step = None if max_query_step is None else int(max_query_step)

    for row in rows:
        seed = int(row["seed"])
        if failures_only and bool(row.get("success")):
            print(f"[dump] skip success seed={seed}", flush=True)
            continue
        obs_path = obs_dir / f"seed_{seed:03d}.npz"
        if not obs_path.is_file():
            raise FileNotFoundError(f"Replay obs missing: {obs_path}")
        blob = np.load(obs_path, allow_pickle=False)
        images = np.asarray(blob["input_image"])
        proprio = np.asarray(blob["proprio"], dtype=np.float32)
        query_steps = np.asarray(blob["query_steps"], dtype=np.int32)
        noise_seeds = np.asarray(blob["noise_seeds"], dtype=np.int32)
        n_replan = int(images.shape[0])
        if max_query_step is not None:
            keep = [
                i
                for i in range(n_replan)
                if int(query_steps[i]) <= int(max_query_step)
            ]
        else:
            keep = list(range(n_replan))
        if not keep:
            print(
                f"[dump] skip seed={seed}: no replans with step<={max_query_step}",
                flush=True,
            )
            continue
        token_rms = []
        chunk_rms = []
        exec_rms = []
        token_rms_nfe = []
        mix_weights = []
        gate_exec_rms = []
        kept_query_steps: list[int] = []
        kept_noise_seeds: list[int] = []
        kept_replan_indices: list[int] = []
        for i in keep:
            observation = {
                "input_image": np.asarray(images[i], dtype=np.float32),
                "proprio": proprio[i],
                **text_obs,
            }
            pred = policy.get_action(
                observation,
                options={
                    "seed": int(noise_seeds[i]),
                    "return_cfg_residual": True,
                    "cfg_exec_horizon": int(args.replan_steps),
                    **(
                        {"adaptive_cfg_tau": float(args.adaptive_cfg_tau)}
                        if args.adaptive_cfg_tau is not None
                        else {}
                    ),
                    **(
                        {
                            "cfg_epsilon_l": float(args.cfg_epsilon_l),
                            "cfg_residual_clip_mode": str(args.cfg_residual_clip_mode),
                        }
                        if args.cfg_epsilon_l is not None
                        else {}
                    ),
                },
            )
            if "cfg_exec_rms" not in pred:
                raise RuntimeError("infer_action did not return CFG residual; update FastWAM.")
            token_rms.append(np.asarray(pred["cfg_token_rms"], dtype=np.float32))
            chunk_rms.append(float(np.asarray(pred["cfg_chunk_rms"])))
            exec_rms.append(float(np.asarray(pred["cfg_exec_rms"])))
            token_rms_nfe.append(np.asarray(pred["cfg_token_rms_nfe"], dtype=np.float32))
            mix_weights.append(float(np.asarray(pred.get("cfg_mix_weight", np.nan))))
            gate_exec_rms.append(float(np.asarray(pred.get("cfg_gate_exec_rms", np.nan))))
            kept_query_steps.append(int(query_steps[i]))
            kept_noise_seeds.append(int(noise_seeds[i]))
            kept_replan_indices.append(int(i))
            print(
                f"[dump] seed={seed} replan={i}/{n_replan - 1} step={int(query_steps[i])} "
                f"E_exec={exec_rms[-1]:.4f}",
                flush=True,
            )
        dest = dump_dir / f"seed_{seed:03d}.npz"
        np.savez_compressed(
            dest,
            env_seed=np.int32(seed),
            success=np.bool_(bool(blob["success"])),
            episode_steps=np.int32(blob["episode_steps"]),
            query_steps=np.asarray(kept_query_steps, dtype=np.int32),
            noise_seeds=np.asarray(kept_noise_seeds, dtype=np.int32),
            replan_indices=np.asarray(kept_replan_indices, dtype=np.int32),
            cfg_token_rms=np.stack(token_rms, axis=0),
            cfg_chunk_rms=np.asarray(chunk_rms, dtype=np.float32),
            cfg_exec_rms=np.asarray(exec_rms, dtype=np.float32),
            cfg_token_rms_nfe=np.stack(token_rms_nfe, axis=0),
            cfg_mix_weights=np.asarray(mix_weights, dtype=np.float32),
            cfg_gate_exec_rms=np.asarray(gate_exec_rms, dtype=np.float32),
            text_cfg_scale=np.float32(args.text_cfg_scale),
            adaptive_cfg_tau=(
                np.float32(args.adaptive_cfg_tau)
                if args.adaptive_cfg_tau is not None
                else np.float32(np.nan)
            ),
            max_query_step=(
                np.int32(max_query_step) if max_query_step is not None else np.int32(-1)
            ),
            failures_only=np.bool_(failures_only),
        )
        print(f"[dump] wrote {dest}", flush=True)


def _mean_sem(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    mean = float(values.mean())
    if values.size == 1:
        return mean, float("nan")
    return mean, float(values.std(ddof=1) / math.sqrt(values.size))


def _auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    ok = np.isfinite(scores)
    labels = labels[ok]
    scores = scores[ok]
    pos = scores[labels]
    neg = scores[~labels]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    greater = float(np.mean(pos[:, None] > neg[None, :]))
    equal = float(np.mean(pos[:, None] == neg[None, :]))
    return greater + 0.5 * equal


def _collect_dump_rows(residual_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(residual_dir.glob("seed_*.npz")):
        blob = np.load(path, allow_pickle=False)
        query = np.asarray(blob["query_steps"], dtype=np.int32)
        exec_rms = np.asarray(blob["cfg_exec_rms"], dtype=np.float32)
        chunk_rms = np.asarray(blob["cfg_chunk_rms"], dtype=np.float32)
        token_rms = np.asarray(blob["cfg_token_rms"], dtype=np.float32)
        mix_weights = (
            np.asarray(blob["cfg_mix_weights"], dtype=np.float32)
            if "cfg_mix_weights" in blob.files
            else np.asarray([], dtype=np.float32)
        )
        gate_values = (
            np.asarray(blob["cfg_gate_exec_rms"], dtype=np.float32)
            if "cfg_gate_exec_rms" in blob.files
            else np.asarray([], dtype=np.float32)
        )
        steps = int(blob["episode_steps"])
        success = bool(blob["success"])
        progress = query.astype(np.float64) / max(1, steps)
        peak = int(np.argmax(exec_rms)) if exec_rms.size else 0
        early = exec_rms[progress < 0.3] if exec_rms.size else np.asarray([])
        late = exec_rms[progress > 0.7] if exec_rms.size else np.asarray([])
        rows.append(
            {
                "seed": int(blob["env_seed"]),
                "success": success,
                "steps": steps,
                "query_steps": query,
                "progress": progress,
                "exec_rms": exec_rms,
                "chunk_rms": chunk_rms,
                "token_rms": token_rms,
                "mix_weights": mix_weights,
                "gate_values": gate_values,
                "guided_fraction": (
                    float(np.mean(mix_weights[np.isfinite(mix_weights)] > 0.0))
                    if mix_weights.size and np.isfinite(mix_weights).any()
                    else float("nan")
                ),
                "peak_step": int(query[peak]) if query.size else 0,
                "peak_progress": float(progress[peak]) if progress.size else float("nan"),
                "early_mean": float(early.mean()) if early.size else float("nan"),
                "late_mean": float(late.mean()) if late.size else float("nan"),
                "cv": float(exec_rms.std() / (exec_rms.mean() + 1e-8)) if exec_rms.size else float("nan"),
            }
        )
    return rows


def phase_summarize(args: argparse.Namespace) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from scripts.analysis._plot_style import INK, MUTED, apply_style, panel_label, polish, save_figure

    out_dir = Path(args.out_dir).expanduser().resolve()
    residual_dir = out_dir / "residual"
    rows = _collect_dump_rows(residual_dir)
    if not rows:
        raise FileNotFoundError(f"No residual dumps in {residual_dir}")

    succ = [row for row in rows if row["success"]]
    fail = [row for row in rows if not row["success"]]

    def stacked_cv(group: list[dict[str, Any]]) -> float:
        if not group:
            return float("nan")
        return float(np.nanmean([row["cv"] for row in group]))

    first_e = np.asarray(
        [float(row["exec_rms"][0]) for row in rows if row["exec_rms"].size],
        dtype=np.float64,
    )
    first_labels = np.asarray(
        [not row["success"] for row in rows if row["exec_rms"].size],
        dtype=bool,
    )
    mean_e = np.asarray(
        [float(row["exec_rms"].mean()) for row in rows if row["exec_rms"].size],
        dtype=np.float64,
    )

    report = {
        "n_episodes": len(rows),
        "n_success": len(succ),
        "n_fail": len(fail),
        "protocol": "diagnostic_on_eval_0_49_inference_NON_STANDARD",
        "aligns_to": "current_closed_loop_inference_states",
        "not_pass20": True,
        "success_exec_rms_mean_sem": _mean_sem(
            np.concatenate([row["exec_rms"] for row in succ]) if succ else np.asarray([])
        ),
        "fail_exec_rms_mean_sem": _mean_sem(
            np.concatenate([row["exec_rms"] for row in fail]) if fail else np.asarray([])
        ),
        "success_early_mean_sem": _mean_sem(np.asarray([row["early_mean"] for row in succ])),
        "success_late_mean_sem": _mean_sem(np.asarray([row["late_mean"] for row in succ])),
        "fail_early_mean_sem": _mean_sem(np.asarray([row["early_mean"] for row in fail])),
        "fail_late_mean_sem": _mean_sem(np.asarray([row["late_mean"] for row in fail])),
        "mean_cv_success": stacked_cv(succ),
        "mean_cv_fail": stacked_cv(fail),
        "auroc_first_replan_fail": _auroc(first_labels, first_e),
        "auroc_episode_mean_fail": _auroc(first_labels, mean_e),
        "median_peak_progress_success": float(
            np.median([row["peak_progress"] for row in succ])
        )
        if succ
        else float("nan"),
        "median_peak_progress_fail": float(
            np.median([row["peak_progress"] for row in fail])
        )
        if fail
        else float("nan"),
        "flat_residual": bool(stacked_cv(rows) < 0.15),
        "guided_chunk_fraction_mean": _mean_sem(
            np.asarray([row["guided_fraction"] for row in rows], dtype=np.float64)
        ),
        "episodes": [
            {
                "seed": row["seed"],
                "success": row["success"],
                "steps": row["steps"],
                "peak_step": row["peak_step"],
                "peak_progress": row["peak_progress"],
                "early_mean": row["early_mean"],
                "late_mean": row["late_mean"],
                "cv": row["cv"],
                "guided_fraction": row["guided_fraction"],
            }
            for row in rows
        ],
    }
    report_path = out_dir / "cfg_residual_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.1))
    ax_step, ax_prog, ax_box = axes

    def _curve(group: list[dict[str, Any]], *, progress: bool):
        xs: list[float] = []
        ys: list[float] = []
        for row in group:
            x = row["progress"] if progress else row["query_steps"].astype(np.float64)
            for xi, yi in zip(x.tolist(), row["exec_rms"].tolist()):
                xs.append(float(xi))
                ys.append(float(yi))
        return np.asarray(xs), np.asarray(ys)

    def _binned(x: np.ndarray, y: np.ndarray, edges: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        centers = 0.5 * (edges[:-1] + edges[1:])
        mean = np.full(centers.shape, np.nan)
        sem = np.full(centers.shape, np.nan)
        for i in range(len(edges) - 1):
            sel = (x >= edges[i]) & (x < edges[i + 1])
            if int(sel.sum()) == 0:
                continue
            m, s = _mean_sem(y[sel])
            mean[i] = m
            sem[i] = s
        return centers, mean, sem

    for group, color, label in (
        (succ, "#3f6f8a", f"success n={len(succ)}"),
        (fail, "#b4533a", f"fail n={len(fail)}"),
    ):
        x, y = _curve(group, progress=False)
        if x.size == 0:
            continue
        centers, mean, sem = _binned(x, y, np.arange(0.0, 1000.0 + 48.0, 48.0))
        ok = np.isfinite(mean)
        ax_step.plot(centers[ok], mean[ok], color=color, lw=1.6, label=label)
        if np.isfinite(sem[ok]).any():
            ax_step.fill_between(
                centers[ok],
                mean[ok] - np.nan_to_num(sem[ok]),
                mean[ok] + np.nan_to_num(sem[ok]),
                color=color,
                alpha=0.18,
                lw=0,
            )
    ax_step.set_xlabel("env step")
    ax_step.set_ylabel(r"CFG residual $E$ (exec 24)")
    ax_step.set_title("On this policy's 0–49 states")
    ax_step.legend(loc="upper right")
    polish(ax_step)
    panel_label(ax_step, "a")

    for group, color, label in (
        (succ, "#3f6f8a", "success"),
        (fail, "#b4533a", "fail"),
    ):
        x, y = _curve(group, progress=True)
        if x.size == 0:
            continue
        centers, mean, sem = _binned(x, y, np.linspace(0.0, 1.0, 11))
        ok = np.isfinite(mean)
        ax_prog.plot(centers[ok], mean[ok], color=color, lw=1.6, label=label)
        if np.isfinite(sem[ok]).any():
            ax_prog.fill_between(
                centers[ok],
                mean[ok] - np.nan_to_num(sem[ok]),
                mean[ok] + np.nan_to_num(sem[ok]),
                color=color,
                alpha=0.18,
                lw=0,
            )
    ax_prog.set_xlabel("episode progress t / T")
    ax_prog.set_ylabel(r"$E$")
    ax_prog.set_title("Same energy vs progress")
    ax_prog.set_xlim(0.0, 1.0)
    polish(ax_prog)
    panel_label(ax_prog, "b")

    def _vals(group: list[dict[str, Any]], key: str) -> list[float]:
        return [float(row[key]) for row in group if math.isfinite(float(row[key]))]

    data = [
        _vals(succ, "early_mean"),
        _vals(succ, "late_mean"),
        _vals(fail, "early_mean"),
        _vals(fail, "late_mean"),
    ]
    bp = ax_box.boxplot(
        data,
        labels=["S early", "S late", "F early", "F late"],
        patch_artist=True,
        widths=0.55,
        medianprops={"color": INK, "lw": 1.2},
        whiskerprops={"color": MUTED},
        capprops={"color": MUTED},
        flierprops={"marker": "o", "markersize": 3, "color": MUTED},
    )
    colors = ["#d7e4ec", "#3f6f8a", "#f4d6ce", "#b4533a"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor(INK)
        patch.set_alpha(0.9)
    ax_box.set_ylabel(r"$E$")
    ax_box.set_title("Early vs late on the same traj")
    polish(ax_box)
    panel_label(ax_box, "c")

    fig.suptitle(
        "v5 CFG residual on water_plant screen 0–49 (step_001000, w=2)",
        color=INK,
        y=1.03,
    )
    fig.tight_layout(w_pad=2.2)
    fig_path = out_dir / "fig_cfg_residual_on_eval_0_49.png"
    save_figure(fig, fig_path)
    print(json.dumps({k: report[k] for k in report if k != "episodes"}, indent=2))
    print(f"wrote {report_path}")
    print(f"wrote {fig_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("replay", "dump", "summarize"), required=True)
    parser.add_argument("--eval-run", type=Path, default=DEFAULT_EVAL_RUN)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--task", type=str, default="water_plant")
    parser.add_argument("--task-config-dir", type=Path, default=DEFAULT_TASK_YAML_DIR)
    parser.add_argument("--dexjoco-py-root", type=Path, default=DEFAULT_DEXJOCO_PY_ROOT)
    parser.add_argument("--seed-start", type=int, default=None)
    parser.add_argument("--seed-end", type=int, default=None)
    parser.add_argument(
        "--failures-only",
        action="store_true",
        help="Dump CFG energy only on failed baseline episodes (skip successes).",
    )
    parser.add_argument(
        "--max-query-step",
        type=int,
        default=None,
        help=(
            "Only dump replans with env step <= this value. "
            "Use to skip late failure nodes past the recoverable window "
            "(e.g. 400 ≈ past water_plant success lengths)."
        ),
    )
    parser.add_argument("--eval-repeat", type=int, default=0)
    parser.add_argument("--action-horizon", type=int, default=32)
    parser.add_argument("--replan-steps", type=int, default=24)
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--text-cfg-scale", type=float, default=2.0)
    parser.add_argument("--adaptive-cfg-tau", type=float, default=None)
    parser.add_argument("--cfg-epsilon-l", type=float, default=None)
    parser.add_argument(
        "--cfg-residual-clip-mode",
        choices=("rms", "elementwise"),
        default="rms",
    )
    parser.add_argument("--inference-seed", type=int, default=20260812)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--backbone-checkpoint", type=str, default=str(DEFAULT_BACKBONE))
    parser.add_argument("--uncond-adapter", type=str, default=None)
    parser.add_argument("--dataset-stats", type=Path, default=DEFAULT_STATS)
    parser.add_argument(
        "--text-embedding-cache-dir",
        type=str,
        default=os.environ.get(
            "FASTWAM_TEXT_EMBEDDING_CACHE_DIR",
            str(DEFAULT_TEXT_EMBEDDING_CACHE_DIR),
        ),
    )
    return parser.parse_args()


def _canonicalize_args(args: argparse.Namespace) -> argparse.Namespace:
    """Resolve every filesystem argument once so logs and consumers agree."""
    for name in (
        "eval_run",
        "out_dir",
        "task_config_dir",
        "dexjoco_py_root",
        "dataset_stats",
    ):
        value = getattr(args, name)
        setattr(args, name, Path(value).expanduser().resolve())
    for name in ("checkpoint", "backbone_checkpoint", "uncond_adapter"):
        value = getattr(args, name)
        if value:
            setattr(args, name, str(Path(value).expanduser().resolve()))
    args.text_embedding_cache_dir = str(
        Path(args.text_embedding_cache_dir).expanduser().resolve()
    )
    return args


def main() -> int:
    args = _canonicalize_args(parse_args())
    print(f"[paths] fastwam_root={ROOT}", flush=True)
    print(
        f"[paths] text_embedding_cache_dir={args.text_embedding_cache_dir}",
        flush=True,
    )
    Path(args.out_dir).expanduser().resolve().mkdir(parents=True, exist_ok=True)
    if args.phase == "replay":
        phase_replay(args)
    elif args.phase == "dump":
        phase_dump(args)
    else:
        phase_summarize(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
