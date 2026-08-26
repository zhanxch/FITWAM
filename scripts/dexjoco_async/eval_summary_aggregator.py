"""Merge sharded DexJoCo eval summaries into one combined summary.

Decoupled and dependency-light: only needs the standard library + ``numpy``
(used by the eval script's metric values). It does NOT import ``torch`` or
``mujoco``, so it can run in either conda environment or even bare ``base``.

Usage as a library::

    from eval_summary_aggregator import merge_shard_summaries, write_combined
    combined = merge_shard_summaries(shard_paths, label="blocking_stride24")
    write_combined(combined, out_dir=Path("evaluate_results/.../step_006500"))

Usage as a CLI::

    python eval_summary_aggregator.py shard_0/summary.json shard_1/summary.json \
        --output-dir evaluate_results/dexjoco/water_plant/step_006500 \
        --label blocking_stride24
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

# Keys re-aggregated from per-episode metrics. Must match the ``metric_means``
# block produced by ``eval_dexjoco_fastwam_control.evaluate_task``.
METRIC_KEYS: tuple[str, ...] = (
    "inference_latency_mean_s",
    "inference_latency_p95_s",
    "action_delta_l2_mean",
    "action_jerk_l2_mean",
    "oscillation_sign_flip_rate",
    "queue_underruns",
    "queue_wait_s",
    "async_replan_delays",
)


@dataclass
class ShardRecord:
    """Provenance for one shard in the combined summary."""

    shard_id: int
    summary_path: str
    episodes: int
    successes: int
    success_rate: float
    policy_port: int | None
    base_seed: int | None


def _mean(values: list[float]) -> float | None:
    filtered = [v for v in values if v is not None]
    if not filtered:
        return None
    return float(sum(filtered) / len(filtered))


def _aggregate_metric_from_episodes(
    episodes: list[dict[str, Any]], key: str
) -> float | None:
    vals: list[float] = []
    for ep in episodes:
        metrics = ep.get("metrics") or {}
        v = metrics.get(key)
        if v is not None:
            vals.append(float(v))
    return _mean(vals) if vals else None


def _recompute_task(task: dict[str, Any], merged_episodes: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(merged_episodes)
    successes = sum(1 for ep in merged_episodes if ep.get("success"))
    return {
        "env_name": task.get("env_name"),
        "prompt": task.get("prompt"),
        "cfg_base_prompt": task.get("cfg_base_prompt"),
        "cfg_failure_prompt": task.get("cfg_failure_prompt"),
        "dual_arm": task.get("dual_arm"),
        "camera_key": task.get("camera_key"),
        "episodes": total,
        "successes": int(successes),
        "success_rate": float(successes / total if total else 0.0),
        "metric_means": {
            key: _aggregate_metric_from_episodes(merged_episodes, key)
            for key in METRIC_KEYS
        },
        "episode_results": merged_episodes,
    }


def _merge_task_episodes(
    task_name: str,
    shard_payloads: list[tuple[int, dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Concatenate and globally re-index episode_results for one task across shards.

    Episodes are ordered by (shard_id, local episode index) and renumbered with a
    contiguous global index. ``video_path`` / ``actions_path`` are kept as-is
    (they are absolute paths inside each shard's output directory).
    """
    merged: list[dict[str, Any]] = []
    global_idx = 0
    for shard_id, payload in sorted(shard_payloads, key=lambda x: x[0]):
        task = _find_task(payload, task_name)
        if task is None:
            continue
        for ep in task.get("episode_results", []):
            row = dict(ep)
            row["episode"] = global_idx
            row["shard"] = int(shard_id)
            merged.append(row)
            global_idx += 1
    return merged


def _find_task(payload: dict[str, Any], env_name: str) -> dict[str, Any] | None:
    for task in payload.get("tasks", []):
        if task.get("env_name") == env_name:
            return task
    return None


def _ordered_task_names(shard_payloads: list[tuple[int, dict[str, Any]]]) -> list[str]:
    seen: dict[str, None] = {}
    for _, payload in sorted(shard_payloads, key=lambda x: x[0]):
        for task in payload.get("tasks", []):
            name = task.get("env_name")
            if name is not None:
                seen.setdefault(name, None)
    return list(seen.keys())


def _shard_totals(payload: dict[str, Any]) -> tuple[int, int]:
    episodes = 0
    successes = 0
    for task in payload.get("tasks", []):
        episodes += int(task.get("episodes", 0))
        successes += int(task.get("successes", 0))
    return episodes, successes


def merge_shard_summaries(
    shard_paths: Iterable[Path],
    *,
    label: str | None = None,
    extra_top_level: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Merge multiple per-shard ``summary.json`` files into one combined summary.

    The returned dict has the same shape as a single ``eval_dexjoco_fastwam_control``
    summary, so downstream reporting tools can consume it unchanged.
    work unchanged. A ``shards`` provenance list is added.
    """
    shard_payloads: list[tuple[int, dict[str, Any], Path]] = []
    for path in shard_paths:
        path = Path(path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"shard summary not found: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        shard_id = int(payload.get("shard_id", len(shard_payloads)))
        shard_payloads.append((shard_id, payload, path))

    shard_payloads.sort(key=lambda x: x[0])
    indexed = [(sid, payload) for sid, payload, _ in shard_payloads]

    task_names = _ordered_task_names(indexed)
    merged_tasks: list[dict[str, Any]] = []
    total_episodes = 0
    total_successes = 0
    for name in task_names:
        merged_eps = _merge_task_episodes(name, indexed)
        template_task = _find_task(indexed[0][1], name) or {"env_name": name}
        merged_task = _recompute_task(template_task, merged_eps)
        merged_tasks.append(merged_task)
        total_episodes += int(merged_task["episodes"])
        total_successes += int(merged_task["successes"])

    # Top-level scalar fields: take from the first shard, override below.
    base = dict(shard_payloads[0][1]) if shard_payloads else {}
    action_horizon = base.get("action_horizon")
    replan_steps = base.get("replan_steps")
    overlap_steps = (
        int(max(0, int(action_horizon) - int(replan_steps)))
        if action_horizon is not None and replan_steps is not None
        else base.get("overlap_steps")
    )
    overlap_ratio = (
        float(overlap_steps / int(action_horizon))
        if action_horizon and overlap_steps is not None
        else base.get("overlap_ratio")
    )

    shard_records: list[ShardRecord] = []
    for shard_id, payload, path in shard_payloads:
        eps, succ = _shard_totals(payload)
        shard_records.append(
            ShardRecord(
                shard_id=shard_id,
                summary_path=str(path),
                episodes=eps,
                successes=succ,
                success_rate=float(succ / eps if eps else 0.0),
                policy_port=payload.get("policy_port"),
                base_seed=payload.get("base_seed", payload.get("seed")),
            )
        )

    summary: dict[str, Any] = {
        "label": label or base.get("label"),
        "run_dir": base.get("run_dir"),
        "policy_host": base.get("policy_host"),
        "policy_ports": [r.policy_port for r in shard_records if r.policy_port is not None],
        "control_mode": base.get("control_mode"),
        "async_fallback": base.get("async_fallback"),
        "replan_steps": replan_steps,
        "action_horizon": action_horizon,
        "overlap_steps": overlap_steps,
        "overlap_ratio": overlap_ratio,
        "low_pass_alpha": base.get("low_pass_alpha"),
        "low_pass_continuous_dim": base.get("low_pass_continuous_dim"),
        "episodes_per_task": total_episodes,
        "num_tasks": len(merged_tasks),
        "total_episodes": int(total_episodes),
        "total_successes": int(total_successes),
        "overall_success_rate": float(total_successes / total_episodes if total_episodes else 0.0),
        "randomize": base.get("randomize"),
        "randomize_dynamics": base.get("randomize_dynamics"),
        "seed": base.get("seed"),
        "save_actions": base.get("save_actions"),
        "action_clip": base.get("action_clip"),
        "clip_max_xyz_step": base.get("clip_max_xyz_step"),
        "clip_max_dz_down": base.get("clip_max_dz_down"),
        "num_shards": len(shard_records),
        "shards": [r.__dict__ for r in shard_records],
        "tasks": merged_tasks,
    }
    if extra_top_level:
        summary.update(extra_top_level)
    return summary


def write_combined(summary: dict[str, Any], out_dir: Path) -> dict[str, Path]:
    """Write ``summary.json``, ``summary.csv`` and ``video_manifest.csv`` into ``out_dir``.

    Returns a dict of the paths written. Mirrors the outputs produced by a single
    ``eval_dexjoco_fastwam_control`` run so downstream tooling is unchanged.
    """
    import numpy as np  # local import: keep module importable without numpy

    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    csv_rows: list[dict[str, Any]] = []
    for task in summary.get("tasks", []):
        means = task.get("metric_means", {}) or {}
        csv_rows.append(
            {
                "label": summary.get("label"),
                "control_mode": summary.get("control_mode"),
                "replan_steps": summary.get("replan_steps"),
                "overlap_steps": summary.get("overlap_steps"),
                "low_pass_alpha": summary.get("low_pass_alpha"),
                "env_name": task.get("env_name"),
                "episodes": task.get("episodes"),
                "successes": task.get("successes"),
                "success_rate": task.get("success_rate"),
                **means,
            }
        )
    csv_path = out_dir / "summary.csv"
    if csv_rows:
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)

    video_rows: list[dict[str, Any]] = []
    for task in summary.get("tasks", []):
        for ep in task.get("episode_results", []):
            if ep.get("video_path"):
                video_rows.append(
                    {
                        "label": summary.get("label"),
                        "env_name": task.get("env_name"),
                        "episode": ep.get("episode"),
                        "seed": ep.get("seed"),
                        "success": ep.get("success"),
                        "steps": ep.get("steps"),
                        "shard": ep.get("shard"),
                        "video_path": ep.get("video_path"),
                        "actions_path": ep.get("actions_path"),
                    }
                )
    video_path = out_dir / "video_manifest.csv"
    if video_rows:
        with video_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(video_rows[0].keys()))
            writer.writeheader()
            writer.writerows(video_rows)

    return {"summary": summary_path, "csv": csv_path, "video_manifest": video_path}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge sharded DexJoCo eval summary.json files into one combined summary."
    )
    parser.add_argument("shard_summaries", type=Path, nargs="+", help="Paths to per-shard summary.json")
    parser.add_argument("--output-dir", type=Path, required=True, help="Where to write combined outputs")
    parser.add_argument("--label", type=str, default=None, help="Label for the combined summary")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    combined = merge_shard_summaries(args.shard_summaries, label=args.label)
    paths = write_combined(combined, args.output_dir)
    total = combined["total_episodes"]
    succ = combined["total_successes"]
    rate = combined["overall_success_rate"]
    print(f"[aggregator] shards={combined['num_shards']} tasks={combined['num_tasks']}", flush=True)
    print(f"[aggregator] overall: {succ}/{total} ({100 * rate:.1f}%)", flush=True)
    print(f"[aggregator] summary={paths['summary']}", flush=True)


if __name__ == "__main__":
    main()
