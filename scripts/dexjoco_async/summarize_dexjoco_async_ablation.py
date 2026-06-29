#!/usr/bin/env python3
"""Summarize DexJoCo FastWAM async/LPF ablation outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _phase_name(path: Path, root: Path) -> str:
    rel = path.relative_to(root)
    return rel.parts[0] if rel.parts else ""


def _condition_name(path: Path) -> str:
    return path.parent.name


def _read_summary(path: Path, root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    videos: list[dict[str, Any]] = []
    phase = _phase_name(path, root)
    condition = _condition_name(path)
    for task in payload.get("tasks", []):
        means = task.get("metric_means", {}) or {}
        row = {
            "phase": phase,
            "condition": condition,
            "label": payload.get("label"),
            "control_mode": payload.get("control_mode"),
            "replan_steps": payload.get("replan_steps"),
            "overlap_steps": payload.get("overlap_steps"),
            "overlap_ratio": payload.get("overlap_ratio"),
            "low_pass_alpha": payload.get("low_pass_alpha"),
            "async_fallback": payload.get("async_fallback"),
            "env_name": task.get("env_name"),
            "episodes": task.get("episodes"),
            "successes": task.get("successes"),
            "success_rate": task.get("success_rate"),
            "latency_mean_s": means.get("inference_latency_mean_s"),
            "latency_p95_s": means.get("inference_latency_p95_s"),
            "latency_max_s": means.get("inference_latency_max_s"),
            "action_delta_l2_mean": means.get("action_delta_l2_mean"),
            "action_jerk_l2_mean": means.get("action_jerk_l2_mean"),
            "oscillation_sign_flip_rate": means.get("oscillation_sign_flip_rate"),
            "queue_underruns": means.get("queue_underruns"),
            "queue_wait_s": means.get("queue_wait_s"),
            "async_replan_delays": means.get("async_replan_delays"),
            "summary_path": str(path),
        }
        rows.append(row)
        for ep in task.get("episode_results", []):
            video_path = ep.get("video_path")
            actions_path = ep.get("actions_path")
            if not video_path and not actions_path:
                continue
            videos.append(
                {
                    "phase": phase,
                    "condition": condition,
                    "env_name": task.get("env_name"),
                    "episode": ep.get("episode"),
                    "seed": ep.get("seed"),
                    "success": ep.get("success"),
                    "steps": ep.get("steps"),
                    "elapsed_s": ep.get("elapsed_s"),
                    "video_path": video_path,
                    "actions_path": actions_path,
                }
            )
    return rows, videos


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return str(value)


def _table_lines(title: str, rows: list[dict[str, Any]]) -> list[str]:
    ordered = sorted(
        rows,
        key=lambda r: (
            str(r.get("phase")),
            -float(r.get("success_rate") or 0),
            float(r.get("action_jerk_l2_mean") or 1e9),
        ),
    )
    lines = [
        f"## {title}",
        "",
        "| phase | condition | mode | replan | LPF | success | jerk | sign flip | latency mean | underruns | wait s |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in ordered:
        success = f"{row.get('successes')}/{row.get('episodes')} ({_fmt(100 * float(row.get('success_rate') or 0), 3)}%)"
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("phase")),
                    str(row.get("condition")),
                    str(row.get("control_mode")),
                    _fmt(row.get("replan_steps"), 0),
                    _fmt(row.get("low_pass_alpha")),
                    success,
                    _fmt(row.get("action_jerk_l2_mean")),
                    _fmt(row.get("oscillation_sign_flip_rate")),
                    _fmt(row.get("latency_mean_s")),
                    _fmt(row.get("queue_underruns")),
                    _fmt(row.get("queue_wait_s")),
                ]
            )
            + " |"
        )
    return lines


def _write_md(path: Path, rows: list[dict[str, Any]]) -> None:
    official = [r for r in rows if not str(r["phase"]).startswith("explore")]
    exploratory = [r for r in rows if str(r["phase"]).startswith("explore")]
    lines = ["# DexJoCo FastWAM Async/LPF Summary", ""]
    lines.extend(_table_lines("Official PLAN Phases", official))
    if exploratory:
        lines.extend([""])
        lines.extend(_table_lines("Exploratory Runs", exploratory))
    lines.extend(
        [
            "",
            "## Reading",
            "",
            "- Compare success first, then jerk/sign-flip as motion smoothness proxies.",
            "- Treat queue underruns and wait time as async timing costs.",
            "- Keep exploratory runs separate from official PLAN_dex phases.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    root = args.run_root.resolve()
    out = args.output_dir.resolve() if args.output_dir else root
    rows: list[dict[str, Any]] = []
    videos: list[dict[str, Any]] = []
    for path in sorted(root.glob("**/summary.json")):
        if ".eval.lock" in path.parts:
            continue
        parsed_rows, parsed_videos = _read_summary(path, root)
        rows.extend(parsed_rows)
        videos.extend(parsed_videos)

    _write_csv(out / "combined_summary.csv", rows)
    _write_csv(out / "combined_video_manifest.csv", videos)
    _write_md(out / "combined_summary.md", rows)
    print(f"rows={len(rows)} videos={len(videos)} out={out}")


if __name__ == "__main__":
    main()
