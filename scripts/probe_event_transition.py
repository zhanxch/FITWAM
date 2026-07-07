#!/usr/bin/env python3
"""Estimate event-transition probability from causal action-trend prediction error.

The probe is intentionally small: for each timestep it fits a linear trend to the
previous action window, predicts the next action, and maps the robustly-normalized
prediction error to p in [0, 1]. High p means the recent trend stopped explaining
the next action well.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq


ACTION_GROUPS = {
    "eef": (0, 6),
    "hand": (6, 22),
}


def read_episode(path: Path) -> dict[str, Any]:
    table = pq.ParquetFile(path).read(columns=["action", "timestamp", "frame_index", "episode_index"])
    action_col = table["action"].combine_chunks()
    actions = np.asarray(action_col.to_pylist(), dtype=np.float32)
    return {
        "path": path,
        "actions": actions,
        "timestamp": np.asarray(table["timestamp"].to_pylist(), dtype=np.float32),
        "frame_index": np.asarray(table["frame_index"].to_pylist(), dtype=np.int64),
        "episode_index": int(table["episode_index"][0].as_py()),
    }


def low_pass_filter(values: np.ndarray, alpha: float, release_alpha: float | None = None) -> np.ndarray:
    """Causal one-pole low-pass filter.

    A smaller release_alpha makes p decay slowly after a transition, which avoids
    rapid on/off jitter in event records.
    """
    if release_alpha is None:
        release_alpha = alpha
    out = np.full_like(values, np.nan, dtype=np.float32)
    prev = float("nan")
    for i, value in enumerate(values):
        if not np.isfinite(value):
            continue
        if not np.isfinite(prev):
            prev = float(value)
        else:
            step_alpha = alpha if value >= prev else release_alpha
            prev = (1.0 - step_alpha) * prev + step_alpha * float(value)
        out[i] = prev
    return out


def moving_average(values: np.ndarray, width: int) -> np.ndarray:
    if width <= 1:
        return values.copy()
    out = values.copy()
    valid = np.isfinite(values)
    for i in range(len(values)):
        start = max(0, i - width + 1)
        window = values[start : i + 1]
        window_valid = valid[start : i + 1]
        if window_valid.any():
            out[i] = float(window[window_valid].mean())
    return out


def trend_errors(
    actions: np.ndarray,
    scale: np.ndarray,
    history: int,
    action_groups: list[str],
) -> dict[str, np.ndarray]:
    normalized = actions / scale
    n, dim = normalized.shape
    error = np.full(n, np.nan, dtype=np.float32)
    group_errors = {
        name: np.full(n, np.nan, dtype=np.float32)
        for name in ACTION_GROUPS
        if ACTION_GROUPS[name][1] <= dim
    }

    x = np.arange(history, dtype=np.float32)
    x_centered = x - x.mean()
    denom = float(np.sum(x_centered * x_centered))

    for t in range(history, n):
        window = normalized[t - history : t]
        y_mean = window.mean(axis=0)
        slope = (x_centered[:, None] * (window - y_mean)).sum(axis=0) / denom
        pred = y_mean + slope * history
        residual = normalized[t] - pred

        selected_group_mse = []
        for name, (start, end) in ACTION_GROUPS.items():
            if end > dim:
                continue
            value = float(np.sqrt(np.mean(residual[start:end] ** 2)))
            group_errors[name][t] = value
            if name in action_groups:
                selected_group_mse.append(value * value)
        error[t] = (
            float(np.sqrt(np.mean(selected_group_mse)))
            if selected_group_mse
            else float(np.sqrt(np.mean(residual**2)))
        )

    return {"error": error, **{f"{k}_error": v for k, v in group_errors.items()}}


def robust_probability(errors: np.ndarray, low_q: float, high_q: float) -> tuple[np.ndarray, dict[str, float]]:
    valid = errors[np.isfinite(errors)]
    lo = float(np.quantile(valid, low_q))
    hi = float(np.quantile(valid, high_q))
    if hi <= lo:
        hi = lo + 1e-6
    p = np.clip((errors - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
    p[~np.isfinite(errors)] = np.nan
    return p, {"low": lo, "high": hi, "median": float(np.median(valid))}


def peak_indices(p: np.ndarray, top_k: int, min_gap: int, threshold: float, min_index: int) -> list[int]:
    candidates = [
        i
        for i, value in enumerate(p)
        if i >= min_index and np.isfinite(value) and value >= threshold
    ]
    candidates.sort(key=lambda i: float(p[i]), reverse=True)
    chosen: list[int] = []
    for idx in candidates:
        if all(abs(idx - other) >= min_gap for other in chosen):
            chosen.append(idx)
        if len(chosen) >= top_k:
            break
    return sorted(chosen)


def write_csv(path: Path, episodes: list[dict[str, Any]]) -> None:
    fieldnames = [
        "episode_index",
        "frame_index",
        "timestamp",
        "error",
        "event_transition_p",
        "event_transition_p_lpf",
        "event_transition_p_smooth",
        "eef_error",
        "hand_error",
        "is_peak",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for ep in episodes:
            peaks = set(ep["peaks"])
            for i in range(len(ep["frame_index"])):
                writer.writerow(
                    {
                        "episode_index": ep["episode_index"],
                        "frame_index": int(ep["frame_index"][i]),
                        "timestamp": float(ep["timestamp"][i]),
                        "error": finite_or_blank(ep["error"][i]),
                        "event_transition_p": finite_or_blank(ep["p"][i]),
                        "event_transition_p_lpf": finite_or_blank(ep["p_lpf"][i]),
                        "event_transition_p_smooth": finite_or_blank(ep["p_smooth"][i]),
                        "eef_error": finite_or_blank(ep.get("eef_error", [np.nan])[i]),
                        "hand_error": finite_or_blank(ep.get("hand_error", [np.nan])[i]),
                        "is_peak": int(i in peaks),
                    }
                )


def finite_or_blank(value: float) -> str:
    if not math.isfinite(float(value)):
        return ""
    return f"{float(value):.6f}"


def write_html(path: Path, episodes: list[dict[str, Any]], top_episodes: int) -> None:
    traces = []
    for ep in episodes[:top_episodes]:
        traces.append(
            {
                "x": ep["frame_index"].tolist(),
                "y": ep["p_smooth"].tolist(),
                "mode": "lines",
                "name": f"ep {ep['episode_index']:03d}",
            }
        )
    payload = json.dumps(traces)
    path.write_text(
        f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Event transition probe</title>
  <script src="https://cdn.plot.ly/plotly-2.30.0.min.js"></script>
</head>
<body>
  <div id="plot" style="width:100%;height:720px;"></div>
  <script>
    const traces = {payload};
    Plotly.newPlot('plot', traces, {{
      title: 'Smoothed event transition probability from action-trend probe',
      xaxis: {{title: 'frame'}},
      yaxis: {{title: 'p', range: [0, 1]}},
      hovermode: 'x unified'
    }});
  </script>
</body>
</html>
""",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/event_transition_probe"))
    parser.add_argument("--history", type=int, default=12)
    parser.add_argument(
        "--action-groups",
        nargs="+",
        choices=sorted(ACTION_GROUPS),
        default=["eef", "hand"],
        help="Action groups used to compute p. Group errors are still logged separately.",
    )
    parser.add_argument("--smooth", type=int, default=5)
    parser.add_argument("--lpf-alpha", type=float, default=0.28)
    parser.add_argument("--lpf-release-alpha", type=float, default=0.08)
    parser.add_argument("--low-quantile", type=float, default=0.10)
    parser.add_argument("--high-quantile", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--top-episodes-html", type=int, default=12)
    parser.add_argument("--ignore-initial-frames", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    parquet_paths = sorted((args.dataset_dir / "data").glob("chunk-*/*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"No parquet episodes found under {args.dataset_dir / 'data'}")

    episodes = [read_episode(path) for path in parquet_paths]
    all_actions = np.concatenate([ep["actions"] for ep in episodes], axis=0)
    scale = all_actions.std(axis=0).astype(np.float32)
    scale[scale < 1e-6] = 1.0

    all_errors = []
    for ep in episodes:
        err = trend_errors(ep["actions"], scale, args.history, args.action_groups)
        ep.update(err)
        all_errors.append(err["error"])

    p_all, calibration = robust_probability(
        np.concatenate(all_errors),
        low_q=args.low_quantile,
        high_q=args.high_quantile,
    )

    cursor = 0
    summary_events = []
    for ep in episodes:
        n = len(ep["actions"])
        ep["p"] = p_all[cursor : cursor + n]
        cursor += n
        ep["p_lpf"] = low_pass_filter(ep["p"], args.lpf_alpha, args.lpf_release_alpha)
        ep["p_smooth"] = moving_average(ep["p_lpf"], args.smooth)
        peak_threshold = float(np.nanquantile(ep["p_smooth"], 0.92))
        min_peak_index = args.ignore_initial_frames
        if min_peak_index is None:
            min_peak_index = args.history * 2
        ep["peaks"] = peak_indices(
            ep["p_smooth"],
            args.top_k,
            args.history,
            peak_threshold,
            min_peak_index,
        )
        for idx in ep["peaks"]:
            summary_events.append(
                {
                    "episode_index": ep["episode_index"],
                    "frame_index": int(ep["frame_index"][idx]),
                    "timestamp": float(ep["timestamp"][idx]),
                    "p_smooth": float(ep["p_smooth"][idx]),
                    "error": float(ep["error"][idx]),
                    "eef_error": float(ep.get("eef_error", [np.nan])[idx]),
                    "hand_error": float(ep.get("hand_error", [np.nan])[idx]),
                }
            )

    summary_events.sort(key=lambda x: x["p_smooth"], reverse=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "event_transition_probe.csv", episodes)
    write_html(args.output_dir / "event_transition_probe.html", episodes, args.top_episodes_html)
    summary = {
        "dataset_dir": str(args.dataset_dir),
        "num_episodes": len(episodes),
        "num_frames": int(sum(len(ep["actions"]) for ep in episodes)),
        "history": args.history,
        "action_groups": args.action_groups,
        "smooth": args.smooth,
        "lpf_alpha": args.lpf_alpha,
        "lpf_release_alpha": args.lpf_release_alpha,
        "ignore_initial_frames": args.ignore_initial_frames
        if args.ignore_initial_frames is not None
        else args.history * 2,
        "calibration": calibration,
        "top_events": summary_events[:50],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
