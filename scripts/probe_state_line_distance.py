#!/usr/bin/env python3
"""Probe event transitions from state a3-to-line(a1,a2) deviation.

This probe uses the full state vector described by meta/modality.json:

  observation.state[0:7]   = eef_pose
  observation.state[7:23]  = hand_joints

For each timestep t >= 2, every state dimension is scaled by its dataset std,
then the geometric distance from (2, x_t) to the line through
(0, x_{t-2}) and (1, x_{t-1}) is computed. The final event_transition_score is
the robustly normalized mean distance over all state dimensions.
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


STATE_COLUMN = "observation.state"


def load_state_span(dataset_dir: Path) -> tuple[int, int, dict[str, dict[str, int]]]:
    path = dataset_dir / "meta" / "modality.json"
    modality = json.loads(path.read_text(encoding="utf-8"))
    state = modality.get("state")
    if not isinstance(state, dict):
        raise KeyError(f"'state' not found in {path}")
    required = ["eef_pose", "hand_joints"]
    missing = [name for name in required if name not in state]
    if missing:
        raise KeyError(f"Missing required state modalities in {path}: {missing}")

    spans = {name: {"start": int(state[name]["start"]), "end": int(state[name]["end"])} for name in required}
    if spans["eef_pose"]["start"] != 0 or spans["eef_pose"]["end"] != spans["hand_joints"]["start"]:
        raise ValueError(f"Expected contiguous eef_pose -> hand_joints state layout in {path}")
    return spans["eef_pose"]["start"], spans["hand_joints"]["end"], spans


def read_episode(path: Path) -> dict[str, Any]:
    table = pq.ParquetFile(path).read(columns=[STATE_COLUMN, "timestamp", "frame_index", "episode_index"])
    values = np.asarray(table[STATE_COLUMN].combine_chunks().to_pylist(), dtype=np.float32)
    return {
        "path": path,
        "state": values,
        "timestamp": np.asarray(table["timestamp"].to_pylist(), dtype=np.float32),
        "frame_index": np.asarray(table["frame_index"].to_pylist(), dtype=np.int64),
        "episode_index": int(table["episode_index"][0].as_py()),
    }


def line_distances(values: np.ndarray, scale: np.ndarray) -> np.ndarray:
    normalized = values / scale
    n, dim = normalized.shape
    distances = np.full((n, dim), np.nan, dtype=np.float32)
    if n < 3:
        return distances

    a1 = normalized[:-2]
    a2 = normalized[1:-1]
    a3 = normalized[2:]
    slope = a2 - a1
    residual = (2.0 * a2) - a1 - a3
    distances[2:] = np.abs(residual) / np.sqrt((slope * slope) + 1.0)
    return distances


def mean_finite(values: np.ndarray, axis: int) -> np.ndarray:
    finite = np.isfinite(values)
    count = finite.sum(axis=axis)
    summed = np.where(finite, values, 0.0).sum(axis=axis)
    out = np.full(count.shape, np.nan, dtype=np.float32)
    np.divide(summed, count, out=out, where=count > 0)
    return out


def robust_score(values: np.ndarray, low_q: float, high_q: float) -> tuple[np.ndarray, dict[str, float]]:
    valid = values[np.isfinite(values)]
    if len(valid) == 0:
        raise ValueError("No finite values available for score calibration")
    low = float(np.quantile(valid, low_q))
    high = float(np.quantile(valid, high_q))
    if high <= low:
        high = low + 1e-6
    score = np.clip((values - low) / (high - low), 0.0, 1.0).astype(np.float32)
    score[~np.isfinite(values)] = np.nan
    return score, {"low": low, "high": high, "median": float(np.median(valid))}


def finite_or_blank(value: float) -> str:
    if not math.isfinite(float(value)):
        return ""
    return f"{float(value):.6f}"


def write_csv(path: Path, episodes: list[dict[str, Any]]) -> None:
    fieldnames = ["episode_index", "frame_index", "timestamp", "state_line_distance", "event_transition_score"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for ep in episodes:
            for i in range(len(ep["frame_index"])):
                writer.writerow(
                    {
                        "episode_index": ep["episode_index"],
                        "frame_index": int(ep["frame_index"][i]),
                        "timestamp": float(ep["timestamp"][i]),
                        "state_line_distance": finite_or_blank(ep["distance"][i]),
                        "event_transition_score": finite_or_blank(ep["score"][i]),
                    }
                )


def write_html(path: Path, episodes: list[dict[str, Any]], top_episodes: int) -> None:
    traces = [
        {
            "x": ep["frame_index"].tolist(),
            "y": ep["score"].tolist(),
            "mode": "lines",
            "name": f"ep {ep['episode_index']:03d}",
        }
        for ep in episodes[:top_episodes]
    ]
    payload = json.dumps(traces)
    path.write_text(
        f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>State line-distance event transition probe</title>
  <script src="https://cdn.plot.ly/plotly-2.30.0.min.js"></script>
</head>
<body>
  <div id="plot" style="width:100%;height:760px;"></div>
  <script>
    const traces = {payload};
    Plotly.newPlot('plot', traces, {{
      title: 'Event transition score from state a3-to-line(a1,a2) distance',
      xaxis: {{title: 'frame'}},
      yaxis: {{title: 'event_transition_score', range: [0, 1]}},
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
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--low-quantile", type=float, default=0.10)
    parser.add_argument("--high-quantile", type=float, default=0.95)
    parser.add_argument("--top-episodes-html", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    state_start, state_end, state_modalities = load_state_span(args.dataset_dir)
    parquet_paths = sorted((args.dataset_dir / "data").glob("chunk-*/*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"No parquet episodes found under {args.dataset_dir / 'data'}")

    episodes = [read_episode(path) for path in parquet_paths]
    all_state = np.concatenate([ep["state"] for ep in episodes], axis=0)
    scale = all_state.std(axis=0).astype(np.float32)
    scale[scale < 1e-6] = 1.0

    all_distances = []
    for ep in episodes:
        per_dim_distance = line_distances(ep["state"], scale)
        distance = mean_finite(per_dim_distance[:, state_start:state_end], axis=1)
        ep["distance"] = distance.astype(np.float32)
        all_distances.append(ep["distance"])

    score_all, calibration = robust_score(
        np.concatenate(all_distances),
        args.low_quantile,
        args.high_quantile,
    )
    cursor = 0
    for ep in episodes:
        n = len(ep["state"])
        ep["score"] = score_all[cursor : cursor + n]
        cursor += n

    top_events = []
    for ep in episodes:
        score = ep["score"]
        for idx in np.argsort(np.nan_to_num(score, nan=-1.0))[-10:][::-1]:
            if not np.isfinite(score[idx]):
                continue
            top_events.append(
                {
                    "episode_index": ep["episode_index"],
                    "frame_index": int(ep["frame_index"][idx]),
                    "timestamp": float(ep["timestamp"][idx]),
                    "state_line_distance": float(ep["distance"][idx]),
                    "event_transition_score": float(score[idx]),
                }
            )
    top_events.sort(key=lambda row: row["event_transition_score"], reverse=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "state_line_distance_probe.csv", episodes)
    write_html(args.output_dir / "state_line_distance_probe.html", episodes, args.top_episodes_html)

    summary = {
        "dataset_dir": str(args.dataset_dir),
        "column": STATE_COLUMN,
        "state_modalities": state_modalities,
        "state_span": {"start": state_start, "end": state_end},
        "num_episodes": len(episodes),
        "num_frames": int(sum(len(ep["state"]) for ep in episodes)),
        "distance": "mean per-dimension geometric distance from (2,a3) to line through (0,a1),(1,a2), after per-dimension std scaling",
        "score": {
            "type": "robust quantile normalization clipped to [0,1]",
            "low_quantile": args.low_quantile,
            "high_quantile": args.high_quantile,
            "calibration": calibration,
        },
        "top_events": top_events[:50],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
