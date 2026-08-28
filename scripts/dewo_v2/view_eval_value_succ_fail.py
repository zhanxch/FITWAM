#!/usr/bin/env python3
"""Interactive HTML viewer: per-replan V(t) for success vs fail DexJoCo eval episodes."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

BLUE = "#1f77b4"
RED = "#d62728"


@dataclass(frozen=True)
class EpisodeRow:
    path: Path
    run: int
    episode: int
    success: bool
    steps: np.ndarray
    values: np.ndarray
    value_rels: np.ndarray | None


_RUN_RE = re.compile(r"/run(\d+)/")
_EP_RE = re.compile(r"episode_(\d+)_(success|failure)_actions\.npz$")


def _load_episodes(out_root: Path) -> list[EpisodeRow]:
    rows: list[EpisodeRow] = []
    npzs = sorted(out_root.glob("run*/shard_*/**/*_actions.npz"))
    if not npzs:
        npzs = sorted(out_root.glob("**/*_actions.npz"))
    for path in npzs:
        payload = np.load(path, allow_pickle=True)
        if "cfg_values" not in payload.files:
            continue
        values = np.asarray(payload["cfg_values"], dtype=np.float64).reshape(-1)
        if values.size == 0 or not np.isfinite(values).any():
            continue
        if "policy_query_steps" in payload.files:
            steps = np.asarray(payload["policy_query_steps"], dtype=np.int32).reshape(-1)
        else:
            replan = int(payload["replan_steps"]) if "replan_steps" in payload.files else 24
            steps = np.arange(values.size, dtype=np.int32) * replan
        n = min(steps.size, values.size)
        steps = steps[:n]
        values = values[:n]
        order = np.argsort(steps)
        steps = steps[order]
        values = values[order]
        value_rels = None
        if "cfg_value_rels" in payload.files:
            rel = np.asarray(payload["cfg_value_rels"], dtype=np.float64).reshape(-1)[:n]
            value_rels = rel[order]

        m_run = _RUN_RE.search(path.as_posix())
        m_ep = _EP_RE.search(path.name)
        run = int(m_run.group(1)) if m_run else 0
        episode = int(m_ep.group(1)) if m_ep else -1
        success = m_ep.group(2) == "success" if m_ep else "success" in path.name
        rows.append(
            EpisodeRow(
                path=path,
                run=run,
                episode=episode,
                success=success,
                steps=steps,
                values=values,
                value_rels=value_rels,
            )
        )
    return rows


def _infer_replan(rows: list[EpisodeRow]) -> int:
    if not rows:
        return 24
    diffs = np.diff(np.unique(np.concatenate([r.steps for r in rows[:32]])))
    pos = diffs[diffs > 0]
    return int(np.min(pos)) if pos.size else 24


def _mean_std_at_steps(
    series: list[tuple[np.ndarray, np.ndarray]], step_grid: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    counts = np.zeros(step_grid.size, dtype=np.int32)
    sums = np.zeros(step_grid.size, dtype=np.float64)
    sq = np.zeros(step_grid.size, dtype=np.float64)
    for steps, values in series:
        for s, v in zip(steps, values, strict=False):
            if not np.isfinite(v):
                continue
            idx = int(np.searchsorted(step_grid, s))
            if idx >= step_grid.size or step_grid[idx] != s:
                continue
            counts[idx] += 1
            sums[idx] += v
            sq[idx] += v * v
    mean = np.full(step_grid.size, np.nan)
    std = np.full(step_grid.size, np.nan)
    ok = counts > 0
    mean[ok] = sums[ok] / counts[ok]
    var = np.zeros(step_grid.size, dtype=np.float64)
    var[ok] = np.maximum(sq[ok] / counts[ok] - mean[ok] ** 2, 0.0)
    std[ok] = np.sqrt(var[ok])
    return mean, std


def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _episode_label(row: EpisodeRow) -> str:
    tag = "ok" if row.success else "fail"
    return f"run{row.run} ep{row.episode:02d} ({tag})"


def _add_mean_band(
    fig: go.Figure,
    *,
    step_grid: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    color: str,
    label: str,
    row: int,
) -> None:
    fig.add_trace(
        go.Scatter(
            x=step_grid,
            y=mean,
            mode="lines+markers",
            name=label,
            line={"color": color, "width": 3},
            marker={"size": 6},
            hovertemplate="step=%{x}<br>mean=%{y:.4f}<extra></extra>",
        ),
        row=row,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([step_grid, step_grid[::-1]]),
            y=np.concatenate([mean - std, (mean + std)[::-1]]),
            fill="toself",
            fillcolor=_rgba(color, 0.18),
            line={"color": "rgba(255,255,255,0)"},
            name=f"{label} ±1σ",
            hoverinfo="skip",
            showlegend=False,
        ),
        row=row,
        col=1,
    )


def _build_figure(rows: list[EpisodeRow], *, max_frames: int, title: str) -> go.Figure:
    replan = _infer_replan(rows)
    step_grid = np.arange(0, max_frames + 1, replan, dtype=np.int32)
    runs = sorted({r.run for r in rows if r.run > 0})

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("V(t) per replan", "Relative growth (V_t - V_{t-1}) / |V_{t-1}|"),
        row_heights=[0.62, 0.38],
    )

    ep_trace_runs: list[int] = []
    rel_trace_runs: list[int] = []

    for row in rows:
        color = _rgba(BLUE if row.success else RED, 0.10 if row.success else 0.40)
        fig.add_trace(
            go.Scatter(
                x=row.steps,
                y=row.values,
                mode="lines",
                name=_episode_label(row),
                legendgroup="success" if row.success else "fail",
                line={"color": color, "width": 1.1 if row.success else 1.4},
                hovertemplate=f"{_episode_label(row)}<br>step=%{{x}}<br>V=%{{y:.4f}}<extra></extra>",
            ),
            row=1,
            col=1,
        )
        ep_trace_runs.append(row.run)

    for row in rows:
        if row.value_rels is None or row.value_rels.size <= 1:
            continue
        rel_trace_runs.append(row.run)
        color = _rgba(BLUE if row.success else RED, 0.08 if row.success else 0.28)
        fig.add_trace(
            go.Scatter(
                x=row.steps[1:],
                y=row.value_rels[1:],
                mode="lines",
                name=_episode_label(row),
                showlegend=False,
                line={"color": color, "width": 1},
                hovertemplate=f"{_episode_label(row)}<br>step=%{{x}}<br>rel=%{{y:.4f}}<extra></extra>",
            ),
            row=2,
            col=1,
        )

    succ = [r for r in rows if r.success]
    fail = [r for r in rows if not r.success]
    s_mean, s_std = _mean_std_at_steps([(r.steps, r.values) for r in succ], step_grid)
    f_mean, f_std = _mean_std_at_steps([(r.steps, r.values) for r in fail], step_grid)
    _add_mean_band(
        fig,
        step_grid=step_grid,
        mean=s_mean,
        std=s_std,
        color=BLUE,
        label=f"success mean (n={len(succ)})",
        row=1,
    )
    _add_mean_band(
        fig,
        step_grid=step_grid,
        mean=f_mean,
        std=f_std,
        color=RED,
        label=f"fail mean (n={len(fail)})",
        row=1,
    )

    def vis_for(run_filter: int | None) -> list[bool]:
        out = [run_filter is None or r == run_filter for r in ep_trace_runs]
        out.extend([True] * 4)
        out.extend(run_filter is None or r == run_filter for r in rel_trace_runs)
        return out

    buttons = [{"label": "All runs", "method": "restyle", "args": [{"visible": vis_for(None)}]}]
    for run in runs:
        buttons.append(
            {"label": f"Run {run}", "method": "restyle", "args": [{"visible": vis_for(run)}]}
        )

    fig.update_layout(
        title=title,
        height=900,
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.03,
            "xanchor": "left",
            "x": 0,
            "font": {"size": 9},
        },
        hovermode="closest",
        template="plotly_white",
        margin={"t": 100, "b": 50},
        updatemenus=[
            {
                "type": "dropdown",
                "direction": "down",
                "x": 1.0,
                "xanchor": "right",
                "y": 1.14,
                "yanchor": "top",
                "buttons": buttons,
            }
        ],
    )
    fig.update_xaxes(title_text="env step", range=[0, max_frames], row=2, col=1)
    fig.update_yaxes(title_text="V(t)", row=1, col=1)
    fig.update_yaxes(title_text="rel growth", row=2, col=1)
    for r in (1, 2):
        fig.add_vrect(x0=48, x1=96, fillcolor="gray", opacity=0.10, line_width=0, row=r, col=1)
    return fig


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=1000)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="HTML path (default: <out-root>/value_succ_fail_interactive.html)",
    )
    args = parser.parse_args()
    out_root = args.out_root.expanduser().resolve()
    rows = _load_episodes(out_root)
    if not rows:
        raise SystemExit(f"No episodes with cfg_values under {out_root}")

    n_succ = sum(1 for r in rows if r.success)
    n_fail = sum(1 for r in rows if not r.success)
    title = f"{out_root.name} | success={n_succ} fail={n_fail}"

    fig = _build_figure(rows, max_frames=args.max_frames, title=title)
    html_path = args.output or (out_root / "value_succ_fail_interactive.html")
    html_path.parent.mkdir(parents=True, exist_ok=True)

    meta = {
        "out_root": str(out_root),
        "n_success": n_succ,
        "n_fail": n_fail,
        "episodes": [
            {
                "run": r.run,
                "episode": r.episode,
                "success": r.success,
                "path": str(r.path),
                "steps": r.steps.tolist(),
                "values": [None if not np.isfinite(x) else float(x) for x in r.values],
                "value_rels": (
                    None
                    if r.value_rels is None
                    else [None if not np.isfinite(x) else float(x) for x in r.value_rels]
                ),
            }
            for r in rows
        ],
    }
    meta_path = html_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    fig.write_html(
        html_path,
        include_plotlyjs="cdn",
        config={"scrollZoom": True, "displayModeBar": True},
        div_id="value-viewer",
    )

    note = (
        "<div style='font-family:sans-serif;font-size:13px;margin:8px 16px 16px;color:#444'>"
        "<b>Usage:</b> blue=success, red=fail; thin=individual episodes; bold=mean±1σ. "
        "Run dropdown filters top panel. Legend: click to toggle, double-click to isolate. "
        f"Episode JSON: <code>{meta_path.name}</code></div>"
    )
    html = html_path.read_text(encoding="utf-8")
    html_path.write_text(html.replace("</body>", note + "\n</body>"), encoding="utf-8")

    print(f"wrote {html_path}")
    print(f"wrote {meta_path}")
    print(f"success={n_succ} fail={n_fail}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
