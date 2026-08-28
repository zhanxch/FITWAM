#!/usr/bin/env python3
"""Plot DEWO v9 recoverability event node distributions from pair_index / critic index."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _load_pairs(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "pairs" in payload:
        return list(payload["pairs"])
    if isinstance(payload, dict) and "full_horizon_pairs" in payload:
        return list(payload["full_horizon_pairs"])
    raise ValueError(f"Unrecognized pair index format: {path}")


def _rows_from_pairs(pairs: list[dict], *, replan: int) -> list[dict]:
    rows: list[dict] = []
    for p in pairs:
        t = int(p["t_star_last_recoverable"])
        m = int(p["M_first_zero"])
        cliff = p.get("fail_cliff")
        if cliff is None:
            cliff = [t, min(int(p.get("failure_length", t + 48)), m + 24)]
        lo, hi = int(cliff[0]), int(cliff[1])
        succ_len = int(
            p.get("success_length") or p.get("success_branch", {}).get("total_length") or 1
        )
        rows.append(
            {
                "pair_id": p.get("pair_id"),
                "seed": p.get("seed"),
                "t_star": t,
                "m_first_zero": m,
                "cliff_lo": lo,
                "cliff_hi": hi,
                "cliff_len": hi - lo,
                "replan_node": t // replan,
                "m_node": m // replan,
                "succ_len": succ_len,
                "t_norm_succ": t / max(succ_len, 1),
            }
        )
    return rows


def _summary(rows: list[dict], *, replan: int) -> dict:
    def arr(k: str) -> np.ndarray:
        return np.array([r[k] for r in rows], dtype=np.float64)

    t = arr("t_star")
    nodes = arr("replan_node").astype(int)
    u_nodes, c_nodes = np.unique(nodes, return_counts=True)
    cliff = arr("cliff_len").astype(int)
    u_cliff, c_cliff = np.unique(cliff, return_counts=True)
    return {
        "num_pairs": len(rows),
        "replan_steps": replan,
        "t_star": {
            "min": int(t.min()),
            "max": int(t.max()),
            "mean": float(t.mean()),
            "median": float(np.median(t)),
            "std": float(t.std()),
            "replan_node_counts": {str(int(k)): int(v) for k, v in zip(u_nodes, c_nodes, strict=True)},
        },
        "t_norm_success_length": {
            "min": float(arr("t_norm_succ").min()),
            "median": float(np.median(arr("t_norm_succ"))),
            "max": float(arr("t_norm_succ").max()),
        },
        "cliff_len_counts": {str(int(k)): int(v) for k, v in zip(u_cliff, c_cliff, strict=True)},
        "early_window_48_96": {
            "t_star_before_96": sum(1 for r in rows if r["t_star"] < 96),
            "t_star_in_48_96": sum(1 for r in rows if 48 <= r["t_star"] < 96),
            "t_star_after_96": sum(1 for r in rows if r["t_star"] >= 96),
        },
    }


def _plot(rows: list[dict], *, replan: int, title: str, out_html: Path) -> None:
    t = np.array([r["t_star"] for r in rows])
    nodes = np.array([r["replan_node"] for r in rows])
    norm = np.array([r["t_norm_succ"] for r in rows])

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "t* histogram (env step, bin=24)",
            "t* replan node",
            "t* / success episode length",
            "Pair timelines (blue=prefix [0,t*), red=fail cliff, tick=M)",
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )

    fig.add_trace(
        go.Histogram(
            x=t,
            xbins={"start": 0, "end": int(t.max() + replan), "size": replan},
            marker={"color": "#1f77b4"},
            name="t*",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Histogram(x=nodes, marker={"color": "#2ca02c"}, name="replan node"),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Histogram(x=norm, marker={"color": "#ff7f0e"}, name="t*/succ_len"),
        row=2,
        col=1,
    )

    sorted_rows = sorted(rows, key=lambda x: x["t_star"])
    for r in sorted_rows:
        ylabel = r["pair_id"] or f"seed{r['seed']}"
        fig.add_trace(
            go.Scatter(
                x=[0, r["t_star"]],
                y=[ylabel, ylabel],
                mode="lines",
                line={"color": "rgba(31,119,180,0.55)", "width": 10},
                showlegend=False,
                hovertemplate=f"{ylabel}<br>prefix [0,{r['t_star']})<extra></extra>",
            ),
            row=2,
            col=2,
        )
        fig.add_trace(
            go.Scatter(
                x=[r["cliff_lo"], r["cliff_hi"]],
                y=[ylabel, ylabel],
                mode="lines",
                line={"color": "rgba(214,39,40,0.85)", "width": 10},
                showlegend=False,
                hovertemplate=(
                    f"{ylabel}<br>fail cliff [{r['cliff_lo']},{r['cliff_hi']})<br>"
                    f"M={r['m_first_zero']}<extra></extra>"
                ),
            ),
            row=2,
            col=2,
        )

    for col in (1, 2):
        fig.add_vrect(x0=48, x1=96, fillcolor="gray", opacity=0.12, line_width=0, row=1, col=col)
    fig.add_vrect(x0=48, x1=96, fillcolor="gray", opacity=0.12, line_width=0, row=2, col=2)

    fig.update_layout(
        title=title,
        height=860,
        showlegend=False,
        template="plotly_white",
        margin={"t": 90},
    )
    fig.update_xaxes(title_text="env step", row=1, col=1)
    fig.update_xaxes(title_text="replan node (t*/24)", row=1, col=2)
    fig.update_xaxes(title_text="t* / success length", row=2, col=1)
    fig.update_xaxes(title_text="env step", row=2, col=2)

    note = (
        "<div style='font-family:sans-serif;font-size:13px;margin:8px 16px;color:#444'>"
        "<b>Semantics:</b> t* = last Pass@M recoverable frame; M = first zero-success frame "
        "(typically t*+24); red = D_fail cliff [t*, M+24). Gray band = eval early window 48–96 "
        "where value lacks succ/fail separation.</div>"
    )
    fig.write_html(
        out_html,
        include_plotlyjs="cdn",
        config={"scrollZoom": True},
        div_id="v9-event-nodes",
    )
    html = out_html.read_text(encoding="utf-8")
    out_html.write_text(html.replace("</body>", note + "\n</body>"), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pair-index",
        type=Path,
        required=True,
        help="pair_index.json or v9_critic_index.json",
    )
    parser.add_argument("--replan-steps", type=int, default=24)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--title", type=str, default=None)
    args = parser.parse_args()

    pair_index = args.pair_index.expanduser().resolve()
    pairs = _load_pairs(pair_index)
    rows = _rows_from_pairs(pairs, replan=int(args.replan_steps))
    if not rows:
        raise SystemExit(f"No pairs in {pair_index}")

    out_dir = args.output_dir or pair_index.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    title = args.title or f"v9 event nodes | {pair_index.parent.name} | n={len(rows)}"

    summary = _summary(rows, replan=int(args.replan_steps))
    json_path = out_dir / "v9_event_nodes.json"
    html_path = out_dir / "v9_event_nodes_interactive.html"
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    _plot(rows, replan=int(args.replan_steps), title=title, out_html=html_path)

    early = summary["early_window_48_96"]
    print(f"wrote {json_path}")
    print(f"wrote {html_path}")
    print(
        f"pairs={len(rows)} t* median={summary['t_star']['median']:.0f} "
        f"mean={summary['t_star']['mean']:.1f}"
    )
    print("replan nodes:", summary["t_star"]["replan_node_counts"])
    print(
        f"early window: t*<96={early['t_star_before_96']} "
        f"in[48,96)={early['t_star_in_48_96']} t*>=96={early['t_star_after_96']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
