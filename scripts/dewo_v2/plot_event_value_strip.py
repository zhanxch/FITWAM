#!/usr/bin/env python3
"""Long strip: official 4x50 V(t) aligned with recoverability event t*/M.

V curves come from an official 4x50 本体 eval (`cfg_values` in *_actions.npz).
Event times come from pair_index + oracle-once `results.jsonl`.
These are different seed sets: the figure shows where t* sits on the critic
time profile, not V of the collect-failure episode itself (oracle jsonl only
stores `cfg_value_last`).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


TAG_COLOR = {
    "oracle_only": "#2ca02c",
    "both": "#7f7f7f",
    "base_only": "#ff7f0e",
    "neither": "#d62728",
}


def _load_benti(out_root: Path, max_k: int) -> tuple[np.ndarray, np.ndarray]:
    rows: list[tuple[bool, np.ndarray]] = []
    npzs = sorted(out_root.glob("run*/shard_*/**/*_actions.npz"))
    for path in npzs:
        z = np.load(path, allow_pickle=True)
        if "cfg_values" not in z.files:
            continue
        v = np.asarray(z["cfg_values"], dtype=np.float64).reshape(-1)
        rows.append(("success" in path.name, v))
    mat = np.full((len(rows), max_k), np.nan)
    ok_mask = np.zeros(len(rows), dtype=bool)
    for i, (ok, v) in enumerate(rows):
        n = min(int(v.size), max_k)
        mat[i, :n] = v[:n]
        ok_mask[i] = ok
    return mat, ok_mask


def _load_oracle(oracle_root: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for path in sorted(oracle_root.glob("shard_*/results.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            pid = str(row["pair_id"])
            rec = out.setdefault(
                pid,
                {
                    "pair_id": pid,
                    "seed": row.get("seed"),
                    "t_star": int(row["t_star"]),
                    "m_first_zero": int(row["m_first_zero"]),
                },
            )
            rec[str(row["condition"])] = {
                "pass": bool(row.get("pass_at_m_hit")),
                "success_count": int(row.get("success_count") or 0),
            }
    return out


def _load_pair_index(path: Path) -> dict[str, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    pairs = payload["pairs"] if isinstance(payload, dict) and "pairs" in payload else payload
    return {str(p["pair_id"]): p for p in pairs}


def _oracle_tag(rec: dict) -> str:
    base = rec.get("v9_base") or {}
    ora = rec.get("v9_oracle_once") or {}
    b, o = bool(base.get("pass")), bool(ora.get("pass"))
    if o and not b:
        return "oracle_only"
    if o and b:
        return "both"
    if b and not o:
        return "base_only"
    return "neither"


def _mean_std(mat: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sl = mat[mask]
    with np.errstate(all="ignore"):
        mean = np.nanmean(sl, axis=0)
        std = np.nanstd(sl, axis=0)
    return mean, std


def _at(arr: np.ndarray, t_star: int, replan: int) -> float:
    k = int(np.clip(t_star // replan, 0, arr.size - 1))
    val = float(arr[k])
    return val if np.isfinite(val) else float("nan")


def _short_id(pair_id: str) -> str:
    return pair_id.replace("_frontier_00", "")


def _collect_events(oracle_root: Path, pair_index: Path, replan: int) -> list[dict]:
    oracle = _load_oracle(oracle_root)
    pairs = _load_pair_index(pair_index)
    events = []
    for pid, rec in oracle.items():
        src = pairs.get(pid, {})
        cliff = src.get("fail_cliff") or [rec["t_star"], rec["m_first_zero"] + replan]
        events.append(
            {
                **rec,
                "cliff_lo": int(cliff[0]),
                "cliff_hi": int(cliff[1]),
                "tag": _oracle_tag(rec),
            }
        )
    events.sort(key=lambda r: (r["t_star"], r["pair_id"]))
    return events


def _plot_overview(
    *,
    mat: np.ndarray,
    ok_mask: np.ndarray,
    events: list[dict],
    t: np.ndarray,
    replan: int,
    max_frames: int,
    benti_name: str,
    oracle_name: str,
    out: Path,
) -> None:
    s_mean, s_std = _mean_std(mat, ok_mask)
    f_mean, f_std = _mean_std(mat, ~ok_mask)
    n_fail = int((~ok_mask).sum())
    n_succ = int(ok_mask.sum())
    order = np.argsort(~ok_mask, kind="stable")
    heat = mat[order]
    vmax = float(np.nanpercentile(heat, 92)) if np.isfinite(heat).any() else 0.05
    vmax = max(vmax, 1e-4)
    t_stars = np.array([e["t_star"] for e in events], dtype=float)
    m_vals = np.array([e["m_first_zero"] for e in events], dtype=float)
    n_event = len(events)

    fig_h = 3.8 + 0.028 * heat.shape[0] + 0.28 * n_event
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(14.0, fig_h),
        gridspec_kw={"height_ratios": [3.2, 2.6, max(4.0, 0.28 * n_event)]},
        sharex=True,
    )

    ax = axes[0]
    im = ax.imshow(
        heat,
        aspect="auto",
        interpolation="nearest",
        cmap="magma",
        norm=Normalize(vmin=0.0, vmax=vmax),
        extent=(-replan / 2, t[-1] + replan / 2, heat.shape[0] - 0.5, -0.5),
    )
    ax.axhline(n_fail - 0.5, color="white", lw=0.8, ls="--")
    ax.axvline(float(np.median(t_stars)), color="#2ca02c", ls="--", lw=1.0, alpha=0.85)
    ax.set_ylabel(f"episode  (fail n={n_fail} top, succ n={n_succ} bottom)")
    ax.set_title(
        "Official 4x50 V(t)  vs  collect-event t*/M\n"
        f"V: {benti_name}    events: {oracle_name}    (different seeds)",
        fontsize=11,
    )
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01, label="V")

    ax = axes[1]
    ax.plot(t, s_mean, color="#1f77b4", lw=1.6, label=f"4x50 success n={n_succ}")
    ax.fill_between(t, s_mean - s_std, s_mean + s_std, color="#1f77b4", alpha=0.16)
    ax.plot(t, f_mean, color="#d62728", lw=1.6, label=f"4x50 fail n={n_fail}")
    ax.fill_between(t, f_mean - f_std, f_mean + f_std, color="#d62728", alpha=0.16)
    y_lo = float(np.nanmin(np.concatenate([s_mean - s_std, f_mean - f_std])))
    y_hi = float(np.nanmax(np.concatenate([s_mean + s_std, f_mean + f_std])))
    span = max(y_hi - y_lo, 1e-3)
    ax.set_ylim(y_lo - 0.18 * span, y_hi + 0.10 * span)
    for ev in events:
        ax.plot(
            ev["t_star"],
            _at(f_mean, ev["t_star"], replan),
            marker="v",
            color=TAG_COLOR[ev["tag"]],
            markersize=6.5,
            zorder=6,
        )
        ax.plot(
            ev["t_star"],
            _at(s_mean, ev["t_star"], replan),
            marker="^",
            color=TAG_COLOR[ev["tag"]],
            markersize=5.0,
            alpha=0.85,
            zorder=6,
        )
    ax.scatter(
        m_vals,
        np.full_like(m_vals, y_lo - 0.10 * span),
        marker="|",
        s=90,
        c="#6b2d2d",
        linewidths=1.0,
        label="M first-zero",
        zorder=5,
    )
    ax.axvline(float(np.median(t_stars)), color="#2ca02c", ls="--", lw=1.0, alpha=0.85)
    ax.set_ylabel("V")
    ax.legend(loc="upper left", fontsize=8, ncol=2, framealpha=0.92)
    ax.grid(True, alpha=0.25)
    ax.set_title(
        "mean V; triangles = t* on fail-mean (v) / succ-mean (^), color = oracle outcome",
        fontsize=10,
    )

    ax = axes[2]
    for i, ev in enumerate(events):
        y = n_event - 1 - i
        ax.plot([0, ev["t_star"]], [y, y], color="#4c78a8", lw=7, solid_capstyle="butt", alpha=0.55)
        ax.plot(
            [ev["cliff_lo"], ev["cliff_hi"]],
            [y, y],
            color="#d62728",
            lw=7,
            solid_capstyle="butt",
            alpha=0.85,
        )
        ax.plot(ev["t_star"], y, marker="v", color=TAG_COLOR[ev["tag"]], markersize=7, zorder=6)
        ax.plot(
            ev["m_first_zero"],
            y,
            marker="|",
            color="#111111",
            markersize=9,
            markeredgewidth=1.6,
            zorder=6,
        )
    ax.set_yticks(range(n_event))
    ax.set_yticklabels(
        [f"{_short_id(ev['pair_id'])}  {ev['tag']}" for ev in reversed(events)],
        fontsize=6.5,
        family="monospace",
    )
    ax.set_xlabel("env step")
    ax.set_ylabel("event (sorted by t*)")
    ax.set_xlim(0, int(max_frames))
    ax.set_ylim(-1.0, n_event)
    ax.grid(True, axis="x", alpha=0.25)
    ax.set_title(
        "event strip: blue=[0,t*)  red=fail cliff  v=t* CFG-once  |=M",
        fontsize=9,
    )
    legend_el = [
        Patch(facecolor="#4c78a8", alpha=0.55, label="prefix [0, t*)"),
        Patch(facecolor="#d62728", alpha=0.85, label="fail cliff"),
        Line2D([0], [0], marker="v", color="#2ca02c", ls="", label="oracle_only t*"),
        Line2D([0], [0], marker="v", color="#7f7f7f", ls="", label="both pass t*"),
        Line2D([0], [0], marker="v", color="#ff7f0e", ls="", label="base_only t*"),
        Line2D([0], [0], marker="v", color="#d62728", ls="", label="neither t*"),
    ]
    ax.legend(handles=legend_el, loc="upper right", fontsize=7, ncol=3, framealpha=0.92)

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def _window_slice(t: np.ndarray, x0: int, x1: int) -> slice:
    i0 = max(0, int(np.searchsorted(t, x0, side="left")))
    i1 = min(int(t.size), int(np.searchsorted(t, x1, side="right")))
    return slice(i0, max(i0 + 1, i1))


def _local_ylim(curves: list[np.ndarray], sl: slice, *, pad_frac: float = 0.12) -> tuple[float, float]:
    chunks = [c[sl] for c in curves if c.size > sl.start]
    if not chunks:
        return 0.0, 0.05
    vals = np.concatenate(chunks)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0, 0.05
    lo = float(np.percentile(vals, 2))
    hi = float(np.percentile(vals, 98))
    if hi <= lo:
        hi = lo + max(abs(lo) * 0.2, 1e-4)
    span = hi - lo
    return lo - pad_frac * span, hi + pad_frac * span


def _plot_local_curves(
    *,
    mat: np.ndarray,
    ok_mask: np.ndarray,
    events: list[dict],
    t: np.ndarray,
    replan: int,
    out: Path,
    x_margin_before: int = 96,
    x_margin_after: int = 72,
) -> None:
    """One row per event: raw per-episode V(t), local x/y zoom around t*/M."""
    succ_curves = [mat[i] for i in range(mat.shape[0]) if ok_mask[i]]
    fail_curves = [mat[i] for i in range(mat.shape[0]) if not ok_mask[i]]
    n_event = len(events)

    fig_h = 0.78 * n_event + 1.2
    fig, axes = plt.subplots(n_event, 1, figsize=(14.0, fig_h), sharex=False, sharey=False)
    if n_event == 1:
        axes = [axes]

    for i, ev in enumerate(events):
        ax = axes[i]
        x0 = max(0, int(ev["t_star"]) - int(x_margin_before))
        x1 = min(int(t[-1]), int(ev["cliff_hi"]) + int(x_margin_after))
        sl = _window_slice(t, x0, x1)
        tw = t[sl]

        for row in fail_curves:
            ax.plot(tw, row[sl], color="#d62728", lw=0.55, alpha=0.22, zorder=1)
        for row in succ_curves:
            ax.plot(tw, row[sl], color="#1f77b4", lw=0.55, alpha=0.14, zorder=1)

        f_med = np.full(tw.size, np.nan)
        s_med = np.full(tw.size, np.nan)
        if fail_curves:
            f_med = np.nanmedian(np.stack([r[sl] for r in fail_curves], axis=0), axis=0)
            ax.plot(tw, f_med, color="#d62728", lw=1.5, alpha=0.95, zorder=3)
        if succ_curves:
            s_med = np.nanmedian(np.stack([r[sl] for r in succ_curves], axis=0), axis=0)
            ax.plot(tw, s_med, color="#1f77b4", lw=1.5, alpha=0.95, zorder=3)

        ax.axvspan(ev["cliff_lo"], ev["cliff_hi"], color="#d62728", alpha=0.10, zorder=0)
        ax.axvline(ev["t_star"], color=TAG_COLOR[ev["tag"]], lw=1.8, zorder=5)
        ax.axvline(ev["m_first_zero"], color="#111111", lw=1.0, ls=":", zorder=5)

        vf = float(f_med[int(np.argmin(np.abs(tw - ev["t_star"])))]) if fail_curves else float("nan")
        vs = float(s_med[int(np.argmin(np.abs(tw - ev["t_star"])))]) if succ_curves else float("nan")
        ax.plot(ev["t_star"], vf, marker="v", color=TAG_COLOR[ev["tag"]], markersize=7, zorder=6)
        ax.plot(ev["t_star"], vs, marker="^", color=TAG_COLOR[ev["tag"]], markersize=6, zorder=6)

        ax.set_xlim(x0, x1)
        ax.set_ylim(*_local_ylim(fail_curves + succ_curves, sl))
        ax.tick_params(axis="y", labelsize=6)
        label = (
            f"{_short_id(ev['pair_id'])}  {ev['tag']}\n"
            f"t*={ev['t_star']}  M={ev['m_first_zero']}  "
            f"Vf={vf:.4f}  Vs={vs:.4f}"
        )
        ax.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=7, family="monospace")
        ax.grid(True, axis="both", alpha=0.22)
        if i == 0:
            ax.set_title(
                "Per-event V (raw 4x50 episodes, local zoom): thin=individual, bold=median. "
                "Solid=t*  dotted=M  red band=fail cliff.",
                fontsize=10,
            )
        if i == n_event - 1:
            ax.set_xlabel("env step")
            ax.legend(
                handles=[
                    Line2D([0], [0], color="#1f77b4", lw=1.5, label="succ median"),
                    Line2D([0], [0], color="#d62728", lw=1.5, label="fail median"),
                    Line2D([0], [0], color="#1f77b4", lw=0.6, alpha=0.3, label="succ raw"),
                    Line2D([0], [0], color="#d62728", lw=0.6, alpha=0.4, label="fail raw"),
                    Line2D([0], [0], color="#2ca02c", label="t* oracle_only"),
                    Line2D([0], [0], color="#7f7f7f", label="t* both"),
                    Line2D([0], [0], color="#ff7f0e", label="t* base_only"),
                    Line2D([0], [0], color="#d62728", ls=":", label="M"),
                ],
                loc="upper right",
                fontsize=7,
                ncol=4,
                framealpha=0.92,
            )

    fig.tight_layout(h_pad=0.22)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def _plot_html(
    *,
    mat: np.ndarray,
    ok_mask: np.ndarray,
    events: list[dict],
    t: np.ndarray,
    replan: int,
    out: Path,
) -> None:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    s_mean, s_std = _mean_std(mat, ok_mask)
    f_mean, f_std = _mean_std(mat, ~ok_mask)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.45, 0.55], vertical_spacing=0.06)
    fig.add_trace(
        go.Scatter(x=t, y=s_mean, name="4x50 success mean V", line=dict(color="#1f77b4", width=2)),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=t, y=f_mean, name="4x50 fail mean V", line=dict(color="#d62728", width=2)),
        row=1,
        col=1,
    )
    for ev in events:
        fig.add_trace(
            go.Scatter(
                x=[ev["t_star"]],
                y=[_at(f_mean, ev["t_star"], replan)],
                mode="markers",
                marker=dict(symbol="triangle-down", size=10, color=TAG_COLOR[ev["tag"]]),
                name=f"{_short_id(ev['pair_id'])} {ev['tag']}",
                hovertemplate=(
                    f"{ev['pair_id']}<br>tag={ev['tag']}<br>t*={ev['t_star']}"
                    f"<br>M={ev['m_first_zero']}<br>Vfail=%{{y:.4f}}<extra></extra>"
                ),
                showlegend=False,
            ),
            row=1,
            col=1,
        )
    for i, ev in enumerate(events):
        y = len(events) - 1 - i
        fig.add_trace(
            go.Scatter(
                x=[0, ev["t_star"]],
                y=[y, y],
                mode="lines",
                line=dict(color="#4c78a8", width=10),
                hoverinfo="skip",
                showlegend=False,
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=[ev["cliff_lo"], ev["cliff_hi"]],
                y=[y, y],
                mode="lines",
                line=dict(color="#d62728", width=10),
                hoverinfo="skip",
                showlegend=False,
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=[ev["t_star"]],
                y=[y],
                mode="markers+text",
                marker=dict(symbol="triangle-down", size=12, color=TAG_COLOR[ev["tag"]]),
                text=[ev["tag"]],
                textposition="middle right",
                name=_short_id(ev["pair_id"]),
                hovertemplate=(
                    f"{ev['pair_id']}<br>t*={ev['t_star']} M={ev['m_first_zero']}"
                    f"<br>{ev['tag']}<extra></extra>"
                ),
                showlegend=False,
            ),
            row=2,
            col=1,
        )
    fig.update_yaxes(title_text="V", row=1, col=1)
    fig.update_yaxes(
        title_text="event (sorted by t*)",
        tickmode="array",
        tickvals=list(range(len(events))),
        ticktext=[_short_id(ev["pair_id"]) for ev in reversed(events)],
        row=2,
        col=1,
    )
    fig.update_xaxes(title_text="env step", row=2, col=1)
    fig.update_layout(
        height=220 + 18 * len(events),
        width=1100,
        title="Event t* on official 4x50 V(t)  |  v = CFG-once time, red = fail cliff",
        template="plotly_white",
        margin=dict(l=160, r=30, t=60, b=40),
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out, include_plotlyjs="cdn")
    print(f"wrote {out}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benti-root", type=Path, required=True)
    parser.add_argument("--oracle-root", type=Path, required=True)
    parser.add_argument(
        "--pair-index",
        type=Path,
        default=Path("/data_all/xiangchengzhan/FastWAM/data/fold_glasses_dewo_v9_pair_full_lerobot/pair_index.json"),
    )
    parser.add_argument("--replan-steps", type=int, default=24)
    parser.add_argument("--max-frames", type=int, default=504)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    replan = int(args.replan_steps)
    max_k = max(1, int(np.ceil(args.max_frames / replan)))
    t = np.arange(max_k) * replan
    mat, ok_mask = _load_benti(args.benti_root, max_k)
    events = _collect_events(args.oracle_root, args.pair_index, replan)
    out_dir = args.output_dir or args.oracle_root

    _plot_overview(
        mat=mat,
        ok_mask=ok_mask,
        events=events,
        t=t,
        replan=replan,
        max_frames=int(args.max_frames),
        benti_name=args.benti_root.name,
        oracle_name=args.oracle_root.name,
        out=out_dir / "event_value_strip.png",
    )
    _plot_local_curves(
        mat=mat,
        ok_mask=ok_mask,
        events=events,
        t=t,
        replan=replan,
        out=out_dir / "event_value_curves_long.png",
    )
    try:
        _plot_html(
            mat=mat,
            ok_mask=ok_mask,
            events=events,
            t=t,
            replan=replan,
            out=out_dir / "event_value_strip.html",
        )
    except ImportError as exc:
        print(f"skip html ({exc})")
    t_stars = np.array([e["t_star"] for e in events], dtype=float)
    print(
        "benti episodes={} fail={} events={} t* median={:.0f} "
        "oracle_only={} both={} base_only={} neither={}".format(
            mat.shape[0],
            int((~ok_mask).sum()),
            len(events),
            float(np.median(t_stars)),
            sum(e["tag"] == "oracle_only" for e in events),
            sum(e["tag"] == "both" for e in events),
            sum(e["tag"] == "base_only" for e in events),
            sum(e["tag"] == "neither" for e in events),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
