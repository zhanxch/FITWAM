#!/usr/bin/env python3
"""Plot per-replan V(t) for success vs fail episodes in a DexJoCo eval OUT_ROOT."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _load_episodes(out_root: Path) -> tuple[list[np.ndarray], list[np.ndarray]]:
    succ: list[np.ndarray] = []
    fail: list[np.ndarray] = []
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
        if "success" in path.name:
            succ.append(values)
        elif "failure" in path.name or "fail" in path.name:
            fail.append(values)
    return succ, fail


def _stack_mean_std(series: list[np.ndarray], max_k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    counts = np.zeros(max_k, dtype=np.int32)
    sums = np.zeros(max_k, dtype=np.float64)
    sq = np.zeros(max_k, dtype=np.float64)
    for row in series:
        n = min(int(row.size), max_k)
        finite = np.isfinite(row[:n])
        idx = np.arange(n)[finite]
        vals = row[:n][finite]
        counts[idx] += 1
        sums[idx] += vals
        sq[idx] += vals * vals
    mean = np.full(max_k, np.nan)
    std = np.full(max_k, np.nan)
    ok = counts > 0
    mean[ok] = sums[ok] / counts[ok]
    var = np.zeros(max_k, dtype=np.float64)
    var[ok] = np.maximum(sq[ok] / counts[ok] - mean[ok] ** 2, 0.0)
    std[ok] = np.sqrt(var[ok])
    return mean, std, counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--replan-steps", type=int, default=24)
    parser.add_argument("--max-frames", type=int, default=180)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    out_root = args.out_root.expanduser().resolve()
    succ, fail = _load_episodes(out_root)
    max_k = max(1, int(np.ceil(args.max_frames / args.replan_steps)))
    t = np.arange(max_k) * int(args.replan_steps)
    s_mean, s_std, s_n = _stack_mean_std(succ, max_k)
    f_mean, f_std, f_n = _stack_mean_std(fail, max_k)

    summary = {
        "out_root": str(out_root),
        "n_success": len(succ),
        "n_fail": len(fail),
        "replan_steps": int(args.replan_steps),
        "max_frames": int(args.max_frames),
        "t": t.tolist(),
        "success_mean": [None if not np.isfinite(x) else float(x) for x in s_mean],
        "success_std": [None if not np.isfinite(x) else float(x) for x in s_std],
        "success_n": s_n.tolist(),
        "fail_mean": [None if not np.isfinite(x) else float(x) for x in f_mean],
        "fail_std": [None if not np.isfinite(x) else float(x) for x in f_std],
        "fail_n": f_n.tolist(),
    }
    json_path = args.output.with_suffix(".json") if args.output else out_root / "value_succ_fail.json"
    png_path = args.output if args.output else out_root / "value_succ_fail.png"
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.plot(t, s_mean, color="#1f77b4", label=f"success n={len(succ)}")
    ax.fill_between(t, s_mean - s_std, s_mean + s_std, color="#1f77b4", alpha=0.2)
    ax.plot(t, f_mean, color="#d62728", label=f"fail n={len(fail)}")
    ax.fill_between(t, f_mean - f_std, f_mean + f_std, color="#d62728", alpha=0.2)
    ax.axvspan(48, 96, color="#bbbbbb", alpha=0.25, label="48–96")
    ax.set_xlim(0, args.max_frames)
    ax.set_xlabel("env step (replan nodes)")
    ax.set_ylabel("V(t)")
    ax.set_title(out_root.name)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(png_path, dpi=140)
    plt.close(fig)

    print(f"wrote {png_path}")
    print(f"wrote {json_path}")
    print(f"success={len(succ)} fail={len(fail)}")
    print("t    V_succ±std (n)          V_fail±std (n)")
    for i, frame in enumerate(t):
        print(
            f"{int(frame):4d}  "
            f"{s_mean[i]:.4f}±{s_std[i]:.4f} ({int(s_n[i]):3d})   "
            f"{f_mean[i]:.4f}±{f_std[i]:.4f} ({int(f_n[i]):3d})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
