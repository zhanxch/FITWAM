#!/usr/bin/env python3
"""A3 + C2: extract open-loop action metrics and loss curves for sim vs real
FastWAM runs.

Primary data source: wandb-summary.json (final reported metrics per run).
Secondary: wandb .wandb datastore history (when parseable).
Fallback: output.log parsing (when wandb files are incomplete).

Per the plan (A3):
  - If both runs have similar action L1 but real rollout fails -> deploy/闭环 (H5)
  - If real action L1 >> sim action L1 -> model/data issue (H1/H2/H3)

Per the plan (C2):
  - Compare loss_action / loss_video convergence between sim and real.

NOTE on the action_l1 caveat (important for interpretation):
  eval/action_l1 is computed on NORMALIZED actions (see trainer.py:520-522, the
  L1 is taken on denormalized actions but the denorm uses the SAME min/max stats
  for pred and gt). Because both pred and gt pass through the same per-dim scale,
  a systematic warp of the rot6d manifold (H2) affects pred and gt identically,
  so action_l1 may look fine even though the model is learning a warped target.
  The real failure shows up in CLOSED-LOOP rollout, not open-loop L1. This is
  exactly the "action L1 comparable but rollout fails" scenario -> points to H2
  (the warp makes the learned manifold fragile to sampling noise) and H1 (no
  proprio makes closed-loop drift unrecoverable).

Usage:
    python scripts/diagnose/extract_wandb_metrics.py
    python scripts/diagnose/extract_wandb_metrics.py --out scripts/diagnose/metrics_output.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

RUNS = {
    "spray_water_real": {
        "dirs": [
            "runs/spray_water_rot6d_rosbag_ts_filter_uncond_3cam_384_1e-4/2026-06-22_00-40-40",
            "runs/spray_water_rot6d_rosbag_ts_filter_uncond_3cam_384_1e-4/2026-06-20_00-02-34",
            "runs/spray_water_rot6d_rosbag_ts_filter_uncond_3cam_384_1e-4/2026-06-17_16-31-58",
        ],
        "label": "spray_water (real, rot6d, 58-dim)",
        "action_dim": 58,
        "rotation": "rot6d",
    },
    "dexjoco_sim": {
        "dirs": [
            "runs/dexjoco_microwave_cook_uncond_3cam_384_1e-4/2026-06-15_15-21-45",
            "runs/dexjoco_microwave_cook_uncond_3cam_384_1e-4/2026-06-09_16-54-35",
        ],
        "label": "dexjoco_microwave_cook (sim, rotvec, 44-dim)",
        "action_dim": 44,
        "rotation": "rotvec",
    },
}

METRICS = (
    "eval/action_l1",
    "eval/action_l2",
    "train/loss_action",
    "train/loss_video",
    "train/loss",
    "eval/val_loss",
    "eval/psnr_rd",
    "eval/ssim_rd",
    "eval/psnr_rg",
    "eval/ssim_rg",
)


def load_wandb_summary(run_dir: str) -> dict | None:
    """Read wandb-summary.json from any wandb run subdir."""
    base = Path(run_dir) / "wandb"
    if not base.exists():
        return None
    for p in base.rglob("wandb-summary.json"):
        try:
            text = p.read_text()
            if not text.strip():
                continue
            d = json.loads(text)
            return {"path": str(p), "summary": {k: v for k, v in d.items() if isinstance(v, (int, float))}}
        except (json.JSONDecodeError, OSError):
            continue
    return None


def parse_output_log(run_dir: str) -> dict:
    """Parse trainer.py [train]/[eval] log lines from output.log to recover
    the loss history. Handles ANSI escape codes."""
    base = Path(run_dir) / "wandb"
    if not base.exists():
        return {"history": {}}
    ansi_re = re.compile(r"\x1b\[[0-9;]*m")
    history: dict[str, list[tuple[int, float]]] = {}
    for log in base.rglob("output.log"):
        try:
            text = log.read_text(errors="ignore")
        except OSError:
            continue
        for line in text.splitlines():
            clean = ansi_re.sub("", line)
            # trainer.py prints e.g. "[train] epoch=0 step=10/... loss=0.1234 loss_action=0.01 loss_video=0.1"
            m_step = re.search(r"step=(\d+)", clean)
            if not m_step:
                continue
            step = int(m_step.group(1))
            for key in ("loss", "loss_action", "loss_video"):
                m = re.search(rf"{key}=([0-9.eE+-]+)", clean)
                if m:
                    full_key = f"train/{key}" if "loss" in key and "val" not in key else key
                    history.setdefault(full_key, []).append((step, float(m.group(1))))
            # [eval] step=N val_loss=... infer_psnr=... infer_ssim=... action_l2=... action_l1=...
            for key in ("val_loss", "action_l1", "action_l2", "infer_psnr", "infer_ssim"):
                m = re.search(rf"{key}=([0-9.eE+-]+)", clean)
                if m:
                    mapped = {
                        "val_loss": "eval/val_loss",
                        "action_l1": "eval/action_l1",
                        "action_l2": "eval/action_l2",
                        "infer_psnr": "eval/psnr_rd",
                        "infer_ssim": "eval/ssim_rd",
                    }[key]
                    history.setdefault(mapped, []).append((step, float(m.group(1))))
    return {"history": history}


def summarize_history(history: dict[str, list[tuple[int, float]]]) -> dict:
    out = {}
    for key, series in history.items():
        if not series:
            continue
        vals = [v for _, v in series]
        out[key] = {
            "n_points": len(series),
            "first_step": series[0][0],
            "first_value": series[0][1],
            "last_step": series[-1][0],
            "last_value": series[-1][1],
            "min_value": min(vals),
            "max_value": max(vals),
        }
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    results: dict[str, dict] = {}
    for run_key, info in RUNS.items():
        print(f"\n{'=' * 78}\n{info['label']}  ({run_key})\n{'=' * 78}")
        summaries = []
        for d in info["dirs"]:
            s = load_wandb_summary(d)
            if s:
                summaries.append((d, s))
                print(f"  summary found: {d}")
        # parse output.log history from all dirs
        histories = []
        for d in info["dirs"]:
            h = parse_output_log(d)
            if h["history"]:
                histories.append((d, h["history"]))
                n = sum(len(v) for v in h["history"].values())
                print(f"  output.log history: {d} ({n} points)")

        # Pick the latest run by directory name (date) for the representative.
        summaries.sort(key=lambda x: x[0], reverse=True)
        histories.sort(key=lambda x: x[0], reverse=True)

        rep_summary = summaries[0][1]["summary"] if summaries else {}
        rep_summary_path = summaries[0][1]["path"] if summaries else None
        rep_history = histories[0][1] if histories else {}
        conv = summarize_history(rep_history)

        print(f"\n  --- Final wandb-summary metrics ---")
        for k in METRICS:
            if k in rep_summary and rep_summary[k] is not None:
                print(f"    {k:22s} = {rep_summary[k]:.6f}")
        if not rep_summary:
            print("    (no wandb-summary.json found)")

        print(f"\n  --- Training loss convergence (from output.log) ---")
        for k in ("train/loss", "train/loss_action", "train/loss_video"):
            if k in conv:
                c = conv[k]
                print(f"    {k:22s}: {c['first_value']:.6f} (step {c['first_step']}) "
                      f"-> {c['last_value']:.6f} (step {c['last_step']})  [{c['n_points']} pts]")
        if not conv:
            print("    (no output.log history recovered)")

        results[run_key] = {
            "label": info["label"],
            "action_dim": info["action_dim"],
            "rotation": info["rotation"],
            "summary": rep_summary,
            "summary_path": rep_summary_path,
            "convergence": conv,
            "history": rep_history if args.out else None,
        }

    # Cross-run comparison
    print(f"\n{'=' * 78}\nCROSS-RUN COMPARISON (sim vs real)\n{'=' * 78}")
    sim = results.get("dexjoco_sim", {})
    real = results.get("spray_water_real", {})
    sim_s = sim.get("summary", {})
    real_s = real.get("summary", {})

    print(f"\n  {'metric':22s} {'sim (rotvec)':>16s} {'real (rot6d)':>16s} {'real/sim':>10s}")
    print(f"  {'-'*22} {'-'*16} {'-'*16} {'-'*10}")
    comparison = []
    for k in ("eval/action_l1", "eval/action_l2", "train/loss_action", "train/loss_video", "eval/psnr_rd", "eval/ssim_rd", "eval/val_loss"):
        sv = sim_s.get(k)
        rv = real_s.get(k)
        if sv is not None and rv is not None:
            ratio = rv / sv if abs(sv) > 1e-12 else float("inf")
            print(f"  {k:22s} {sv:>16.6f} {rv:>16.6f} {ratio:>9.3f}x")
            comparison.append((k, sv, rv, ratio))
        else:
            print(f"  {k:22s} {str(sv):>16s} {str(rv):>16s} {'N/A':>10s}")

    # Verdict
    print(f"\n{'-' * 78}\nVERDICT (A3: open-loop action quality sim vs real):")
    sim_l1 = sim_s.get("eval/action_l1")
    real_l1 = real_s.get("eval/action_l1")
    if sim_l1 is None:
        print(f"  sim (dexjoco) eval/action_l1 NOT available locally (wandb run incomplete).")
        print(f"  -> To complete A3, run a fresh open-loop eval on the sim checkpoint:")
        print(f"     Use the trainer eval path or run the sim eval server + open-loop client")
        print(f"     and compare eval/action_l1 to the real value below.")
    if real_l1 is not None:
        print(f"  real (spray_water) eval/action_l1 = {real_l1:.6f}  (action_l2 = {real_s.get('eval/action_l2'):.6f})")
    if sim_l1 is not None and real_l1 is not None:
        ratio = real_l1 / sim_l1 if sim_l1 > 1e-12 else float("inf")
        print(f"  ratio real/sim = {ratio:.2f}x")
        if ratio > 2.0:
            print("  -> real open-loop action error MUCH larger than sim => model/data (H1/H2/H3).")
        elif ratio < 1.5:
            print("  -> real open-loop action error comparable to sim.")
            print("     Combined with real rollout FAILURE, this means the problem is NOT open-loop")
            print("     action quality but CLOSED-LOOP behavior. Most consistent with H1 (no proprio")
            print("     -> no drift correction) and H2 (rot6d warp makes the manifold fragile to")
            print("     sampling noise + GS orthonormalization changes commanded orientation).")
            print("     The action_l1 metric hides H2 because pred & gt share the same warp.")
        else:
            print("  -> moderate difference; interpret with H2 caveat (action_l1 is warp-invariant).")

    print(f"\nVERDICT (C2: loss convergence):")
    sim_la = sim_s.get("train/loss_action")
    real_la = real_s.get("train/loss_action")
    sim_lv = sim_s.get("train/loss_video")
    real_lv = real_s.get("train/loss_video")
    print(f"  train/loss_action: sim={sim_la}  real={real_la}")
    print(f"  train/loss_video:  sim={sim_lv}  real={real_lv}")
    if real_la is not None and real_lv is not None:
        if real_la < real_lv * 0.1:
            print("  -> real loss_action << loss_video: action expert converges, so the action")
            print("     expert CAN fit the (warped) target. Failure is in closed-loop generalization,")
            print("     consistent with H1 (no proprio) + H2 (warp fragility under sampling noise).")
        else:
            print("  -> real loss_action not much smaller than loss_video: action expert may be")
            print("     under-trained or struggling with the warped rot6d manifold (H2).")

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2, default=str))
        print(f"\nWrote full results to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
