#!/usr/bin/env python3
"""Aggregate opensource baseline 4x50 results: pooled rate + mean±std over 4 repeats."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def _var(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return sum((x - m) ** 2 for x in xs) / len(xs)


def _std(xs: list[float]) -> float:
    return math.sqrt(_var(xs))


def summarize_job(job_dir: Path) -> dict | None:
    episodes = list(job_dir.glob("step_*/episodes.csv"))
    summaries = list(job_dir.glob("step_*/summary.json"))
    if not summaries and not episodes:
        return None

    pooled = None
    if summaries:
        pooled = json.loads(summaries[0].read_text())

    per_repeat: dict[int, list[bool]] = {}
    if episodes:
        with episodes[0].open() as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            rep = int(row["repeat"])
            suc = str(row["success"]).lower() in {"1", "true", "yes"}
            per_repeat.setdefault(rep, []).append(suc)

    rates = []
    run_rows = []
    for rep in sorted(per_repeat):
        vals = per_repeat[rep]
        rate = sum(vals) / len(vals) if vals else float("nan")
        rates.append(rate)
        run_rows.append(
            {
                "run": rep + 1,
                "repeat": rep,
                "successes": int(sum(vals)),
                "episodes": len(vals),
                "rate": rate,
            }
        )

    out = {
        "job_dir": str(job_dir),
        "job_id": job_dir.name,
        "pooled": pooled,
        "runs": run_rows,
        "mean_success_rate": _mean(rates) if rates else (pooled or {}).get("success_rate"),
        "std_success_rate": _std(rates) if rates else None,
        "var_success_rate": _var(rates) if rates else None,
        "n_runs": len(rates),
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--master-log", type=Path, required=True)
    args = ap.parse_args()

    rows = []
    for job_dir in sorted(p for p in args.out_root.iterdir() if p.is_dir() and p.name != "logs"):
        item = summarize_job(job_dir)
        if item is None:
            continue
        rows.append(item)
        (job_dir / "official_4x50_metrics.json").write_text(
            json.dumps(item, indent=2, sort_keys=True) + "\n"
        )

    agg_path = args.out_root / "aggregate_official_4x50.json"
    agg_path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")

    md_lines = [
        "# Opensource baseline official 4×50",
        "",
        f"Root: `{args.out_root}`",
        "",
        "| Job | Pooled | Mean±Std (4 runs) | Runs |",
        "|---|---:|---:|---|",
    ]
    with args.master_log.open("a") as log:
        log.write(f"[aggregate] -> {agg_path}\n")
        for r in rows:
            pooled = r.get("pooled") or {}
            pooled_s = pooled.get("successes")
            pooled_e = pooled.get("episodes")
            pooled_rate = pooled.get("success_rate")
            mean = r.get("mean_success_rate")
            std = r.get("std_success_rate")
            runs = ", ".join(
                f"r{x['run']}={x['successes']}/{x['episodes']}" for x in r.get("runs") or []
            )
            line = (
                f"| `{r['job_id']}` | "
                f"{pooled_s}/{pooled_e} ({None if pooled_rate is None else 100*float(pooled_rate):.1f}%) | "
                f"{None if mean is None else 100*float(mean):.1f}%±"
                f"{None if std is None else 100*float(std):.1f}% | {runs} |"
            )
            md_lines.append(line)
            log.write(
                f"[aggregate] {r['job_id']} pooled={pooled_s}/{pooled_e} "
                f"mean={mean} std={std}\n"
            )

    md_path = args.out_root / "RESULTS.md"
    md_path.write_text("\n".join(md_lines) + "\n")
    print(agg_path)
    print(md_path)


if __name__ == "__main__":
    main()
