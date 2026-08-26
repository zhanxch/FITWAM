#!/usr/bin/env python3
"""Rank once-CFG policies from closed-loop node-outcome tables.

Each FORCE_REPLAN=i run gives: intervene at replan i, then 本体, success/fail.
Policy class evaluated here (no energy needed):
  K: skip 0..K-1; fire CFG once at replan K; rest 本体.
That is exactly the i=K lookup.

After several i tables exist, pick K maximizing rescued - fragile_broken
(both_ok assumed unchanged). Then run official 4×50.

Energy-based C (first i>=K with E in band) can be layered later using the
same tables + residual dumps; do not rank energy bands without node outcomes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_eps(out_root: Path) -> list[dict]:
    summary = json.loads((out_root / "rescue_summary.json").read_text(encoding="utf-8"))
    return list(summary.get("episodes") or [])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design-json", type=Path, required=True)
    parser.add_argument(
        "--runs",
        nargs="+",
        help="Pairs INDEX=OUT_ROOT, e.g. 0=/path/at0 1=/path/at1",
    )
    parser.add_argument("--baseline-agg", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    design = json.loads(args.design_json.read_text(encoding="utf-8"))
    baseline = json.loads(args.baseline_agg.read_text(encoding="utf-8"))
    fail = {
        (int(rep) + 1, int(s))
        for rep, seeds in design["fail_seeds_by_eval_repeat"].items()
        for s in seeds
    }
    frag = {
        (int(rep) + 1, int(s))
        for rep, seeds in design["fragile_seeds_by_eval_repeat"].items()
        for s in seeds
    }
    base_pooled = int(baseline["pooled_successes"])
    base_n = int(baseline["pooled_episodes"])
    base_rate = float(baseline["pooled_success_rate"])

    by_i: dict[int, list[dict]] = {}
    for item in args.runs or []:
        idx_s, path = item.split("=", 1)
        by_i[int(idx_s)] = _load_eps(Path(path))

    ranked = []
    for i, rows in sorted(by_i.items()):
        rescued = broken = fail_n = frag_n = 0
        for row in rows:
            k = (int(row["run"]), int(row["seed"]))
            ok = bool(row["success"])
            if k in fail:
                fail_n += 1
                if ok:
                    rescued += 1
            elif k in frag:
                frag_n += 1
                if not ok:
                    broken += 1
        net = rescued - broken
        hyp = (base_pooled + net) / base_n
        ranked.append(
            {
                "policy": f"K={i}: skip 0..{i-1}, CFG once at replan {i}, then 本体",
                "force_replan": i,
                "failures_tried": fail_n,
                "failures_rescued": rescued,
                "fragile_tried": frag_n,
                "fragile_broken": broken,
                "net": net,
                "hypothetical_sr": hyp,
                "delta_sr": hyp - base_rate,
            }
        )
    ranked.sort(key=lambda r: (-int(r["net"]), -float(r["delta_sr"])))
    payload = {
        "baseline_sr": base_rate,
        "note": "both_ok assumed unchanged. Pick best K then official 4x50.",
        "ranked": ranked,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
