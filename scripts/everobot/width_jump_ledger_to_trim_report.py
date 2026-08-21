#!/usr/bin/env python3
"""Convert width-jump failure_events.jsonl into a B1-style trim/collection summary.

Each found event becomes the failure keep-interval supplied by the ledger as
``[core_start_frame, core_end_frame)``. Missing detections remain explicit
full-episode fallbacks so downstream manifest selection can retain or discard
them deliberately.

Example:
  python scripts/everobot/width_jump_ledger_to_trim_report.py \\
    --rollout-root data/.../rollout_raw_200 \\
    --ledger data/.../failure_events.jsonl \\
    --output data/.../width_jump_trim_meta
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rollout-root", type=Path, required=True)
    p.add_argument("--ledger", type=Path, required=True, help="failure_events.jsonl")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--fps", type=int, default=30)
    return p.parse_args()


def load_episode_lengths(rollout_root: Path) -> dict[int, int]:
    info = json.loads((rollout_root / "meta" / "info.json").read_text())
    # Prefer episodes.jsonl / episodes.json when present.
    episodes_jsonl = rollout_root / "meta" / "episodes.jsonl"
    lengths: dict[int, int] = {}
    if episodes_jsonl.exists():
        for line in episodes_jsonl.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            lengths[int(row["episode_index"])] = int(row["length"])
        return lengths
    # Fallback: outcomes + parquet not required if ledger carries n_frames via episodes.
    outcomes = rollout_root / "meta" / "episode_outcomes.jsonl"
    if outcomes.exists():
        for line in outcomes.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            ep = int(row["episode_index"])
            if "length" in row:
                lengths[ep] = int(row["length"])
            elif "n_frames" in row:
                lengths[ep] = int(row["n_frames"])
    if lengths:
        return lengths
    total = int(info.get("total_episodes", 0))
    raise SystemExit(
        f"Could not resolve episode lengths under {rollout_root}/meta "
        f"(episodes.jsonl / episode_outcomes.jsonl). info.total_episodes={total}"
    )


def load_outcomes(rollout_root: Path) -> dict[int, bool]:
    path = rollout_root / "meta" / "episode_outcomes.jsonl"
    out: dict[int, bool] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        out[int(row["episode_index"])] = bool(row.get("success"))
    return out


def main() -> int:
    args = parse_args()
    lengths = load_episode_lengths(args.rollout_root)
    outcomes = load_outcomes(args.rollout_root)

    events_by_ep: dict[int, dict[str, Any]] = {}
    for line in args.ledger.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        ep = int(row["episode_index"])
        events_by_ep[ep] = row

    trim_report: list[dict[str, Any]] = []
    missing_event = 0
    for ep in sorted(outcomes):
        success = bool(outcomes[ep])
        length = int(lengths.get(ep, 0))
        if success:
            trim_report.append(
                {
                    "episode_index": ep,
                    "failure": False,
                    "trimmed": False,
                    "original_length": length,
                    "trimmed_length": length,
                    "trim_start_frame": 0,
                    "trim_end_frame": length,
                    "window_rule": "full_success_episode",
                }
            )
            continue
        ev = events_by_ep.get(ep)
        if ev is None:
            missing_event += 1
            # Fallback: keep full failure episode (caller should prefer re-running extract).
            trim_report.append(
                {
                    "episode_index": ep,
                    "failure": True,
                    "trimmed": False,
                    "original_length": length,
                    "trimmed_length": length,
                    "trim_start_frame": 0,
                    "trim_end_frame": length,
                    "window_rule": "width_jump_missing_event_fallback_full",
                    "skip_reason": "no_width_jump_event",
                }
            )
            continue
        start = int(ev["core_start_frame"])
        end = int(ev["core_end_frame"])
        center = int(ev.get("event_center_frame", (start + end) // 2))
        if start < 0 or end > length or start >= end:
            raise SystemExit(
                f"Bad width-jump window for ep={ep}: [{start},{end}) length={length}"
            )
        trim_report.append(
            {
                "episode_index": ep,
                "failure": True,
                "trimmed": True,
                "original_length": length,
                "trimmed_length": end - start,
                "trimmed_tail_steps": length - end,
                "trim_start_frame": start,
                "trim_end_frame": end,
                "core_start_frame": start,
                "core_end_frame": end,
                "event_center_frame": center,
                "window_rule": "width_jump_first_after_ignore_centered_33",
                "jump_ratio": ev.get("jump_ratio"),
                "width": ev.get("width"),
                "baseline_median": ev.get("baseline_median"),
                "detection": ev.get("detection", "action_width_jump"),
            }
        )

    args.output.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema": "width_jump_trim_report_v1",
        "rollout_root": str(args.rollout_root),
        "ledger": str(args.ledger),
        "fps": int(args.fps),
        "episodes": len(trim_report),
        "failures": sum(1 for r in trim_report if r["failure"]),
        "successes": sum(1 for r in trim_report if not r["failure"]),
        "trimmed_failures": sum(1 for r in trim_report if r["failure"] and r["trimmed"]),
        "missing_width_jump_events": missing_event,
        "window_rule": "use ledger [core_start_frame,core_end_frame) exactly",
        "detected_event_window_lengths": sorted(
            {
                int(r["trim_end_frame"]) - int(r["trim_start_frame"])
                for r in trim_report
                if r["failure"] and r["trimmed"]
            }
        ),
        "trim_report": trim_report,
    }
    out_path = args.output / "collection_summary.json"
    out_path.write_text(json.dumps(summary, indent=2) + "\n")
    # Also write a slim jsonl for debugging.
    with (args.output / "trim_report.jsonl").open("w") as f:
        for row in trim_report:
            f.write(json.dumps(row) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "trim_report"}, indent=2))
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
