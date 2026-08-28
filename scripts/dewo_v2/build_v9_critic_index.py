#!/usr/bin/env python3
"""Build v9 full-horizon critic index from mixed-S0 collect + recoverability scan."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.collect_dexjoco_rollouts import read_json, write_json  # noqa: E402


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def video_paths(raw: Path, ep: int) -> dict[str, str]:
    chunk = int(ep) // 1000
    return {
        "front": str(
            raw
            / f"videos/chunk-{chunk:03d}/observation.images.front/episode_{int(ep):06d}.mp4"
        ),
        "wrist": str(
            raw
            / f"videos/chunk-{chunk:03d}/observation.images.wrist/episode_{int(ep):06d}.mp4"
        ),
    }


def build_index(collect_root: Path, scan_root: Path, raw: Path) -> dict:
    outcomes = {
        int(r["episode_index"]): r
        for r in load_jsonl(raw / "meta" / "episode_outcomes.jsonl")
    }
    episodes = {
        int(r["episode_index"]): r
        for r in load_jsonl(raw / "meta" / "episodes.jsonl")
    }
    prefs = load_jsonl(scan_root / "prefix_results.jsonl")
    pairs = load_jsonl(scan_root / "event_pair_manifest.jsonl")

    d0: list[dict] = []
    fails: list[dict] = []
    for i, o in sorted(outcomes.items()):
        rec = {
            "episode_index": i,
            "seed": o.get("seed"),
            "repeat": o.get("attempt_index"),
            "success": bool(o.get("success")),
            "length": int(episodes[i]["length"]),
            "videos": video_paths(raw, i),
        }
        if rec["success"]:
            d0.append(rec)
        else:
            fails.append(rec)

    pair_rows: list[dict] = []
    for p in pairs:
        if p.get("status") != "complete":
            continue
        se = read_json(Path(p["counterfactual_success_event"]))
        pair_json = Path(p["counterfactual_success_event"]).parent / "pair.json"
        pj = read_json(pair_json) if pair_json.exists() else {}
        fr = pj.get("frontier") or p.get("frontier") or {}
        led_path = se.get("successful_continuation_ledger")
        led = read_json(Path(led_path)) if led_path and Path(led_path).exists() else {}
        ep = int(p["source_failure_episode_index"])
        t = int(fr.get("last_recoverable_frame") or se.get("t_frame"))
        m_first = int(fr.get("first_zero_frame") or se.get("t_plus_24_frame"))
        succ_end = int(
            led.get("final_global_frame_exclusive")
            or (t + int(led.get("steps_executed") or 0))
        )
        fail_len = int(episodes[ep]["length"])
        pair_rows.append(
            {
                "pair_id": p.get("pair_id"),
                "source_failure_episode_index": ep,
                "seed": p.get("seed"),
                "t_star_last_recoverable": t,
                "M_first_zero": m_first,
                "event_crop_33": [
                    fr.get("event_start"),
                    fr.get("event_end_exclusive"),
                ],
                "shared_prefix_frames": [0, t],
                "success_branch": {
                    "outcome": True,
                    "start_frame": t,
                    "end_frame_exclusive": succ_end,
                    "total_length": succ_end,
                    "continuation_arrays": se.get("successful_continuation_arrays"),
                    "continuation_ledger": led_path,
                    "continuation_front": led.get("continuation_front_video")
                    or se.get("full_front_video"),
                    "continuation_wrist": led.get("continuation_wrist_video")
                    or se.get("full_wrist_video"),
                    "prefix_videos": video_paths(raw, ep),
                },
                "failure_branch": {
                    "outcome": False,
                    "start_frame": 0,
                    "end_frame_exclusive": fail_len,
                    "total_length": fail_len,
                    "source": "factual_gt_failure_rollout",
                    "videos": video_paths(raw, ep),
                },
                "length_gap": fail_len - succ_end,
            }
        )

    prefix_labels: list[dict] = []
    for r in prefs:
        prefix_labels.append(
            {
                "source_failure_episode_index": int(r["source_failure_episode_index"]),
                "seed": r.get("seed"),
                "prefix_frame": int(r["prefix_frame"]),
                "success_count": int(r["success_count"]),
                "success_rate": float(r["success_rate"]),
                "pass_m": int(r.get("pass_m") or 4),
                "training_eligible": bool(r.get("training_eligible")),
                "trajectory_ledgers": r.get("trajectory_ledgers"),
            }
        )

    by_ep: dict[int, list[dict]] = defaultdict(list)
    for r in prefix_labels:
        by_ep[r["source_failure_episode_index"]].append(r)

    never_recoverable: list[dict] = []
    for ep, rs in by_ep.items():
        if all(x["success_count"] == 0 for x in rs):
            never_recoverable.append(
                {
                    "episode_index": ep,
                    "min_prefix_scanned": min(x["prefix_frame"] for x in rs),
                    "n_prefixes": len(rs),
                    "length": int(episodes[ep]["length"]),
                }
            )

    def _stats(values: list[int]) -> dict[str, int]:
        if not values:
            return {"min": 0, "median": 0, "max": 0}
        s = sorted(values)
        return {"min": s[0], "median": s[len(s) // 2], "max": s[-1]}

    return {
        "format": "v9_critic_index_v0",
        "note": (
            "Full-horizon critic sources already on disk. Do not train V on 33-frame pair crops. "
            "Target is counterfactual Pass@M / P(success|s,S0), not factual MC return. "
            "Do not pad success branches to failure length."
        ),
        "collect_root": str(collect_root.resolve()),
        "rollout_raw": str(raw.resolve()),
        "scan_root": str(scan_root.resolve()),
        "counts": {
            "d0_full_success_episodes": len(d0),
            "factual_failure_episodes": len(fails),
            "pass_at_m_prefix_labels": len(prefix_labels),
            "unique_failures_scanned": len(by_ep),
            "complete_recoverability_pairs": len(pair_rows),
            "failures_never_recoverable_in_scan": len(never_recoverable),
        },
        "length_stats": {
            "d0_success_len": _stats([x["length"] for x in d0]),
            "factual_fail_len": _stats([x["length"] for x in fails]),
            "pair_success_branch_len": _stats(
                [x["success_branch"]["total_length"] for x in pair_rows]
            ),
            "pair_failure_branch_len": _stats(
                [x["failure_branch"]["total_length"] for x in pair_rows]
            ),
            "t_star": Counter(x["t_star_last_recoverable"] for x in pair_rows).most_common(),
        },
        "d0_success_episodes": d0,
        "factual_failure_episodes": fails,
        "pass_at_m_prefix_labels": prefix_labels,
        "full_horizon_pairs": pair_rows,
        "never_recoverable_failures": never_recoverable,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--collect-root",
        type=Path,
        required=True,
        help="Mixed S0 collect root (contains rollout_raw_200).",
    )
    parser.add_argument(
        "--scan-root",
        type=Path,
        default=None,
        help="Recoverability scan root (default: collect/recoverability_pairs_v2).",
    )
    parser.add_argument(
        "--raw-dataset",
        type=Path,
        default=None,
        help="Rollout LeRobot root (default: collect_root/rollout_raw_200).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON (default: collect_root/v9_critic_index.json).",
    )
    args = parser.parse_args(argv)

    collect_root = args.collect_root.expanduser().resolve()
    raw = (
        args.raw_dataset.expanduser().resolve()
        if args.raw_dataset is not None
        else collect_root / "rollout_raw_200"
    )
    scan_root = (
        args.scan_root.expanduser().resolve()
        if args.scan_root is not None
        else collect_root / "recoverability_pairs_v2"
    )
    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else collect_root / "v9_critic_index.json"
    )

    for required in (
        raw / "meta" / "episode_outcomes.jsonl",
        raw / "meta" / "episodes.jsonl",
        scan_root / "prefix_results.jsonl",
        scan_root / "event_pair_manifest.jsonl",
    ):
        if not required.exists():
            raise SystemExit(f"Missing required input: {required}")

    index = build_index(collect_root, scan_root, raw)
    write_json(output, index)
    print(f"wrote {output}")
    print(json.dumps(index["counts"], indent=2))
    if not index["full_horizon_pairs"]:
        raise SystemExit("No complete full_horizon_pairs; scan may be incomplete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
