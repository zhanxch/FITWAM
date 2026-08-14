#!/usr/bin/env python3
"""Rewrite rollout-success units: same-seed pairing, drop never-failed seeds.

Rules:
  - Never-failed seed: drop all success rollouts.
  - Early failure (center / min(success_len) < 0.5): keep the failure
    [core_start, core_end) on each paired success episode.
  - Mid/late failure: drop the first half of each paired success ([L/2, L)).
  - Mixed seeds: union of the above intervals.
  - Expert success and failure events are left unchanged.
  - sample_id is preserved so VAE/FAST cache keys still hit.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from fastwam.everobot_schema import validate_manifest, with_manifest_hash

EARLY_REL_THRESHOLD = 0.5
NUM_FRAMES = 33


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _merge_intervals(intervals: list[tuple[int, int]]) -> list[list[int]]:
    cleaned = sorted((int(a), int(b)) for a, b in intervals if int(b) > int(a))
    if not cleaned:
        return []
    merged = [list(cleaned[0])]
    for start, end in cleaned[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def _nwin(intervals: list[list[int]]) -> int:
    total = 0
    for start, end in intervals:
        total += max(0, int(end) - int(start) - NUM_FRAMES + 1)
    return total


def _build_seed_index(
    *,
    outcomes_path: Path,
    episode_meta_path: Path,
    failure_ledger_path: Path,
) -> dict[int, dict]:
    lens: dict[int, int] = {}
    for row in _load_jsonl(episode_meta_path):
        dataset_id = row.get("dataset_id")
        if dataset_id not in (None, "fold_glasses_s0_rollout"):
            continue
        lens[int(row["episode_index"])] = int(row["length"])

    fail_by_ep = {
        int(row["episode_index"]): row for row in _load_jsonl(failure_ledger_path)
    }

    by_seed: dict[int, dict] = defaultdict(lambda: {"success": [], "failure": []})
    for row in _load_jsonl(outcomes_path):
        seed = int(row["seed"])
        ep = int(row["episode_index"])
        rec = {"ep": ep, "len": lens.get(ep)}
        if row.get("success") or row.get("outcome") == "success":
            by_seed[seed]["success"].append(rec)
        else:
            rec["fail_ev"] = fail_by_ep.get(ep)
            by_seed[seed]["failure"].append(rec)
    return by_seed


def _success_intervals_for_seed(d: dict) -> dict[int, list[list[int]]]:
    succs = [s for s in d["success"] if s.get("len")]
    fails = d["failure"]
    if not succs or not fails:
        return {}

    early_iv: list[tuple[int, int]] = []
    has_late = False
    for fail in fails:
        ev = fail.get("fail_ev")
        if ev is None:
            has_late = True
            continue
        center = int(
            ev.get("event_center_frame")
            or (int(ev["core_start_frame"]) + int(ev["core_end_frame"])) // 2
        )
        rel = min(center / max(int(s["len"]), 1) for s in succs)
        if rel < EARLY_REL_THRESHOLD:
            early_iv.append((int(ev["core_start_frame"]), int(ev["core_end_frame"])))
        else:
            has_late = True

    out: dict[int, list[list[int]]] = {}
    for succ in succs:
        length = int(succ["len"])
        raw: list[tuple[int, int]] = []
        for start, end in early_iv:
            start = max(0, start)
            end = min(end, length)
            if end - start >= NUM_FRAMES:
                raw.append((start, end))
        if has_late:
            mid = length // 2
            if length - mid >= NUM_FRAMES:
                raw.append((mid, length))
        merged = _merge_intervals(raw)
        if merged:
            out[int(succ["ep"])] = merged
    return out


def rewrite_manifest(
    *,
    manifest_path: Path,
    outcomes_path: Path,
    episode_meta_path: Path,
    failure_ledger_path: Path,
    output_path: Path,
) -> dict[str, int]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    by_seed = _build_seed_index(
        outcomes_path=outcomes_path,
        episode_meta_path=episode_meta_path,
        failure_ledger_path=failure_ledger_path,
    )
    ep_to_intervals: dict[int, list[list[int]]] = {}
    for seed_data in by_seed.values():
        ep_to_intervals.update(_success_intervals_for_seed(seed_data))

    kept: list[dict] = []
    stats = {
        "kept_expert": 0,
        "kept_failure": 0,
        "kept_success": 0,
        "dropped_success": 0,
        "success_windows": 0,
    }
    for sample in manifest["samples"]:
        dataset_id = sample.get("dataset_id")
        if dataset_id == "fold_glasses_expert_success":
            kept.append(sample)
            stats["kept_expert"] += 1
            continue
        if sample.get("sample_type") == "event" or sample.get("episode_outcome") == "failure":
            kept.append(sample)
            stats["kept_failure"] += 1
            continue
        ep = int(sample["episode_index"])
        intervals = ep_to_intervals.get(ep)
        if not intervals:
            stats["dropped_success"] += 1
            continue
        rewritten = dict(sample)
        rewritten["valid_intervals"] = intervals
        rewritten["annotation"] = {
            "source": "auto",
            "method": "seedpair_early_crop_late_second_half",
            "version": "dewo_v2_seedpair_v1",
            "early_rel_threshold": EARLY_REL_THRESHOLD,
        }
        kept.append(rewritten)
        stats["kept_success"] += 1
        stats["success_windows"] += _nwin(intervals)

    manifest["samples"] = kept
    manifest["num_samples"] = len(kept)
    selection = dict(manifest.get("selection") or {})
    selection["success_rollout_policy"] = (
        "same_seed; drop never-failed; early crop failure interval; "
        "late drop first half"
    )
    manifest["selection"] = selection
    manifest["manifest_name"] = output_path.stem
    hashed = with_manifest_hash(manifest)
    validate_manifest(hashed, verify_hash=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(hashed, indent=2) + "\n", encoding="utf-8")
    stats["total_units"] = len(kept)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--outcomes", type=Path, required=True)
    parser.add_argument("--episode-meta", type=Path, required=True)
    parser.add_argument("--failure-ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    stats = rewrite_manifest(
        manifest_path=args.manifest,
        outcomes_path=args.outcomes,
        episode_meta_path=args.episode_meta,
        failure_ledger_path=args.failure_ledger,
        output_path=args.output,
    )
    print(json.dumps({"output": str(args.output), **stats}, indent=2))


if __name__ == "__main__":
    main()
