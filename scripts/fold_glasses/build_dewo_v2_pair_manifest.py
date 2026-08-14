#!/usr/bin/env python3
"""Build the DEWO v2 pair training manifest.

Primary: expert success episodes mixed with pair success events, action loss on.
Aux: one pair-success copy (action loss off) and the paired failure event
(action loss off). Original S0 success rollouts are not included.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from fastwam.everobot_schema import validate_manifest, with_manifest_hash

PROMPT = "Fold the glasses and place them into the case."
NUM_FRAMES = 33


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expert-manifest", type=Path, required=True)
    parser.add_argument("--pair-dataset", type=Path, required=True)
    parser.add_argument("--pair-dataset-id", default="fold_glasses_pair_events")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    expert = load_json(args.expert_manifest)
    pair_root = args.pair_dataset.expanduser().resolve()
    outcomes = {
        int(row["episode_index"]): row
        for row in load_jsonl(pair_root / "meta" / "episode_outcomes.jsonl")
    }
    episodes = load_jsonl(pair_root / "meta" / "episodes.jsonl")
    pair_index = load_json(pair_root / "pair_index.json")["pairs"]
    by_success = {int(row["success_episode_index"]): row for row in pair_index}
    by_failure = {int(row["failure_episode_index"]): row for row in pair_index}

    samples = [
        dict(row)
        for row in expert.get("samples", [])
        if row.get("episode_outcome") == "success"
        and row.get("batch_role", "primary") == "primary"
    ]
    for row in samples:
        row["batch_role"] = "primary"
        row["action_loss"] = "enabled"

    for ep in episodes:
        ep_idx = int(ep["episode_index"])
        length = int(ep["length"])
        if length != NUM_FRAMES:
            raise ValueError(f"pair episode {ep_idx} length {length} != {NUM_FRAMES}")
        outcome_row = outcomes[ep_idx]
        success = bool(outcome_row.get("success"))
        pair_meta = by_success.get(ep_idx) or by_failure.get(ep_idx)
        if pair_meta is None:
            raise ValueError(f"episode {ep_idx} missing from pair_index.json")
        common = {
            "sample_type": "event",
            "dataset_id": args.pair_dataset_id,
            "dataset_root": str(pair_root),
            "episode_id": f"{args.pair_dataset_id}_ep{ep_idx:06d}",
            "episode_index": ep_idx,
            "round_id": f"{args.pair_dataset_id}::r1",
            "collection_round": 1,
            "task": PROMPT,
            "start_frame": 0,
            "end_frame": NUM_FRAMES,
            "sample_stride": 1,
            "split": "train",
            "pair_id": pair_meta["pair_id"],
            "core_start_frame": 0,
            "core_end_frame": NUM_FRAMES,
            "source_window_rule": "recoverability_pair_33",
            "effector": "global",
        }
        if success:
            samples.append(
                {
                    **common,
                    "sample_id": f"{args.pair_dataset_id}_ep{ep_idx:06d}_success_primary",
                    "event_id": f"{args.pair_dataset_id}_ep{ep_idx:06d}_success_primary",
                    "episode_outcome": "success",
                    "event_outcome": "success",
                    "event_type": "success_event",
                    "action_loss": "enabled",
                    "batch_role": "primary",
                    "sample_role": "success_event_primary",
                }
            )
            samples.append(
                {
                    **common,
                    "sample_id": f"{args.pair_dataset_id}_ep{ep_idx:06d}_success_aux",
                    "event_id": f"{args.pair_dataset_id}_ep{ep_idx:06d}_success_aux",
                    "episode_outcome": "success",
                    "event_outcome": "success",
                    "event_type": "success_event",
                    "action_loss": "disabled",
                    "batch_role": "auxiliary",
                    "sample_role": "success_auxiliary",
                }
            )
        else:
            samples.append(
                {
                    **common,
                    "sample_id": f"{args.pair_dataset_id}_ep{ep_idx:06d}_failure",
                    "event_id": f"{args.pair_dataset_id}_ep{ep_idx:06d}_failure",
                    "episode_outcome": "failure",
                    "event_outcome": "failure",
                    "event_type": "failure_event",
                    "action_loss": "disabled",
                    "batch_role": "auxiliary",
                    "sample_role": "failure_context",
                }
            )

    n_primary_success_events = sum(
        1
        for row in samples
        if row.get("event_type") == "success_event" and row.get("batch_role") == "primary"
    )
    n_aux_success = sum(
        1
        for row in samples
        if row.get("event_type") == "success_event" and row.get("batch_role") == "auxiliary"
    )
    n_aux_failure = sum(
        1 for row in samples if row.get("event_type") == "failure_event"
    )
    if n_primary_success_events != n_aux_success or n_aux_success != n_aux_failure:
        raise SystemExit(
            "Pair counts must match: "
            f"success_primary={n_primary_success_events} "
            f"success_aux={n_aux_success} failure={n_aux_failure}"
        )

    dataset_roots = dict(expert.get("dataset_roots") or {})
    dataset_roots[args.pair_dataset_id] = str(pair_root)
    source_round_ids = sorted({str(row["round_id"]) for row in samples})
    manifest = {
        "schema_version": expert.get("schema_version", "0.2"),
        "format": expert.get("format", "EveRobotTrainManifest"),
        "manifest_name": "offline_b1_jump_fast_pair",
        "eve_root": expert.get("eve_root"),
        "frame_interval": expert.get("frame_interval", "half_open"),
        "dataset_roots": dataset_roots,
        "source_round_ids": source_round_ids,
        "source_hashes": dict(expert.get("source_hashes") or {}),
        "samples": samples,
        "selection": {
            "recipe": "fold_glasses_dewo_v2_recoverability_pairs",
            "primary": "expert_success_plus_pair_success_action",
            "auxiliary": "pair_success_video_plus_pair_failure_video",
            "include_s0_success_rollouts": False,
            "num_pairs": len(pair_index),
        },
    }
    hashed = with_manifest_hash(manifest)
    validate_manifest(hashed, verify_hash=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(hashed, indent=2, sort_keys=True) + "\n")
    print(
        f"wrote {args.output} samples={len(samples)} "
        f"pairs={len(pair_index)} expert_primary="
        f"{len(samples) - 3 * len(pair_index)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
