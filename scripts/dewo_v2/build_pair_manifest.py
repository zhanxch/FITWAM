#!/usr/bin/env python3
"""Build the DEWO v9 pair training manifest.

Primary: success episodes from --expert-manifest (expert or S0 success
rollouts) mixed with pair success events, action loss on.
Aux: paired failure events (action loss off). Optional success_auxiliary
copies are off for v9 (``--skip-aux-success``).

Task identity comes from ``--prompt`` / ``--pair-dataset-id`` / ``--recipe``;
this file is not fold_glasses-specific.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from fastwam.everobot_schema import validate_manifest, with_manifest_hash

NUM_FRAMES = 33


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expert-manifest", type=Path, required=True)
    parser.add_argument("--pair-dataset", type=Path, required=True)
    parser.add_argument(
        "--pair-dataset-id",
        required=True,
        help="Manifest dataset_id, typically ${TASK}_pair_events.",
    )
    parser.add_argument("--prompt", required=True, help="Task success instruction.")
    parser.add_argument(
        "--recipe",
        required=True,
        help="Selection recipe label, typically ${TASK}_dewo_v9_recoverability_pairs.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--primary-source",
        choices=("expert", "success_rollouts", "all_success_seeds"),
        default="expert",
    )
    parser.add_argument(
        "--horizon",
        choices=("crop33", "full"),
        default="full",
        help="crop33: 33-frame pair events. full: variable-length rollouts.",
    )
    parser.add_argument(
        "--skip-aux-success",
        action="store_true",
        help="Do not write success_auxiliary copies (v9 pool drops them).",
    )
    args = parser.parse_args(argv)
    prompt = str(args.prompt)
    full_horizon = args.horizon == "full"

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
        if full_horizon:
            if length < NUM_FRAMES:
                raise ValueError(
                    f"pair episode {ep_idx} length {length} < {NUM_FRAMES}"
                )
            end_frame = length
            window_rule = "recoverability_pair_full"
        else:
            if length != NUM_FRAMES:
                raise ValueError(
                    f"pair episode {ep_idx} length {length} != {NUM_FRAMES}"
                )
            end_frame = NUM_FRAMES
            window_rule = "recoverability_pair_33"
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
            "task": prompt,
            "start_frame": 0,
            "end_frame": end_frame,
            "sample_stride": 1,
            "split": "train",
            "pair_id": pair_meta["pair_id"],
            "core_start_frame": 0,
            "core_end_frame": end_frame,
            "source_window_rule": window_rule,
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
            if not args.skip_aux_success:
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
    if args.skip_aux_success:
        if n_primary_success_events != n_aux_failure:
            raise SystemExit(
                "Pair counts must match: "
                f"success_primary={n_primary_success_events} "
                f"failure={n_aux_failure}"
            )
    elif n_primary_success_events != n_aux_success or n_aux_success != n_aux_failure:
        raise SystemExit(
            "Pair counts must match: "
            f"success_primary={n_primary_success_events} "
            f"success_aux={n_aux_success} failure={n_aux_failure}"
        )

    dataset_roots = dict(expert.get("dataset_roots") or {})
    dataset_roots[args.pair_dataset_id] = str(pair_root)
    source_round_ids = sorted({str(row["round_id"]) for row in samples})
    include_s0 = args.primary_source in {"success_rollouts", "all_success_seeds"}
    if args.primary_source == "all_success_seeds":
        primary_label = "all_success_seed_episodes_plus_pair_success_action"
    elif args.primary_source == "success_rollouts":
        primary_label = "success_rollout_episodes_plus_pair_success_action"
    else:
        primary_label = "expert_success_plus_pair_success_action"
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
            "recipe": str(args.recipe),
            "primary": primary_label,
            "auxiliary": "pair_success_video_plus_pair_failure_video",
            "include_s0_success_rollouts": include_s0,
            "primary_source": args.primary_source,
            "num_pairs": len(pair_index),
            "horizon": args.horizon,
            "skip_aux_success": bool(args.skip_aux_success),
        },
    }
    hashed = with_manifest_hash(manifest)
    validate_manifest(hashed, verify_hash=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(hashed, indent=2, sort_keys=True) + "\n")
    n_pair_units = n_primary_success_events + n_aux_success + n_aux_failure
    n_rollout_primary = len(samples) - n_pair_units
    print(
        f"wrote {args.output} samples={len(samples)} "
        f"pairs={len(pair_index)} primary_source={args.primary_source} "
        f"horizon={args.horizon} episode_primary={n_rollout_primary}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
