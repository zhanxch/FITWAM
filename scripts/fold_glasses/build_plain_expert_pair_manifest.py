#!/usr/bin/env python3
"""Build plain offline ablation manifest: expert success + pair success.

No DEWO aux duplicates, no failure-context samples, all action_loss enabled,
all batch_role=primary. Prompts unchanged (task string only).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from fastwam.everobot_schema import validate_manifest, with_manifest_hash


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dewo-pair-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    src = load_json(args.dewo_pair_manifest)
    samples: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for row in src.get("samples", []):
        if row.get("sample_role") in {"success_auxiliary", "failure_context"}:
            continue
        if row.get("episode_outcome") != "success":
            continue
        item = dict(row)
        item["batch_role"] = "primary"
        item["action_loss"] = "enabled"
        key = (
            item.get("dataset_id"),
            item.get("episode_index"),
            item.get("start_frame"),
            item.get("end_frame"),
        )
        if key in seen:
            continue
        seen.add(key)
        samples.append(item)

    manifest = {
        "schema_version": src.get("schema_version", "0.2"),
        "format": src.get("format", "EveRobotTrainManifest"),
        "manifest_name": "offline_plain_expert_pair",
        "eve_root": src.get("eve_root"),
        "frame_interval": src.get("frame_interval", "half_open"),
        "dataset_roots": dict(src.get("dataset_roots") or {}),
        "source_round_ids": sorted({str(s["round_id"]) for s in samples}),
        "source_hashes": dict(src.get("source_hashes") or {}),
        "samples": samples,
        "selection": {
            "recipe": "fold_glasses_plain_offline_expert_plus_pair_success",
            "primary": "expert_success_plus_pair_success_action",
            "auxiliary": None,
            "include_s0_success_rollouts": False,
            "notes": (
                "Ablation baseline vs DEWOv2: same expert+pair-success clips, "
                "uniform sampling, action on, no CFG / role mix."
            ),
        },
    }
    hashed = with_manifest_hash(manifest)
    validate_manifest(hashed, verify_hash=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(hashed, indent=2, sort_keys=True) + "\n")
    n_expert = sum(
        1 for s in samples if s.get("dataset_id") == "fold_glasses_expert_success"
    )
    n_pair = len(samples) - n_expert
    print(
        f"wrote {args.output} samples={len(samples)} "
        f"expert={n_expert} pair_success={n_pair}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
