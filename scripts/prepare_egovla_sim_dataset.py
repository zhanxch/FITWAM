#!/usr/bin/env python3
"""Convert EgoVLA simulator HDF5 episodes to FastWAM's LeRobot-style format."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if SRC_ROOT.exists() and str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fastwam.datasets.egovla_sim.constants import EGOVLA_SIM_ROOT
from fastwam.datasets.egovla_sim.converter import (
    prepare_dataset,
    prepare_merged_dataset,
    resolve_split_sources,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=None, help="Single task HDF5 directory.")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--sim-root", type=Path, default=EGOVLA_SIM_ROOT)
    parser.add_argument(
        "--split",
        choices=("long", "short"),
        default=None,
        help="Merge preset EgoVLA_SIM task groups into one dataset.",
    )
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--include-hand-pose",
        action="store_true",
        help="Include hand pose arrays in parquet/meta fields.",
    )
    parser.add_argument(
        "--overlay-hand-pose",
        action="store_true",
        help="Draw projected left/right EE + fingertip pose on the output videos.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.split is not None:
        sources = resolve_split_sources(args.sim_root, args.split)
        prepare_merged_dataset(
            sources,
            args.output_root,
            sim_root=args.sim_root,
            fps=args.fps,
            overwrite=args.overwrite,
            include_hand_pose=args.include_hand_pose,
            overlay_hand_pose=args.overlay_hand_pose,
        )
        return

    if args.source_root is None:
        raise ValueError("Either --split or --source-root must be provided.")

    prepare_dataset(
        source_root=args.source_root,
        output_root=args.output_root,
        task=args.task,
        fps=args.fps,
        overwrite=args.overwrite,
        include_hand_pose=args.include_hand_pose,
        overlay_hand_pose=args.overlay_hand_pose,
    )


if __name__ == "__main__":
    main()
