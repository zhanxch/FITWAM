#!/usr/bin/env python3
"""Export FastWAM .pt text embedding caches to numpy .npz for dexjoco-only eval."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert .pt text caches to .npz.")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=PROJECT_ROOT / "data/text_embeds_cache/dexjoco_ego",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache_dir = args.cache_dir.expanduser().resolve()
    if not cache_dir.exists():
        raise FileNotFoundError(cache_dir)

    pt_files = sorted(cache_dir.glob("*.wan22ti2v5b.pt"))
    if not pt_files:
        raise FileNotFoundError(f"No .pt caches under {cache_dir}")

    for pt_path in pt_files:
        npz_path = pt_path.with_suffix("").with_suffix(".npz")
        # e.g. <hash>.t5_len128.wan22ti2v5b.pt -> <hash>.t5_len128.npz
        npz_path = pt_path.parent / pt_path.name.replace(".wan22ti2v5b.pt", ".npz")
        payload = torch.load(pt_path, map_location="cpu")
        context = payload["context"].to(dtype=torch.float32).numpy()
        mask = payload["mask"].bool().numpy()
        np.savez(npz_path, context=context, mask=mask)
        print(f"Wrote {npz_path}")


if __name__ == "__main__":
    main()
