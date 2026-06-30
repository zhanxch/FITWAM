"""Export T5 text-embedding caches from .pt → .npz (torch-free for eval clients).

Run this in the ``fastwam`` conda env (which has torch) after
``precompute_text_embeds.py``. The resulting ``.npz`` files can be loaded by
the DexJoCo eval client without any torch dependency.

Usage:
    python scripts/water_plant/export_text_embed_cache_npz.py --cache-dir data/text_embeds_cache/water_plant
    python scripts/water_plant/export_text_embed_cache_npz.py --cache-dir data/text_embeds_cache/dexjoco_microwave_cook
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def export_cache(cache_dir: Path) -> tuple[int, int]:
    if not cache_dir.exists():
        raise FileNotFoundError(f"Cache directory not found: {cache_dir}")

    try:
        import torch
    except ImportError:
        print(
            "ERROR: torch is required to read .pt caches. Run this script in the "
            "`fastwam` conda env, not `dexjoco`.",
            file=sys.stderr,
        )
        raise

    pt_files = sorted(cache_dir.glob("*.t5_len*.wan22ti2v5b.pt"))
    if not pt_files:
        print(f"No .pt caches found in {cache_dir}")
        return 0, 0

    exported = 0
    skipped = 0
    for pt_path in pt_files:
        npz_path = pt_path.with_suffix(".npz")
        # The filename is like <hash>.t5_len128.wan22ti2v5b.pt → keep prefix
        npz_path = pt_path.parent / (pt_path.stem.replace(".wan22ti2v5b", "") + ".npz")
        if npz_path.exists():
            skipped += 1
            continue

        payload = torch.load(pt_path, map_location="cpu")
        context = payload["context"].to(dtype=torch.float32).numpy()
        mask = payload["mask"].bool().numpy()

        import numpy as np

        np.savez(npz_path, context=context, mask=mask)
        exported += 1
        print(f"  {pt_path.name} → {npz_path.name}  context={context.shape} mask={mask.shape}")

    return exported, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description="Export T5 text caches .pt → .npz")
    parser.add_argument(
        "--cache-dir",
        type=str,
        required=True,
        help="Text embedding cache directory (e.g. data/text_embeds_cache/water_plant).",
    )
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir).resolve()
    print(f"Exporting text caches from: {cache_dir}")
    exported, skipped = export_cache(cache_dir)
    print(f"Done. Exported: {exported}, already existed (skipped): {skipped}")


if __name__ == "__main__":
    main()
