#!/usr/bin/env python3
"""Strip HuggingFace parquet metadata that breaks older FastWAM loaders."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import pyarrow.parquet as pq


def _link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def prepare_fastwam_dataset(source_root: Path, output_root: Path, *, overwrite: bool) -> None:
    if not source_root.exists():
        raise FileNotFoundError(f"Missing source dataset: {source_root}")

    marker = output_root / ".fastwam_prepared"
    if output_root.exists():
        if marker.exists() and not overwrite:
            print(f"[fix_lerobot_parquet_metadata] already prepared: {output_root}")
            return
        if overwrite:
            shutil.rmtree(output_root)

    output_root.mkdir(parents=True, exist_ok=True)

    for rel in sorted(source_root.rglob("*.parquet")):
        rel_path = rel.relative_to(source_root)
        out_path = output_root / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        table = pq.read_table(rel).replace_schema_metadata()
        pq.write_table(table, out_path)

    for rel in sorted(source_root.rglob("*")):
        if not rel.is_file() or rel.suffix == ".parquet":
            continue
        out_path = output_root / rel.relative_to(source_root)
        if out_path.exists():
            continue
        _link_or_copy(rel, out_path)

    videos_src = source_root / "videos"
    videos_dst = output_root / "videos"
    if videos_src.exists() and not videos_dst.exists():
        os.symlink(videos_src.resolve(), videos_dst, target_is_directory=True)

    marker.write_text(f"source={source_root.resolve()}\n", encoding="utf-8")
    print(f"[fix_lerobot_parquet_metadata] prepared {output_root} from {source_root}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    prepare_fastwam_dataset(args.source_root, args.output_root, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
