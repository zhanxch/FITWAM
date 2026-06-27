#!/usr/bin/env python3
"""Resize LeRobot dataset videos to a fixed square resolution for faster training I/O."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

DEFAULT_SOURCE_ROOT = Path("data/egodex_part2_basic_pnp_fastwam_video_pretrain")
DEFAULT_OUTPUT_ROOT = Path("data/egodex_part2_basic_pnp_fastwam_video_pretrain_384")
DEFAULT_SIZE = 384


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def _link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _list_video_keys(features: dict[str, Any]) -> list[str]:
    return sorted(key for key, value in features.items() if value.get("dtype") == "video")


def _resize_one(src: Path, dst: Path, size: int, crf: int) -> tuple[str, str | None]:
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(src),
        "-vf",
        f"scale={size}:{size}",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-an",
        str(dst),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return str(src), result.stderr[-1000:]
    return str(src), None


def _copy_non_video_assets(source_root: Path, output_root: Path) -> None:
    if output_root.exists():
        raise FileExistsError(f"Output already exists: {output_root}. Pass --overwrite to replace it.")
    output_root.mkdir(parents=True, exist_ok=True)

    for name in ("meta", "data"):
        src_dir = source_root / name
        if not src_dir.exists():
            raise FileNotFoundError(f"Missing {src_dir}")
        for src in src_dir.rglob("*"):
            if src.is_dir():
                continue
            rel = src.relative_to(source_root)
            dst = output_root / rel
            _link_or_copy(src, dst)


def _update_info_video_shape(info: dict[str, Any], video_keys: list[str], size: int) -> dict[str, Any]:
    updated = dict(info)
    features = dict(updated["features"])
    for key in video_keys:
        feature = dict(features[key])
        feature["shape"] = [size, size, 3]
        video_info = dict(feature.get("info", {}))
        video_info["video.height"] = size
        video_info["video.width"] = size
        video_info["video.codec"] = "h264"
        video_info["video.pix_fmt"] = "yuv420p"
        feature["info"] = video_info
        features[key] = feature
    updated["features"] = features
    return updated


def resize_dataset(
    source_root: Path,
    output_root: Path,
    *,
    size: int,
    workers: int,
    crf: int,
    overwrite: bool,
    video_keys: list[str] | None,
) -> None:
    if not source_root.exists():
        raise FileNotFoundError(f"Missing source dataset: {source_root}")

    info = _read_json(source_root / "meta" / "info.json")
    video_path_tmpl = info["video_path"]
    keys = video_keys or _list_video_keys(info["features"])
    if not keys:
        raise ValueError("No video features found in info.json")

    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {output_root}. Pass --overwrite to replace it.")
        shutil.rmtree(output_root)

    _copy_non_video_assets(source_root, output_root)

    jobs: list[tuple[Path, Path]] = []
    for key in keys:
        src_glob = source_root / "videos"
        for src_video in sorted(src_glob.rglob(f"{key}/*.mp4")):
            rel = src_video.relative_to(source_root)
            jobs.append((src_video, output_root / rel))

    print(f"Resizing {len(jobs)} videos to {size}x{size} with {workers} workers -> {output_root}")
    failures: list[tuple[str, str]] = []
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_resize_one, src, dst, size, crf): (src, dst)
            for src, dst in jobs
        }
        for future in as_completed(futures):
            src, err = future.result()
            done += 1
            if err is not None:
                failures.append((src, err))
            if done % 500 == 0 or done == len(jobs):
                print(f"Resized {done}/{len(jobs)} videos")

    if failures:
        sample = failures[0]
        raise RuntimeError(
            f"Failed to resize {len(failures)} videos. First error for {sample[0]}:\n{sample[1]}"
        )

    updated_info = _update_info_video_shape(info, keys, size)
    _write_json(output_root / "meta" / "info.json", updated_info)
    print(f"Updated video shape in {output_root / 'meta/info.json'}")
    print(f"Done. Dataset ready at {output_root}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE, help="Output square side length.")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--crf", type=int, default=23)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--video-key",
        action="append",
        default=None,
        help="Optional video feature key(s). Defaults to all video features in info.json.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    resize_dataset(
        source_root=args.source_root,
        output_root=args.output_root,
        size=args.size,
        workers=args.workers,
        crf=args.crf,
        overwrite=args.overwrite,
        video_keys=args.video_key,
    )


if __name__ == "__main__":
    main()
