"""Convert a LeRobot v2.1 dataset to EveRobot format.

EveRobot format keeps the original LeRobot data files (mp4 videos, parquet
tables, meta JSON) unchanged and adds:

1. ``everobot/everobot_manifest.json`` — episode-level metadata for fast
   dataset initialization (episode boundaries, video paths, task strings).
2. ``everobot/episode_XXXXXX.npz`` — per-episode action and state arrays
   extracted from parquet, for fast numpy loading without pyarrow.

The original dataset directory is **not modified**. All EveRobot artifacts are
written to a separate ``everobot/`` subdirectory (or a user-specified output
directory).

Usage:
    python scripts/water_plant/convert_lerobot_to_everobot.py \
        --dataset-dir data/water_plant_fastwam \
        --video-keys front wrist \
        --output-dir data/water_plant_fastwam/everobot
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from tqdm import tqdm

from fastwam.datasets.everobot.manifest import build_episode_records, save_manifest
from fastwam.utils.logging_config import setup_logging


def load_parquet_action_state(parquet_path: str):
    """Load action and state arrays from a single-episode parquet file."""
    import pyarrow.parquet as pq

    table = pq.read_table(parquet_path, columns=["action", "observation.state", "frame_index", "timestamp"])
    df = table.to_pandas()
    action = np.stack(df["action"].values)  # [L, action_dim]
    state = np.stack(df["observation.state"].values)  # [L, state_dim]
    frame_index = df["frame_index"].values.astype(np.int64)  # [L]
    timestamp = df["timestamp"].values.astype(np.float32)  # [L]
    return action, state, frame_index, timestamp


def main():
    parser = argparse.ArgumentParser(description="Convert LeRobot dataset to EveRobot format.")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        required=True,
        help="Path to the LeRobot v2.1 dataset directory (e.g. data/water_plant_fastwam).",
    )
    parser.add_argument(
        "--video-keys",
        type=str,
        nargs="+",
        required=True,
        help="Camera keys as named in meta/modality.json (e.g. front wrist).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for EveRobot artifacts. Defaults to <dataset-dir>/everobot/.",
    )
    parser.add_argument(
        "--extract-arrays",
        action="store_true",
        default=True,
        help="Extract per-episode action/state arrays to .npz for fast loading.",
    )
    parser.add_argument(
        "--no-extract-arrays",
        dest="extract_arrays",
        action="store_false",
        help="Skip .npz extraction (manifest only).",
    )
    args = parser.parse_args()

    setup_logging()
    dataset_dir = Path(args.dataset_dir).resolve()
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    output_dir = Path(args.output_dir) if args.output_dir else dataset_dir / "everobot"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[everobot] Converting {dataset_dir} → {output_dir}")

    # Read info.json for fps, dims, resolution
    with open(dataset_dir / "meta" / "info.json", "r", encoding="utf-8") as f:
        info = json.load(f)
    fps = int(info["fps"])

    # Read modality.json for action/state dims
    modality_path = dataset_dir / "meta" / "modality.json"
    modality = {}
    if modality_path.exists():
        with open(modality_path, "r", encoding="utf-8") as f:
            modality = json.load(f)

    # Determine dims from modality or parquet
    action_dim = state_dim = 0
    if modality:
        if "action" in modality:
            action_dim = max(v["end"] for v in modality["action"].values()) - min(
                v["start"] for v in modality["action"].values()
            )
        if "state" in modality:
            state_dim = max(v["end"] for v in modality["state"].values()) - min(
                v["start"] for v in modality["state"].values()
            )

    # Build episode records
    episodes = build_episode_records(
        str(dataset_dir),
        video_keys=args.video_keys,
    )

    # Extract arrays and update records with npz paths
    if args.extract_arrays:
        print(f"[everobot] Extracting action/state arrays for {len(episodes)} episodes...")
        for ep in tqdm(episodes, desc="Extracting arrays"):
            action, state, frame_index, timestamp = load_parquet_action_state(ep["parquet_path"])
            if action_dim == 0:
                action_dim = action.shape[1]
            if state_dim == 0:
                state_dim = state.shape[1]

            npz_path = output_dir / f"episode_{ep['episode_index']:06d}.npz"
            np.savez_compressed(
                npz_path,
                action=action.astype(np.float32),
                state=state.astype(np.float32),
                frame_index=frame_index,
                timestamp=timestamp,
            )
            ep["npz_path"] = str(npz_path)

    # Read video resolution from first video
    video_resolution = [0, 0]
    if episodes:
        first_video = episodes[0]["video_paths"][args.video_keys[0]]
        if os.path.exists(first_video):
            try:
                import imageio

                reader = imageio.get_reader(first_video)
                meta = reader.get_meta_data()
                video_resolution = [meta.get("size", [0, 0])[1], meta.get("size", [0, 0])[0]]
                reader.close()
            except Exception as e:
                print(f"[everobot] Warning: could not read video resolution: {e}")
    if video_resolution == [0, 0]:
        # Fallback to info.json
        imgs = info.get("features", {}).get("observation.images.front", {})
        if "shape" in imgs:
            shape = imgs["shape"]
            video_resolution = [shape[1], shape[2]]  # [H, W]

    # Locate stats
    stats_path = str(dataset_dir / "meta" / "stats.json")
    if not os.path.exists(stats_path):
        stats_path = None

    manifest_path = save_manifest(
        str(output_dir / "everobot_manifest.json"),
        dataset_dir=str(dataset_dir),
        episodes=episodes,
        fps=fps,
        action_dim=int(action_dim),
        state_dim=int(state_dim),
        video_keys=args.video_keys,
        video_resolution=video_resolution,
        modality=modality,
        stats_path=stats_path,
    )

    print(f"[everobot] Manifest saved: {manifest_path}")
    print(f"[everobot] Episodes: {len(episodes)}")
    print(f"[everobot] Action dim: {action_dim}, State dim: {state_dim}")
    print(f"[everobot] Video keys: {args.video_keys}")
    print(f"[everobot] Video resolution: {video_resolution}")
    if args.extract_arrays:
        print(f"[everobot] NPZ arrays: {output_dir}/episode_XXXXXX.npz")
    print("[everobot] Done. Use this with EveRobotDataset(manifest_path=...) or dataset_dirs=[...].")


if __name__ == "__main__":
    main()
