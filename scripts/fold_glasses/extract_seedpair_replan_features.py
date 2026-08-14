#!/usr/bin/env python3
"""Extract replan-level rollout features for same-seed event discovery.

The collection policy emits a 32-step chunk and executes the first 24 steps.
This script therefore samples observations once per replan and stores the exact
executed 24-step action block. Visual features are deliberately action-free so
they can be used to align task phase without circularly defining phase by the
policy action being compared.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

# This extractor is PyTorch-only. Some hosts have an incompatible optional
# TensorFlow/protobuf install that Transformers otherwise imports eagerly.
os.environ.setdefault("USE_TF", "0")

import av
import numpy as np
import pyarrow.parquet as pq
import torch
from transformers import CLIPImageProcessor, CLIPVisionModel


DEFAULT_MODEL = "openai/clip-vit-large-patch14-336"
CAMERAS = ("observation.images.front", "observation.images.wrist")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_sampled_video(path: Path, frame_indices: np.ndarray) -> list[np.ndarray]:
    """Decode once in temporal order and return selected RGB frames."""

    wanted = [int(index) for index in frame_indices.tolist()]
    if not wanted:
        return []
    selected: list[np.ndarray] = []
    target_pos = 0
    with av.open(str(path)) as container:
        for frame_index, frame in enumerate(container.decode(video=0)):
            if frame_index < wanted[target_pos]:
                continue
            if frame_index != wanted[target_pos]:
                raise RuntimeError(
                    f"Decoder skipped requested frame {wanted[target_pos]} in {path}"
                )
            selected.append(frame.to_ndarray(format="rgb24"))
            target_pos += 1
            if target_pos == len(wanted):
                break
    if len(selected) != len(wanted):
        missing = wanted[len(selected) :]
        raise IndexError(f"{path} is missing requested frames {missing[:5]}")
    return selected


def embed_images(
    images: list[np.ndarray],
    *,
    processor: CLIPImageProcessor,
    model: CLIPVisionModel,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for start in range(0, len(images), batch_size):
        pixel_values = processor(
            images=images[start : start + batch_size], return_tensors="pt"
        )["pixel_values"].to(device=device, dtype=dtype)
        with torch.inference_mode():
            pooled = model(pixel_values=pixel_values).pooler_output.float()
            pooled = torch.nn.functional.normalize(pooled, dim=-1)
        chunks.append(pooled.cpu().numpy().astype(np.float16))
    if not chunks:
        return np.zeros((0, int(model.config.hidden_size)), dtype=np.float16)
    return np.concatenate(chunks, axis=0)


def build_action_blocks(
    actions: np.ndarray, starts: np.ndarray, block_size: int
) -> tuple[np.ndarray, np.ndarray]:
    blocks = np.zeros(
        (len(starts), block_size, actions.shape[1]), dtype=np.float32
    )
    valid = np.zeros((len(starts), block_size), dtype=bool)
    for block_index, start in enumerate(starts.tolist()):
        count = min(block_size, len(actions) - int(start))
        if count <= 0:
            continue
        blocks[block_index, :count] = actions[int(start) : int(start) + count]
        valid[block_index, :count] = True
    return blocks, valid


def extract_episode(
    dataset: Path,
    output: Path,
    outcome: dict[str, Any],
    *,
    stride: int,
    processor: CLIPImageProcessor,
    model: CLIPVisionModel,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
) -> dict[str, Any]:
    episode_index = int(outcome["episode_index"])
    parquet = (
        dataset
        / "data"
        / "chunk-000"
        / f"episode_{episode_index:06d}.parquet"
    )
    table = pq.read_table(parquet, columns=["action", "observation.state"])
    actions = np.asarray(table.column("action").to_pylist(), dtype=np.float32)
    states = np.asarray(
        table.column("observation.state").to_pylist(), dtype=np.float32
    )
    if actions.ndim != 2 or actions.shape[1] not in {22, 23}:
        raise ValueError(f"Unexpected action shape {actions.shape} in {parquet}")
    if actions.shape[1] == 23:
        actions = actions[:, 1:]
    if states.ndim != 2 or states.shape[1] < 23:
        raise ValueError(f"Unexpected state shape {states.shape} in {parquet}")
    states = states[:, :23]
    episode_length = min(len(actions), len(states))
    actions = actions[:episode_length]
    states = states[:episode_length]
    frame_indices = np.arange(0, episode_length, stride, dtype=np.int32)
    action_blocks, action_valid = build_action_blocks(
        actions, frame_indices, stride
    )

    visual: dict[str, np.ndarray] = {}
    for camera in CAMERAS:
        video = (
            dataset
            / "videos"
            / "chunk-000"
            / camera
            / f"episode_{episode_index:06d}.mp4"
        )
        frames = read_sampled_video(video, frame_indices)
        visual[camera] = embed_images(
            frames,
            processor=processor,
            model=model,
            device=device,
            dtype=dtype,
            batch_size=batch_size,
        )

    destination = output / "episodes" / f"ep{episode_index:06d}.npz"
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        episode_index=np.asarray(episode_index, dtype=np.int32),
        seed=np.asarray(int(outcome["seed"]), dtype=np.int32),
        success=np.asarray(bool(outcome.get("success")), dtype=bool),
        episode_length=np.asarray(episode_length, dtype=np.int32),
        stride=np.asarray(stride, dtype=np.int32),
        frame_indices=frame_indices,
        states=states[frame_indices],
        action_blocks=action_blocks,
        action_valid=action_valid,
        front_visual=visual[CAMERAS[0]],
        wrist_visual=visual[CAMERAS[1]],
    )
    return {
        "episode_index": episode_index,
        "seed": int(outcome["seed"]),
        "outcome": "success" if bool(outcome.get("success")) else "failure",
        "episode_length": episode_length,
        "num_replans": len(frame_indices),
        "feature_path": str(destination.resolve()),
    }


def parse_episode_filter(raw: str) -> set[int] | None:
    if not raw.strip():
        return None
    return {int(value.strip()) for value in raw.split(",") if value.strip()}


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=24)
    parser.add_argument("--visual-model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--episodes", default="")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    dataset = args.dataset.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if args.stride <= 0:
        raise ValueError("--stride must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    outcomes_path = dataset / "meta" / "episode_outcomes.jsonl"
    outcomes = read_jsonl(outcomes_path)
    episode_filter = parse_episode_filter(args.episodes)
    if episode_filter is not None:
        outcomes = [
            row
            for row in outcomes
            if int(row["episode_index"]) in episode_filter
        ]
        missing = episode_filter - {
            int(row["episode_index"]) for row in outcomes
        }
        if missing:
            raise ValueError(f"Unknown episode indices: {sorted(missing)}")

    device = torch.device(args.device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    processor = CLIPImageProcessor.from_pretrained(
        args.visual_model, local_files_only=True
    )
    model = CLIPVisionModel.from_pretrained(
        args.visual_model, local_files_only=True
    ).eval()
    model.to(device=device, dtype=dtype)

    output.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for position, outcome in enumerate(outcomes, start=1):
        episode_index = int(outcome["episode_index"])
        destination = output / "episodes" / f"ep{episode_index:06d}.npz"
        if destination.exists() and not args.overwrite:
            with np.load(destination) as cached:
                records.append(
                    {
                        "episode_index": episode_index,
                        "seed": int(np.asarray(cached["seed"]).item()),
                        "outcome": (
                            "success"
                            if bool(np.asarray(cached["success"]).item())
                            else "failure"
                        ),
                        "episode_length": int(
                            np.asarray(cached["episode_length"]).item()
                        ),
                        "num_replans": len(cached["frame_indices"]),
                        "feature_path": str(destination.resolve()),
                        "cached": True,
                    }
                )
                print(
                    f"[{position}/{len(outcomes)}] cached ep={episode_index}",
                    flush=True,
                )
            continue
        record = extract_episode(
            dataset,
            output,
            outcome,
            stride=int(args.stride),
            processor=processor,
            model=model,
            device=device,
            dtype=dtype,
            batch_size=int(args.batch_size),
        )
        records.append(record)
        print(
            f"[{position}/{len(outcomes)}] ep={episode_index} "
            f"seed={record['seed']} {record['outcome']} "
            f"replans={record['num_replans']}",
            flush=True,
        )

    records.sort(key=lambda row: int(row["episode_index"]))
    index_path = output / "episode_features.jsonl"
    index_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    config = {
        "format": "FoldGlassesSeedPairReplanFeatures",
        "version": "1.0",
        "dataset": str(dataset),
        "outcomes_sha256": sha256_file(outcomes_path),
        "stride": int(args.stride),
        "executed_action_block": int(args.stride),
        "visual_model": str(args.visual_model),
        "visual_feature": "L2-normalized CLIP vision pooler output per camera",
        "phase_features_are_action_free": True,
        "num_episodes": len(records),
    }
    (output / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(output), **config}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
