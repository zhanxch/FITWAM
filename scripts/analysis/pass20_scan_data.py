"""Load Pass@20 action-chunk scans and locate recoverability events."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.analysis.pass20_future_metrics import mean_pairwise_rms
from scripts.fold_glasses.discover_seedpair_branch_events import normalize_actions

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATS = (
    ROOT.parent
    / "FastWAM-infer-in-DexJoco"
    / "artifacts"
    / "fold_glasses"
    / "dataset_stats.json"
)


def load_chunks(scan_root: Path) -> dict[str, np.ndarray]:
    records: list[dict[str, Any]] = []
    for traj_path in sorted(scan_root.glob("shard*/prefixes/*/replicate_*/trajectory.json")):
        payload = json.loads(traj_path.read_text(encoding="utf-8"))
        ledger = payload.get("ledger") or {}
        npz_path = traj_path.with_name("action_chunk.npz")
        if not npz_path.is_file():
            raw = ledger.get("action_chunk_arrays")
            npz_path = Path(raw) if raw else npz_path
        if not npz_path.is_file():
            continue
        arrays = np.load(npz_path)
        chunk = np.asarray(arrays["first_action_chunk"], dtype=np.float32)
        if chunk.shape != (32, 22):
            continue
        records.append(
            {
                "episode_index": int(ledger["episode_index"]),
                "prefix_frame": int(ledger["prefix_frame"]),
                "success": bool(ledger["success"]),
                "chunk": chunk,
            }
        )
    if not records:
        raise SystemExit(f"No action chunks under {scan_root}")
    return {
        "episode_index": np.asarray([row["episode_index"] for row in records], dtype=np.int32),
        "prefix_frame": np.asarray([row["prefix_frame"] for row in records], dtype=np.int32),
        "success": np.asarray([row["success"] for row in records], dtype=bool),
        "chunk": np.stack([row["chunk"] for row in records], axis=0),
    }


def node_metrics(
    *,
    chunks: np.ndarray,
    episode_index: np.ndarray,
    prefix_frame: np.ndarray,
    success: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    min_replicates: int,
) -> dict[str, np.ndarray]:
    z = normalize_actions(chunks, mean, std)
    episodes: list[int] = []
    frames: list[int] = []
    spread: list[float] = []
    pass_rate: list[float] = []
    keys = np.unique(
        np.stack([episode_index.astype(np.int64), prefix_frame.astype(np.int64)], axis=1),
        axis=0,
    )
    for ep, frame in keys:
        idx = np.where((episode_index == ep) & (prefix_frame == frame))[0]
        if idx.size < min_replicates:
            continue
        episodes.append(int(ep))
        frames.append(int(frame))
        spread.append(mean_pairwise_rms(z[idx]))
        pass_rate.append(float(np.mean(success[idx])))
    return {
        "episode": np.asarray(episodes, dtype=np.int32),
        "frame": np.asarray(frames, dtype=np.int32),
        "spread": np.asarray(spread, dtype=np.float64),
        "pass_rate": np.asarray(pass_rate, dtype=np.float64),
    }


def sorted_nodes(
    frames: np.ndarray, pass_rate: np.ndarray, spread: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(frames)
    return (
        frames[order].astype(np.int32),
        pass_rate[order].astype(np.float64),
        spread[order].astype(np.float64),
    )


def first_pass_zero(frames: np.ndarray, pass_rate: np.ndarray) -> int | None:
    for t, rate in zip(frames.tolist(), pass_rate.tolist()):
        if rate <= 0.0:
            return int(t)
    return None


def last_recoverable_before(
    frames: np.ndarray, pass_rate: np.ndarray, cutoff: int
) -> int | None:
    last: int | None = None
    for t, rate in zip(frames.tolist(), pass_rate.tolist()):
        if int(t) >= int(cutoff):
            break
        if rate > 0.0:
            last = int(t)
    return last
