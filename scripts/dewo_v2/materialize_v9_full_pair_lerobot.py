#!/usr/bin/env python3
"""Materialize full-horizon recoverability pair LeRobot for DEWO v9.

D+: stitch original fail prefix ``[0, t)`` with the counterfactual success
continuation that starts at ``t``. D_fail: factual fail cliff
``[t, min(len, M+24))``, expanded to at least 33 frames if needed.
Shared prefix lives only on the success episode (except the cliff-start
frame ``t``, which is the first fail frame with ``G=0``).

Do not overwrite the 33-frame ``pair_lerobot`` used by v6/v7/v8.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import av
import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.collect_dexjoco_rollouts import (  # noqa: E402
    aggregate_stats,
    append_jsonl,
    compute_episode_stats,
    prepare_dataset,
    read_json,
    save_episode_video,
    serialize_dict,
    update_info,
    write_episode_parquet,
    write_json,
)
from v9_pair_geometry import (  # noqa: E402
    MIN_EVENT_FRAMES,
    fail_cliff_span,
    stitch_prefix_plus_continuation,
)

FAILURE_PHRASE = "Failed to finish the whole process."
DEFAULT_SUCCESS_PROMPT = "Grasp the watering can and apply water to the plant."


def load_rgb_frames(path: Path) -> list[np.ndarray]:
    container = av.open(str(path))
    try:
        frames = [
            frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)
        ]
    finally:
        container.close()
    if not frames:
        raise ValueError(f"empty video: {path}")
    return frames


def _column_as_array(table: Any, name: str) -> np.ndarray:
    return np.asarray(table.column(name).to_pylist(), dtype=np.float32)


def load_episode_arrays(
    dataset: Path,
    info: dict[str, Any],
    episode_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    chunk = int(episode_index) // int(info["chunks_size"])
    path = dataset / info["data_path"].format(
        episode_chunk=chunk,
        episode_index=int(episode_index),
    )
    table = pq.read_table(path)
    return _column_as_array(table, "action"), _column_as_array(
        table, "observation.state"
    )


def video_path(dataset: Path, info: dict[str, Any], episode_index: int, video_key: str) -> Path:
    chunk = int(episode_index) // int(info["chunks_size"])
    return dataset / info["video_path"].format(
        episode_chunk=chunk,
        video_key=video_key,
        episode_index=int(episode_index),
    )


def save_variable_episode(
    *,
    output_dataset: Path,
    info: dict[str, Any],
    stats_list: list[dict[str, dict]],
    episode_index: int,
    global_start_index: int,
    actions: np.ndarray,
    states: np.ndarray,
    front: list[np.ndarray],
    wrist: list[np.ndarray],
    task_text: str,
    fps: int,
) -> int:
    length = int(actions.shape[0])
    if length < MIN_EVENT_FRAMES:
        raise ValueError(f"episode {episode_index} length {length} < {MIN_EVENT_FRAMES}")
    if states.shape[0] != length:
        raise ValueError(
            f"episode {episode_index} state/action length mismatch: "
            f"{states.shape[0]} vs {length}"
        )
    if len(front) != length or len(wrist) != length:
        raise ValueError(
            f"episode {episode_index} video length mismatch: "
            f"front={len(front)} wrist={len(wrist)} actions={length}"
        )
    timestamps = np.arange(length, dtype=np.float32) / float(fps)
    write_episode_parquet(
        output_dataset
        / info["data_path"].format(
            episode_chunk=episode_index // int(info["chunks_size"]),
            episode_index=episode_index,
        ),
        actions=np.asarray(actions, dtype=np.float32),
        states=np.asarray(states, dtype=np.float32),
        timestamps=timestamps,
        frame_indices=np.arange(length, dtype=np.int64),
        episode_indices=np.full((length,), episode_index, dtype=np.int64),
        global_indices=np.arange(
            global_start_index, global_start_index + length, dtype=np.int64
        ),
        task_indices=np.zeros((length,), dtype=np.int64),
    )
    for video_key, frames in (
        ("observation.images.front", front),
        ("observation.images.wrist", wrist),
    ):
        save_episode_video(
            frames,
            video_path(output_dataset, info, episode_index, video_key),
            fps,
        )
    append_jsonl(
        output_dataset / "meta" / "episodes.jsonl",
        {"episode_index": episode_index, "tasks": [task_text], "length": length},
    )
    ep_stats = compute_episode_stats(
        {
            "action": np.asarray(actions, dtype=np.float32),
            "observation.state": np.asarray(states, dtype=np.float32),
            "timestamp": timestamps,
            "frame_index": np.arange(length, dtype=np.int64),
            "episode_index": np.full((length,), episode_index, dtype=np.int64),
            "index": np.arange(
                global_start_index, global_start_index + length, dtype=np.int64
            ),
            "task_index": np.zeros((length,), dtype=np.int64),
        },
        info["features"],
        is_compute_episode_stats_image=False,
    )
    stats_list.append(ep_stats)
    append_jsonl(
        output_dataset / "meta" / "episodes_stats.jsonl",
        {"episode_index": episode_index, "stats": serialize_dict(ep_stats)},
    )
    return length


def _continuation_start_frame(npz: Any, t_star: int) -> int:
    if "global_frames" in npz.files:
        frames = np.asarray(npz["global_frames"])
        if frames.size:
            return int(frames.reshape(-1)[0])
    return int(t_star)


def _success_branch(pair: dict[str, Any]) -> dict[str, Any]:
    branch = pair.get("success_branch")
    if isinstance(branch, dict):
        return branch
    return pair


def materialize_pair(
    pair: dict[str, Any],
    *,
    source_dataset: Path,
    source_info: dict[str, Any],
    output_dataset: Path,
    info: dict[str, Any],
    stats_list: list[dict[str, dict]],
    episode_index: int,
    global_start_index: int,
    success_prompt: str,
    fps: int,
) -> tuple[int, int, dict[str, Any], int]:
    t_star = int(pair["t_star_last_recoverable"])
    m_first = int(pair["M_first_zero"])
    fail_ep = int(
        pair.get("source_failure_episode_index", pair.get("failure_episode_index"))
    )
    succ = _success_branch(pair)
    orig_actions, orig_states = load_episode_arrays(source_dataset, source_info, fail_ep)
    fail_len = int(orig_actions.shape[0])
    orig_front = load_rgb_frames(
        video_path(source_dataset, source_info, fail_ep, "observation.images.front")
    )
    orig_wrist = load_rgb_frames(
        video_path(source_dataset, source_info, fail_ep, "observation.images.wrist")
    )
    n_vid = min(len(orig_front), len(orig_wrist), fail_len)
    orig_actions = orig_actions[:n_vid]
    orig_states = orig_states[:n_vid]
    orig_front = orig_front[:n_vid]
    orig_wrist = orig_wrist[:n_vid]
    fail_len = n_vid

    cont_npz_path = Path(
        succ.get("continuation_arrays")
        or pair.get("success_continuation_npz")
        or ""
    )
    if not cont_npz_path:
        raise ValueError(f"{pair.get('pair_id')}: missing continuation npz")
    cont = np.load(cont_npz_path)
    cont_start = _continuation_start_frame(cont, t_star)
    expected_start = int(succ.get("start_frame", t_star))
    if cont_start != expected_start:
        raise ValueError(
            f"{pair.get('pair_id')}: continuation starts at {cont_start}, "
            f"expected t={expected_start}"
        )
    prefix = orig_actions[:t_star]
    success_actions = stitch_prefix_plus_continuation(prefix, np.asarray(cont["actions"]))
    success_states = stitch_prefix_plus_continuation(
        orig_states[:t_star], np.asarray(cont["states"])
    )
    cont_front = load_rgb_frames(
        Path(succ.get("continuation_front") or pair["success_continuation_front"])
    )
    cont_wrist = load_rgb_frames(
        Path(succ.get("continuation_wrist") or pair["success_continuation_wrist"])
    )
    success_front = orig_front[:t_star] + cont_front
    success_wrist = orig_wrist[:t_star] + cont_wrist
    n_succ = int(success_actions.shape[0])
    success_front = success_front[:n_succ]
    success_wrist = success_wrist[:n_succ]
    success_states = success_states[:n_succ]
    if len(success_front) != n_succ or len(success_wrist) != n_succ:
        raise ValueError(
            f"{pair.get('pair_id')}: stitched success video/action mismatch "
            f"front={len(success_front)} wrist={len(success_wrist)} act={n_succ}"
        )

    success_ep = episode_index
    n_written = save_variable_episode(
        output_dataset=output_dataset,
        info=info,
        stats_list=stats_list,
        episode_index=success_ep,
        global_start_index=global_start_index,
        actions=success_actions,
        states=success_states,
        front=success_front,
        wrist=success_wrist,
        task_text=success_prompt,
        fps=fps,
    )
    global_start_index += n_written
    episode_index += 1

    lo, hi = fail_cliff_span(t_star, m_first, fail_len)
    fail_actions = orig_actions[lo:hi]
    fail_states = orig_states[lo:hi]
    fail_front = orig_front[lo:hi]
    fail_wrist = orig_wrist[lo:hi]
    failure_ep = episode_index
    n_fail = save_variable_episode(
        output_dataset=output_dataset,
        info=info,
        stats_list=stats_list,
        episode_index=failure_ep,
        global_start_index=global_start_index,
        actions=fail_actions,
        states=fail_states,
        front=fail_front,
        wrist=fail_wrist,
        task_text=success_prompt,
        fps=fps,
    )
    record = {
        "pair_id": pair["pair_id"],
        "seed": pair.get("seed"),
        "success_episode_index": success_ep,
        "failure_episode_index": failure_ep,
        "t_star_last_recoverable": t_star,
        "M_first_zero": m_first,
        "fail_cliff": [lo, hi],
        "success_length": n_written,
        "failure_length": n_fail,
        "source_failure_episode_index": fail_ep,
        "source_window_rule": "recoverability_pair_full",
    }
    return episode_index + 1, global_start_index + n_fail, record, n_written + n_fail


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--critic-index", type=Path, required=True)
    parser.add_argument("--source-dataset", type=Path, required=True)
    parser.add_argument("--output-dataset", type=Path, required=True)
    parser.add_argument("--success-prompt", default=DEFAULT_SUCCESS_PROMPT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    success_prompt = str(args.success_prompt)

    critic = read_json(args.critic_index.expanduser().resolve())
    pairs = list(critic.get("full_horizon_pairs") or [])
    if not pairs:
        raise SystemExit(f"No full_horizon_pairs in {args.critic_index}")

    source = args.source_dataset.expanduser().resolve()
    source_info = read_json(source / "meta" / "info.json")
    info, n_ep, global_i, stats_list, _attempts = prepare_dataset(
        source,
        args.output_dataset.expanduser().resolve(),
        success_prompt,
        f"{success_prompt} {FAILURE_PHRASE}",
        overwrite=bool(args.overwrite),
        resume=False,
        save_all_trajectories=True,
        outcome_task_mode="clean",
    )
    fps = int(info.get("fps", 30))
    output = args.output_dataset.expanduser().resolve()
    pair_index: list[dict[str, Any]] = []
    for pair in pairs:
        n_ep, global_i, record, _n = materialize_pair(
            pair,
            source_dataset=source,
            source_info=source_info,
            output_dataset=output,
            info=info,
            stats_list=stats_list,
            episode_index=n_ep,
            global_start_index=global_i,
            success_prompt=success_prompt,
            fps=fps,
        )
        append_jsonl(
            output / "meta" / "episode_outcomes.jsonl",
            {
                "episode_index": record["success_episode_index"],
                "success": True,
                "outcome": "success",
                "pair_id": record["pair_id"],
                "event_role": "success_event",
                "seed": record.get("seed"),
                "source_failure_episode_index": record["source_failure_episode_index"],
            },
        )
        append_jsonl(
            output / "meta" / "episode_outcomes.jsonl",
            {
                "episode_index": record["failure_episode_index"],
                "success": False,
                "outcome": "failure",
                "pair_id": record["pair_id"],
                "event_role": "failure_event",
                "seed": record.get("seed"),
                "source_failure_episode_index": record["source_failure_episode_index"],
            },
        )
        pair_index.append(record)
        update_info(
            output,
            info,
            num_episodes=n_ep,
            total_frames=global_i,
            total_tasks=1,
        )

    write_json(output / "meta" / "stats.json", serialize_dict(aggregate_stats(stats_list)))
    write_json(
        output / "pair_index.json",
        {
            "pairs": pair_index,
            "num_pairs": len(pair_index),
            "horizon": "full",
            "source_window_rule": "recoverability_pair_full",
            "critic_index": str(args.critic_index.expanduser().resolve()),
        },
    )
    print(
        f"wrote {n_ep} episodes ({len(pair_index)} pairs) to {output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
