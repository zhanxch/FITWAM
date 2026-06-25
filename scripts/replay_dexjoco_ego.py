#!/usr/bin/env python3
"""Open-loop DexJoCo sim replay for the merged ``dexjoco_ego`` LeRobot dataset.

Mirrors the evaluation loop in ``dexjoco-openpi-eval``: create a task env with
``policy_mode=True``, ``env.reset()``, then execute actions step-by-step and
record fresh simulator videos. Actions are loaded from dataset parquet (rotvec
44/22-dim), not from an OpenPI policy server and not from pre-encoded MP4s.

Use ``scripts/run_dexjoco_ego_sim_replay.sh`` for the full-dataset batch test.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import zarr
from tqdm import tqdm

os.environ.setdefault("MUJOCO_GL", "egl")

from dexjoco.data.video_writer import Mp4VideoWriter
from dexjoco.tasks.mappings import CONFIG_MAPPING
from dexjoco.tasks.state_restorers import has_restorer, restore_initial_state

DEFAULT_DATASET_DIR = Path("data/dexjoco_ego")
DEFAULT_OUTPUT_DIR = Path("data/dexjoco_ego/sim_replay")
CLICK_MOUSE_ALIGN_ROTVEC = np.array(
    [
        -4.4294e-01,
        1.3729e-06,
        1.5170e00,
        -3.14156462e00,
        -6.91584035e-05,
        -1.40317984e-03,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0.263,
        0,
        0,
        0,
    ],
    dtype=np.float64,
)
CLICK_MOUSE_ALIGN_STEPS = 30
DEFAULT_ZARR_ROOT = Path("third_party/dexjoco/datasets/raw/dexjoco_raw_datasets")
CHUNKS_SIZE = 1000

# Must match the merge order in scripts/prepare_dexjoco_ego_dataset.py.
TASK_NAMES: tuple[str, ...] = (
    "bimanual_assembly",
    "bimanual_hanoi",
    "bimanual_microwave_cook",
    "bimanual_photograph",
    "bimanual_unlock_ipad",
    "click_mouse",
    "fold_glasses",
    "hammer_nail",
    "pick_bucket",
    "pinch_tongs",
    "water_plant",
)
EPISODES_PER_TASK = 100

# Primary camera key in sim when randomize=False (matches dexjoco_ego ego view).
TASK_PRIMARY_CAMERA: dict[str, str] = {
    "bimanual_assembly": "ego",
    "bimanual_hanoi": "ego",
    "bimanual_microwave_cook": "ego",
    "bimanual_photograph": "ego",
    "bimanual_unlock_ipad": "ego",
    "click_mouse": "ego_right",
    "fold_glasses": "front",
    "hammer_nail": "front",
    "pick_bucket": "front",
    "pinch_tongs": "front",
    "water_plant": "front",
}


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _episode_parquet_path(dataset_dir: Path, episode_index: int, chunks_size: int) -> Path:
    chunk = episode_index // chunks_size
    return dataset_dir / "data" / f"chunk-{chunk:03d}" / f"episode_{episode_index:06d}.parquet"


def _resolve_task(episode_index: int, *, task_name: str | None = None) -> tuple[str, int]:
    if task_name is not None:
        if task_name not in TASK_NAMES:
            raise ValueError(f"Unknown task_name={task_name!r}; expected one of {TASK_NAMES}")
        return task_name, episode_index
    task_index = episode_index // EPISODES_PER_TASK
    if task_index >= len(TASK_NAMES):
        raise ValueError(
            f"Episode {episode_index} is out of range for {len(TASK_NAMES)} tasks "
            f"with {EPISODES_PER_TASK} episodes each."
        )
    local_index = episode_index % EPISODES_PER_TASK
    return TASK_NAMES[task_index], local_index


def _is_bimanual(task_name: str) -> bool:
    return task_name.startswith("bimanual_")


def _rotvec_to_quat_wxyz(rotvec: np.ndarray) -> np.ndarray:
    """Convert axis-angle rotvec to MuJoCo/DexJoCo wxyz quaternion."""
    rotvec = np.asarray(rotvec, dtype=np.float64)
    angle = np.linalg.norm(rotvec)
    if angle < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    axis = rotvec / angle
    half = 0.5 * angle
    sin_half = np.sin(half)
    return np.array(
        [np.cos(half), axis[0] * sin_half, axis[1] * sin_half, axis[2] * sin_half],
        dtype=np.float64,
    )


def _load_episode_actions(parquet_path: Path, *, dual_arm: bool) -> np.ndarray:
    table = pq.read_table(parquet_path)
    actions = np.asarray([row.as_py() for row in table["action"]], dtype=np.float64)
    if dual_arm:
        return actions[:, :44]
    return actions[:, :22]


def _rotvec_action_to_policy_quat(action_rotvec: np.ndarray, *, dual_arm: bool) -> np.ndarray:
    action_rotvec = np.asarray(action_rotvec, dtype=np.float64)
    if dual_arm:
        r_xyz = action_rotvec[:3]
        r_rotvec = action_rotvec[3:6]
        r_hand = action_rotvec[6:22]
        l_xyz = action_rotvec[22:25]
        l_rotvec = action_rotvec[25:28]
        l_hand = action_rotvec[28:44]
        r_quat = _rotvec_to_quat_wxyz(r_rotvec)
        l_quat = _rotvec_to_quat_wxyz(l_rotvec)
        return np.concatenate([r_xyz, r_quat, l_xyz, l_quat, r_hand, l_hand])

    xyz = action_rotvec[:3]
    rotvec = action_rotvec[3:6]
    hand = action_rotvec[6:22]
    quat = _rotvec_to_quat_wxyz(rotvec)
    return np.concatenate([xyz, quat, hand])


def _find_zarr_path(zarr_root: Path, task_name: str, local_episode_index: int) -> Path | None:
    task_dir = zarr_root / task_name
    if not task_dir.exists():
        return None

    direct = task_dir / f"episode_{local_episode_index:06d}" / "replay.zarr"
    if direct.exists():
        return direct

    demo_dirs = sorted(p.parent for p in task_dir.glob("*/replay.zarr"))
    if local_episode_index < len(demo_dirs):
        return demo_dirs[local_episode_index] / "replay.zarr"
    return None


def _load_initial_state_from_zarr(zarr_path: Path) -> np.ndarray | None:
    root = zarr.open(str(zarr_path), mode="r")
    if "state" not in root["data"]:
        return None
    return np.asarray(root["data"]["state"][0]).ravel()


def _safe_squeeze_image(img: np.ndarray) -> np.ndarray:
    if img is None:
        return None
    arr = np.asarray(img)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = np.concatenate([arr, arr, arr], axis=2)
    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating):
            if np.nanmax(arr) <= 1.0:
                arr = np.clip(arr, 0.0, 1.0) * 255.0
            else:
                arr = np.clip(arr, 0.0, 255.0)
            arr = arr.astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)
    return arr


def _click_mouse_align(env, *, dual_arm: bool) -> None:
    """Match the warmup used in ``eval_dexjoco_openpi.py`` for click_mouse."""
    align_action = _rotvec_action_to_policy_quat(CLICK_MOUSE_ALIGN_ROTVEC, dual_arm=dual_arm)
    for _ in range(CLICK_MOUSE_ALIGN_STEPS):
        env.step(align_action)


def _episode_output_path(
    output_dir: Path,
    *,
    task_name: str,
    local_episode_index: int,
    global_episode_index: int,
    camera_key: str,
    group_by_task: bool,
) -> Path:
    if group_by_task:
        return (
            output_dir
            / task_name
            / f"episode_{local_episode_index:06d}_{camera_key}.mp4"
        )
    return output_dir / f"episode_{global_episode_index:06d}_{camera_key}.mp4"


def _write_replay_video(
    trajectory: list[dict[str, Any]],
    *,
    output_path: Path,
    camera_key: str,
    video_fps: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = Mp4VideoWriter.create_h264(
        fps=video_fps,
        codec="h264",
        input_pix_fmt="rgb24",
        crf=21,
        thread_type="FRAME",
        thread_count=2,
    )
    writer.start(str(output_path))
    for step in trajectory:
        img = step["observations"].get(camera_key)
        if img is None:
            continue
        writer.write_frame(_safe_squeeze_image(img))
    writer.stop()


def replay_episode(
    dataset_dir: Path,
    episode_index: int,
    *,
    output_dir: Path,
    zarr_root: Path | None,
    chunks_size: int,
    randomize: bool,
    restore_state: bool,
    seed: int,
    video_fps: int,
    save_failed: bool,
    group_by_task: bool,
    skip_existing: bool,
    camera_screen_effect: bool,
    ipad_screen_effect: bool,
    task_name_override: str | None = None,
) -> dict[str, Any] | None:
    task_name, local_episode_index = _resolve_task(
        episode_index, task_name=task_name_override
    )
    dual_arm = _is_bimanual(task_name)
    parquet_path = _episode_parquet_path(dataset_dir, episode_index, chunks_size)
    if not parquet_path.exists():
        print(f"[skip] missing parquet for episode {episode_index}")
        return None

    actions_rotvec = _load_episode_actions(parquet_path, dual_arm=dual_arm)
    policy_actions = np.stack(
        [_rotvec_action_to_policy_quat(a, dual_arm=dual_arm) for a in actions_rotvec],
        axis=0,
    )

    camera_key = "random_camera" if randomize else TASK_PRIMARY_CAMERA[task_name]
    output_path = _episode_output_path(
        output_dir,
        task_name=task_name,
        local_episode_index=local_episode_index,
        global_episode_index=episode_index,
        camera_key=camera_key,
        group_by_task=group_by_task,
    )
    if skip_existing and output_path.exists():
        return {
            "episode_index": episode_index,
            "task_name": task_name,
            "local_episode_index": local_episode_index,
            "output_path": str(output_path),
            "num_steps": None,
            "succeed": None,
            "skipped_existing": True,
        }

    initial_state = None
    if restore_state and zarr_root is not None:
        zarr_path = _find_zarr_path(zarr_root, task_name, local_episode_index)
        if zarr_path is None:
            print(
                f"[warn] episode {episode_index}: no raw zarr for {task_name} "
                f"local={local_episode_index}; scene will use reset() only."
            )
        else:
            initial_state = _load_initial_state_from_zarr(zarr_path)
            if initial_state is None:
                print(f"[warn] episode {episode_index}: zarr has no state[0] at {zarr_path}")

    config = CONFIG_MAPPING[task_name]()
    env_extra_kwargs: dict[str, Any] = {}
    if task_name == "bimanual_photograph":
        env_extra_kwargs["camera_screen_effect"] = camera_screen_effect
    if task_name == "bimanual_unlock_ipad":
        env_extra_kwargs["ipad_screen_effect"] = ipad_screen_effect

    env = config.get_environment(
        policy_mode=True,
        render_mode="rgb_array",
        randomize=randomize,
        randomize_dynamics=False,
        seed=seed + episode_index,
        **env_extra_kwargs,
    )

    trajectory: list[dict[str, Any]] = []
    succeed = False

    try:
        obs, _info = env.reset()
        if restore_state and initial_state is not None and has_restorer(task_name):
            obs = restore_initial_state(env, task_name, config, initial_state)

        if task_name == "click_mouse":
            _click_mouse_align(env, dual_arm=dual_arm)
            obs = env.observation(env.unwrapped._compute_observation())

        for action in tqdm(policy_actions, desc=f"ep {episode_index:06d} ({task_name})", leave=False):
            next_obs, _rew, done, _trunc, info = env.step(action)
            trajectory.append(copy.deepcopy({"observations": obs, "actions": action, "infos": info}))
            obs = next_obs
            if info.get("succeed", False):
                succeed = True
    finally:
        try:
            env.close()
        except Exception:
            pass

    if not trajectory:
        return None
    if not (succeed or save_failed):
        print(f"[skip] episode {episode_index}: sim did not report success")
        return {
            "episode_index": episode_index,
            "task_name": task_name,
            "local_episode_index": local_episode_index,
            "output_path": None,
            "num_steps": len(trajectory),
            "succeed": succeed,
            "skipped_existing": False,
        }

    _write_replay_video(
        trajectory,
        output_path=output_path,
        camera_key=camera_key,
        video_fps=video_fps,
    )
    print(f"[saved] {output_path} (steps={len(trajectory)}, succeed={succeed})")
    return {
        "episode_index": episode_index,
        "task_name": task_name,
        "local_episode_index": local_episode_index,
        "output_path": str(output_path),
        "num_steps": len(trajectory),
        "succeed": succeed,
        "skipped_existing": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help="Merged LeRobot dataset root (default: data/dexjoco_ego).",
    )
    parser.add_argument(
        "--task-name",
        type=str,
        default=None,
        help="Fixed task for single-task datasets (e.g. bimanual_microwave_cook). "
        "Episode indices are treated as local indices within that task.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for sim-rendered replay videos.",
    )
    parser.add_argument(
        "--zarr-root",
        type=Path,
        default=DEFAULT_ZARR_ROOT,
        help="Root containing per-task raw Zarr demos for scene restore "
        "(e.g. third_party/dexjoco/datasets/raw/dexjoco_raw_datasets).",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        nargs="*",
        default=None,
        help="Episode indices to replay. Default: all episodes in meta/episodes.jsonl.",
    )
    parser.add_argument("--max-episodes", type=int, default=None, help="Replay at most N episodes.")
    parser.add_argument("--seed", type=int, default=0, help="Base env seed; episode index is added.")
    parser.add_argument("--video-fps", type=int, default=30, help="Output video FPS.")
    parser.add_argument(
        "--randomize",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable rand_full visual randomization at reset (default: false).",
    )
    parser.add_argument(
        "--restore-state",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Optionally restore object poses from raw Zarr state[0] (default: false).",
    )
    parser.add_argument(
        "--save-failed",
        "--save-all",
        action="store_true",
        dest="save_failed",
        help="Save replay video even when sim does not report success (default for batch test).",
    )
    parser.add_argument(
        "--group-by-task",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write videos under output_dir/<task_name>/ (default: true).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip episodes whose output MP4 already exists.",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=None,
        help="JSON summary path (default: <output-dir>/summary.json).",
    )
    parser.add_argument("--camera-screen-effect", action="store_true", help="bimanual_photograph only.")
    parser.add_argument("--ipad-screen-effect", action="store_true", help="bimanual_unlock_ipad only.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    meta_dir = dataset_dir / "meta"
    if not meta_dir.exists():
        raise FileNotFoundError(f"Missing dataset meta directory: {meta_dir}")

    info = _read_json(meta_dir / "info.json")
    chunks_size = int(info.get("chunks_size", CHUNKS_SIZE))

    if args.episodes is None:
        with (meta_dir / "episodes.jsonl").open("r", encoding="utf-8") as f:
            episode_indices = [int(json.loads(line)["episode_index"]) for line in f if line.strip()]
    else:
        episode_indices = list(args.episodes)
    if args.max_episodes is not None:
        episode_indices = episode_indices[: args.max_episodes]

    zarr_root = None
    if args.restore_state:
        zarr_root = args.zarr_root.resolve()
        if not zarr_root.exists():
            print(
                f"[warn] --restore-state requested but --zarr-root not found ({zarr_root}); "
                "falling back to env.reset() only."
            )
            zarr_root = None

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = (args.summary_path or output_dir / "summary.json").resolve()

    results: list[dict[str, Any]] = []
    saved_count = 0
    for episode_index in tqdm(episode_indices, desc="Sim replay"):
        result = replay_episode(
            dataset_dir,
            episode_index,
            output_dir=output_dir,
            zarr_root=zarr_root,
            chunks_size=chunks_size,
            randomize=args.randomize,
            restore_state=args.restore_state,
            seed=args.seed,
            video_fps=args.video_fps,
            save_failed=args.save_failed,
            group_by_task=args.group_by_task,
            skip_existing=args.skip_existing,
            camera_screen_effect=args.camera_screen_effect,
            ipad_screen_effect=args.ipad_screen_effect,
            task_name_override=args.task_name,
        )
        if result is None:
            continue
        results.append(result)
        if result.get("output_path") and not result.get("skipped_existing"):
            saved_count += 1

    task_stats: dict[str, dict[str, float]] = {}
    task_names_for_stats = (args.task_name,) if args.task_name else TASK_NAMES
    for task_name in task_names_for_stats:
        task_rows = [r for r in results if r["task_name"] == task_name and not r.get("skipped_existing")]
        evaluated = [r for r in task_rows if r.get("succeed") is not None]
        if not evaluated:
            continue
        task_stats[task_name] = {
            "episodes": len(evaluated),
            "saved": sum(1 for r in evaluated if r.get("output_path")),
            "successes": sum(1 for r in evaluated if r.get("succeed")),
            "success_rate": sum(1 for r in evaluated if r.get("succeed")) / len(evaluated),
        }

    evaluated = [r for r in results if r.get("succeed") is not None and not r.get("skipped_existing")]
    overall_successes = sum(1 for r in evaluated if r.get("succeed"))
    overall_success_rate = overall_successes / len(evaluated) if evaluated else 0.0

    summary = {
        "dataset_dir": str(dataset_dir),
        "task_name": args.task_name,
        "output_dir": str(output_dir),
        "total_episodes": len(episode_indices),
        "processed": len(results),
        "saved_videos": saved_count,
        "randomize": args.randomize,
        "restore_state": args.restore_state,
        "overall_successes": overall_successes,
        "overall_success_rate": overall_success_rate,
        "task_stats": task_stats,
        "episodes": results,
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"\nSim replay finished: saved {saved_count} videos -> {output_dir}")
    print(f"Summary: {summary_path}")
    if evaluated:
        print(
            f"Overall success rate: {overall_success_rate:.1%} "
            f"({overall_successes}/{len(evaluated)})"
        )
    for task_name, stats in task_stats.items():
        print(
            f"  {task_name}: success_rate={stats['success_rate']:.1%} "
            f"({int(stats['successes'])}/{int(stats['episodes'])})"
        )


if __name__ == "__main__":
    main()
