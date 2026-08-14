#!/usr/bin/env python3
"""Validate deterministic replay of a recorded Fold Glasses attempt.

The collector reused one environment for all repeats of a seed. This validator
recreates that reset sequence, replays the recorded 22-D FastWAM actions through
the same robot-action conversion, and checks selected observations plus the final
outcome. A counterfactual action-block intervention is invalid unless this
factual control first reproduces the source attempt.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import av
import numpy as np
import pyarrow.parquet as pq
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
OPEN = Path(
    os.environ.get(
        "FASTWAM_OPEN_REPO", "/data_all/xiangchengzhan/FastWAM-infer-in-DexJoco"
    )
)
DEXJOCO = ROOT / "third_party" / "dexjoco" / "dexjoco"


def setup_paths() -> None:
    for path in (OPEN / "src", ROOT / "src", DEXJOCO):
        value = str(path)
        if value in sys.path:
            sys.path.remove(value)
        sys.path.insert(0, value)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def parse_ints(raw: str) -> list[int]:
    return sorted({int(value.strip()) for value in raw.split(",") if value.strip()})


def attempt_for_episode(dataset: Path, episode_index: int) -> dict[str, Any]:
    summary = read_json(dataset / "collection_summary.json")
    matches = [
        row
        for row in summary.get("attempt_log", [])
        if int(row.get("saved_episode_index", -1)) == episode_index
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one attempt for episode {episode_index}, found {len(matches)}"
        )
    attempt = dict(matches[0])
    required = {"seed", "repeat", "success", "saved_episode_index"}
    missing = required - set(attempt)
    if missing:
        raise ValueError(f"Attempt mapping is missing {sorted(missing)}")
    return attempt


def load_episode(dataset: Path, episode_index: int) -> tuple[np.ndarray, np.ndarray]:
    path = (
        dataset
        / "data"
        / "chunk-000"
        / f"episode_{episode_index:06d}.parquet"
    )
    table = pq.read_table(path, columns=["action", "observation.state"])
    actions = np.asarray(table.column("action").to_pylist(), dtype=np.float32)
    states = np.asarray(
        table.column("observation.state").to_pylist(), dtype=np.float32
    )
    if actions.ndim != 2 or actions.shape[1] not in {22, 23}:
        raise ValueError(f"Unexpected action shape {actions.shape}")
    if actions.shape[1] == 23:
        actions = actions[:, 1:]
    if states.ndim != 2 or states.shape[1] < 23:
        raise ValueError(f"Unexpected state shape {states.shape}")
    length = min(len(actions), len(states))
    return actions[:length], states[:length, :23]


def read_video_frames(path: Path, frame_indices: Sequence[int]) -> dict[int, np.ndarray]:
    wanted = sorted(set(int(value) for value in frame_indices))
    output: dict[int, np.ndarray] = {}
    position = 0
    with av.open(str(path)) as container:
        for frame_index, frame in enumerate(container.decode(video=0)):
            if position >= len(wanted):
                break
            if frame_index == wanted[position]:
                output[frame_index] = frame.to_ndarray(format="rgb24")
                position += 1
    missing = set(wanted) - set(output)
    if missing:
        raise IndexError(f"{path} is missing frames {sorted(missing)}")
    return output


def video_paths(dataset: Path, episode_index: int) -> dict[str, Path]:
    root = dataset / "videos" / "chunk-000"
    return {
        name: root
        / f"observation.images.{name}"
        / f"episode_{episode_index:06d}.mp4"
        for name in ("front", "wrist")
    }


def image_metrics(replayed: np.ndarray, recorded: np.ndarray) -> dict[str, float]:
    replayed = np.asarray(replayed, dtype=np.float32)
    recorded = np.asarray(recorded, dtype=np.float32)
    if replayed.shape != recorded.shape:
        raise ValueError(
            f"Image shape mismatch: replay={replayed.shape}, record={recorded.shape}"
        )
    error = replayed - recorded
    mse = float(np.mean(error**2))
    return {
        "mae_0_255": float(np.mean(np.abs(error))),
        "rmse_0_255": float(math.sqrt(mse)),
        "psnr_db": float(20.0 * math.log10(255.0 / max(math.sqrt(mse), 1e-12))),
    }


def progress_metrics(env: Any) -> dict[str, Any]:
    raw = env.unwrapped
    glass = np.asarray(raw._data.sensor("glass_pos").data, dtype=np.float64)
    box = np.asarray(raw._model.body("open_box").pos, dtype=np.float64)
    delta = glass - box
    hinge0 = float(raw._data.sensor("glass_joint_0_pos").data[0])
    hinge1 = float(raw._data.sensor("glass_joint_1_pos").data[0])
    margins = {
        "inside_x": float(0.145 - abs(delta[0])),
        "inside_y": float(0.145 - abs(delta[1])),
        "inside_z_low": float(delta[2] - (-0.045)),
        "inside_z_high": float(0.0275 - delta[2]),
    }
    return {
        "hinge_0": hinge0,
        "hinge_1": hinge1,
        "hinge_min": min(hinge0, hinge1),
        "glass_minus_box_xyz": delta.tolist(),
        "placement_margins": margins,
        "trigger_active": bool(
            hinge0 > 1.1
            and hinge1 > 1.1
            and all(value >= 0.0 for value in margins.values())
        ),
        "success_trigger_count": int(raw._success_trigger_count),
    }


def create_environment(seed: int) -> tuple[Any, Any]:
    setup_paths()
    from dexjoco.tasks import CONFIG_MAPPING

    config = CONFIG_MAPPING["fold_glasses"]()
    env = config.get_environment(
        policy_mode=True,
        render_mode="rgb_array",
        randomize=False,
        seed=int(seed),
        randomize_dynamics=False,
    )
    # Rendering and physics dominate runtime; real-time sleeping is irrelevant to
    # deterministic replay and was not part of the simulator state transition.
    env.unwrapped.hz = 1_000_000_000
    return config, env


def reset_to_repeat(env: Any, repeat: int) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    observation: dict[str, np.ndarray] | None = None
    info: dict[str, Any] | None = None
    for _ in range(repeat + 1):
        observation, info = env.reset()
    assert observation is not None and info is not None
    return observation, info


def render_current_observation(env: Any) -> dict[str, np.ndarray]:
    """Render one wrapped observation without changing the physics state."""

    raw = env.unwrapped
    previous = bool(raw.image_obs)
    raw.image_obs = True
    try:
        return env.observation(raw._compute_observation())
    finally:
        raw.image_obs = previous


def factual_replay(
    dataset: Path,
    episode_index: int,
    *,
    check_frames: Sequence[int],
    output: Path,
    state_atol: float = 2e-4,
    image_mae_threshold: float = 5.0,
    max_steps: int | None = None,
) -> dict[str, Any]:
    setup_paths()
    from fastwam_dexjoco.policy import fastwam_action_to_dexjoco

    attempt = attempt_for_episode(dataset, episode_index)
    actions, recorded_states = load_episode(dataset, episode_index)
    limit = len(actions) if max_steps is None else min(len(actions), int(max_steps))
    checks = sorted(
        frame for frame in set(int(value) for value in check_frames) if 0 <= frame < limit
    )
    if not checks:
        raise ValueError("No check frames lie within the replay interval")
    recorded_images = {
        camera: read_video_frames(path, checks)
        for camera, path in video_paths(dataset, episode_index).items()
    }

    _, env = create_environment(int(attempt["seed"]))
    output.mkdir(parents=True, exist_ok=True)
    observations: list[dict[str, Any]] = []
    terminated_early = False
    final_info: Mapping[str, Any] = {"succeed": False}
    try:
        observation, _ = reset_to_repeat(env, int(attempt["repeat"]))
        # Rendering four 640x640 cameras is not part of the physics transition.
        # Disable it between sparse audit frames so long factual controls remain
        # tractable; success and state sensors are still computed every step.
        env.unwrapped.image_obs = False
        for frame in range(limit):
            if frame in checks:
                observation = render_current_observation(env)
                replayed_state = np.asarray(observation["state"], dtype=np.float32)[:23]
                state_error = np.abs(replayed_state - recorded_states[frame])
                camera_metrics: dict[str, dict[str, float]] = {}
                for camera in ("front", "wrist"):
                    replayed = np.asarray(observation[camera], dtype=np.uint8)
                    camera_metrics[camera] = image_metrics(
                        replayed, recorded_images[camera][frame]
                    )
                    Image.fromarray(replayed).save(
                        output / f"replay_{camera}_f{frame:04d}.png"
                    )
                    Image.fromarray(recorded_images[camera][frame]).save(
                        output / f"recorded_{camera}_f{frame:04d}.png"
                    )
                observations.append(
                    {
                        "frame": frame,
                        "state_max_abs": float(state_error.max()),
                        "state_rmse": float(np.sqrt(np.mean(state_error**2))),
                        "cameras": camera_metrics,
                        "progress": progress_metrics(env),
                    }
                )
            observation, _, terminated, truncated, final_info = env.step(
                fastwam_action_to_dexjoco(actions[frame])
            )
            if terminated or truncated:
                if frame + 1 < limit:
                    terminated_early = True
                break
        final_progress = progress_metrics(env)
    finally:
        env.close()

    full_replay = limit == len(actions)
    recorded_success = bool(attempt["success"])
    replayed_success = bool(final_info.get("succeed", False))
    state_passed = all(
        float(row["state_max_abs"]) <= state_atol for row in observations
    )
    image_passed = all(
        float(metrics["mae_0_255"]) <= image_mae_threshold
        for row in observations
        for metrics in row["cameras"].values()
    )
    outcome_passed = bool(
        not full_replay or replayed_success == recorded_success
    )
    passed = bool(
        state_passed
        and image_passed
        and outcome_passed
        and not terminated_early
    )
    result = {
        "format": "FoldGlassesFactualReplayValidation",
        "version": "1.0",
        "dataset": str(dataset),
        "episode_index": episode_index,
        "seed": int(attempt["seed"]),
        "repeat": int(attempt["repeat"]),
        "recorded_success": recorded_success,
        "replayed_success": replayed_success,
        "episode_length": len(actions),
        "replayed_steps_requested": limit,
        "full_replay": full_replay,
        "terminated_early": terminated_early,
        "thresholds": {
            "state_max_abs": float(state_atol),
            "image_mae_0_255": float(image_mae_threshold),
        },
        "checks": observations,
        "final_progress": final_progress,
        "state_replay_passed": state_passed,
        "image_replay_passed": image_passed,
        "outcome_replay_passed": outcome_passed,
        "factual_replay_passed": passed,
    }
    (output / "factual_replay.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--episode-index", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check-frames", default="0,144,168,1199")
    parser.add_argument("--state-atol", type=float, default=2e-4)
    parser.add_argument("--image-mae-threshold", type=float, default=5.0)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="0 replays the full recorded attempt.",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if args.state_atol < 0.0 or args.image_mae_threshold < 0.0:
        raise ValueError("Replay thresholds must be non-negative")
    result = factual_replay(
        args.dataset.expanduser().resolve(),
        int(args.episode_index),
        check_frames=parse_ints(str(args.check_frames)),
        output=args.output.expanduser().resolve(),
        state_atol=float(args.state_atol),
        image_mae_threshold=float(args.image_mae_threshold),
        max_steps=None if int(args.max_steps) == 0 else int(args.max_steps),
    )
    print(json.dumps(result, sort_keys=True))
    return 0 if result["factual_replay_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
