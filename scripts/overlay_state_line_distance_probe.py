#!/usr/bin/env python3
"""Render state line-distance event scores on top of LeRobot episode videos."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import av
import cv2
import numpy as np


def float_or_nan(value: str) -> float:
    if value == "":
        return float("nan")
    return float(value)


def read_probe_csv(path: Path) -> dict[int, dict[str, np.ndarray]]:
    rows_by_ep: dict[int, list[dict[str, str]]] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            ep = int(row["episode_index"])
            rows_by_ep.setdefault(ep, []).append(row)

    out = {}
    for ep, rows in rows_by_ep.items():
        rows.sort(key=lambda r: int(r["frame_index"]))
        out[ep] = {
            "frame_index": np.asarray([int(r["frame_index"]) for r in rows], dtype=np.int64),
            "timestamp": np.asarray([float(r["timestamp"]) for r in rows], dtype=np.float32),
            "distance": np.asarray([float_or_nan(r["state_line_distance"]) for r in rows], dtype=np.float32),
            "score": np.asarray([float_or_nan(r["event_transition_score"]) for r in rows], dtype=np.float32),
        }
    return out


def select_episodes(args: argparse.Namespace) -> list[int]:
    if args.episodes:
        return args.episodes
    with args.summary_json.open() as f:
        summary = json.load(f)
    episodes = []
    for event in summary["top_events"]:
        ep = int(event["episode_index"])
        if ep not in episodes:
            episodes.append(ep)
        if len(episodes) >= args.num_episodes:
            break
    return episodes


def decode_video(path: Path) -> tuple[list[np.ndarray], float]:
    frames = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate) if stream.average_rate is not None else 30.0
        for frame in container.decode(stream):
            frames.append(frame.to_ndarray(format="bgr24"))
    return frames, fps


def color_for_score(score: float) -> tuple[int, int, int]:
    if not np.isfinite(score):
        return (160, 160, 160)
    score = float(np.clip(score, 0.0, 1.0))
    if score < 0.5:
        a = score / 0.5
        return (0, int(210 * a + 170 * (1 - a)), int(60 * (1 - a)))
    a = (score - 0.5) / 0.5
    return (0, int(210 * (1 - a) + 60 * a), int(255 * a + 60 * (1 - a)))


def resize_height(frame: np.ndarray, height: int) -> np.ndarray:
    h, w = frame.shape[:2]
    width = int(round(w * height / h))
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def draw_panel(width: int, height: int, probe: dict[str, np.ndarray], idx: int, trail: int) -> np.ndarray:
    panel = np.full((height, width, 3), 248, dtype=np.uint8)
    left, right = 72, width - 20
    top, bottom = 20, height - 34
    cv2.rectangle(panel, (left, top), (right, bottom), (215, 215, 215), 1)

    for y_value in [0.25, 0.5, 0.75]:
        y = int(bottom - y_value * (bottom - top))
        cv2.line(panel, (left, y), (right, y), (232, 232, 232), 1)
        cv2.putText(panel, f"{y_value:.2f}", (18, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (90, 90, 90), 1, cv2.LINE_AA)

    score = probe["score"]
    start = max(0, idx - trail + 1)
    end = min(len(score), idx + 1)
    if end - start >= 2:
        xs = np.linspace(left, right, end - start)
        ys = []
        for value in score[start:end]:
            if np.isfinite(value):
                ys.append(int(bottom - np.clip(value, 0.0, 1.0) * (bottom - top)))
            else:
                ys.append(None)
        for i in range(1, len(xs)):
            if ys[i - 1] is None or ys[i] is None:
                continue
            cv2.line(panel, (int(xs[i - 1]), ys[i - 1]), (int(xs[i]), ys[i]), (45, 85, 220), 2, cv2.LINE_AA)

    current_score = float(score[idx]) if idx < len(score) else float("nan")
    frame_index = int(probe["frame_index"][idx]) if idx < len(probe["frame_index"]) else idx
    timestamp = float(probe["timestamp"][idx]) if idx < len(probe["timestamp"]) else idx / 30.0
    label = "score=nan" if not np.isfinite(current_score) else f"score={current_score:.3f}"
    cv2.putText(
        panel,
        f"frame {frame_index}  t={timestamp:.2f}s  state_all {label}",
        (left, 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (40, 40, 40),
        1,
        cv2.LINE_AA,
    )
    return panel


def render_episode(
    dataset_dir: Path,
    output_dir: Path,
    episode: int,
    probe: dict[str, np.ndarray],
    camera_keys: list[str],
    video_height: int,
    panel_height: int,
    trail: int,
) -> Path:
    chunk = episode // 1000
    videos = []
    fps_values = []
    for key in camera_keys:
        path = dataset_dir / "videos" / f"chunk-{chunk:03d}" / f"observation.images.{key}" / f"episode_{episode:06d}.mp4"
        frames, fps = decode_video(path)
        videos.append(frames)
        fps_values.append(fps)

    n = min([len(v) for v in videos] + [len(probe["score"])])
    if n == 0:
        raise ValueError(f"No frames to render for episode {episode}")

    resized_first = [resize_height(v[0], video_height) for v in videos]
    video_width = sum(frame.shape[1] for frame in resized_first)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"episode_{episode:06d}_state_line_distance_overlay.mp4"
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps_values[0],
        (video_width, video_height + panel_height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {out_path}")

    for i in range(n):
        video_row = np.concatenate([resize_height(v[i], video_height) for v in videos], axis=1)
        score = float(probe["score"][i])
        cv2.putText(video_row, f"episode {episode:03d}", (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(video_row, (video_width - 28, 26), 12, color_for_score(score), -1)
        panel = draw_panel(video_width, panel_height, probe, i, trail)
        writer.write(np.concatenate([video_row, panel], axis=0))
    writer.release()
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--probe-csv", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, nargs="*", default=None)
    parser.add_argument("--num-episodes", type=int, default=6)
    parser.add_argument("--cameras", nargs="+", default=["front", "wrist"])
    parser.add_argument("--video-height", type=int, default=360)
    parser.add_argument("--panel-height", type=int, default=150)
    parser.add_argument("--trail", type=int, default=90)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    probe_by_ep = read_probe_csv(args.probe_csv)
    episodes = select_episodes(args)
    rendered = []
    for ep in episodes:
        if ep not in probe_by_ep:
            raise KeyError(f"Episode {ep} not found in {args.probe_csv}")
        out_path = render_episode(
            dataset_dir=args.dataset_dir,
            output_dir=args.output_dir,
            episode=ep,
            probe=probe_by_ep[ep],
            camera_keys=args.cameras,
            video_height=args.video_height,
            panel_height=args.panel_height,
            trail=args.trail,
        )
        rendered.append(str(out_path))
        print(out_path)

    manifest = args.output_dir / "overlay_manifest.json"
    manifest.write_text(json.dumps({"score": "state_all", "videos": rendered}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
