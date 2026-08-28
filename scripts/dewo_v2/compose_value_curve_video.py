#!/usr/bin/env python3
"""Compose DexJoCo rollout mp4 with a synced V(t) curve panel."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np


def load_v_curve(npz_path: Path) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(npz_path)
    steps = np.asarray(z["policy_query_steps"], dtype=np.int32)
    values = np.asarray(z["cfg_values"], dtype=np.float64)
    order = np.argsort(steps)
    return steps[order], values[order]


def render_plot_panel(
    steps: np.ndarray,
    values: np.ndarray,
    *,
    current_step: int,
    x_max: int,
    y_max: float,
    width: int,
    height: int,
    title: str,
) -> np.ndarray:
    dpi = 100
    fig_w, fig_h = width / dpi, height / dpi
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)

    mask = steps <= x_max
    xs = steps[mask]
    ys = values[mask]
    ax.plot(xs, ys, color="#2ecc71", linewidth=2.5, label="V(s)")
    ax.scatter(xs, ys, color="#2ecc71", s=28, zorder=3)

    visible = xs[xs <= current_step]
    if visible.size:
        ax.axvline(current_step, color="#e74c3c", linewidth=2, alpha=0.85)
        idx = np.searchsorted(xs, current_step, side="right") - 1
        idx = int(np.clip(idx, 0, len(xs) - 1))
        ax.scatter([xs[idx]], [ys[idx]], color="#e74c3c", s=64, zorder=4)

    ax.set_xlim(0, x_max)
    ax.set_ylim(0, y_max)
    ax.set_xlabel("env step")
    ax.set_ylabel("V")
    ax.set_title(title, fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.text(
        0.02,
        0.98,
        f"t = {current_step}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
    )

    fig.tight_layout(pad=0.8)
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)
    rgb = rgba[..., :3].copy()
    plt.close(fig)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def compose(
    *,
    video_path: Path,
    npz_path: Path,
    output_path: Path,
    max_step: int | None,
    y_max: float,
    fps: float | None,
) -> None:
    steps, values = load_v_curve(npz_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    out_fps = fps or src_fps
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    plot_w, plot_h = frame_w, frame_h
    out_w, out_h = frame_w + plot_w, frame_h

    if max_step is None:
        max_step = int(steps[-1])
    title = f"V(t) · {npz_path.stem.replace('_actions', '')}"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(".tmp.avi")
    writer = cv2.VideoWriter(
        str(tmp),
        cv2.VideoWriter_fourcc(*"MJPG"),
        out_fps,
        (out_w, out_h),
    )
    if not writer.isOpened():
        raise RuntimeError("VideoWriter failed")

    frame_idx = 0
    written = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx > max_step:
            break
        panel = render_plot_panel(
            steps,
            values,
            current_step=frame_idx,
            x_max=max_step,
            y_max=y_max,
            width=plot_w,
            height=plot_h,
            title=title,
        )
        if panel.shape[0] != frame_h or panel.shape[1] != plot_w:
            panel = cv2.resize(panel, (plot_w, frame_h))
        writer.write(np.hstack([frame, panel]))
        frame_idx += 1
        written += 1

    cap.release()
    writer.release()

    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(tmp),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "20",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)
    tmp.unlink(missing_ok=True)
    print(f"[compose-value-video] wrote {written} frames -> {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--npz", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-step", type=int, default=384)
    parser.add_argument("--y-max", type=float, default=0.12)
    parser.add_argument("--fps", type=float, default=None)
    args = parser.parse_args()
    compose(
        video_path=args.video.expanduser().resolve(),
        npz_path=args.npz.expanduser().resolve(),
        output_path=args.output.expanduser().resolve(),
        max_step=args.max_step,
        y_max=args.y_max,
        fps=args.fps,
    )


if __name__ == "__main__":
    main()
