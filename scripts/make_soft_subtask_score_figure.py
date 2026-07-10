#!/usr/bin/env python3
"""Build a static video-strip + soft-subtask-score figure from probe outputs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def float_or_nan(value: str) -> float:
    if value == "":
        return float("nan")
    return float(value)


def read_episode_probe(csv_path: Path, episode: int) -> dict[str, np.ndarray]:
    rows = []
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            if int(row["episode_index"]) == episode:
                rows.append(row)

    if not rows:
        raise KeyError(f"Episode {episode} was not found in {csv_path}")

    rows.sort(key=lambda row: int(row["frame_index"]))
    return {
        "frame_index": np.asarray([int(row["frame_index"]) for row in rows], dtype=np.int64),
        "timestamp": np.asarray([float(row["timestamp"]) for row in rows], dtype=np.float32),
        "score": np.asarray([float_or_nan(row["event_transition_score"]) for row in rows], dtype=np.float32),
    }


def video_info(path: Path) -> tuple[int, int, int, float]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    cap.release()
    return width, height, frame_count, fps


def default_sample_frames(frame_count: int, n: int) -> list[int]:
    if frame_count <= 0:
        raise ValueError("Video reports zero frames")
    if n <= 1:
        return [frame_count // 2]
    positions = np.linspace(0.06, 0.94, n)
    return [int(round(p * (frame_count - 1))) for p in positions]


def read_video_frames(path: Path, indices: list[int], crop_height: int | None, frame_height: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")

    frames = []
    for index in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame_bgr = cap.read()
        if not ok:
            raise ValueError(f"Could not read frame {index} from {path}")
        if crop_height is not None and crop_height > 0:
            frame_bgr = frame_bgr[: min(crop_height, frame_bgr.shape[0]), :]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w = frame_rgb.shape[:2]
        width = int(round(w * frame_height / h))
        frames.append(cv2.resize(frame_rgb, (width, frame_height), interpolation=cv2.INTER_AREA))

    cap.release()
    return frames


def make_strip(frames: list[np.ndarray], gap: int) -> np.ndarray:
    if not frames:
        raise ValueError("No frames were decoded")
    h = frames[0].shape[0]
    spacer = np.full((h, gap, 3), 255, dtype=np.uint8)
    pieces = []
    for i, frame in enumerate(frames):
        if i:
            pieces.append(spacer)
        pieces.append(frame)
    return np.concatenate(pieces, axis=1)


def load_font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    names = ["DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf", "Arial.ttf"]
    dirs = [
        Path("/usr/share/fonts/truetype/dejavu"),
        Path("/usr/share/fonts/truetype/liberation2"),
        Path("/usr/share/fonts"),
    ]
    for directory in dirs:
        for name in names:
            path = directory / name
            if path.exists():
                return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont | ImageFont.ImageFont) -> tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def draw_centered_text(
    draw: ImageDraw.ImageDraw,
    center: tuple[int, int],
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    fill: tuple[int, int, int],
) -> None:
    w, h = text_size(draw, text, font)
    draw.text((center[0] - w // 2, center[1] - h // 2), text, font=font, fill=fill)


def paste_rotated_label(
    canvas: Image.Image,
    center: tuple[int, int],
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    fill: tuple[int, int, int],
) -> None:
    probe = Image.new("RGBA", (1, 1), (0, 0, 0, 0))
    probe_draw = ImageDraw.Draw(probe)
    w, h = text_size(probe_draw, text, font)
    label = Image.new("RGBA", (w + 8, h + 8), (0, 0, 0, 0))
    label_draw = ImageDraw.Draw(label)
    label_draw.text((4, 4), text, font=font, fill=fill + (255,))
    rotated = label.rotate(90, expand=True)
    canvas.alpha_composite(rotated, (center[0] - rotated.width // 2, center[1] - rotated.height // 2))


def draw_dashed_line(
    draw: ImageDraw.ImageDraw,
    xy0: tuple[int, int],
    xy1: tuple[int, int],
    fill: tuple[int, int, int],
    width: int = 2,
    dash: int = 10,
    gap: int = 8,
) -> None:
    x0, y0 = xy0
    x1, y1 = xy1
    if x0 != x1:
        raise ValueError("Only vertical dashed lines are supported")
    y = y0
    while y < y1:
        draw.line((x0, y, x1, min(y + dash, y1)), fill=fill, width=width)
        y += dash + gap


def timestamp_for_frame(probe: dict[str, np.ndarray], frame_index: int, fps: float) -> float:
    probe_frames = probe["frame_index"]
    matches = np.flatnonzero(probe_frames == frame_index)
    if len(matches):
        return float(probe["timestamp"][matches[0]])
    nearest = int(np.argmin(np.abs(probe_frames - frame_index)))
    if abs(int(probe_frames[nearest]) - frame_index) <= 1:
        return float(probe["timestamp"][nearest])
    return float(frame_index) / fps


def draw_figure(
    probe: dict[str, np.ndarray],
    strip: np.ndarray,
    sample_indices: list[int],
    fps: float,
    title: str,
    output: Path,
) -> None:
    sample_times = [timestamp_for_frame(probe, index, fps) for index in sample_indices]
    x = probe["timestamp"]
    y = probe["score"]

    strip_img = Image.fromarray(strip)
    left_margin = 130
    right_margin = 40
    top_margin = 22
    text_h = 90
    video_gap = 12
    plot_gap = 34
    plot_h = 285
    bottom_margin = 54

    width = left_margin + strip_img.width + right_margin
    video_y = top_margin + text_h + video_gap
    plot_x0 = left_margin
    plot_y0 = video_y + strip_img.height + plot_gap
    plot_w = strip_img.width
    height = plot_y0 + plot_h + bottom_margin

    canvas = Image.new("RGBA", (width, height), (255, 255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    font_label = load_font(26, bold=True)
    font_title = load_font(34, bold=True)
    font_tick = load_font(20)
    font_axis = load_font(23)

    title_w, title_text_h = text_size(draw, title, font_title)
    title_pad_x = 58
    title_pad_y = 20
    title_box = [
        max(left_margin, (width - title_w - 2 * title_pad_x) // 2),
        top_margin + 4,
        min(width - right_margin, (width + title_w + 2 * title_pad_x) // 2),
        top_margin + 4 + title_text_h + 2 * title_pad_y,
    ]
    draw.rounded_rectangle(title_box, radius=22, fill=(242, 234, 215), outline=(242, 234, 215), width=2)
    draw_centered_text(draw, ((title_box[0] + title_box[2]) // 2, (title_box[1] + title_box[3]) // 2), title, font_title, (28, 28, 28))

    canvas.alpha_composite(strip_img.convert("RGBA"), (left_margin, video_y))
    draw.rounded_rectangle(
        [left_margin - 2, video_y - 2, left_margin + strip_img.width + 2, video_y + strip_img.height + 2],
        radius=4,
        outline=(230, 230, 230),
        width=2,
    )

    plot_bg = [plot_x0, plot_y0, plot_x0 + plot_w, plot_y0 + plot_h]
    draw.rectangle(plot_bg, fill=(255, 255, 255), outline=(205, 205, 205), width=2)
    for tick in [0.0, 0.25, 0.5, 0.75, 1.0]:
        yy = int(round(plot_y0 + plot_h - tick * plot_h))
        draw.line((plot_x0, yy, plot_x0 + plot_w, yy), fill=(226, 226, 226), width=2)
        label = f"{tick:.2f}".rstrip("0").rstrip(".")
        tw, th = text_size(draw, label, font_tick)
        draw.text((plot_x0 - tw - 16, yy - th // 2), label, font=font_tick, fill=(92, 92, 92))

    x_min = float(np.nanmin(x))
    x_max = float(np.nanmax(x))
    x_span = max(x_max - x_min, 1e-6)
    for t in sample_times:
        xx = int(round(plot_x0 + (t - x_min) / x_span * plot_w))
        draw_dashed_line(draw, (xx, plot_y0), (xx, plot_y0 + plot_h), fill=(180, 180, 180), width=2)

    points: list[tuple[int, int]] = []
    for t, score in zip(x, y):
        if not np.isfinite(score):
            if len(points) >= 2:
                draw.line(points, fill=(45, 111, 183), width=5, joint="curve")
            points = []
            continue
        xx = int(round(plot_x0 + (float(t) - x_min) / x_span * plot_w))
        yy = int(round(plot_y0 + plot_h - np.clip(float(score), 0.0, 1.0) * plot_h))
        points.append((xx, yy))
    if len(points) >= 2:
        draw.line(points, fill=(45, 111, 183), width=5, joint="curve")

    tick_step = 2.0 if x_max <= 10 else 5.0
    x_ticks = np.arange(0.0, x_max + 1e-6, tick_step)
    for tick in x_ticks:
        xx = int(round(plot_x0 + (tick - x_min) / x_span * plot_w))
        draw.line((xx, plot_y0 + plot_h, xx, plot_y0 + plot_h + 8), fill=(65, 65, 65), width=2)
        label = f"{tick:.0f}"
        tw, th = text_size(draw, label, font_tick)
        draw.text((xx - tw // 2, plot_y0 + plot_h + 12), label, font=font_tick, fill=(65, 65, 65))

    axis_label = "Time (s)"
    aw, ah = text_size(draw, axis_label, font_axis)
    draw.text((plot_x0 + plot_w // 2 - aw // 2, plot_y0 + plot_h + 34), axis_label, font=font_axis, fill=(42, 42, 42))

    paste_rotated_label(canvas, (42, top_margin + text_h // 2), "Text", font_label, (100, 100, 100))
    paste_rotated_label(canvas, (42, video_y + strip_img.height // 2), "Video", font_label, (100, 100, 100))
    paste_rotated_label(canvas, (42, plot_y0 + plot_h // 2), "Soft subtask score", font_axis, (100, 100, 100))

    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--probe-dir",
        type=Path,
        default=Path("results/state_line_probe/hammer_nail_fastwam_state_line_distance"),
    )
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--video", type=Path, default=None)
    parser.add_argument("--probe-csv", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--title", type=str, default=None)
    parser.add_argument("--num-frames", type=int, default=5)
    parser.add_argument("--frame-indices", type=int, nargs="*", default=None)
    parser.add_argument(
        "--video-crop-height",
        type=int,
        default=360,
        help="Crop overlay videos to their video row. Use 0 to keep the full frame.",
    )
    parser.add_argument("--frame-height", type=int, default=230)
    parser.add_argument("--frame-gap", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    probe_csv = args.probe_csv or args.probe_dir / "state_line_distance_probe.csv"
    video = args.video or args.probe_dir / "videos" / f"episode_{args.episode:06d}_state_line_distance_overlay.mp4"
    output = args.output or args.probe_dir / f"episode_{args.episode:06d}_soft_subtask_score_figure.png"
    title = args.title or f"Hammer nail episode {args.episode:06d}"

    probe = read_episode_probe(probe_csv, args.episode)
    _, height, frame_count, fps = video_info(video)
    if args.frame_indices is None or len(args.frame_indices) == 0:
        sample_indices = default_sample_frames(min(frame_count, len(probe["frame_index"])), args.num_frames)
    else:
        sample_indices = [max(0, min(frame_count - 1, int(index))) for index in args.frame_indices]

    crop_height = None if args.video_crop_height <= 0 else min(args.video_crop_height, height)
    frames = read_video_frames(video, sample_indices, crop_height, args.frame_height)
    strip = make_strip(frames, args.frame_gap)
    draw_figure(probe, strip, sample_indices, fps, title, output)
    print(output)


if __name__ == "__main__":
    main()
