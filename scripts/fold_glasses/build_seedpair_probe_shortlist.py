#!/usr/bin/env python3
"""Build and render a width-stratified branch-event probe shortlist.

Failure policy width is attached as a diagnostic, not used as an event gate.
The shortlist deliberately contains high- and low-width candidates with the
same observational branch criteria so downstream policy probes can test whether
a width rise is actually informative.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import av
import numpy as np
from PIL import Image, ImageDraw


CAMERAS = ("front", "wrist")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(dict(row), sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def width_diagnostic(
    width_root: Path, episode_index: int, frame: int
) -> dict[str, Any] | None:
    path = width_root / "npz" / f"ep{episode_index:06d}_widths.npz"
    if not path.exists():
        return None
    with np.load(path) as payload:
        steps = np.asarray(payload["probe_steps"], dtype=np.int32)
        hits = np.flatnonzero(steps == int(frame))
        if len(hits) != 1:
            return None
        index = int(hits[0])
        width = float(payload["probe_widths"][index])
        baseline = float(payload["baseline_median"][index])
        ratio = (
            float(width / baseline)
            if np.isfinite(baseline) and baseline > 1e-8
            else None
        )
        return {
            "source": str(path.resolve()),
            "space": "denormalized_robot_units",
            "definition": "l2_norm_std_of_first_step_over_8_policy_samples",
            "width": width,
            "baseline_median": baseline if np.isfinite(baseline) else None,
            "ratio_to_baseline": ratio,
            "detector_found_jump": bool(payload["found_event"]),
            "detector_jump_frame": int(payload["event_center_frame"]),
            "used_as_event_gate": False,
        }


def attach_widths(
    candidates: Sequence[Mapping[str, Any]], width_root: Path
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        row = dict(candidate)
        row["failure_policy_width_diagnostic"] = width_diagnostic(
            width_root,
            int(row["failure_episode_index"]),
            int(row["failure_frame"]),
        )
        rows.append(row)
    return rows


def select_stratum(
    rows: Sequence[Mapping[str, Any]],
    *,
    name: str,
    minimum_ratio: float | None,
    maximum_ratio: float | None,
    count: int,
    min_supported_successes: int,
) -> list[dict[str, Any]]:
    eligible: list[dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        diagnostic = row.get("failure_policy_width_diagnostic")
        ratio = None if diagnostic is None else diagnostic.get("ratio_to_baseline")
        if ratio is None or not np.isfinite(float(ratio)):
            continue
        if int(row["shared_context"]["supported_success_count"]) < min_supported_successes:
            continue
        onset = str(int(row["future_context"]["persistence"]["onset_horizon"]))
        source_counts = row["future_context"].get("evidence_source_counts", {})
        if int(source_counts.get(onset, 0)) < min_supported_successes:
            continue
        if minimum_ratio is not None and float(ratio) < minimum_ratio:
            continue
        if maximum_ratio is not None and float(ratio) > maximum_ratio:
            continue
        eligible.append(row)
    eligible.sort(
        key=lambda row: (
            -float(row["selection_score"]),
            int(row["global_observational_rank"]),
        )
    )
    selected: list[dict[str, Any]] = []
    seen_seeds: set[int] = set()
    for row in eligible:
        seed = int(row["seed"])
        if seed in seen_seeds:
            continue
        seen_seeds.add(seed)
        row["width_stratum"] = name
        row["shortlist_rank_within_stratum"] = len(selected) + 1
        row["probe_shortlist_eligible"] = True
        selected.append(row)
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError(
            f"Stratum {name!r} has only {len(selected)} distinct-seed candidates; "
            f"requested {count}"
        )
    return selected


def build_shortlist(
    candidates: Sequence[Mapping[str, Any]],
    width_root: Path,
    *,
    per_stratum: int = 4,
    min_supported_successes: int = 2,
    high_ratio_min: float = 1.5,
    low_ratio_max: float = 1.0,
) -> list[dict[str, Any]]:
    if per_stratum <= 0 or min_supported_successes <= 0:
        raise ValueError("Shortlist sizes must be positive")
    if not 0.0 < low_ratio_max < high_ratio_min:
        raise ValueError("Expected 0 < low_ratio_max < high_ratio_min")
    rows = attach_widths(candidates, width_root)
    high = select_stratum(
        rows,
        name="high_failure_width",
        minimum_ratio=high_ratio_min,
        maximum_ratio=None,
        count=per_stratum,
        min_supported_successes=min_supported_successes,
    )
    low = select_stratum(
        rows,
        name="low_failure_width_control",
        minimum_ratio=None,
        maximum_ratio=low_ratio_max,
        count=per_stratum,
        min_supported_successes=min_supported_successes,
    )
    output = high + low
    output.sort(
        key=lambda row: (
            str(row["width_stratum"]),
            int(row["shortlist_rank_within_stratum"]),
        )
    )
    return output


def video_path(dataset: Path, episode: int, camera: str) -> Path:
    return (
        dataset
        / "videos"
        / "chunk-000"
        / f"observation.images.{camera}"
        / f"episode_{episode:06d}.mp4"
    )


def read_frames(
    path: Path, frames: Sequence[int]
) -> dict[int, tuple[np.ndarray, int]]:
    """Read requested frames and explicitly proxy requests past a terminal frame."""

    wanted = sorted(set(int(value) for value in frames))
    exact: dict[int, np.ndarray] = {}
    position = 0
    last_frame: np.ndarray | None = None
    last_index = -1
    with av.open(str(path)) as container:
        for index, frame in enumerate(container.decode(video=0)):
            last_frame = frame.to_ndarray(format="rgb24")
            last_index = index
            if index == wanted[position]:
                exact[index] = last_frame
                position += 1
                if position >= len(wanted):
                    break
    if last_frame is None:
        raise RuntimeError(f"{path} contains no frames")
    output: dict[int, tuple[np.ndarray, int]] = {}
    for requested in wanted:
        if requested in exact:
            output[requested] = (exact[requested], requested)
        elif requested > last_index:
            output[requested] = (last_frame, last_index)
        else:
            raise IndexError(f"{path} decoder skipped frame {requested}")
    return output


def collect_render_requests(
    shortlist: Sequence[Mapping[str, Any]],
) -> dict[tuple[int, str], set[int]]:
    requests: dict[tuple[int, str], set[int]] = defaultdict(set)
    for candidate in shortlist:
        horizon = int(candidate["future_context"]["persistence"]["onset_horizon"])
        stride = int(candidate["stride"])
        trajectories = [
            {
                "episode": int(candidate["failure_episode_index"]),
                "frame": int(candidate["failure_frame"]),
            }
        ] + [
            {
                "episode": int(row["success_episode_index"]),
                "frame": int(row["success_frame"]),
            }
            for row in candidate["success_alignments"]
        ]
        for trajectory in trajectories:
            for camera in CAMERAS:
                requests[(trajectory["episode"], camera)].update(
                    {trajectory["frame"], trajectory["frame"] + horizon * stride}
                )
    return requests


def render_candidate(
    candidate: Mapping[str, Any],
    frame_cache: Mapping[tuple[int, str, int], tuple[np.ndarray, int]],
    output: Path,
    *,
    tile_size: int = 224,
) -> Path:
    horizon = int(candidate["future_context"]["persistence"]["onset_horizon"])
    stride = int(candidate["stride"])
    rows = [
        {
            "role": "FAIL",
            "episode": int(candidate["failure_episode_index"]),
            "frame": int(candidate["failure_frame"]),
            "cost": float(candidate["shared_context"]["current_context_cost_median"]),
        }
    ] + [
        {
            "role": f"SUCCESS {position}",
            "episode": int(alignment["success_episode_index"]),
            "frame": int(alignment["success_frame"]),
            "cost": float(alignment["current_context_cost"]),
        }
        for position, alignment in enumerate(candidate["success_alignments"], start=1)
    ]
    columns = (
        ("front", 0, "front anchor"),
        ("wrist", 0, "wrist anchor"),
        ("front", horizon * stride, f"front +{horizon} replans"),
        ("wrist", horizon * stride, f"wrist +{horizon} replans"),
    )
    title_height = 64
    header_height = 28
    label_width = 170
    canvas = Image.new(
        "RGB",
        (label_width + len(columns) * tile_size, title_height + header_height + len(rows) * tile_size),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    width = candidate["failure_policy_width_diagnostic"]
    draw.text(
        (8, 6),
        (
            f"{candidate['candidate_id']}  {candidate['width_stratum']}  "
            f"obs-rank={candidate['global_observational_rank']}\n"
            f"width/base={width['ratio_to_baseline']:.2f}  "
            f"branch-rms={candidate['executed_action_branch']['nearest_success_mode_block_rms']:.3f}  "
            f"future onset=+{horizon} replans"
        ),
        fill="black",
    )
    for column, (_, _, label) in enumerate(columns):
        draw.text(
            (label_width + column * tile_size + 6, title_height + 6),
            label,
            fill="black",
        )
    for row_index, trajectory in enumerate(rows):
        top = title_height + header_height + row_index * tile_size
        draw.multiline_text(
            (6, top + 8),
            (
                f"{trajectory['role']}\n"
                f"ep{trajectory['episode']:06d}\n"
                f"f{trajectory['frame']:04d}\n"
                f"ctx={trajectory['cost']:.3f}"
            ),
            fill="black",
            spacing=4,
        )
        for column, (camera, offset, _) in enumerate(columns):
            frame = trajectory["frame"] + offset
            array, actual_frame = frame_cache[
                (trajectory["episode"], camera, frame)
            ]
            image = Image.fromarray(array).resize(
                (tile_size, tile_size), Image.Resampling.LANCZOS
            )
            if actual_frame != frame:
                overlay = ImageDraw.Draw(image)
                overlay.rectangle(
                    (0, tile_size - 34, tile_size, tile_size), fill=(150, 0, 0)
                )
                overlay.text(
                    (6, tile_size - 29),
                    f"SUCCESS TERMINAL proxy f{actual_frame}",
                    fill="white",
                )
            canvas.paste(image, (label_width + column * tile_size, top))
    destination = output / "previews" / f"{candidate['candidate_id']}.jpg"
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, quality=92)
    return destination


def render_shortlist(
    dataset: Path,
    shortlist: Sequence[Mapping[str, Any]],
    output: Path,
) -> list[Path]:
    requests = collect_render_requests(shortlist)
    cache: dict[tuple[int, str, int], tuple[np.ndarray, int]] = {}
    for (episode, camera), frames in sorted(requests.items()):
        decoded = read_frames(video_path(dataset, episode, camera), sorted(frames))
        for requested_frame, value in decoded.items():
            cache[(episode, camera, requested_frame)] = value
    return [render_candidate(row, cache, output) for row in shortlist]


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--width-root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-stratum", type=int, default=4)
    parser.add_argument("--min-supported-successes", type=int, default=2)
    parser.add_argument("--high-ratio-min", type=float, default=1.5)
    parser.add_argument("--low-ratio-max", type=float, default=1.0)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    candidates_path = args.candidates.expanduser().resolve()
    width_root = args.width_root.expanduser().resolve()
    dataset = args.dataset.expanduser().resolve()
    output = args.output.expanduser().resolve()
    shortlist = build_shortlist(
        read_jsonl(candidates_path),
        width_root,
        per_stratum=int(args.per_stratum),
        min_supported_successes=int(args.min_supported_successes),
        high_ratio_min=float(args.high_ratio_min),
        low_ratio_max=float(args.low_ratio_max),
    )
    output.mkdir(parents=True, exist_ok=True)
    previews = render_shortlist(dataset, shortlist, output)
    for row, preview in zip(shortlist, previews):
        row["preview"] = str(preview.resolve())
    write_jsonl(output / "probe_shortlist.jsonl", shortlist)
    summary = {
        "format": "FoldGlassesWidthStratifiedProbeShortlist",
        "version": "1.0",
        "candidates": str(candidates_path),
        "width_root": str(width_root),
        "dataset": str(dataset),
        "per_stratum": int(args.per_stratum),
        "min_supported_successes": int(args.min_supported_successes),
        "high_ratio_min": float(args.high_ratio_min),
        "low_ratio_max": float(args.low_ratio_max),
        "width_is_stratification_not_event_gate": True,
        "num_candidates": len(shortlist),
        "candidate_ids": [str(row["candidate_id"]) for row in shortlist],
        "output": str(output),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
