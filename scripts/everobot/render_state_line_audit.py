#!/usr/bin/env python3
"""Render a stratified visual audit of EveRobot state-line candidates."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Callable, Mapping, Sequence


AUDIT_FORMAT = "EveRobotStateLineAudit"
AUDIT_VERSION = "0.1"
CAMERA_KEYS = (
    "observation.images.front",
    "observation.images.wrist",
)
STRATA = (
    ("train", "success"),
    ("train", "failure"),
    ("val", "success"),
    ("val", "failure"),
)


@dataclass(frozen=True)
class AuditSelection:
    """One episode and its highest-confidence interaction candidate."""

    episode: dict[str, Any]
    event: dict[str, Any]


FrameLoader = Callable[[Path, int], Any]


def canonical_json(payload: Any) -> str:
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                value = json.loads(stripped)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from error
            if not isinstance(value, dict):
                raise ValueError(
                    f"{path}:{line_number}: expected one JSON object"
                )
            rows.append(value)
    return rows


def _nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < minimum
    ):
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _confidence(event: Mapping[str, Any], label: str) -> float:
    value = event.get("absolute_confidence")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label}.absolute_confidence must be numeric")
    confidence = float(value)
    if not 0.0 <= confidence <= 1.0:
        raise ValueError(f"{label}.absolute_confidence must be in [0, 1]")
    return confidence


def _derived_seed(seed: int, split: str, outcome: str) -> int:
    payload = f"{seed}\0{split}\0{outcome}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def select_audit_events(
    episode_rows: Sequence[Mapping[str, Any]],
    event_rows: Sequence[Mapping[str, Any]],
    *,
    num_episodes: int,
    seed: int,
) -> list[AuditSelection]:
    """Select one best candidate per episode, balanced across four strata."""

    if num_episodes < len(STRATA):
        raise ValueError(
            f"num_episodes must be at least {len(STRATA)} for four-way "
            "stratification"
        )

    episodes: dict[str, dict[str, Any]] = {}
    for index, raw_episode in enumerate(episode_rows):
        episode = dict(raw_episode)
        episode_id = _nonempty_string(
            episode.get("episode_id"),
            f"episode_meta[{index}].episode_id",
        )
        if episode_id in episodes:
            raise ValueError(f"Duplicate episode_id: {episode_id}")
        split = _nonempty_string(
            episode.get("split"), f"episode_meta[{index}].split"
        )
        outcome = _nonempty_string(
            episode.get("episode_outcome"),
            f"episode_meta[{index}].episode_outcome",
        )
        if (split, outcome) in STRATA:
            episodes[episode_id] = episode

    best_by_episode: dict[str, dict[str, Any]] = {}
    event_ids: set[str] = set()
    for index, raw_event in enumerate(event_rows):
        if raw_event.get("event_type") != "interaction_candidate":
            continue
        event = dict(raw_event)
        event_id = _nonempty_string(
            event.get("event_id"), f"event_meta[{index}].event_id"
        )
        if event_id in event_ids:
            raise ValueError(f"Duplicate event_id: {event_id}")
        event_ids.add(event_id)
        episode_id = _nonempty_string(
            event.get("episode_id"), f"event_meta[{index}].episode_id"
        )
        episode = episodes.get(episode_id)
        if episode is None:
            continue
        if (
            event.get("split") != episode.get("split")
            or event.get("episode_outcome")
            != episode.get("episode_outcome")
        ):
            raise ValueError(
                f"event {event_id} split/outcome disagrees with episode "
                f"{episode_id}"
            )
        confidence = _confidence(event, f"event_meta[{index}]")
        previous = best_by_episode.get(episode_id)
        if previous is None:
            best_by_episode[episode_id] = event
            continue
        previous_key = (
            _confidence(previous, f"event {previous['event_id']}"),
            str(previous["event_id"]),
        )
        current_key = (confidence, event_id)
        if current_key[0] > previous_key[0] or (
            current_key[0] == previous_key[0]
            and current_key[1] < previous_key[1]
        ):
            best_by_episode[episode_id] = event

    queues: dict[tuple[str, str], list[AuditSelection]] = {
        stratum: [] for stratum in STRATA
    }
    for episode_id, event in best_by_episode.items():
        episode = episodes[episode_id]
        stratum = (
            str(episode["split"]),
            str(episode["episode_outcome"]),
        )
        queues[stratum].append(AuditSelection(episode=episode, event=event))

    for split, outcome in STRATA:
        queue = queues[(split, outcome)]
        if not queue:
            raise ValueError(
                f"No interaction_candidate episodes for {split}/{outcome}"
            )
        queue.sort(
            key=lambda selection: (
                str(selection.episode["episode_id"]),
                str(selection.event["event_id"]),
            )
        )
        random.Random(_derived_seed(seed, split, outcome)).shuffle(queue)

    total_available = sum(len(queue) for queue in queues.values())
    if total_available < num_episodes:
        raise ValueError(
            f"Requested {num_episodes} episodes, but only "
            f"{total_available} stratified candidates are available"
        )

    selected: list[AuditSelection] = []
    offsets = {stratum: 0 for stratum in STRATA}
    while len(selected) < num_episodes:
        progress = False
        for stratum in STRATA:
            offset = offsets[stratum]
            queue = queues[stratum]
            if offset >= len(queue):
                continue
            selected.append(queue[offset])
            offsets[stratum] = offset + 1
            progress = True
            if len(selected) == num_episodes:
                break
        if not progress:
            raise RuntimeError("Stratified sampler exhausted unexpectedly")
    return selected


def candidate_frame_indices(
    episode: Mapping[str, Any],
    event: Mapping[str, Any],
) -> dict[str, int]:
    """Return one frame before, inside, and after a half-open candidate."""

    length = _integer(episode.get("length"), "episode.length", minimum=1)
    start = _integer(event.get("start_frame"), "event.start_frame")
    end = _integer(event.get("end_frame"), "event.end_frame", minimum=1)
    core_start = _integer(
        event.get("core_start_frame", start),
        "event.core_start_frame",
    )
    core_end = _integer(
        event.get("core_end_frame", end),
        "event.core_end_frame",
        minimum=1,
    )
    if not 0 <= start < end <= length:
        raise ValueError(
            f"candidate interval [{start}, {end}) exceeds episode length "
            f"{length}"
        )
    if not start <= core_start < core_end <= end:
        raise ValueError(
            f"core interval [{core_start}, {core_end}) is outside candidate "
            f"[{start}, {end})"
        )

    peak: int | None = None
    annotation = event.get("annotation")
    containers: list[Mapping[str, Any]] = [event]
    if isinstance(annotation, Mapping):
        containers.append(annotation)
    for container in containers:
        for key in ("core_peak_frame", "peak_frame"):
            value = container.get(key)
            if value is None:
                continue
            candidate_peak = _integer(value, f"event.{key}")
            if not core_start <= candidate_peak < core_end:
                raise ValueError(
                    f"event.{key}={candidate_peak} is outside core interval "
                    f"[{core_start}, {core_end})"
                )
            peak = candidate_peak
            break
        if peak is not None:
            break
    if peak is None:
        peak = (core_start + core_end - 1) // 2

    return {
        "before": max(0, start - 1),
        "core": peak,
        "after": min(length - 1, end),
    }


def _read_lerobot_info(dataset_root: Path) -> dict[str, Any]:
    path = dataset_root / "meta" / "info.json"
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{path}: invalid JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def locate_video(
    episode: Mapping[str, Any],
    video_key: str,
) -> Path:
    dataset_root = Path(
        _nonempty_string(episode.get("dataset_root"), "episode.dataset_root")
    ).expanduser().resolve()
    episode_index = _integer(
        episode.get("episode_index"), "episode.episode_index"
    )
    info = _read_lerobot_info(dataset_root)
    chunks_size = int(info.get("chunks_size", 1000))
    if chunks_size <= 0:
        raise ValueError("meta/info.json chunks_size must be positive")
    pattern = str(
        info.get(
            "video_path",
            "videos/chunk-{episode_chunk:03d}/{video_key}/"
            "episode_{episode_index:06d}.mp4",
        )
    )
    expected = dataset_root / pattern.format(
        episode_chunk=episode_index // chunks_size,
        episode_index=episode_index,
        video_key=video_key,
    )
    if expected.is_file():
        return expected
    matches = sorted(
        (dataset_root / "videos").glob(
            f"chunk-*/{video_key}/episode_{episode_index:06d}.mp4"
        )
    )
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(
            f"Missing {video_key} video for episode {episode_index}: "
            f"expected {expected}"
        )
    raise ValueError(
        f"Multiple {video_key} videos found for episode {episode_index}: "
        f"{matches}"
    )


def _load_pillow() -> tuple[Any, Any, Any]:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as error:
        raise RuntimeError(
            "Pillow is required to render state-line audit contact sheets; "
            "install the project's pillow dependency"
        ) from error
    return Image, ImageDraw, ImageFont


def _resolve_ffmpeg_executable() -> str:
    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    try:
        import imageio_ffmpeg
    except ImportError as error:
        raise RuntimeError(
            "ffmpeg is required to decode LeRobot MP4 frames. Install ffmpeg "
            "or the project's imageio-ffmpeg dependency."
        ) from error
    executable = imageio_ffmpeg.get_ffmpeg_exe()
    if not executable or not Path(executable).is_file():
        raise RuntimeError(
            "imageio-ffmpeg did not provide a usable ffmpeg executable"
        )
    return executable


def load_video_frame(path: Path, frame_index: int) -> Any:
    """Decode one exact video frame through an ffmpeg subprocess."""

    if not path.is_file():
        raise FileNotFoundError(path)
    frame_index = _integer(frame_index, "frame_index")
    executable = _resolve_ffmpeg_executable()
    command = [
        executable,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(path),
        "-vf",
        f"select=eq(n\\,{frame_index})",
        "-frames:v",
        "1",
        "-f",
        "image2pipe",
        "-vcodec",
        "png",
        "pipe:1",
    ]
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            f"ffmpeg failed to decode frame {frame_index} from {path}: "
            f"{detail or 'unknown ffmpeg error'}"
        )
    if not result.stdout:
        raise RuntimeError(
            f"ffmpeg returned no image for frame {frame_index} from {path}"
        )
    Image, _, _ = _load_pillow()
    try:
        image = Image.open(io.BytesIO(result.stdout))
        image.load()
    except Exception as error:
        raise RuntimeError(
            f"ffmpeg output for frame {frame_index} from {path} is not a "
            "readable image"
        ) from error
    return image.convert("RGB")


def _fit_image(image: Any, size: tuple[int, int]) -> Any:
    Image, _, _ = _load_pillow()
    converted = image.convert("RGB")
    converted.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, (18, 18, 18))
    left = (size[0] - converted.width) // 2
    top = (size[1] - converted.height) // 2
    canvas.paste(converted, (left, top))
    return canvas


def render_contact_sheet(
    output_path: Path,
    selection: AuditSelection,
    frame_indices: Mapping[str, int],
    frames: Mapping[str, Mapping[str, Any]],
) -> None:
    """Write one two-camera by three-timepoint JPEG contact sheet."""

    Image, ImageDraw, ImageFont = _load_pillow()
    font = ImageFont.load_default()
    cell_width = 360
    image_height = 240
    label_height = 34
    header_height = 58
    margin = 12
    sheet_width = margin * 2 + cell_width * 3
    sheet_height = header_height + margin + (image_height + label_height) * 2
    sheet = Image.new("RGB", (sheet_width, sheet_height), (245, 245, 245))
    draw = ImageDraw.Draw(sheet)

    episode = selection.episode
    event = selection.event
    confidence = _confidence(event, f"event {event['event_id']}")
    header = (
        f"episode={episode['episode_id']}  split={episode['split']}  "
        f"outcome={episode['episode_outcome']}  "
        f"confidence={confidence:.4f}"
    )
    draw.text((margin, 10), header, fill=(20, 20, 20), font=font)
    draw.text(
        (margin, 31),
        f"event={event['event_id']}",
        fill=(55, 55, 55),
        font=font,
    )

    roles = ("before", "core", "after")
    for row_index, video_key in enumerate(CAMERA_KEYS):
        y = header_height + margin + row_index * (image_height + label_height)
        for column_index, role in enumerate(roles):
            x = margin + column_index * cell_width
            frame = _fit_image(frames[video_key][role], (cell_width, image_height))
            sheet.paste(frame, (x, y))
            label = (
                f"{video_key.rsplit('.', 1)[-1]} | {role} | "
                f"frame={frame_indices[role]}"
            )
            draw.rectangle(
                (x, y + image_height, x + cell_width, y + image_height + label_height),
                fill=(232, 232, 232),
            )
            draw.text(
                (x + 6, y + image_height + 9),
                label,
                fill=(20, 20, 20),
                font=font,
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, format="JPEG", quality=92, optimize=True)


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = (canonical_json(dict(payload)) + "\n").encode("utf-8")
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def render_audit(
    *,
    eve_root: Path,
    output_dir: Path,
    num_episodes: int,
    seed: int,
    frame_loader: FrameLoader = load_video_frame,
) -> dict[str, Any]:
    eve_root = eve_root.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    episode_path = eve_root / "episode_meta.jsonl"
    event_path = eve_root / "event_meta.jsonl"
    episode_rows = read_jsonl(episode_path)
    event_rows = read_jsonl(event_path)
    selected = select_audit_events(
        episode_rows,
        event_rows,
        num_episodes=num_episodes,
        seed=seed,
    )

    episode_hash = file_sha256(episode_path)
    event_hash = file_sha256(event_path)
    selection_input_hash = hashlib.sha256(
        canonical_json(
            {
                "episode_meta_sha256": episode_hash,
                "event_meta_sha256": event_hash,
                "num_episodes": num_episodes,
                "seed": seed,
            }
        ).encode("utf-8")
    ).hexdigest()

    strata_counts: dict[str, int] = defaultdict(int)
    index_rows: list[dict[str, Any]] = []
    for audit_index, selection in enumerate(selected):
        episode = selection.episode
        event = selection.event
        frame_indices = candidate_frame_indices(episode, event)
        video_paths = {
            video_key: locate_video(episode, video_key)
            for video_key in CAMERA_KEYS
        }
        frames = {
            video_key: {
                role: frame_loader(path, frame_index)
                for role, frame_index in frame_indices.items()
            }
            for video_key, path in video_paths.items()
        }
        episode_index = _integer(
            episode.get("episode_index"), "episode.episode_index"
        )
        sheet_name = (
            f"audit_{audit_index:03d}_episode_{episode_index:06d}.jpg"
        )
        sheet_path = output_dir / sheet_name
        render_contact_sheet(
            sheet_path,
            selection,
            frame_indices,
            frames,
        )
        split = str(episode["split"])
        outcome = str(episode["episode_outcome"])
        strata_counts[f"{split}/{outcome}"] += 1
        index_rows.append(
            {
                "audit_index": audit_index,
                "episode_id": str(episode["episode_id"]),
                "dataset_id": str(episode.get("dataset_id", "")),
                "dataset_root": str(
                    Path(str(episode["dataset_root"]))
                    .expanduser()
                    .resolve()
                ),
                "episode_index": episode_index,
                "split": split,
                "episode_outcome": outcome,
                "event_id": str(event["event_id"]),
                "absolute_confidence": _confidence(
                    event, f"event {event['event_id']}"
                ),
                "candidate_interval": [
                    int(event["start_frame"]),
                    int(event["end_frame"]),
                ],
                "core_interval": [
                    int(event.get("core_start_frame", event["start_frame"])),
                    int(event.get("core_end_frame", event["end_frame"])),
                ],
                "frames": dict(frame_indices),
                "videos": {
                    key: str(path) for key, path in video_paths.items()
                },
                "contact_sheet": sheet_name,
            }
        )

    payload = {
        "format": AUDIT_FORMAT,
        "version": AUDIT_VERSION,
        "inputs": {
            "eve_root": str(eve_root),
            "episode_meta": {
                "path": str(episode_path),
                "sha256": episode_hash,
            },
            "event_meta": {
                "path": str(event_path),
                "sha256": event_hash,
            },
            "selection_input_sha256": selection_input_hash,
        },
        "sampling": {
            "seed": seed,
            "num_episodes_requested": num_episodes,
            "num_episodes_selected": len(index_rows),
            "strata_counts": dict(sorted(strata_counts.items())),
            "policy": (
                "highest_absolute_confidence_candidate_per_episode_then_"
                "balanced_seeded_stratified_sampling"
            ),
        },
        "episodes": index_rows,
    }
    write_json_atomic(output_dir / "audit_index.json", payload)
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eve-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = render_audit(
            eve_root=args.eve_root,
            output_dir=args.output_dir,
            num_episodes=args.num_episodes,
            seed=args.seed,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
