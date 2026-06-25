#!/usr/bin/env python3
"""Web-based subtask annotator for DexJoCo LeRobot datasets.

Split each episode video into segments and label left/right hand subtasks per segment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
from urllib.parse import unquote

from flask import Flask, abort, jsonify, render_template, request, send_file

DEFAULT_DATASET = Path("data/dexjoco_microwave_cook")
ANNOTATION_FILENAME = "dual_hand_subtasks.jsonl"

LEFT_SUBTASKS = [
    "Open the microwave door",
    "nothing to do",
    "close the microwave door",
]

RIGHT_SUBTASKS = [
    "pick up the food",
    "place the grasped food inside the microwave",
    "move out",
    "nothing to do",
    "press the start button",
]

DEFAULT_TASK = (
    "Open the microwave door, place the food inside the microwave, "
    "close the door, and press the start button."
)

CAMERAS = (
    "observation.images.ego",
    "observation.images.wrist_left",
    "observation.images.wrist_right",
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


class DatasetAnnotator:
    def __init__(self, dataset_root: Path) -> None:
        self.dataset_root = dataset_root.resolve()
        self.meta_dir = self.dataset_root / "meta"
        self.info = _read_json(self.meta_dir / "info.json")
        self.fps = int(self.info.get("fps", 30))
        self.episodes = _read_jsonl(self.meta_dir / "episodes.jsonl")
        self.tasks = {
            row["task_index"]: row["task"]
            for row in _read_jsonl(self.meta_dir / "tasks.jsonl")
        }
        self.annotation_path = self.dataset_root / "annotations" / ANNOTATION_FILENAME
        self.annotations = {
            row["episode_index"]: row
            for row in _read_jsonl(self.annotation_path)
        }

    def episode_task(self, episode_index: int) -> str:
        ep = self.episodes[episode_index]
        if ep.get("tasks"):
            return ep["tasks"][0]
        task_idx = ep.get("task_index", 0)
        return self.tasks.get(task_idx, DEFAULT_TASK)

    def video_path(self, episode_index: int, camera: str) -> Path:
        if camera not in CAMERAS:
            abort(400, description=f"Unknown camera: {camera}")
        chunk = episode_index // int(self.info.get("chunks_size", 1000))
        rel = self.info["video_path"].format(
            episode_chunk=chunk,
            video_key=camera,
            episode_index=episode_index,
        )
        path = self.dataset_root / rel
        if not path.exists():
            abort(404, description=f"Video not found: {path}")
        return path

    def episode_payload(self, episode_index: int) -> dict[str, Any]:
        if episode_index < 0 or episode_index >= len(self.episodes):
            abort(404, description=f"Episode {episode_index} not found")
        ep = self.episodes[episode_index]
        saved = self.annotations.get(episode_index, {})
        return {
            "episode_index": episode_index,
            "length": ep["length"],
            "task": saved.get("task") or self.episode_task(episode_index),
            "splits": saved.get("splits", []),
            "segments": saved.get("segments", []),
        }

    def save_annotation(self, payload: dict[str, Any]) -> None:
        episode_index = int(payload["episode_index"])
        if episode_index < 0 or episode_index >= len(self.episodes):
            abort(404, description=f"Episode {episode_index} not found")

        segments = payload.get("segments", [])
        for seg in segments:
            if seg.get("left_subtask") and seg["left_subtask"] not in LEFT_SUBTASKS:
                abort(400, description=f"Invalid left_subtask: {seg['left_subtask']}")
            if seg.get("right_subtask") and seg["right_subtask"] not in RIGHT_SUBTASKS:
                abort(400, description=f"Invalid right_subtask: {seg['right_subtask']}")

        row = {
            "episode_index": episode_index,
            "task": payload.get("task") or self.episode_task(episode_index),
            "length": int(payload.get("length") or self.episodes[episode_index]["length"]),
            "splits": [int(s) for s in payload.get("splits", [])],
            "segments": [
                {
                    "start_frame": int(seg["start_frame"]),
                    "end_frame": int(seg["end_frame"]),
                    "left_subtask": seg.get("left_subtask", ""),
                    "right_subtask": seg.get("right_subtask", ""),
                }
                for seg in segments
            ],
        }
        self.annotations[episode_index] = row
        ordered = [self.annotations[i] for i in sorted(self.annotations)]
        _write_jsonl(self.annotation_path, ordered)


def create_app(dataset_root: Path) -> Flask:
    app = Flask(
        __name__,
        template_folder=str(Path(__file__).resolve().parent / "templates"),
    )
    annotator = DatasetAnnotator(dataset_root)

    @app.get("/")
    def index() -> str:
        return render_template(
            "dexjoco_subtask_annotator.html",
            left_subtasks=LEFT_SUBTASKS,
            right_subtasks=RIGHT_SUBTASKS,
            cameras=CAMERAS,
            fps=annotator.fps,
        )

    @app.get("/api/episodes")
    def api_episodes() -> Any:
        return jsonify(
            [
                {
                    "episode_index": ep["episode_index"],
                    "length": ep["length"],
                    "annotated": ep["episode_index"] in annotator.annotations,
                }
                for ep in annotator.episodes
            ]
        )

    @app.get("/api/episode/<int:episode_index>")
    def api_episode(episode_index: int) -> Any:
        return jsonify(annotator.episode_payload(episode_index))

    @app.get("/video/<int:episode_index>/<path:camera>")
    def serve_video(episode_index: int, camera: str) -> Any:
        camera = unquote(camera)
        path = annotator.video_path(episode_index, camera)
        return send_file(path, mimetype="video/mp4", conditional=True)

    @app.post("/api/annotation/<int:episode_index>")
    def save_annotation(episode_index: int) -> Any:
        payload = request.get_json(force=True)
        payload["episode_index"] = episode_index
        annotator.save_annotation(payload)
        return jsonify({"ok": True, "path": str(annotator.annotation_path)})

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="Path to DexJoCo LeRobot dataset root.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6678)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset.resolve()
    if not (dataset_root / "meta" / "episodes.jsonl").exists():
        raise SystemExit(f"Dataset not found: {dataset_root}")

    app = create_app(dataset_root)
    print(f"Dataset: {dataset_root}")
    print(f"Annotations: {dataset_root / 'annotations' / ANNOTATION_FILENAME}")
    print(f"Open http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)


if __name__ == "__main__":
    main()
