from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from PIL import Image

from scripts.everobot import render_state_line_audit as audit


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def build_fixture(
    root: Path,
    *,
    episodes_per_stratum: int = 6,
) -> tuple[Path, list[dict[str, object]], list[dict[str, object]]]:
    eve_root = root / "eve"
    dataset_root = root / "dataset"
    eve_root.mkdir()
    (dataset_root / "meta").mkdir(parents=True)
    (dataset_root / "meta" / "info.json").write_text(
        json.dumps(
            {
                "chunks_size": 1000,
                "video_path": (
                    "videos/chunk-{episode_chunk:03d}/{video_key}/"
                    "episode_{episode_index:06d}.mp4"
                ),
            }
        ),
        encoding="utf-8",
    )

    episodes: list[dict[str, object]] = []
    events: list[dict[str, object]] = []
    episode_index = 0
    for split, outcome in audit.STRATA:
        for local_index in range(episodes_per_stratum):
            episode_id = (
                f"{split}-{outcome}-episode-{local_index:03d}"
            )
            episode = {
                "episode_id": episode_id,
                "dataset_id": "rollout-r0",
                "dataset_root": str(dataset_root),
                "episode_index": episode_index,
                "split": split,
                "episode_outcome": outcome,
                "length": 100,
            }
            episodes.append(episode)
            for candidate_index, confidence in enumerate((0.2, 0.8)):
                events.append(
                    {
                        "event_id": (
                            f"{episode_id}-candidate-{candidate_index}"
                        ),
                        "episode_id": episode_id,
                        "event_type": "interaction_candidate",
                        "split": split,
                        "episode_outcome": outcome,
                        "start_frame": 20 + candidate_index,
                        "end_frame": 40 + candidate_index,
                        "core_start_frame": 24 + candidate_index,
                        "core_end_frame": 30 + candidate_index,
                        "absolute_confidence": confidence,
                    }
                )
            for video_key in audit.CAMERA_KEYS:
                path = (
                    dataset_root
                    / "videos"
                    / "chunk-000"
                    / video_key
                    / f"episode_{episode_index:06d}.mp4"
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            episode_index += 1

    write_jsonl(eve_root / "episode_meta.jsonl", episodes)
    write_jsonl(eve_root / "event_meta.jsonl", events)
    return eve_root, episodes, events


class RenderStateLineAuditTest(unittest.TestCase):
    def test_selection_is_stratified_deterministic_and_uses_best_event(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary:
            eve_root, episodes, events = build_fixture(Path(temporary))
            del eve_root
            first = audit.select_audit_events(
                episodes,
                events,
                num_episodes=20,
                seed=17,
            )
            second = audit.select_audit_events(
                list(reversed(episodes)),
                list(reversed(events)),
                num_episodes=20,
                seed=17,
            )

            self.assertEqual(
                [row.event["event_id"] for row in first],
                [row.event["event_id"] for row in second],
            )
            counts = Counter(
                (
                    str(row.episode["split"]),
                    str(row.episode["episode_outcome"]),
                )
                for row in first
            )
            self.assertEqual(
                counts,
                Counter({stratum: 5 for stratum in audit.STRATA}),
            )
            self.assertTrue(
                all(
                    float(row.event["absolute_confidence"]) == 0.8
                    for row in first
                )
            )
            self.assertTrue(
                all(
                    str(row.event["event_id"]).endswith("candidate-1")
                    for row in first
                )
            )

    def test_render_audit_uses_mock_frames_and_writes_complete_index(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            eve_root, _, _ = build_fixture(
                root,
                episodes_per_stratum=2,
            )
            output_dir = root / "audit"
            calls: list[tuple[Path, int]] = []

            def frame_loader(path: Path, frame_index: int) -> Image.Image:
                calls.append((path, frame_index))
                color = (
                    frame_index % 255,
                    60 if "front" in str(path) else 120,
                    180,
                )
                return Image.new("RGB", (96, 72), color)

            result = audit.render_audit(
                eve_root=eve_root,
                output_dir=output_dir,
                num_episodes=4,
                seed=11,
                frame_loader=frame_loader,
            )

            self.assertEqual(len(calls), 4 * 2 * 3)
            self.assertEqual(
                result["sampling"]["strata_counts"],
                {
                    "train/failure": 1,
                    "train/success": 1,
                    "val/failure": 1,
                    "val/success": 1,
                },
            )
            self.assertEqual(
                result["sampling"]["num_episodes_selected"],
                4,
            )
            self.assertEqual(
                len(result["inputs"]["selection_input_sha256"]),
                64,
            )
            for row in result["episodes"]:
                self.assertEqual(
                    row["frames"],
                    {"before": 20, "core": 27, "after": 41},
                )
                self.assertEqual(
                    set(row["videos"]),
                    set(audit.CAMERA_KEYS),
                )
                sheet = output_dir / str(row["contact_sheet"])
                self.assertTrue(sheet.is_file())
                with Image.open(sheet) as image:
                    self.assertEqual(image.format, "JPEG")
                    self.assertGreater(image.width, image.height)

            stored = json.loads(
                (output_dir / "audit_index.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(stored, result)

    def test_frame_choice_prefers_explicit_peak_and_clamps_boundaries(
        self,
    ) -> None:
        episode = {"length": 30}
        event = {
            "start_frame": 0,
            "end_frame": 30,
            "core_start_frame": 8,
            "core_end_frame": 20,
            "core_peak_frame": 14,
        }
        self.assertEqual(
            audit.candidate_frame_indices(episode, event),
            {"before": 0, "core": 14, "after": 29},
        )

    def test_missing_stratum_is_rejected(self) -> None:
        with TemporaryDirectory() as temporary:
            _, episodes, events = build_fixture(
                Path(temporary),
                episodes_per_stratum=1,
            )
            episodes = [
                row
                for row in episodes
                if not (
                    row["split"] == "val"
                    and row["episode_outcome"] == "failure"
                )
            ]
            with self.assertRaisesRegex(
                ValueError,
                "No interaction_candidate episodes for val/failure",
            ):
                audit.select_audit_events(
                    episodes,
                    events,
                    num_episodes=4,
                    seed=0,
                )


if __name__ == "__main__":
    unittest.main()
