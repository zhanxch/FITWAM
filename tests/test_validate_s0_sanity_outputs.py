from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "validate_s0_sanity_outputs_under_test",
    ROOT / "scripts" / "water_plant" / "validate_s0_sanity_outputs.py",
)
validate = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(validate)


def write_action_archive(path: Path, *, steps: int, nonfinite: bool = False) -> None:
    actions = np.zeros((steps, 22), dtype=np.float32)
    if nonfinite:
        actions[0, 0] = np.nan
    np.savez_compressed(
        path,
        initial_state=np.zeros((38,), dtype=np.float32),
        executed_actions=actions,
        raw_policy_actions=np.zeros((steps, 22), dtype=np.float32),
        low_pass_actions=np.zeros((steps, 22), dtype=np.float32),
        executed_is_fallback=np.zeros((steps,), dtype=bool),
        policy_query_steps=np.asarray([0], dtype=np.int32),
        policy_arrival_steps=np.asarray([0], dtype=np.int32),
        policy_latencies=np.asarray([0.1], dtype=np.float32),
        policy_chunks=np.zeros((1, 32, 22), dtype=np.float32),
        replan_steps=np.int32(25),
        action_horizon=np.int32(32),
        action_dim=np.int32(22),
    )


class ValidateS0SanityOutputsTest(unittest.TestCase):
    def fixture(self, root: Path, *, nonfinite_episode: int | None = None):
        protocol = {
            "model": {
                "config": {
                    "camera_keys": ["front", "wrist"],
                    "proprio_dim": 23,
                }
            },
            "collection": {"base_seed": 100, "max_env_steps": 1500},
        }
        protocol_path = root / "protocol.json"
        protocol_path.write_text(json.dumps(protocol), encoding="utf-8")

        episode_results = []
        for episode in range(4):
            steps = episode + 1
            video = root / f"episode_{episode}.mp4"
            video.write_bytes(b"video")
            actions = root / f"episode_{episode}_actions.npz"
            write_action_archive(
                actions,
                steps=steps,
                nonfinite=episode == nonfinite_episode,
            )
            episode_results.append(
                {
                    "episode": episode,
                    "seed": 100 + episode,
                    "steps": steps,
                    "success": episode % 2 == 0,
                    "video_path": str(video),
                    "actions_path": str(actions),
                }
            )
        summary = {
            "total_episodes": 4,
            "save_actions": True,
            "randomize": False,
            "randomize_dynamics": False,
            "action_clip": False,
            "tasks": [
                {
                    "env_name": "water_plant",
                    "episode_results": episode_results,
                }
            ],
        }
        summary_path = root / "summary.json"
        summary_path.write_text(json.dumps(summary), encoding="utf-8")
        return summary_path, protocol_path

    @staticmethod
    def fake_video_inspector(path: Path) -> dict:
        episode = int(path.stem.split("_")[-1])
        return {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "decoded_frames": episode + 2,
            "height": 384,
            "width": 384,
            "maximum_dynamic_range": 200,
        }

    def test_valid_sanity_outputs_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            summary, protocol = self.fixture(Path(tmp))
            report = validate.validate_sanity_outputs(
                summary,
                protocol,
                expected_episodes=4,
                video_inspector=self.fake_video_inspector,
            )
            self.assertEqual(report["status"], "valid")
            self.assertEqual(report["camera_keys"], ["front", "wrist"])
            self.assertEqual(report["model_proprio_dim"], 23)
            self.assertEqual(report["raw_initial_state_dim"], 38)
            self.assertEqual(len(report["episodes"]), 4)

    def test_nonfinite_action_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            summary, protocol = self.fixture(Path(tmp), nonfinite_episode=2)
            with self.assertRaisesRegex(ValueError, "NaN or infinite"):
                validate.validate_sanity_outputs(
                    summary,
                    protocol,
                    expected_episodes=4,
                    video_inspector=self.fake_video_inspector,
                )

    def test_protocol_without_wrist_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            summary, protocol = self.fixture(Path(tmp))
            payload = json.loads(protocol.read_text(encoding="utf-8"))
            payload["model"]["config"]["camera_keys"] = ["front"]
            protocol.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "front\\+wrist"):
                validate.validate_sanity_outputs(
                    summary,
                    protocol,
                    expected_episodes=4,
                    video_inspector=self.fake_video_inspector,
                )

    def test_model_proprio_dim_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            summary, protocol = self.fixture(Path(tmp))
            payload = json.loads(protocol.read_text(encoding="utf-8"))
            payload["model"]["config"]["proprio_dim"] = 38
            protocol.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "model proprio_dim must be 23"):
                validate.validate_sanity_outputs(
                    summary,
                    protocol,
                    expected_episodes=4,
                    video_inspector=self.fake_video_inspector,
                )

    def test_model_proprio_shape_is_not_accepted_as_raw_initial_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary, protocol = self.fixture(root)
            action_path = root / "episode_0_actions.npz"
            with np.load(action_path, allow_pickle=False) as archive:
                payload = {key: archive[key] for key in archive.files}
            payload["initial_state"] = np.zeros((23,), dtype=np.float32)
            np.savez_compressed(action_path, **payload)
            with self.assertRaisesRegex(ValueError, "raw initial_state must have shape"):
                validate.validate_sanity_outputs(
                    summary,
                    protocol,
                    expected_episodes=4,
                    video_inspector=self.fake_video_inspector,
                )
