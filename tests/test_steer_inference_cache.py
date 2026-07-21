from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch


ROOT = Path(__file__).resolve().parents[1]


def _load_server_module():
    hydra = types.ModuleType("hydra")
    hydra_utils = types.ModuleType("hydra.utils")
    hydra_utils.instantiate = lambda *_args, **_kwargs: None
    hydra.utils = hydra_utils

    omegaconf = types.ModuleType("omegaconf")
    omegaconf.OmegaConf = SimpleNamespace()

    policy_server = types.ModuleType("fastwam_policy_server")
    policy_server.DEFAULT_SERVER_PORT = 5555
    policy_server.PolicyServer = object

    policy_io = types.ModuleType("policy_io")
    for name, value in {
        "KEY_ACTION": "action",
        "KEY_CONTEXT": "context",
        "KEY_CONTEXT_MASK": "context_mask",
        "KEY_INPUT_IMAGE": "input_image",
        "KEY_PROMPT": "prompt",
        "KEY_PROPRIO": "proprio",
    }.items():
        setattr(policy_io, name, value)
    policy_io.to_inference_tensors = lambda *_args, **_kwargs: {}
    policy_io.validate_policy_observation = lambda *_args, **_kwargs: None

    stubs = {
        "hydra": hydra,
        "hydra.utils": hydra_utils,
        "omegaconf": omegaconf,
        "fastwam_policy_server": policy_server,
        "policy_io": policy_io,
    }
    previous = {name: sys.modules.get(name) for name in stubs}
    sys.modules.update(stubs)
    try:
        spec = importlib.util.spec_from_file_location(
            "run_fastwam_server_steer_cache_under_test",
            ROOT / "scripts" / "run_fastwam_server.py",
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        for name, original in previous.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


server = _load_server_module()


def _protocol(
    *,
    episodes: int = 1,
    replan_steps: int = 25,
    max_env_steps: int = 25,
    checkpoint_sha: str = "a" * 64,
    config_sha: str = "b" * 64,
):
    return {
        "schema": "fastwam.steer_protocol",
        "schema_version": 1,
        "task": "water_plant",
        "environment_seeds": {
            "global_base": 100,
            "global_end_exclusive": 100 + episodes,
            "shard_base": 100,
            "shard_end_exclusive": 100 + episodes,
        },
        "episodes": {
            "global_start": 0,
            "global_end_exclusive": episodes,
            "shard_id": 0,
            "shard_global_start": 0,
            "shard_global_end_exclusive": episodes,
            "local_start": 0,
            "local_end_exclusive": episodes,
        },
        "inference": {
            "seed": 314159,
            "replan_steps": replan_steps,
            "max_env_steps": max_env_steps,
            "max_requests_per_episode": (
                max_env_steps + replan_steps - 1
            ) // replan_steps,
            "control_mode": "blocking",
            "async_fallback": "wait",
            "action_horizon_override": None,
            "num_inference_steps_override": None,
        },
        "environment_options": {
            "randomize": False,
            "randomize_dynamics": False,
            "action_clip": False,
            "clip_max_xyz_step": 0.05,
            "clip_max_dz_down": 0.03,
            "task_config_dir": "/tmp/rand_obj",
        },
        "model": {
            "checkpoint_path": "/tmp/step_006000.pt",
            "checkpoint_sha256": checkpoint_sha,
            "config_path": "/tmp/config.yaml",
            "config_sha256": config_sha,
        },
    }


def _promote_recording_to_full_horizon(path: Path) -> None:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows[0]["coverage_policy"] = server.STEER_CACHE_COVERAGE_FULL
    rows[-1]["coverage_policy"] = server.STEER_CACHE_COVERAGE_FULL
    path.write_text(
        "".join(server._canonical_json(row) + "\n" for row in rows),
        encoding="utf-8",
    )


class SteerInferenceCacheTest(unittest.TestCase):
    @staticmethod
    def _make_policy(*, mode: str, cache=None, recorder=None):
        class Model:
            device = "cpu"
            torch_dtype = torch.float32

            def __init__(self):
                self.calls = []

            def infer_action(
                self,
                *,
                prompt,
                input_image,
                action_horizon,
                proprio,
                negative_prompt,
                text_cfg_scale,
                num_inference_steps,
                sigma_shift,
                seed,
                rand_device,
                tiled,
                steer_inference_mode=None,
                steer_embedding=None,
                return_steer_embedding=False,
            ):
                del (
                    prompt,
                    input_image,
                    proprio,
                    negative_prompt,
                    text_cfg_scale,
                    num_inference_steps,
                    sigma_shift,
                    seed,
                    rand_device,
                    tiled,
                )
                self.calls.append(
                    {
                        "mode": steer_inference_mode,
                        "embedding": steer_embedding,
                    }
                )
                result = {"action": torch.zeros(action_horizon, 2)}
                if return_steer_embedding:
                    result["steer_embedding"] = torch.tensor([[1.0, 2.0]])
                return result

        model = Model()
        processor = SimpleNamespace(shape_meta={})
        policy = server.FastWAMPolicy(
            model=model,
            processor=processor,
            device="cpu",
            action_horizon=2,
            num_inference_steps=1,
            steer_inference_mode=mode,
            steer_cache=cache,
            steer_cache_recorder=recorder,
        )
        policy._denormalize_action = lambda tensor: tensor.numpy()
        return policy, model

    def test_recorded_cache_roundtrip_is_keyed_and_hash_bound(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "steer.jsonl"
            checkpoint_sha = "a" * 64
            config_sha = "b" * 64
            recorder = server.SteerEmbeddingRecorder(
                path,
                checkpoint_sha256=checkpoint_sha,
                config_sha256=config_sha,
                embedding_dim=3,
                protocol=_protocol(episodes=2),
            )
            recorder.record(0, 0, torch.tensor([[1.0, 2.0, 3.0]]))
            recorder.record(1, 0, torch.tensor([[4.0, 5.0, 6.0]]))
            recorder.close()

            cache = server.SteerEmbeddingCache(
                path,
                expected_file_sha256=server._sha256_file(path),
                checkpoint_sha256=checkpoint_sha,
                config_sha256=config_sha,
                embedding_dim=3,
                protocol=_protocol(episodes=2),
                required_coverage_policy=server.STEER_CACHE_COVERAGE_OBSERVED,
            )

            self.assertTrue(
                torch.equal(cache.get(1, 0), torch.tensor([[4.0, 5.0, 6.0]]))
            )
            with self.assertRaisesRegex(KeyError, "Missing steer cache entry"):
                cache.get(0, 1)

    def test_cache_file_hash_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "steer.jsonl"
            recorder = server.SteerEmbeddingRecorder(
                path,
                checkpoint_sha256="a" * 64,
                config_sha256="b" * 64,
                embedding_dim=2,
                protocol=_protocol(),
            )
            recorder.record(0, 0, torch.tensor([[1.0, 2.0]]))
            recorder.close()

            with self.assertRaisesRegex(ValueError, "Steer cache SHA256 mismatch"):
                server.SteerEmbeddingCache(
                    path,
                    expected_file_sha256="0" * 64,
                    checkpoint_sha256="a" * 64,
                    config_sha256="b" * 64,
                    embedding_dim=2,
                    protocol=_protocol(),
                )

    def test_cache_checkpoint_or_shape_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "steer.jsonl"
            recorder = server.SteerEmbeddingRecorder(
                path,
                checkpoint_sha256="a" * 64,
                config_sha256="b" * 64,
                embedding_dim=2,
                protocol=_protocol(),
            )
            recorder.record(0, 0, torch.tensor([[1.0, 2.0]]))
            recorder.close()
            file_sha = server._sha256_file(path)

            with self.assertRaisesRegex(ValueError, "checkpoint_sha256"):
                server.SteerEmbeddingCache(
                    path,
                    expected_file_sha256=file_sha,
                    checkpoint_sha256="c" * 64,
                    config_sha256="b" * 64,
                    embedding_dim=2,
                    protocol=_protocol(checkpoint_sha="c" * 64),
                )
            with self.assertRaisesRegex(ValueError, "config_sha256"):
                server.SteerEmbeddingCache(
                    path,
                    expected_file_sha256=file_sha,
                    checkpoint_sha256="a" * 64,
                    config_sha256="c" * 64,
                    embedding_dim=2,
                    protocol=_protocol(config_sha="c" * 64),
                )
            with self.assertRaisesRegex(ValueError, "embedding_dim"):
                server.SteerEmbeddingCache(
                    path,
                    expected_file_sha256=file_sha,
                    checkpoint_sha256="a" * 64,
                    config_sha256="b" * 64,
                    embedding_dim=3,
                    protocol=_protocol(),
                )

    def test_cached_policy_uses_reset_and_replan_indices_deterministically(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "steer.jsonl"
            recorder = server.SteerEmbeddingRecorder(
                path,
                checkpoint_sha256="a" * 64,
                config_sha256="b" * 64,
                embedding_dim=2,
                protocol=_protocol(replan_steps=25, max_env_steps=50),
            )
            recorder.record(0, 0, torch.tensor([[1.0, 2.0]]))
            recorder.record(0, 1, torch.tensor([[3.0, 4.0]]))
            recorder.close()
            _promote_recording_to_full_horizon(path)
            cache = server.SteerEmbeddingCache(
                path,
                expected_file_sha256=server._sha256_file(path),
                checkpoint_sha256="a" * 64,
                config_sha256="b" * 64,
                embedding_dim=2,
                protocol=_protocol(replan_steps=25, max_env_steps=50),
            )
            policy, model = self._make_policy(mode="cached", cache=cache)
            tensors = {"input_image": torch.zeros(3, 16, 16), "prompt": "task"}
            with mock.patch.object(
                server,
                "to_inference_tensors",
                return_value=tensors,
            ):
                with self.assertRaisesRegex(RuntimeError, "requires reset"):
                    policy.get_action({})
                policy.reset()
                policy.get_action({})
                policy.get_action({})
                with self.assertRaisesRegex(KeyError, "Missing steer cache entry"):
                    policy.get_action({})

        self.assertEqual([call["mode"] for call in model.calls], ["explicit", "explicit"])
        self.assertTrue(
            torch.equal(model.calls[0]["embedding"], torch.tensor([[1.0, 2.0]]))
        )
        self.assertTrue(
            torch.equal(model.calls[1]["embedding"], torch.tensor([[3.0, 4.0]]))
        )

    def test_protocol_mismatch_and_incomplete_cache_fail_before_replay(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "steer.jsonl"
            protocol = _protocol()
            recorder = server.SteerEmbeddingRecorder(
                path,
                checkpoint_sha256="a" * 64,
                config_sha256="b" * 64,
                embedding_dim=2,
                protocol=protocol,
            )
            recorder.record(0, 0, torch.tensor([[1.0, 2.0]]))
            recorder.close()
            _promote_recording_to_full_horizon(path)
            file_sha = server._sha256_file(path)

            wrong_protocol = _protocol()
            wrong_protocol["inference"]["seed"] = 271828
            with self.assertRaisesRegex(ValueError, "header mismatch for protocol"):
                server.SteerEmbeddingCache(
                    path,
                    expected_file_sha256=file_sha,
                    checkpoint_sha256="a" * 64,
                    config_sha256="b" * 64,
                    embedding_dim=2,
                    protocol=wrong_protocol,
                )

            rows = path.read_text(encoding="utf-8").splitlines()
            incomplete = root / "incomplete.jsonl"
            incomplete.write_text("\n".join(rows[:-1]) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing completion footer"):
                server.SteerEmbeddingCache(
                    incomplete,
                    expected_file_sha256=server._sha256_file(incomplete),
                    checkpoint_sha256="a" * 64,
                    config_sha256="b" * 64,
                    embedding_dim=2,
                    protocol=protocol,
                )

    def test_full_horizon_cache_rejects_missing_request_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "steer.jsonl"
            protocol = _protocol(replan_steps=25, max_env_steps=50)
            recorder = server.SteerEmbeddingRecorder(
                path,
                checkpoint_sha256="a" * 64,
                config_sha256="b" * 64,
                embedding_dim=2,
                protocol=protocol,
            )
            recorder.record(0, 0, torch.tensor([[1.0, 2.0]]))
            recorder.close()
            _promote_recording_to_full_horizon(path)

            with self.assertRaisesRegex(ValueError, "full_horizon coverage mismatch"):
                server.SteerEmbeddingCache(
                    path,
                    expected_file_sha256=server._sha256_file(path),
                    checkpoint_sha256="a" * 64,
                    config_sha256="b" * 64,
                    embedding_dim=2,
                    protocol=protocol,
                )

    def test_policy_bypass_is_forwarded_without_explicit_embedding(self) -> None:
        policy, model = self._make_policy(mode="bypass")
        tensors = {"input_image": torch.zeros(3, 16, 16), "prompt": "task"}
        with mock.patch.object(server, "to_inference_tensors", return_value=tensors):
            policy.get_action({})

        self.assertEqual(model.calls[0]["mode"], "bypass")
        self.assertIsNone(model.calls[0]["embedding"])


if __name__ == "__main__":
    unittest.main()
