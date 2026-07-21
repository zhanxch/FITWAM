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


ROOT = Path(__file__).resolve().parents[1]


class _Config(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value


def _as_config(value):
    if isinstance(value, dict) and not isinstance(value, _Config):
        return _Config({key: _as_config(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_as_config(item) for item in value]
    return value


class _OmegaConfStub:
    loaded_config = None

    @classmethod
    def load(cls, _path):
        return cls.loaded_config

    @staticmethod
    def create(value):
        return _as_config(value)

    @staticmethod
    def to_container(value, resolve=True):
        del resolve
        return value


def _load_sync_server():
    torch = types.ModuleType("torch")
    torch.Tensor = object
    torch.cuda = SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None)

    hydra = types.ModuleType("hydra")
    hydra_utils = types.ModuleType("hydra.utils")
    hydra_utils.instantiate = lambda *_args, **_kwargs: None
    hydra.utils = hydra_utils

    omegaconf = types.ModuleType("omegaconf")
    omegaconf.OmegaConf = _OmegaConfStub

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
        "torch": torch,
        "hydra": hydra,
        "hydra.utils": hydra_utils,
        "omegaconf": omegaconf,
        "fastwam_policy_server": policy_server,
        "policy_io": policy_io,
    }
    with mock.patch.dict(sys.modules, stubs):
        spec = importlib.util.spec_from_file_location(
            "run_fastwam_server_seed_under_test",
            ROOT / "scripts" / "run_fastwam_server.py",
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
    return module


SYNC_SERVER = _load_sync_server()


class _FakeModel:
    def load_checkpoint(self, path):
        self.checkpoint = path

    def eval(self):
        return self


class _FakeProcessor:
    shape_meta = {}

    def eval(self):
        return self

    def set_normalizer_from_stats(self, stats):
        self.stats = stats


def _runtime_stubs() -> dict[str, types.ModuleType]:
    packages = {}
    for name in (
        "fastwam",
        "fastwam.datasets",
        "fastwam.datasets.lerobot",
        "fastwam.datasets.lerobot.processors",
        "fastwam.datasets.lerobot.utils",
    ):
        package = types.ModuleType(name)
        package.__path__ = []
        packages[name] = package

    processor_module = types.ModuleType(
        "fastwam.datasets.lerobot.processors.fastwam_processor"
    )
    processor_module.FastWAMProcessor = _FakeProcessor
    normalizer_module = types.ModuleType(
        "fastwam.datasets.lerobot.utils.normalizer"
    )
    normalizer_module.load_dataset_stats_from_json = lambda _path: {"ok": True}
    runtime_module = types.ModuleType("fastwam.runtime")
    runtime_module._normalize_mixed_precision = lambda value: value
    runtime_module._mixed_precision_to_model_dtype = lambda _value: "fake-dtype"
    return {
        **packages,
        processor_module.__name__: processor_module,
        normalizer_module.__name__: normalizer_module,
        runtime_module.__name__: runtime_module,
    }


def _build_policy_with_seed(
    config_seed: int | None,
    inference_seed: int | None,
    *,
    steer_inference_mode: str = "learned",
):
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        (run_dir / "config.yaml").write_text("{}\n", encoding="utf-8")
        checkpoint = run_dir / "checkpoints" / "weights" / "step_000001.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"checkpoint")
        (run_dir / "dataset_stats.json").write_text("{}\n", encoding="utf-8")

        _OmegaConfStub.loaded_config = _as_config(
            {
                "model": {"kind": "model"},
                "data": {
                    "train": {
                        "num_frames": 33,
                        "processor": {
                            "kind": "processor",
                            "norm_stats_source": "compute",
                        },
                    }
                },
                "mixed_precision": "bf16",
                "eval_num_inference_steps": 10,
                "EVALUATION": {
                    "seed": config_seed,
                    "rand_device": "cpu",
                },
            }
        )

        def instantiate(config, **_kwargs):
            return _FakeModel() if config.get("kind") == "model" else _FakeProcessor()

        with (
            mock.patch.dict(sys.modules, _runtime_stubs()),
            mock.patch.object(SYNC_SERVER, "instantiate", side_effect=instantiate),
        ):
            return SYNC_SERVER._build_policy_from_run(
                run_dir=run_dir,
                checkpoint=str(checkpoint),
                dataset_stats_path=None,
                norm_stats_meta_dir=None,
                device="cpu",
                action_horizon=None,
                num_inference_steps=None,
                load_text_encoder=False,
                inference_seed=inference_seed,
                steer_inference_mode=steer_inference_mode,
            )


def _load_async_server():
    async_policy_server = types.ModuleType("fastwam_policy_server_async")
    async_policy_server.DEFAULT_ASYNC_SERVER_PORT = 5556

    class PolicyServerAsync:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def run(self):
            return None

    async_policy_server.PolicyServerAsync = PolicyServerAsync
    with mock.patch.dict(
        sys.modules,
        {
            "fastwam_policy_server_async": async_policy_server,
            "run_fastwam_server": SYNC_SERVER,
        },
    ):
        spec = importlib.util.spec_from_file_location(
            "run_fastwam_server_async_seed_under_test",
            ROOT / "scripts" / "run_fastwam_server_async.py",
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
    return module


ASYNC_SERVER = _load_async_server()


def _load_orchestrator():
    utils = types.ModuleType("multi_gpu_eval_utils")
    utils.ServerSpec = type("ServerSpec", (), {})
    utils.ShardSpec = type("ShardSpec", (), {})
    for name in (
        "build_conda_command",
        "find_free_ports",
        "launch_subprocess",
        "locate_conda_sh",
        "shard_episodes",
        "terminate_process",
        "wait_for_server",
    ):
        setattr(utils, name, lambda *_args, **_kwargs: None)
    aggregator = types.ModuleType("eval_summary_aggregator")
    aggregator.merge_shard_summaries = lambda *_args, **_kwargs: None
    aggregator.write_combined = lambda *_args, **_kwargs: None
    with mock.patch.dict(
        sys.modules,
        {
            "multi_gpu_eval_utils": utils,
            "eval_summary_aggregator": aggregator,
        },
    ):
        spec = importlib.util.spec_from_file_location(
            "run_multi_gpu_dexjoco_eval_seed_under_test",
            ROOT / "scripts" / "dexjoco_async" / "run_multi_gpu_dexjoco_eval.py",
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
    return module


ORCHESTRATOR = _load_orchestrator()


def _model_provenance(root: Path) -> dict:
    return {
        "run_dir": str(root),
        "checkpoint_path": str(root / "step_000001.pt"),
        "checkpoint_sha256": "a" * 64,
        "config_path": str(root / "config.yaml"),
        "config_sha256": "b" * 64,
    }


def _steer_protocol_args(root: Path) -> dict:
    return {
        "tasks": ["water_plant"],
        "episodes": 8,
        "seed": 100,
        "replan_steps": 25,
        "max_env_steps": 1500,
        "control_mode": "blocking",
        "async_fallback": "wait",
        "randomize": False,
        "randomize_dynamics": False,
        "action_clip": False,
        "clip_max_xyz_step": 0.05,
        "clip_max_dz_down": 0.03,
        "task_config_dir": root / "rand_obj",
        "_model_provenance": _model_provenance(root),
    }


class DexJoCoInferenceSeedRoutingTest(unittest.TestCase):
    def test_policy_builder_seed_override_wins_over_config(self):
        policy = _build_policy_with_seed(config_seed=17, inference_seed=0)
        self.assertEqual(policy.seed, 0)

    def test_policy_builder_falls_back_to_config_seed(self):
        policy = _build_policy_with_seed(config_seed=17, inference_seed=None)
        self.assertEqual(policy.seed, 17)

    def test_bypass_builder_still_loads_the_full_checkpoint(self):
        policy = _build_policy_with_seed(
            config_seed=17,
            inference_seed=0,
            steer_inference_mode="bypass",
        )

        self.assertEqual(policy.steer_inference_mode, "bypass")
        self.assertEqual(Path(policy.model.checkpoint).name, "step_000001.pt")

    def test_async_server_forwards_inference_seed_to_policy_builder(self):
        base_policy = SimpleNamespace(
            model=object(),
            processor=object(),
            device="cpu",
            action_horizon=32,
            num_inference_steps=10,
            num_video_frames=33,
            text_cfg_scale=1.0,
            negative_prompt="",
            sigma_shift=None,
            seed=314,
            rand_device="cpu",
            tiled=False,
        )
        args = SimpleNamespace(
            mock=False,
            run_dir="/tmp/run",
            checkpoint="step_000001.pt",
            dataset_stats_path=None,
            norm_stats_meta_dir=None,
            device="cpu",
            action_horizon=None,
            num_inference_steps=None,
            inference_seed=314,
            load_text_encoder=False,
            host="127.0.0.1",
            port=5556,
            api_token=None,
            num_workers=2,
            steer_cache_path=None,
            steer_cache_record_path=None,
            steer_protocol_json=None,
        )
        with (
            mock.patch.object(ASYNC_SERVER, "parse_args", return_value=args),
            mock.patch.object(
                ASYNC_SERVER,
                "_build_policy_from_run",
                return_value=base_policy,
            ) as build_policy,
            mock.patch.object(
                ASYNC_SERVER,
                "_resolve_run_dir",
                side_effect=lambda path: path,
            ),
        ):
            ASYNC_SERVER.main()
        self.assertEqual(build_policy.call_args.kwargs["inference_seed"], 314)
        self.assertEqual(
            build_policy.call_args.kwargs["steer_inference_mode"],
            "learned",
        )
        self.assertIsNone(build_policy.call_args.kwargs["steer_cache_path"])

    def test_multi_gpu_servers_receive_the_same_inference_seed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = SimpleNamespace(
                server_script=root / "server.py",
                server_num_workers=8,
                mock=False,
                run_dir=root,
                checkpoint="step_000001.pt",
                dataset_stats_path=None,
                norm_stats_meta_dir=None,
                action_horizon=None,
                num_inference_steps=None,
                inference_seed=2718,
                load_text_encoder=False,
                api_token=None,
            )
            servers = [
                SimpleNamespace(device="cuda", bind_host="0.0.0.0", port=5570),
                SimpleNamespace(device="cuda", bind_host="0.0.0.0", port=5571),
            ]
            argvs = [
                ORCHESTRATOR._build_server_argv(args, server)
                for server in servers
            ]
        for argv in argvs:
            self.assertEqual(argv.count("--inference-seed"), 1)
            self.assertEqual(argv[argv.index("--inference-seed") + 1], "2718")

    def test_combined_summary_records_inference_seed(self):
        args = SimpleNamespace(
            client_host="127.0.0.1",
            seed=100,
            inference_seed=2718,
            tasks=["water_plant"],
            max_env_steps=1500,
            video_fps=30,
            randomize=False,
            randomize_dynamics=False,
            save_video=True,
            save_actions=True,
            action_clip=False,
            clip_max_xyz_step=0.05,
            clip_max_dz_down=0.03,
            _model_provenance=_model_provenance(Path("/tmp/run")),
        )
        metadata = ORCHESTRATOR._combined_summary_metadata(
            args,
            gpus=[0, 1, 2, 3],
            ports=[5570, 5571, 5572, 5573],
        )
        self.assertEqual(metadata["inference_seed"], 2718)
        self.assertEqual(metadata["steer_inference"]["mode"], "learned")
        self.assertEqual(len(metadata["steer_inference"]["servers"]), 4)
        self.assertEqual(
            metadata["model_provenance"]["checkpoint_sha256"],
            "a" * 64,
        )
        args.inference_seed = None
        metadata = ORCHESTRATOR._combined_summary_metadata(
            args,
            gpus=[0, 1, 2, 3],
            ports=[5570, 5571, 5572, 5573],
        )
        self.assertIsNone(metadata["inference_seed"])

    def test_cached_steer_routes_shard_path_hash_and_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = SimpleNamespace(
                server_script=root / "server.py",
                server_num_workers=8,
                mock=False,
                run_dir=root,
                checkpoint="step_000001.pt",
                dataset_stats_path=None,
                norm_stats_meta_dir=None,
                action_horizon=None,
                num_inference_steps=None,
                inference_seed=2718,
                load_text_encoder=False,
                api_token=None,
                steer_inference_mode="cached",
                steer_cache_path=str(root / "cache-{shard}-{gpu}-{port}.jsonl"),
                steer_cache_sha256=f"{'a' * 64},{'b' * 64}",
                steer_cache_record_path=None,
                **_steer_protocol_args(root),
            )
            server = SimpleNamespace(
                gpu=5,
                device="cuda",
                bind_host="0.0.0.0",
                port=5571,
            )
            shard = SimpleNamespace(
                shard_id=1,
                base_seed=102,
                num_episodes=2,
                global_episode_start=2,
            )

            argv = ORCHESTRATOR._build_server_argv(
                args, server, shard_id=1, shard=shard
            )

        self.assertEqual(argv[argv.index("--steer-inference-mode") + 1], "cached")
        self.assertTrue(
            argv[argv.index("--steer-cache-path") + 1].endswith(
                "cache-1-5-5571.jsonl"
            )
        )
        self.assertEqual(
            argv[argv.index("--steer-cache-sha256") + 1],
            "b" * 64,
        )
        self.assertEqual(argv[argv.index("--num-workers") + 1], "1")
        protocol = json.loads(argv[argv.index("--steer-protocol-json") + 1])
        self.assertEqual(protocol["task"], "water_plant")
        self.assertEqual(protocol["environment_seeds"]["shard_base"], 102)
        self.assertEqual(protocol["episodes"]["shard_global_start"], 2)
        self.assertEqual(protocol["inference"]["max_requests_per_episode"], 60)

    def test_bypass_and_learned_recording_are_first_class(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = {
                "server_script": root / "server.py",
                "server_num_workers": 8,
                "mock": False,
                "run_dir": root,
                "checkpoint": "step_000001.pt",
                "dataset_stats_path": None,
                "norm_stats_meta_dir": None,
                "action_horizon": None,
                "num_inference_steps": None,
                "inference_seed": 2718,
                "load_text_encoder": False,
                "api_token": None,
                "steer_cache_path": None,
                "steer_cache_sha256": None,
                **_steer_protocol_args(root),
            }
            server = SimpleNamespace(
                gpu=4,
                device="cuda",
                bind_host="0.0.0.0",
                port=5570,
            )
            bypass = ORCHESTRATOR._build_server_argv(
                SimpleNamespace(
                    **base,
                    steer_inference_mode="bypass",
                    steer_cache_record_path=None,
                ),
                server,
            )
            learned = ORCHESTRATOR._build_server_argv(
                SimpleNamespace(
                    **base,
                    steer_inference_mode="learned",
                    steer_cache_record_path=str(root / "record-{shard}.jsonl"),
                ),
                server,
                shard_id=2,
                shard=SimpleNamespace(
                    shard_id=2,
                    base_seed=104,
                    num_episodes=2,
                    global_episode_start=4,
                ),
            )

        self.assertEqual(
            bypass[bypass.index("--steer-inference-mode") + 1],
            "bypass",
        )
        self.assertEqual(
            learned[learned.index("--steer-inference-mode") + 1],
            "learned",
        )
        self.assertTrue(
            learned[learned.index("--steer-cache-record-path") + 1].endswith(
                "record-2.jsonl"
            )
        )
        self.assertEqual(learned[learned.index("--num-workers") + 1], "1")
        self.assertIn("--steer-protocol-json", learned)

    def test_no_launch_servers_fails_closed_for_real_model(self):
        with mock.patch.object(
            ORCHESTRATOR,
            "parse_args",
            return_value=SimpleNamespace(launch_servers=False, mock=False),
        ):
            with self.assertRaisesRegex(ValueError, "disabled for real-model"):
                ORCHESTRATOR.main()

    def test_async_cache_mode_rejects_concurrent_workers(self):
        args = SimpleNamespace(
            mock=False,
            steer_cache_path="/tmp/cache.jsonl",
            steer_cache_record_path=None,
            num_workers=2,
        )
        with mock.patch.object(ASYNC_SERVER, "parse_args", return_value=args):
            with self.assertRaisesRegex(ValueError, "requires --num-workers 1"):
                ASYNC_SERVER.main()

    def test_omitting_flag_preserves_cli_and_server_argv_behavior(self):
        with mock.patch.object(sys, "argv", ["server", "--mock"]):
            self.assertIsNone(SYNC_SERVER.parse_args().inference_seed)
            self.assertEqual(SYNC_SERVER.parse_args().steer_inference_mode, "learned")
        with mock.patch.object(sys, "argv", ["server-async", "--mock"]):
            self.assertIsNone(ASYNC_SERVER.parse_args().inference_seed)
            self.assertEqual(ASYNC_SERVER.parse_args().steer_inference_mode, "learned")
        with mock.patch.object(
            sys,
            "argv",
            [
                "orchestrator",
                "--gpus",
                "0",
                "--episodes",
                "1",
                "--run-dir",
                "/tmp/run",
                "--replan-steps",
                "25",
                "--output-dir",
                "/tmp/out",
            ],
        ):
            self.assertIsNone(ORCHESTRATOR.parse_args().inference_seed)
            self.assertEqual(
                ORCHESTRATOR.parse_args().steer_inference_mode,
                "learned",
            )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            legacy_args = SimpleNamespace(
                server_script=root / "server.py",
                server_num_workers=8,
                mock=False,
                run_dir=root,
                checkpoint="step_000001.pt",
                dataset_stats_path=None,
                norm_stats_meta_dir=None,
                action_horizon=None,
                num_inference_steps=None,
                load_text_encoder=False,
                api_token=None,
            )
            server = SimpleNamespace(
                device="cuda",
                bind_host="0.0.0.0",
                port=5570,
            )
            argv = ORCHESTRATOR._build_server_argv(legacy_args, server)
        self.assertNotIn("--inference-seed", argv)


if __name__ == "__main__":
    unittest.main()
