from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import pickle
import random
import sys
import tempfile
import types
import unittest
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "everobot"
    / "build_common_init_checkpoint.py"
)
SPEC = importlib.util.spec_from_file_location("build_common_init_checkpoint", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
common_init = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = common_init
SPEC.loader.exec_module(common_init)


class FakeTensor:
    def __init__(self, values, *, shape=None, dtype="float32"):
        self.values = tuple(float(value) for value in values)
        self.shape = tuple(shape or (len(self.values),))
        self.dtype = dtype
        if math.prod(self.shape) != len(self.values):
            raise ValueError("shape does not match values")

    def detach(self):
        return self

    def cpu(self):
        return self

    def clone(self):
        return copy.deepcopy(self)


class FakeBoolean:
    def __init__(self, value: bool):
        self.value = bool(value)

    def all(self):
        return self.value

    def __bool__(self):
        return self.value


class FakeScalar:
    def __init__(self, value: int):
        self.value = int(value)

    def item(self):
        return self.value


class FakeTorch:
    bfloat16 = "bfloat16"
    float16 = "float16"
    float32 = "float32"

    def __init__(self, events):
        self.events = events
        self.seed = 0
        self.cuda = types.SimpleNamespace(
            is_available=lambda: True,
            manual_seed_all=lambda seed: self.events.append(("cuda_seed", int(seed))),
        )

    def manual_seed(self, seed):
        self.seed = int(seed)
        self.events.append(("torch_seed", self.seed))

    @staticmethod
    def save(payload, path):
        with Path(path).open("wb") as stream:
            pickle.dump(payload, stream, protocol=4)

    @staticmethod
    def load(path, **_kwargs):
        with Path(path).open("rb") as stream:
            return pickle.load(stream)

    @staticmethod
    def equal(left, right):
        return (
            left.shape == right.shape
            and left.dtype == right.dtype
            and left.values == right.values
        )

    @staticmethod
    def isfinite(value):
        return FakeBoolean(all(math.isfinite(item) for item in value.values))

    @staticmethod
    def count_nonzero(value):
        return FakeScalar(sum(item != 0.0 for item in value.values))


class FakeNumpy:
    def __init__(self, events):
        self.random = types.SimpleNamespace(
            seed=lambda seed: events.append(("numpy_seed", int(seed)))
        )


class FakeOmegaConf:
    @staticmethod
    def load(path):
        return json.loads(Path(path).read_text(encoding="utf-8"))

    @staticmethod
    def resolve(_cfg):
        return None

    @staticmethod
    def to_container(cfg, resolve=True):
        del resolve
        return copy.deepcopy(cfg)


class FakeModule:
    def __init__(self, state):
        self._state = copy.deepcopy(state)

    def state_dict(self):
        return self._state

    def load_state_dict(self, state, strict=True):
        del strict
        self._state = copy.deepcopy(state)


class FakeModel:
    def __init__(self, torch_module, *, student_value):
        self.torch = torch_module
        self.mot = FakeModule(
            {
                "mixtures.video.weight": FakeTensor([91.0, 92.0], shape=(1, 2)),
                "mixtures.action.weight": FakeTensor([93.0, 94.0], shape=(1, 2)),
            }
        )
        self.proprio_encoder = FakeModule(
            {"weight": FakeTensor([81.0, 82.0], shape=(1, 2))}
        )
        self.outcome_encoder = None
        self.offline_steer_enabled = True
        self.offline_steer_config = {
            "enabled": True,
            "hidden_dim": 8,
            "embedding_dim": 4,
            "num_heads": 1,
            "dropout": 0.0,
            "detach_backbone_inputs": True,
            "pair_loss_weight": 0.1,
            "pair_loss_margin": 0.2,
            "pair_loss_warmup_steps": 5,
        }
        self.offline_steer_student = FakeModule(
            {"projection.weight": FakeTensor([student_value, student_value + 1.0])}
        )
        self.offline_steer_residual = FakeModule(
            {
                "projection.weight": FakeTensor([0.0, 0.0]),
                "projection.bias": FakeTensor([0.0]),
            }
        )
        self.torch_dtype = "bfloat16"

    def load_checkpoint(self, path, optimizer=None, experts=None):
        del optimizer, experts
        payload = self.torch.load(path)
        self.mot.load_state_dict(payload["mot"])
        self.proprio_encoder.load_state_dict(payload["proprio_encoder"])
        return payload

    def save_checkpoint(self, path, optimizer=None, step=None):
        del optimizer
        self.torch.save(
            {
                "mot": self.mot.state_dict(),
                "step": step,
                "torch_dtype": str(self.torch_dtype),
                "proprio_encoder": self.proprio_encoder.state_dict(),
                "offline_steer_student": self.offline_steer_student.state_dict(),
                "offline_steer_residual": self.offline_steer_residual.state_dict(),
                "offline_steer_config": dict(self.offline_steer_config),
            },
            path,
        )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_config(root: Path, baseline: Path, baseline_hash: str, *, seed=42) -> Path:
    config = {
        "seed": seed,
        "mixed_precision": "bf16",
        "resume": str(baseline.resolve()),
        "resume_experts": None,
        "experiment_provenance": {
            "source_checkpoint_sha256": baseline_hash,
        },
        "model": {
            "_target_": "fastwam.runtime.create_fastwam",
            "skip_dit_load_from_pretrain": True,
            "offline_steer": {
                "enabled": True,
                "hidden_dim": 8,
                "embedding_dim": 4,
                "num_heads": 1,
                "dropout": 0.0,
                "detach_backbone_inputs": True,
                "pair_loss_weight": 0.1,
                "pair_loss_margin": 0.2,
                "pair_loss_warmup_steps": 5,
            },
        },
    }
    path = root / "resolved.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def make_baseline(root: Path, torch_module: FakeTorch, *, steer=False, mismatch=False):
    mot = {
        "mixtures.video.weight": FakeTensor([1.0, 2.0], shape=(1, 2)),
        "mixtures.action.weight": FakeTensor(
            [3.0, 4.0, 5.0] if mismatch else [3.0, 4.0],
            shape=(1, 3) if mismatch else (1, 2),
        ),
    }
    payload = {
        "mot": mot,
        "proprio_encoder": {"weight": FakeTensor([5.0, 6.0], shape=(1, 2))},
        "step": 6500,
        "torch_dtype": "torch.bfloat16",
    }
    if steer:
        payload["offline_steer_student"] = {
            "projection.weight": FakeTensor([9.0, 9.0])
        }
        payload["offline_steer_residual"] = {
            "projection.weight": FakeTensor([9.0, 9.0])
        }
        payload["offline_steer_config"] = {"enabled": True}
    path = root / "s0.pt"
    torch_module.save(payload, path)
    return path


def make_args(root: Path, config: Path, baseline: Path, suffix="a"):
    return argparse.Namespace(
        resolved_config=config,
        baseline_checkpoint=baseline,
        output=root / f"common-init-{suffix}.pt",
        proof_output=root / f"common-init-{suffix}.proof.json",
        seed=42,
        model_dtype="auto",
        device="cuda:0",
        expected_config_sha256=sha256(config),
        expected_baseline_sha256=sha256(baseline),
    )


def make_dependencies(events):
    fake_torch = FakeTorch(events)
    fake_numpy = FakeNumpy(events)

    def instantiate(_model_cfg, *, model_dtype, device):
        events.append(("instantiate", fake_torch.seed, model_dtype, device))
        rng = random.Random(fake_torch.seed)
        return FakeModel(fake_torch, student_value=rng.random())

    dependencies = common_init.RuntimeDependencies(
        torch=fake_torch,
        numpy=fake_numpy,
        omega_conf=FakeOmegaConf,
        instantiate=instantiate,
    )
    return dependencies, fake_torch


class CommonInitializationCheckpointTest(unittest.TestCase):
    def test_builds_complete_seeded_artifact_and_proof(self):
        events = []
        dependencies, fake_torch = make_dependencies(events)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = make_baseline(root, fake_torch)
            config = make_config(root, baseline, sha256(baseline))
            args = make_args(root, config, baseline)

            report = common_init.run(args, dependencies=dependencies)

            self.assertEqual(
                events[:3],
                [("numpy_seed", 42), ("torch_seed", 42), ("cuda_seed", 42)],
            )
            self.assertEqual(events[3], ("instantiate", 42, "bfloat16", "cuda:0"))
            self.assertTrue(args.output.is_file())
            self.assertTrue(args.proof_output.is_file())

            payload = fake_torch.load(args.output)
            baseline_payload = fake_torch.load(baseline)
            self.assertEqual(payload["mot"].keys(), baseline_payload["mot"].keys())
            for key in payload["mot"]:
                self.assertTrue(fake_torch.equal(payload["mot"][key], baseline_payload["mot"][key]))
            self.assertIn("offline_steer_student", payload)
            self.assertIn("offline_steer_residual", payload)
            self.assertEqual(payload["step"], 0)
            self.assertNotIn("optimizer", payload)
            self.assertEqual(report["output"]["checkpoint"]["sha256"], sha256(args.output))
            self.assertEqual(
                report["proof_sha256"],
                hashlib.sha256(
                    common_init._canonical_json_bytes(
                        {key: value for key, value in report.items() if key != "proof_sha256"}
                    )
                ).hexdigest(),
            )
            proof = json.loads(args.proof_output.read_text(encoding="utf-8"))
            self.assertEqual(proof, report)
            self.assertTrue(report["invariants"]["steer_unchanged_by_s0_load"])
            self.assertTrue(report["invariants"]["complete_weight_only_checkpoint"])

    def test_same_seed_produces_identical_common_initialization(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            events_a = []
            deps_a, torch_a = make_dependencies(events_a)
            baseline = make_baseline(root, torch_a)
            config = make_config(root, baseline, sha256(baseline))
            args_a = make_args(root, config, baseline, "a")
            report_a = common_init.run(args_a, dependencies=deps_a)

            events_b = []
            deps_b, _torch_b = make_dependencies(events_b)
            args_b = make_args(root, config, baseline, "b")
            report_b = common_init.run(args_b, dependencies=deps_b)

            self.assertEqual(args_a.output.read_bytes(), args_b.output.read_bytes())
            self.assertEqual(
                report_a["output"]["checkpoint"]["sections"],
                report_b["output"]["checkpoint"]["sections"],
            )

    def test_rejects_baseline_with_existing_steer_weights(self):
        events = []
        dependencies, fake_torch = make_dependencies(events)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = make_baseline(root, fake_torch, steer=True)
            config = make_config(root, baseline, sha256(baseline))
            args = make_args(root, config, baseline)

            with self.assertRaisesRegex(ValueError, "must be steer-free"):
                common_init.run(args, dependencies=dependencies)
            self.assertFalse(args.output.exists())
            self.assertFalse(args.proof_output.exists())

    def test_rejects_partial_or_shape_mismatched_s0(self):
        events = []
        dependencies, fake_torch = make_dependencies(events)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = make_baseline(root, fake_torch, mismatch=True)
            config = make_config(root, baseline, sha256(baseline))
            args = make_args(root, config, baseline)

            with self.assertRaisesRegex(ValueError, "not exactly compatible"):
                common_init.run(args, dependencies=dependencies)
            self.assertFalse(args.output.exists())

    def test_refuses_overwrite_before_model_instantiation(self):
        events = []
        dependencies, fake_torch = make_dependencies(events)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = make_baseline(root, fake_torch)
            config = make_config(root, baseline, sha256(baseline))
            args = make_args(root, config, baseline)
            args.output.write_bytes(b"keep-me")

            with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
                common_init.run(args, dependencies=dependencies)
            self.assertEqual(args.output.read_bytes(), b"keep-me")
            self.assertFalse(args.proof_output.exists())
            self.assertNotIn("instantiate", [event[0] for event in events])

    def test_rejects_config_that_is_not_bound_to_the_supplied_s0(self):
        events = []
        dependencies, fake_torch = make_dependencies(events)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = make_baseline(root, fake_torch)
            config = make_config(root, baseline, sha256(baseline))
            payload = json.loads(config.read_text(encoding="utf-8"))
            payload["resume"] = str(root / "different.pt")
            config.write_text(json.dumps(payload), encoding="utf-8")
            args = make_args(root, config, baseline)

            with self.assertRaisesRegex(ValueError, "does not bind"):
                common_init.run(args, dependencies=dependencies)
            self.assertFalse(args.output.exists())


if __name__ == "__main__":
    unittest.main()
