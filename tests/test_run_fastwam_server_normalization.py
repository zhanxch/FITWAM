from __future__ import annotations

import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]


def _load_server_module():
    stubs: dict[str, types.ModuleType] = {}

    torch = types.ModuleType("torch")
    torch.Tensor = object
    torch.cuda = SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None)
    stubs["torch"] = torch

    hydra = types.ModuleType("hydra")
    hydra_utils = types.ModuleType("hydra.utils")
    hydra_utils.instantiate = lambda *_args, **_kwargs: None
    hydra.utils = hydra_utils
    stubs["hydra"] = hydra
    stubs["hydra.utils"] = hydra_utils

    omegaconf = types.ModuleType("omegaconf")
    omegaconf.OmegaConf = object
    stubs["omegaconf"] = omegaconf

    policy_server = types.ModuleType("fastwam_policy_server")
    policy_server.DEFAULT_SERVER_PORT = 5555
    policy_server.PolicyServer = object
    stubs["fastwam_policy_server"] = policy_server

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
    stubs["policy_io"] = policy_io

    previous = {name: sys.modules.get(name) for name in stubs}
    sys.modules.update(stubs)
    try:
        spec = importlib.util.spec_from_file_location(
            "run_fastwam_server_normalization_under_test",
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


def _write_meta(meta_dir: Path) -> None:
    meta_dir.mkdir(parents=True)
    (meta_dir / "stats.json").write_text("{}\n", encoding="utf-8")
    (meta_dir / "modality.json").write_text("{}\n", encoding="utf-8")


class RunFastWamServerNormalizationTest(unittest.TestCase):
    def test_explicit_meta_dir_overrides_stale_config_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir()
            meta_dir = Path(tmp) / "relocated-meta"
            _write_meta(meta_dir)
            processor_cfg = {
                "norm_stats_source": "meta",
                "norm_stats_meta_dir": "stale/meta",
            }

            kind, path = server._resolve_normalization_binding(
                processor_cfg,
                run_dir=run_dir,
                dataset_stats_path=None,
                norm_stats_meta_dir=str(meta_dir),
            )

            self.assertEqual(kind, "meta")
            self.assertEqual(path, meta_dir.resolve())
            self.assertEqual(processor_cfg["norm_stats_meta_dir"], str(meta_dir.resolve()))

    def test_meta_config_rejects_dataset_stats(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            stats = run_dir / "dataset_stats.json"
            stats.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "norm_stats_source=meta"):
                server._resolve_normalization_binding(
                    {"norm_stats_source": "meta"},
                    run_dir=run_dir,
                    dataset_stats_path=str(stats),
                    norm_stats_meta_dir=None,
                )

    def test_compute_config_rejects_meta_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            meta_dir = run_dir / "meta"
            _write_meta(meta_dir)
            with self.assertRaisesRegex(ValueError, "not allowed"):
                server._resolve_normalization_binding(
                    {"norm_stats_source": "compute"},
                    run_dir=run_dir,
                    dataset_stats_path=None,
                    norm_stats_meta_dir=str(meta_dir),
                )

    def test_meta_files_must_be_nonempty(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            meta_dir = run_dir / "meta"
            _write_meta(meta_dir)
            (meta_dir / "stats.json").write_bytes(b"")
            with self.assertRaisesRegex(FileNotFoundError, "non-empty"):
                server._resolve_normalization_binding(
                    {"norm_stats_source": "meta"},
                    run_dir=run_dir,
                    dataset_stats_path=None,
                    norm_stats_meta_dir=str(meta_dir),
                )

    def test_legacy_meta_config_path_still_works_outside_strict_s0_entrypoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            configured_meta = run_dir / "legacy-meta"
            _write_meta(configured_meta)
            kind, path = server._resolve_normalization_binding(
                {
                    "norm_stats_source": "meta",
                    "norm_stats_meta_dir": "legacy-meta",
                },
                run_dir=run_dir,
                dataset_stats_path=None,
                norm_stats_meta_dir=None,
            )
            self.assertEqual(kind, "meta")
            self.assertEqual(path, configured_meta.resolve())


if __name__ == "__main__":
    unittest.main()
