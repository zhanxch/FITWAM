from __future__ import annotations

import ast
import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]


def _load_eval_orchestrator():
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
    previous_utils = sys.modules.get("multi_gpu_eval_utils")
    previous_aggregator = sys.modules.get("eval_summary_aggregator")
    sys.modules["multi_gpu_eval_utils"] = utils
    sys.modules["eval_summary_aggregator"] = aggregator
    try:
        spec = importlib.util.spec_from_file_location(
            "run_multi_gpu_dexjoco_eval_under_test",
            ROOT / "scripts" / "dexjoco_async" / "run_multi_gpu_dexjoco_eval.py",
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        if previous_utils is None:
            sys.modules.pop("multi_gpu_eval_utils", None)
        else:
            sys.modules["multi_gpu_eval_utils"] = previous_utils
        if previous_aggregator is None:
            sys.modules.pop("eval_summary_aggregator", None)
        else:
            sys.modules["eval_summary_aggregator"] = previous_aggregator


def _declared_options(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    options = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "add_argument":
            continue
        for argument in node.args:
            if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                options.add(argument.value)
    return options


def _call_keyword_names(path: Path, function_name: str) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        called = node.func.id if isinstance(node.func, ast.Name) else None
        if called == function_name:
            names.update(keyword.arg for keyword in node.keywords if keyword.arg)
    return names


class S0NormalizationArgumentRoutingTest(unittest.TestCase):
    def test_eval_orchestrator_forwards_meta_override_to_server(self):
        orchestrator = _load_eval_orchestrator()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            meta_dir = root / "meta"
            meta_dir.mkdir()
            args = SimpleNamespace(
                server_script=root / "server.py",
                server_num_workers=8,
                mock=False,
                run_dir=root,
                checkpoint="step_006500.pt",
                dataset_stats_path=None,
                norm_stats_meta_dir=meta_dir,
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
            argv = orchestrator._build_server_argv(args, server)
            self.assertEqual(
                argv[argv.index("--norm-stats-meta-dir") + 1],
                str(meta_dir.resolve()),
            )
            self.assertNotIn("--dataset-stats-path", argv)

    def test_sync_and_async_servers_declare_and_forward_meta_override(self):
        sync_server = ROOT / "scripts" / "run_fastwam_server.py"
        async_server = ROOT / "scripts" / "run_fastwam_server_async.py"
        for path in (sync_server, async_server):
            self.assertIn("--norm-stats-meta-dir", _declared_options(path))
            self.assertIn(
                "norm_stats_meta_dir",
                _call_keyword_names(path, "_build_policy_from_run"),
            )

    def test_water_plant_scripts_route_both_sources_without_action_clipping(self):
        for relative in (
            "scripts/water_plant/sanity_s0_rollout_4.sh",
            "scripts/water_plant/collect_offline_s0_rollout_200.sh",
        ):
            text = (ROOT / relative).read_text(encoding="utf-8")
            self.assertIn("--dataset-stats-path", text)
            self.assertIn("--norm-stats-meta-dir", text)
            self.assertIn("--no-action-clip", text)


if __name__ == "__main__":
    unittest.main()
