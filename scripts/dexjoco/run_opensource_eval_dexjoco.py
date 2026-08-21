#!/usr/bin/env python3
"""Thin wrapper around OPEN eval_dexjoco.py with a local tasks.yaml overlay."""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

ROOT = Path(os.environ.get("FASTWAM_ROOT", Path(__file__).resolve().parents[2]))
OPEN = Path(os.environ.get("OPEN_REPO", str(ROOT.parent / "FastWAM-infer-in-DexJoco")))
TASKS_YAML = Path(
    os.environ.get(
        "FASTWAM_DEXJOCO_TASKS_YAML",
        str(ROOT / "configs/eval/dexjoco/opensource_baseline_tasks.yaml"),
    )
)

# Prefer OPEN src, then pin FastWAM, then dexjoco (already on PYTHONPATH from launcher).
sys.path.insert(0, str(OPEN / "src"))

import fastwam_dexjoco.tasks as tasks  # noqa: E402

# Default args are bound at function definition time; patch both the module
# constant and the function defaults so load_task_spec() uses our overlay.
tasks.DEFAULT_TASKS_CONFIG = TASKS_YAML
tasks.load_task_specs.__defaults__ = (TASKS_YAML,)
tasks.load_task_spec.__defaults__ = (TASKS_YAML,)

# Run OPEN evaluator as __main__ with current argv.
sys.argv[0] = str(OPEN / "scripts/eval_dexjoco.py")
runpy.run_path(str(OPEN / "scripts/eval_dexjoco.py"), run_name="__main__")
