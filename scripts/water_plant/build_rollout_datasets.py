#!/usr/bin/env python3
"""Compatibility wrapper for the generic rollout dataset builder."""

from __future__ import annotations

import runpy
from pathlib import Path


if __name__ == "__main__":
    runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "build_rollout_datasets.py"),
        run_name="__main__",
    )
