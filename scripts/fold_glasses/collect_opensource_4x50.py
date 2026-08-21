#!/usr/bin/env python3
"""Deprecated path. Use scripts/dexjoco/collect_opensource_4x50.py."""
from pathlib import Path
import runpy

if __name__ == "__main__":
    runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "dexjoco" / "collect_opensource_4x50.py"),
        run_name="__main__",
    )
