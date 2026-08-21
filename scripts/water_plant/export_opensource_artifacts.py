#!/usr/bin/env python3
"""Thin water_plant wrapper around the shared DEWO v2 artifact exporter."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from dewo_v2.export_opensource_artifacts import main

if __name__ == "__main__":
    if "--task" not in sys.argv:
        sys.argv[1:1] = ["--task", "water_plant"]
    raise SystemExit(main())
