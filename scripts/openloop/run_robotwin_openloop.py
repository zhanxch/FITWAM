"""Deprecated compatibility entrypoint for historical RoboTwin open-loop commands.

Use `scripts/openloop/run_openloop.py` with `configs/openloop*.yaml` for new runs.
This shim keeps older scripts and notebooks working.
"""

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fastwam.utils.config_resolvers import register_default_resolvers

OPENLOOP_DIR = Path(__file__).resolve().parent
if str(OPENLOOP_DIR) not in sys.path:
    sys.path.insert(0, str(OPENLOOP_DIR))

from openloop_eval import run_openloop

register_default_resolvers()


@hydra.main(version_base="1.3", config_path="../../configs", config_name="openloop_robotwin.yaml")
def main(cfg: DictConfig) -> None:
    run_openloop(cfg)


if __name__ == "__main__":
    main()
