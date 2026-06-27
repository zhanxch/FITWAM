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

from archive.openloop.openloop_eval import run_openloop

register_default_resolvers()


@hydra.main(version_base="1.3", config_path="../../configs", config_name="openloop.yaml")
def main(cfg: DictConfig) -> None:
    run_openloop(cfg)


if __name__ == "__main__":
    main()
