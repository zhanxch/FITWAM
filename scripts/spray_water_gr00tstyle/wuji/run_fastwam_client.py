#!/usr/bin/env python3
"""Run a Wuji/Astribot real-robot client against a FastWAM policy server.

Uses the same robot control loop as ``run_gr00t_client.py`` but talks to
``scripts/spray_water_gr00tstyle/wuji/run_fastwam_server.py`` over the FastWAM ZMQ protocol.

Example:

  scripts/spray_water_gr00tstyle/wuji/run_fastwam_client_with_env.sh \\
    --host <server-ip> \\
    --port 5560 \\
    --task "Pick up the spray bottle, pump it to build up pressure, then spray water on the flowers" \\
    --execute-horizon 32 \\
    --no-home \\
    --eef-control-way direct \\
    --arm-interp-hz 30 \\
    --hand-interp-hz 30 \\
    --no-wbc \\
    --workspace-radius 2 2 2 \\
    --max-eef-step 0.05 \\
    --max-eef-rotation-step-deg 10
"""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_ROOT = SCRIPT_DIR.parent
for path in (SCRIPTS_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from fastwam_policy_server import DEFAULT_SERVER_PORT, PolicyClient
import run_gr00t_client as robot_client


def main() -> None:
    args = robot_client.parse_args(default_port=DEFAULT_SERVER_PORT)
    robot_client.run_robot_client(
        args,
        PolicyClient,
        server_label="FastWAM",
        node_name="fastwam_wuji_rot6d_client",
    )


if __name__ == "__main__":
    main()
