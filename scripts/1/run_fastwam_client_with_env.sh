#!/usr/bin/env bash

# Example:
# scripts/1/run_fastwam_client_with_env.sh \
#   --host <server-ip> \
#   --port 5560 \
#   --task "Pick up the spray bottle, pump it to build up pressure, then spray water on the flowers" \
#   --execute-horizon 32 \
#   --no-home \
#   --eef-control-way direct \
#   --arm-interp-hz 30 \
#   --hand-interp-hz 30 \
#   --no-wbc \
#   --workspace-radius 2 2 2 \
#   --max-eef-step 0.05 \
#   --max-eef-rotation-step-deg 10 \
#   --actual-log-mode stream \
#   --actual-stream-hz 30 \
#   --log-prefix client_fastwam_direct_30hz_stream

set -o pipefail

REPO_ROOT="${FASTWAM_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
SDK="${ASTRIBOT_SDK_ROOT:-/home/zxc/cenyj/astribot_sdk/astribot_sdk_ros2-master}"
SHIM="${ASTRIBOT_PYTHON_SHIMS:-/home/zxc/cenyj/astribot_sdk/python_shims}"
WUJI_SETUP="${WUJI_HAND_SETUP:-/home/zxc/Desktop/wuji/wuji-teleop/wujihandros2/install/setup.bash}"
ROS_SETUP="${ROS_SETUP:-/opt/ros/humble/setup.bash}"
CLIENT="${REPO_ROOT}/scripts/1/run_fastwam_client.py"
RESET_RAN=0

source_if_present() {
    local path="$1"
    if [[ -f "${path}" ]]; then
        # shellcheck disable=SC1090
        source "${path}"
    else
        echo "[run_fastwam_client_with_env.sh][WARN] setup file not found: ${path}" >&2
    fi
}

activate_conda_env_if_present() {
    local env_name="$1"
    if declare -F conda >/dev/null 2>&1; then
        conda activate "${env_name}"
        return
    fi

    if command -v conda >/dev/null 2>&1; then
        local conda_base
        conda_base="$(conda info --base 2>/dev/null)" || return 1
        # shellcheck disable=SC1090
        source "${conda_base}/etc/profile.d/conda.sh"
        conda activate "${env_name}"
        return
    fi

    return 1
}

reset_robot_to_init() {
    (
        set +e
        cd "${SDK}" || return 1
        source_if_present "${SDK}/env.sh"
        source_if_present "${WUJI_SETUP}"
        if ! activate_conda_env_if_present astribot; then
            echo "[run_fastwam_client_with_env.sh][WARN] conda env not available: astribot" >&2
        fi
        python examples/112-move_to_init_joints_with_hands.py
    )
}

cleanup() {
    local status=$?
    trap - EXIT INT TERM

    if [[ "${RESET_RAN}" -eq 0 ]]; then
        RESET_RAN=1
        echo "[run_fastwam_client_with_env.sh] resetting robot to init joints..." >&2
        if ! reset_robot_to_init; then
            echo "[run_fastwam_client_with_env.sh][WARN] robot reset failed" >&2
        fi
    fi

    exit "${status}"
}

trap cleanup EXIT INT TERM

source_if_present "${ROS_SETUP}"
source_if_present "${SDK}/env.sh"
source_if_present "${SDK}/install/setup.sh"
source_if_present "${WUJI_SETUP}"

if [[ -f "${REPO_ROOT}/.venv/bin/activate" ]]; then
    # shellcheck disable=SC1090
    source "${REPO_ROOT}/.venv/bin/activate"
elif ! activate_conda_env_if_present astribot; then
    echo "[run_fastwam_client_with_env.sh][WARN] astribot conda env not activated" >&2
fi

export PYTHONPATH="${REPO_ROOT}/scripts:${REPO_ROOT}/scripts/1:${REPO_ROOT}/src:${REPO_ROOT}:${SHIM}:${SDK}/third_party/software/astribot_ros_middleware/lib/python3.10/site-packages:${SDK}/astribot_msgs/local/lib/python3.10/dist-packages:${SDK}:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${SDK}/astribot_msgs/lib:${SDK}/astribot_msgs/local/lib:${SDK}/astribot_sdk/core/common/robotics_library_py:${SDK}/astribot_sdk/core/common/whole_body_control/third_party:${SDK}/third_party/third_pkg/pinocchio/lib:${SDK}/third_party/drake/lib:${LD_LIBRARY_PATH:-}"

cd "${REPO_ROOT}" || exit 1
set -e

python - "${CLIENT}" "$@" <<'PY'
import importlib
import runpy
import sys

client_path = sys.argv[1]
client_args = sys.argv[2:]

pkg = importlib.import_module("robotics_library_py")
print("[bootstrap] robotics_library_py =", getattr(pkg, "__file__", None))
print("[bootstrap] robotics_library_py.__path__ =", list(getattr(pkg, "__path__", [])))

if not hasattr(pkg, "__path__"):
    raise RuntimeError("robotics_library_py was loaded as a module, not a package")

sys.argv = [client_path, *client_args]
runpy.run_path(client_path, run_name="__main__")
PY
