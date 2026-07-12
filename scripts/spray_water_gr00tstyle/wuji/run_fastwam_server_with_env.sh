#!/usr/bin/env bash

# Example:
# scripts/spray_water_gr00tstyle/wuji/run_fastwam_server_with_env.sh \
#   --run-dir runs/spray_water_rot6d_rosbag_ts_filter_uncond_3cam_384_1e-4/2026-06-17_16-31-58 \
#   --checkpoint checkpoints/weights/step_017025.pt \
#   --device cuda:0 \
#   --host 0.0.0.0 \
#   --port 5560

set -o pipefail

REPO_ROOT="${FASTWAM_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
SDK="${ASTRIBOT_SDK_ROOT:?Set ASTRIBOT_SDK_ROOT to the Astribot SDK root}"
SHIM="${ASTRIBOT_PYTHON_SHIMS:?Set ASTRIBOT_PYTHON_SHIMS to the Python shims directory}"
WUJI_SETUP="${WUJI_HAND_SETUP:?Set WUJI_HAND_SETUP to the Wuji ROS setup.bash}"
ROS_SETUP="${ROS_SETUP:-/opt/ros/humble/setup.bash}"
SERVER="${REPO_ROOT}/scripts/spray_water_gr00tstyle/wuji/run_fastwam_server.py"

source_if_present() {
    local path="$1"
    if [[ -f "${path}" ]]; then
        # shellcheck disable=SC1090
        source "${path}"
    else
        echo "[run_fastwam_server_with_env.sh][WARN] setup file not found: ${path}" >&2
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

source_if_present "${ROS_SETUP}"
source_if_present "${SDK}/env.sh"
source_if_present "${SDK}/install/setup.sh"
source_if_present "${WUJI_SETUP}"

if [[ -f "${REPO_ROOT}/.venv/bin/activate" ]]; then
    # shellcheck disable=SC1090
    source "${REPO_ROOT}/.venv/bin/activate"
elif ! activate_conda_env_if_present fastwam; then
    echo "[run_fastwam_server_with_env.sh][WARN] FastWAM virtualenv/conda env not activated" >&2
fi

export PYTHONPATH="${REPO_ROOT}/scripts:${REPO_ROOT}/scripts/spray_water_gr00tstyle:${REPO_ROOT}/scripts/spray_water_gr00tstyle/wuji:${REPO_ROOT}/src:${REPO_ROOT}:${SHIM}:${SDK}/third_party/software/astribot_ros_middleware/lib/python3.10/site-packages:${SDK}/astribot_msgs/local/lib/python3.10/dist-packages:${SDK}:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${SDK}/astribot_msgs/lib:${SDK}/astribot_msgs/local/lib:${SDK}/astribot_sdk/core/common/robotics_library_py:${SDK}/astribot_sdk/core/common/whole_body_control/third_party:${SDK}/third_party/third_pkg/pinocchio/lib:${SDK}/third_party/drake/lib:${LD_LIBRARY_PATH:-}"

cd "${REPO_ROOT}" || exit 1
set -e

exec python "${SERVER}" "$@"
