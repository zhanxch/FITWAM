#!/usr/bin/env bash
#
# B2 diagnostic: run FastWAM real-robot client with deploy post-processing
# DISABLED to isolate whether interpolation / workspace clip / filtering (H5)
# is the cause of the sim-vs-real gap.
#
# This script is a thin wrapper around run_fastwam_client_with_env.sh that
# INJECTS diagnostic flags unless the caller overrides them:
#   --execute-horizon 1        : execute only the first action of each chunk
#                                (skips multi-step interpolation between chunk steps)
#   --arm-interp-hz 0          : disable EEF SLERP interpolation between chunks
#   --hand-interp-hz 0         : disable hand interpolation between chunks
#   --eef-control-way direct   : no low-pass filter
#   --max-eef-step 1.0         : effectively no per-step position clip
#   --max-eef-rotation-step-deg 180 : effectively no per-step rotation clip
#   --workspace-radius 5 5 5   : wide workspace, effectively no workspace clip
#   --dump-raw-actions <dir>   : save raw policy actions before post-processing (B1)
#
# Comparison protocol:
#   1. Run the NORMAL launch (run_fastwam_client_with_env.sh with your usual
#      flags) and record the failure (video + logs).
#   2. Run THIS script with the same server/checkpoint.
#   3. Compare:
#        - If symptoms PERSIST (still stuck/oscillating/drops bottle) with
#          post-processing disabled -> the action OUTPUT itself is wrong;
#          H5 (post-processing) is NOT the cause. Points to H1 (no proprio) /
#          H2 (rot6d warp) / H3 (decoupled experts).
#        - If symptoms IMPROVE or disappear -> H5 (post-processing) is a
#          contributing cause; tune interpolation/clip/filter.
#
# Usage:
#   scripts/wuji/run_fastwam_client_diagnostic.sh \
#       --host <server-ip> --port 5560 \
#       --task "Pick up the spray bottle, pump it to build up pressure, then spray water on the flowers" \
#       --no-home --no-wbc \
#       --log-dir runs/diag_b2_nopostproc \
#       --log-prefix client_fastwam_diag_nopostproc
#
# The server should be launched with:
#   python scripts/wuji/run_fastwam_server.py \
#       --run-dir runs/spray_water_rot6d_rosbag_ts_filter_uncond_3cam_384_1e-4/<latest> \
#       --checkpoint <ckpt> \
#       --dump-dir runs/diag_b2_nopostproc/server_dump
#
set -o pipefail

REPO_ROOT="${FASTWAM_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
WRAPPER="${REPO_ROOT}/scripts/wuji/run_fastwam_client_with_env.sh"

if [[ ! -f "${WRAPPER}" ]]; then
    echo "[run_fastwam_client_diagnostic.sh][ERROR] wrapper not found: ${WRAPPER}" >&2
    exit 1
fi

# Default diagnostic flags (B2: disable post-processing).
# These are PREPENDED so that any user-provided value for the same flag wins
# (argparse takes the last occurrence).
DIAG_FLAGS=(
    --execute-horizon 1
    --arm-interp-hz 0
    --hand-interp-hz 0
    --eef-control-way direct
    --max-eef-step 1.0
    --max-eef-rotation-step-deg 180
    --workspace-radius 5 5 5
)

# B1 dump: save raw policy actions. Defaults to a dir under the log-dir if set,
# else runs/diag_dump/client_raw. User can override by passing --dump-raw-actions.
DUMP_DIR=""
for arg in "$@"; do
    if [[ "${arg}" == "--dump-raw-actions" ]]; then
        DUMP_DIR="__user_set__"
    fi
done
if [[ "${DUMP_DIR}" != "__user_set__" ]]; then
    # try to extract --log-dir to colocate the dump
    LOG_DIR=""
    PREV=""
    for arg in "$@"; do
        if [[ "${PREV}" == "--log-dir" ]]; then
            LOG_DIR="${arg}"
        fi
        PREV="${arg}"
    done
    if [[ -n "${LOG_DIR}" ]]; then
        DUMP_DIR="${LOG_DIR}/client_raw"
    else
        DUMP_DIR="runs/diag_dump/client_raw"
    fi
    DIAG_FLAGS+=(--dump-raw-actions "${DUMP_DIR}")
fi

echo "[run_fastwam_client_diagnostic.sh] B2: post-processing DISABLED."
echo "[run_fastwam_client_diagnostic.sh] injected flags: ${DIAG_FLAGS[*]}"
echo "[run_fastwam_client_diagnostic.sh] raw action dump -> ${DUMP_DIR}"
echo "[run_fastwam_client_diagnostic.sh] forwarding to: ${WRAPPER}"
echo

exec bash "${WRAPPER}" "${DIAG_FLAGS[@]}" "$@"
