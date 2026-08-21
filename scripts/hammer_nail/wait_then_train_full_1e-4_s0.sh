#!/usr/bin/env bash
# Compatibility wrapper. Prefer:
#   TASK=hammer_nail ENV_FILE=... GPUS=... TRAIN_LAUNCHER=... \
#     bash scripts/dewo_v2/wait_then_train.sh
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export TASK="${TASK:-hammer_nail}"
export TRAIN_LAUNCHER="${TRAIN_LAUNCHER:-scripts/hammer_nail/train_dewo_v2_jump_fast_full_1e-4_s0.sh}"
exec bash "${ROOT_DIR}/scripts/dewo_v2/wait_then_train.sh"
