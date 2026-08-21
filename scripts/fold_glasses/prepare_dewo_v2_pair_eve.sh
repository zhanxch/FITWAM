#!/usr/bin/env bash
# Compatibility wrapper. Prefer:
#   TASK=fold_glasses PAIR_DATASET=... EXP_ROOT=... GPUS=... \
#     bash scripts/dewo_v2/prepare_pair_eve.sh
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export TASK="${TASK:-fold_glasses}"
exec bash "${ROOT_DIR}/scripts/dewo_v2/prepare_pair_eve.sh"
