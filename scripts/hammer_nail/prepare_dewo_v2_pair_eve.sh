#!/usr/bin/env bash
# Compatibility wrapper. Prefer:
#   TASK=hammer_nail PAIR_DATASET=... EXP_ROOT=... GPUS=... \
#     bash scripts/dewo_v2/prepare_pair_eve.sh
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export TASK="${TASK:-hammer_nail}"
exec bash "${ROOT_DIR}/scripts/dewo_v2/prepare_pair_eve.sh"
