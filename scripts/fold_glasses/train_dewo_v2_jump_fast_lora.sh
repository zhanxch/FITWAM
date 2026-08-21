#!/usr/bin/env bash
# Compatibility wrapper. Prefer:
#   source .../offline_v1_b1_jump_fast.env
#   TASK=fold_glasses GPUS=... bash scripts/dewo_v2/train_jump_fast_lora.sh
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export TASK="${TASK:-fold_glasses}"
exec bash "${ROOT_DIR}/scripts/dewo_v2/train_jump_fast_lora.sh"
