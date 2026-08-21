#!/usr/bin/env bash
# Compatibility wrapper (WAIT_IDLE=0). Prefer:
#   TASK=fold_glasses GPUS=... WAIT_IDLE=0 bash scripts/dexjoco/eval_opensource_4x50.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export TASK="${TASK:-fold_glasses}"
export WAIT_IDLE="${WAIT_IDLE:-0}"
exec bash "${ROOT}/scripts/dexjoco/eval_opensource_4x50.sh"
