#!/usr/bin/env bash
# Compatibility wrapper. Prefer:
#   TASK=fold_glasses GPUS=... bash scripts/dexjoco/eval_opensource_4x50.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export TASK="${TASK:-fold_glasses}"
exec bash "${ROOT}/scripts/dexjoco/eval_opensource_4x50.sh"
