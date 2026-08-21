#!/usr/bin/env bash
# Compatibility wrapper. Prefer:
#   TASK=fold_glasses SOURCE_ROOT=... GPUS=... bash scripts/dewo_v2/run_pair_pipeline.sh
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export TASK="${TASK:-fold_glasses}"
exec bash "${ROOT_DIR}/scripts/dewo_v2/run_pair_pipeline.sh"
