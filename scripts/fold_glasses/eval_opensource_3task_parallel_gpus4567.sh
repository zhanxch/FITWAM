#!/usr/bin/env bash
# Compatibility wrapper. Prefer:
#   TASKS=fold_glasses,hammer_nail,pick_bucket GPUS=... \
#     bash scripts/dexjoco/eval_opensource_4x50.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export TASKS="${TASKS:-fold_glasses,hammer_nail,pick_bucket}"
exec bash "${ROOT}/scripts/dexjoco/eval_opensource_4x50.sh"
