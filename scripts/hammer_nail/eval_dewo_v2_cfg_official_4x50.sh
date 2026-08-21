#!/usr/bin/env bash
# Compatibility wrapper. Prefer:
#   TASK=hammer_nail RUN_DIR=... CKPT=... TEXT_EMBEDDING_CACHE_DIR=... GPUS=... \
#     bash scripts/dewo_v2/eval_cfg_official_4x50.sh
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export TASK="${TASK:-hammer_nail}"
exec bash "${ROOT_DIR}/scripts/dewo_v2/eval_cfg_official_4x50.sh"
