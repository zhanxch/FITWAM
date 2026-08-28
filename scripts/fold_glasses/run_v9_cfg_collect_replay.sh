#!/usr/bin/env bash
# Deploy gate test: v9_base vs value_growth CFG at collect t* (secondary to oracle-once base eval).
# Prefer base eval: bash scripts/dewo_v2/eval_v9_collect_event_replay.sh
# This wrapper: CONDITIONS=v9_base,v9_cfg
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export TASK="${TASK:-fold_glasses}"
export GPUS="${GPUS:-4,5,6,7}"
export CONDITIONS="${CONDITIONS:-v9_base,v9_cfg}"
export RUN_DIR="${RUN_DIR:-${ROOT_DIR}/runs/dexjoco_fold_glasses_dewo_v9/2026-08-27_11-21-16_B1-jump-fast-v9-uncond-adapter}"
exec bash "${ROOT_DIR}/scripts/dewo_v2/eval_v9_collect_event_replay.sh"
