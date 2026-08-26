#!/usr/bin/env bash
# Causal once-CFG at a fixed replan index on fail ∪ fragile (exclude both_ok).
#   FORCE_REPLAN=0 MAX_ENV_STEPS=500 GPUS=0,1,2,3 \
#     bash scripts/dewo_v2/eval_cfg_once_at_index.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
FORCE_REPLAN="${FORCE_REPLAN:?Set FORCE_REPLAN=0,1,2,...}"
SCREEN="${SCREEN:-search}"
MAX_ENV_STEPS="${MAX_ENV_STEPS:-500}"
GPUS="${GPUS:-0,1,2,3}"
CFG_SCALE="${CFG_SCALE:-1.2}"
BASE_PORT="${BASE_PORT:-$((13000 + FORCE_REPLAN * 100))}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-${ROOT}/evaluate_results/dexjoco/water_plant_dewo_v7_step1500_once_at${FORCE_REPLAN}_${SCREEN}_${STAMP}}"
export FORCE_REPLAN SCREEN MAX_ENV_STEPS GPUS CFG_SCALE BASE_PORT STAMP OUT_ROOT
bash "${ROOT}/scripts/dewo_v2/eval_cfg_once_intervene_failures.sh"
