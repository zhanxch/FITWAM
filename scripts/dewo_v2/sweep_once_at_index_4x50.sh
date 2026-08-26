#!/usr/bin/env bash
# Causal 4×50 once-CFG sweep on early replans i=0..M-1, then 本体.
# After all i finish, rank policies: skip first k, fire at first i>=k (always,
# or later with energy X from logged gate E). Search uses max_steps=500.
#
#   GPUS=0,1,2,3 MAX_REPLAN_INDEX=4 CFG_SCALE=1.2 \
#     bash scripts/dewo_v2/sweep_once_at_index_4x50.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

TASK="${TASK:-water_plant}"
GPUS="${GPUS:-0,1,2,3}"
WAIT_IDLE="${WAIT_IDLE:-0}"
CFG_SCALE="${CFG_SCALE:-1.2}"
MAX_ENV_STEPS="${MAX_ENV_STEPS:-500}"
# i=0..MAX_REPLAN_INDEX inclusive → 5 nodes when MAX_REPLAN_INDEX=4
MAX_REPLAN_INDEX="${MAX_REPLAN_INDEX:-4}"
RUN_DIR="${RUN_DIR:-${ROOT}/runs/dexjoco_water_plant_dewo_v7/2026-08-25_12-51-39_B1-jump-fast-v7-uncond-adapter}"
CKPT="${CKPT:-${RUN_DIR}/checkpoints/weights/step_001500.pt}"
BACKBONE_CKPT="${BACKBONE_CKPT:-${ROOT}/artifacts/opensource_ckpt_links/mixed_5task/step_055000.pt}"
PRETRAINED_NORM_STATS="${PRETRAINED_NORM_STATS:-${ROOT}/artifacts/mixed_5task/dataset_stats.json}"
TEXT_EMBEDDING_CACHE_DIR="${TEXT_EMBEDDING_CACHE_DIR:-${ROOT}/data/water_plant_mixed_s0_dewo_v2_pair_20260820_182236/text_embeds_cache}"
CFG_TASK_DIR="${CFG_TASK_DIR:-${ROOT}/configs/eval/dexjoco/water_plant_dewo_v7_cfg}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
SWEEP_ROOT="${SWEEP_ROOT:-${ROOT}/evaluate_results/dexjoco/${TASK}_dewo_v7_step1500_once_at0to${MAX_REPLAN_INDEX}_4x50_max${MAX_ENV_STEPS}_${STAMP}}"
BASE_PORT="${BASE_PORT:-14000}"
BASELINE_AGG="${BASELINE_AGG:-${ROOT}/evaluate_results/dexjoco/water_plant_dewo_v7_step1500_cfg1_本体_4x50_20260825_135146/aggregate.json}"

mkdir -p "${SWEEP_ROOT}/schedules" "${SWEEP_ROOT}/logs"
echo "[sweep] root=${SWEEP_ROOT} i=0..${MAX_REPLAN_INDEX} max_steps=${MAX_ENV_STEPS} cfg=${CFG_SCALE}"

python3 - "${SWEEP_ROOT}/schedules" "${MAX_REPLAN_INDEX}" <<'PY'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
m = int(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
for idx in range(m + 1):
    by = {f"{rep}:{seed}": idx for rep in range(4) for seed in range(50)}
    path = out / f"force_replan_{idx}_flat.json"
    path.write_text(json.dumps(by, indent=2) + "\n")
    print(f"wrote {path} n={len(by)}")
PY

for i in $(seq 0 "${MAX_REPLAN_INDEX}"); do
  OUT_I="${SWEEP_ROOT}/at${i}"
  echo "[sweep] ===== FORCE_REPLAN=${i} OUT=${OUT_I} ====="
  TASK="${TASK}" \
  GPUS="${GPUS}" \
  WAIT_IDLE="${WAIT_IDLE}" \
  REPEATS=4 \
  ENV_SEED=0 \
  EPISODES=50 \
  RUN_DIR="${RUN_DIR}" \
  CKPT="${CKPT}" \
  BACKBONE_CKPT="${BACKBONE_CKPT}" \
  PRETRAINED_NORM_STATS="${PRETRAINED_NORM_STATS}" \
  TEXT_EMBEDDING_CACHE_DIR="${TEXT_EMBEDDING_CACHE_DIR}" \
  CFG_TASK_DIR="${CFG_TASK_DIR}" \
  CFG_SCALE="${CFG_SCALE}" \
  ADAPTIVE_CFG_TAU= \
  MAX_ENV_STEPS="${MAX_ENV_STEPS}" \
  BASE_PORT="$((BASE_PORT + i * 80))" \
  OUT_ROOT="${OUT_I}" \
  METHOD="dewo_v7_once_at${i}_4x50_max${MAX_ENV_STEPS}_SEARCH" \
  CFG_GATE_MODE=schedule \
  CFG_INTERVENE_SCHEDULE="${SWEEP_ROOT}/schedules/force_replan_${i}_flat.json" \
  bash "${ROOT}/scripts/dewo_v2/eval_cfg_official_4x50.sh"
done

python3 "${ROOT}/scripts/analysis/rank_4x50_once_at_index.py" \
  --本体-agg "${BASELINE_AGG}" \
  --本体-root "${ROOT}/evaluate_results/dexjoco/water_plant_dewo_v7_step1500_cfg1_本体_4x50_20260825_135146" \
  --sweep-root "${SWEEP_ROOT}" \
  --max-replan "${MAX_REPLAN_INDEX}" \
  --out "${SWEEP_ROOT}/rank_policies.json" \
  | tee "${SWEEP_ROOT}/logs/rank.log"

echo "[sweep] DONE ${SWEEP_ROOT}"
