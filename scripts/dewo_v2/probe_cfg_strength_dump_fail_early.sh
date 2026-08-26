#!/usr/bin/env bash
# Dump-only CFG strength on existing replay obs: failures + early replans only.
# Skips success episodes and late failure nodes (past recoverable window).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

EVAL_ROOT="${EVAL_ROOT:-${ROOT}/evaluate_results/dexjoco/water_plant_dewo_v7_step1500_cfg1_本体_4x50_20260825_135146}"
PROBE_ROOT="${PROBE_ROOT:-${ROOT}/evaluate_results/dexjoco/water_plant_dewo_v7_step1500_cfg_strength_probe_20260825}"
RUN_DIR="${RUN_DIR:-${ROOT}/runs/dexjoco_water_plant_dewo_v7/2026-08-25_12-51-39_B1-jump-fast-v7-uncond-adapter}"
CKPT="${CKPT:-${RUN_DIR}/checkpoints/weights/step_001500.pt}"
BACKBONE="${BACKBONE_CKPT:-${ROOT}/artifacts/opensource_ckpt_links/mixed_5task/step_055000.pt}"
STATS="${PRETRAINED_NORM_STATS:-${ROOT}/artifacts/mixed_5task/dataset_stats.json}"
TEXT_CACHE="${TEXT_EMBEDDING_CACHE_DIR:-${ROOT}/data/water_plant_mixed_s0_dewo_v2_pair_20260820_182236/text_embeds_cache}"
CFG_TASK_DIR="${CFG_TASK_DIR:-${ROOT}/configs/eval/dexjoco/water_plant_dewo_v7_cfg}"
GPUS="${GPUS:-0,1,2,3}"
CFG_SCALE="${CFG_SCALE:-1.2}"
PROBE_TAU="${PROBE_TAU:-1e9}"
# Past water_plant success lengths (~max 364); late 1000-step failure nodes are not rescue targets.
MAX_QUERY_STEP="${MAX_QUERY_STEP:-400}"

IFS=',' read -r -a GPU_ARR <<< "${GPUS}"
mkdir -p "${PROBE_ROOT}/logs"

echo "[probe-dump] PROBE_ROOT=${PROBE_ROOT} failures_only max_query_step=${MAX_QUERY_STEP}"
test -f "${CKPT}"
test -f "${BACKBONE}"
for i in 1 2 3 4; do
  test -d "${PROBE_ROOT}/run${i}/obs" || { echo "missing replay obs run${i}"; exit 2; }
  # Drop partial/full residuals from the previous all-episode dump.
  rm -rf "${PROBE_ROOT}/run${i}/residual"
done

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fastwam
export PYTHONPATH="${ROOT}/src:${ROOT}/scripts:${ROOT}/scripts/dexjoco_async:${PYTHONPATH:-}"

dump_pids=()
for i in 1 2 3 4; do
  gpu="${GPU_ARR[$((i - 1))]}"
  out="${PROBE_ROOT}/run${i}"
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    python "${ROOT}/scripts/analysis/dump_cfg_residual_on_eval_traj.py" \
      --phase dump \
      --eval-run "${EVAL_ROOT}/run${i}" \
      --out-dir "${out}" \
      --task water_plant \
      --task-config-dir "${CFG_TASK_DIR}" \
      --device cuda:0 \
      --checkpoint "${CKPT}" \
      --backbone-checkpoint "${BACKBONE}" \
      --dataset-stats "${STATS}" \
      --text-embedding-cache-dir "${TEXT_CACHE}" \
      --text-cfg-scale "${CFG_SCALE}" \
      --adaptive-cfg-tau "${PROBE_TAU}" \
      --failures-only \
      --max-query-step "${MAX_QUERY_STEP}" \
      --eval-repeat "$((i - 1))" \
      --inference-seed "$((20260812 + i - 1))" \
      > "${PROBE_ROOT}/logs/dump_fail_early_run${i}.log" 2>&1
  ) &
  dump_pids+=("$!")
done
fail=0
for pid in "${dump_pids[@]}"; do
  if ! wait "${pid}"; then fail=1; fi
done
if [[ "${fail}" -ne 0 ]]; then
  echo "[probe-dump] ERROR; see ${PROBE_ROOT}/logs/dump_fail_early_run*.log"
  exit 2
fi

python "${ROOT}/scripts/analysis/partition_cfg_strength_once_intervene.py" \
  --probe-root "${PROBE_ROOT}" \
  --out-dir "${PROBE_ROOT}/partitions" \
  --n-bands 4 \
  --pool fail \
  | tee "${PROBE_ROOT}/logs/partition.log"

echo "[probe-dump] OK -> ${PROBE_ROOT}/partitions"
