#!/usr/bin/env bash
# Offline CFG-strength probe on existing v7 本体 4×50 rollouts, then partition.
# Uses GPUs 0,1,2,3 (one run each). Tau gating is NOT used.
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
DEXJOCO_PY="${DEXJOCO_PY_ROOT:-${ROOT}/third_party/dexjoco/dexjoco}"
GPUS="${GPUS:-0,1,2,3}"
CFG_SCALE="${CFG_SCALE:-1.2}"
# Probe: compute gate E, never fire guided mix.
PROBE_TAU="${PROBE_TAU:-1e9}"

IFS=',' read -r -a GPU_ARR <<< "${GPUS}"
mkdir -p "${PROBE_ROOT}/logs"

echo "[probe] EVAL_ROOT=${EVAL_ROOT}"
echo "[probe] PROBE_ROOT=${PROBE_ROOT}"
test -f "${CKPT}"
test -f "${BACKBONE}"
test -f "${STATS}"

# --- Phase 1: replay obs (dexjoco env, CPU; 4 runs in parallel) ---
replay_pids=()
for i in 1 2 3 4; do
  out="${PROBE_ROOT}/run${i}"
  mkdir -p "${out}"
  (
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate dexjoco
    export MUJOCO_GL=egl
    export PYTHONPATH="${ROOT}/src:${ROOT}/scripts:${ROOT}/scripts/dexjoco_async:${DEXJOCO_PY}:${PYTHONPATH:-}"
    python "${ROOT}/scripts/analysis/dump_cfg_residual_on_eval_traj.py" \
      --phase replay \
      --eval-run "${EVAL_ROOT}/run${i}" \
      --out-dir "${out}" \
      --task water_plant \
      --task-config-dir "${CFG_TASK_DIR}" \
      --dexjoco-py-root "${DEXJOCO_PY}" \
      --eval-repeat "$((i - 1))" \
      --text-embedding-cache-dir "${TEXT_CACHE}" \
      > "${PROBE_ROOT}/logs/replay_run${i}.log" 2>&1
  ) &
  replay_pids+=("$!")
done
fail=0
for pid in "${replay_pids[@]}"; do
  if ! wait "${pid}"; then fail=1; fi
done
if [[ "${fail}" -ne 0 ]]; then
  echo "[probe] ERROR: replay failed; see ${PROBE_ROOT}/logs/replay_run*.log"
  exit 2
fi
echo "[probe] replay done"

# --- Phase 2: dump gate energy on GPUs (fastwam) ---
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
      --eval-repeat "$((i - 1))" \
      --inference-seed "$((20260812 + i - 1))" \
      > "${PROBE_ROOT}/logs/dump_run${i}.log" 2>&1
  ) &
  dump_pids+=("$!")
done
fail=0
for pid in "${dump_pids[@]}"; do
  if ! wait "${pid}"; then fail=1; fi
done
if [[ "${fail}" -ne 0 ]]; then
  echo "[probe] ERROR: dump failed; see ${PROBE_ROOT}/logs/dump_run*.log"
  exit 2
fi
echo "[probe] dump done"

python "${ROOT}/scripts/analysis/partition_cfg_strength_once_intervene.py" \
  --probe-root "${PROBE_ROOT}" \
  --out-dir "${PROBE_ROOT}/partitions" \
  --n-bands 4 \
  --pool all \
  | tee "${PROBE_ROOT}/logs/partition.log"

echo "[probe] OK -> ${PROBE_ROOT}/partitions"
