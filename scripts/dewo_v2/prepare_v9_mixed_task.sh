#!/usr/bin/env bash
# Prepare one mixed-S0 task for DEWO v9 train: scan (optional) → critic index →
# materialize full pair LeRobot → Eve + VAE pre-encode.
#
#   TASK=fold_glasses COLLECT_ROOT=... GPUS=4,5,6,7 \
#     RUN_SCAN=0 QUEUE_FILE=logs/mixed_v9_pipeline.env \
#     bash scripts/dewo_v2/prepare_v9_mixed_task.sh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"
# shellcheck disable=SC1091
source "${ROOT_DIR}/scripts/dewo_v2/lib.sh"

dewo_v2_require_task
dewo_v2_load_task "${TASK}"

CKPT="${ROOT_DIR}/checkpoints/dexjoco/mixed_5task_fastwam/weights/step_055000.pt"
STATS="${ROOT_DIR}/artifacts/mixed_5task/dataset_stats.json"
export CKPT STATS PRETRAINED_NORM_STATS="${STATS}" INIT_WEIGHTS="${CKPT}" SOURCE_CHECKPOINT="${CKPT}"

COLLECT_ROOT="${COLLECT_ROOT:?Set COLLECT_ROOT to data/${TASK}_mixed_s0_collect_*}"
COLLECT_ROOT="$(realpath -e "${COLLECT_ROOT}")"
RAW="${COLLECT_ROOT}/rollout_raw_200"
SCAN_ROOT="${SCAN_ROOT:-${COLLECT_ROOT}/recoverability_pairs_v2}"
RUN_SCAN="${RUN_SCAN:-0}"
GPUS="${GPUS:-4,5,6,7}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
EXP_ROOT="${EXP_ROOT:-${ROOT_DIR}/data/${TASK}_dewo_v9_pair_${STAMP}}"
PAIR_OUT="${PAIR_OUT:-${ROOT_DIR}/data/${TASK}_dewo_v9_pair_full_lerobot}"
dewo_v2_assert_path_for_task COLLECT_ROOT "${COLLECT_ROOT}"
dewo_v2_assert_path_for_task EXP_ROOT "${EXP_ROOT}"
dewo_v2_assert_path_for_task PAIR_OUT "${PAIR_OUT}"
QUEUE_FILE="${QUEUE_FILE:-${ROOT_DIR}/logs/mixed_v9_pipeline.env}"
LOG="${EXP_ROOT}/logs/prepare_v9.log"
mkdir -p "${EXP_ROOT}/logs" "$(dirname "${QUEUE_FILE}")"

log() { echo "[v9-prep ${TASK} $(date -Is)] $*" | tee -a "${LOG}"; }

test -f "${CKPT}" || { log "ERROR missing CKPT=${CKPT}"; exit 2; }
test -f "${STATS}" || { log "ERROR missing STATS=${STATS}"; exit 2; }
test -d "${RAW}/meta" || { log "ERROR missing rollout ${RAW}"; exit 2; }

dewo_v2_activate_fastwam

scan_complete() {
  [[ -f "${SCAN_ROOT}/summary.json" ]] || return 1
  python3 - <<PY
import json, sys
s = json.load(open("${SCAN_ROOT}/summary.json"))
n = int(s.get("num_complete_event_pairs") or 0)
sys.exit(0 if s.get("status") == "complete" and n > 0 else 1)
PY
}

if [[ "${RUN_SCAN}" == "1" ]] && ! scan_complete; then
  log "LAUNCH recoverability scan GPUS=${GPUS} -> ${SCAN_ROOT}"
  OPEN_REPO="${OPEN_REPO:-${ROOT_DIR}/../FastWAM-infer-in-DexJoco}"
  OPEN_REPO="$(cd "${OPEN_REPO}" && pwd)"
  test -f "${TEXT_EMB}" || { log "ERROR missing TEXT_EMB=${TEXT_EMB}"; exit 2; }
  python "${ROOT_DIR}/scripts/fold_glasses/run_recoverability_pair_scan.py" \
    --gpus "${GPUS}" \
    --dataset "${RAW}" \
    --output "${SCAN_ROOT}" \
    --checkpoint "${CKPT}" \
    --model-config "${OPEN_REPO}/configs/fastwam_dexjoco.yaml" \
    --dataset-stats "${STATS}" \
    --text-embedding "${TEXT_EMB}" \
    --task-name "${TASK}" \
    --max-steps 1000 \
    --overwrite \
    2>&1 | tee -a "${LOG}"
fi

if ! scan_complete; then
  log "ERROR scan incomplete at ${SCAN_ROOT}"
  exit 2
fi

CRITIC_INDEX="${COLLECT_ROOT}/v9_critic_index.json"
if [[ -f "${CRITIC_INDEX}" ]]; then
  log "reuse critic index ${CRITIC_INDEX}"
elif [[ -f "${COLLECT_ROOT}/v8_critic_index.json" ]]; then
  CRITIC_INDEX="${COLLECT_ROOT}/v8_critic_index.json"
  log "reuse existing ${CRITIC_INDEX} (legacy filename)"
else
  log "build critic index -> ${CRITIC_INDEX}"
  python "${ROOT_DIR}/scripts/dewo_v2/build_v9_critic_index.py" \
    --collect-root "${COLLECT_ROOT}" \
    --scan-root "${SCAN_ROOT}" \
    --output "${CRITIC_INDEX}" \
    2>&1 | tee -a "${LOG}"
fi

log "materialize full-horizon pair LeRobot -> ${PAIR_OUT}"
python "${ROOT_DIR}/scripts/dewo_v2/materialize_v9_full_pair_lerobot.py" \
  --critic-index "${CRITIC_INDEX}" \
  --source-dataset "${RAW}" \
  --output-dataset "${PAIR_OUT}" \
  --success-prompt "${SUCCESS_PROMPT}" \
  --overwrite \
  2>&1 | tee -a "${LOG}"

export PAIR_DATASET="${PAIR_OUT}"
export PRIMARY_KIND=all_success_seeds
export PAIR_HORIZON=full
export ROLLOUT_RAW="${RAW}"
export BASE_DATASET="${RAW}"
export DEWO_TASK=dexjoco/dexjoco_dewo_v9_offline_b1_jump_fast_uncond
export EXP_ROOT GPUS CUDA_VISIBLE_DEVICES="${GPUS}"

log "prepare Eve + VAE GPUS=${GPUS} EXP_ROOT=${EXP_ROOT}"
bash "${ROOT_DIR}/scripts/dewo_v2/prepare_pair_eve.sh" 2>&1 | tee -a "${LOG}"

ENV_FILE="${EXP_ROOT}/eve_v02/protocol/offline_v1_b1_jump_fast.env"
test -f "${ENV_FILE}" || { log "ERROR missing ${ENV_FILE}"; exit 2; }

{
  echo "TASK=${TASK}"
  echo "ENV_FILE=${ENV_FILE}"
  echo "EXP_ROOT=${EXP_ROOT}"
  echo "PAIR_OUT=${PAIR_OUT}"
  echo "COLLECT_ROOT=${COLLECT_ROOT}"
  echo "PREPARED_AT=$(date -Is)"
} >> "${QUEUE_FILE}"

log "READY env=${ENV_FILE} appended to ${QUEUE_FILE}"
log "DONE ${TASK}"
