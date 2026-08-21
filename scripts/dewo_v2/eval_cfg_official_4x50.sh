#!/usr/bin/env bash
# Official 4×50 DEWO v2 CFG eval (success vs base). Opensource 224 / z-score.
#   TASK=water_plant RUN_DIR=... CKPT=... GPUS=4,5,6,7 CFG_SCALE=2.0 \
#     bash scripts/dewo_v2/eval_cfg_official_4x50.sh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"
# shellcheck disable=SC1091
source "${ROOT_DIR}/scripts/dewo_v2/lib.sh"

dewo_v2_require_task
dewo_v2_require_gpus
# Preserve caller overrides: load_task exports release CKPT/STATS and would clobber them.
CALLER_CKPT="${CKPT:-}"
CALLER_STATS="${PRETRAINED_NORM_STATS:-}"
CALLER_CFG_TASK_DIR="${CFG_TASK_DIR:-}"
dewo_v2_load_task "${TASK}"

RUN_DIR="${RUN_DIR:?Set RUN_DIR to the DEWO v2 training run directory}"
CKPT="${CALLER_CKPT:-${CKPT:-${RUN_DIR}/checkpoints/weights/step_002500.pt}}"
PRETRAINED_NORM_STATS="${CALLER_STATS:-${PRETRAINED_NORM_STATS:-${STATS}}}"
dewo_v2_align_opensource_stack
TEXT_EMBEDDING_CACHE_DIR="${TEXT_EMBEDDING_CACHE_DIR:?Set TEXT_EMBEDDING_CACHE_DIR}"
CFG_TASK_DIR="${CALLER_CFG_TASK_DIR:-${CFG_TASK_DIR:-${ROOT_DIR}/configs/eval/dexjoco/${TASK}_dewo_v2_cfg}}"
DEXJOCO_PY_ROOT="${DEXJOCO_PY_ROOT:-${ROOT_DIR}/third_party/dexjoco/dexjoco}"

CFG_SCALE="${CFG_SCALE:-2.0}"
EPISODES="${EPISODES:-50}"
ENV_SEED="${ENV_SEED:-0}"
INFERENCE_SEED="${INFERENCE_SEED:-20260812}"
REPLAN_STEPS="${REPLAN_STEPS:-24}"
MAX_ENV_STEPS="${MAX_ENV_STEPS:-${MAX_STEPS}}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-10}"
ACTION_HORIZON="${ACTION_HORIZON:-32}"
BASE_PORT="${BASE_PORT:-6100}"
WAIT_IDLE="${WAIT_IDLE:-1}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
CKPT_TAG="${CKPT_TAG:-$(basename "${CKPT}" .pt)}"
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/evaluate_results/dexjoco/${TASK}_dewo_v2_pair_${CKPT_TAG}_cfg${CFG_SCALE}_4x50_${STAMP}}"

dewo_v2_activate_fastwam
export PYTHONPATH="${ROOT_DIR}/src:${ROOT_DIR}/scripts:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

mkdir -p "${OUT_ROOT}/logs"
LOG="${OUT_ROOT}/logs/orchestrator.log"
log() { echo "[dewo-v2-cfg-4x50 ${TASK} $(date -Is)] $*" | tee -a "${LOG}"; }

for required in \
  "${RUN_DIR}/config.yaml" \
  "${CKPT}" \
  "${PRETRAINED_NORM_STATS}" \
  "${TEXT_EMBEDDING_CACHE_DIR}" \
  "${CFG_TASK_DIR}/${TASK}.yaml"
do
  [[ -e "${required}" ]] || { log "ERROR missing ${required}"; exit 2; }
done

IFS=',' read -r -a GPU_ARR <<< "${GPUS}"
if [[ "${#GPU_ARR[@]}" -ne 4 ]]; then
  log "ERROR: official 4×50 concurrent layout expects 4 GPUs, got ${GPUS}"
  exit 2
fi

if [[ "${WAIT_IDLE}" == "1" ]]; then
  dewo_v2_wait_gpus_idle "${LOG}" "dewo-v2-cfg-4x50"
fi

log "CFG success-vs-base scale=${CFG_SCALE} (not suffix-only scale=1)"
log "ckpt=${CKPT} gpus=${GPUS} protocol=seeds 0..49 × 4, replan=${REPLAN_STEPS}, max_steps=${MAX_ENV_STEPS}"
log "out=${OUT_ROOT}"

pids=()
for i in 1 2 3 4; do
  gpu="${GPU_ARR[$((i - 1))]}"
  out_dir="${OUT_ROOT}/run${i}"
  mkdir -p "${out_dir}"
  log "launch run${i} gpu=${gpu} env_seed=${ENV_SEED} infer_seed=$((INFERENCE_SEED + i - 1))"
  "${ENV_PREFIX}/bin/python" scripts/dexjoco_async/run_multi_gpu_dexjoco_eval.py \
    --gpus "${gpu}" \
    --base-port "$((BASE_PORT + i * 20))" \
    --episodes "${EPISODES}" \
    --seed "${ENV_SEED}" \
    --eval-repeat "$((i - 1))" \
    --inference-seed "$((INFERENCE_SEED + i - 1))" \
    --text-cfg-scale "${CFG_SCALE}" \
    --action-horizon "${ACTION_HORIZON}" \
    --server-conda-env "${ENV_PREFIX}" \
    --client-conda-env dexjoco \
    --run-dir "${RUN_DIR}" \
    --checkpoint "${CKPT}" \
    --dataset-stats-path "${PRETRAINED_NORM_STATS}" \
    --text-embedding-cache-dir "${TEXT_EMBEDDING_CACHE_DIR}" \
    --no-load-text-encoder \
    --task-config-dir "${CFG_TASK_DIR}" \
    --tasks "${TASK}" \
    --dexjoco-py-root "${DEXJOCO_PY_ROOT}" \
    --replan-steps "${REPLAN_STEPS}" \
    --control-mode blocking \
    --max-env-steps "${MAX_ENV_STEPS}" \
    --num-inference-steps "${NUM_INFERENCE_STEPS}" \
    --video-fps 30 \
    --no-randomize \
    --no-randomize-dynamics \
    --save-actions \
    --save-video \
    --no-action-clip \
    --output-dir "${out_dir}" \
    > "${OUT_ROOT}/logs/run${i}.log" 2>&1 &
  pids+=("$!")
done

fail=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    fail=1
  fi
done
if [[ "${fail}" -ne 0 ]]; then
  log "ERROR: one or more CFG eval runs failed"
  exit 2
fi

"${ENV_PREFIX}/bin/python" - "${OUT_ROOT}" "${CKPT_TAG}" "${CFG_SCALE}" "${MAX_ENV_STEPS}" "${TASK}" <<'PY' | tee -a "${LOG}"
import json, statistics, sys
from pathlib import Path
root = Path(sys.argv[1])
ckpt_tag = sys.argv[2]
cfg_scale = float(sys.argv[3])
max_steps = int(sys.argv[4])
task = sys.argv[5]
rates, rows, pooled_s, pooled_n = [], [], 0, 0
for i in range(1, 5):
    d = json.loads((root / f"run{i}" / "summary.json").read_text())
    rate = float(d["overall_success_rate"])
    s, n = int(d["total_successes"]), int(d["total_episodes"])
    rates.append(rate)
    pooled_s += s
    pooled_n += n
    rows.append({"run": i, "successes": s, "episodes": n, "rate": rate})
agg = {
    "task": task,
    "method": "dewo_v2_success_vs_base_cfg",
    "ckpt_tag": ckpt_tag,
    "text_cfg_scale": cfg_scale,
    "protocol": "official_4x50_seeds_0_49",
    "max_env_steps": max_steps,
    "runs": rows,
    "mean_success_rate": statistics.fmean(rates),
    "var_success_rate": statistics.pvariance(rates),
    "std_success_rate": statistics.pstdev(rates),
    "pooled_successes": pooled_s,
    "pooled_episodes": pooled_n,
    "pooled_success_rate": pooled_s / pooled_n if pooled_n else None,
}
(root / "aggregate.json").write_text(json.dumps(agg, indent=2) + "\n")
print(json.dumps(agg, indent=2))
print(
    f"mean={agg['mean_success_rate']:.4f} var={agg['var_success_rate']:.6f} "
    f"pooled={pooled_s}/{pooled_n}"
)
PY

log "DONE ${OUT_ROOT}/aggregate.json"
