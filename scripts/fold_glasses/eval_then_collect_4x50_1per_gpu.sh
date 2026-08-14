#!/usr/bin/env bash
# fold_glasses S0: official 4×50 eval then 4×50 rollout collect.
# Launch mode: 1 policy server per GPU (no packed / no idle wait).
#
# Usage:
#   GPUS=0,1,2,3,4,5,6,7 bash scripts/fold_glasses/eval_then_collect_4x50_1per_gpu.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

source /home/xiangchengzhan/anaconda3/etc/profile.d/conda.sh
conda activate fastwam
ENV_PREFIX="${FITWAM_ENV_PREFIX:-/home/xiangchengzhan/anaconda3/envs/fastwam}"
export PATH="${ENV_PREFIX}/bin:${PATH}"
export PYTHONPATH="${ROOT}/src:${ROOT}/scripts:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export TOKENIZERS_PARALLELISM=false

CKPT="${CKPT:-${ROOT}/checkpoints/dexjoco/fold_glasses_fastwam/weights/step_010000.pt}"
RUN_DIR="${RUN_DIR:-${ROOT}/checkpoints/dexjoco/fold_glasses_fastwam/s0_bundle}"
NORM_META="${NORM_STATS_META_DIR:-${ROOT}/data/fold_glasses_fastwam/meta}"
TEXT_CACHE="${TEXT_CACHE:-${ROOT}/data/text_embeds_cache/fold_glasses}"
TASK_CONFIG_DIR="${TASK_CONFIG_DIR:-${ROOT}/third_party/dexjoco/configs/rand_obj}"
DEXJOCO_PY_ROOT="${DEXJOCO_PY_ROOT:-${ROOT}/third_party/dexjoco/dexjoco}"
SOURCE_DATASET="${SOURCE_DATASET:-${ROOT}/data/dexjoco/dexjoco_lerobot_datasets/fold_glasses}"
SUCCESS_PROMPT="${SUCCESS_PROMPT:-Fold the glasses and place them into the case.}"

GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
REPLAN_STEPS="${REPLAN_STEPS:-25}"
MAX_ENV_STEPS="${MAX_ENV_STEPS:-1200}"
VIDEO_FPS="${VIDEO_FPS:-30}"
EPISODES="${EPISODES:-50}"
NUM_REPEATS="${NUM_REPEATS:-4}"
EVAL_SEED="${EVAL_SEED:-0}"
COLLECT_SEED="${COLLECT_SEED:-10086}"
BASE_PORT="${BASE_PORT:-5700}"

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
CKPT_TAG="$(basename "${CKPT}" .pt)"
OUT_ROOT="${OUT_ROOT:-${ROOT}/evaluate_results/dexjoco/fold_glasses_s0_1pergpu_4x50_${CKPT_TAG}_${STAMP}}"
EVAL_ROOT="${EVAL_ROOT:-${OUT_ROOT}/eval_official_4x50}"
COLLECT_ROOT="${COLLECT_ROOT:-${OUT_ROOT}/collection}"
ROLLOUT_RAW="${ROLLOUT_RAW:-${OUT_ROOT}/rollout_raw_200}"
LOG_DIR="${OUT_ROOT}/logs"
mkdir -p "${EVAL_ROOT}" "${COLLECT_ROOT}" "${LOG_DIR}"
MASTER_LOG="${LOG_DIR}/orchestrator_${STAMP}.log"
EVAL_LOG="${LOG_DIR}/eval_${STAMP}.log"
COLLECT_LOG="${LOG_DIR}/collect_${STAMP}.log"

log() { echo "[fold-glasses-1pergpu $(date -Is)] $*" | tee -a "${MASTER_LOG}"; }

for p in "${CKPT}" "${RUN_DIR}/config.yaml" "${NORM_META}/stats.json" \
  "${TASK_CONFIG_DIR}/fold_glasses.yaml" "${DEXJOCO_PY_ROOT}"; do
  [[ -e "$p" ]] || { log "ERROR missing $p"; exit 1; }
done

log "start gpus=${GPUS} mode=1_server_per_gpu (no packed, no idle wait)"
log "ckpt=${CKPT}"
log "out=${OUT_ROOT}"
log "eval: seeds ${EVAL_SEED}..$((EVAL_SEED + EPISODES - 1)) × ${NUM_REPEATS} | replan=${REPLAN_STEPS} max_env_steps=${MAX_ENV_STEPS}"
log "collect: seeds ${COLLECT_SEED}..$((COLLECT_SEED + EPISODES - 1)) × ${NUM_REPEATS}"

# ----- Phase 1: official 4×50 eval (sequential repeats) -----
for run_i in $(seq 1 "${NUM_REPEATS}"); do
  out="${EVAL_ROOT}/run${run_i}"
  if [[ -f "${out}/summary.json" ]]; then
    log "=== eval run${run_i}/${NUM_REPEATS} SKIP (summary exists) ==="
    continue
  fi
  mkdir -p "${out}"
  log "=== eval run${run_i}/${NUM_REPEATS} start ==="
  set -o pipefail
  "${ENV_PREFIX}/bin/python" scripts/dexjoco_async/run_multi_gpu_dexjoco_eval.py \
    --gpus "${GPUS}" \
    --base-port "$((BASE_PORT + run_i * 20))" \
    --episodes "${EPISODES}" \
    --seed "${EVAL_SEED}" \
    --server-conda-env "${ENV_PREFIX}" \
    --client-conda-env dexjoco \
    --run-dir "${RUN_DIR}" \
    --checkpoint "${CKPT}" \
    --norm-stats-meta-dir "${NORM_META}" \
    --text-embedding-cache-dir "${TEXT_CACHE}" \
    --no-load-text-encoder \
    --task-config-dir "${TASK_CONFIG_DIR}" \
    --tasks fold_glasses \
    --dexjoco-py-root "${DEXJOCO_PY_ROOT}" \
    --replan-steps "${REPLAN_STEPS}" \
    --control-mode blocking \
    --max-env-steps "${MAX_ENV_STEPS}" \
    --video-fps "${VIDEO_FPS}" \
    --no-randomize \
    --no-randomize-dynamics \
    --save-video \
    --save-actions \
    --no-action-clip \
    --output-dir "${out}" \
    2>&1 | tee -a "${EVAL_LOG}"
  ec=${PIPESTATUS[0]}
  log "=== eval run${run_i}/${NUM_REPEATS} EXIT=${ec} ==="
  [[ "${ec}" -eq 0 ]] || exit "${ec}"
done

"${ENV_PREFIX}/bin/python" - "${EVAL_ROOT}" "${CKPT_TAG}" "${MAX_ENV_STEPS}" <<'PY' | tee -a "${EVAL_LOG}" "${MASTER_LOG}"
import json, statistics, sys
from pathlib import Path
root = Path(sys.argv[1])
ckpt_tag = sys.argv[2]
max_env_steps = int(sys.argv[3])
rates, rows = [], []
pooled_s = pooled_n = 0
for i in range(1, 5):
    d = json.loads((root / f"run{i}" / "summary.json").read_text())
    rate = float(d["overall_success_rate"])
    s, n = int(d["total_successes"]), int(d["total_episodes"])
    rates.append(rate); pooled_s += s; pooled_n += n
    rows.append({"run": i, "successes": s, "episodes": n, "rate": rate, "seed": d.get("seed")})
mean = statistics.fmean(rates)
var = statistics.pvariance(rates) if len(rates) > 1 else 0.0
std = statistics.pstdev(rates) if len(rates) > 1 else 0.0
protocol = "official_4x50_seeds_0_49" if max_env_steps == 1500 else f"4x50_seeds_0_49_max_env_steps_{max_env_steps}"
agg = {
    "method": "fold_glasses_s0_1pergpu",
    "ckpt_tag": ckpt_tag,
    "protocol": protocol,
    "max_env_steps": max_env_steps,
    "packing": "1_server_per_gpu_sequential_runs",
    "runs": rows,
    "mean_success_rate": mean,
    "var_success_rate": var,
    "std_success_rate": std,
    "pooled_successes": pooled_s,
    "pooled_episodes": pooled_n,
    "pooled_success_rate": pooled_s / pooled_n if pooled_n else None,
}
(root / "aggregate.json").write_text(json.dumps(agg, indent=2) + "\n")
print(f"[eval] aggregate mean={mean:.4f} ± std={std:.4f} var={var:.6f} pooled={pooled_s}/{pooled_n}")
print(json.dumps(agg, indent=2))
PY

log "eval phase DONE -> ${EVAL_ROOT}/aggregate.json"

# ----- Phase 2: 4×50 rollout collect (seeds disjoint from eval 0..49) -----
for run_i in $(seq 1 "${NUM_REPEATS}"); do
  out_dir="${COLLECT_ROOT}/run${run_i}"
  raw_ds="${out_dir}/raw"
  if [[ -f "${out_dir}/summary.json" ]] && [[ -d "${raw_ds}/meta" ]]; then
    log "=== collect run${run_i}/${NUM_REPEATS} SKIP (already complete) ==="
    continue
  fi
  mkdir -p "${out_dir}"
  log "=== collect run${run_i}/${NUM_REPEATS} seed=${COLLECT_SEED}..$((COLLECT_SEED + EPISODES - 1)) ==="
  set -o pipefail
  "${ENV_PREFIX}/bin/python" scripts/dexjoco_async/run_multi_gpu_dexjoco_collect.py \
    --gpus "${GPUS}" \
    --base-port "$((BASE_PORT + 100 + run_i * 20))" \
    --episodes "${EPISODES}" \
    --seed "${COLLECT_SEED}" \
    --server-conda-env "${ENV_PREFIX}" \
    --client-conda-env dexjoco \
    --run-dir "${RUN_DIR}" \
    --checkpoint "${CKPT}" \
    --no-load-text-encoder \
    --norm-stats-meta-dir "${NORM_META}" \
    --text-embedding-cache-dir "${TEXT_CACHE}" \
    --task-config-dir "${TASK_CONFIG_DIR}" \
    --tasks fold_glasses \
    --source-dataset "${SOURCE_DATASET}" \
    --success-prompt "${SUCCESS_PROMPT}" \
    --replan-steps "${REPLAN_STEPS}" \
    --max-env-steps "${MAX_ENV_STEPS}" \
    --video-fps "${VIDEO_FPS}" \
    --no-randomize \
    --no-randomize-dynamics \
    --no-action-clip \
    --outcome-task-mode clean \
    --output-dir "${out_dir}" \
    --raw-output-dataset "${raw_ds}" \
    --overwrite \
    2>&1 | tee -a "${COLLECT_LOG}"
  ec=${PIPESTATUS[0]}
  log "=== collect run${run_i}/${NUM_REPEATS} EXIT=${ec} ==="
  [[ "${ec}" -eq 0 ]] || exit "${ec}"
done

log "merging ${NUM_REPEATS} raw datasets -> ${ROLLOUT_RAW}"
shard_args=()
for i in $(seq 1 "${NUM_REPEATS}"); do
  shard_args+=("${COLLECT_ROOT}/run${i}/raw")
done
"${ENV_PREFIX}/bin/python" scripts/build_rollout_datasets.py merge-shards \
  --shard-datasets "${shard_args[@]}" \
  --output-dataset "${ROLLOUT_RAW}" \
  --overwrite
"${ENV_PREFIX}/bin/python" scripts/build_rollout_datasets.py validate-outcomes \
  --dataset "${ROLLOUT_RAW}" \
  --expected-episodes "$((EPISODES * NUM_REPEATS))" \
  --report "${OUT_ROOT}/rollout_outcome_validation.json"

log "ALL DONE"
log "eval_aggregate=${EVAL_ROOT}/aggregate.json"
log "rollout_raw=${ROLLOUT_RAW}"
log "validation=${OUT_ROOT}/rollout_outcome_validation.json"
