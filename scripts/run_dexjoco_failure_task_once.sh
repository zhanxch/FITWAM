#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=${ROOT:-/data_all/share/FastWAM_zhaoyc_failure}
RESULT_ROOT=${RESULT_ROOT:-/data_all/share/dexjoco_fastwam_results}
PY=${PY:-/home/gzr1/miniconda3/envs/residual/bin/python}
ACCEL=${ACCEL:-/home/zhaoyc/.local/bin/accelerate}
ACCEL_CFG=${ACCEL_CFG:-scripts/accelerate_configs/accelerate_zero1_ds.yaml}
TASK=${TASK:?Set TASK, e.g. TASK=hammer_nail}
VARIANT=${VARIANT:-failure_embedding}
GPUS=${GPUS:-0,1,2,3}
MAX_STEPS=${MAX_STEPS:-6000}
SAVE_EVERY=${SAVE_EVERY:-500}
EVAL_EVERY=${EVAL_EVERY:-500}
EVAL_EPISODES=${EVAL_EPISODES:-50}
EVAL_SEED=${EVAL_SEED:-0}
REPLAN_STEPS=${REPLAN_STEPS:-24}
MAX_ENV_STEPS=${MAX_ENV_STEPS:-1500}
EVAL_SERVER_CONDA_ENV=${EVAL_SERVER_CONDA_ENV:-residual}
EVAL_CLIENT_CONDA_ENV=${EVAL_CLIENT_CONDA_ENV:-residual}
RUN_ID=${RUN_ID:-$(date +%Y-%m-%d_%H-%M-%S)}
WAIT_FOR_GPUS=${WAIT_FOR_GPUS:-0}
REQUIRE_FREE_MB=${REQUIRE_FREE_MB:-70000}
START_PRUNER=${START_PRUNER:-1}

case "$VARIANT" in
  success|failure_embedding|text_failure) ;;
  *) echo "Unknown VARIANT=$VARIANT; expected success, failure_embedding, or text_failure" >&2; exit 2 ;;
esac

IFS=',' read -r -a GPU_LIST <<< "$GPUS"
NPROC=${#GPU_LIST[@]}
FIRST_GPU=${GPU_LIST[0]}
TASK_CFG="dexjoco/dexjoco_${TASK}_${VARIANT}_2cam_proprio_1e-4"
WANDB_NAME="dexjoco_${TASK}_${VARIANT}_2cam_proprio_1e-4"
RUN_DIR="${RESULT_ROOT}/${WANDB_NAME}/${RUN_ID}"
LOGDIR="${ROOT}/artifacts/logs"
TRAIN_LOG="${LOGDIR}/train_${TASK}_${VARIANT}_${RUN_ID}.log"
EVAL_DIR="${ROOT}/artifacts/evals/${TASK}_${VARIANT}_step_${MAX_STEPS}_seed${EVAL_SEED}_${EVAL_EPISODES}ep_${RUN_ID}"
SUCCESS_DATASET="/data_all/share/datasets/dexjoco/dexjoco_lerobot_datasets/${TASK}"
FAILURE_DATASET="${ROOT}/artifacts/datasets/${TASK}_failure_fastwam_2cam_text"
STATS_PATH="${ROOT}/artifacts/dataset_stats/dexjoco_${TASK}_success_action_state.json"

if [[ "$VARIANT" == "success" || "$VARIANT" == "failure_embedding" ]]; then
  CACHE_DIR="${ROOT}/artifacts/text_embeds_cache/dexjoco_${TASK}_2cam_success"
else
  CACHE_DIR="${ROOT}/artifacts/text_embeds_cache/dexjoco_${TASK}_2cam_text_failure"
fi
EXPORT_TEXT_CACHE_SCRIPT=${EXPORT_TEXT_CACHE_SCRIPT:-scripts/export_text_embed_cache_npz.py}
if [[ ! -f "${ROOT}/${EXPORT_TEXT_CACHE_SCRIPT}" ]]; then
  EXPORT_TEXT_CACHE_SCRIPT=scripts/water_plant/export_text_embed_cache_npz.py
fi

cd "$ROOT"
mkdir -p "$LOGDIR" "$EVAL_DIR"
PY_BIN_DIR=$(dirname "$PY")
export PATH="$PY_BIN_DIR":/home/zhaoyc/.local/bin:$PATH
export PYTHONPATH="$ROOT/src:$ROOT/scripts:${PYTHONPATH:-}"
export DIFFSYNTH_MODEL_BASE_PATH="$ROOT/checkpoints"
export PYTHONUNBUFFERED=1
export WANDB_MODE=online

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "$TRAIN_LOG"
}

gpu_ready() {
  local gpu used
  for gpu in "${GPU_LIST[@]}"; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu" | tr -d ' ')
    if (( 81920 - used < REQUIRE_FREE_MB )); then
      return 1
    fi
  done
  return 0
}

if [[ "$WAIT_FOR_GPUS" == "1" ]]; then
  log "waiting for GPUs=$GPUS free_mb_each>=$REQUIRE_FREE_MB"
  until gpu_ready; do
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader | tee -a "$TRAIN_LOG" || true
    sleep 180
  done
fi

if [[ "$START_PRUNER" == "1" && -f "${ROOT}/artifacts/scripts/run_ckpt_pruner.sh" ]]; then
  if ! tmux has-session -t fastwam_ckpt_pruner 2>/dev/null; then
    log "starting checkpoint pruner: keep all weights, latest state"
    tmux new-session -d -s fastwam_ckpt_pruner "cd $ROOT; KEEP_WEIGHTS=-1 KEEP_STATES=1 PRUNE_INTERVAL=300 bash artifacts/scripts/run_ckpt_pruner.sh"
  fi
fi

required_datasets=("$SUCCESS_DATASET")
if [[ "$VARIANT" != "success" ]]; then
  required_datasets+=("$FAILURE_DATASET")
fi
for path in "${required_datasets[@]}"; do
  [[ -d "$path" ]] || { log "missing dataset: $path"; exit 10; }
done

if [[ ! -f "$STATS_PATH" ]]; then
  log "missing stats; computing $STATS_PATH"
  "$PY" scripts/compute_dexjoco_success_stats.py --tasks "$TASK"
fi

log "generating configs for $TASK"
"$PY" scripts/create_dexjoco_failure_configs.py --tasks "$TASK" --max-steps "$MAX_STEPS" --save-every "$SAVE_EVERY" --eval-every "$EVAL_EVERY"

log "precomputing text embeddings for task=$TASK_CFG cache=$CACHE_DIR"
CUDA_VISIBLE_DEVICES="$FIRST_GPU" "$PY" scripts/precompute_text_embeds.py "task=${TASK_CFG}" >> "$TRAIN_LOG" 2>&1
CUDA_VISIBLE_DEVICES="$FIRST_GPU" "$PY" "$EXPORT_TEXT_CACHE_SCRIPT" --cache-dir "$CACHE_DIR" >> "$TRAIN_LOG" 2>&1

train_args=(
  "task=${TASK_CFG}"
  "output_dir=${RUN_DIR}"
  "wandb.name=${WANDB_NAME}"
  "max_steps=${MAX_STEPS}"
  "save_every=${SAVE_EVERY}"
  "eval_every=${EVAL_EVERY}"
)
if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
  train_args+=("resume=${RESUME_CHECKPOINT}")
fi

log "training task=$TASK variant=$VARIANT run_dir=$RUN_DIR gpus=$GPUS"
CUDA_VISIBLE_DEVICES="$GPUS" "$ACCEL" launch \
  --config_file "$ACCEL_CFG" \
  --num_processes "$NPROC" \
  scripts/train.py \
  "${train_args[@]}" >> "$TRAIN_LOG" 2>&1

CKPT="${RUN_DIR}/checkpoints/weights/step_$(printf '%06d' "$MAX_STEPS").pt"
if [[ ! -f "$CKPT" ]]; then
  log "missing final checkpoint: $CKPT"
  exit 11
fi

log "rolling out checkpoint=$CKPT episodes=$EVAL_EPISODES seed=$EVAL_SEED"
"$PY" scripts/water_plant/dexjoco_async/run_multi_gpu_dexjoco_eval.py \
  --gpus "$GPUS" \
  --run-dir "$RUN_DIR" \
  --checkpoint "$CKPT" \
  --no-load-text-encoder \
  --server-conda-env "$EVAL_SERVER_CONDA_ENV" \
  --client-conda-env "$EVAL_CLIENT_CONDA_ENV" \
  --task-config-dir third_party/dexjoco/configs/rand_obj \
  --tasks "$TASK" \
  --episodes "$EVAL_EPISODES" \
  --seed "$EVAL_SEED" \
  --replan-steps "$REPLAN_STEPS" \
  --control-mode blocking \
  --max-env-steps "$MAX_ENV_STEPS" \
  --output-dir "$EVAL_DIR" >> "$TRAIN_LOG" 2>&1

log "done task=$TASK variant=$VARIANT run_dir=$RUN_DIR eval_dir=$EVAL_DIR"
