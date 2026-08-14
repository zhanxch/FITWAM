#!/usr/bin/env bash
# Evaluate one frozen checkpoint under base, success, and success-vs-base CFG.
# Every branch uses identical environment and diffusion seeds and task settings.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

: "${RUN_DIR:?Set RUN_DIR to the DEWO v2 training run directory}"
: "${CKPT:?Set CKPT to the single selected checkpoint}"
: "${PRETRAINED_NORM_STATS:?Set PRETRAINED_NORM_STATS to the OPEN fold_glasses dataset_stats.json}"
: "${TEXT_EMBEDDING_CACHE_DIR:?Set TEXT_EMBEDDING_CACHE_DIR to the DEWO v2 text cache}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${FITWAM_ENV:-fastwam}"
ENV_PREFIX="${FITWAM_ENV_PREFIX:-$(conda info --base)/envs/fastwam}"
export PATH="${ENV_PREFIX}/bin:${PATH}"
export PYTHONPATH="${ROOT_DIR}/src:${ROOT_DIR}/scripts:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

GPUS="${GPUS:-0,1,2,3}"
EPISODES="${EPISODES:-50}"
REPEATS="${REPEATS:-4}"
ENV_SEED="${ENV_SEED:-0}"
INFERENCE_SEED="${INFERENCE_SEED:-20260812}"
CFG_SCALE="${CFG_SCALE:-2.0}"
REPLAN_STEPS="${REPLAN_STEPS:-24}"
MAX_ENV_STEPS="${MAX_ENV_STEPS:-1200}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-10}"
BASE_PORT="${BASE_PORT:-5800}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/evaluate_results/dexjoco/fold_glasses_dewo_v2_cfg_ablation_${STAMP}}"
BASE_TASK_DIR="${ROOT_DIR}/configs/eval/dexjoco/fold_glasses_dewo_v2_base"
CFG_TASK_DIR="${ROOT_DIR}/configs/eval/dexjoco/fold_glasses_dewo_v2_cfg"
DEXJOCO_PY_ROOT="${DEXJOCO_PY_ROOT:-${ROOT_DIR}/third_party/dexjoco/dexjoco}"
mkdir -p "${OUT_ROOT}/logs"

for required in \
  "${RUN_DIR}/config.yaml" \
  "${CKPT}" \
  "${PRETRAINED_NORM_STATS}" \
  "${TEXT_EMBEDDING_CACHE_DIR}" \
  "${BASE_TASK_DIR}/fold_glasses.yaml" \
  "${CFG_TASK_DIR}/fold_glasses.yaml"
do
  [[ -e "${required}" ]] || { echo "Missing required input: ${required}" >&2; exit 2; }
done

run_branch() {
  local branch="$1"
  local task_dir="$2"
  local scale="$3"
  local branch_offset="$4"
  local repeat
  for repeat in $(seq 1 "${REPEATS}"); do
    local output_dir="${OUT_ROOT}/${branch}/run${repeat}"
    local repeat_env_seed="$((ENV_SEED + repeat - 1))"
    local repeat_inference_seed="$((INFERENCE_SEED + repeat - 1))"
    mkdir -p "${output_dir}"
    "${ENV_PREFIX}/bin/python" scripts/dexjoco_async/run_multi_gpu_dexjoco_eval.py \
      --gpus "${GPUS}" \
      --base-port "$((BASE_PORT + branch_offset + repeat * 20))" \
      --episodes "${EPISODES}" \
      --seed "${repeat_env_seed}" \
      --inference-seed "${repeat_inference_seed}" \
      --text-cfg-scale "${scale}" \
      --server-conda-env "${ENV_PREFIX}" \
      --client-conda-env dexjoco \
      --run-dir "${RUN_DIR}" \
      --checkpoint "${CKPT}" \
      --dataset-stats-path "${PRETRAINED_NORM_STATS}" \
      --text-embedding-cache-dir "${TEXT_EMBEDDING_CACHE_DIR}" \
      --no-load-text-encoder \
      --task-config-dir "${task_dir}" \
      --tasks fold_glasses \
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
      --output-dir "${output_dir}" \
      2>&1 | tee "${OUT_ROOT}/logs/${branch}_run${repeat}.log"
  done
}

# scale=1 is conditional-only and intentionally avoids the second CFG forward.
run_branch base "${BASE_TASK_DIR}" 1.0 0
run_branch success "${CFG_TASK_DIR}" 1.0 200
run_branch success_vs_base_cfg "${CFG_TASK_DIR}" "${CFG_SCALE}" 400

"${ENV_PREFIX}/bin/python" - "${OUT_ROOT}" "${REPEATS}" <<'PY'
import json
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1])
repeats = int(sys.argv[2])
summary = {}
for branch in ("base", "success", "success_vs_base_cfg"):
    runs = []
    for repeat in range(1, repeats + 1):
        payload = json.loads((root / branch / f"run{repeat}" / "summary.json").read_text())
        runs.append(float(payload["overall_success_rate"]))
    summary[branch] = {
        "rates": runs,
        "mean": statistics.fmean(runs),
        "population_std": statistics.pstdev(runs) if len(runs) > 1 else 0.0,
    }
(root / "cfg_ablation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
PY

echo "DEWO v2 CFG ablation complete: ${OUT_ROOT}/cfg_ablation_summary.json"
