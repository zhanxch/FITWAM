#!/usr/bin/env bash
# Official 4×50 fold_glasses DEWO v2 **CFG** eval (success vs base, scale=2).
# This is not success-suffix-only (scale=1). Opensource 224 / z-score.
# 4 GPUs → 4 concurrent repeats, 1 server per GPU (packing 4×VAE-DiT OOMs).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

RUN_DIR="${RUN_DIR:-${ROOT_DIR}/runs/dexjoco_fold_glasses_dewo_v2/2026-08-13_20-55-10_B1-jump-fast-lora}"
CKPT="${CKPT:-${RUN_DIR}/checkpoints/weights/step_002500.pt}"
PRETRAINED_NORM_STATS="${PRETRAINED_NORM_STATS:-/data_all/xiangchengzhan/FastWAM-infer-in-DexJoco/artifacts/fold_glasses/dataset_stats.json}"
TEXT_EMBEDDING_CACHE_DIR="${TEXT_EMBEDDING_CACHE_DIR:-${ROOT_DIR}/data/fold_glasses_dewo_v2_pair_20260813/text_embeds_cache}"
CFG_TASK_DIR="${CFG_TASK_DIR:-${ROOT_DIR}/configs/eval/dexjoco/fold_glasses_dewo_v2_cfg}"
DEXJOCO_PY_ROOT="${DEXJOCO_PY_ROOT:-${ROOT_DIR}/third_party/dexjoco/dexjoco}"

GPUS="${GPUS:-4,5,6,7}"
CFG_SCALE="${CFG_SCALE:-2.0}"
EPISODES="${EPISODES:-50}"
ENV_SEED="${ENV_SEED:-0}"
INFERENCE_SEED="${INFERENCE_SEED:-20260812}"
REPLAN_STEPS="${REPLAN_STEPS:-24}"
MAX_ENV_STEPS="${MAX_ENV_STEPS:-1200}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-10}"
ACTION_HORIZON="${ACTION_HORIZON:-32}"
BASE_PORT="${BASE_PORT:-6100}"
WAIT_IDLE="${WAIT_IDLE:-1}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
CKPT_TAG="${CKPT_TAG:-$(basename "${CKPT}" .pt)}"
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/evaluate_results/dexjoco/fold_glasses_dewo_v2_pair_${CKPT_TAG}_cfg${CFG_SCALE}_4x50_${STAMP}}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${FITWAM_ENV:-fastwam}"
ENV_PREFIX="${FITWAM_ENV_PREFIX:-$(conda info --base)/envs/fastwam}"
export PATH="${ENV_PREFIX}/bin:${PATH}"
export PYTHONPATH="${ROOT_DIR}/src:${ROOT_DIR}/scripts:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

mkdir -p "${OUT_ROOT}/logs"
LOG="${OUT_ROOT}/logs/orchestrator.log"
log() { echo "[dewo-v2-cfg-4x50 $(date -Is)] $*" | tee -a "${LOG}"; }

for required in \
  "${RUN_DIR}/config.yaml" \
  "${CKPT}" \
  "${PRETRAINED_NORM_STATS}" \
  "${TEXT_EMBEDDING_CACHE_DIR}" \
  "${CFG_TASK_DIR}/fold_glasses.yaml"
do
  [[ -e "${required}" ]] || { log "ERROR missing ${required}"; exit 2; }
done

IFS=',' read -r -a GPU_ARR <<< "${GPUS}"
if [[ "${#GPU_ARR[@]}" -ne 4 ]]; then
  log "ERROR: official 4×50 concurrent layout expects 4 GPUs, got ${GPUS}"
  exit 2
fi

gpus_idle() {
  python - "${GPUS}" <<'PY'
import subprocess, sys
gpus = [int(x) for x in sys.argv[1].split(",") if x.strip()]
out = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
    text=True,
)
rows = {}
for line in out.strip().splitlines():
    idx, mem, util = [p.strip() for p in line.split(",")]
    rows[int(idx)] = (int(float(mem)), int(float(util)))
ok = True
for g in gpus:
    mem, util = rows[g]
    print(f"gpu{g}: mem={mem}MiB util={util}%")
    if mem > 1500 or util > 10:
        ok = False
raise SystemExit(0 if ok else 1)
PY
}

if [[ "${WAIT_IDLE}" == "1" ]]; then
  log "waiting for GPUs ${GPUS} idle"
  idle_streak=0
  while true; do
    if gpus_idle >>"${LOG}" 2>&1; then
      idle_streak=$((idle_streak + 1))
      log "idle streak ${idle_streak}/2"
      [[ "${idle_streak}" -ge 2 ]] && break
    else
      idle_streak=0
    fi
    sleep 60
  done
fi

log "CFG success-vs-base scale=${CFG_SCALE} (not suffix-only scale=1)"
log "ckpt=${CKPT}"
log "gpus=${GPUS} protocol=seeds 0..49 × 4, replan=${REPLAN_STEPS}, max_steps=${MAX_ENV_STEPS}, nfe=${NUM_INFERENCE_STEPS}"
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

"${ENV_PREFIX}/bin/python" - "${OUT_ROOT}" "${CKPT_TAG}" "${CFG_SCALE}" "${MAX_ENV_STEPS}" <<'PY' | tee -a "${LOG}"
import json, statistics, sys
from pathlib import Path
root = Path(sys.argv[1])
ckpt_tag = sys.argv[2]
cfg_scale = float(sys.argv[3])
max_steps = int(sys.argv[4])
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
