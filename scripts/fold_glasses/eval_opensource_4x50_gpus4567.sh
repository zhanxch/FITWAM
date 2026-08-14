#!/usr/bin/env bash
# Exact open-source fold_glasses reproduction (MichaelGaoZT/FastWAM-infer-in-DexJoco).
#
# Pins matching their README:
#   FastWAM  45d8e14
#   DexJoco  8d23b0f  (third_party/dexjoco)
# Protocol: seeds 0..49 × 4 repeats, replan=24, max_steps=1200, nfe=10, no DR.
# Uses THEIR artifacts (dataset_stats + text embedding), THEIR eval_dexjoco.py.
set -euo pipefail

ROOT=/data_all/xiangchengzhan/FastWAM
OPEN=/data_all/xiangchengzhan/FastWAM-infer-in-DexJoco
FASTWAM_PIN="${FASTWAM_PIN:-${ROOT}/third_party/FastWAM_pin_45d8e14}"
ENV_PREFIX=/home/xiangchengzhan/anaconda3/envs/fastwam
GPUS="${GPUS:-4,5,6,7}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-${ROOT}/evaluate_results/dexjoco/fold_glasses_opensource_exact_4x50_${STAMP}}"
CKPT_DIR="${CKPT_DIR:-${OPEN}/checkpoints/fold_glasses}"
LOG_DIR="${OUT_ROOT}/logs"
mkdir -p "${LOG_DIR}"
MASTER_LOG="${LOG_DIR}/orchestrator_${STAMP}.log"

source /home/xiangchengzhan/anaconda3/etc/profile.d/conda.sh
conda activate fastwam
export PATH="${ENV_PREFIX}/bin:${PATH}"
# Prefer pinned FastWAM over current workspace src.
export PYTHONPATH="${OPEN}/src:${FASTWAM_PIN}/src:${ROOT}/third_party/dexjoco/dexjoco:${PYTHONPATH:-}"
export MUJOCO_GL=egl
export TOKENIZERS_PARALLELISM=false
export DIFFSYNTH_SKIP_DOWNLOAD=true
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-${ROOT}/checkpoints}"

log() { echo "[fold-glasses-opensource-exact $(date -Is)] $*" | tee -a "${MASTER_LOG}"; }

# Sanity: exact pins
pin_head="$(git -C "${FASTWAM_PIN}" rev-parse HEAD)"
dex_head="$(git -C "${ROOT}/third_party/dexjoco" rev-parse HEAD)"
log "FastWAM_PIN=${FASTWAM_PIN} HEAD=${pin_head}"
log "DexJoco HEAD=${dex_head}"
if [[ "${pin_head}" != 45d8e1458921d83f8ad6cf9ce993d371208dabd0 ]]; then
  log "ERROR: FastWAM pin mismatch"
  exit 1
fi
if [[ "${dex_head}" != 8d23b0fab23b17a58c4b55f3942e17013aaf8267 ]]; then
  log "ERROR: DexJoco pin mismatch (expected 8d23b0f)"
  exit 1
fi
for p in \
  "${OPEN}/scripts/eval_dexjoco.py" \
  "${OPEN}/configs/fastwam_dexjoco.yaml" \
  "${OPEN}/artifacts/fold_glasses/dataset_stats.json" \
  "${OPEN}/artifacts/fold_glasses/0c3367ce1d74848cc46b93c6d2eee5e2097dca410a2c95f3da48bd8c8673fa20.t5_len128.wan22ti2v5b.pt" \
  "${CKPT_DIR}/step_010000.pt"
do
  [[ -e "$p" ]] || { log "ERROR missing $p"; exit 1; }
done

gpus_idle() {
  python - "${GPUS}" <<'PY'
import subprocess
import sys

gpus = [int(x) for x in sys.argv[1].split(",") if x.strip()]
ps = subprocess.check_output(["ps", "-eo", "pid,cmd"], text=True)
for line in ps.splitlines():
    if "max600" in line and "gpus4567" in line:
        print(f"busy_proc: {line.strip()[:200]}")
        raise SystemExit(1)
    if "run_multi_gpu_dexjoco_eval.py" in line and (
        "--gpus 4,5,6,7" in line
        or "--gpus 4 " in (line + " ")
        or "--gpus 5 " in (line + " ")
        or "--gpus 6 " in (line + " ")
        or "--gpus 7 " in (line + " ")
    ):
        print(f"busy_proc: {line.strip()[:200]}")
        raise SystemExit(1)

out = subprocess.check_output(
    [
        "nvidia-smi",
        "--query-gpu=index,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ],
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

log "waiting for GPUs ${GPUS} idle (and max600 gone)"
idle_streak=0
while true; do
  if gpus_idle >>"${MASTER_LOG}" 2>&1; then
    idle_streak=$((idle_streak + 1))
    log "idle streak ${idle_streak}/2"
    if [[ "${idle_streak}" -ge 2 ]]; then
      break
    fi
  else
    idle_streak=0
  fi
  sleep 60
done
log "GPUs ${GPUS} idle; launching EXACT open-source eval -> ${OUT_ROOT}"

# Run from OPEN so relative defaults match their repo; absolute paths still passed.
cd "${OPEN}"
"${ENV_PREFIX}/bin/python" scripts/eval_dexjoco.py \
  --task-name fold_glasses \
  --checkpoint-dir "${CKPT_DIR}" \
  --checkpoint-steps 10000 \
  --model-config "${OPEN}/configs/fastwam_dexjoco.yaml" \
  --dataset-stats "${OPEN}/artifacts/fold_glasses/dataset_stats.json" \
  --text-embedding "${OPEN}/artifacts/fold_glasses/0c3367ce1d74848cc46b93c6d2eee5e2097dca410a2c95f3da48bd8c8673fa20.t5_len128.wan22ti2v5b.pt" \
  --gpus "${GPUS}" \
  --seed-start 0 \
  --seed-end 49 \
  --repeats 4 \
  --action-horizon 32 \
  --replan-steps 24 \
  --num-inference-steps 10 \
  --max-steps 1200 \
  --output-dir "${OUT_ROOT}" \
  2>&1 | tee -a "${LOG_DIR}/eval_${STAMP}.log"

log "DONE summary=${OUT_ROOT}/summary.json"
