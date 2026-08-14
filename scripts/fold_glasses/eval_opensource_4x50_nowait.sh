#!/usr/bin/env bash
# fold_glasses official 4×50 via open-source FastWAM-infer-in-DexJoco stack.
# 1 process/GPU (no packed). No idle wait — launch immediately.
#
# Usage:
#   GPUS=0,1,2,3,4,5,6,7 bash scripts/fold_glasses/eval_opensource_4x50_nowait.sh
set -euo pipefail

ROOT=/data_all/xiangchengzhan/FastWAM
OPEN=/data_all/xiangchengzhan/FastWAM-infer-in-DexJoco
FASTWAM_PIN="${FASTWAM_PIN:-${ROOT}/third_party/FastWAM_pin_45d8e14}"
ENV_PREFIX=/home/xiangchengzhan/anaconda3/envs/fastwam
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-${ROOT}/evaluate_results/dexjoco/fold_glasses_opensource_exact_4x50_${STAMP}}"
CKPT_DIR="${CKPT_DIR:-${OPEN}/checkpoints/fold_glasses}"
LOG_DIR="${OUT_ROOT}/logs"
mkdir -p "${LOG_DIR}"
MASTER_LOG="${LOG_DIR}/orchestrator_${STAMP}.log"

source /home/xiangchengzhan/anaconda3/etc/profile.d/conda.sh
conda activate fastwam
export PATH="${ENV_PREFIX}/bin:${PATH}"
export PYTHONPATH="${OPEN}/src:${FASTWAM_PIN}/src:${ROOT}/third_party/dexjoco/dexjoco:${PYTHONPATH:-}"
export MUJOCO_GL=egl
export TOKENIZERS_PARALLELISM=false
export DIFFSYNTH_SKIP_DOWNLOAD=true
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-${ROOT}/checkpoints}"

log() { echo "[fold-glasses-opensource $(date -Is)] $*" | tee -a "${MASTER_LOG}"; }

pin_head="$(git -C "${FASTWAM_PIN}" rev-parse HEAD)"
dex_head="$(git -C "${ROOT}/third_party/dexjoco" rev-parse HEAD)"
log "FastWAM_PIN=${FASTWAM_PIN} HEAD=${pin_head}"
log "DexJoco HEAD=${dex_head}"
if [[ "${pin_head}" != 45d8e1458921d83f8ad6cf9ce993d371208dabd0 ]]; then
  log "ERROR: FastWAM pin mismatch"; exit 1
fi
if [[ "${dex_head}" != 8d23b0fab23b17a58c4b55f3942e17013aaf8267 ]]; then
  log "ERROR: DexJoco pin mismatch"; exit 1
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

log "NO idle wait; launch immediate gpus=${GPUS}"
log "ckpt_dir=${CKPT_DIR} (same weights as fold_glasses_fastwam/step_010000.pt)"
log "out=${OUT_ROOT}"
log "protocol: seeds 0..49 × 4 repeats, replan=24, max_steps=1200, nfe=10"

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
# Print mean±var across repeats if summary has the fields
"${ENV_PREFIX}/bin/python" - "${OUT_ROOT}/summary.json" <<'PY' | tee -a "${MASTER_LOG}"
import json, statistics, sys
from pathlib import Path
p = Path(sys.argv[1])
if not p.exists():
    print("missing summary"); raise SystemExit(1)
d = json.loads(p.read_text())
print(json.dumps(d, indent=2)[:4000])
# try extract per-repeat rates if present
rates = d.get("repeat_success_rates") or d.get("per_repeat_success_rate")
if rates:
    mean = statistics.fmean(rates)
    var = statistics.pvariance(rates) if len(rates) > 1 else 0.0
    print(f"mean={mean:.4f} var={var:.6f} pooled={d.get('success_rate') or d.get('overall_success_rate')}")
PY
