#!/usr/bin/env bash
# fold_glasses: wait for opensource eval to free GPUs 0-7 (or enough VRAM),
# then squeeze 4×50 opensource-aligned collection on 0-7, then pilot
# failure-event extraction via action-width jump on the collected failures.
set -euo pipefail

ROOT=/data_all/xiangchengzhan/FastWAM
ENV_PREFIX=/home/xiangchengzhan/anaconda3/envs/fastwam
cd "${ROOT}"

source /home/xiangchengzhan/anaconda3/etc/profile.d/conda.sh
conda activate fastwam
export PATH="${ENV_PREFIX}/bin:${PATH}"
export MUJOCO_GL=egl
export TOKENIZERS_PARALLELISM=false
export DIFFSYNTH_SKIP_DOWNLOAD=true
export DIFFSYNTH_MODEL_BASE_PATH="${ROOT}/checkpoints"

GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
EVAL_OUT="${EVAL_OUT:-${ROOT}/evaluate_results/dexjoco/fold_glasses_opensource_exact_4x50_20260812_102114}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
COLLECT_OUT="${COLLECT_OUT:-${ROOT}/data/fold_glasses_opensource_s0_collect_4x50_${STAMP}}"
PILOT_OUT="${PILOT_OUT:-${COLLECT_OUT}/failure_event_width_jump_pilot}"
MIN_FREE_MIB="${MIN_FREE_MIB:-18000}"
LOG_DIR="${COLLECT_OUT}/logs"
mkdir -p "${LOG_DIR}"
MASTER="${LOG_DIR}/orchestrator_${STAMP}.log"

log() { echo "[fg-collect-pilot $(date -Is)] $*" | tee -a "${MASTER}"; }

gpus_ready() {
  python - "${GPUS}" "${MIN_FREE_MIB}" <<'PY'
import subprocess, sys
gpus=[int(x) for x in sys.argv[1].split(",") if x.strip()]
need=int(sys.argv[2])
# block while opensource eval_dexjoco still owns these GPUs
ps=subprocess.check_output(["ps","-eo","pid,cmd"], text=True)
for line in ps.splitlines():
    if "scripts/eval_dexjoco.py" in line and "fold_glasses" in line:
        print(f"eval_still_running: {line.strip()[:180]}")
        raise SystemExit(1)
out=subprocess.check_output(
    ["nvidia-smi","--query-gpu=index,memory.free","--format=csv,noheader,nounits"],
    text=True,
)
rows={}
for line in out.strip().splitlines():
    idx, free = [p.strip() for p in line.split(",")]
    rows[int(idx)] = int(float(free))
ok=True
for g in gpus:
    free=rows.get(g, 0)
    print(f"gpu{g}: free={free}MiB need>={need}")
    if free < need:
        ok=False
raise SystemExit(0 if ok else 1)
PY
}

log "wait until eval_dexjoco gone AND each GPU free>=${MIN_FREE_MIB}MiB (gpus=${GPUS})"
idle=0
while true; do
  if gpus_ready >>"${MASTER}" 2>&1; then
    idle=$((idle + 1))
    log "ready streak ${idle}/2"
    [[ "${idle}" -ge 2 ]] && break
  else
    idle=0
  fi
  sleep 45
done

log "LAUNCH collect -> ${COLLECT_OUT}"
"${ENV_PREFIX}/bin/python" scripts/fold_glasses/collect_opensource_4x50.py \
  --gpus "${GPUS}" \
  --seed-start 10086 \
  --seed-end 10135 \
  --repeats 4 \
  --max-steps 1200 \
  --overwrite \
  --output-dir "${COLLECT_OUT}" \
  2>&1 | tee -a "${LOG_DIR}/collect_${STAMP}.log"

RAW="${COLLECT_OUT}/rollout_raw_200"
[[ -d "${RAW}/meta" ]] || { log "ERROR missing ${RAW}"; exit 2; }

log "LAUNCH failure-event pilot on failures in ${RAW}"
# Prefer a free GPU; default 0 after collect exited.
"${ENV_PREFIX}/bin/python" scripts/analysis/pilot_failure_event_width_jump_on_dataset.py \
  --dataset "${RAW}" \
  --output "${PILOT_OUT}" \
  --gpu 0 \
  --num-samples 8 \
  --stride 24 \
  --ignore-seconds 4 \
  --jump-ratio 2.0 \
  --jump-abs 0.02 \
  --stop-on-first-event \
  2>&1 | tee -a "${LOG_DIR}/pilot_${STAMP}.log"

log "DONE collect=${COLLECT_OUT} pilot=${PILOT_OUT}"
log "aggregate=${COLLECT_OUT}/aggregate.json pilot_summary=${PILOT_OUT}/summary.json"
