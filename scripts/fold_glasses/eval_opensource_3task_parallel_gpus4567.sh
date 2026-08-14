#!/usr/bin/env bash
# Exact open-source multi-task eval (MichaelGaoZT/FastWAM-infer-in-DexJoco).
#
# Simultaneously runs best checkpoints for:
#   fold_glasses  step_010000
#   hammer_nail   step_002500
#   pick_bucket   step_010000
# Each task uses all target GPUs (round-robin seeds) → 3 MoT models / GPU on A100-80GB.
#
# Pins: FastWAM 45d8e14, DexJoco 8d23b0f
# Protocol: seeds 0..49 × 4 repeats, replan=24, max_steps=1200, nfe=10, no DR
set -euo pipefail

ROOT=/data_all/xiangchengzhan/FastWAM
OPEN=/data_all/xiangchengzhan/FastWAM-infer-in-DexJoco
FASTWAM_PIN="${FASTWAM_PIN:-${ROOT}/third_party/FastWAM_pin_45d8e14}"
ENV_PREFIX=/home/xiangchengzhan/anaconda3/envs/fastwam
GPUS="${GPUS:-0,1,2,3}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-${ROOT}/evaluate_results/dexjoco/opensource_exact_3task_4x50_${STAMP}}"
LOG_DIR="${OUT_ROOT}/logs"
mkdir -p "${LOG_DIR}"
MASTER_LOG="${LOG_DIR}/orchestrator_${STAMP}.log"

TASKS=(
  "fold_glasses|10000|${OPEN}/checkpoints/fold_glasses"
  "hammer_nail|2500|${OPEN}/checkpoints/hammer_nail"
  "pick_bucket|10000|${OPEN}/checkpoints/pick_bucket"
)

source /home/xiangchengzhan/anaconda3/etc/profile.d/conda.sh
conda activate fastwam
export PATH="${ENV_PREFIX}/bin:${PATH}"
export PYTHONPATH="${OPEN}/src:${FASTWAM_PIN}/src:${ROOT}/third_party/dexjoco/dexjoco:${PYTHONPATH:-}"
export MUJOCO_GL=egl
export TOKENIZERS_PARALLELISM=false
export DIFFSYNTH_SKIP_DOWNLOAD=true
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-${ROOT}/checkpoints}"

log() { echo "[opensource-3task $(date -Is)] $*" | tee -a "${MASTER_LOG}"; }

pin_head="$(git -C "${FASTWAM_PIN}" rev-parse HEAD)"
dex_head="$(git -C "${ROOT}/third_party/dexjoco" rev-parse HEAD)"
log "FastWAM_PIN HEAD=${pin_head}"
log "DexJoco HEAD=${dex_head}"
log "GPUS=${GPUS}"
[[ "${pin_head}" == 45d8e1458921d83f8ad6cf9ce993d371208dabd0 ]] || { log "ERROR FastWAM pin"; exit 1; }
[[ "${dex_head}" == 8d23b0fab23b17a58c4b55f3942e17013aaf8267 ]] || { log "ERROR DexJoco pin"; exit 1; }

for spec in "${TASKS[@]}"; do
  IFS='|' read -r task step ckpt_dir <<<"${spec}"
  ckpt="${ckpt_dir}/step_$(printf '%06d' "${step}").pt"
  stats="${OPEN}/artifacts/${task}/dataset_stats.json"
  emb=$(ls "${OPEN}/artifacts/${task}/"*.t5_len128.wan22ti2v5b.pt | head -1)
  [[ -e "${ckpt}" ]] || { log "ERROR missing ${ckpt}"; exit 1; }
  [[ -f "${stats}" ]] || { log "ERROR missing ${stats}"; exit 1; }
  [[ -f "${emb}" ]] || { log "ERROR missing text embed for ${task}"; exit 1; }
  bytes=$(stat -L -c%s "${ckpt}")
  log "ready ${task} step=${step} ckpt_bytes=${bytes} emb=$(basename "${emb}")"
done

gpus_exclusive_idle() {
  python - "${GPUS}" <<'PY'
import subprocess
import sys

gpus = [int(x) for x in sys.argv[1].split(",") if x.strip()]
gpu_set = set(gpus)

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
    if mem > 2000 or util > 10:
        ok = False

# Also reject foreign multi-gpu evals that claim any of our GPUs.
ps = subprocess.check_output(["ps", "-eo", "pid,cmd"], text=True)
for line in ps.splitlines():
    if "run_multi_gpu_dexjoco_eval.py" not in line:
        continue
    if "opensource_exact_3task" in line or "FastWAM-infer-in-DexJoco" in line:
        continue
    claimed = set()
    if "--gpus" in line:
        try:
            part = line.split("--gpus", 1)[1].strip().split()[0]
            claimed = {int(x) for x in part.split(",") if x.strip().isdigit()}
        except Exception:
            claimed = set()
    if claimed & gpu_set:
        print(f"busy_proc: {line.strip()[:220]}")
        raise SystemExit(1)

raise SystemExit(0 if ok else 1)
PY
}

log "waiting for exclusive idle on GPUs ${GPUS}"
idle_streak=0
while true; do
  if gpus_exclusive_idle >>"${MASTER_LOG}" 2>&1; then
    idle_streak=$((idle_streak + 1))
    log "idle streak ${idle_streak}/2"
    [[ "${idle_streak}" -ge 2 ]] && break
  else
    idle_streak=0
  fi
  sleep 30
done

log "launching 3 tasks in parallel on GPUs ${GPUS} -> ${OUT_ROOT}"
cd "${OPEN}"

pids=()
for spec in "${TASKS[@]}"; do
  IFS='|' read -r task step ckpt_dir <<<"${spec}"
  emb=$(ls "${OPEN}/artifacts/${task}/"*.t5_len128.wan22ti2v5b.pt | head -1)
  task_out="${OUT_ROOT}/${task}"
  mkdir -p "${task_out}"
  log "start ${task} step=${step} out=${task_out}"
  (
    "${ENV_PREFIX}/bin/python" scripts/eval_dexjoco.py \
      --task-name "${task}" \
      --checkpoint-dir "${ckpt_dir}" \
      --checkpoint-steps "${step}" \
      --model-config "${OPEN}/configs/fastwam_dexjoco.yaml" \
      --dataset-stats "${OPEN}/artifacts/${task}/dataset_stats.json" \
      --text-embedding "${emb}" \
      --gpus "${GPUS}" \
      --seed-start 0 \
      --seed-end 49 \
      --repeats 4 \
      --action-horizon 32 \
      --replan-steps 24 \
      --num-inference-steps 10 \
      --max-steps 1200 \
      --output-dir "${task_out}" \
      > "${LOG_DIR}/eval_${task}_${STAMP}.log" 2>&1
  ) &
  pids+=($!)
done

fail=0
for i in "${!pids[@]}"; do
  pid="${pids[$i]}"
  IFS='|' read -r task _ _ <<<"${TASKS[$i]}"
  if wait "${pid}"; then
    log "DONE ${task} pid=${pid}"
  else
    log "FAIL ${task} pid=${pid} exit=$?"
    fail=1
  fi
done

python - <<'PY' "${OUT_ROOT}" "${MASTER_LOG}"
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
log_path = Path(sys.argv[2])
rows = []
for task in ("fold_glasses", "hammer_nail", "pick_bucket"):
    candidates = list(root.glob(f"{task}/step_*/summary.json")) + list(root.glob(f"{task}/summary.json"))
    for p in candidates:
        try:
            data = json.loads(p.read_text())
        except Exception as e:
            print(f"skip {p}: {e}")
            continue
        if isinstance(data, list):
            for item in data:
                item = dict(item)
                item["task"] = task
                item["path"] = str(p)
                rows.append(item)
        elif isinstance(data, dict):
            item = dict(data)
            item["task"] = task
            item["path"] = str(p)
            rows.append(item)
out = root / "aggregate_summary.json"
out.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
with log_path.open("a") as f:
    f.write(f"[opensource-3task] aggregate -> {out}\n")
    for r in rows:
        rate = r.get("success_rate")
        ep = r.get("episodes")
        suc = r.get("successes")
        f.write(
            f"[opensource-3task] {r.get('task')} step={r.get('checkpoint_step')} "
            f"{suc}/{ep} ({None if rate is None else 100*float(rate):.2f}%)\n"
        )
print(out)
PY

log "ALL complete fail=${fail} aggregate=${OUT_ROOT}/aggregate_summary.json"
exit "${fail}"
