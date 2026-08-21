#!/usr/bin/env bash
# Official CFG 4×50 for every weight ckpt in a DEWO v2 run (newest → oldest).
#   TASK=fold_glasses RUN_DIR=... TEXT_EMBEDDING_CACHE_DIR=... GPUS=4,5,6,7 \
#     bash scripts/dewo_v2/eval_cfg_ckpt_ladder.sh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"
# shellcheck disable=SC1091
source "${ROOT_DIR}/scripts/dewo_v2/lib.sh"

dewo_v2_require_task
dewo_v2_require_gpus
dewo_v2_load_task "${TASK}"

RUN_DIR="${RUN_DIR:?Set RUN_DIR to the DEWO v2 training run directory}"
WEIGHTS_DIR="${WEIGHTS_DIR:-${RUN_DIR}/checkpoints/weights}"
CKPT_STEPS="${CKPT_STEPS:-15000 12500 10000 7500 5000 2500}"
WAIT_IDLE="${WAIT_IDLE:-0}"
CFG_SCALE="${CFG_SCALE:-2.0}"
BASE_PORT="${BASE_PORT:-6600}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
LADDER_ROOT="${LADDER_ROOT:-${ROOT_DIR}/evaluate_results/dexjoco/${TASK}_dewo_v2_cfg${CFG_SCALE}_ladder_${STAMP}}"
TEXT_EMBEDDING_CACHE_DIR="${TEXT_EMBEDDING_CACHE_DIR:?Set TEXT_EMBEDDING_CACHE_DIR}"
LOG_DIR="${LADDER_ROOT}/logs"
mkdir -p "${LOG_DIR}"
MASTER_LOG="${LOG_DIR}/ladder.log"

log() { echo "[dewo-cfg-ladder $(date -Is)] $*" | tee -a "${MASTER_LOG}"; }

log "task=${TASK} run_dir=${RUN_DIR}"
log "steps(reverse)=${CKPT_STEPS}"
log "gpus=${GPUS} wait_idle=${WAIT_IDLE} cfg_scale=${CFG_SCALE}"
log "ladder_root=${LADDER_ROOT}"

port_offset=0
for step in ${CKPT_STEPS}; do
  tag="$(printf 'step_%06d' "${step}")"
  ckpt="${WEIGHTS_DIR}/${tag}.pt"
  if [[ ! -f "${ckpt}" ]]; then
    log "ERROR missing ${ckpt}"
    exit 2
  fi
  out="${LADDER_ROOT}/${tag}_cfg${CFG_SCALE}_4x50"
  log "===== eval ${tag} → ${out} ====="
  TASK="${TASK}" \
  RUN_DIR="${RUN_DIR}" \
  CKPT="${ckpt}" \
  CKPT_TAG="${tag}" \
  GPUS="${GPUS}" \
  WAIT_IDLE="${WAIT_IDLE}" \
  CFG_SCALE="${CFG_SCALE}" \
  BASE_PORT="$((BASE_PORT + port_offset))" \
  OUT_ROOT="${out}" \
  TEXT_EMBEDDING_CACHE_DIR="${TEXT_EMBEDDING_CACHE_DIR}" \
  PRETRAINED_NORM_STATS="${PRETRAINED_NORM_STATS:-${STATS}}" \
  bash "${ROOT_DIR}/scripts/dewo_v2/eval_cfg_official_4x50.sh" \
    2>&1 | tee -a "${MASTER_LOG}"
  port_offset=$((port_offset + 100))
  log "===== done ${tag} ====="
done

python - "${LADDER_ROOT}" "${CFG_SCALE}" "${TASK}" <<'PY' | tee -a "${MASTER_LOG}"
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
cfg = sys.argv[2]
task = sys.argv[3]
rows = []
for agg in sorted(root.glob("step_*/aggregate.json"), reverse=True):
    d = json.loads(agg.read_text())
    rows.append({
        "ckpt_tag": d.get("ckpt_tag"),
        "mean_success_rate": d.get("mean_success_rate"),
        "var_success_rate": d.get("var_success_rate"),
        "std_success_rate": d.get("std_success_rate"),
        "pooled_successes": d.get("pooled_successes"),
        "pooled_episodes": d.get("pooled_episodes"),
        "pooled_success_rate": d.get("pooled_success_rate"),
        "path": str(agg.parent),
    })
out = {
    "task": task,
    "method": "dewo_v2_cfg_ladder_reverse",
    "text_cfg_scale": float(cfg),
    "protocol": "official_4x50_seeds_0_49",
    "checkpoints": rows,
}
(root / "ladder_summary.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out, indent=2))
for r in rows:
    print(
        f"{r['ckpt_tag']}: mean={r['mean_success_rate']:.4f} "
        f"var={r['var_success_rate']:.6f} "
        f"pooled={r['pooled_successes']}/{r['pooled_episodes']}"
    )
PY

log "DONE ladder_summary=${LADDER_ROOT}/ladder_summary.json"
