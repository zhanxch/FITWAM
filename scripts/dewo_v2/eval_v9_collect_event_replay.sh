#!/usr/bin/env bash
# v9 base CFG eval on mixed-S0 collect recoverability events (oracle-once protocol).
#
# Mechanism validation: replay factual prefix to t*, then compare
#   v9_base (本体) vs v9_oracle_once (one forced CFG replan at t*, then 本体).
# Optional deploy test: CONDITIONS=v9_base,v9_cfg (value_growth gate).
#
#   TASK=fold_glasses RUN_DIR=... CKPT=... GPUS=4,5,6,7 \
#     bash scripts/dewo_v2/eval_v9_collect_event_replay.sh
#
# Paths (collect rollout, pair_index, stats, text cache) resolve from tasks.py unless overridden.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"
# shellcheck disable=SC1091
source "${ROOT_DIR}/scripts/dewo_v2/lib.sh"

dewo_v2_require_task
dewo_v2_require_gpus

CALLER_CKPT="${CKPT:-}"
CALLER_STATS="${PRETRAINED_NORM_STATS:-}"
CALLER_BACKBONE="${BACKBONE_CKPT:-}"
CALLER_COLLECT="${COLLECT_ROLLOUT:-}"
CALLER_PAIR_INDEX="${PAIR_INDEX:-}"
CALLER_PREFIX="${PREFIX_RESULTS:-}"
CALLER_TEXT="${TEXT_CACHE:-}"
CALLER_TASK_YAML="${TASK_YAML:-}"
dewo_v2_load_task "${TASK}"

RUN_DIR="${RUN_DIR:?Set RUN_DIR to the DEWO v9 training run directory}"
CKPT="${CALLER_CKPT:-${CKPT:-${RUN_DIR}/checkpoints/weights/step_005000.pt}}"
BACKBONE_CKPT="${CALLER_BACKBONE:-${BACKBONE_CKPT:-${ROOT_DIR}/checkpoints/dexjoco/mixed_5task_fastwam/weights/step_055000.pt}}"
dewo_v2_align_opensource_stack

PATHS_JSON="$(python "${ROOT_DIR}/scripts/dewo_v2/tasks.py" dump-collect-replay-paths \
  --task "${TASK}" \
  ${CALLER_COLLECT:+--collect-rollout "${CALLER_COLLECT}"})"
eval "$(python - "${PATHS_JSON}" <<'PY'
import json, shlex, sys
paths = json.loads(sys.argv[1])
for key, value in paths.items():
    print(f"export {key}={shlex.quote(value)}")
PY
)"

COLLECT_ROLLOUT="${CALLER_COLLECT:-${COLLECT_ROLLOUT}}"
PAIR_INDEX="${CALLER_PAIR_INDEX:-${PAIR_INDEX}}"
PREFIX_RESULTS="${CALLER_PREFIX:-${PREFIX_RESULTS}}"
PRETRAINED_NORM_STATS="${CALLER_STATS:-${PRETRAINED_NORM_STATS:-${STATS}}}"
TEXT_CACHE="${CALLER_TEXT:-${TEXT_CACHE:-${TEXT_EMBEDDING_CACHE_DIR:-}}}"
TASK_YAML="${CALLER_TASK_YAML:-}"

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/evaluate_results/dexjoco/${TASK}_v9_collect_event_replay_${STAMP}}"
CONDITIONS="${CONDITIONS:-v9_base,v9_oracle_once}"
ORACLE_CFG_SCALE="${ORACLE_CFG_SCALE:-1.1}"
PASS_M="${PASS_M:-4}"
REPLAN_STEPS="${REPLAN_STEPS:-24}"
MAX_STEPS="${MAX_STEPS:-1000}"

dewo_v2_activate_fastwam
export PYTHONPATH="${ROOT_DIR}/src:${ROOT_DIR}/scripts:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export ORACLE_CFG_SCALE TASK

mkdir -p "${OUT_ROOT}/logs"
LOG="${OUT_ROOT}/logs/orchestrator.log"
log() { echo "[v9-collect-replay $(date -Is)] $*" | tee -a "${LOG}"; }

for required in \
  "${CKPT}" \
  "${BACKBONE_CKPT}" \
  "${PRETRAINED_NORM_STATS}" \
  "${PAIR_INDEX}" \
  "${COLLECT_ROLLOUT}/collection_summary.json"
do
  [[ -e "${required}" ]] || { log "ERROR missing ${required}"; exit 2; }
done

IFS=',' read -r -a GPU_ARR <<< "${GPUS}"
NUM_SHARDS="${#GPU_ARR[@]}"
log "TASK=${TASK} OUT=${OUT_ROOT} ckpt=${CKPT} conditions=${CONDITIONS} oracle_scale=${ORACLE_CFG_SCALE} gpus=${GPUS}"

REPLAY_ARGS=(
  --collect-dataset "${COLLECT_ROLLOUT}"
  --pair-index "${PAIR_INDEX}"
  --prefix-results "${PREFIX_RESULTS}"
  --run-dir "${RUN_DIR}"
  --checkpoint "$(basename "${CKPT}")"
  --backbone-checkpoint "${BACKBONE_CKPT}"
  --uncond-adapter "${CKPT}"
  --dataset-stats "${PRETRAINED_NORM_STATS}"
  --text-cache "${TEXT_CACHE}"
  --task "${TASK}"
  --conditions "${CONDITIONS}"
  --oracle-cfg-scale "${ORACLE_CFG_SCALE}"
  --replan-steps "${REPLAN_STEPS}"
  --max-steps "${MAX_STEPS}"
  --pass-m "${PASS_M}"
  --output "${OUT_ROOT}"
)
if [[ -n "${TASK_YAML}" ]]; then
  REPLAY_ARGS+=(--task-yaml "${TASK_YAML}")
fi

PIDS=()
for i in "${!GPU_ARR[@]}"; do
  gpu="${GPU_ARR[$i]}"
  shard_log="${OUT_ROOT}/logs/shard_${i}.log"
  CUDA_VISIBLE_DEVICES="${gpu}" python "${ROOT_DIR}/scripts/fold_glasses/replay_v9_cfg_at_collect_event.py" \
    "${REPLAY_ARGS[@]}" \
    --device cuda:0 \
    --shard-index "${i}" \
    --num-shards "${NUM_SHARDS}" \
    > "${shard_log}" 2>&1 &
  PIDS+=("$!")
  log "shard ${i} gpu=${gpu} pid=$! log=${shard_log}"
done

FAIL=0
for pid in "${PIDS[@]}"; do
  if ! wait "${pid}"; then
    FAIL=1
  fi
done

python - <<'PY' "${OUT_ROOT}" "${NUM_SHARDS}" "${CONDITIONS}" "${TASK}" | tee -a "${LOG}"
import json, sys
from pathlib import Path

out = Path(sys.argv[1])
num_shards = int(sys.argv[2])
cond_names = [c.strip() for c in sys.argv[3].split(",") if c.strip()]
task = sys.argv[4]
rows = []
for i in range(num_shards):
    p = out / f"shard_{i}" / "results.jsonl"
    if not p.exists():
        continue
    rows.extend(json.loads(line) for line in p.read_text().splitlines() if line.strip())

def agg(name):
    sub = [r for r in rows if r["condition"] == name]
    hits = sum(bool(r["pass_at_m_hit"]) for r in sub)
    succs = sum(int(r["success_count"]) for r in sub)
    trials = sum(int(r["pass_m"]) for r in sub)
    return {
        "pairs": len(sub),
        "pass_at_m_pairs": hits,
        "pair_rate": hits / max(len(sub), 1),
        "replicate_successes": succs,
        "replicate_trials": trials,
        "replicate_rate": succs / max(trials, 1),
    }

base_name = cond_names[0] if cond_names else "v9_base"
cfg_name = cond_names[1] if len(cond_names) > 1 else "v9_oracle_once"
base = agg(base_name)
cfg = agg(cfg_name)
both = cfg_only = base_only = neither = 0
for pid in sorted({r["pair_id"] for r in rows}):
    b = next((r for r in rows if r["pair_id"] == pid and r["condition"] == base_name), None)
    c = next((r for r in rows if r["pair_id"] == pid and r["condition"] == cfg_name), None)
    bh = bool(b and b["pass_at_m_hit"])
    ch = bool(c and c["pass_at_m_hit"])
    if bh and ch:
        both += 1
    elif ch:
        cfg_only += 1
    elif bh:
        base_only += 1
    else:
        neither += 1

summary = {
    "protocol": "v9_base_cfg_eval_oracle_once_at_tstar",
    "task": task,
    "conditions": cond_names,
    "num_result_rows": len(rows),
    base_name: base,
    cfg_name: cfg,
    "pair_outcomes": {
        "both_success": both,
        f"{cfg_name}_only": cfg_only,
        f"{base_name}_only": base_only,
        "neither": neither,
    },
}
(out / "aggregate.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
PY

if [[ "${FAIL}" -ne 0 ]]; then
  log "ERROR one or more shards failed"
  exit 1
fi
log "done ${OUT_ROOT}/aggregate.json"
