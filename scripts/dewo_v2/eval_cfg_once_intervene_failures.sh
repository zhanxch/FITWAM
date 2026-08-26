#!/usr/bin/env bash
# Failure-only once-CFG rescue from a strength-band schedule (no tau).
# Expects partitions from probe_cfg_strength_once_intervene.sh.
#
#   BAND=q75_100 GPUS=0,1,2,3 CFG_SCALE=1.2 \
#     bash scripts/dewo_v2/eval_cfg_once_intervene_failures.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
# shellcheck disable=SC1091
source "${ROOT}/scripts/dewo_v2/lib.sh"

PROBE_ROOT="${PROBE_ROOT:-${ROOT}/evaluate_results/dexjoco/water_plant_dewo_v7_step1500_cfg_strength_probe_20260825}"
DESIGN_JSON="${DESIGN_JSON:-${PROBE_ROOT}/partitions/search_design/search_design.json}"
# Causal search: FORCE_REPLAN=i fires CFG once at that replan index (no energy band).
FORCE_REPLAN="${FORCE_REPLAN:-}"
if [[ -n "${FORCE_REPLAN}" ]]; then
  BAND="${BAND:-at${FORCE_REPLAN}}"
else
  BAND="${BAND:?Set BAND=q75_100|... or FORCE_REPLAN=i}"
fi
# fail | fragile | search (fail ∪ fragile). both_ok excluded from search.
SCREEN="${SCREEN:-fail}"
SCHEDULE="${SCHEDULE:-${PROBE_ROOT}/partitions/schedule_${BAND}_flat.json}"
PART_SUMMARY="${PART_SUMMARY:-${PROBE_ROOT}/partitions/partition_summary.json}"
FRAGILE_JSON="${FRAGILE_JSON:-${PROBE_ROOT}/partitions/fragile_from_always_cfg105.json}"
if [[ "${SCREEN}" != "fail" && "${SCREEN}" != "fragile" && "${SCREEN}" != "search" ]]; then
  echo "ERROR: SCREEN must be fail|fragile|search, got ${SCREEN}" >&2
  exit 2
fi

TASK="${TASK:-water_plant}"
GPUS="${GPUS:-0,1,2,3}"
WAIT_IDLE="${WAIT_IDLE:-0}"
CFG_SCALE="${CFG_SCALE:-1.2}"
RUN_DIR="${RUN_DIR:-${ROOT}/runs/dexjoco_water_plant_dewo_v7/2026-08-25_12-51-39_B1-jump-fast-v7-uncond-adapter}"
CKPT="${CKPT:-${RUN_DIR}/checkpoints/weights/step_001500.pt}"
BACKBONE_CKPT="${BACKBONE_CKPT:-${ROOT}/artifacts/opensource_ckpt_links/mixed_5task/step_055000.pt}"
PRETRAINED_NORM_STATS="${PRETRAINED_NORM_STATS:-${ROOT}/artifacts/mixed_5task/dataset_stats.json}"
TEXT_EMBEDDING_CACHE_DIR="${TEXT_EMBEDDING_CACHE_DIR:-${ROOT}/data/water_plant_mixed_s0_dewo_v2_pair_20260820_182236/text_embeds_cache}"
CFG_TASK_DIR="${CFG_TASK_DIR:-${ROOT}/configs/eval/dexjoco/water_plant_dewo_v7_cfg}"
DEXJOCO_PY_ROOT="${DEXJOCO_PY_ROOT:-${ROOT}/third_party/dexjoco/dexjoco}"
BASELINE_AGG="${BASELINE_AGG:-${ROOT}/evaluate_results/dexjoco/water_plant_dewo_v7_step1500_cfg1_本体_4x50_20260825_135146/aggregate.json}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-${ROOT}/evaluate_results/dexjoco/${TASK}_dewo_v7_step1500_once_${BAND}_${SCREEN}_${STAMP}}"
BASE_PORT="${BASE_PORT:-12000}"
REPLAN_STEPS="${REPLAN_STEPS:-24}"
# Search screen: 500 is enough (successes finish ~260–360). Official score still uses 1000.
MAX_ENV_STEPS="${MAX_ENV_STEPS:-500}"
ACTION_HORIZON="${ACTION_HORIZON:-32}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-10}"
INFERENCE_SEED="${INFERENCE_SEED:-20260812}"

dewo_v2_activate_fastwam
export PYTHONPATH="${ROOT}/src:${ROOT}/scripts:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
mkdir -p "${OUT_ROOT}/logs"
LOG="${OUT_ROOT}/logs/orchestrator.log"
log() { echo "[once-intervene ${BAND}/${SCREEN} $(date -Is)] $*" | tee -a "${LOG}"; }

IFS=',' read -r -a GPU_ARR <<< "${GPUS}"
if [[ "${#GPU_ARR[@]}" -ne 4 ]]; then
  log "ERROR: need 4 GPUs for 4 repeats, got ${GPUS}"
  exit 2
fi

log "schedule=${SCHEDULE} screen=${SCREEN} force_replan=${FORCE_REPLAN:-none}"
log "ckpt=${CKPT} cfg=${CFG_SCALE} gpus=${GPUS} max_steps=${MAX_ENV_STEPS}"
log "baseline_agg=${BASELINE_AGG}"

SEED_SRC="${DESIGN_JSON}"
if [[ ! -f "${SEED_SRC}" ]]; then
  if [[ "${SCREEN}" == "fragile" ]]; then SEED_SRC="${FRAGILE_JSON}"
  elif [[ "${SCREEN}" == "fail" ]]; then SEED_SRC="${PART_SUMMARY}"
  else
    log "ERROR missing ${DESIGN_JSON}"; exit 2
  fi
fi

mapfile -t SEED_LISTS < <("${ENV_PREFIX}/bin/python" - "${SEED_SRC}" "${SCREEN}" <<'PY'
import json, sys
from pathlib import Path
s = json.loads(Path(sys.argv[1]).read_text())
screen = sys.argv[2]
fail = s.get("fail_seeds_by_eval_repeat") or {}
frag = s.get("fragile_seeds_by_eval_repeat") or {}
for i in range(4):
    a = list(fail.get(str(i), []))
    b = list(frag.get(str(i), []))
    if screen == "fail":
        seeds = a
    elif screen == "fragile":
        seeds = b
    else:
        seeds = sorted(set(a) | set(b))
    print(",".join(str(x) for x in seeds))
PY
)
FAIL_SEEDS=("${SEED_LISTS[@]}")

if [[ -n "${FORCE_REPLAN}" ]]; then
  SCHEDULE="${OUT_ROOT}/force_replan_${FORCE_REPLAN}_flat.json"
  "${ENV_PREFIX}/bin/python" - "${SEED_SRC}" "${SCREEN}" "${FORCE_REPLAN}" "${SCHEDULE}" <<'PY'
import json, sys
from pathlib import Path
s = json.loads(Path(sys.argv[1]).read_text())
screen = sys.argv[2]
idx = int(sys.argv[3])
out = Path(sys.argv[4])
fail = s.get("fail_seeds_by_eval_repeat") or {}
frag = s.get("fragile_seeds_by_eval_repeat") or {}
by = {}
for i in range(4):
    a = list(fail.get(str(i), []))
    b = list(frag.get(str(i), []))
    if screen == "fail":
        seeds = a
    elif screen == "fragile":
        seeds = b
    else:
        seeds = sorted(set(a) | set(b))
    for seed in seeds:
        by[f"{i}:{int(seed)}"] = idx
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(by, indent=2) + "\n")
print(f"wrote {out} n={len(by)} force_replan={idx}")
PY
fi
test -f "${SCHEDULE}"

eval_one() {
  local run_i="$1"
  local gpu="$2"
  local port="$3"
  local eval_repeat="$4"
  local infer_seed="$5"
  local seed_list="$6"
  local out_dir="${OUT_ROOT}/run${run_i}"
  mkdir -p "${out_dir}"
  if [[ -z "${seed_list}" ]]; then
    log "run${run_i}: no seeds; skip"
    mkdir -p "${out_dir}"
    echo '{"overall_success_rate":0,"total_successes":0,"total_episodes":0,"tasks":[]}' > "${out_dir}/summary.json"
    return 0
  fi
  log "launch run${run_i} gpu=${gpu} seeds=${seed_list} eval_repeat=${eval_repeat}"
  "${ENV_PREFIX}/bin/python" scripts/dexjoco_async/run_multi_gpu_dexjoco_eval.py \
    --gpus "${gpu}" \
    --base-port "${port}" \
    --episodes 1 \
    --seed-list "${seed_list}" \
    --seed 0 \
    --eval-repeat "${eval_repeat}" \
    --inference-seed "${infer_seed}" \
    --text-cfg-scale "${CFG_SCALE}" \
    --cfg-gate-mode schedule \
    --cfg-intervene-schedule "${SCHEDULE}" \
    --action-horizon "${ACTION_HORIZON}" \
    --server-conda-env "${ENV_PREFIX}" \
    --client-conda-env dexjoco \
    --run-dir "${RUN_DIR}" \
    --checkpoint "${CKPT}" \
    --backbone-checkpoint "${BACKBONE_CKPT}" \
    --dataset-stats-path "${PRETRAINED_NORM_STATS}" \
    --text-embedding-cache-dir "${TEXT_EMBEDDING_CACHE_DIR}" \
    --no-load-text-encoder \
    --task-config-dir "${CFG_TASK_DIR}" \
    --tasks "${TASK}" \
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
    > "${OUT_ROOT}/logs/run${run_i}.log" 2>&1
}

pids=()
for i in 1 2 3 4; do
  (
    eval_one "${i}" "${GPU_ARR[$((i - 1))]}" "$((BASE_PORT + i * 20))" "$((i - 1))" "$((INFERENCE_SEED + i - 1))" "${FAIL_SEEDS[$((i - 1))]}"
  ) &
  pids+=("$!")
done
fail=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then fail=1; fi
done
if [[ "${fail}" -ne 0 ]]; then
  log "ERROR: one or more rescue runs failed"
  exit 2
fi

"${ENV_PREFIX}/bin/python" - "${OUT_ROOT}" "${BAND}" "${BASELINE_AGG}" "${SCHEDULE}" "${CFG_SCALE}" "${SCREEN}" "${SEED_SRC}" "${FORCE_REPLAN:-}" <<'PY' | tee -a "${LOG}"
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
band = sys.argv[2]
baseline = json.loads(Path(sys.argv[3]).read_text())
schedule = json.loads(Path(sys.argv[4]).read_text())
cfg = float(sys.argv[5])
screen = sys.argv[6]
seed_src = Path(sys.argv[7]) if len(sys.argv) > 7 else None
force_replan = sys.argv[8] if len(sys.argv) > 8 else ""
ok_n = 0
tried = 0
rows = []
for i in range(1, 5):
    summary = json.loads((root / f"run{i}" / "summary.json").read_text())
    for task in summary.get("tasks", []):
        for ep in task.get("episode_results", []):
            tried += 1
            success = bool(ep.get("success"))
            if success:
                ok_n += 1
            rows.append({"run": i, "seed": ep.get("seed"), "success": success})
base_pooled = int(baseline["pooled_successes"])
base_n = int(baseline["pooled_episodes"])
base_rate = float(baseline.get("pooled_success_rate") or 0.0)
out = {
    "band": band,
    "screen": screen,
    "force_replan": (None if force_replan == "" else int(force_replan)),
    "cfg_scale": cfg,
    "n_schedule_keys": len(schedule),
    "episodes_tried": tried,
    "episodes_success": ok_n,
    "episodes": rows,
}
fail_keys = set()
frag_keys = set()
if seed_src is not None and seed_src.is_file():
    payload = json.loads(seed_src.read_text())
    for rep, seeds in (payload.get("fail_seeds_by_eval_repeat") or {}).items():
        for seed in seeds:
            fail_keys.add((int(rep) + 1, int(seed)))
    for rep, seeds in (payload.get("fragile_seeds_by_eval_repeat") or {}).items():
        for seed in seeds:
            frag_keys.add((int(rep) + 1, int(seed)))

def _split(keys):
    hit = miss = 0
    for row in rows:
        k = (int(row["run"]), int(row["seed"]))
        if k not in keys:
            continue
        if row["success"]:
            hit += 1
        else:
            miss += 1
    return hit, miss

if screen == "fail":
    hyp_pooled = base_pooled + ok_n
    hyp_rate = hyp_pooled / base_n if base_n else float("nan")
    out.update(
        {
            "failures_tried": tried,
            "failures_rescued": ok_n,
            "rescue_rate": (ok_n / tried) if tried else None,
            "baseline_pooled_success_rate": base_rate,
            "hypothetical_pooled_successes": hyp_pooled,
            "hypothetical_pooled_success_rate": hyp_rate,
            "delta_pooled_success_rate": hyp_rate - base_rate,
        }
    )
elif screen == "fragile":
    broken = tried - ok_n
    out.update(
        {
            "fragile_tried": tried,
            "fragile_survived": ok_n,
            "fragile_broken": broken,
            "break_rate": (broken / tried) if tried else None,
            "baseline_pooled_success_rate": base_rate,
        }
    )
else:
    rescued, fail_still = _split(fail_keys)
    frag_ok, broken = _split(frag_keys)
    net = rescued - broken
    hyp_pooled = base_pooled + rescued - broken
    hyp_rate = hyp_pooled / base_n if base_n else float("nan")
    out.update(
        {
            "failures_rescued": rescued,
            "failures_still_fail": fail_still,
            "fragile_survived": frag_ok,
            "fragile_broken": broken,
            "net_rescued_minus_broken": net,
            "baseline_pooled_success_rate": base_rate,
            "hypothetical_pooled_successes": hyp_pooled,
            "hypothetical_pooled_success_rate": hyp_rate,
            "delta_pooled_success_rate": hyp_rate - base_rate,
            "note": "both_ok assumed unchanged. Official 4x50 still required.",
        }
    )
(root / "rescue_summary.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out, indent=2))
PY

log "done OUT_ROOT=${OUT_ROOT}"
