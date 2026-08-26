#!/usr/bin/env bash
# Dump CFG gate energy only on search strata: 本体 failures ∪ fragile successes.
# Excludes both_ok (全程 CFG 仍成功). Early window only (default step<=400).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

EVAL_ROOT="${EVAL_ROOT:-${ROOT}/evaluate_results/dexjoco/water_plant_dewo_v7_step1500_cfg1_本体_4x50_20260825_135146}"
PROBE_ROOT="${PROBE_ROOT:-${ROOT}/evaluate_results/dexjoco/water_plant_dewo_v7_step1500_cfg_strength_probe_20260825}"
DESIGN_DIR="${DESIGN_DIR:-${PROBE_ROOT}/partitions/search_design}"
RUN_DIR="${RUN_DIR:-${ROOT}/runs/dexjoco_water_plant_dewo_v7/2026-08-25_12-51-39_B1-jump-fast-v7-uncond-adapter}"
CKPT="${CKPT:-${RUN_DIR}/checkpoints/weights/step_001500.pt}"
BACKBONE="${BACKBONE_CKPT:-${ROOT}/artifacts/opensource_ckpt_links/mixed_5task/step_055000.pt}"
STATS="${PRETRAINED_NORM_STATS:-${ROOT}/artifacts/mixed_5task/dataset_stats.json}"
TEXT_CACHE="${TEXT_EMBEDDING_CACHE_DIR:-${ROOT}/data/water_plant_mixed_s0_dewo_v2_pair_20260820_182236/text_embeds_cache}"
CFG_TASK_DIR="${CFG_TASK_DIR:-${ROOT}/configs/eval/dexjoco/water_plant_dewo_v7_cfg}"
GPUS="${GPUS:-0,1,2,3}"
CFG_SCALE="${CFG_SCALE:-1.2}"
PROBE_TAU="${PROBE_TAU:-1e9}"
MAX_QUERY_STEP="${MAX_QUERY_STEP:-400}"

test -f "${DESIGN_DIR}/search_design.json"
IFS=',' read -r -a GPU_ARR <<< "${GPUS}"
mkdir -p "${PROBE_ROOT}/logs"

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fastwam
export PYTHONPATH="${ROOT}/src:${ROOT}/scripts:${ROOT}/scripts/dexjoco_async:${PYTHONPATH:-}"

# Per-run seed lists = fail ∪ fragile
mapfile -t SEED_LISTS < <(python3 - "${DESIGN_DIR}/search_design.json" <<'PY'
import json, sys
from pathlib import Path
d = json.loads(Path(sys.argv[1]).read_text())
fail = d["fail_seeds_by_eval_repeat"]
frag = d["fragile_seeds_by_eval_repeat"]
for i in range(4):
    seeds = sorted(set(fail.get(str(i), [])) | set(frag.get(str(i), [])))
    print(",".join(str(x) for x in seeds))
PY
)

dump_pids=()
for i in 1 2 3 4; do
  gpu="${GPU_ARR[$((i - 1))]}"
  seeds="${SEED_LISTS[$((i - 1))]}"
  out="${PROBE_ROOT}/run${i}"
  mkdir -p "${out}/residual"
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    # Dump only listed seeds via repeated --seed-start/--seed-end single seeds
    IFS=',' read -r -a arr <<< "${seeds}"
    for seed in "${arr[@]}"; do
      dest="${out}/residual/seed_$(printf '%03d' "${seed}").npz"
      if [[ -f "${dest}" ]]; then
        echo "[skip existing] ${dest}"
        continue
      fi
      python "${ROOT}/scripts/analysis/dump_cfg_residual_on_eval_traj.py" \
        --phase dump \
        --eval-run "${EVAL_ROOT}/run${i}" \
        --out-dir "${out}" \
        --task water_plant \
        --task-config-dir "${CFG_TASK_DIR}" \
        --device cuda:0 \
        --checkpoint "${CKPT}" \
        --backbone-checkpoint "${BACKBONE}" \
        --dataset-stats "${STATS}" \
        --text-embedding-cache-dir "${TEXT_CACHE}" \
        --text-cfg-scale "${CFG_SCALE}" \
        --adaptive-cfg-tau "${PROBE_TAU}" \
        --max-query-step "${MAX_QUERY_STEP}" \
        --seed-start "${seed}" \
        --seed-end "${seed}" \
        --eval-repeat "$((i - 1))" \
        --inference-seed "$((20260812 + i - 1))"
    done
  ) > "${PROBE_ROOT}/logs/dump_strata_run${i}.log" 2>&1 &
  dump_pids+=("$!")
done
fail=0
for pid in "${dump_pids[@]}"; do
  if ! wait "${pid}"; then fail=1; fi
done
if [[ "${fail}" -ne 0 ]]; then
  echo "ERROR dump strata; see ${PROBE_ROOT}/logs/dump_strata_run*.log"
  exit 2
fi

python3 "${ROOT}/scripts/analysis/design_once_cfg_search.py" \
  --本体-root "${EVAL_ROOT}" \
  --always-cfg-root "${ALWAYS_CFG_ROOT:-${ROOT}/evaluate_results/dexjoco/water_plant_dewo_v7_step1000_cfg1.05_4x50_20260825_150247}" \
  --probe-root "${PROBE_ROOT}" \
  --out-dir "${DESIGN_DIR}" \
  --min-replan-grid 0,1,2,3 \
  | tee "${PROBE_ROOT}/logs/search_design_rerank.log"

echo "OK design -> ${DESIGN_DIR}/search_design.json"
