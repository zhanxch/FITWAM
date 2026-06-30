#!/usr/bin/env bash
# Run DexJoCo async/LPF eval clients against already-running FastWAM policy servers.

set -euo pipefail

FASTWAM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${FASTWAM_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DEXJOCO_PY_ROOT="${DEXJOCO_PY_ROOT:-${FASTWAM_ROOT}/third_party/dexjoco/dexjoco}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-${FASTWAM_ROOT}/evaluate_results/dexjoco_async_ablation/${RUN_ID}/evals}"
RUN_DIR="${RUN_DIR:?set RUN_DIR to a FastWAM DexJoCo training run directory}"
TASK_CONFIG_DIR="${TASK_CONFIG_DIR:-${FASTWAM_ROOT}/third_party/dexjoco/configs/rand_obj}"
TASKS="${TASKS:-bimanual_microwave_cook}"
EPISODES="${EPISODES:-5}"
SEED="${SEED:-0}"
MAX_ENV_STEPS="${MAX_ENV_STEPS:-1500}"
SAVE_VIDEO="${SAVE_VIDEO:-1}"
SAVE_ACTIONS="${SAVE_ACTIONS:-1}"
ASYNC_FALLBACK="${ASYNC_FALLBACK:-wait}"
POLICY_HOST="${POLICY_HOST:-127.0.0.1}"
WAIT_SECONDS="${WAIT_SECONDS:-1200}"

# Format: label,control_mode,replan_steps,low_pass_alpha,policy_port
CONDITIONS_STR="${CONDITIONS:-sync_stride24,blocking,24,none,5570 overlap_stride24_lpf05,overlap,24,0.5,5571 overlap_stride16_lpf05,overlap,16,0.5,5572 overlap_stride8_lpf07,overlap,8,0.7,5573}"

mkdir -p "${OUT_ROOT}"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYTHONPATH="${FASTWAM_ROOT}/src:${FASTWAM_ROOT}/scripts:${DEXJOCO_PY_ROOT}:${PYTHONPATH:-}"

echo "[eval-clients] FASTWAM_ROOT=${FASTWAM_ROOT}"
echo "[eval-clients] PYTHON_BIN=${PYTHON_BIN}"
echo "[eval-clients] RUN_ID=${RUN_ID}"
echo "[eval-clients] OUT_ROOT=${OUT_ROOT}"
echo "[eval-clients] RUN_DIR=${RUN_DIR}"
echo "[eval-clients] TASK_CONFIG_DIR=${TASK_CONFIG_DIR}"
echo "[eval-clients] TASKS=${TASKS} EPISODES=${EPISODES} SEED=${SEED}"
echo "[eval-clients] CONDITIONS=${CONDITIONS_STR}"

printf 'run_id,out_root,run_dir,tasks,episodes,seed,conditions\n' > "${OUT_ROOT}/run_manifest.csv"
printf '%s,%s,%s,%s,%s,%s,"%s"\n' \
  "${RUN_ID}" "${OUT_ROOT}" "${RUN_DIR}" "${TASKS}" "${EPISODES}" "${SEED}" "${CONDITIONS_STR}" \
  >> "${OUT_ROOT}/run_manifest.csv"

read -r -a CONDITIONS <<< "${CONDITIONS_STR}"

wait_for_policy_server() {
  local port="$1"
  local deadline=$((SECONDS + WAIT_SECONDS))
  while (( SECONDS < deadline )); do
    if "${PYTHON_BIN}" - <<PY >/dev/null 2>&1; then
import sys
sys.path.insert(0, "${FASTWAM_ROOT}/scripts")
from policy_client_async import PolicyClientAsync
c = PolicyClientAsync(host="${POLICY_HOST}", port=${port}, timeout_ms=5000, identity="probe-${port}")
try:
    ok = c.ping()
finally:
    c.close()
raise SystemExit(0 if ok else 1)
PY
      return 0
    fi
    sleep 5
  done
  echo "[eval-clients] server ${POLICY_HOST}:${port} did not become ready in ${WAIT_SECONDS}s"
  return 1
}

eval_pids=()
status=0

for spec in "${CONDITIONS[@]}"; do
  IFS=',' read -r label mode replan lpf port <<< "${spec}"
  cond_dir="${OUT_ROOT}/${label}"
  mkdir -p "${cond_dir}"
  eval_log="${cond_dir}/eval_port${port}.log"
  lock_dir="${cond_dir}/.eval.lock"

  if [[ -f "${cond_dir}/summary.json" ]]; then
    echo "[eval-clients] skipping label=${label}; summary already exists"
    continue
  fi

  if ! mkdir "${lock_dir}" 2>/dev/null; then
    echo "[eval-clients] label=${label} is already running; waiting for summary"
    while [[ -d "${lock_dir}" && ! -f "${cond_dir}/summary.json" ]]; do
      sleep 10
    done
    if [[ -f "${cond_dir}/summary.json" ]]; then
      echo "[eval-clients] skipping label=${label}; concurrent run completed"
      continue
    fi
    echo "[eval-clients] stale or failed lock for label=${label}: ${lock_dir}"
    status=1
    continue
  fi
  trap 'rm -rf "${lock_dir}"' EXIT

  echo "[eval-clients] waiting label=${label} port=${port}"
  if ! wait_for_policy_server "${port}"; then
    rm -rf "${lock_dir}"
    status=1
    continue
  fi

  eval_args=(
    scripts/water_plant/dexjoco_async/eval_dexjoco_fastwam_control.py
    --run-dir "${RUN_DIR}"
    --task-config-dir "${TASK_CONFIG_DIR}"
    --policy-host "${POLICY_HOST}"
    --policy-port "${port}"
    --tasks ${TASKS}
    --episodes "${EPISODES}"
    --seed "${SEED}"
    --replan-steps "${replan}"
    --max-env-steps "${MAX_ENV_STEPS}"
    --output-dir "${cond_dir}"
    --control-mode "${mode}"
    --async-fallback "${ASYNC_FALLBACK}"
  )
  if [[ "${lpf}" != "none" ]]; then
    eval_args+=(--low-pass-alpha "${lpf}")
  fi
  if [[ "${SAVE_VIDEO}" == "0" ]]; then
    eval_args+=(--no-save-video)
  fi
  if [[ "${SAVE_ACTIONS}" == "0" ]]; then
    eval_args+=(--no-save-actions)
  fi

  printf '%s,%s,%s,%s,%s,%s\n' \
    "${label}" "${mode}" "${replan}" "${lpf}" "${port}" "${cond_dir}" \
    >> "${OUT_ROOT}/conditions.csv"
  echo "[eval-clients] launching label=${label} port=${port} log=${eval_log}"
  (
    set +e
    "${PYTHON_BIN}" -u "${eval_args[@]}" >"${eval_log}" 2>&1
    rc=$?
    rm -rf "${lock_dir}"
    exit "${rc}"
  ) &
  eval_pids+=("$!")
done

for pid in "${eval_pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done

"${PYTHON_BIN}" - <<'PY' "${OUT_ROOT}"
from pathlib import Path
import csv
import json
import sys

root = Path(sys.argv[1])
rows = []
videos = []
for path in sorted(root.glob("*/summary.json")):
    payload = json.loads(path.read_text())
    for task in payload.get("tasks", []):
        means = task.get("metric_means", {})
        rows.append({
            "label": payload.get("label"),
            "control_mode": payload.get("control_mode"),
            "replan_steps": payload.get("replan_steps"),
            "overlap_steps": payload.get("overlap_steps"),
            "low_pass_alpha": payload.get("low_pass_alpha"),
            "env_name": task.get("env_name"),
            "episodes": task.get("episodes"),
            "successes": task.get("successes"),
            "success_rate": task.get("success_rate"),
            "latency_mean_s": means.get("inference_latency_mean_s"),
            "latency_p95_s": means.get("inference_latency_p95_s"),
            "jerk": means.get("action_jerk_l2_mean"),
            "delta": means.get("action_delta_l2_mean"),
            "sign_flip": means.get("oscillation_sign_flip_rate"),
            "queue_underruns": means.get("queue_underruns"),
            "queue_wait_s": means.get("queue_wait_s"),
            "summary_path": str(path),
        })
        for ep in task.get("episode_results", []):
            if ep.get("video_path"):
                videos.append({
                    "label": payload.get("label"),
                    "env_name": task.get("env_name"),
                    "episode": ep.get("episode"),
                    "seed": ep.get("seed"),
                    "success": ep.get("success"),
                    "steps": ep.get("steps"),
                    "video_path": ep.get("video_path"),
                    "actions_path": ep.get("actions_path"),
                })
if rows:
    with (root / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# DexJoCo FastWAM Async/LPF Summary", ""]
    for row in rows:
        lines.append(
            f"- {row['label']}: {row['successes']}/{row['episodes']} success, "
            f"jerk={row['jerk']}, sign_flip={row['sign_flip']}, "
            f"underruns={row['queue_underruns']}, wait_s={row['queue_wait_s']}"
        )
    (root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
if videos:
    with (root / "video_manifest.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(videos[0].keys()))
        writer.writeheader()
        writer.writerows(videos)
PY

echo "[eval-clients] done status=${status}"
echo "[eval-clients] output=${OUT_ROOT}"
exit "${status}"
