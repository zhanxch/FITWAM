#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=${ROOT:-/data_all/zhaoyc/Summer2/FastWAM_zhaoyc_failure_moved_from_share_20260703}
RESULT_ROOT=${RESULT_ROOT:-/data_all/zhaoyc/Summer2/dexjoco_fastwam_results_moved_from_share_20260703}
PY=${PY:-/home/gzr1/miniconda3/envs/residual/bin/python}
WANDB_NAME=${WANDB_NAME:-dexjoco_water_plant_structured_failure_2cam_proprio_1e-4}
RUN_ID=${RUN_ID:-2026-06-30_02-41-07}
RUN_DIR=${RUN_DIR:-$RESULT_ROOT/$WANDB_NAME/$RUN_ID}
OUT_ROOT=${OUT_ROOT:-$ROOT/artifacts/evals/structure_up_early_ckpt_sweep_$(date '+%Y%m%d_%H%M%S')}
LOG=${LOG:-$ROOT/artifacts/logs/structure_up_early_ckpt_sweep.log}

GPUS_CSV=${GPUS_CSV:-0,1,2,3}
EPISODES=${EPISODES:-25}
SEED=${SEED:-0}
TASKS=${TASKS:-water_plant}
BATCH_SIZE=${BATCH_SIZE:-5}
INFER_BATCH_SIZE=${INFER_BATCH_SIZE:-$BATCH_SIZE}
MAX_ENV_STEPS=${MAX_ENV_STEPS:-600}
REPLAN_STEPS=${REPLAN_STEPS:-25}
SERVER_WORKERS=${SERVER_WORKERS:-8}
POLICY_TIMEOUT_MS=${POLICY_TIMEOUT_MS:-300000}
SERVER_WAIT_S=${SERVER_WAIT_S:-900}
CHECKPOINT_STEPS=${CHECKPOINT_STEPS:-500 1000 1500 2000 2500 3000 3500 4000 4500 5000 5500 6000 6500}

mkdir -p "$OUT_ROOT" "$(dirname "$LOG")"

export PATH=/home/zhaoyc/.local/bin:$PATH
export PYTHONPATH="$ROOT/src:$ROOT/scripts:$ROOT/third_party/dexjoco/dexjoco:$ROOT/third_party/dexjoco:${PYTHONPATH:-}"
export DIFFSYNTH_MODEL_BASE_PATH="$ROOT/checkpoints"
export MUJOCO_GL=${MUJOCO_GL:-egl}
export PYTHONUNBUFFERED=1

IFS=',' read -r -a GPUS <<< "$GPUS_CSV"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "$LOG"
}

pick_ports() {
  "$PY" - "${#GPUS[@]}" <<'PY'
import socket
import sys

n = int(sys.argv[1])
socks = []
ports = []
for _ in range(n):
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    ports.append(s.getsockname()[1])
    socks.append(s)
print(" ".join(str(p) for p in ports))
for s in socks:
    s.close()
PY
}

wait_server_ready() {
  local pid="$1"
  local port="$2"
  local server_log="$3"
  local deadline=$((SECONDS + SERVER_WAIT_S))
  while (( SECONDS < deadline )); do
    if ! kill -0 "$pid" 2>/dev/null; then
      log "ERROR: server pid=$pid exited before ready port=$port"
      tail -n 220 "$server_log" | tee -a "$LOG" || true
      return 1
    fi
    if "$PY" - "$port" <<'PY' >/dev/null 2>&1
import sys
from policy_zmq_client_async import PolicyClientAsync

client = PolicyClientAsync(host="127.0.0.1", port=int(sys.argv[1]), timeout_ms=10000)
try:
    ok = client.ping()
finally:
    client.close()
raise SystemExit(0 if ok else 1)
PY
    then
      return 0
    fi
    sleep 2
  done
  log "ERROR: server not ready after ${SERVER_WAIT_S}s port=$port pid=$pid"
  tail -n 220 "$server_log" | tee -a "$LOG" || true
  return 1
}

merge_checkpoint_summary() {
  "$PY" - "$1" "$2" "$3" "$4" "$5" "$6" <<'PY'
import json
import sys
from pathlib import Path

ckpt_dir = Path(sys.argv[1])
run_dir = sys.argv[2]
ckpt = sys.argv[3]
step = int(sys.argv[4])
episodes = int(sys.argv[5])
seed = int(sys.argv[6])

shards = []
episode_results = []
total_episodes = 0
total_successes = 0
task_name = None
prompt = None
metric_weighted = {}
metric_weights = {}

for shard_dir in sorted(ckpt_dir.glob("shard_*")):
    summary_candidates = sorted(
        shard_dir.rglob("summary.json"),
        key=lambda p: p.stat().st_mtime,
    )
    if not summary_candidates:
        continue
    summary_path = summary_candidates[-1]
    data = json.loads(summary_path.read_text())
    eps = int(data.get("total_episodes", 0))
    succ = int(data.get("total_successes", 0))
    total_episodes += eps
    total_successes += succ
    task = (data.get("tasks") or [{}])[0]
    task_name = task_name or task.get("env_name")
    prompt = prompt or task.get("prompt")
    for result in task.get("episode_results", []):
        episode_results.append(result)
    for key, value in (task.get("metric_means") or {}).items():
        if isinstance(value, (int, float)):
            metric_weighted[key] = metric_weighted.get(key, 0.0) + float(value) * eps
            metric_weights[key] = metric_weights.get(key, 0) + eps
    shards.append({
        "shard_id": int(shard_dir.name.split("_")[-1]),
        "summary_path": str(summary_path),
        "episodes": eps,
        "successes": succ,
        "success_rate": (succ / eps) if eps else None,
        "policy_port": data.get("policy_port"),
        "base_seed": data.get("seed"),
    })

metric_means = {
    key: metric_weighted[key] / metric_weights[key]
    for key in sorted(metric_weighted)
    if metric_weights.get(key)
}
summary = {
    "label": "structure_up_early_ckpt_sweep",
    "run_dir": run_dir,
    "checkpoint": ckpt,
    "step": step,
    "control_mode": "blocking",
    "replan_steps": 25,
    "action_horizon": 32,
    "max_env_steps": 600,
    "episodes_per_task": episodes,
    "num_tasks": 1,
    "total_episodes": total_episodes,
    "total_successes": total_successes,
    "overall_success_rate": (total_successes / total_episodes) if total_episodes else None,
    "seed": seed,
    "save_video": False,
    "save_actions": False,
    "num_shards": len(shards),
    "shards": shards,
    "tasks": [{
        "env_name": task_name or "water_plant",
        "prompt": prompt,
        "episodes": total_episodes,
        "successes": total_successes,
        "success_rate": (total_successes / total_episodes) if total_episodes else None,
        "metric_means": metric_means,
        "episode_results": sorted(episode_results, key=lambda x: x.get("seed", 0)),
    }],
}
if total_episodes <= 0:
    raise SystemExit(f"no shard summaries with episodes found under {ckpt_dir}")
out = ckpt_dir / "summary.json"
out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(out)
PY
}

append_summary_row() {
  "$PY" - "$OUT_ROOT" <<'PY'
import csv
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
rows = []
for summary_path in sorted(root.glob("step_*/summary.json")):
    data = json.loads(summary_path.read_text())
    rows.append({
        "step": data.get("step"),
        "checkpoint": data.get("checkpoint"),
        "episodes": data.get("total_episodes"),
        "successes": data.get("total_successes"),
        "success_rate": data.get("overall_success_rate"),
        "summary_path": str(summary_path),
    })
rows.sort(key=lambda r: int(r["step"]))
csv_path = root / "sweep_summary.csv"
with csv_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["step", "checkpoint", "episodes", "successes", "success_rate", "summary_path"])
    writer.writeheader()
    writer.writerows(rows)
(root / "sweep_summary.json").write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")
print(csv_path)
PY
}

cleanup_pids=()
cleanup() {
  for pid in "${cleanup_pids[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT

log "starting Structure Up early checkpoint sweep"
log "run_dir=$RUN_DIR"
log "out_root=$OUT_ROOT"
log "gpus=${GPUS_CSV} episodes=$EPISODES seed=$SEED max_env_steps=$MAX_ENV_STEPS checkpoint_steps=[$CHECKPOINT_STEPS]"

for step in $CHECKPOINT_STEPS; do
  step_pad=$(printf "%06d" "$step")
  ckpt="$RUN_DIR/checkpoints/weights/step_${step_pad}.pt"
  ckpt_dir="$OUT_ROOT/step_${step_pad}"
  mkdir -p "$ckpt_dir"
  if [[ ! -s "$ckpt" ]]; then
    log "missing checkpoint step=$step ckpt=$ckpt; skipping"
    continue
  fi
  if [[ -f "$ckpt_dir/summary.json" ]]; then
    log "summary exists for step=$step; skipping"
    append_summary_row >/dev/null
    continue
  fi

  log "evaluating step=$step ckpt=$ckpt"
  read -r -a PORTS <<< "$(pick_ports)"
  server_pids=()
  client_pids=()
  shard_count=${#GPUS[@]}
  base=$((EPISODES / shard_count))
  rem=$((EPISODES % shard_count))

  for i in "${!GPUS[@]}"; do
    gpu="${GPUS[$i]}"
    port="${PORTS[$i]}"
    shard_dir="$ckpt_dir/shard_$i"
    mkdir -p "$shard_dir"
    server_log="$shard_dir/server.log"
    log "  shard=$i gpu=$gpu port=$port starting server"
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$ROOT/scripts/run_fastwam_server_async.py" \
      --run-dir "$RUN_DIR" \
      --checkpoint "$ckpt" \
      --device cuda:0 \
      --host 127.0.0.1 \
      --port "$port" \
      --no-load-text-encoder \
      --num-workers "$SERVER_WORKERS" > "$server_log" 2>&1 &
    server_pids+=("$!")
    cleanup_pids+=("$!")
  done

  for i in "${!server_pids[@]}"; do
    wait_server_ready "${server_pids[$i]}" "${PORTS[$i]}" "$ckpt_dir/shard_$i/server.log"
    log "  shard=$i server ready"
  done

  for i in "${!GPUS[@]}"; do
    eps="$base"
    if (( i < rem )); then
      eps=$((eps + 1))
    fi
    shard_seed=$((SEED + i * base + (i < rem ? i : rem)))
    shard_dir="$ckpt_dir/shard_$i"
    client_log="$shard_dir/client.log"
    port="${PORTS[$i]}"
    log "  shard=$i starting client eps=$eps seed=$shard_seed"
    "$PY" "$ROOT/scripts/eval_dexjoco_fastwam_async.py" \
      --run-dir "$RUN_DIR" \
      --policy-host 127.0.0.1 \
      --policy-port "$port" \
      --policy-timeout-ms "$POLICY_TIMEOUT_MS" \
      --task-config-dir "$ROOT/third_party/dexjoco/configs/rand_obj" \
      --tasks $TASKS \
      --episodes "$eps" \
      --seed "$shard_seed" \
      --batch-size "$BATCH_SIZE" \
      --infer-batch-size "$INFER_BATCH_SIZE" \
      --use-batch-infer \
      --replan-steps "$REPLAN_STEPS" \
      --max-env-steps "$MAX_ENV_STEPS" \
      --output-dir "$shard_dir" \
      --no-save-video \
      --no-save-actions > "$client_log" 2>&1 &
    client_pids+=("$!")
  done

  failed=0
  for pid in "${client_pids[@]}"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done

  for pid in "${server_pids[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
  wait "${server_pids[@]}" 2>/dev/null || true
  cleanup_pids=()

  if (( failed != 0 )); then
    log "ERROR: one or more clients failed at step=$step"
    exit 3
  fi

  summary_path=$(merge_checkpoint_summary "$ckpt_dir" "$RUN_DIR" "$ckpt" "$step" "$EPISODES" "$SEED")
  append_summary_row >/dev/null
  rate=$("$PY" - "$summary_path" <<'PY'
import json
import sys
data = json.loads(open(sys.argv[1]).read())
rate = data.get("overall_success_rate")
rate_s = f"{rate:.3f}" if isinstance(rate, (int, float)) else "nan"
print(f"{data['total_successes']}/{data['total_episodes']} ({rate_s})")
PY
)
  log "finished step=$step result=$rate summary=$summary_path"
done

append_summary_row | tee -a "$LOG"
log "completed Structure Up early checkpoint sweep"
