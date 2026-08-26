#!/usr/bin/env bash
# Verify an eval/smoke OUT dir is on CUDA, not CPU-fallback.
# Usage: bash verify_eval_cuda.sh /path/to/OUT_ROOT
set -euo pipefail

OUT="${1:?OUT_ROOT required}"
SOCK="${FASTWAM_TMUX_SOCK:-/tmp/fastwam_dewo_v2_pm1p5.sock}"

if [[ ! -d "${OUT}" ]]; then
  echo "ERROR: missing OUT ${OUT}" >&2
  exit 2
fi

mapfile -t SERVER_LOGS < <(find "${OUT}" -type f -name 'server.log' 2>/dev/null | sort)
if [[ "${#SERVER_LOGS[@]}" -eq 0 ]]; then
  echo "WAIT: no server.log under ${OUT} yet"
  exit 3
fi

fail=0
cuda_hits=0
for f in "${SERVER_LOGS[@]}"; do
  if grep -q 'falling back to CPU\|CUDA unavailable' "${f}"; then
    echo "FAIL CPU-fallback: ${f}"
    fail=1
  fi
  if grep -qE 'Loading FastWAM model on cuda|device cuda|CUDA_VISIBLE' "${f}"; then
    cuda_hits=$((cuda_hits + 1))
    echo "OK cuda-ish: ${f}"
  else
    # still loading is OK if no CPU fallback line
    echo "INFO no explicit cuda line yet: ${f}"
    tail -n 3 "${f}" | sed 's/^/  | /'
  fi
done

if [[ "${fail}" -ne 0 ]]; then
  echo "RESULT: FAIL (CPU fallback detected) — kill and relaunch via host tmux"
  exit 1
fi

# GPU snapshot via host tmux when possible
GPU_OUT="/tmp/fw_verify_gpu_$$.csv"
if tmux -S "${SOCK}" list-sessions >/dev/null 2>&1; then
  tmux -S "${SOCK}" new-session -d -s "_gpu_verify_$$" \
    "nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv > '${GPU_OUT}' 2>&1"
  sleep 1
  if [[ -f "${GPU_OUT}" ]]; then
    echo "=== nvidia-smi (host tmux) ==="
    cat "${GPU_OUT}"
  fi
elif command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  echo "=== nvidia-smi (local) ==="
  nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv
else
  echo "WARN: could not query nvidia-smi (sandbox?); rely on server.log only"
fi

if [[ "${cuda_hits}" -eq 0 ]]; then
  echo "RESULT: PENDING (no CPU fallback, but cuda load line not seen yet)"
  exit 3
fi

echo "RESULT: OK (${cuda_hits}/${#SERVER_LOGS[@]} server logs show cuda load)"
exit 0
