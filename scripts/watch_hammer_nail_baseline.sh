#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/data_all/zhaoyc/Summer2/FastWAM_zhaoyc_failure_moved_from_share_20260703}"
SEARCH_ROOT="${SEARCH_ROOT:-/data_all/share}"
OUT_DIR="${OUT_DIR:-${ROOT}/artifacts/baseline_watch}"
TASK_PATTERN="${TASK_PATTERN:-*hammer*nail*}"
MAXDEPTH="${MAXDEPTH:-8}"
INNER_MAXDEPTH="${INNER_MAXDEPTH:-6}"
MIN_BYTES="${MIN_BYTES:-1000000000}"
POLL_SECONDS="${POLL_SECONDS:-300}"
EXIT_ON_FOUND="${EXIT_ON_FOUND:-1}"

mkdir -p "${OUT_DIR}" "${ROOT}/artifacts/logs"
LOG="${LOG:-${ROOT}/artifacts/logs/watch_hammer_nail_baseline.log}"
CANDIDATES="${OUT_DIR}/hammer_nail_candidates.tsv"
LATEST="${OUT_DIR}/hammer_nail_latest_checkpoint.txt"

scan_once() {
  local tmp_dirs tmp_files
  tmp_dirs="$(mktemp)"
  tmp_files="$(mktemp)"
  trap 'rm -f "${tmp_dirs}" "${tmp_files}"' RETURN

  timeout 120 find "${SEARCH_ROOT}" -maxdepth "${MAXDEPTH}" -type d -iname "${TASK_PATTERN}" 2>/dev/null \
    | grep -Ev '/\\.git($|/)|/wandb($|/)|/state($|/)|/states($|/)' \
    | sort -u > "${tmp_dirs}" || true

  while IFS= read -r dir; do
    timeout 60 find "${dir}" -maxdepth "${INNER_MAXDEPTH}" -type f \
      \( -name "step_*.pt" -o -name "*.pt" \) 2>/dev/null \
      | grep -Ev '/state(s)?/|optimizer|zero|mp_rank|latest' >> "${tmp_files}" || true
  done < "${tmp_dirs}"

  timeout 120 find "${SEARCH_ROOT}" -maxdepth "${MAXDEPTH}" -type f \
    \( -name "step_*.pt" -o -name "*.pt" \) 2>/dev/null \
    | grep -Ei 'hammer[_-]?nail|hammer.nail' \
    | grep -Ev '/state(s)?/|optimizer|zero|mp_rank|latest' >> "${tmp_files}" || true

  sort -u "${tmp_files}" | while IFS= read -r path; do
    [[ -f "${path}" ]] || continue
    size="$(stat -c '%s' "${path}" 2>/dev/null || echo 0)"
    if [[ "${size}" =~ ^[0-9]+$ ]] && (( size >= MIN_BYTES )); then
      mtime="$(stat -c '%Y' "${path}" 2>/dev/null || echo 0)"
      printf '%s\t%s\t%s\n' "${mtime}" "${size}" "${path}"
    fi
  done | sort -nr
}

echo "===== $(date) watch_hammer_nail_baseline start =====" | tee -a "${LOG}"
echo "search_root=${SEARCH_ROOT} maxdepth=${MAXDEPTH} min_bytes=${MIN_BYTES}" | tee -a "${LOG}"

while true; do
  echo "===== $(date) scan =====" | tee -a "${LOG}"
  scan_once > "${CANDIDATES}"
  count="$(wc -l < "${CANDIDATES}" | tr -d ' ')"
  echo "candidates=${count}" | tee -a "${LOG}"
  if (( count > 0 )); then
    latest_path="$(head -1 "${CANDIDATES}" | cut -f3-)"
    printf '%s\n' "${latest_path}" > "${LATEST}"
    echo "latest=${latest_path}" | tee -a "${LOG}"
    head -20 "${CANDIDATES}" | tee -a "${LOG}"
    if [[ "${EXIT_ON_FOUND}" == "1" ]]; then
      exit 0
    fi
  fi
  sleep "${POLL_SECONDS}"
done
