#!/usr/bin/env bash
# Run a command in a host tmux session (outside Cursor sandbox).
# Usage:
#   bash host_tmux_run.sh --session NAME --cmd 'bash /path/to/launcher.sh'
#   bash host_tmux_run.sh --session NAME --cmd '...' --sock /tmp/fastwam_dewo_v2_pm1p5.sock
set -euo pipefail

SOCK="${FASTWAM_TMUX_SOCK:-/tmp/fastwam_dewo_v2_pm1p5.sock}"
SESSION=""
CMD=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --sock) SOCK="$2"; shift 2 ;;
    --session) SESSION="$2"; shift 2 ;;
    --cmd) CMD="$2"; shift 2 ;;
    -h|--help)
      sed -n '2,6p' "$0"
      exit 0
      ;;
    *)
      echo "unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "${SESSION}" || -z "${CMD}" ]]; then
  echo "ERROR: require --session and --cmd" >&2
  exit 2
fi

if ! tmux -S "${SOCK}" list-sessions >/dev/null 2>&1; then
  echo "[host-tmux] creating keepalive on ${SOCK}"
  tmux -S "${SOCK}" new-session -d -s _keepalive "sleep infinity"
fi

if tmux -S "${SOCK}" has-session -t "${SESSION}" 2>/dev/null; then
  echo "[host-tmux] killing existing session ${SESSION}"
  tmux -S "${SOCK}" kill-session -t "${SESSION}"
fi

# Wrap so a failing cmd still leaves a readable pane briefly.
WRAP="set +e; ${CMD}; ec=\$?; echo \"[host-tmux] DONE_EXIT=\${ec}\"; exit \${ec}"
tmux -S "${SOCK}" new-session -d -s "${SESSION}" "bash -lc $(printf '%q' "${WRAP}")"
echo "[host-tmux] started session=${SESSION} sock=${SOCK}"
tmux -S "${SOCK}" list-sessions
