#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"
export PYTHONPATH="${PWD}/src:${PWD}:${PYTHONPATH:-}"
LOG_DIR="${ROOT_DIR}/logs/openloop"
mkdir -p "${LOG_DIR}"
PYTHON="/home/xiangchengzhan/anaconda3/envs/fastwam/bin/python"
unset CUDA_VISIBLE_DEVICES
# GPU 1 typically has the most headroom beside occupied training jobs.
export CUDA_VISIBLE_DEVICES=1
echo "Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

RUN_ID="2026-06-26_19-16-40"
RUN_DIR="runs/spray_water_rot6d_gr00tstyle_uncond_3cam_384_1e-4/${RUN_ID}"
OUT_BASE="./evaluate_results/openloop_episode_gr00tstyle/filtered_out"

STEPS=(015000 020000 025000 030000)
MODES=(gt_window autoregressive)

COMMON=(
  --config-name=openloop_episode
  task=spray_water_rot6d_gr00tstyle_uncond_3cam_384_1e-4
  data=spray_water_rot6d_gr00tstyle_filtered_out
  OPENLOOP.split=train
  OPENLOOP.episode_indices=[0]
  OPENLOOP.frame_stride=32
  OPENLOOP.max_samples=null
  OPENLOOP.predict_video=true
  OPENLOOP.save_episode_video=true
  OPENLOOP.save_episode_action_dim_plots=true
  OPENLOOP.video_action_panel=none
)

for step in "${STEPS[@]}"; do
  ckpt="${RUN_DIR}/checkpoints/weights/step_${step}.pt"
  if [[ ! -f "${ckpt}" ]]; then
    echo "SKIP missing checkpoint: ${ckpt}"
    continue
  fi

  for mode in "${MODES[@]}"; do
    out_dir="${OUT_BASE}/step_${step}/${mode}"
    log="${LOG_DIR}/run_gr00tstyle_openloop_step${step}_${mode}.log"
    if compgen -G "${out_dir}/*/summary.json" > /dev/null; then
      echo "SKIP existing: step=${step} mode=${mode}"
      continue
    fi

    echo "=== step=${step} mode=${mode} -> ${out_dir} ==="
    "${PYTHON}" scripts/openloop/run_openloop.py \
      "${COMMON[@]}" \
      "ckpt=${ckpt}" \
      "OPENLOOP.rollout_mode=${mode}" \
      "OPENLOOP.output_dir=${out_dir}" \
      2>&1 | tee "${log}"
  done
done

echo "=== All checkpoints done ==="
