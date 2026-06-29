#!/usr/bin/env bash
set -euo pipefail

cd /data_all/xiangchengzhan/FastWAM
export PYTHONPATH="${PWD}/src:${PWD}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"

CKPT="runs/spray_water_rot6d_gr00tstyle_uncond_3cam_384_1e-4/2026-06-26_19-16-40/checkpoints/weights/step_005000.pt"
OUT_BASE="./evaluate_results/openloop_episode_gr00tstyle/filtered_out"

COMMON=(
  --config-name=openloop_episode
  task=spray_water_rot6d_gr00tstyle_uncond_3cam_384_1e-4
  data=spray_water_rot6d_gr00tstyle_filtered_out
  "ckpt=${CKPT}"
  OPENLOOP.split=train
  OPENLOOP.episode_indices=[0]
  # 32 matches the trained action/video rollout window. `max_samples=null`
  # walks the entire episode instead of the config default 8 chunks.
  OPENLOOP.frame_stride=32
  OPENLOOP.max_samples=null
  OPENLOOP.predict_video=true
  OPENLOOP.save_episode_video=true
  OPENLOOP.save_episode_action_dim_plots=true
  OPENLOOP.video_action_panel=none
)

run_one() {
  local mode="$1"
  local out_dir="$2"
  local log="$3"
  echo "=== Running rollout_mode=${mode} -> ${out_dir} ==="
  conda run -n fastwam --no-capture-output python scripts/openloop/run_openloop.py \
    "${COMMON[@]}" \
    "OPENLOOP.rollout_mode=${mode}" \
    "OPENLOOP.output_dir=${out_dir}" \
    2>&1 | tee "${log}"
}

run_one gt_window "${OUT_BASE}/gt_window" run_gr00tstyle_openloop_gt.log
run_one autoregressive "${OUT_BASE}/autoregressive" run_gr00tstyle_openloop_ar.log

echo "=== Done ==="
