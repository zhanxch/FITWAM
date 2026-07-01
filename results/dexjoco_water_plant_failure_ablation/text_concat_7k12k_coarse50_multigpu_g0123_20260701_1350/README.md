# Text-Failure Concat 7k-12k Coarse Rollout

This directory contains the 50-episode multi-GPU DexJoCo water_plant rollout sweep for the text-concat failure-aware FastWAM run.

## Setup

- Source run: `/data_all/share/dexjoco_fastwam_results/dexjoco_water_plant_text_failure_2cam_proprio_1e-4/2026-06-29_00-21-57`
- Checkpoints: `step_007000.pt` to `step_012000.pt`, interval 1000
- Episodes: 50 per checkpoint, seed 0
- GPUs: `0,1,2,3`
- Control: blocking, `replan_steps=25`, `max_env_steps=600`
- Randomization: disabled
- Saved media/actions: disabled

The 6500-step checkpoint is not included here because it had already been evaluated separately with a 100-episode budget.

## Results

| Checkpoint step | Successes | Episodes | Success rate |
|---:|---:|---:|---:|
| 7000 | 25 | 50 | 50% |
| 8000 | 35 | 50 | 70% |
| 9000 | 35 | 50 | 70% |
| 10000 | 37 | 50 | 74% |
| 11000 | 39 | 50 | 78% |
| 12000 | 40 | 50 | 80% |

The coarse trend rises from 50% at 7000 steps to 80% at 12000 steps. The best checkpoint in this sweep is 12000 under the 50-episode budget.

## Run Notes

- `step=10000` attempt 1 hit a GPU OOM on one shard and produced a partial 25/37 aggregation. That attempt is discarded. The valid result is attempt 2: 37/50.
- `step=12000` attempt 1 hit a GPU OOM on one shard. That invalid attempt was terminated so the wrapper could retry. The valid result is attempt 2: 40/50.
- These failures are infrastructure/runtime caveats, not model outcome measurements.

## Artifacts

- Local summary: `coarse50_summary.csv`, `coarse50_summary.json`
- Per-checkpoint summaries: `concat_step_*/summary.json`
- Remote run log copy: `remote_run.log`
- Curve: `text_concat_7k12k_success_curve.svg`

Remote source directory:

`/data_all/share/FastWAM_zhaoyc_failure/artifacts/evals/text_concat_7k12k_coarse50_multigpu_g0123_20260701_1350`
