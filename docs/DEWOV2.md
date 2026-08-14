# DEWOv2: fold_glasses recoverability-pair pipeline

This document describes the current DEWOv2 implementation for `fold_glasses`.
The active recipe uses recoverability pairs. The older width-jump / pm1p5-minb1
pipeline is obsolete and must not be used for the main result.

## Pipeline

```text
S0 rollout_raw_200
  -> recoverability intervention scan
  -> paired success/failure LeRobot dataset
  -> Eve train/validation manifests
  -> base, outcome and FAST text caches + VAE latent cache
  -> Video LoRA training
  -> success-vs-base CFG inference (4 x 50 seeds)
```

The training mixture is:

- primary: expert-success episodes and recoverable pair-success events, with
  action loss enabled;
- auxiliary success: the same pair-success events, with action loss disabled;
- auxiliary failure: paired failure events, with action loss disabled.

Original S0 success rollouts are deliberately excluded. The implementation uses
the open-source DexJoCo-compatible stack: two cameras at 224 x 224 and z-score
normalization from the released `dataset_stats.json`.

## Prerequisites

- Conda environment `fastwam` (override with `FITWAM_ENV`).
- The open inference repository at
  `/data_all/xiangchengzhan/FastWAM-infer-in-DexJoco` (override `OPEN_REPO`).
- Released `fold_glasses` checkpoint/config/statistics in that repository.
- An S0 collection containing `rollout_raw_200`.
- Four GPUs by default; override `GPUS` with a comma-separated list.

Generated datasets, caches, checkpoints and evaluation results are intentionally
not versioned in Git.

## Data preparation and training

The end-to-end entry point scans the S0 rollouts, materializes the paired
LeRobot dataset, builds Eve manifests and caches, pre-encodes VAE latents, and
starts training inline:

```bash
SOURCE_ROOT=/path/to/fold_glasses_s0_collection \
GPUS=0,1,2,3 \
bash scripts/fold_glasses/run_dewo_v2_pair_pipeline.sh
```

Useful overrides are `ROLLOUT_RAW`, `SCAN_ROOT`, `PAIR_DATASET`, `EXP_ROOT`,
and `SKIP_SCAN=1`. Re-running the command reuses an existing scan summary and
pair dataset when present.

The preparation stage writes:

```text
${EXP_ROOT}/
  eve_v02/manifests/offline_b1_jump_fast_pair.json
  eve_v02/manifests/offline_selection_primary_success.json
  eve_v02/protocol/offline_v1_b1_jump_fast.env
  eve_v02/protocol/offline_v1_b1_jump_fast.json
  text_embeds_cache/
  vae_latent_cache/
  logs/
```

To run preparation and training separately:

```bash
PAIR_DATASET=/path/to/pair_lerobot EXP_ROOT=/path/to/experiment \
  bash scripts/fold_glasses/prepare_dewo_v2_pair_eve.sh

source /path/to/experiment/eve_v02/protocol/offline_v1_b1_jump_fast.env
GPUS=0,1,2,3 RUN_INLINE=1 \
  bash scripts/fold_glasses/train_dewo_v2_jump_fast_lora.sh
```

Training uses
`configs/task/dexjoco/dexjoco_fold_glasses_offline_b1_jump_fast_lora_3e-5.yaml`.
Its defaults are Video LoRA rank 32, learning rate `3e-5`, 15,000 steps, and
checkpoints every 2,500 steps. Set `DEWO_HYDRA_OVERRIDES` for explicit Hydra
overrides. Without `RUN_INLINE=1`, the launcher creates a tmux session.

## Inference and evaluation

Run the official success-vs-base CFG evaluation after selecting a checkpoint:

```bash
RUN_DIR=/path/to/training/run \
CKPT=/path/to/step_xxxxxx.pt \
TEXT_EMBEDDING_CACHE_DIR=/path/to/text_embeds_cache \
GPUS=0,1,2,3 \
bash scripts/fold_glasses/eval_dewo_v2_cfg_official_4x50.sh
```

The default CFG scale is 2.0. Override `CFG_SCALE`, `SEEDS_PER_GPU`, `OUT_ROOT`,
or `CFG_TASK_DIR` as needed. The launcher runs four disjoint 50-seed shards,
merges their summaries, and records the checkpoint and evaluation protocol in
the output directory.

For validation-only scale selection, use:

```bash
RUN_DIR=/path/to/training/run \
TEXT_EMBEDDING_CACHE_DIR=/path/to/text_embeds_cache \
bash scripts/fold_glasses/eval_dewo_v2_cfg_ablation.sh
```

Choose the CFG scale once on validation seeds, then report the final checkpoint
on fresh test seeds that were not used for scale or checkpoint selection.

## Key files

- `scripts/fold_glasses/run_dewo_v2_pair_pipeline.sh`: end-to-end entry point.
- `scripts/fold_glasses/run_recoverability_pair_scan.py`: intervention scan.
- `scripts/fold_glasses/materialize_recoverability_pairs_lerobot.py`: paired
  LeRobot materialization.
- `scripts/fold_glasses/prepare_dewo_v2_pair_eve.sh`: manifests and caches.
- `scripts/fold_glasses/train_dewo_v2_jump_fast_lora.sh`: training launcher.
- `scripts/fold_glasses/eval_dewo_v2_cfg_official_4x50.sh`: official inference.
- `DEWO.md`: motivation, design history, and experiment-level notes.

