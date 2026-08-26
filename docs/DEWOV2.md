# DEWOv2: recoverability-pair pipeline

This document describes the current DEWOv2 implementation. The active recipe
uses recoverability pairs on the open-source DexJoCo stack (224×224, z-score).
The older width-jump / pm1p5-minb1 pipeline is obsolete.

## Launchers (change env, not scripts)

Session knobs (`GPUS`, `RUN_DIR`, `CKPT`, `WAIT_IDLE`, `OUT_ROOT`) are
environment variables. Task identity lives in `scripts/dewo_v2/tasks.py`.

```bash
# Collect S0 rollouts (seeds 10086..10135 × 4)
TASK=fold_glasses GPUS=4,5,6,7 bash scripts/dewo_v2/collect_opensource_4x50.sh

# Scan + pair LeRobot + Eve/text/FAST
TASK=fold_glasses SOURCE_ROOT=/path/to/s0_collect GPUS=4,5,6,7 \
  bash scripts/dewo_v2/run_pair_pipeline.sh

# Train (after sourcing the prepare env)
source /path/to/experiment/eve_v02/protocol/offline_v1_b1_jump_fast.env
TASK=fold_glasses INIT=s0 GPUS=4,5,6,7 bash scripts/dewo_v2/train.sh

# Official CFG 4×50
TASK=fold_glasses RUN_DIR=/path/to/run CKPT=/path/to/step_xxxxxx.pt \
  TEXT_EMBEDDING_CACHE_DIR=/path/to/text_embeds_cache GPUS=4,5,6,7 \
  bash scripts/dewo_v2/eval_cfg_official_4x50.sh
```

Per-task paths under `scripts/fold_glasses/` and `scripts/hammer_nail/` are
compatibility wrappers that only set `TASK`.

Release-ckpt eval (not DEWO v2) uses:

```bash
TASK=fold_glasses GPUS=4,5,6,7 bash scripts/dexjoco/eval_opensource_4x50.sh
```

## Pipeline

```text
S0 rollout_raw_200
  -> recoverability intervention scan
  -> paired success/failure LeRobot dataset
  -> Eve train/validation manifests
  -> base, outcome and FAST text caches
  -> full DiT training (INIT=scratch or INIT=s0; online VAE by default)
  -> success-vs-base CFG inference (4 x 50 seeds)
```

VAE latent pre-encode is **off by default** (encode cost was high, train-step
speedup was small once DiT dominated). Opt back in with
`USE_VAE_LATENT_CACHE=1` on the train launcher.

The training mixture is:

- primary: expert-success episodes and recoverable pair-success events, with
  action loss enabled;
- auxiliary success: the same pair-success events, with action loss disabled;
- auxiliary failure: paired failure events, with action loss disabled.

Original S0 success rollouts are deliberately excluded.

### Text embeds vs FAST

- **Expert / primary never uses FAST** (`CFG_PRIMARY_FAST` must be `0`). Primary
  mixes outcome + base only, so `precompute_text_embeds.py` (base + success /
  failure suffixes) **is still required for expert**.
- **FAST text embeds** are only for aux channels when `CFG_AUX_*_FAST > 0`.
  `precompute_fast_cfg_text_embeds.py` already skips `eve_batch_role=primary`
  windows. If all `CFG_*_FAST` are `0`, or `SKIP_FAST_TEXT_PRECOMPUTE=1`,
  prepare skips the FAST precompute entirely.

## Prerequisites

- Conda environment `fastwam` (override with `FITWAM_ENV`).
- Open inference repo: `../FastWAM-infer-in-DexJoco` (override `OPEN_REPO`).
- Released task checkpoint/config/statistics in that repository (or
  `checkpoints/dexjoco/<task>_fastwam`).
- `GPUS` must be set; there is no hidden default card list.

Generated datasets, caches, checkpoints and evaluation results are not
versioned in Git.

Useful overrides: `ROLLOUT_RAW`, `SCAN_ROOT`, `PAIR_DATASET`, `EXP_ROOT`,
`SKIP_SCAN=1`, `CFG_SCALE`, `WAIT_IDLE`, `OUT_ROOT`.

Training uses the Hydra task from `INIT` in `scripts/dewo_v2/train.sh`:
`dexjoco_dewo_v2_offline_b1_jump_fast_full_1e-4` (scratch) or
`dexjoco_dewo_v2_offline_b1_jump_fast_full_s0` (continue from S0). Full DiT
only; there is no LoRA recipe. Set `DEWO_HYDRA_OVERRIDES` for extra Hydra
knobs. Without `RUN_INLINE=1`, the launcher creates a tmux session. Val is
off (`eval_every=0`); VAE pre-encode does not encode the val split unless
`VAE_ENCODE_VAL=true`.

CFG scale selection: `scripts/dewo_v2/eval_cfg_ablation.sh`. Choose the scale
once on validation seeds, then report the final checkpoint on held-out test
seeds.

## Key files

- `scripts/dewo_v2/tasks.py`: task registry + CFG recipe
- `scripts/dewo_v2/collect_opensource_4x50.sh`
- `scripts/dewo_v2/run_pair_pipeline.sh`
- `scripts/dewo_v2/prepare_pair_eve.sh`
- `scripts/dewo_v2/train.sh`
- `scripts/dewo_v2/eval_cfg_official_4x50.sh`
- `scripts/fold_glasses/run_recoverability_pair_scan.py`
- `scripts/fold_glasses/materialize_recoverability_pairs_lerobot.py`
- `DEWO.md`: motivation, design history, and experiment-level notes
