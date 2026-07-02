# DexJoCo Hammer Nail Failure-Embedding Sweep

Task: `hammer_nail` (`rand_obj`, blocking control, 50 episodes, seed 0).

Gate for this task: `min(pi0.5=84.7, GR00T N1.5=67.3) - 1.0 = 66.3%`.

| Policy | Checkpoint | Success | Rate | Gate | Notes |
|--------|------------|---------|------|------|-------|
| Success-only baseline | step 006650 | 36/50 | 72.0% | pass | Uploaded baseline from `/data_all/share/dexjoco_fastwam_results/hammer_nail_uncond_2cam_384_1e-4` |
| Failure embedding | step 006000 | 32/50 | 64.0% | fail | Final checkpoint missed the gate |
| Failure embedding | step 005500 | 34/50 | 68.0% | pass | Best passing checkpoint in this sweep |
| Failure embedding | step 005000 | 27/50 | 54.0% | fail | Early checkpoint under-trained |

Remote artifacts:

- Baseline eval: `/data_all/share/FastWAM_zhaoyc_failure/artifacts/evals/hammer_nail_baseline_step_006650_seed0_50ep_retry6_2026-07-01_19-05-07`
- Failure embedding run: `/data_all/share/dexjoco_fastwam_results/dexjoco_hammer_nail_failure_embedding_2cam_proprio_1e-4/2026-07-01_20-55-26`
- Step 006000 eval: `/data_all/share/FastWAM_zhaoyc_failure/artifacts/evals/hammer_nail_failure_embedding_step_6000_seed0_50ep_retry_residual_2026-07-02_08-12-52`
- Step 005500 eval: `/data_all/share/FastWAM_zhaoyc_failure/artifacts/evals/hammer_nail_failure_embedding_step_5500_seed0_50ep_residual_2026-07-02_08-29-08`
- Step 005000 eval: `/data_all/share/FastWAM_zhaoyc_failure/artifacts/evals/hammer_nail_failure_embedding_step_5000_seed0_50ep_residual_2026-07-02_08-43-53`

Interpretation: the checkpoint sweep found a passing failure-embedding checkpoint at step 005500, but it did not outperform the observed success-only baseline on the same 50 seeds.
