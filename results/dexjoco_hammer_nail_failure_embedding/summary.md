# DexJoCo Hammer Nail Failure-Embedding Sweep

Task: `hammer_nail` (`rand_obj`, blocking control, 50 episodes, seed 0).

Gate for this task: `min(pi0.5=84.7, GR00T N1.5=67.3) - 1.0 = 66.3%`.

| Variant | Ckpt step | Protocol | N | success@600 | success@1000 | success @1500/final | median success step |
|---------|-----------|----------|---|-------------|--------------|--------------------|---------------------|
| baseline_success_only | 6650 | replan24 | 50 | 34/50 = 68% | 36/50 = 72% | 36/50 = 72% | 239.5 |
| failure_embedding | 3500 | replan25, env1500 | 50 | 19/50 = 38% | 21/50 = 42% | 21/50 = 42% | 380 |
| failure_embedding | 4000 | replan25, env1500 | 50 | 15/50 = 30% | 21/50 = 42% | 21/50 = 42% | 331 |
| failure_embedding | 4500 | replan25, env1500 | 50 | 15/50 = 30% | 23/50 = 46% | 23/50 = 46% | 353 |
| failure_embedding | 5000 | replan24 | 50 | 24/50 = 48% | 27/50 = 54% | 27/50 = 54% | 363 |
| failure_embedding | 5500 | replan24 | 50 | 29/50 = 58% | 34/50 = 68% | 34/50 = 68% | 322.5 |
| failure_embedding | 6000 | replan24 | 50 | 28/50 = 56% | 32/50 = 64% | 32/50 = 64% | 374.5 |
| text_failure | 4000 | replan25, env1500 | 50 | 21/50 = 42% | 27/50 = 54% | 27/50 = 54% | 352 |
| text_failure | 4500 | replan25, env1500 | 50 | 18/50 = 36% | 19/50 = 38% | 19/50 = 38% | 252 |
| text_failure | 5000 | replan25, env1500 | 50 | 22/50 = 44% | 23/50 = 46% | 23/50 = 46% | 278 |
| text_failure | 6000 | replan25, env1500 | 50 | 28/50 = 56% | 31/50 = 62% | 31/50 = 62% | 315 |
| text_failure | 6000 | replan25, env600 | 50 | 18/50 = 36% | — | — | 219 |

Remote artifacts:

- Baseline eval: `/data_all/zhaoyc/Summer2/FastWAM_zhaoyc_failure_moved_from_share_20260703/artifacts/evals/hammer_nail_baseline_step_006650_seed0_50ep_retry6_2026-07-01_19-05-07`
- Failure embedding run: `/data_all/zhaoyc/Summer2/dexjoco_fastwam_results_moved_from_share_20260703/dexjoco_hammer_nail_failure_embedding_2cam_proprio_1e-4/2026-07-01_20-55-26`
- Step 006000 eval: `/data_all/zhaoyc/Summer2/FastWAM_zhaoyc_failure_moved_from_share_20260703/artifacts/evals/hammer_nail_failure_embedding_step_6000_seed0_50ep_retry_residual_2026-07-02_08-12-52`
- Step 005500 eval: `/data_all/zhaoyc/Summer2/FastWAM_zhaoyc_failure_moved_from_share_20260703/artifacts/evals/hammer_nail_failure_embedding_step_5500_seed0_50ep_residual_2026-07-02_08-29-08`
- Step 005000 eval: `/data_all/zhaoyc/Summer2/FastWAM_zhaoyc_failure_moved_from_share_20260703/artifacts/evals/hammer_nail_failure_embedding_step_5000_seed0_50ep_residual_2026-07-02_08-43-53`
- Text failure step 006000 eval (env1500): `/data_all/zhaoyc/Summer2/FastWAM_zhaoyc_failure_moved_from_share_20260703/artifacts/evals/hammer_nail_text_failure_step_6000_seed0_50ep_2026-07-02_1111_hammer_text_failure_6000_g4567_replan25_env1500`
- Text failure step 006000 eval (env600): `/data_all/zhaoyc/Summer2/FastWAM_zhaoyc_failure_moved_from_share_20260703/artifacts/evals/hammer_nail_text_failure_step_006000_seed0_50ep_env600_replan25_20260703_0010`

Interpretation: step 005500 is the best failure-embedding checkpoint in this sweep, but it remains below the success-only baseline.
