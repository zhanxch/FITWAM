# DexJoCo FastWAM Async/LPF Summary

## Official PLAN Phases

| phase | condition | mode | replan | LPF | success | jerk | sign flip | latency mean | underruns | wait s |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| phase0_smoke_blocking_v5 | phase0_blocking_stride24 | blocking | 24 | NA | 0/1 (0%) | 0.3458 | 0.647 | 0.9064 | 0 | 0 |
| phase1_minimal_batchA_v5 | sync_stride24_lpf | blocking | 24 | 0.5 | 1/5 (20%) | 0.1381 | 0.4955 | 0.9862 | 0 | 0 |
| phase1_minimal_batchA_v5 | sync_stride24 | blocking | 24 | NA | 1/5 (20%) | 0.3307 | 0.6499 | 1.006 | 0 | 0 |
| phase1_minimal_batchA_v5 | overlap_stride24 | overlap | 24 | NA | 0/5 (0%) | 0.3657 | 0.6493 | 0.9954 | 46 | 18.4 |
| phase1_minimal_batchA_v5 | sync_default25 | blocking | 25 | NA | 0/5 (0%) | 0.3695 | 0.6518 | 0.9285 | 0 | 0 |
| phase1_minimal_batchB_v5 | overlap_stride24_lpf | overlap | 24 | 0.5 | 1/5 (20%) | 0.1342 | 0.499 | 0.9839 | 45 | 20.36 |
| phase2_stride_supplement_v5 | overlap_stride8_lpf | overlap | 8 | 0.5 | 3/5 (60%) | 0.151 | 0.4629 | 0.9613 | 10.2 | 1.98 |
| phase2_stride_supplement_v5 | overlap_stride32_lpf | overlap | 32 | 0.5 | 2/5 (40%) | 0.1416 | 0.4928 | 0.9414 | 30.8 | 28.92 |
| phase2_stride_supplement_v5 | overlap_stride16_lpf | overlap | 16 | 0.5 | 1/5 (20%) | 0.1499 | 0.4874 | 0.993 | 20.4 | 2.664 |
| phase3_20ep_primary_v5 | overlap_stride8_lpf | overlap | 8 | 0.5 | 7/20 (35%) | 0.1647 | 0.4696 | 1.054 | 25.75 | 6.095 |
| phase3_20ep_primary_v5 | overlap_stride32_lpf | overlap | 32 | 0.5 | 3/20 (15%) | 0.1396 | 0.4994 | 0.8881 | 33.4 | 29.68 |
| phase3_20ep_primary_v5 | sync_stride24 | blocking | 24 | NA | 3/20 (15%) | 0.3822 | 0.6507 | 0.9356 | 0 | 0 |
| phase3_20ep_primary_v5 | sync_stride24_lpf | blocking | 24 | 0.5 | 2/20 (10%) | 0.137 | 0.5041 | 0.9334 | 0 | 0 |

## Exploratory Runs

| phase | condition | mode | replan | LPF | success | jerk | sign flip | latency mean | underruns | wait s |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| explore_lpf_alpha_v5 | overlap_stride8_lpf07 | overlap | 8 | 0.7 | 3/10 (30%) | 0.2425 | 0.5381 | 0.9313 | 43.1 | 12.98 |
| explore_stride8_controls_v5 | overlap_stride8 | overlap | 8 | NA | 4/10 (40%) | 0.3925 | 0.6445 | 0.9493 | 46.6 | 14.26 |
| explore_stride8_controls_v5 | overlap_stride8_lpf03 | overlap | 8 | 0.3 | 2/10 (20%) | 0.08267 | 0.3923 | 0.9495 | 52.3 | 15.82 |

## Reading

- Compare success first, then jerk/sign-flip as motion smoothness proxies.
- Treat queue underruns and wait time as async timing costs.
- Keep exploratory runs separate from official PLAN_dex phases.
