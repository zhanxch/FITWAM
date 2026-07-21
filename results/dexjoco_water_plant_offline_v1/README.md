# Water Plant Offline Checkpoint Screening

Four variants were trained from the same source checkpoint. Evaluation used
200 paired simulator seeds, front and wrist views, 23-dimensional
proprioception, replan 25, and the same inference seed.
The wrapper limit was 1500 environment steps; the task terminated remaining
unsuccessful episodes at its 1000-step cap. Videos and action arrays were
retained outside Git.

| Checkpoint | B0 | B1 | C | M |
| --- | ---: | ---: | ---: | ---: |
| step 5000 | 82.0% | 84.0% | 74.5% | **88.0%** |
| step 6000 | 70.5% | 75.5% | 75.5% | **84.5%** |
| step 6500 | 79.5% | 87.0% | 74.5% | **87.5%** |
| validation-best | 72.5% (5500) | 80.5% (4500) | 74.5% (6500) | **88.0% (5000)** |

| M vs B1 | Delta | 95% paired bootstrap CI | Exact McNemar p |
| --- | ---: | ---: | ---: |
| step 5000 | +4.0pp | [-2.5pp, +10.5pp] | 0.2912 |
| step 6000 | +9.0pp | [+3.0pp, +15.0pp] | 0.0064 |
| step 6500 | +0.5pp | [-4.5pp, +5.5pp] | 1.0000 |
| validation-best | +7.5pp | [+1.0pp, +14.0pp] | 0.0357 |

This table is E0 development-set checkpoint screening. M improves over B1 at
step 6000 and under validation-based checkpoint selection, while the step-6500
difference is negligible.

On the separate E1 fresh seeds `20262000..20262199`, fixed step-6000 B1 scored
`170/200 = 85.0%` and M scored `175/200 = 87.5%`. The paired difference was
`+2.5pp` (95% paired bootstrap CI `[-3.5pp, +8.5pp]`; exact McNemar
`p=0.511`). E1 therefore did not confirm the E0 gain. The matching S0 source
policy scored `150/200 = 75.0%`. Paired B1-S0 was `+10.0pp` (95% CI
`[+2.5pp, +17.5pp]`, McNemar `p=0.0169`); paired M-S0 was `+12.5pp` (95% CI
`[+5.5pp, +19.5pp]`, `p=0.00126`). These efficacy results do not establish a
causal benefit from episode-specific steer content. Machine-readable E1 statistics are in
[`fresh_seed_confirmation_200.json`](./fresh_seed_confirmation_200.json) and
[`fresh_seed_confirmation_200.csv`](./fresh_seed_confirmation_200.csv).

Using the same M step-6000 checkpoint and E1 seeds, learned steer scored
`175/200 = 87.5%`, exact residual bypass scored `20/200 = 10.0%`, and
cross-episode shuffled steer scored `176/200 = 88.0%`. Bypass minus learned was
`-77.5pp` (95% CI `[-83.5pp, -71.5pp]`; McNemar `p=1.73e-45`), while shuffled
minus learned was `+0.5pp` (95% CI `[-5.0pp, +6.0pp]`; `p=1.0`). The steer
path is necessary for this checkpoint, but these results do not show that its
episode-specific embedding content improves success. Machine-readable results
are in [`inference_causality_step6000_200.json`](./inference_causality_step6000_200.json)
and [`inference_causality_step6000_200.csv`](./inference_causality_step6000_200.csv).

Machine-readable statistics are in
[`checkpoint_screening_200.csv`](./checkpoint_screening_200.csv) and
[`paired_comparison_200.csv`](./paired_comparison_200.csv). The earlier
50-episode screening files remain in this directory for provenance.
