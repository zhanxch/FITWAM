# Water Plant Offline Checkpoint Screening

Four variants were trained from the same source checkpoint. Evaluation used
200 paired simulator seeds, front and wrist views, 23-dimensional
proprioception, replan 25, and the same inference seed.
The wrapper limit was 1500 environment steps; the task terminated remaining
unsuccessful episodes at its 1000-step cap. Videos and action arrays were
retained outside Git.

| Checkpoint | B0 | B1 | C | M |
| --- | ---: | ---: | ---: | ---: |
| validation-best (primary) | 72.5% (5500) | 80.5% (4500) | 74.5% (6500) | **88.0% (5000)** |
| step 5000 | 82.0% | 84.0% | 74.5% | **88.0%** |
| step 6000 | 70.5% | 75.5% | 75.5% | **84.5%** |
| step 6500 | 79.5% | 87.0% | 74.5% | **87.5%** |

| M vs B1 | Delta | 95% paired bootstrap CI | Exact McNemar p |
| --- | ---: | ---: | ---: |
| step 5000 | +4.0pp | [-2.5pp, +10.5pp] | 0.2912 |
| step 6000 | +9.0pp | [+3.0pp, +15.0pp] | 0.0064 |
| step 6500 | +0.5pp | [-4.5pp, +5.5pp] | 1.0000 |
| validation-best | +7.5pp | [+1.0pp, +14.0pp] | 0.0357 |

This table is E0 development-set checkpoint screening. Validation-best is the
primary checkpoint rule; fixed-step rows are secondary checkpoint-sensitivity
diagnostics. M improves over B1 under validation-based selection and at step
6000, while the step-6500 difference is negligible.

On the separate E1 fresh seeds `20262000..20262199`, the secondary fixed
step-6000 comparison scored
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

The primary Strict E2 checkpoint diagnostic evaluated each arm's
validation-best checkpoint on the existing E2 seeds `20262200..20262399`.
Residual-only C at step 5000 scored `170/200 = 85.0%`, M-pair-shuffle at step
6500 scored `142/200 = 71.0%`, and strict M at step 6500 scored
`109/200 = 54.5%`.

| Validation-best comparison | Delta | 95% paired bootstrap CI | Exact McNemar p |
| --- | ---: | ---: | ---: |
| C - strict M | +30.5pp | [+23.0pp, +38.0pp] | 3.14e-13 |
| C - M-pair-shuffle | +14.0pp | [+7.0pp, +21.0pp] | 2.34e-4 |
| M-pair-shuffle - strict M | +16.5pp | [+8.5pp, +24.5pp] | 1.12e-4 |

This follow-up reuses E2 development seeds and is not a new holdout. It
confirms that checkpoint selection does not rescue the current Teacher/pair
objective. Machine-readable statistics and per-seed outcomes are in
[`strict_e2_validation_best_200.json`](./strict_e2_validation_best_200.json)
and [`strict_e2_validation_best_200.csv`](./strict_e2_validation_best_200.csv).

Strict E2 then retrained M, M-pair-shuffle, and residual-only C from one serialized common
initialization. The first, secondary fixed-step evaluation used step 6000 on seeds
`20262200..20262399`. S0 scored `154/200 = 77.0%`, B1 scored
`164/200 = 82.0%`, C scored `164/200 = 82.0%`, strict M scored `117/200 = 58.5%`, and
M-pair-shuffle scored `147/200 = 73.5%`. Strict M was `-23.5pp` below B1
(95% CI `[-31.5pp, -15.5pp]`) and `-15.0pp` below M-pair-shuffle
(95% CI `[-23.0pp, -7.0pp]`). C and B1 had zero aggregate difference
(paired 95% CI `[-6.5pp, +6.5pp]`; McNemar `p=1.0`). Strict M was `-23.5pp`
below C (95% CI `[-31.0pp, -16.0pp]`). The residual scaffold recovered the B1
success rate when Teacher/pair supervision was disabled, while the current
Teacher/pair objective failed the causal gate. Machine-readable statistics
are in [`strict_e2_common_init_200.json`](./strict_e2_common_init_200.json) and
[`strict_e2_common_init_200.csv`](./strict_e2_common_init_200.csv).
One same-seed C repeat preserved the initial state and success outcome but not
the exact action trajectory. Here, paired means matched simulator seeds rather
than bitwise-deterministic rollouts.

Machine-readable statistics are in
[`checkpoint_screening_200.csv`](./checkpoint_screening_200.csv) and
[`paired_comparison_200.csv`](./paired_comparison_200.csv). The earlier
50-episode screening files remain in this directory for provenance.
