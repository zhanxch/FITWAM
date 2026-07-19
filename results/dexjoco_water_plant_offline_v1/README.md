# Water Plant Offline Fixed-Budget Checkpoint Screening

Four variants were trained from the same source checkpoint for 6500 optimizer
steps. Evaluation used 50 paired simulator seeds, front and wrist views,
23-dimensional proprioception, replan 25, and the same inference seed.
The wrapper limit was 1500 environment steps; the task terminated remaining
unsuccessful episodes at its 1000-step cap, so success@1000 and success@1500
are identical in this screening.

| Variant | Successes | Success rate | Median successful step |
| --- | ---: | ---: | ---: |
| B1: failure video | 45/50 | 90% | 256 |
| B0: success-only | 40/50 | 80% | 273.5 |
| C: residual-only | 37/50 | 74% | 260 |
| M: contrastive steer | 45/50 | 90% | 267 |

| Comparison | Delta | 95% paired bootstrap CI | McNemar p |
| --- | ---: | ---: | ---: |
| B0 vs B1 | -10pp | [-24pp, +4pp] | 0.2668 |
| C vs B1 | -16pp | [-30pp, -2pp] | 0.0574 |
| M vs B1 | 0pp | [-10pp, +10pp] | 1.0000 |
| M vs C | +16pp | [+2pp, +30pp] | 0.0574 |

The fixed-budget checkpoint-screening gate required `M - B1 >= 4pp` and did not pass.
This is a single-training-seed, 50-episode screening result. It does not support
a publication-level gain claim or expansion to additional tasks.

Machine-readable statistics are in
[`paired_comparison.json`](./paired_comparison.json) and
[`paired_comparison.csv`](./paired_comparison.csv). Validation curves are in
[`validation_curve.csv`](./validation_curve.csv). Sanitized per-episode outcomes,
protocol settings, and provenance hashes are in
[`screening_records.json`](./screening_records.json).
