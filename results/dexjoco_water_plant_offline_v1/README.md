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

M improves over B1 at step 6000 and under validation-based checkpoint
selection. The step-6500 difference is negligible. These results use one
training seed and show checkpoint sensitivity, so multi-seed training remains
required before a publication-level gain claim.

Machine-readable statistics are in
[`checkpoint_screening_200.csv`](./checkpoint_screening_200.csv) and
[`paired_comparison_200.csv`](./paired_comparison_200.csv). The earlier
50-episode screening files remain in this directory for provenance.
