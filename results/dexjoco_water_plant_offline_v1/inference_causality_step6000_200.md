# Paired Rollout Comparison

| Variant | Checkpoint step | Successes | Success rate |
| --- | ---: | ---: | ---: |
| learned | 6000 | 175/200 | 87.5% |
| bypass | 6000 | 20/200 | 10.0% |
| shuffled | 6000 | 176/200 | 88.0% |

| Comparison | Delta | 95% paired bootstrap CI | Exact McNemar p |
| --- | ---: | ---: | ---: |
| bypass_vs_learned | -77.5pp | [-83.5pp, -71.5pp] | 1.72973e-45 |
| shuffled_vs_learned | +0.5pp | [-5.0pp, +6.0pp] | 1 |
