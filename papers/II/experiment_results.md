# Candidate Paper Experiment Results

This file records experiment numbers that may be used in the paper later.
Historical pilots are retained with their evidence status. Only rows with restored raw summaries and a frozen common protocol should enter the final paper table.

## Closed-Loop Success Rate

| Task | Method / Setting | Success Rate | Successes / Trials | Reported Variance | Status | Notes |
| --- | --- | ---: | ---: | ---: | --- | --- |
| water_plant | pi0.5 | 88.7% | TBD | ±3.1 | tested | Need confirm eval protocol / seeds. |
| water_plant | gr00t n1.5 | 72.7% | TBD | ±1.2 | tested | Need confirm eval protocol / seeds. |
| water_plant | fastwam success-only, step 6500 | 70.0% | 70 / 100 | TBD | reported pilot | Raw rollout summary not tracked in this checkout. |
| water_plant | fastwam | 75.5% | 151 / 200 | TBD | tested | Raw count from current run. |
| water_plant | text failure scratch, step 6500 | 38.0% | 38 / 100 | TBD | reported pilot | Raw rollout summary not tracked in this checkout. |
| water_plant | text failure scratch, step 11000 | 81.0% | 81 / 100 | TBD | reported pilot | Raw rollout summary not tracked in this checkout. |
| water_plant | text failure scratch, step 12240 | 82.0% | 82 / 100 | TBD | tested | Raw count from current run. |
| water_plant | lora continuation | 81.5% | 163 / 200 | TBD | tested | LoRA continued training. |
| hammer_nail | fastwam success-only, step 6500 | 67.5% | 135 / 200 | TBD | reported pilot | Protocol summary not tracked in this checkout. |

## Items To Confirm Before Paper Use

- Evaluation protocol: simulator / real robot, number of seeds, and whether all methods share the same test split.
- For `pi0.5` and `gr00t n1.5`, add raw successes / trials if available.
- Standardize method names before moving the table into `iclr2026_conference.tex`.
