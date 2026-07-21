# Candidate Paper Experiment Results

This file records experiment numbers that may be used in the paper later.
Historical pilots are retained with their evidence status. Only rows with restored raw summaries and a frozen common protocol should enter the final paper table.

## Closed-Loop Success Rate

| Task | Method / Setting | Success Rate | Successes / Trials | Reported Variance | Status | Notes |
| --- | --- | ---: | ---: | ---: | --- | --- |
| water_plant | pi0.5 | 88.7% | -- | ±3.1 | external result | Raw trials and protocol are not tracked here. |
| water_plant | GR00T N1.5 | 72.7% | -- | ±1.2 | external result | Raw trials and protocol are not tracked here. |
| water_plant | FastWAM success-only, step 6500 | 70.0% | 70 / 100 | -- | historical pilot | Raw rollout summary is not tracked in this checkout. |
| water_plant | FastWAM rollout provenance | 75.5% | 151 / 200 | -- | historical raw count | Source-policy rollout used to build the failure pool. |
| water_plant | text-failure pilot, step 6500 | 38.0% | 38 / 100 | -- | historical pilot | Raw rollout summary is not tracked in this checkout. |
| water_plant | text-failure pilot, step 11000 | 81.0% | 81 / 100 | -- | historical pilot | Training provenance is incomplete in this checkout. |
| water_plant | text-failure pilot, step 12240 | 82.0% | 82 / 100 | -- | historical raw count | Training provenance is incomplete and the protocol differs from the controlled continuation study. |
| water_plant | rollout text-failure LoRA continuation | 81.5% | 163 / 200 | -- | historical raw count | Continued from the success-only source policy. |
| water_plant | B1 failure-video control, E0 step 6500 | 87.0% | 174 / 200 | -- | development checkpoint screening | Paired seeds, replan 25; this E0 set was used to screen checkpoints. |
| water_plant | B0 success-only control, E0 step 6500 | 79.5% | 159 / 200 | -- | development checkpoint screening | Same checkpoint step and paired E0 seeds as B1. |
| water_plant | C residual-only, E0 step 6500 | 74.5% | 149 / 200 | -- | development checkpoint screening | Same checkpoint step and paired E0 seeds as B1. |
| water_plant | M contrastive steer, E0 step 6500 | 87.5% | 175 / 200 | -- | development checkpoint screening | `M - B1 = +0.5pp`, 95% paired bootstrap CI `[-4.5pp, +5.5pp]`; the +4pp gate did not pass. |
| water_plant | S0 source policy, E1 step 6500 | 75.0% | 150 / 200 | -- | fresh-seed reference | Seeds `20262000..20262199`, replan 25, max environment steps 1500. |
| water_plant | B1 failure-video control, E1 step 6000 | 85.0% | 170 / 200 | -- | fresh-seed confirmation | `B1 - S0 = +10.0pp`, 95% paired CI `[+2.5pp, +17.5pp]`, McNemar `p=0.0169`. |
| water_plant | M contrastive steer, E1 step 6000 | 87.5% | 175 / 200 | -- | fresh-seed confirmation | `M - B1 = +2.5pp`, 95% paired CI `[-3.5pp, +8.5pp]`, McNemar `p=0.511`; the +4pp gate did not pass. |
| water_plant | M residual bypass, E1 step 6000 | 10.0% | 20 / 200 | -- | inference causal control | `bypass - learned = -77.5pp`, 95% paired CI `[-83.5pp, -71.5pp]`. |
| water_plant | M cross-episode shuffled steer, E1 step 6000 | 88.0% | 176 / 200 | -- | inference causal control | `shuffled - learned = +0.5pp`, 95% paired CI `[-5.0pp, +6.0pp]`; episode-specific steer content is not established. |
| hammer_nail | FastWAM success-only, step 6500 | 67.5% | 135 / 200 | -- | historical pilot | Protocol summary is not tracked in this checkout. |

## Items To Confirm Before Paper Use

- Evaluation protocol: simulator / real robot, number of seeds, and whether all methods share the same test split.
- For `pi0.5` and `gr00t n1.5`, add raw successes / trials if available.
- Move only frozen-protocol rows with complete provenance into the final paper table.
