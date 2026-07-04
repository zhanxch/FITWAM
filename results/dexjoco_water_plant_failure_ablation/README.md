# DexJoCo Water Plant Failure Ablation

This folder records the lightweight evidence for the DexJoCo `water_plant`
failure-data ablation. Large rollout videos, action `.npz` files, and model
checkpoints stay on the shared server; this repository tracks only summaries,
manifest paths, and the interactive HTML report.

## Protocol

- Task: DexJoCo `water_plant`
- Observations: 2 cameras + proprioception
- Control: blocking, `replan_steps=25`
- Evaluation cap: `max_env_steps=600`
- Saved artifacts: rollout mp4 and action `.npz` for new multi-GPU runs

## Results

| Variant | Ckpt step | Protocol | N | success@600 | median success step |
|---------|-----------|----------|---|-------------|---------------------|
| text_failure | 6500 | replan25, 600step | 100 | 38/100 = 38% | 263.5 |
| text_failure | 7000 | replan25, 600step | 50 | 25/50 = 50% | 267 |
| text_failure | 8000 | replan25, 600step | 50 | 35/50 = 70% | 255 |
| text_failure | 9000 | replan25, 600step | 50 | 35/50 = 70% | 278 |
| text_failure | 10000 | replan25, 600step | 50 | 37/50 = 74% | 270 |
| text_failure | 11000 | replan25, 600step | 100 | 81/100 = 81% | 264 |
| text_failure | 12000 | replan25, 600step | 50 | 40/50 = 80% | 272 |
| text_failure | 12240 | replan25, 600step | 100 | 82/100 = 82% | 261 |
| structured_failure | 1000 | replan25, 600step | 50 | 1/50 = 2% | 298 |
| structured_failure | 2000 | replan25, 600step | 50 | 4/50 = 8% | 250 |
| structured_failure | 3000 | replan25, 600step | 50 | 5/50 = 10% | 300 |
| structured_failure | 4000 | replan25, 600step | 50 | 40/50 = 80% | 279.5 |
| structured_failure | 5000 | replan25, 600step | 50 | 19/50 = 38% | 261 |
| structured_failure | 6000 | replan25, 600step | 50 | 41/50 = 82% | 253 |
| structured_failure | 6500 | replan25, 600step | 100 | 74/100 = 74% | 265 |
| structured_failure | 11500 | replan25, 600step | 100 | 59/100 = 59% | 257 |
| structured_failure | 12240 | replan25, 600step | 100 | 4/100 = 4% | 525 |

The main observation is the opposite training trajectory:

- B improves with longer training: `38 -> 81 -> 82`.
- C works at the mid checkpoint but collapses late: `74 -> 59 -> 4`.

This means the structured variant should not be dismissed as unusable from
deploy alone; the immediate diagnosis target is late-checkpoint stability.

## Files

- `summary.csv`: compact metric table.
- `artifact_manifest.csv`: remote summary/video/action locations.
- `interactive_report.html`: local copy of the current interactive report.

The remote canonical report is:

```text
/data_all/zhaoyc/Summer2/FastWAM_zhaoyc_failure_moved_from_share_20260703/artifacts/reports/latest_fastwam_water_plant_failure_ablation.html
```
