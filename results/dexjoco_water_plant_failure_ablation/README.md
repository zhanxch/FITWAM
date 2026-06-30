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

| Method | Checkpoint | Episodes | Success |
|--------|------------|----------|---------|
| B text failure | step_006500 | 100 | 38/100 |
| B text failure | step_011000 | 100 | 81/100 |
| B text failure | step_012240 | 100 | 82/100 |
| C structured failure | step_006500 | 100 | 74/100 |
| C structured failure | step_011500 | 100 | 59/100 |
| C structured failure | step_012240 | 100 | 4/100 |

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
/data_all/share/FastWAM_zhaoyc_failure/artifacts/reports/latest_fastwam_water_plant_failure_ablation.html
```
