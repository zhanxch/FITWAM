# DexJoCo Failure Datasets

Shared LeRobot-format failure datasets:

```text
/data_all/share/dexjoco_failure_datasets
```

| Dataset | Episodes | Frames | FPS | Videos | Notes |
|---------|----------|--------|-----|--------|-------|
| water_plant_failure_fastwam_2cam_text | 100 | 51417 | 30 | 200 | two-camera failure rollouts |
| hammer_nail_failure_fastwam_2cam_text | 100 | 60000 | 30 | 200 | two-camera failure rollouts |
| fold_glasses_failure_fastwam_2cam_text | 100 | 60000 | 30 | 200 | two-camera failure rollouts; collection baseline was 20/123 success |

Each directory contains `data/`, `videos/`, and `meta/` with front/wrist video, action, and state features.
