# DexJoCo Async Microwave Result Subset

This is a compact artifact subset from a historical DexJoCo `bimanual_microwave_cook` async/LPF inference ablation.

It is included as supporting engineering evidence only. The current FITWAM simulation focus is the `water_plant` and `hammer_nail` failure-video/self-evolution sequence described in the root README.

## What Is Included

| Path | Contents |
|------|----------|
| `combined_summary.md` / `.csv` | Full condition-level metric table. |
| `combined_video_manifest.csv` | Sanitized manifest of the original video/action outputs. |
| `phase3_20ep_primary_v5/` | Main 20-episode comparison summaries and manifests. |
| `samples/` | Representative success/failure video and action pairs for each primary condition. |

The full raw media set was about 400 MB, so the repository keeps only representative `mp4` and `actions.npz` files plus complete summaries/manifests.

## Main 20-Episode Result

| Condition | Success | Action jerk | Sign flip | Interpretation |
|-----------|---------|-------------|-----------|----------------|
| `sync_stride24` | 3/20 | 0.3822 | 0.6507 | Blocking baseline. |
| `sync_stride24_lpf` | 2/20 | 0.1370 | 0.5041 | Smoother, but no success gain. |
| `overlap_stride8_lpf` | 7/20 | 0.1647 | 0.4696 | Best success/smoothness trade-off in this pass. |
| `overlap_stride32_lpf` | 3/20 | 0.1396 | 0.4994 | Smoother, but no success gain over baseline. |

Use these results as context for async inference behavior, not as evidence for the current failure-data training claim.
