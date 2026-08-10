# Frozen figures: L3 transition 1NN + L2-visual frame coverage

Approved / style-locked pair (2026-08-09). **This is the only active L2/L3 experience-coverage deliverable.**

Historical iteration dirs/scripts → `archive/l2l3_experience_coverage_20260809/`.

## Figures

| File | Panel |
|------|--------|
| `L3_transition_nn_hist.png` | L3 transition novelty (original binned hist) |
| `L3_transition_nn_smooth.png` / `.pdf` | L3 same data, **KDE smooth** (preferred style) |
| `L2_visual_frame_all.png` / `.pdf` | L2–Visual · Frame-level, **log-x** |
| `L2_visual_frame_all_linear.png` / `.pdf` | L2–Visual linear-x (polished L2 chrome) |
| `L2_visual_frame_all_linear_l3style_smooth.png` / `.pdf` | L2 linear distances, **L3 palette + KDE smooth** |

### L3: is there trimming?
**No head/tail trim.** This frozen L3 is *not* the later cfg10086 headtrim / trim8s runs.

What it *does* do:
1. **Episode subsample**: all 45 failures; 50 / 155 successes (seed `20260808`)
2. **Frame stride**: keep every 5th frame (`stride=5`)
3. **Transition lag**: feature `[s_t, a_t, s_{t+5}]`; starts with `t+5 ≥ T` are dropped
4. **Dataset**: raw S0 rollout `water_plant_s0_rollout_b0_b1_20260718/rollout` (failures keep long tails; many hit length 1000)

## Provenance

### L3
- Archived source run: `archive/l2l3_experience_coverage_20260809/results/experience_distribution_coverage_20260808/`
- Cache here: `L3_distances.npz`, `L3_meta.json`

### L2-visual
- Feature: `concat(s, z_VAE, a)` · global 1NN to Expert · **all frames, no trim**
- Rollout: S0 4×50 cfg seed 10086
- Cache: `L2_visual_distances.npz`, `L2_visual_meta.json`
- Features for recompute: `L2_visual_features_fulltraj.npz`

## Regenerate

```bash
conda activate web
python scripts/analysis/render_frozen_l2_visual_l3_transition.py --mode replot
# optional full L2 NN recompute:
python scripts/analysis/render_frozen_l2_visual_l3_transition.py --mode recompute-l2
```

Code snapshots under `code/`.
