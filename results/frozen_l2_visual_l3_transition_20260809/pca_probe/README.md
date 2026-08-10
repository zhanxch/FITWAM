# PCA probe (kept artifacts)

Only two generations are retained here.

## 1) Original PCA (no filter)

Script: `scripts/analysis/plot_frozen_l2_l3_pca.py`

| File | Content |
|------|---------|
| `fig_pca_joint_scatter_L2_L3.{png,pdf}` | Joint PCA scatter (E/S/F) |
| `fig_pca_joint_density_L2_L3.{png,pdf}` | Joint PCA density contours |
| `fig_pca_expertfit_scatter_L2_L3.{png,pdf}` | PCA fit on Expert, project S/F |
| `pca_report.json` | stats |

```bash
conda activate web
python scripts/analysis/plot_frozen_l2_l3_pca.py
```

## 2) Final: PCA-plane filter → drop → refit

Script: `scripts/analysis/plot_pca2d_filter_refit.py`

Protocol: PCA#1 on E+S+F → drop Success inside Expert coverage **in PCA#1 plane** → PCA#2 refit → scatter + circles.

| File | Content |
|------|---------|
| `fig_pca2d_filter_refit_rms_{L2,L3,overview}.*` | RMS-circle membership (main) |
| `fig_pca2d_filter_refit_ellipse_{L2,L3,overview}.*` | Mahalanobis ellipse membership |
| `pca2d_filter_refit_report.json` | drop rates / shifts |

```bash
conda activate web
python scripts/analysis/plot_pca2d_filter_refit.py
```

Intermediate probes (high-D 1NN filter, no-fail PCA, coverage-only circles, etc.) were removed.
