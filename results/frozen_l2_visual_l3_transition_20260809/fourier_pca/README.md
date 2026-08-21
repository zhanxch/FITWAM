# Fourier spectrum + episode PCA (L2-visual / L3)

Probe on the same frozen diversity features as `pca_probe/`.
**All PCA panels are episode-level** (one point = one episode).

## Protocol

1. Rebuild per-episode trajectories:
   - **L2**: `concat(s, z_VAE, a)` z-scored on Expert (all frames)
   - **L3**: `[s_t, a_t, s_{t+5}]` z-scored on Expert (`stride=5`)
2. Per-episode rFFT PSD (per-dim mean removal + Hann; mean over feature dims).
3. **Episode log-PSD PCA**: `log10(PSD)` over all frequencies → joint PC1–PC2.
4. **Episode high-pass PCA**: same, but only bins with `f ≥ 0.125` cycles/frame (= 25% Nyquist).

## Figures

| File | Content |
|------|---------|
| `fig_fourier_spectrum_{L2,L3}.*` | Mean PSD + cumulative energy |
| `fig_fourier_episode_spectral_pca_{L2,L3}.*` | Episode log-PSD (full band) PCA |
| `fig_fourier_highpass_pca_{L2,L3}.*` | Episode HF-band log-PSD PCA |
| `fig_fourier_highpass_overview_L2_L3.*` | 2×2: full vs HF, L2/L3 |

## Reproduce

```bash
conda activate web
python scripts/analysis/plot_frozen_l2_l3_fourier_pca.py
```
