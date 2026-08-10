# Frozen: open-loop interval split (B1 ep4 + S0 ep0)

Locked 2026-08-09 as the preferred visualization for:

- **B1**: front-wide / mid-narrow predicted-action interval on expert ep4
- **S0**: stable interval on expert ep0 (separate episode)

## Protocol (locked)

- Metric: open-loop multi-seed `‖σ(a0)‖₂` interval width (centered band)
- Cut: progress `< 0.32` dropped
- Mid-window frames: progress `[0.40, 0.60]` (front|wrist thumbs)
- Style: smoothed fill only (no outline); panel A = B1 only, panel B = S0 only
- Coarse screen: stride=12, K=4 (formal denser infer optional later)

## Canonical files

- `fig_recommended_split_ep4_ep0_interval_band.{png,pdf}`
- `ep4_B1_interval_with_frames.mp4`
- `ep0_S0_interval_with_frames.mp4`
- `interval_frames_meta.json`
- `ep004_widths.npz`, `ep000_widths.npz` (screen widths)

Superseded alternate open-loop / interaction-sensitivity schemes live under
`results/arxiv/openloop_interval_alts_20260809/`.
