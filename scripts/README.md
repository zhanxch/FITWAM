# Scripts

This repo is aligned with the official FastWAM pipeline
(`third_party/FastWAM`). The active scripts cover the official training path
plus the real-robot deploy stack for the spray_water experiment. Historical
experiments (DexJoCo / EgoDex / EgoVLA / G1) are kept locally under `archive/`
but are not tracked in git.

**Research direction:** see root [`README.md`](../README.md) (Interaction-centric WAM).  
**Upstream FastWAM setup:** [`docs/FASTWAM_UPSTREAM.md`](../docs/FASTWAM_UPSTREAM.md).

## Official-aligned training path

| File | Role |
|------|------|
| `train.py` | Hydra training entrypoint |
| `train_zero1.sh` / `train_zero2.sh` | ZeRO-1 / ZeRO-2 launchers via `accelerate` |
| `precompute_text_embeds.py` | Precompute T5 text-embedding caches for a task |
| `preprocess_action_dit_backbone.py` | Prepare ActionDiT backbone checkpoint |

Standard flow (mirrors the official README):

```bash
# 1. precompute text embeds for the task
python scripts/precompute_text_embeds.py \
  task=spray_water_rot6d_rosbag_ts_filter_uncond_3cam_384_1e-4

# 2. train (baseline, no proprio)
bash scripts/train_spray_water_rot6d.sh
```

spray_water variants (diagnostics from `scripts/diagnose/`):

| Script | Task config | Purpose |
|--------|-------------|---------|
| `train_spray_water_rot6d.sh` | `..._uncond_3cam_384_1e-4` | Baseline (proprio_dim=null) |
| `train_spray_water_rot6d_proprio58.sh` | `..._uncond_3cam_384_1e-4_proprio58` | B3: proprio_dim=58 (H1 fix) |
| `train_spray_water_rot6d_skip_rot6d.sh` | `..._uncond_3cam_384_1e-4_skip_rot6d` | C3: skip rot6d normalization (H2 fix, forked normalizer patch) |

## Real-robot deploy (Wuji/Astribot) — `scripts/1/`

Official FastWAM deploys in-process via `experiments/robotwin/fastwam_policy/deploy_policy.py`
(for RoboTwin sim). The real Wuji robot uses a ZMQ server/client split instead:

```
run_fastwam_client.py (robot)  --ZMQ-->  run_fastwam_server.py (GPU, :5560)
```

| File | Role |
|------|------|
| `1/run_fastwam_server.py` | Load checkpoint; ZMQ server; GR00T obs via `wuji_fastwam_adapter` |
| `1/run_fastwam_client.py` | Real Wuji robot client loop |
| `1/run_gr00t_client.py` | GR00T-server baseline client (for A/B comparison in B1) |
| `1/*_with_env.sh` | Launchers that source the ROS/Astribot environment first |
| `1/run_fastwam_client_diagnostic.sh` | B2: deploy with post-processing disabled (execute-horizon=1, no clip/filter) |
| `wuji_fastwam_adapter.py` | GR00T obs <-> FastWAM policy obs; 58-dim action split |
| `robotwin_camera_utils.py` | Shared 3-cam mosaic + image helpers (robotwin concat layout) |
| `fastwam_policy_server.py` | ZMQ transport (server) |
| `policy_io.py` | Training-aligned policy-server observation contract |
| `fastwam_server_io.json` | Machine-readable API spec for the server |

### Policy server (GPU machine)

```bash
bash scripts/1/run_fastwam_server_with_env.sh \
  --run-dir runs/spray_water_rot6d_rosbag_ts_filter_uncond_3cam_384_1e-4/<run_id> \
  --checkpoint runs/.../step_XXXX.pt \
  --device cuda:0 --host 0.0.0.0 --port 5560
```

### Client (robot)

```bash
bash scripts/1/run_fastwam_client_with_env.sh \
  --policy-host <gpu-ip> --policy-port 5560
```

The server prints `Use proprio: True` when the task config sets `proprio_dim: 58`,
and feeds the 58-dim robot state as a proprio context token to the action expert.

## Diagnostics — `scripts/diagnose/`

See `scripts/diagnose/README.md` for the full sim-vs-real gap diagnosis
(A1-A3, B1-B3, C1-C3) and the spray_water data-pipeline documentation.

## Open-loop evaluation — `scripts/openloop/`

Custom open-loop eval (not in official FastWAM). Entry point:

```bash
python scripts/openloop/run_openloop.py --config-name=openloop_episode \
  task=... data=... ckpt=...
```

Batch helpers: `run_gr00tstyle_openloop_filtered_out.sh`, `run_gr00tstyle_openloop_checkpoints.sh`.

Legacy shims: `run_robotwin_openloop*.py` (deprecated; use `run_openloop.py` + `configs/openloop*.yaml`).
