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
| `convert_lerobot_to_everobot.py` | Convert LeRobot dataset → EveRobot manifest + arrays |
| `train_everobot.py` | EveRobot training entrypoint (episode-level sampling) |
| `train_everobot.sh` | Multi-GPU launcher for EveRobot training |

Standard flow (mirrors the official README):

```bash
# 1. precompute text embeds for the task
python scripts/precompute_text_embeds.py \
  task=spray_water_rot6d_rosbag_ts_filter_uncond_3cam_384_1e-4

# 2. train (baseline, no proprio)
bash scripts/train_spray_water_rot6d.sh
```

spray_water variants:

| Script | Task config | Purpose |
|--------|-------------|---------|
| `train_spray_water_rot6d.sh` | `..._uncond_3cam_384_1e-4` | Baseline (proprio_dim=null) |
| `train_spray_water_rot6d_proprio58.sh` | `..._uncond_3cam_384_1e-4_proprio58` | B3: proprio_dim=58 (H1 fix) |
| `train_spray_water_rot6d_skip_rot6d.sh` | `..._uncond_3cam_384_1e-4_skip_rot6d` | C3: skip rot6d normalization (H2 fix, forked normalizer patch) |

## EveRobot training (episode-level, DiffSynth-style)

Each episode (one mp4 per camera) is treated as a single training sample,
following `third_party/diffsynth-studio` WAN fine-tuning where each video is
one sample. Instead of DiffSynth's `dataset_repeat` (seeing the same clip N
times), EveRobot drops the first N frames to create diverse temporal windows.

```bash
# 1. Convert LeRobot → EveRobot manifest + per-episode .npz arrays
python scripts/convert_lerobot_to_everobot.py \
  --dataset-dir data/water_plant_fastwam \
  --video-keys front wrist

# 2. Precompute text embeds (same as standard training)
python scripts/precompute_text_embeds.py task=everobot_water_plant

# 3. Train
bash scripts/train_everobot.sh task=everobot_water_plant
```

Key EveRobot overrides (in `configs/data/everobot_water_plant.yaml`):

| Override | Default | Meaning |
|----------|---------|---------|
| `drop_first_frames` | 60 | Max leading frames to drop per episode |
| `drop_first_frames_random` | true | Random drop in [0, N] each access (augmentation) |
| `samples_per_episode` | 3 | Sub-clips per episode per epoch (replaces `dataset_repeat`) |

Output format is identical to standard `RobotVideoDataset`, so `FastWAM.training_loss()`
and the deploy stack work unchanged. EveRobot is a **parallel** training path —
the original `train.py` / `train_zero1.sh` flow is unaffected.

### Full-episode training (DiffSynth-style, variable T)

Use `EveRobotFullEpisodeDataset` when each **whole mp4** should be one sample
(no fixed `num_frames=33` window). Video frames are subsampled every 4 control
steps; episode tail is trimmed so `T_video % 4 == 1` and
`action_horizon == 4 * (T_video - 1)`.

```bash
# Same convert + text-embed steps as above, then:
bash scripts/train_everobot.sh task=everobot_water_plant_full
bash scripts/train_everobot.sh task=everobot_water_plant_full_lora
```

| Config | Meaning |
|--------|---------|
| `task=everobot_water_plant_full` | Full episode, full MoT FT |
| `task=everobot_water_plant_full_lora` | Full episode + video LoRA |

Data config: `configs/data/everobot_water_plant_full.yaml` (`batch_size: 1` required).

**Default (no augmentation):** 95 episodes × 1 sample/epoch = 95 steps/epoch, 5 epochs ≈ 475 steps.

**Optional temporal augmentation** (randomly drop leading control steps, DiffSynth `dataset_repeat`-style):

```bash
bash scripts/train_everobot.sh task=everobot_water_plant_full_lora \
  data.train.drop_first_frames=60 \
  data.train.drop_first_frames_random=true \
  data.train.samples_per_episode=3
```

This yields 95 × 3 = **285** samples/epoch; each access samples a new random start in `[0, 60]`.

| Override | Default | With augmentation |
|----------|---------|-------------------|
| `drop_first_frames` | 0 | 60 |
| `drop_first_frames_random` | false | true |
| `samples_per_episode` | 1 | 3 |
| Train samples / epoch | 95 | 285 |

```bash
# preset task with augmentation enabled
bash scripts/train_everobot.sh task=everobot_water_plant_full_dropaug
```

### Video LoRA + ActionDiT full fine-tune (optional)

DiffSynth-style LoRA on the **video DiT only**; ActionDiT and proprio encoder remain
full-parameter trainable. Default `configs/model/fastwam.yaml` is unchanged (full MoT FT).

| Config | Entry |
|--------|--------|
| `model=fastwam_video_lora` | Enables `model.video_lora.enabled=true` (rank 32, self-attn + FFN targets) |
| `task=everobot_water_plant_lora` | EveRobot + LoRA |
| `task=everobot_water_plant_full` | Full episode (variable T) |
| `task=everobot_water_plant_full_lora` | Full episode + LoRA |
| `task=water_plant_uncond_2cam_384_1e-4_lora` | LeRobot sliding-window + LoRA |

```bash
# EveRobot + video LoRA
bash scripts/train_everobot.sh task=everobot_water_plant_lora

# LeRobot water_plant + video LoRA
bash scripts/train_water_plant_2cam.sh task=water_plant_uncond_2cam_384_1e-4_lora
```

Checkpoints use `checkpoint_format: video_lora_v1` (compact: `video_lora` + `mot_action` +
`proprio_encoder`). Deploy/inference loads them when the run `config.yaml` has
`model.video_lora.enabled=true` (written automatically from the task config).

Requires `peft` (`pip install -e .`).

## Real-robot deploy (Wuji/Astribot) — `scripts/wuji/`

Official FastWAM deploys in-process via `experiments/robotwin/fastwam_policy/deploy_policy.py`
(for RoboTwin sim). The real Wuji robot uses a ZMQ server/client split instead:

```
run_fastwam_client.py (robot)  --ZMQ-->  run_fastwam_server.py (GPU, :5560)
```

| File | Role |
|------|------|
| `wuji/run_fastwam_server.py` | Load checkpoint; ZMQ server; GR00T obs via `wuji_fastwam_adapter` |
| `wuji/run_fastwam_client.py` | Real Wuji robot client loop |
| `wuji/run_gr00t_client.py` | GR00T-server baseline client (for A/B comparison in B1) |
| `wuji/*_with_env.sh` | Launchers that source the ROS/Astribot environment first |
| `wuji/run_fastwam_client_diagnostic.sh` | B2: deploy with post-processing disabled (execute-horizon=1, no clip/filter) |
| `wuji_fastwam_adapter.py` | GR00T obs <-> FastWAM policy obs; 58-dim action split |
| `robotwin_camera_utils.py` | Shared 3-cam mosaic + image helpers (robotwin concat layout) |
| `fastwam_policy_server.py` | ZMQ transport (server) |
| `policy_io.py` | Training-aligned policy-server observation contract |
| `fastwam_server_io.json` | Machine-readable API spec for the server |

### Policy server (GPU machine)

```bash
bash scripts/wuji/run_fastwam_server_with_env.sh \
  --run-dir runs/spray_water_rot6d_rosbag_ts_filter_uncond_3cam_384_1e-4/<run_id> \
  --checkpoint runs/.../step_XXXX.pt \
  --device cuda:0 --host 0.0.0.0 --port 5560
```

### Client (robot)

```bash
bash scripts/wuji/run_fastwam_client_with_env.sh \
  --policy-host <gpu-ip> --policy-port 5560
```

The server prints `Use proprio: True` when the task config sets `proprio_dim: 58`,
and feeds the 58-dim robot state as a proprio context token to the action expert.

## Open-loop evaluation — `scripts/openloop/`

Custom open-loop eval (not in official FastWAM). Entry point:

```bash
python scripts/openloop/run_openloop.py --config-name=openloop_episode \
  task=... data=... ckpt=...
```

Batch helpers: `run_gr00tstyle_openloop_filtered_out.sh`, `run_gr00tstyle_openloop_checkpoints.sh`.

Legacy shims: `run_robotwin_openloop*.py` (deprecated; use `run_openloop.py` + `configs/openloop*.yaml`).
