# B3: Validate the no-proprio hypothesis (H1)

> **历史记录 / 已被基线吸收。** 官方 FastWAM 默认 `proprio_dim` 自动从数据
> config 的 `proprio_output_dim` 继承（=58），即官方训练本来就带 proprio。
> 之前 spray_water 基线显式写 `proprio_dim: null` 主动关掉 proprio，那是 H1
> 假设的来源，**不是官方默认**。现已删除该覆盖，基线走官方默认带 proprio=58，
> 因此 **H1（无 proprio）在官方流程下本就不成立**，B3 不再作为独立实验。
> 本文档保留作历史记录；如需测"无 proprio"对照（诊断 H1 是否真的是问题），
> 可临时建一个 `proprio_dim: null` 的 task config 做 A/B，但这偏离官方默认。

## Goal

Determine whether FastWAM's lack of proprioception (`proprio_dim: null` in the
task config) is a root cause of the sim-vs-real gap. This is the
**highest-priority** hypothesis per the analysis plan.

## Why retrain (not inject)

The plan listed two options:

1. **Retrain** with `proprio_dim=58`.
2. **Inject external proprio** into the existing checkpoint at deploy time.

Option 2 is **not feasible** for the existing spray_water checkpoint because
the model was instantiated with `proprio_dim=null`, which means
`self.proprio_encoder = None` (see `src/fastwam/models/wan22/fastwam.py`). There
is no network module to receive the proprio input, so feeding a 58-dim state at
inference would have no effect (the `infer_action` proprio branch is skipped
when `proprio_encoder` is None).

Therefore B3 requires a **controlled retrain** with proprioception enabled.

## Provided artifacts

| File | Purpose |
|---|---|
| `configs/task/spray_water_rot6d_rosbag_ts_filter_uncond_3cam_384_1e-4_proprio58.yaml` | New task config: identical to baseline except `proprio_dim: 58` |
| `scripts/train_spray_water_rot6d_proprio58.sh` | Training launch script (A/B to `scripts/train_spray_water_rot6d.sh`) |

The new config enables the **official proprio context-token** path
(`proprio_dim: 58`). The deploy server
(`scripts/1/run_fastwam_server.py`) decides `use_proprio` by checking
`model.proprio_dim is not None`, so the context-token path is automatically
wired end-to-end without server changes.

## Run the experiment

### 1. Train (quick smoke test first)

```bash
# Smoke test (~200 steps, confirms the proprio path trains without error):
bash scripts/train_spray_water_rot6d_proprio58.sh \
    max_steps=200 save_every=100 eval_every=100

# Full run (matches baseline schedule):
bash scripts/train_spray_water_rot6d_proprio58.sh
```

The run lands under
`runs/spray_water_rot6d_rosbag_ts_filter_uncond_3cam_384_1e-4_proprio58/<timestamp>/`.
The wandb name is the same with a `_proprio58` suffix.

### 2. Deploy (A/B against the proprio=null checkpoint)

Use the **same** server/client scripts — the server auto-detects
`model.proprio_dim` and will print `Use proprio: True`:

```bash
# Server (note: no code change needed; use_proprio is auto-detected):
python scripts/1/run_fastwam_server.py \
    --run-dir runs/spray_water_rot6d_rosbag_ts_filter_uncond_3cam_384_1e-4_proprio58/<ts> \
    --checkpoint <ckpt> \
    --dump-dir runs/diag_b3_proprio/server_dump

# Client (same as baseline; the robot state is already in the GR00T obs):
bash scripts/1/run_fastwam_client_with_env.sh \
    --host <ip> --port 5560 \
    --task "Pick up the spray bottle, pump it to build up pressure, then spray water on the flowers" \
    --execute-horizon 32 --no-home --eef-control-way direct \
    --arm-interp-hz 30 --hand-interp-hz 30 --no-wbc \
    --workspace-radius 2 2 2 --max-eef-step 0.05 --max-eef-rotation-step-deg 10 \
    --dump-raw-actions runs/diag_b3_proprio/client_raw \
    --log-dir runs/diag_b3_proprio --log-prefix client_fastwam_proprio
```

### 3. Analyze

```bash
python scripts/diagnose/analyze_deploy_dumps.py \
    --server-dir runs/diag_b3_proprio/server_dump \
    --client-dir runs/diag_b3_proprio/client_raw
```

The analyzer will confirm `use_proprio=True` and `normalized_proprio_is_none=False`.

## Verdict criteria (from the plan)

- If grasp (right hand) holds the bottle and lifts it, AND the left arm reaches
  the trigger instead of oscillating at the bottle -> **H1 confirmed**:
  proprioception was the missing anchor. Fix = always set `proprio_dim` for
  real-robot tasks.
- If symptoms persist -> H1 is not the primary cause; focus on H2 (rot6d
  normalization, see C3) and H3 (action-video decoupling).

## Why sim is unaffected (predicted)

Sim renders are deterministic and visually unambiguous, so the single input
frame is enough for the action expert to infer absolute pose. Real ROS images
are noisy/compressed/occluded, so without proprio the action expert has no
reliable absolute-pose anchor -> drift and oscillation. GR00T/pi0 always
condition on state, so they don't suffer this. (See the plan's H1 writeup.)
