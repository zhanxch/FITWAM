# Scripts

## Sim inference (three processes)

```
fastwam_sim_agent  --TCP-->  sim_server.py (Isaac, :5570)
                 --ZMQ-->  run_fastwam_server.py (GPU, :5560)
```

### Modules

| File | Role |
|------|------|
| `policy_io.py` | **Fixed** policy-server observation contract (training-aligned) |
| `run_fastwam_server.py` | Load checkpoint; ZMQ server; only accepts `policy_io` format |
| `fastwam_policy_server.py` | ZMQ transport |
| `sim_adapter.py` | **Sim-only** obs/action conversion (Ego Humanoid 50-dim, text cache) |
| `fastwam_sim_agent.py` | Closed-loop client; uses `sim_adapter` |
| `sim_server.py` | Isaac env RPC; env ops on **main thread** via `MainThreadRpcBridge` |
| `sim_protocol.py` | TCP JSON + main-thread bridge |
| `launch_fastwam_sim_eval.sh` | Launcher |

### Policy observation (do not change for sim)

Sent to `get_action` after `EgoHumanoidSimAdapter.sim_obs_to_policy_obs`:

- `input_image`: `[1, 3, H, W]` float in `[-1, 1]`
- `proprio`: `[50]` float (raw qpos; server normalizes like training)
- `prompt` **or** `context` + `context_mask`

### Sim server (Isaac machine)

```bash
${ISAACLAB_PATH}/isaaclab.sh -p scripts/sim_server.py \
  --task Humanoid-Stack-Can-v0 --enable_cameras --camera-key rgb \
  --host 0.0.0.0 --port 5570
```

### Policy server (GPU machine)

```bash
python scripts/run_fastwam_server.py \
  --run-dir runs/ego_vla_short_uncond_1cam_384_1e-4/2026-05-21_19-59-24 \
  --checkpoint runs/.../step_006445.pt \
  --device cuda:0 --host 0.0.0.0 --port 5560
```

### Agent

```bash
python scripts/fastwam_sim_agent.py \
  --policy-host 127.0.0.1 --policy-port 5560 \
  --sim-host <sim-ip> --sim-port 5570 \
  --save-video
```

With `--save-video`, ego RGB is written to `logs/fastwam_sim_eval/<timestamp>/episode_001_success0.mp4`.

Agent exits **without** stopping the policy server (no re-load). Use `--kill-policy-on-exit` only if you want to shut the server down.

Policy server returns `{"error": ...}` on failed requests and **keeps listening** for the next call.

Copy `sim_server.py`, `sim_protocol.py` to the Benchmark repo on the sim machine when deploying there.

If you see `'StackCanEnv' object has no attribute 'scene'`, the sim machine is on an old
`sim_server.py` or the Kit app is not ticking (`simulation_app.update()` in the main loop).

## Training & data

`train.py`, `prepare_*.py`, `precompute_text_embeds.py`, `train_zero*.sh`, etc.
