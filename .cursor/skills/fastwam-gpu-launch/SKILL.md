---
name: fastwam-gpu-launch
description: >-
  Launch FastWAM / DexJoCo GPU train, eval, smoke, and CFG 4×50 jobs without
  Cursor-sandbox CUDA failures. Use when starting or restarting any GPU job
  (smoke test, eval, 4x50, dewo train/collect, inference server), when
  nvidia-smi fails in the agent shell, when server.log shows CPU fallback, or
  when the user asks to squeeze a test onto busy GPUs.
---

# FastWAM GPU launch (no sandbox)

## Why this exists

Cursor Shell often runs in a **sandbox that breaks NVIDIA**. Symptoms:

- `nvidia-smi`: *couldn't communicate with the NVIDIA driver*
- `server.log`: `CUDA unavailable; falling back to CPU`
- Job “starts” but never takes GPU memory; eval is uselessly slow/wrong

**Never** start train/eval/smoke with Cursor Shell `nohup` / `setsid` / background `&` and hope it got CUDA. Always use a **host tmux** session that already lives outside the sandbox.

## Mandatory workflow

Copy and track:

```
Launch checklist:
- [ ] 0. Resolve CKPT / BACKBONE_CKPT / STATS via Shell (fastwam-weight-resolve) — never Glob *.pt
- [ ] 1. Write/reuse launcher (session logs/ or scripts/dewo_v2|dexjoco with env knobs)
- [ ] 2. Start via host tmux (script below), not Cursor Shell background
- [ ] 3. Confirm server.log says cuda (not CPU fallback)
- [ ] 4. Confirm target GPUs gained memory (host-tmux nvidia-smi)
- [ ] 5. Only then tell the user it is running
```

Weights under `checkpoints/`, `runs/`, and `*.pt` are **gitignored**. Empty Glob does not mean missing files — see [fastwam-weight-resolve](../fastwam-weight-resolve/SKILL.md).

### 1. Prefer existing launchers

- Official 4×50 CFG: `scripts/dewo_v2/eval_cfg_official_4x50.sh` (`TASK`, `GPUS`, `RUN_DIR`, `CKPT`, `WAIT_IDLE=0` if squeezing)
- Opensource 4×50: `scripts/dexjoco/eval_opensource_4x50.sh`
- Smoke: reuse prior `logs/launch_smoke20_*.sh` pattern (seeds 0..19, label NON_STANDARD)
- Do **not** bake dated experiment/`GPUS` into a new permanent repo launcher

### 2. Start on host tmux

Default socket (keepalive already present on this machine):

```bash
SOCK="${FASTWAM_TMUX_SOCK:-/tmp/fastwam_dewo_v2_pm1p5.sock}"
```

Launch:

```bash
bash .cursor/skills/fastwam-gpu-launch/scripts/host_tmux_run.sh \
  --session smoke_v3_500 \
  --cmd "bash /path/to/launcher.sh"
```

Or equivalent:

```bash
tmux -S "$SOCK" new-session -d -s <session> '<command>'
```

If the socket is missing, create a keepalive **once** from a host context (not Cursor sandbox):

```bash
tmux -S /tmp/fastwam_dewo_v2_pm1p5.sock new-session -d -s _keepalive 'sleep infinity'
```

### 3. Verify before claiming success

```bash
bash .cursor/skills/fastwam-gpu-launch/scripts/verify_eval_cuda.sh <OUT_DIR>
```

Must see:

- `Loading FastWAM model on cuda:` (or equivalent) in `**/server.log`
- **No** `falling back to CPU` / `CUDA unavailable`
- Host-tmux `nvidia-smi` memory on target GPUs **up** vs pre-launch (co-tenant OK; our process must appear)

If any check fails: kill that job tree, discard that `OUT` dir, relaunch via host tmux. Do not “wait and see” on CPU servers.

### 4. Query GPU / logs from host tmux when unsure

If Cursor Shell `nvidia-smi` fails, run it **inside** host tmux:

```bash
tmux -S "$SOCK" new-session -d -s _gpu_chk \
  'nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv > /tmp/fw_gpu_chk.csv'
sleep 1; cat /tmp/fw_gpu_chk.csv
```

## Hard bans (repeat failures)

| Ban | Why |
|-----|-----|
| `nohup` / `setsid` GPU jobs from Cursor Shell | Children inherit broken CUDA |
| `nvidia-smi \| head` under `set -o pipefail` | SIGPIPE → script exits mid-launch |
| `pkill -f` pattern that matches the **current** agent cmdline | Self-kill; command aborts silently |
| Declare “running” without cuda + mem checks | False success, wasted user time |

Safer kill: match a **unique OUT path** or known PIDs from `pgrep -af`, never a short substring that appears in the agent’s own command text.

## Squeeze onto busy GPUs

- Check free MiB via host-tmux `nvidia-smi` (FastWAM infer server often ~12–16G)
- Set `WAIT_IDLE=0` when co-tenancy is intentional
- Use distinct `--base-port` ranges
- Label smoke as **NON_STANDARD**; official score still needs 4×50 seeds 0–49 × 4

## Related

- Weight / ckpt resolve (Shell only): `.cursor/skills/fastwam-weight-resolve/SKILL.md`
- Always-on reminder: `.cursor/rules/gpu-launch-no-sandbox.mdc`
- Weight ignore trap: `.cursor/rules/weight-paths-shell-only.mdc`
- DexJoCo eval protocol: `.cursor/rules/dexjoco-eval-standard.mdc`
