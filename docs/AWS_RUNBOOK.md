# FITWAM AWS Runbook

Last verified: 2026-07-10. Treat instance state, public IP, GPU occupancy, and Spot status as live values.

## Project identity

- Research project: **FITWAM** (`Failure-Improvement Tactile WAM`).
- Repository: `Interaction-centricWAM`.
- Upstream backbone and Python package: `FastWAM` / `fastwam`.
- Current offline method working name: `FITWAM-Steer`.
- Method source of truth: [`OFFLINE_SELF_IMPROVING_PLAN.md`](./OFFLINE_SELF_IMPROVING_PLAN.md).

The current offline task learns failure-contrastive steer tokens from successful and failed trajectories. Failed actions remain excluded from behavior-cloning loss. The minimum evidence is a controlled gain over success-only FastWAM and the failure-data baseline without outcome-label leakage.

Current status: SSH, the `fitwam` and `dexjoco` environments, the 5-task success dataset, text cache, and base weights are present. The GPU lane is **not currently training-ready**: `nvidia-smi` is missing and PyTorch reports CUDA error 802 (`system not yet initialized`). The failure datasets also require staging and manifest validation. `FITWAM-Steer` remains a design in the plan document; its architecture module and controlled runs have not yet been implemented.

## Server and connection

```text
AWS region:     us-east-1
Instance ID:    i-0a14bfdef63fad1b6
Instance type:  p4de.24xlarge Spot
GPU:            8 x NVIDIA A100-SXM4-80GB
SSH alias:      fitwam-aws
Remote user:    zhaoyc
SSH port:       22
Project root:   /data_all/zhaoyc/Summer2/FITWAM
```

Connect:

```bash
ssh fitwam-aws
conda activate fitwam
cd /data_all/zhaoyc/Summer2/FITWAM
```

VS Code Remote SSH:

1. Choose `Remote-SSH: Connect to Host...` from the lower-left remote menu.
2. Select `fitwam-aws` and choose Linux if prompted.
3. Open `/data_all/zhaoyc/Summer2/FITWAM`.
4. Select `/home/zhaoyc/miniforge3/envs/fitwam/bin/python`.

Authentication uses the private key configured in `/Users/yiche/.ssh/config`. The `zhaoyc` password is locked; a password prompt means the SSH alias/key was not used correctly.

## Storage boundaries

- `/data_all/zhaoyc` is the private personal area. Its traversal permission is restricted to `zhaoyc`.
- The `zhaoyc` shell uses `umask 077`, and the project sync helper strips group/other permissions from synchronized files.
- `/data_all/shared` and `/data_all/share` are the team shared area. `sharegroup` members may write there and other server users may read it.
- Keep personal code, logs, temporary checkpoints, and credentials under `/data_all/zhaoyc` or `$HOME`.
- Shared datasets and model weights may remain in team-owned locations and be consumed through symlinks from the project.

Current shared read-only sources:

```text
data/dexjoco             -> /home/zhanxch/FITWAM/data/dexjoco
data/text_embeds_cache   -> /home/zhanxch/FITWAM/data/text_embeds_cache
checkpoints/Wan-AI       -> /home/zhanxch/FITWAM/checkpoints/Wan-AI
checkpoints/ActionDiT... -> /home/zhanxch/FITWAM/checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt
```

Do not modify a symlink target owned by another user without coordination.

## Environments

### Model environment

```bash
conda activate fitwam
```

- Python 3.10
- PyTorch `2.7.1+cu128`
- torchvision `0.22.1+cu128`
- repository installed editable from `/data_all/zhaoyc/Summer2/FITWAM`
- dependencies resolved from `pyproject.toml`
- W&B entity and project supplied by Conda environment variables

The older `fastwam` Conda environment is retained only as rollback context. Current scripts and VS Code must use `fitwam`.

### DexJoCo evaluation environment

```bash
conda activate dexjoco
export MUJOCO_GL=egl
```

The policy server/model runs in `fitwam`; the simulator client runs in `dexjoco`. Keep these environments separate because their Python and numerical dependencies differ.

## Smoke-test history and current health

The following checks passed earlier on 2026-07-10, before the current GPU regression:

- `fitwam`: `pip check`, editable `fastwam` import from the private project root, PyTorch CUDA 12.8, and a BF16 matrix multiply on an idle A100.
- `dexjoco`: `pip check`, DexJoCo/OpenPI client imports, MuJoCo 3.4, and a 64 x 64 EGL render.
- Multi-task data loader: one sample yielded video `(3, 9, 384, 768)`, action `(32, 22)`, proprio `(33, 23)`, context `(128, 4096)`, and context mask `(128,)`.
- W&B setup run: <https://wandb.ai/yicheng132024-southern-university-of-science-technology/fitwam/runs/4natl3fp>.

Current live check on 2026-07-10:

- SSH, project import, package checks, DexJoCo imports, and W&B authentication pass.
- `data/dexjoco/multi-task-5/meta/info.json` reports 500 episodes, 186,313 frames, 5 tasks, 1,000 videos, action dim 22, and state dim 23.
- `nvidia-smi` returns `command not found`.
- PyTorch reports 8 enumerated devices but `torch.cuda.is_available() == False`, with CUDA error 802.
- No training or rollout may start until the GPU checks in the next section pass again.

## W&B

```text
Entity:  yicheng132024-southern-university-of-science-technology
Project: fitwam
Setup run: https://wandb.ai/yicheng132024-southern-university-of-science-technology/fitwam/runs/4natl3fp
```

Authentication is stored privately in `/home/zhaoyc/.netrc` and the local Codex secret store. Never put the API key in this repository, shell history, a launch command, or chat.

Every serious run must initialize W&B at process start, preserve local JSONL/CSV curves, run periodic validation, save the best checkpoint, and record an explicit stop reason.

The base configs read `WANDB_ENTITY` and `WANDB_PROJECT`. Historical task configs may intentionally override them with `fast-wam`; for a new FITWAM experiment, verify the resolved Hydra config or pass `wandb.workspace=${WANDB_ENTITY} wandb.project=${WANDB_PROJECT}` explicitly.

## Before a run

```bash
bash /Users/yiche/.codex/skills/deeplearning-gpu-burst/scripts/check_fitwam_aws.sh
ssh fitwam-aws nvidia-smi
```

Then record:

- Git commit and uncommitted diff.
- Training config and exact launch command.
- Dataset, text-cache, normalization-statistics, and resume-checkpoint paths.
- Allocated GPU IDs after checking process ownership.
- W&B run URL and local output directory.

Treat any failure of `nvidia-smi`, `torch.cuda.is_available()`, a 4-rank collective smoke, or DexJoCo EGL as a hard stop. Do not infer GPU readiness from `/dev/nvidia*` nodes or `torch.cuda.device_count()` alone.

This is a shared Spot instance. Never kill another user's process or stop, reboot, or terminate the instance without explicit team confirmation.

## Security

- Port 22 is currently reachable from `0.0.0.0/0`; reachability does not bypass authentication.
- `zhaoyc` uses one personal authorized key and has no usable password.
- The emergency `WBC.pem` logs in as `ubuntu` and is shared access; possession of that file grants server access.
- Server-wide password authentication remains enabled for other team accounts.
- Keep private keys under `/Users/yiche/.codex/secrets/` with restrictive permissions.

Future hardening should use per-user public keys, retire shared private keys, and restrict SSH ingress to team IP/VPN after coordinating with all users.

## Recovery

If SSH stops working after an instance stop/start, query the current IP with AWS profile `fastwam-aws`, update `HostName` for both aliases in `/Users/yiche/.ssh/config`, and verify the host key out of band before replacing the pinned known-host entry.

The root EBS volume is Spot-attached, unencrypted, and configured to delete on termination. Sync irreplaceable checkpoints and reports elsewhere promptly.
