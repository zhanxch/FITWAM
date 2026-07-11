# FITWAM GPU Runbook

This runbook records the portable workflow only. Hostnames, account names, instance identifiers, IP addresses, credentials, private paths, and W&B entities belong in local environment or SSH configuration, never in this repository.

## Local configuration

Set these values outside Git:

```bash
export FITWAM_SSH_ALIAS=<ssh-alias>
export FITWAM_PROJECT_ROOT=<remote-project-root>
export FITWAM_ARTIFACT_ROOT=<shared-artifact-root>
export WANDB_ENTITY=<wandb-entity>
export WANDB_PROJECT=fitwam
```

Use an SSH alias backed by a private local key. Do not put passwords, private keys, AWS resource identifiers, public IPs, or W&B API keys in tracked files or command-line arguments.

## Connect

```bash
ssh "${FITWAM_SSH_ALIAS}"
conda activate fitwam
cd "${FITWAM_PROJECT_ROOT}"
```

VS Code Remote SSH should use the same alias. Select the `fitwam` interpreter from the remote environment rather than recording its account-specific absolute path here.

## Storage

- Keep personal code, logs, temporary checkpoints, and credentials in the account's private storage.
- Treat shared datasets and checkpoints as read-only unless ownership explicitly permits writes.
- Consume team-owned assets through environment variables or symlinks.
- Never commit local mount points, account names, credentials, or private access instructions.

## Environments

Training and policy serving use:

```bash
conda activate fitwam
```

DexJoCo evaluation uses:

```bash
conda activate dexjoco
export MUJOCO_GL=egl
```

Keep the policy server and simulator environments separate because their Python and numerical dependencies differ.

## Before a run

Verify:

```bash
ssh "${FITWAM_SSH_ALIAS}" nvidia-smi
```

Then record in the private run metadata:

- Git commit and uncommitted diff hash;
- resolved training config and launch command;
- EveRobot manifest hash and dataset fingerprint;
- allocated GPU IDs after checking process ownership;
- W&B run URL and local output directory;
- validation schedule, best-checkpoint rule, and stop reason.

Treat failure of `nvidia-smi`, `torch.cuda.is_available()`, a distributed collective smoke test, or DexJoCo EGL as a hard stop. Never kill another user's process or stop shared infrastructure without explicit team confirmation.

## W&B

Every serious run initializes W&B at process start and also preserves local JSONL/CSV curves. Credentials stay in the user's private credential store. The repository may refer only to `WANDB_ENTITY` and `WANDB_PROJECT`; it must not contain an entity name, API key, or personal run URL.

## Security

- Use per-user public keys and restrictive permissions for private material.
- Keep cloud profiles, SSH config, known-host entries, and secret files outside the repository.
- Do not document live ingress rules or shared emergency credentials in Git.
- Query live cloud state through the authorized local tooling when needed.
