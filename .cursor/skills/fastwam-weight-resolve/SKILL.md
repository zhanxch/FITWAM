---
name: fastwam-weight-resolve
description: >-
  Resolve FastWAM / DexJoCo / DEWO checkpoint, adapter, stats, and T5 cache
  paths when Glob returns empty because gitignore hides weights. Use before
  any train/eval/collect/CFG launch, when CKPT/BACKBONE_CKPT/STATS is missing,
  when Glob finds zero *.pt under checkpoints or runs, or when the user asks
  where weights live.
---

# FastWAM weight resolve (Shell only)

## Why Glob fails

Repo `.gitignore` includes (among others):

- `checkpoints` (symlink → `/data_all/xiangchengzhan/models/fastwam`)
- `runs`, `evaluate_results`, `data/*`, `third_party/`
- `*.pt` (and many other binary extensions)

Cursor **Glob / ripgrep skip ignored paths**. Empty Glob ≠ missing file.

**Rule:** discover and verify weights with the **Shell tool** (`ls` / `find` / `test -f`). Opening this SKILL.md with Read is only for instructions — it does not verify files. Rules/skills never disable Shell.

## Mandatory resolve checklist

```
Weight checklist:
- [ ] 1. Identify role: release S0 | DEWO adapter | backbone | stats | T5
- [ ] 2. Prefer tasks.py pin / known RUN_DIR (do not Glob)
- [ ] 3. Shell ls / test -f the candidate path
- [ ] 4. Export absolute CKPT / BACKBONE_CKPT / STATS for the launcher
- [ ] 5. Continue with fastwam-gpu-launch (host tmux + cuda verify)
```

## Canonical roots

| Role | Path |
|------|------|
| Workspace | `/data_all/xiangchengzhan/FastWAM` (`FASTWAM_ROOT`) |
| Release ckpt store | `$FASTWAM_ROOT/checkpoints` → `/data_all/xiangchengzhan/models/fastwam` |
| OPEN infer repo | sibling `FastWAM-infer-in-DexJoco` (`OPEN_REPO`) |
| Train / adapter runs | `$FASTWAM_ROOT/runs/` |

Do not treat the symlink spelling and the realpath as two different trees.

## Release S0 (open-source stack)

Pins live in `scripts/dewo_v2/tasks.py` → `TaskSpec.ckpt_rel` / `resolve_ckpt()`.

| Task | Local release weight |
|------|----------------------|
| `water_plant` | `checkpoints/dexjoco/water_plant_fastwam/weights/step_012500.pt` |
| `fold_glasses` | `checkpoints/dexjoco/fold_glasses_fastwam/weights/step_010000.pt` |
| `hammer_nail` | `checkpoints/dexjoco/hammer_nail_fastwam/weights/step_002500.pt` |
| `pick_bucket` | `checkpoints/dexjoco/pick_bucket_fastwam/weights/step_010000.pt` |
| `pinch_tongs` | `checkpoints/dexjoco/pinch_tongs_fastwam/weights/step_010000.pt` |
| mixed S0 backbone | `checkpoints/dexjoco/mixed_5task_fastwam/weights/step_055000.pt` |

Fallback if local missing: `OPEN_REPO/checkpoints/<task>/step_*.pt` (`open_ckpt_rel`).

Verify:

```bash
FASTWAM_ROOT="$(realpath -e /data_all/xiangchengzhan/FastWAM)"
ls -la "$FASTWAM_ROOT/checkpoints/dexjoco/${TASK}_fastwam/weights/"
test -f "$FASTWAM_ROOT/checkpoints/dexjoco/${TASK}_fastwam/weights/step_XXXXX.pt"
```

Or let the script resolve:

```bash
cd "$FASTWAM_ROOT"
python scripts/dewo_v2/tasks.py "$TASK"   # prints exports including CKPT / STATS / TEXT_EMB
```

## DEWO v5 / v6 adapter weights

Layout:

```text
runs/dexjoco_<task>_dewo_v5|v6/<run_id>/checkpoints/weights/step_XXXXXX.pt
```

Example:

```bash
ls -la runs/dexjoco_water_plant_dewo_v5/*/checkpoints/weights/
```

For CFG eval:

- `CKPT` = adapter `step_*.pt` under that run’s `checkpoints/weights/`
- `BACKBONE_CKPT` = release S0 (or mixed) weight from the table above
- Do **not** pass DeepSpeed `checkpoints/state/**` as `CKPT`

## Stats / T5 (OPEN artifacts)

| Piece | Path |
|-------|------|
| Per-task stats | `$OPEN_REPO/artifacts/<task>/dataset_stats.json` |
| Mixed stats | `$OPEN_REPO/artifacts/mixed_5task/dataset_stats.json` |
| Base T5 cache | `$OPEN_REPO/artifacts/<task>/*.t5_len128*.pt` |

Still Shell-verify; do not Glob `*.pt` under OPEN either if that tree is ignored.

## Shell recipes (use these, not Glob)

List release weights:

```bash
find -L "$FASTWAM_ROOT/checkpoints/dexjoco" -path '*/weights/*.pt' -type f | sort
```

List adapter steps for a run:

```bash
ls -1 "$RUN_DIR/checkpoints/weights"/step_*.pt
```

Assert before launch:

```bash
test -f "$CKPT" || { echo "missing CKPT=$CKPT"; exit 1; }
test -f "${BACKBONE_CKPT:-$CKPT}" || { echo "missing BACKBONE"; exit 1; }
test -f "$STATS" || { echo "missing STATS=$STATS"; exit 1; }
```

## Hard bans

| Ban | Why |
|-----|-----|
| `Glob **/*.pt` under `checkpoints` / `runs` / `evaluate_results` | gitignore → false “not found” |
| Inventing a new path after empty Glob | Weights already pinned in `tasks.py` |
| Using `checkpoints/state/**` as eval `CKPT` | Optimizer shards, not infer weights |
| Pairing release ckpt with local async `s0_bundle` packing | Wrong stack for open-source 224/z-score |

## After resolve

Hand off to [fastwam-gpu-launch](../fastwam-gpu-launch/SKILL.md):

1. Shell `ls` confirmed paths (this skill)
2. Host-tmux GPU check
3. `bash .cursor/skills/fastwam-gpu-launch/scripts/host_tmux_run.sh ...`
4. `verify_eval_cuda.sh` before claiming “running”

## Related

- Always-on: `.cursor/rules/weight-paths-shell-only.mdc`
- Eval stack pins: `.cursor/rules/dexjoco-eval-standard.mdc`
- Path roles: `AGENTS.md`
