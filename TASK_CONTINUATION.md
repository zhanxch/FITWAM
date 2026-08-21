# Task Continuation

## Checkpoint 1 - 2026-08-13

### Goal

Improve the Fold Glasses policy's closed-loop rollout success rate by learning
from deployment rollouts without flooding training with redundant successful
episodes. Same-seed success/failure comparison is a diagnostic and data-selection
mechanism, not the objective itself. The final method must be accepted or rejected
by held-out-seed closed-loop success rate.

### User Request Reference

- Window ID: unavailable in the current tool interface.
- Item ID: unavailable in the current tool interface.
- Request: design and implement a better method than blindly using all successful
  rollouts; use same-seed success/failure evidence when useful, exclude all-success
  seeds, decide how to treat all-failure seeds, and prioritize the goal over the
  user's initial proposal. The prior 0-3 GPU training process is already stopped.
- Continuation request: checkpoint after each phase and continue across automatic
  context transitions without asking the user to open a new Codex window.

### Decisions

- Do not revive or modify the stopped 0-3 GPU training run.
- Exclude all-success seeds from rollout training data.
- Exclude all-failure seeds from the main intervention-training set because no
  same-seed successful counterfactual exists. Keep them as a hard-transfer
  evaluation slice. Current all-failure seeds are 10106 and 10119.
- Reject direct absolute-frame action-distance selection. It selected frame 0 in
  34/35 mixed seeds, mostly measuring diffusion sampling noise.
- Select the earliest stable, plausibly recoverable branch point, not the largest
  outcome-correlated difference.
- Required evidence gates: monotone task-phase alignment; failure state/context
  still supported by successful trajectories; successful actions locally
  consistent; persistent success/failure action separation; future visual/state
  divergence after action divergence.
- Use successful windows for action supervision. Failure windows remain auxiliary
  context with action loss disabled.
- Pairing is used for event discovery and audit. Do not add an unvalidated
  contrastive/pair loss merely because paired samples exist.
- Split rollout seeds as groups. Never allow episodes from one seed to cross
  train/validation.
- Control expert replay explicitly so stride-1 expert windows do not overwhelm
  selected rollout windows.
- The proprioceptive state contains only end-effector pose and hand joints; it has
  no object state. Visual evidence is therefore required for credible phase and
  progress alignment.

### Done / Files Changed

- Audited the 200-rollout dataset and existing rule-based manifest rewriter.
- Confirmed dataset composition: 50 seeds x 4 attempts, 35 mixed, 13 all-success,
  2 all-failure; 22-D action, 23-D proprioception, 30 FPS, 33-frame train windows.
- Confirmed the current split is outcome-stratified by episode and leaks seeds
  across splits.
- Confirmed the current role-balanced sampler does not guarantee same-seed pairs.
- Ran an action/state prototype and rejected absolute-time action difference due
  to frame-0 noise selection.
- Started the Idea-Spark bottleneck audit. Retrieval connectors ran but returned
  zero papers; the empty literature table was written truthfully. This is degraded
  grounding and will not override local evidence.
- Added this checkpoint file: `TASK_CONTINUATION.md`.

### Key Result Paths

- Rollouts: `data/fold_glasses_opensource_s0_collect_4x50_20260812_112113/rollout_raw_200`
- Outcomes: `data/fold_glasses_opensource_s0_collect_4x50_20260812_112113/rollout_raw_200/meta/episode_outcomes.jsonl`
- Width-jump ledger: `data/fold_glasses_opensource_s0_collect_4x50_20260812_112113/failure_events_pm1p5_minb1/failure_events.jsonl`
- Existing manifest: `data/fold_glasses_dewo_v2_opensource_20260812_195749/eve_v02/manifests/offline_b1_jump_fast.json`
- Existing rule rewriter: `scripts/fold_glasses/rewrite_manifest_seedpair_success.py`
- Existing seed-leaking split report: `data/fold_glasses_dewo_v2_opensource_20260812_195749/eve_v02/splits/episode_splits.report.json`
- Idea-Spark run: `ideaspark_run/seed-conditioned-critical-event`
- Empty literature result: `ideaspark_run/seed-conditioned-critical-event/phase0/lit_results.json`
- Early outcome probe evidence: `results/s0_early_frame_60_70_action_latent_probe_20260806/world_vs_action_relevance/summary.json`

### Next 1-3 Steps

1. Finish the degraded bottleneck audit and pin down falsification criteria.
2. Inspect rollout control cadence, videos, and reusable visual features; define a
   tractable phase-alignment and causal-order scoring algorithm.
3. Implement a standalone selector with audit JSON/plots and synthetic tests.

## Checkpoint 2 - 2026-08-13

### Goal

Turn the same-seed rollout evidence into a falsifiable intervention-data
selection method whose only acceptance criterion is improved held-out closed-loop
Fold Glasses success rate.

### User Request Reference

- Window ID: unavailable in the current tool interface.
- Item ID: unavailable in the current tool interface.
- Same request as Checkpoint 1, plus the explicit instruction to persist through
  context transitions and checkpoint every completed phase.

### Decisions

- Score policy decisions at the 24-frame replan cadence, not per frame. Each
  diffusion query emits 32 actions but only the first 24 are executed; frame-level
  differences inside a chunk are largely interpolated/redundant.
- The intervention primitive is: pre-action context at replan k, the next executed
  action block, and later context/progress. Action divergence must precede later
  visual/proprioceptive divergence.
- Use visual features for phase and progress because proprioception omits object
  state. Do not use action to define phase alignment.
- A local cached CLIP vision encoder is available. Use it (or a documented lighter
  fallback) to build reusable replan-level front/wrist embeddings.
- Failure rollout pairing alone is not a learning signal: `pair_weight` is exposed
  by the dataset but unused by the model/trainer loss. Do not claim pair-aware
  batches implement a contrastive objective.
- The model does have an explicit binary outcome token. Failure action loss is
  disabled while video loss remains active and outcome-conditioned. Treat failure
  video training as an ablation, not the core mechanism. Core training signal is
  successful action supervision at selected branch points.
- Idea-Spark terminated with `do_not_generate` because all three available
  literature connectors returned zero papers. This degraded audit is complete and
  must not be reopened or used as evidence for novelty.

### Done / Files Changed

- Completed Idea-Spark Phase 0 full-text gate with an empty cache and Phase 1
  degraded routing.
- Verified collection policy cadence and deterministic diffusion noise seeds from
  `scripts/fold_glasses/collect_opensource_4x50.py`.
- Verified local CLIP ViT-L/14-336 weights load offline.
- Verified failure samples have an outcome token, disabled action loss, and active
  video loss; verified pair/event weights currently do not weight training loss.
- Updated `TASK_CONTINUATION.md` with this checkpoint.

### Key Result Paths

- Degraded bottleneck result:
  `ideaspark_run/seed-conditioned-critical-event/phase1/phase1_output.json`
- Degraded stop explanation:
  `ideaspark_run/seed-conditioned-critical-event/phase1/do_not_generate.md`
- Collection cadence/config:
  `data/fold_glasses_opensource_s0_collect_4x50_20260812_112113/config.json`
- Collection implementation: `scripts/fold_glasses/collect_opensource_4x50.py`
- Outcome conditioning/action masking: `src/fastwam/models/wan22/fastwam.py`
- Manifest expansion/roles: `src/fastwam/datasets/eve/manifest_dataset.py`

### Next 1-3 Steps

1. Extract reusable replan-level visual/proprio/action block features for all 200
   rollouts and characterize same-seed alignment quality.
2. Implement and calibrate earliest-persistent-branch selection with rejection
   gates and exact small-sample diagnostics.
3. Add synthetic tests and generate audit plots before rewriting any manifest.

## Checkpoint 3 - 2026-08-13

### Goal

Identify learnable pre-divergence decision events where visually and
proprioceptively similar same-seed rollouts take different action branches and
only later diverge in task progress. Move policy probability mass from the
failure branch to the successful branch, and accept the method only if it raises
held-out-seed closed-loop success.

### User Request Reference

- Window ID: unavailable in the current tool interface.
- Item ID: unavailable in the current tool interface.
- Clarification: the user expects a visually similar region with a broad or
  multimodal action distribution containing both failure and success paths,
  followed by visual divergence, and asks whether that is the policy-useful
  event.

### Decisions

- Low success width is not a required gate. A broad success distribution may
  contain several valid modes; blindly narrowing it can destroy valid behavior.
- A failure width rise is only a cheap candidate anchor. Width alone neither
  identifies the successful mode nor establishes causality.
- The event anchor is the last supported common context before outcome-relevant
  divergence. The supervised successful action block begins there; frames after
  obvious visual divergence are evidence of consequence, not the event input.
- Separate three evidence levels in all outputs: observational candidate,
  learnable successful action support, and causal intervention evidence. Never
  collapse these into one self-validating score.
- Required observational order is: current visual/proprio context supported by
  same-seed successes; successful and failed executed action blocks differ in a
  persistent, structured way; future visual/progress divergence occurs after
  that action difference. Temporal ordering is necessary but not causal proof.
- Learnability means successful executed blocks agree enough to define a target
  support set and the policy assigns insufficient mass to that set at the shared
  failure context. It does not mean every success sample or policy draw is tight.
- The decisive pre-training falsification is action-block intervention/replay:
  replace the failure block at the common context with a matched successful
  block and test whether future progress/outcome improves versus matched random,
  late, and same-distance controls.
- The completed degraded Idea-Spark run remains terminal (`do_not_generate`);
  its workflow informed the falsification discipline but is not reopened and is
  not evidence for novelty.

### Done / Files Changed

- Reframed the proposed event from `failure width high + success width low` to a
  pre-divergence branching event with an explicit downstream causal test.
- Audited the three implementation drafts at a structural level; they exist but
  have not yet passed synthetic or real-data validation.
- Updated `TASK_CONTINUATION.md` with this checkpoint.

### Key Result Paths

- Feature extraction draft:
  `scripts/fold_glasses/extract_seedpair_replan_features.py`
- Candidate alignment draft:
  `scripts/fold_glasses/build_seedpair_width_candidates.py`
- Policy distribution probe draft:
  `scripts/fold_glasses/probe_seedpair_action_distributions.py`
- Existing failure widths:
  `data/fold_glasses_opensource_s0_collect_4x50_20260812_112113/failure_width_curves_full/npz`
- Existing failure jump events:
  `data/fold_glasses_opensource_s0_collect_4x50_20260812_112113/failure_events_pm1p5_minb1/failure_events.jsonl`

### Next 1-3 Steps

1. Add synthetic tests for monotone alignment, seed exclusions, head-noise
   rejection, and action-before-future-visual-divergence ordering; repair the
   drafts until they pass.
2. Run a bounded real-data feature extraction/candidate build and inspect aligned
   contexts and future divergence for false positives.
3. Complete the action-support selector and intervention protocol before any new
   training manifest or GPU training is launched.

## Checkpoint 4 - 2026-08-13

### Goal

Validate the pre-divergence event definition mechanically before spending GPU
time on real rollout probing or training.

### User Request Reference

- Window ID: unavailable in the current tool interface.
- Item ID: unavailable in the current tool interface.
- Same request and clarification as Checkpoint 3.

### Decisions

- Calibration uses successful rollouts from mixed seeds only. All-success seeds
  are excluded not only from training but also from selector threshold fitting.
- Future visual/proprio divergence must exceed successful-support variation for
  at least two consecutive replan horizons. A transient peak is rejected.
- The exact anchor-frame context distance, rather than a backward local average,
  determines whether observations are still similar at the decision point.
- Policy samples and actual action blocks are compared in the checkpoint's exact
  z-score training space. Robot-unit actions are retained separately for replay.
- Existing width values in robot units remain candidate anchors only and are not
  directly comparable to the normalized action-support metrics.

### Done / Files Changed

- Updated `scripts/fold_glasses/build_seedpair_width_candidates.py` with
  mixed-seed-only calibration and persistent future-divergence logic.
- Updated `scripts/fold_glasses/probe_seedpair_action_distributions.py` to store
  both normalized and robot-unit action blocks and calculate widths only in the
  normalized training space.
- Added `tests/test_fold_glasses_seedpair_events.py` covering monotone alignment,
  all-success/all-failure exclusion, persistent versus transient future
  divergence, head-noise rejection, normalized action conversion, and probe
  eligibility.
- Verification: `python -m pytest -q tests/test_fold_glasses_seedpair_events.py`
  completed with 6 passed.

### Key Result Paths

- Candidate implementation:
  `scripts/fold_glasses/build_seedpair_width_candidates.py`
- Distribution probe implementation:
  `scripts/fold_glasses/probe_seedpair_action_distributions.py`
- Synthetic tests: `tests/test_fold_glasses_seedpair_events.py`

### Next 1-3 Steps

1. Run feature extraction and candidate construction for seed 10086's four real
   rollouts, then render/inspect anchor and future contexts.
2. Repair real-data alignment failures and expand extraction to the 200-rollout
   population only after the bounded audit is credible.
3. Implement support-mass scoring and executable action-block interventions.

## Checkpoint 5 - 2026-08-13

### Goal

Determine on real rollout data whether a failure width jump actually denotes a
shared pre-divergence context whose policy distribution contains both successful
and failed action branches.

### User Request Reference

- Window ID: unavailable in the current tool interface.
- Item ID: unavailable in the current tool interface.
- Same request and clarification as Checkpoint 3: test whether a visually similar,
  action-wide region followed by visual divergence is useful for policy learning.

### Decisions

- Split candidate status into `context_probe_eligible` and
  `observational_event_supported`. A shared current context may proceed to action
  probing even when passive future CLIP divergence is weak; only intervention can
  establish causality.
- Do not interpret similar scalar width as a shared multimodal action support.
  Distribution geometry and support mass must be measured explicitly.
- Do not train on a success block merely because its aligned success observation
  looks similar. The block must either be executable from the failure prefix or a
  successful counterfactual rollout must be generated from that exact prefix.
- Before any intervention result is trusted, factual action replay must reproduce
  the recorded failure prefix/outcome. The merged collection summary preserves
  seed/repeat/source mappings needed to reconstruct attempts.

### Done / Files Changed

- Fixed CLIP extraction compatibility by disabling the unused Transformers
  TensorFlow backend; used the repository `fastwam` environment to avoid the
  base environment's incompatible optional flash-attn install.
- Extracted dual-camera CLIP, proprio, and executed 24-step blocks for seed 10086
  episodes 0-3 (2 success, 2 failure).
- Added bounded `--seeds` support to the candidate builder.
- Updated candidate state semantics so passive future divergence is evidence, not
  a prerequisite for the expensive action probe.
- Built the seed 10086 audit: ep0/f240 is rejected because its current context is
  already outside success support; ep3/f144 is a shared-context probe candidate
  but lacks persistent passive future divergence.
- Completed paired-noise policy probing at ep1/f144, ep2/f144, and ep3/f144 with
  8 diffusion samples each, storing normalized and robot-unit 24-step blocks.
- Real result: normalized first-step widths are similar (success1 0.1245,
  success2 0.1169, failure 0.1137). Failure-context samples stay close to the
  factual failure block (block RMS 0.023-0.064) and far from either success block
  (0.232-0.288). The two actual success blocks disagree strongly (RMS 0.486).
  Thus this candidate does not currently show a broad failure distribution that
  contains an agreed successful branch.
- Re-ran the synthetic suite after status separation: 6 passed.

### Key Result Paths

- Real features:
  `data/fold_glasses_opensource_s0_collect_4x50_20260812_112113/seedpair_replan_features_clip_v1`
- Seed 10086 candidate audit:
  `data/fold_glasses_opensource_s0_collect_4x50_20260812_112113/seedpair_candidate_audit_seed10086_v1`
- Seed 10086 action probe:
  `data/fold_glasses_opensource_s0_collect_4x50_20260812_112113/seedpair_action_probe_seed10086_v1`
- Attempt repeat/source mapping:
  `data/fold_glasses_opensource_s0_collect_4x50_20260812_112113/rollout_raw_200/collection_summary.json`

### Next 1-3 Steps

1. Implement an explicit action-support scorer that reports success-consensus,
   factual branch separation, and policy mass near successful support rather than
   reducing distributions to width.
2. Implement and validate factual prefix replay, then run success-block and
   matched-control interventions from the exact ep3/f144 MuJoCo state.
3. Use the intervention result to decide whether to broaden candidate discovery
   beyond existing width jumps before extracting all 200 rollout features.

## Checkpoint 6 - 2026-08-13

### Goal

Establish a reproducible action-support metric and validate that the simulator can
recreate the exact factual failure state required for causal block intervention.

### User Request Reference

- Window ID: unavailable in the current tool interface.
- Item ID: unavailable in the current tool interface.
- Same request and clarification as Checkpoint 3.

### Decisions

- Define success support as a union of per-success-mode balls in normalized
  action-block space. Do not average distinct successful modes into one target.
- Estimate each mode's radius from policy samples at that success observation to
  the executed successful block; report binomial Wilson intervals because K=8 is
  small.
- `training_eligible` is always false after geometric scoring alone. A generated
  exact-prefix successful intervention is required before an action block becomes
  supervised rollout data.
- The collector reused one environment across repeats, so factual recreation must
  replay the reset sequence through the recorded repeat number. The merged
  `collection_summary.json` is the authoritative seed/repeat mapping.
- Sparse rendering is valid for replay acceleration: image rendering does not
  participate in MuJoCo physics, while state sensors and success predicates are
  still evaluated every step.

### Done / Files Changed

- Added `scripts/fold_glasses/score_seedpair_action_support.py` and tests for
  mixed sampled branches and multiple successful modes.
- Formal seed 10086 score confirms 0/8 failure-context samples enter either
  successful support mode; Wilson 95% upper bound is 0.324 because K is small.
  The nearest-success radius ratio is 3.00-3.29, while samples remain close to the
  factual failure block.
- Added `scripts/fold_glasses/validate_factual_replay.py` with seed/repeat
  reconstruction, sparse visual audits, task-progress metrics, and hard gates.
- Short ep3 replay passed at f0/f144/f168: 23-D state max error exactly 0 and
  front/wrist MAE 1.37-2.04 on the 0-255 scale.
- Full ep3 replay passed through 1200 steps: reproduced failure outcome, no early
  termination, f1199 state max error 0, image MAE 1.13-1.89. Final hinge values
  are 0.575 and 1.260, confirming failure because both hinges did not exceed 1.1.
- Synthetic suite now has 8 passing tests.

### Key Result Paths

- Support scorer: `scripts/fold_glasses/score_seedpair_action_support.py`
- Formal support result:
  `data/fold_glasses_opensource_s0_collect_4x50_20260812_112113/seedpair_action_support_seed10086_v1`
- Factual replay validator:
  `scripts/fold_glasses/validate_factual_replay.py`
- Short factual replay:
  `data/fold_glasses_opensource_s0_collect_4x50_20260812_112113/factual_replay_ep3_v2_short/factual_replay.json`
- Full factual replay:
  `data/fold_glasses_opensource_s0_collect_4x50_20260812_112113/factual_replay_ep3_v2_full/factual_replay.json`

### Next 1-3 Steps

1. Execute block-only interventions from exact ep3/f144 using factual, two aligned
   success blocks, eight failure-context policy samples, and phase-shift controls.
2. For any branch that improves task-relevant hinge/placement progress beyond
   controls, resume the baseline policy closed-loop from the intervened state and
   test final outcome.
3. Use results to decide whether width-jump anchors are useful or candidate search
   must scan for earlier action-separation points independent of width jumps.

## Checkpoint 7 - 2026-08-13

### Goal

Falsify or validate seed 10086 episode 3 frame 144 as a policy-useful branching
event, then use that result to choose the next candidate-discovery strategy.

### User Request Reference

- Window ID: unavailable in the current tool interface.
- Item ID: unavailable in the current tool interface.
- Clarification: a useful event may be a visually shared context with a broad or
  multimodal action distribution containing success and failure branches, with
  visual/task-state divergence only afterward. Determine whether learning that
  event can improve policy success rather than assuming the heuristic is valid.

### Decisions

- The proposed temporal pattern is a useful candidate definition but is not a
  training label. Width can arise from harmless multimodality, diffusion noise,
  or phase mismatch, and later visual divergence can be correlated rather than
  caused by the selected action block.
- A policy-useful event requires branch-specific causal evidence: from the exact
  failure prefix, a success-supported action block must improve task progress
  beyond phase-shift and equal-RMS controls, and a closed-loop continuation must
  ultimately succeed before the generated rollout enters training.
- Reject seed 10086 episode 3 frame 144. Do not generate training data from it or
  spend additional closed-loop continuation compute on this candidate.
- Broaden discovery beyond failure width jumps. Search all mixed seeds directly
  for the earliest shared visual/proprio context where executed success/failure
  action blocks separate before future visual/task-state divergence. Retain
  failure-policy width only as a diagnostic and prioritization feature.
- Keep all-success seeds excluded and all-failure seeds as hard-transfer tests.
  Keep the old GPU 0-3 training stopped.

### Done / Files Changed

- Executed 17/17 exact-prefix branches at seed 10086 episode 3 frame 144:
  factual, two aligned-success blocks, four phase-shift controls, two orthogonal
  equal-RMS controls, and eight failure-context policy samples.
- No branch succeeded during the intervention horizon.
- Mean delayed hinge-min effect for aligned-success blocks was +0.01222, slightly
  below equal-RMS controls at +0.01270. Failure-policy samples averaged +0.00719.
  The best individual phase-shift control (+0.0260, recorded in branch results)
  also exceeded the aligned-success mean. The alleged success branch therefore
  has no specific advantage over negative controls.
- No training data was generated. Added this checkpoint to
  `TASK_CONTINUATION.md`.

### Key Result Paths

- Intervention summary:
  `data/fold_glasses_opensource_s0_collect_4x50_20260812_112113/block_intervention_seed10086_ep3_f144_v1/summary.json`
- Per-branch evidence:
  `data/fold_glasses_opensource_s0_collect_4x50_20260812_112113/block_intervention_seed10086_ep3_f144_v1/branch_results.jsonl`
- Factual replay prerequisite:
  `data/fold_glasses_opensource_s0_collect_4x50_20260812_112113/factual_replay_ep3_v2_full/factual_replay.json`

### Next 1-3 Steps

1. Implement width-independent mixed-seed discovery over all replan blocks using
   action-free phase alignment and normalized executed-block separation.
2. Extract the remaining rollout features and rank the earliest candidates by
   shared-context support, persistent action separation, and later divergence.
3. Run bounded exact-prefix interventions with phase-shift and equal-RMS controls
   on the strongest candidates before generating data or launching training.

## Checkpoint 8 - 2026-08-13

### Goal

Replace failure-width-jump event discovery with a width-independent scan for the
actual observational pattern: shared pre-divergence context, separated executed
success/failure action branches, then persistent future context divergence.

### User Request Reference

- Window ID: unavailable in the current tool interface.
- Item ID: unavailable in the current tool interface.
- Same request as Checkpoint 7. The user specifically questioned whether the
  visually similar, action-broad region before visual divergence is the useful
  policy-learning event.

### Decisions

- Distinguish two meanings of action width. Policy-sample width at one observation
  is uncertainty/multimodality and is a downstream probe. Distance between
  executed same-context success/failure blocks is the cheap discovery signal.
- Never average successful actions into one target. Treat every observed success
  block as its own mode and compare a failure block to the nearest mode.
- Do not calibrate a success-mode radius from distances between successful modes.
  Two far-apart actions can both be valid and would inflate that threshold. The
  offline scan uses only a low normalized-distance screening floor; each mode's
  support radius is later estimated from policy samples at that exact successful
  observation.
- The observational order is explicit: current action-free context support,
  persistent within-block action separation, then fixed-lag future visual/state
  divergence. Every selected event still has `training_eligible=false`.
- The action comparison uses the checkpoint's actual global z-score statistics
  (`use_stepwise_action_norm=false`) with the same [-5, 5] clipping as inference.
- Retain all scored blocks for audit and select only the earliest block of each
  contiguous qualifying interval to reduce temporal redundancy.

### Done / Files Changed

- Added `scripts/fold_glasses/discover_seedpair_branch_events.py`.
- Added four synthetic tests covering checkpoint normalization, monotone endpoint
  alignment, action separation before later visual divergence, preservation of
  multiple successful modes, and rejection when a failure action matches any
  successful mode.
- The first test run exposed that cross-success mode distances cannot define a
  support threshold; repaired that methodological error before real-data use.
- Verification: `python -m pytest -q tests/test_fold_glasses_seedpair_events.py`
  completed with 14 passed.

### Key Result Paths

- Width-independent scanner:
  `scripts/fold_glasses/discover_seedpair_branch_events.py`
- Synthetic/utility tests: `tests/test_fold_glasses_seedpair_events.py`
- Checkpoint stats used for action z-score:
  `/data_all/xiangchengzhan/FastWAM-infer-in-DexJoco/artifacts/fold_glasses/dataset_stats.json`
- Stats SHA256:
  `961ecadb6098d0b687b4d9eb34371281bb23f3e05dd0bfdf12c8116fd9782289`

### Next 1-3 Steps

1. Run the new scanner on the existing four seed 10086 feature files and audit
   whether it avoids or appropriately contextualizes the already falsified f144
   event.
2. Extract action-free replan features for the remaining 196 rollouts and run the
   all-mixed-seed scan.
3. Render/rank a bounded candidate set, then policy-probe and exact-prefix test
   only the strongest nonredundant events.

## Checkpoint 9 - 2026-08-13

### Goal

Create the complete action-free, replan-level feature cache needed to search all
mixed Fold Glasses seeds without using failure width jumps or absolute time.

### User Request Reference

- Window ID: unavailable in the current tool interface.
- Item ID: unavailable in the current tool interface.
- Same request as Checkpoints 7-8: determine and implement whether a shared
  visual context, separated action branch, and later visual divergence defines
  useful event data for increasing policy success.

### Decisions

- Feature extraction is not training and did not resume the old GPU 0-3 run.
- Keep all 200 feature files for audit and hard-transfer evaluation, while the
  selector itself continues to exclude all-success and all-failure seeds.
- Successful and failed episodes are aligned only through front/wrist CLIP plus
  proprioception. Executed actions are stored but never enter phase alignment.

### Done / Files Changed

- Completed dual-camera CLIP/replan feature extraction for all 200 rollout
  episodes. The first four cached seed 10086 files were reused; the other 196
  were extracted on GPU 0 in the `fastwam` environment.
- The extraction process exited successfully and released its GPU process.
- Integrity audit passed: index and NPZ counts are both 200; episode indices are
  exactly 0-199; total replan points are 6,618; every file has feature dimensions
  `(front=1024, wrist=1024, state=23, block=24x22)`; no NaN or infinity exists.
- No training data or manifest was generated at this stage.

### Key Result Paths

- Complete feature cache:
  `data/fold_glasses_opensource_s0_collect_4x50_20260812_112113/seedpair_replan_features_clip_v1`
- Feature index:
  `data/fold_glasses_opensource_s0_collect_4x50_20260812_112113/seedpair_replan_features_clip_v1/episode_features.jsonl`
- Extraction config:
  `data/fold_glasses_opensource_s0_collect_4x50_20260812_112113/seedpair_replan_features_clip_v1/config.json`

### Next 1-3 Steps

1. Run the width-independent branch scanner over all 35 mixed seeds using the
   full mixed-seed calibration population.
2. Audit candidate counts, evidence margins, timing, confidence tiers, and the
   known seed 10086 intervention-negative control.
3. Build a bounded high-confidence probe set and render anchor/future frames
   before spending GPU compute on policy distributions.

## Checkpoint 10 - 2026-08-13

### Goal

Test whether a failure action-width rise adds useful information beyond the core
event pattern, rather than assuming width jump is either necessary or sufficient.

### User Request Reference

- Window ID: unavailable in the current tool interface.
- Item ID: unavailable in the current tool interface.
- The user specifically asked why not align existing failure action-width plots
  with success distributions, and whether a visually shared, action-wide region
  followed by visual divergence is useful for policy learning.

### Decisions

- Treat failure policy width as a pre-registered stratification variable, not an
  event gate. Compare high-width candidates against low-width candidates that
  satisfy the same context/action/future observational criteria.
- Require at least two successful current-context supports and at least two
  future evidence sources for the expensive shortlist. Use distinct seeds within
  each width stratum to avoid one seed dominating a group.
- A successful episode that terminates before a requested future horizon is not
  missing at random: task success has already occurred. Represent it explicitly
  with the last available successful state as a terminal proxy; never label that
  image as an exact fixed-lag frame.
- Store future evidence source counts and terminal-proxy counts at every horizon.
  Preview terminal proxies with a visible red annotation.
- Freeze the shortlist before new action-distribution probes: four candidates
  with failure width/baseline >=1.5 and four with <=1.0, ranked only by the
  bounded observational score within each stratum. The new probe uses diffusion
  seeds independent from the prior samples used to define width strata.

### Done / Files Changed

- Updated `scripts/fold_glasses/discover_seedpair_branch_events.py` with explicit
  successful-terminal future evidence and per-horizon provenance counts.
- Added `scripts/fold_glasses/build_seedpair_probe_shortlist.py` to attach the
  existing failure width curves, select pre-registered high/low-width strata,
  and render anchor/future dual-camera audit sheets.
- Added tests for terminal proxy semantics and distinct-seed width strata.
- Regenerated the all-seed audit as v2: 35 mixed seeds, 3,000 scored failure
  blocks, 95 nonredundant observational candidates, 0 training-eligible events.
  Of these, 65 have at least two current success supports; 45 use at least one
  explicit successful-terminal source at divergence onset.
- Built and manually inspected eight preview sheets. Both high- and low-width
  groups show the same task phase at the anchor (glasses held near/outside the
  box) followed by success-in-box versus failure-outside divergence. The visual
  audit therefore rejects width rise as a necessary event condition.
- Verification: `python -m pytest -q tests/test_fold_glasses_seedpair_events.py`
  completed with 16 passed.

### Key Result Paths

- All-seed v2 audit:
  `data/fold_glasses_opensource_s0_collect_4x50_20260812_112113/seedpair_branch_audit_all_v2`
- Frozen width-stratified shortlist:
  `data/fold_glasses_opensource_s0_collect_4x50_20260812_112113/seedpair_probe_shortlist_width_stratified_v2/probe_shortlist.jsonl`
- Preview sheets:
  `data/fold_glasses_opensource_s0_collect_4x50_20260812_112113/seedpair_probe_shortlist_width_stratified_v2/previews`
- Shortlist summary:
  `data/fold_glasses_opensource_s0_collect_4x50_20260812_112113/seedpair_probe_shortlist_width_stratified_v2/summary.json`
- Existing width diagnostic source:
  `data/fold_glasses_opensource_s0_collect_4x50_20260812_112113/failure_width_curves_full`

### Next 1-3 Steps

1. Probe all unique failure/success contexts in the frozen eight-candidate
   shortlist with a fresh common set of diffusion noise seeds.
2. Score per-success-mode support mass and compare high- versus low-width strata;
   reject candidates whose policy samples do not expose a success branch.
3. Factual-replay and exact-prefix intervene only on the strongest surviving
   candidates, using phase-shift and equal-RMS controls before any training data.

## Checkpoint 11 - 2026-08-13

### Goal

Measure whether the frozen policy at visually shared failure observations already
assigns probability to any same-seed successful action mode, and separate that
question from whether a transferred success block is causally useful.

### User Request Reference

- Window ID: unavailable in the current tool interface.
- Item ID: unavailable in the current tool interface.
- Same user question as Checkpoint 10: determine whether aligning failure width
  with success action distributions identifies events whose supervision can
  concentrate the policy on correct actions and raise rollout success.

### Decisions

- Estimate one support mode per successful rollout. Its radius is the q90 distance
  from 16 fresh policy samples at that exact successful observation to the
  executed successful block. Never average successful modes.
- Freeze `q90 x 1.0` as the strict primary support criterion. Multipliers 0.75,
  1.25, and 1.5 are sensitivity analyses only. Do not choose the multiplier
  after seeing which candidates hit.
- A strict support miss does not reject causal intervention. It means the current
  failure-conditioned policy does not expose the candidate success action with
  measurable probability. The actual success block may still be a useful
  low-probability correction if exact-prefix intervention validates it.
- Conversely, a sampled support hit does not make data trainable. It must still
  beat phase-shift/equal-RMS controls and succeed under closed-loop continuation.
- Use a fresh common set of 16 diffusion seeds across all contexts, independent
  of the eight samples used in the prior width curves.

### Done / Files Changed

- Probed all 26 unique contexts from the frozen eight-candidate shortlist: eight
  failure anchors and 18 successful targets. Every context has 16 finite
  normalized action blocks of shape 24x22.
- Strict support result (`q90 x 1.0`): all 8 candidates have 0/16 failure-policy
  samples in any successful mode. The Wilson 95% upper bound per candidate is
  0.194. Multipliers 0.75 and 1.25 also yield zero branching candidates.
- At the permissive `q90 x 1.5` sensitivity only seed 10102 episode 11 frame 408
  has one hit (1/16, mass 0.0625, Wilson 95% 0.011-0.283). This is not robust:
  the hit distance is 0.503 while the corresponding success-context sample
  maximum is 0.391; it appears only after radius inflation.
- Fresh normalized widths preserve the pre-registered stratification: high-width
  failure anchors have first-step widths 0.120-0.246 (except all above the low
  group range), while low-width anchors are 0.090-0.103. Yet strict success-mode
  mass is zero in both strata.
- No event is training eligible and no training manifest was produced.
- Verification remained 16 passing tests.

### Key Result Paths

- Fresh paired-noise action probe:
  `data/fold_glasses_opensource_s0_collect_4x50_20260812_112113/seedpair_action_probe_width_stratified_v2_k16`
- Strict support scores (`q90 x 1.0`):
  `data/fold_glasses_opensource_s0_collect_4x50_20260812_112113/seedpair_action_support_width_stratified_v2_k16_m100`
- Sensitivity scores (`x0.75`, `x1.25`, `x1.5`):
  `data/fold_glasses_opensource_s0_collect_4x50_20260812_112113/seedpair_action_support_width_stratified_v2_k16_m075`
  `data/fold_glasses_opensource_s0_collect_4x50_20260812_112113/seedpair_action_support_width_stratified_v2_k16_m125`
  `data/fold_glasses_opensource_s0_collect_4x50_20260812_112113/seedpair_action_support_width_stratified_v2_k16_m150`

### Next 1-3 Steps

1. Complete full factual replay gates for the eight failure episodes in the
   frozen shortlist.
2. Run exact-prefix actual-success blocks against factual, phase-shift, and
   equal-normalized-RMS controls. Compare high- versus low-width strata without
   changing candidate membership.
3. Only for interventions with a success-specific effect, run frozen-policy
   closed-loop continuation and save successful counterfactual trajectories for
   event/CFG training.

## Checkpoint 12 - 2026-08-13

### Goal

Locate the policy-relevant event directly from a recorded failure trajectory by
estimating the frozen policy's closed-loop recoverability from exact failure
prefixes. The target interval is the transition from a recoverable prefix to a
low-recoverability prefix; same-seed successes then identify and validate the
correct action branch within that interval.

### User Request Reference

- Window ID: unavailable in the current tool interface.
- Item ID: unavailable in the current tool interface.
- Request: replay a failure rollout to frame K, resample the policy several times
  from that exact state, advance K while Pass@4 remains nonzero, and treat the
  interval K-N..K where recovery disappears as the candidate critical event.
- Goal thread ID: `019ff910-7aa4-7aa1-be9c-aa4720eb026b`.

### Decisions

- Adopt the recoverability frontier as the primary event locator. It is more
  directly tied to Pass@1 than action-width or observational trajectory
  divergence.
- Scan only 24-frame replan boundaries. At each boundary discard the factual
  rollout's pending action chunk and query the frozen policy anew; otherwise the
  experiment changes the controller cadence.
- Do not equate 0/4 with irrecoverability. If true recovery probability is 10%,
  four failures occur with probability 65.6%. Use an adaptive number of
  continuations and report uncertainty / censoring rather than a binary claim
  from four samples.
- Define practical recoverability under the original 1200-step task budget, not
  physical reachability under unlimited time. Exclude or separately diagnose
  frontiers caused only by insufficient remaining time.
- Treat K-N..K as a candidate event, not automatically training data. Require a
  repeated recoverability drop plus a success-supported action intervention from
  K-N that crosses the boundary and yields successful closed-loop continuation.
- Recoverability need not be monotone in a single finite Monte Carlo scan. Use a
  coarse-to-fine scan and confirm adjacent blocks/common noise seeds rather than
  stopping at the first observed zero.
- Same-seed success trajectories supply possible corrective modes after the
  frontier is located. Action width remains a stratification/interpretation
  variable; it is neither necessary nor sufficient.
- Keep all-success seeds excluded. Use mixed-seed failures for discovery and
  reserve all-failure seeds 10106/10119 for hard-transfer evaluation. Keep the
  old GPU 0-3 training stopped.

### Done / Files Changed

- Reframed the selector around an exact-prefix recoverability frontier.
- Verified the deployed policy is memoryless at each replan: it consumes the
  current front/wrist images and 23-D proprioception, so exact simulator state
  replay plus a newly rendered observation is sufficient to restart inference.
- Verified the collection/controller cadence is 32 predicted actions with 24
  executed actions per replan.
- Verified existing intervention code already snapshots/restores MuJoCo state,
  wrapper counters, and the success-trigger counter after factual prefix replay.
- Added this checkpoint to `TASK_CONTINUATION.md`. The requested notes/context
  tools are not exposed in the current tool interface, so this repository file
  is the durable continuation record.

### Key Result Paths

- Policy inference implementation:
  `/data_all/xiangchengzhan/FastWAM-infer-in-DexJoco/src/fastwam_dexjoco/policy.py`
- Exact-prefix state utilities:
  `scripts/fold_glasses/run_seedpair_block_interventions.py`
- Deterministic replay gate:
  `scripts/fold_glasses/validate_factual_replay.py`
- Frozen shortlist:
  `data/fold_glasses_opensource_s0_collect_4x50_20260812_112113/seedpair_probe_shortlist_width_stratified_v2/probe_shortlist.jsonl`

### Next 1-3 Steps

1. Implement a reusable exact-prefix recoverability scanner with adaptive
   sampling, common noise seeds, time-limit controls, and resumable JSONL output.
2. Add unit tests for statistical classification, non-monotone/censored scans,
   and event interval selection.
3. Run factual replay plus a bounded one-episode pilot, inspect the recovery
   curve, then scale only if the pilot validates the experiment mechanics.

## Checkpoint 13 - 2026-08-13

### Goal

Implement a compute-pragmatic Pass@K failure-prefix scan that locates a slightly
wide critical event window without requiring formal sequential hypothesis tests.

### User Request Reference

- Window ID: unavailable in the current tool interface.
- Item ID: unavailable in the current tool interface.
- Request: rigorous confidence testing is unnecessary; fix a reasonable Pass@K
  and scan at an action-chunk/replan cadence, and allow the selected event to be
  somewhat wider.
- Goal thread ID: `019ff910-7aa4-7aa1-be9c-aa4720eb026b`.

### Decisions

- Use the user's notation `Pass@K` for the number of continuations. Use `t` for
  the failure-prefix frame so it is not confused with K.
- Fix K=4 by default, matching the existing Pass@4 evaluation intuition.
- Scan on the actual 24-frame replan grid. The model predicts 32 actions but the
  deployed controller executes only 24 before replanning, so a 32-frame grid is
  not controller-aligned.
- At each t, report both `success_count/4` and the Boolean Pass@4 hit (at least
  one success). No formal confidence interval is required for event discovery.
- A candidate downward frontier requires a previously hit scan point followed by
  at least two consecutive zero-hit scan points. One isolated 0/4 is ignored.
- Expand the raw frontier by one 24-frame block on both sides. A wider event is
  preferred to cutting out the causal action transition.
- Allow multiple frontiers and explicit recovery islands; never describe the
  heuristic result as physically irreversible.
- A trajectory with no earlier Pass@4 hit or no later persistent zero produces
  no event. Zero-hit prefixes are diagnostic only; positive training data come
  from successful closed-loop continuations at the last recoverable prefix.
- Same-seed successes remain useful for action-mode comparison and controls after
  the frontier scan, but do not locate the frontier. All-success seeds remain
  excluded and all-failure seeds remain held-out hard-transfer tests.

### Done / Files Changed

- Frozen the simplified experimental rule above after the user explicitly chose
  practicality over formal statistical classification.
- Completed independent code audits confirming that a new closed-loop scanner is
  required and that exact replan-boundary restart is supported by the memoryless
  inference policy.
- Added this checkpoint to `TASK_CONTINUATION.md`. Notes/context-management tools
  requested by the user remain unavailable in the current tool interface.

### Key Result Paths

- Durable continuation record: `TASK_CONTINUATION.md`
- Planned scanner: `scripts/fold_glasses/scan_failure_recoverability_frontier.py`
- Planned tests: `tests/test_fold_glasses_recoverability_frontier.py`
- Existing exact-prefix utilities:
  `scripts/fold_glasses/run_seedpair_block_interventions.py`

### Next 1-3 Steps

1. Finish the scanner and pure unit tests for the fixed K=4 / stride=24 rule.
2. Run one factual-replay-gated mixed-seed failure pilot and inspect its complete
   Pass@4-over-prefix curve plus any recovery islands.
3. Use a detected frontier to validate the widened event with a paired successful
   continuation/action-block intervention before constructing training data.

## Checkpoint 14 - 2026-08-13

### Goal

Produce at most one exact-prefix recoverability event pair per mixed seed and
retain both the successful counterfactual event and its factual failure event for
training/audit.

### User Request Reference

- Window ID: unavailable in the current tool interface.
- Item ID: unavailable in the current tool interface.
- Request: save both success and failure events for training; when a seed has
  failures, use only one failure rollout for that seed; skip seeds with no
  failures.
- Goal thread ID: `019ff910-7aa4-7aa1-be9c-aa4720eb026b`.

### Decisions

- Preselect exactly one failure episode per mixed seed before scanning: the
  lowest episode index (equivalently the earliest recorded failed repeat).
- Do not scan a second failure for a seed even if the preselected episode yields
  no frontier. This avoids post-hoc selection of the most favorable failure.
- Skip all-success seeds because they have no failure episode.
- Keep the two all-failure seeds 10106 and 10119 out of event selection/training;
  retain them as hard-transfer evaluation seeds as previously decided.
- Save paired event provenance under one pair id. `failure_event` is the factual
  source observation/action window; `success_event` is a counterfactual rollout
  from the same exact prefix that ultimately succeeds. Both retain states,
  actions, outcome, prefix t, widened window, policy/noise seeds, and source
  episode/seed metadata.
- A zero-hit prefix is diagnostic and is not used as a positive action target.
  Successful counterfactual event trajectories from the last recoverable prefix
  are the positive supervision source. Failure event data may be retained for
  outcome-conditioned video/CFG use, with failure action imitation still masked.

### Done / Files Changed

- Completed the full deterministic factual replay gate for the first pilot:
  seed 10102, repeat 3, episode 11, 1200 steps.
- Replay passed state, image, outcome, and no-early-termination checks at frames
  0, 408, 432, and 1199. Recorded and replayed outcomes are both failure.
- Added the one-failure-per-seed and paired-event persistence contract to this
  continuation checkpoint.

### Key Result Paths

- Passed pilot replay:
  `data/fold_glasses_opensource_s0_collect_4x50_20260812_112113/factual_replay_ep11_frontier_v1/factual_replay.json`
- Replay audit images:
  `data/fold_glasses_opensource_s0_collect_4x50_20260812_112113/factual_replay_ep11_frontier_v1`
- Source outcomes:
  `data/fold_glasses_opensource_s0_collect_4x50_20260812_112113/rollout_raw_200/meta/episode_outcomes.jsonl`
- Source attempt ledger:
  `data/fold_glasses_opensource_s0_collect_4x50_20260812_112113/rollout_raw_200/collection_summary.json`

### Next 1-3 Steps

1. Finish scanner implementation, seed-level preselection, paired event artifacts,
   and unit tests.
2. Run a bounded Pass@4 prefix pilot on episode 11 using the passed replay gate;
   start around the previously observed 408-frame region before full-grid scale.
3. If a persistent hit-to-zero frontier appears, materialize the widened paired
   event and verify it can be consumed by the existing EVE dataset path.

## Checkpoint 15 - 2026-08-13

### Goal

Finalize the minimal paired-event contract so that saved success/failure data is
causally aligned and cannot silently train on failure actions.

### User Request Reference

- Window ID: unavailable in the current tool interface.
- Item ID: unavailable in the current tool interface.
- Request refinement: retain success and failure events; one failure per seed;
  skip seeds without failures.
- Goal thread ID: `019ff910-7aa4-7aa1-be9c-aa4720eb026b`.

### Decisions

- A training pair must be formed at one identical prefix frame `t` and one live
  simulator snapshot. Select one successful and one failed closed-loop replicate
  from the same four-run Pass@4 batch at that `t` (success count must be 1, 2, or
  3). Do not pair a success from `t-24` with a failure from `t`.
- If a scan point is 0/4 or 4/4, it can locate/diagnose a frontier but produces no
  training pair at that point. This preserves the same-observation action branch
  requirement.
- Raw paired episodes retain the exact factual prefix through `t`; the branch
  suffix is the policy continuation. Prefix observations/actions must be byte
  identical between pair members (or carry an explicit hash mismatch reason).
- EVE manifest contains exactly two event units per pair, a 33-frame window that
  contains `[t, t+24)`, `core_start_frame=t`, `core_end_frame=t+24`, and
  `window_selection=core_start_anchor`.
- Success unit: `event_outcome=success`, `batch_role=primary`,
  `action_loss=enabled`, `action_loss_window=[t,t+24]`.
- Failure unit: `event_outcome=failure`, `batch_role=auxiliary`,
  `action_loss=disabled` for the entire unit. Failure data is outcome/video
  context, never a positive action target.
- `pair_id`/`pair_weight` remain provenance metadata; current trainer does not
  implement pairwise contrastive loss. Do not claim that metadata alone creates
  paired learning.
- Every pair records seed, source failure episode/repeat, t, snapshot/prefix hash,
  selected replicate IDs, per-replan noise ledgers, checkpoint/config/stat/text
  hashes, and terminal outcomes in `event_pairs.jsonl`.

### Done / Files Changed

- Received and incorporated the implementation audit's final pair contract.
- Clarified that a mixed Pass@4 point, not merely a recoverability frontier, is the
  minimum condition for training-event materialization.
- Updated this durable continuation record.

### Key Result Paths

- EVE semantics: `src/fastwam/datasets/eve/manifest_dataset.py`
- Manifest schema: `src/fastwam/everobot_schema.py`
- Existing pair examples:
  `data/fold_glasses_dewo_v2_opensource_20260812_195749/eve_v02/manifests/offline_b1_jump_fast.json`
- Existing policy loss behavior: `src/fastwam/models/wan22/fastwam.py`

### Next 1-3 Steps

1. Review the new scanner against this contract and repair any mismatch before
   running it.
2. Run CPU tests and one GPU pilot on the preselected failure episode for seed
   10102 (episode 10, not the mechanism-only episode-11 pilot).
3. Materialize only mixed-prefix successful/failure pairs, then validate the EVE
   manifest and dataset action masks before any training.

## Checkpoint 16 - 2026-08-13

### Goal

Resolve the EVE 33-frame window question without diluting the critical action
signal or leaking factual failure actions into positive supervision.

### User Request Reference

- Window ID: unavailable in the current tool interface.
- Item ID: unavailable in the current tool interface.
- Request: ask whether action masking is necessary and whether a few frames can
  be concatenated before/after an event to form the required 33-frame window.
- Goal thread ID: `019ff910-7aa4-7aa1-be9c-aa4720eb026b`.

### Decisions

- Prefer a window anchored exactly at the recoverability prefix `t`, with 33
  consecutive observations `[t, t+33)` and 32 actions `[t, t+32)`. Append
  post-event continuation frames rather than prepending factual frames.
- For a success counterfactual, all 32 actions in this suffix are from the same
  closed-loop branch that reached success, so no success-side action mask is
  needed. The event anchor itself keeps the sample focused.
- Failure event windows retain the same anchor and frame count but set
  `action_loss=disabled` for the whole unit. This remains necessary because
  concatenating frames does not make failure actions valid positive labels.
- If a future implementation must prepend frames before `t`, it must either mask
  every pre-`t` action or use those frames only as visual context; never silently
  supervise the factual failure prefix. This is a fallback, not the primary
  format.
- The prior narrow success `action_loss_window=[t,t+24)` contract is superseded
  for the primary format by full post-`t` supervision through `t+32`; the last
  eight actions are continuation stabilization, not an additional event label.
- If a branch terminates before `t+33`, do not fabricate repeated terminal frames
  for training. Use another successful replicate/window or mark the pair
  incomplete; terminal proxies remain diagnostic only.

### Done / Files Changed

- Decided the mask/concatenation tradeoff: append-after requires no success mask;
  prepend-before requires masking. Failure action masking remains mandatory.
- Updated this durable continuation record before code changes.

### Key Result Paths

- EVE window expansion/masking:
  `src/fastwam/datasets/eve/manifest_dataset.py`
- Existing 33-frame convention:
  `src/fastwam/datasets/lerobot/robot_video_dataset.py`
- Pair manifest examples:
  `data/fold_glasses_dewo_v2_opensource_20260812_195749/eve_v02/manifests/offline_b1_jump_fast.json`

### Next 1-3 Steps

1. Review the scanner/materializer implementation against the new append-after
   contract and remove the success action mask from generated manifests.
2. Keep the failure unit's full action disable and add a verifier for exactly 33
   observations / 32 post-`t` action labels.
3. Run CPU tests, then the single selected seed pilot before any training.

## Checkpoint 17 - 2026-08-13

### Goal

Use the corrected recoverability event convention requested by the user:
`t` is the last mixed/recoverable prefix, and `t+24` is the first confirmed
zero-hit prefix. Materialize the 33-frame event as `[t-9, t+24)`.

### User Request Reference

- Window ID: unavailable in the current tool interface.
- Item ID: unavailable in the current tool interface.
- Correction: the user's `t` denotes a prefix with successful continuations;
  `t+24` denotes the next replan point with `0/4` unrecoverable continuations.
  The event should run from `t-9` through `t+24` (33 frames).
- Goal thread ID: `019ff910-7aa4-7aa1-be9c-aa4720eb026b`.

### Decisions

- Do not use the first-zero point as the event anchor. It is the event's right
  boundary (`t+24`), not `t`.
- Define `t` as the immediately preceding scan point with `1 <= success_count <
  M`; this is the same observation prefix at which a successful and failed
  continuation can be paired.
- Require the next scan point `t+24` to have `0/M`, and the following scan point
  `t+48` to also have `0/M` for the simple persistence confirmation.
- Set the primary event interval to the exact half-open range `[t-9, t+24)`;
  clamp the left edge to zero only for very early events, and reject/clasify
  intervals shorter than 33 frames rather than fabricating frames.
- Pair success and failure replicates from the same `t` snapshot only. A
  success from `t` must not be paired with a failure from `t+24`.
- Keep scan stride/replan stride at 24. `Pass@M` remains M=4 by default.
- Keep the pre-encode requirement: before training, verify text embeddings and
  VAE latent cache completeness against the final event manifest; compute any
  missing artifacts first, then re-run the verifier.

### Done / Files Changed

- Corrected the semantic specification after identifying the anchor inversion in
  the previous checkpoint/implementation review.
- Recorded this checkpoint before applying code edits.

### Key Result Paths

- Scanner under correction:
  `scripts/fold_glasses/scan_failure_recoverability_frontier.py`
- Scanner tests:
  `tests/test_fold_glasses_recoverability_frontier.py`
- Durable task record: `TASK_CONTINUATION.md`

### Next 1-3 Steps

1. Change frontier detection/materialization to use mixed `t`, confirmed zero at
   `t+24`, and `[t-9,t+24)` exactly.
2. Fix one-failure-per-seed selection to use the earliest selected failure and
   require mixed `1..M-1` prefix results before producing a training pair.
3. Add preflight checks/commands for text embedding and VAE caches, then run
   CPU tests and the selected seed pilot.

## Checkpoint 18 - 2026-08-13

### Goal

Make the recoverability event implementation match the user's exact temporal
definition and produce data that cannot silently train on the wrong branch.

### User Request Reference

- Window ID: unavailable in the current tool interface.
- Item ID: unavailable in the current tool interface.
- Latest clarification: `t` is a snapshot with successful continuations;
  `t+24` is the first `0/4` unrecoverable scan point. Use the event
  `[t-9, t+24)` and retain same-`t` success/failure events.
- Goal thread ID: `019ff910-7aa4-7aa1-be9c-aa4720eb026b`.

### Decisions

- `t` means `1 <= success_count < M` at one exact simulator snapshot; it is
  never the zero point.
- `t+24` must be the next aligned `0/M` point and `t+48` is a confirmation
  point. A single zero is not enough.
- The training candidate is exactly 33 observations `[t-9, t+24)` and 32
  action labels `[t-9, t+23)`. Early events that cannot supply 33 real frames
  are diagnostic only and are excluded from training.
- A pair selects one successful and one failed continuation from the same
  Pass@4 batch at the same `t` snapshot. Failure actions are retained for
  audit/context but have action loss disabled.
- Bump scanner artifact/version signatures before any real scan so stale
  prefix/pair caches cannot be reused.
- Do not start the stopped 0-3 GPU training or any new training in this phase.

### Done / Files Changed

- Scanner pure selector currently passes 8 semantic unit tests and has explicit
  `t_frame`, `t_plus_24_frame`, and `[t-9,t+24)` fields.
- Scanner persists every same-prefix replicate, including failures, but event
  materialization still needs branch-specific failure rerun/images and the
  33-observation/32-action shape fix.
- Existing scanner path:
  `scripts/fold_glasses/scan_failure_recoverability_frontier.py`.
- Existing tests:
  `tests/test_fold_glasses_recoverability_frontier.py`.

### Key Tool Results

- `python -m pytest -q tests/test_fold_glasses_recoverability_frontier.py`:
  8 passed (output was returned by the tool; no large output file needed).
- `python -m py_compile scripts/fold_glasses/scan_failure_recoverability_frontier.py`:
  passed.
- Rollout source:
  `data/fold_glasses_opensource_s0_collect_4x50_20260812_112113/rollout_raw_200`.

### Next 1-3 Steps

1. Fix scanner version/signature, exact event array shapes, and same-`t`
   success/failure branch materialization.
2. Add/run a pure recoverability pair verifier and tests, including explicit
   failure action masking metadata.
3. Add pre-training text/VAE cache preflight and only then prepare a pilot
   command; leave GPU training stopped until the artifacts pass verification.

## Checkpoint 18 - 2026-08-13 (EVE Pair Verifier Phase)

### Goal

Provide a read-only, dependency-light verifier for recoverability pair artifacts
before they are allowed into EVE training. The verifier must enforce the exact
scanner event geometry and prevent factual failure actions from becoming action
imitation targets.

### User Request Reference

- Window ID: unavailable in the current tool interface.
- Item ID: unavailable in the current tool interface.
- Clarification: inspect the EVE/LeRobot path and ensure exactly 33-frame paired
  windows, with success action supervision and failure action loss disabled.
- Goal thread ID: `019ff910-7aa4-7aa1-be9c-aa4720eb026b`.

### Decisions

- This phase changes only a pure verifier and its tests; it does not modify the
  existing `build_lerobot_eve_dataset.py` episode copier.
- A complete training pair requires mixed `Pass@4` at `t`, confirmed zeros at
  `t+24` and `t+48`, and the exact half-open event interval
  `[max(0,t-9), t+24)`. Training eligibility additionally requires the full 33
  frames (so an early clamped event is reported but rejected for training).
- Success and failure NPZs must have matching frame indices and `[T,22]` action /
  `[T,23]` state arrays; factual prefix rows before `t` must match exactly.
- Failure outcome is never a positive action target. Explicit `action_loss=enabled`
  on a failure artifact is an error; absent legacy metadata is reported as an
  implicit outcome-derived disablement, with strict mode available for future
  materializers.
- Pair identity is checked through seed, source episode, `t`, replicate roles,
  run signature, and any supplied snapshot/prefix hashes. Missing hashes are
  surfaced as warnings rather than fabricated.

### Done / Files Changed

- Audited `scripts/fold_glasses/scan_failure_recoverability_frontier.py`,
  `scripts/everobot/build_lerobot_eve_dataset.py`,
  `src/fastwam/datasets/eve/manifest_dataset.py`, and
  `src/fastwam/everobot_schema.py`.
- Confirmed the current EVE loader consumes actual LeRobot episode/video data;
  scanner sidecar NPZ/MP4 files are not automatically trainable references.
- Confirmed the current 33-frame loader contract exposes 32 action tokens; a
  `[t,t+24)` action mask therefore supervises 23 tokens under the existing
  indexing convention.

### Key Tool Results

- Scanner/materializer body: `scripts/fold_glasses/scan_failure_recoverability_frontier.py:837-1089`.
- EVE window expansion/mask: `src/fastwam/datasets/eve/manifest_dataset.py:238-385`.
- LeRobot 33-frame/action-size contract: `src/fastwam/datasets/lerobot/robot_video_dataset.py:47-99,341-423`.
- EVE schema validation: `src/fastwam/everobot_schema.py:107-332`.

### Next 1-3 Steps

1. Add `scripts/everobot/validate_recoverability_event_pairs.py` with structural,
   array, prefix, and role checks plus a JSON report/CLI.
2. Add synthetic unit tests covering valid pairs and every critical rejection
   (all-success, non-adjacent zeros, widened/short windows, prefix mismatch,
   failure action enabled).
3. Run the focused tests and report the verifier path to the parent agent; do
   not start training or alter the stopped GPU jobs.

## Checkpoint 19 - 2026-08-13 (Pair Verifier Complete)

### Goal

Make recoverability sidecar pairs auditable and safe to hand to the EVE
materialization/training stage. A pair must represent one same-seed mixed prefix,
not an arbitrary success and failure episode.

### Decisions

- `scripts/everobot/validate_recoverability_event_pairs.py` is read-only and has
  no MuJoCo/LeRobot/training dependency.
- A training pair must explicitly declare `seed_classification="mixed"`,
  `training_eligible=true`, and `evaluation_only=false`. All-success and
  all-failure seeds are rejected.
- The verifier enforces `Pass@4` mixed count (1..3), exact zero points at
  `t+24` and `t+48`, and exact `[max(0,t-9),t+24)` geometry with 33 rows.
- Both branch NPZs require contiguous matching frame IDs, `[T,22]` actions,
  `[T,23]` states, finite values, and byte-identical action/state prefix rows
  before `t`.
- Success may declare only `action_loss_window=[t,t+24)`; failure must not carry
  an action-loss window and must resolve to `action_loss=disabled`. Missing
  legacy action-loss metadata is inferred safely but warned; strict CLI mode
  (`--require-explicit-action-loss`) rejects it.
- Optional snapshot/prefix hashes are validated for SHA-256 syntax and must agree
  across branches; `--require-hashes` makes their absence fatal. Optional
  trajectory ledgers are checked for exact `t`, replicate index, and outcome;
  `--require-trajectory-ledgers` makes missing ledgers fatal.
- The existing LeRobot episode copier remains untouched in this phase. Sidecar
  NPZ/MP4 artifacts are not trainable references until a later materializer
  converts them into actual LeRobot episodes or extends the loader.

### Done / Files Changed

- Added verifier: `scripts/everobot/validate_recoverability_event_pairs.py`.
- Added synthetic tests: `tests/test_validate_recoverability_event_pairs.py`.
- Added this checkpoint to `TASK_CONTINUATION.md`.

### Key Tool Results

- Focused result: `PYTHONPATH=. pytest -q tests/test_validate_recoverability_event_pairs.py`
  -> 14 passed.
- Combined scanner/selector result:
  `PYTHONPATH=. pytest -q tests/test_validate_recoverability_event_pairs.py tests/test_fold_glasses_recoverability_frontier.py`
  -> 18 passed.
- Static checks: `python -m py_compile ...` and `git diff --check` passed.

### Next 1-3 Steps

1. Main agent should add explicit `action_loss`, `action_loss_window`, and
   snapshot/prefix hashes to scanner descriptors/pair metadata, then run the
   verifier in strict mode before materialization.
2. Define the actual LeRobot materialization bridge for sidecar arrays; do not
   point an EVE manifest at `event_pairs.jsonl` alone.
3. Re-run the verifier on the first real pair and inspect the resulting JSON
   report before any training.
