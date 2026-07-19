# FITWAM Related Work

The bibliography is maintained in [`fitwam_related_work.bib`](./fitwam_related_work.bib).

## World-Action Models

World-action models (WAMs) couple action prediction with objectives that model how observations evolve under action. UVA and UWM established unified video-action representations and coupled video-action diffusion \citep{li2025uva,zhu2025uwm}. DreamZero scaled this formulation with a pretrained video diffusion backbone and showed that video-only demonstrations can transfer physical behavior across embodiments \citep{ye2026dreamzero}. Fast-WAM then isolated the role of future imagination, showing that video co-training can retain most of its control benefit when future generation is removed from action inference \citep{yuan2026fastwam}.

Recent work studies how predictive representations should be exposed to control. AGRA aligns WAM representations with low-level action structure, while World Guidance and World Pilot compress future information into conditions or priors that guide action generation \citep{qiu2026agra,su2026worldguidance,lin2026worldpilot}. WALL-WM replaces arbitrary fixed clips with semantically coherent events across language, vision, and action \citep{li2026wallwm}. FITWAM uses the explicit world/action interface for self-improvement: deployment observations supervise world modeling, while event-local success/failure contrasts supervise a residual action interface.

## Post-deployment Self-Improvement

VLA post-training commonly improves deployed policies from rewards, advantages, interventions, or on-policy experience. RECAP uses value-based advantage conditioning to train on demonstrations, autonomous rollouts, and corrections; SOP organizes this process as an asynchronous deployment-learning loop \citep{amin2025pi06,pan2026sop}. RL Token exposes a compact representation from a pretrained VLA and trains a lightweight actor-critic to refine its actions online \citep{xu2026rltoken}. These methods establish policy-centered self-improvement after deployment.

WAMs expose a complementary post-deployment objective because observed transitions can directly supervise an explicit world model. World Action Verifier detects world-model errors through state plausibility and action reachability, while AdaJEPA updates a latent world model from the observed transition before replanning \citep{liu2026wav,wang2026adajepa}. World Guidance provides an action-readable interface for predicted futures, and RL Token provides a lightweight interface for residual policy improvement \citep{su2026worldguidance,xu2026rltoken}. FITWAM combines these roles in its online route: predicted futures are checked against subsequent observations to update the world interface, while reward or advantage signals update a residual action interface.

## Learning from Failed Experience

Behavior cloning treats its action targets as desirable, so autonomous failures require selective supervision. SSDF scores trajectory segments and reuses high-quality portions of imperfect demonstrations, while $\pi_{0.7}$ represents task performance through metadata-conditioned robot data \citep{wu2025ssdf,intelligence2026pi07}. DALI-R learns latent dynamics from suboptimal and failed trajectories and reranks imagined action chunks; AFIL models successful and failed action distributions and guides sampling away from failure at deployment \citep{luo2026dalir,zheng2026afil}.

FITWAM's offline route converts state-line transition scores into confidence-weighted candidate intervals, rather than assuming exact semantic subtask boundaries. Candidate intervals from failed rollouts retain world-model supervision but contribute no direct action-imitation target. Paired success/failure candidates train a compact steer representation and a residual action module through contrastive supervision; deployment requires neither an outcome label nor a separate failure generator.

## Tactile Learning for Contact-Rich Manipulation

Touch exposes contact state, force, and slip that can remain ambiguous in RGB observations. Sparsh learns reusable self-supervised tactile representations, while Sparsh-X compresses complementary visual, audio, motion, and pressure cues through multisensory bottleneck tokens \citep{higuera2024sparsh,higuera2025sparshx}. FTP-1 further scales tactile representations into a generalist policy across sensors and contact-rich tasks \citep{yuan2026ftp1}. These works motivate a compact tactile interface rather than treating raw tactile streams as additional image channels.

Tactile-conditioned action models provide the closest architectural line. VTAM uses tactile perception as a grounding stream for video-action modeling, while Tactile-WAM models touch changes through asymmetric cross-modal attention \citep{yuan2026vtam,wu2026tactilewam}. FITWAM treats contact feedback as event evidence for both world prediction and failure-localized residual correction, connecting tactile WAM training with deployment-driven self-improvement.

## Positioning

FITWAM formulates WAM self-improvement as two complementary routes. The online route updates world and residual-action interfaces from newly observed transitions, rewards, and advantages. The offline route filters rollout experience into a replay buffer of soft failure-event candidates and learns from success/failure contrast without directly imitating failed actions.
