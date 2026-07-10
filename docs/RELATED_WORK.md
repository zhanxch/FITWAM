# FITWAM Related Work

This draft keeps three lines of work that directly constrain the paper claim. The bibliography is in [`fitwam_related_work.bib`](./fitwam_related_work.bib).

## Positioning

| Closest work | What it already establishes | FITWAM's remaining question |
|---|---|---|
| Fast-WAM | World supervision during training with action-only inference | Can offline failures improve this action policy? |
| WALL-WM | Events are better temporal units than arbitrary fixed clips | Can event-local failure labels support policy improvement? |
| $\pi_{0.7}$ | Metadata and subgoals can steer a generalist policy | Can a steer be learned specifically from failed actions? |
| AFIL | Failure actions can provide negative guidance | Can this work with one deployed generator and no failure input at inference? |
| Sparsh-X | Bottleneck tokens can compact multisensory interaction information | Can a bottleneck token carry success-versus-failure action structure? |

RL Token and AdaJEPA inform the later online stage. They are not used to justify the current offline contribution.

## Paper-ready text

### World-Action Models and Event-Grounded Learning

World-action models (WAMs) couple predictive world representations with robot action generation. Motus unifies several video, inverse-dynamics, and action modes through specialist transformer components, while DreamZero and LingBot-VA turn video generative models into closed-loop robot policies \citep{bi2025motus,ye2026dreamzero,li2026lingbotva}. Fast-WAM studies a particularly relevant design point: future-video prediction remains a training objective, but deployment directly samples actions without generating a future video \citep{yuan2026fastwam}. World Guidance further shows that action-relevant conditions can replace photorealistic future generation \citep{su2026worldguidance}. These works motivate FITWAM's backbone, but the use of world supervision or action-only inference is not our contribution.

Most WAMs still train on fixed windows whose boundaries need not match physical interactions. WALL-WM instead treats semantically coherent events as the unit of video-action learning and organizes data around task, subtask, action, and segment structure \citep{li2026wallwm}. FITWAM adopts this event-centered view for data organization. Its additional problem is outcome-aware credit assignment: a failed episode often contains a correct prefix, a short failure-relevant interaction, and an uninformative terminal tail. EveRobot therefore records episode provenance, event boundaries, and soft subtask scores so that failure supervision can be applied to the relevant interaction rather than every clip in the episode.

### Learning from Autonomous Failures and Robot Post-training

RLDS, LeRobot, and ARIO provide standardized episodic storage, robot-learning pipelines, and unified heterogeneous robot data \citep{ramos2021rlds,cadene2026lerobot,wang2024ario}. EveRobot builds on this data layer rather than replacing it: its scope is the provenance of repeated policy rollouts, overlapping interaction events, outcome labels, and reproducible round/subset manifests required by self-improvement experiments.

Mixed-quality demonstrations contain useful behavior as well as actions that should not be imitated. SSDF addresses this by estimating segment quality and selecting reusable parts of imperfect robot demonstrations \citep{wu2025ssdf}. Recent foundation policies use richer context to consume such heterogeneous data: $\pi_{0.7}$ conditions on task-performance metadata, subgoal images, and other strategy information, while $\pi^{*}_{0.6}$ combines demonstrations, autonomous experience, corrections, and advantage conditioning \citep{intelligence2026pi07,intelligence2025pi06star}. These results establish that outcome and strategy context can steer a policy; they do not isolate how a WAM should learn directly from the action structure of failed interaction segments.

AFIL is the closest mechanism-level comparison. It trains successful and failed action generators with a shared backbone and uses their difference as adaptive negative guidance during diffusion or flow sampling \citep{zheng2026afil}. FAR likewise learns from failure contrast, but places it inside a test-time retry and continual-improvement loop \citep{hao2026far}. FITWAM studies a narrower offline setting: failed actions receive no imitation or flow-matching loss, localized success and failure trajectories train an action-side contrastive representation, and one success-directed steer token conditions the deployed action expert. Inference uses neither an outcome label nor a failure generator, critic, retry controller, or model update.

Online adaptation is complementary. RL Token exposes a compact VLA representation to a lightweight actor-critic, whereas AdaJEPA updates a latent world model from observed transitions during plan-execute-replan control \citep{xu2026rltoken,wang2026adajepa}. They motivate a later online FITWAM stage, after the offline steer and EveRobot data loop have been validated.

### Tactile Representations and Tactile World Models

Touch directly observes contact events that may be ambiguous in RGB. Sparsh and AnyTouch learn reusable representations across tactile tasks and sensors, while Sparsh-X fuses image, audio, motion, and pressure through bottleneck tokens \citep{higuera2024sparsh,feng2025anytouch,higuera2025sparshx}. FTP-1 further studies a generalist tactile policy across sensors and embodiments \citep{yuan2026ftp1}. FITWAM borrows the compact-token principle from Sparsh-X for its steer interface; the offline simulation stage does not claim a tactile contribution.

Predictive tactile models provide the closer endpoint for the full project. Visuo-Tactile World Models predict future visual and tactile latents, VTAM jointly models video, touch, and actions, and Tactile-WAM and VT-WAM introduce tactile-aware WAM architectures for contact-rich manipulation \citep{higuera2026vtwm,yuan2026vtam,wu2026tactilewam,tian2026vtwam}. FITWAM's intended distinction is not simply adding touch to a WAM. The tactile stage must show that contact-localized failure signals improve offline or iterative policy updates beyond a tactile WAM trained only on successful trajectories.

## Claim Boundary

> FITWAM uses event-local failed trajectories as action-side negative supervision to learn a success-directed steer for a Fast-WAM policy, without imitating failed actions or requiring failure context at deployment.

No robotics paper with the exact title or established method name `UniData` was identified. ARIO is included because it is the closest verified unified-data standard; the name `UniData` should not appear without an exact source.
