# FITWAM Related Work

This draft keeps three lines of work that directly constrain the paper claim. The bibliography is in [`fitwam_related_work.bib`](./fitwam_related_work.bib).

## Positioning

| Closest work | What it already establishes | FITWAM's remaining question |
|---|---|---|
| Fast-WAM | World supervision during training with action-only inference | Can offline failures improve this action policy? |
| WALL-WM | Events are better temporal units than arbitrary fixed clips | Can event-local failure labels support policy improvement? |
| World Pilot | Latent and action priors from a WAM can steer a VLA | Can the steer be distilled specifically from localized failures? |
| $\pi_{0.7}$ | Metadata and subgoals can steer a generalist policy | Can a steer be learned specifically from failed actions? |
| DALI-R | Failed trajectories can train a latent world model for action reranking | Can their signal be compressed offline without test-time imagination? |
| AFIL | Failure actions can provide negative guidance | Can this work with one deployed generator and no failure input at inference? |
| Sparsh-X | Bottleneck tokens can compact multisensory interaction information | Can a bottleneck token carry success-versus-failure action structure? |

RL Token and AdaJEPA inform the later online stage. They are not used to justify the current offline contribution.

## Paper-ready text

### World-Action Models and Event-Grounded Learning

World-action models (WAMs) couple predictive world representations with executable robot actions and now span cascaded and joint architectures \citep{wang2026wamsurvey}. Early unified formulations already exposed the main design choices. UVA learns a shared video-action latent with decoupled diffusion heads, UWM couples independently noised video and action diffusion in one transformer, VideoVLA jointly forecasts actions and visual outcomes, and Cosmos Policy represents actions, future states, and values inside a pretrained video model \citep{li2025uva,zhu2025uwm,shen2025videovla,kim2026cosmospolicy}. These works establish video-model pretraining and joint future/action prediction as a policy-learning interface rather than a separate visual-planning module.

Recent joint WAMs scale that interface in different ways. Motus supports video generation, inverse dynamics, and action prediction through specialist transformer components; MotuBrain extends it with multiview, language, cross-embodiment action, and deployment machinery \citep{bi2025motus,motubrain2026}. DreamZero turns a pretrained video diffusion model into a zero-shot policy, while LingBot-VA uses causal video-action diffusion with closed-loop asynchronous execution \citep{ye2026dreamzero,li2026lingbotva}. These systems define the joint video-action branch of the literature, but do not make failed interaction segments an explicit action-learning signal.

A second branch asks how much future generation must remain at deployment. Fast-WAM retains future-video supervision during training while directly sampling actions at test time; GigaWorld-Policy uses causal factorization to make video generation optional; and Being-H0.7 transfers future information through latent queries without visual rollout \citep{yuan2026fastwam,ye2026gigaworld,luo2026beingh07}. Light-WAM and Efficient-WAM compress the video/action experts or future tokens, while Flash-WAM distills iterative joint denoising to one step per modality \citep{li2026lightwam,li2026efficientwam,akbari2026flashwam}. FITWAM inherits Fast-WAM's action-only inference regime; inference efficiency itself is not our novelty.

Other WAMs replace dense future video with more action-readable predictive structure. World Guidance predicts compact condition-space futures, LaWAM uses latent visual subgoals, ImageWAM repurposes image-editing features, and Bridge-WA distills future tokens, change maps, and motion flow into an action policy \citep{su2026worldguidance,chen2026lawam,zhang2026imagewam,bai2026bridgewa}. AGRA directly regularizes the world-action interface so video features attend to task-relevant interaction regions, while EgoWAM shows that agent-invariant feature and motion targets transfer in-the-wild human video more effectively than pixel prediction \citep{qiu2026agra,li2026egowam}. This line is especially relevant to FITWAM because it shows that extra trajectories and plausible future pixels help only when their world target is organized for control.

Steering and temporal organization form the remaining axis. World Pilot routes WAM priors into a VLA through latent and action steering, while Temporal Ratio shows that an action head's reliance on future latents varies with task phase and can drive adaptive guidance \citep{lin2026worldpilot,mishra2026temporalratio}. WALL-WM changes the training unit itself, replacing arbitrary fixed windows with semantically coherent task, subtask, action, and segment events \citep{li2026wallwm}. FITWAM therefore does not claim steering or event decomposition alone. It addresses outcome credit assignment: a failed episode often contains a correct prefix, a short failure-relevant interaction, and an uninformative terminal tail. EveRobot is designed to record episode provenance, event boundaries, and soft subtask scores so failure supervision can target the relevant interaction rather than every clip in the episode.

### Learning from Autonomous Failures and Robot Post-training

RLDS, LeRobot, and ARIO provide standardized episodic storage, robot-learning pipelines, and unified heterogeneous robot data \citep{ramos2021rlds,cadene2026lerobot,wang2024ario}. EveRobot builds on this data layer rather than replacing it: its scope is the provenance of repeated policy rollouts, overlapping interaction events, outcome labels, and reproducible round/subset manifests required by self-improvement experiments.

Mixed-quality demonstrations contain useful behavior as well as actions that should not be imitated. SSDF addresses this by estimating segment quality and selecting reusable parts of imperfect robot demonstrations \citep{wu2025ssdf}. Recent foundation policies use richer context to consume such heterogeneous data: $\pi_{0.7}$ conditions on task-performance metadata, subgoal images, and other strategy information, while $\pi^{*}_{0.6}$ combines demonstrations, autonomous experience, corrections, and advantage conditioning \citep{intelligence2026pi07,intelligence2025pi06star}. These results establish that outcome and strategy context can steer a policy; they do not isolate how a WAM should learn directly from the action structure of failed interaction segments.

DALI-R is the closest world-model comparison for mixed-quality trajectories: it learns latent 3D dynamics from suboptimal and failed data, then imagines and reranks candidate action chunks at inference \citep{luo2026dalir}. AFIL is the closest negative-guidance comparison, training successful and failed action generators with a shared backbone and steering diffusion or flow sampling away from the latter \citep{zheng2026afil}. FAR places failure contrast inside a test-time retry and continual-improvement loop \citep{hao2026far}. FITWAM studies a narrower offline setting: failed actions receive no imitation or flow-matching loss, localized success and failure trajectories train an action-side contrastive representation, and one success-directed steer token conditions the deployed action expert. Inference uses neither latent reranking, an outcome label, a failure generator, critic, retry controller, nor model update.

Online adaptation is complementary. RL Token exposes a compact VLA representation to a lightweight actor-critic, whereas AdaJEPA updates a latent world model from observed transitions during plan-execute-replan control \citep{xu2026rltoken,wang2026adajepa}. They motivate a later online FITWAM stage, after the offline steer and EveRobot data loop have been validated.

### Tactile Representations and Tactile World Models

Touch directly observes contact events that may be ambiguous in RGB. Sparsh and AnyTouch learn reusable representations across tactile tasks and sensors, while Sparsh-X fuses image, audio, motion, and pressure through bottleneck tokens \citep{higuera2024sparsh,feng2025anytouch,higuera2025sparshx}. FTP-1 further studies a generalist tactile policy across sensors and embodiments \citep{yuan2026ftp1}. FITWAM borrows the compact-token principle from Sparsh-X for its steer interface; the offline simulation stage does not claim a tactile contribution.

Predictive tactile models provide the closer endpoint for the full project. Visuo-Tactile World Models predict future visual and tactile latents, VTAM jointly models video, touch, and actions, and Tactile-WAM and VT-WAM introduce tactile-aware WAM architectures for contact-rich manipulation \citep{higuera2026vtwm,yuan2026vtam,wu2026tactilewam,tian2026vtwam}. FITWAM's intended distinction is not simply adding touch to a WAM. The tactile stage must show that contact-localized failure signals improve offline or iterative policy updates beyond a tactile WAM trained only on successful trajectories.

## Claim Boundary

> FITWAM uses event-local failed trajectories as action-side negative supervision to learn a success-directed steer for a Fast-WAM policy, without imitating failed actions or requiring failure context at deployment.

No robotics paper with the exact title or established method name `UniData` was identified. ARIO is included because it is the closest verified unified-data standard; the name `UniData` should not appear without an exact source.
