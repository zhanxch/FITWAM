#!/usr/bin/env python3
"""C1: verify whether action tokens cross-attend to video latents at inference (H3).

This is a STATIC code analysis (no GPU/model needed). It inspects
FastWAM.infer_action and the MoT attention mask construction to determine
what the action expert can actually "see" when predicting actions:

  - Does the action expert attend to the INPUT image (first video frame)? -> YES
  - Does the action expert attend to the IMAGINED future video? -> NO

If the action expert only sees the first frame + text (and proprio when
enabled), then the "world model future imagination" never reaches the action
expert at inference. This structurally explains the observed failure mode
where the predicted VIDEO shows the left hand reaching for the trigger
(video expert correct) but the ACTION stays stuck at the spray bottle
(action expert wrong): the two experts are decoupled at inference time.

Implication for the sim-vs-real gap:
  - In SIM (DexJoCo), the single input frame is a clean, deterministic MuJoCo
    render with rich, unambiguous state -> first-frame conditioning is enough
    for the action expert to predict correct actions.
  - In REAL (spray_water), the single frame is a noisy ROS-compressed image
    with lighting/occlusion/calibration noise -> first-frame conditioning is
    weaker, and WITHOUT proprio (H1) the action expert has no absolute pose
    anchor. The video imagination that COULD disambiguate the scene is not
    fed to the action expert.
  - GR00T/pi0 always feed robot state to the action head; they do not rely on
    future video imagination either, but the state conditioning compensates.

Usage:
    python scripts/diagnose/action_video_coupling_analysis.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
FASTWAM = REPO / "src" / "fastwam" / "models" / "wan22" / "fastwam.py"
MOT = REPO / "src" / "fastwam" / "models" / "wan22" / "mot.py"


def read_lines(path: Path, start: int, end: int) -> str:
    lines = path.read_text().splitlines()
    return "\n".join(f"{i+1:4d}| {lines[i]}" for i in range(start - 1, min(end, len(lines))))


def main() -> int:
    print("=" * 78)
    print("C1: action-video coupling analysis at inference (H3)")
    print("=" * 78)

    print("\n[1] infer_action video cache source (fastwam.py:1213-1276)")
    print("-" * 78)
    print(read_lines(FASTWAM, 1213, 1220))
    print("  ...")
    print(read_lines(FASTWAM, 1251, 1280))
    print("\n  KEY: video_pre is built from `first_frame_latents` ONLY (the input image).")
    print("       The video K/V cache passed to the action branch contains ONLY the")
    print("       first-frame tokens. No imagined/forecast future video is encoded.")

    print("\n[2] attention mask: what action tokens can attend to (fastwam.py:504-525)")
    print("-" * 78)
    print(read_lines(FASTWAM, 504, 525))
    print("\n  KEY: mask[video_seq_len:, :first_frame_tokens] = True")
    print("       Action queries attend to the FIRST FRAME of video tokens ONLY.")
    print("       They do NOT attend to any future/imagined video frames.")

    print("\n[3] mixed attention uses cached video K/V (mot.py:425-433)")
    print("-" * 78)
    print(read_lines(MOT, 425, 433))
    print("\n  KEY: k_cat = [k_video (first frame only), k_action]. Action attends to")
    print("       first-frame video + action self. Confirms coupling is first-frame only.")

    print("\n[4] does infer_action ever call the video diffusion loop? (fastwam.py:1281-1316)")
    print("-" * 78)
    print(read_lines(FASTWAM, 1281, 1316))
    print("\n  KEY: The loop only denoises `latents_action`. There is NO video latent")
    print("       denoising in infer_action. The video expert is run once (pre_dit on the")
    print("       first frame) to build the K/V cache, then frozen. Future imagination is")
    print("       NOT generated during action inference.")

    print("\n[5] contrast with infer_joint (fastwam.py:978) which DOES generate video")
    print("-" * 78)
    # find infer_joint signature
    txt = FASTWAM.read_text()
    m = re.search(r"def infer_joint\(", txt)
    if m:
        line_no = txt[: m.start()].count("\n") + 1
        print(read_lines(FASTWAM, line_no, line_no + 6))
    print("  infer_joint co-denoises video + action (the 'test-time future imagination').")
    print("  But the DEPLOY server (run_fastwam_server.py / run_gr00t_client.py) calls")
    print("  infer_action, NOT infer_joint. So imagination is never used at deploy time.")

    print("\n" + "=" * 78)
    print("VERDICT (C1 / H3):")
    print("=" * 78)
    print("  CONFIRMED: At inference (infer_action), the action expert attends ONLY to")
    print("  the first-frame video tokens (input image) + text context (+ proprio if set).")
    print("  It does NOT attend to imagined future video frames, and infer_action does not")
    print("  run the video diffusion loop at all.")
    print()
    print("  => The 'video expert correct, action expert wrong' failure mode is STRUCTURAL:")
    print("     the two experts are decoupled at deploy. The video imagination that the")
    print("     paper pitches as 'test-time future imagination' is only exercised in")
    print("     infer_joint, which the real/sim deploy servers do not call.")
    print()
    print("  => For the sim-vs-real gap: this is an EQUAL-OPPORTUNITY limitation (affects")
    print("     sim and real identically). It does NOT by itself explain why sim > real.")
    print("     But it COMPOUNDS with H1 (no proprio): in sim the single clean frame is")
    print("     enough; in real the single noisy frame + no proprio + no imagination leaves")
    print("     the action expert with weak conditioning -> drift / oscillation.")
    print()
    print("  => If you want imagination to help actions, switch deploy to infer_joint or")
    print("     add a 2-stage pipeline: infer video first, then infer_action conditioned on")
    print("     the imagined frames (would require extending the attention mask).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
