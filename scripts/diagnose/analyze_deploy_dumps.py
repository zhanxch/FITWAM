#!/usr/bin/env python3
"""B1 analysis: inspect server/client dumps to localize action failure.

After running the real robot with:
  - server: python scripts/1/run_fastwam_server.py ... --dump-dir runs/diag_dump
  - client: ... --dump-raw-actions runs/diag_dump/client_raw

this script reads the per-step .npz dumps and reports:

  1. Confirm H1: is `normalized_proprio` None in every server dump? (it should be,
     given proprio_dim=null in the task config).
  2. Inspect H2 on REAL MODEL OUTPUT: take the denormalized action's rot6d
     components and measure orthonormality + GS-induced rotation change. If the
     model's actual output is non-orthonormal, the deploy GS step changes the
     commanded orientation -> explains the oscillation/stuck behavior.
  3. Inspect the hand joints: are they in a "closed/grasp" range when the robot
     is supposed to be gripping? (relates to the "drops the bottle" symptom).
  4. Compare server dump vs client dump (if both present) to check whether
     split_wuji_action / transport changed anything.

Usage:
    python scripts/diagnose/analyze_deploy_dumps.py --server-dir runs/diag_dump
    python scripts/diagnose/analyze_deploy_dumps.py --server-dir runs/diag_dump --client-dir runs/diag_dump/client_raw
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from rot6d_roundtrip_test import rot6d_to_quat_xyzw, quat_distance  # noqa: E402
from normalization_rot6d_test import orthonormality_error, gs_rotation_change_deg  # noqa: E402


def load_server_dumps(server_dir: Path) -> list[Path]:
    return sorted(server_dir.glob("step_*.npz"))


def load_client_dumps(client_dir: Path) -> list[Path]:
    return sorted(client_dir.glob("raw_actions_loop_*.npz"))


def analyze_server_dump(npz_path: Path) -> dict:
    d = np.load(npz_path, allow_pickle=True)
    out: dict = {"file": npz_path.name, "step": int(d["step"]) if "step" in d else -1}
    # H1: proprio presence
    out["use_proprio"] = bool(d["use_proprio"]) if "use_proprio" in d else None
    if "raw_proprio_is_none" in d:
        out["raw_proprio_is_none"] = bool(d["raw_proprio_is_none"])
    if "normalized_proprio_is_none" in d:
        out["normalized_proprio_is_none"] = bool(d["normalized_proprio_is_none"])
    # action
    if "action_denorm" in d:
        act = d["action_denorm"]
        if act.ndim == 1:
            act = act[None, :]
        out["action_shape"] = act.shape
        # H2: rot6d orthonormality of the MODEL's actual output
        left_rot6d = act[:, 3:9]
        right_rot6d = act[:, 12:18]
        out["left_rot6d_orth_err_mean"] = float(np.mean([orthonormality_error(r) for r in left_rot6d]))
        out["left_rot6d_orth_err_max"] = float(np.max([orthonormality_error(r) for r in left_rot6d]))
        out["right_rot6d_orth_err_mean"] = float(np.mean([orthonormality_error(r) for r in right_rot6d]))
        out["right_rot6d_orth_err_max"] = float(np.max([orthonormality_error(r) for r in right_rot6d]))
        out["left_gs_change_deg_max"] = float(np.max([gs_rotation_change_deg(r) for r in left_rot6d]))
        out["right_gs_change_deg_max"] = float(np.max([gs_rotation_change_deg(r) for r in right_rot6d]))
        # hand joints (grasp check)
        if "left_hand_denorm" in d:
            lh = d["left_hand_denorm"]
            if lh.ndim == 1:
                lh = lh[None, :]
            out["left_hand_mean"] = float(lh.mean())
            out["left_hand_min"] = float(lh.min())
            out["left_hand_max"] = float(lh.max())
        if "right_hand_denorm" in d:
            rh = d["right_hand_denorm"]
            if rh.ndim == 1:
                rh = rh[None, :]
            out["right_hand_mean"] = float(rh.mean())
            out["right_hand_min"] = float(rh.min())
            out["right_hand_max"] = float(rh.max())
        # eef position range (stuck/oscillation check)
        if "left_eef_denorm" in d:
            le = d["left_eef_denorm"]
            if le.ndim == 1:
                le = le[None, :]
            out["left_eef_xyz_mean"] = le[:, :3].mean(axis=0).tolist()
            out["left_eef_xyz_std"] = le[:, :3].std(axis=0).tolist()
        if "right_eef_denorm" in d:
            re = d["right_eef_denorm"]
            if re.ndim == 1:
                re = re[None, :]
            out["right_eef_xyz_mean"] = re[:, :3].mean(axis=0).tolist()
            out["right_eef_xyz_std"] = re[:, :3].std(axis=0).tolist()
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--server-dir", required=True, help="Directory of server step_*.npz dumps")
    parser.add_argument("--client-dir", default=None, help="Directory of client raw_actions_loop_*.npz dumps")
    parser.add_argument("--max-steps", type=int, default=20)
    args = parser.parse_args()

    server_dir = Path(args.server_dir)
    if not server_dir.exists():
        print(f"ERROR: server dump dir not found: {server_dir}")
        return 1
    server_files = load_server_dumps(server_dir)
    if not server_files:
        print(f"ERROR: no step_*.npz files in {server_dir}")
        print("Did you run the server with --dump-dir?")
        return 1
    print(f"Found {len(server_files)} server dump files in {server_dir}")

    print("\n" + "=" * 78)
    print("B1: deploy dump analysis")
    print("=" * 78)

    # H1 confirmation
    first = np.load(server_files[0], allow_pickle=True)
    use_proprio = bool(first["use_proprio"]) if "use_proprio" in first else None
    norm_proprio_none = bool(first["normalized_proprio_is_none"]) if "normalized_proprio_is_none" in first else None
    print("\n[H1 confirmation] proprioception at deploy:")
    print(f"  use_proprio flag        = {use_proprio}")
    print(f"  normalized_proprio None = {norm_proprio_none}")
    if use_proprio is False or norm_proprio_none is True:
        print("  -> CONFIRMED H1: proprio is NOT fed to the model at deploy. The action")
        print("     expert has no absolute pose anchor; relies solely on the input image.")
    else:
        print("  -> proprio IS being used; H1 not active for this checkpoint.")

    # Per-step analysis
    print(f"\n[Per-step model output analysis] (first {min(args.max_steps, len(server_files))} steps)")
    print(f"  {'step':>5} {'L_orth_max':>12} {'R_orth_max':>12} {'L_GS_deg':>10} {'R_GS_deg':>10} "
          f"{'L_hand_mn':>10} {'R_hand_mn':>10} {'L_eef_x':>9} {'L_eef_z':>9}")
    all_orth = []
    all_gs = []
    for f in server_files[: args.max_steps]:
        r = analyze_server_dump(f)
        all_orth.append(r.get("left_rot6d_orth_err_max", 0.0))
        all_gs.append(r.get("left_gs_change_deg_max", 0.0))
        print(f"  {r.get('step', -1):>5} "
              f"{r.get('left_rot6d_orth_err_max', float('nan')):>12.4e} "
              f"{r.get('right_rot6d_orth_err_max', float('nan')):>12.4e} "
              f"{r.get('left_gs_change_deg_max', float('nan')):>10.4f} "
              f"{r.get('right_gs_change_deg_max', float('nan')):>10.4f} "
              f"{r.get('left_hand_mean', float('nan')):>10.4f} "
              f"{r.get('right_hand_mean', float('nan')):>10.4f} "
              f"{r.get('left_eef_xyz_mean', [float('nan')]*3)[0]:>9.4f} "
              f"{r.get('left_eef_xyz_mean', [float('nan')]*3)[2]:>9.4f}")

    print("\n[H2 on real model output] rot6d orthonormality + GS rotation change:")
    if all_orth:
        print(f"  left rot6d orth_err max over steps = {max(all_orth):.4e}")
        print(f"  left GS rotation change max over steps = {max(all_gs):.4f} deg")
        if max(all_orth) > 1e-2:
            print("  -> CONFIRMED H2 on real output: the model's denormalized rot6d is")
            print("     non-orthonormal, so the deploy GS step changes the commanded")
            print("     orientation by up to the GS rotation change above -> oscillation.")
        else:
            print("  -> model output rot6d is orthonormal; H2 not manifesting in output.")
            print("     (The warp still makes training harder, but output is clean.)")

    # Client dump comparison
    if args.client_dir:
        client_dir = Path(args.client_dir)
        if client_dir.exists():
            client_files = load_client_dumps(client_dir)
            print(f"\n[Client raw-action dump] {len(client_files)} files in {client_dir}")
            if client_files:
                c0 = np.load(client_files[0], allow_pickle=True)
                print(f"  keys in first client dump: {[k for k in c0.files if not k.startswith('_')]}")
                # Compare server action_denorm vs client action_left_eef for step 0
                s0 = np.load(server_files[0], allow_pickle=True)
                if "action_denorm" in s0 and "action_left_eef" in c0:
                    sa = s0["action_denorm"]
                    if sa.ndim == 1:
                        sa = sa[None, :]
                    ca = c0["action_left_eef"]
                    if ca.ndim == 3 and ca.shape[0] == 1:
                        ca = ca[0]
                    if sa.shape[0] >= 1 and ca.shape[0] >= 1:
                        server_left_eef = sa[0, 0:9]
                        client_left_eef = ca[0, 0:9]
                        diff = np.linalg.norm(server_left_eef - client_left_eef)
                        print(f"  server vs client left_eef[0] L2 diff = {diff:.6e}")
                        if diff > 1e-4:
                            print("  -> transport/split changed the action; investigate wuji_fastwam_adapter.split_wuji_action")
                        else:
                            print("  -> server and client action match; post-processing after split is the only difference.")
        else:
            print(f"\n[Client dump] dir not found: {client_dir}")

    print("\n" + "-" * 78)
    print("HOW TO USE THESE RESULTS:")
    print("  - If H1 confirmed + H2 confirmed on output: primary fixes are")
    print("    (1) enable proprio_dim=58 (official proprio context token), (2) per-modality rot6d normalization.")
    print("  - If hand joints are NOT in grasp range when they should be: the action expert")
    print("    is failing to predict a closing grasp -> consistent with H1 (no proprio, so it")
    print("    cannot know current hand state to decide close vs open).")
    print("  - If eef xyz std across the chunk is large but mean barely moves: the chunk")
    print("    oscillates around a stuck position -> consistent with H1+H2.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
