#!/usr/bin/env python3
"""A1: rot6d -> quat -> rot6d round-trip test for deploy rotation conversion (H5).

Loads real action rot6d components from the spray_water dataset and verifies that
the deploy-time conversions in scripts/1/run_gr00t_client.py are identity up to
numerical precision. Any significant error would indicate a conversion bug
contributing to the sim-to-real gap.

Conversion path under test (matches real deploy):
    obs:  pose7 (xyz + quat_xyzw)  --quat_xyzw_to_rot6d-->  rot6d   (data side)
    act:  rot6d                    --rot6d_to_quat_xyzw-->  quat_xyzw (robot side)

We test three round-trips:
  1. quat_xyzw -> rot6d -> quat_xyzw   (obs path: data stores rot6d, deploy needs quat)
  2. rot6d -> quat_xyzw -> rot6d       (act path: model predicts rot6d, robot needs quat)
  3. Full eef9 round-trip via eef9_to_astribot_pose / astribot_pose_to_eef9

We also measure the orthonormality error of rot6d recovered from a quaternion,
which tells us how much the Gram-Schmidt step in rot6d_to_quat_xyzw "repairs"
non-orthonormal inputs (relevant for H2, where the model may output
non-orthonormal rot6d after denormalization).

Usage:
    python scripts/diagnose/rot6d_roundtrip_test.py \
        --data-dir data/spray_water_rot6d_rosbag_ts_filter \
        --num-frames 100
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


# ---------------------------------------------------------------------------
# Reproduce the exact conversion code from scripts/1/run_gr00t_client.py
# (lines 159-258). Duplicated here so the test is self-contained and does not
# require importing ROS/Astribot dependencies.
# ---------------------------------------------------------------------------


def _normalize(vector: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-8:
        return fallback.astype(np.float32)
    return (vector / norm).astype(np.float32)


def _quat_wxyz_to_matrix(quat_wxyz: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat_wxyz, dtype=np.float64)
    norm = np.linalg.norm(quat)
    if norm <= 1e-12:
        quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    else:
        quat = quat / norm
    if quat[0] < 0.0:
        quat = -quat
    w, x, y, z = quat
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def _matrix_to_quat_xyzw(matrix: np.ndarray) -> np.ndarray:
    m = np.asarray(matrix, dtype=np.float64)
    trace = float(np.trace(m))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (m[2, 1] - m[1, 2]) / s
        qy = (m[0, 2] - m[2, 0]) / s
        qz = (m[1, 0] - m[0, 1]) / s
    else:
        diag = np.diagonal(m)
        idx = int(np.argmax(diag))
        if idx == 0:
            s = math.sqrt(max(1.0 + m[0, 0] - m[1, 1] - m[2, 2], 1e-12)) * 2.0
            qw = (m[2, 1] - m[1, 2]) / s
            qx = 0.25 * s
            qy = (m[0, 1] + m[1, 0]) / s
            qz = (m[0, 2] + m[2, 0]) / s
        elif idx == 1:
            s = math.sqrt(max(1.0 + m[1, 1] - m[0, 0] - m[2, 2], 1e-12)) * 2.0
            qw = (m[0, 2] - m[2, 0]) / s
            qx = (m[0, 1] + m[1, 0]) / s
            qy = 0.25 * s
            qz = (m[1, 2] + m[2, 1]) / s
        else:
            s = math.sqrt(max(1.0 + m[2, 2] - m[0, 0] - m[1, 1], 1e-12)) * 2.0
            qw = (m[1, 0] - m[0, 1]) / s
            qx = (m[0, 2] + m[2, 0]) / s
            qy = (m[1, 2] + m[2, 1]) / s
            qz = 0.25 * s

    quat = np.array([qx, qy, qz, qw], dtype=np.float64)
    norm = np.linalg.norm(quat)
    if norm <= 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    quat = quat / norm
    if quat[3] < 0.0:
        quat = -quat
    return quat.astype(np.float32)


def quat_xyzw_to_rot6d(quat_xyzw: np.ndarray) -> np.ndarray:
    quat_xyzw = np.asarray(quat_xyzw, dtype=np.float32)
    quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
    return _quat_wxyz_to_matrix(quat_wxyz)[:2, :].reshape(-1).astype(np.float32)


def rot6d_to_quat_xyzw(rot6d: np.ndarray) -> np.ndarray:
    rows = np.asarray(rot6d, dtype=np.float32).reshape(2, 3)
    row0 = _normalize(rows[0], np.array([1.0, 0.0, 0.0], dtype=np.float32))
    row1 = rows[1] - float(np.dot(rows[1], row0)) * row0
    row1 = _normalize(row1, np.array([0.0, 1.0, 0.0], dtype=np.float32))
    row2 = np.cross(row0, row1).astype(np.float32)
    rotation = np.stack([row0, row1, row2], axis=0)
    return _matrix_to_quat_xyzw(rotation)


def eef9_to_astribot_pose(eef9: np.ndarray) -> np.ndarray:
    eef9 = np.asarray(eef9, dtype=np.float32)
    quat_xyzw = rot6d_to_quat_xyzw(eef9[3:])
    return np.concatenate([eef9[:3], quat_xyzw]).astype(np.float32)


def astribot_pose_to_eef9(pose7: np.ndarray | list[float]) -> np.ndarray:
    pose7 = np.asarray(pose7, dtype=np.float32)
    rot6d = quat_xyzw_to_rot6d(pose7[3:])
    return np.concatenate([pose7[:3], rot6d]).astype(np.float32)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    """Recover a 3x3 rotation matrix from rot6d using the SAME Gram-Schmidt
    procedure as rot6d_to_quat_xyzw, so orthonormality of the result reflects
    what the robot actually receives."""
    rows = np.asarray(rot6d, dtype=np.float64).reshape(2, 3)
    row0 = rows[0] / max(np.linalg.norm(rows[0]), 1e-12)
    row1 = rows[1] - float(np.dot(rows[1], row0)) * row0
    row1 = row1 / max(np.linalg.norm(row1), 1e-12)
    row2 = np.cross(row0, row1)
    return np.stack([row0, row1, row2], axis=0)


def orthonormality_error(rot6d: np.ndarray) -> float:
    """||R R^T - I||_F for the matrix recovered from rot6d (before GS repair)."""
    rows = np.asarray(rot6d, dtype=np.float64).reshape(2, 3)
    # Build a 3x3 matrix WITHOUT Gram-Schmidt to measure raw deviation.
    row2 = np.cross(rows[0], rows[1])
    R = np.stack([rows[0], rows[1], row2], axis=0)
    I = np.eye(3, dtype=np.float64)
    return float(np.linalg.norm(R @ R.T - I, ord="fro"))


def quat_distance(q1: np.ndarray, q2: np.ndarray) -> float:
    """Geodesic distance (radians) between two quaternions (xyzw), accounting
    for double-cover sign ambiguity."""
    q1 = np.asarray(q1, dtype=np.float64)
    q2 = np.asarray(q2, dtype=np.float64)
    dot = abs(float(np.dot(q1, q2)))
    dot = min(dot, 1.0)
    return 2.0 * math.acos(dot)


def load_rot6d_samples(data_dir: str, num_frames: int) -> np.ndarray:
    """Load rot6d components (left dim 3:9, right dim 12:18) from actions.

    Returns array of shape (N, 2, 6): [sample, arm_index(0=left,1=right), rot6d_dim].
    """
    parquet_dir = Path(data_dir) / "data" / "chunk-000"
    files = sorted(parquet_dir.glob("episode_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet episodes under {parquet_dir}")

    samples: list[np.ndarray] = []
    frames_per_file = max(1, num_frames // max(1, len(files)))
    for f in files:
        if len(samples) * 2 >= num_frames:
            break
        table = pq.read_table(f)
        n = table.num_rows
        idxs = np.linspace(0, n - 1, num=min(frames_per_file, n), dtype=int)
        col = table.column("action")
        for i in idxs:
            arr = np.array(col[i].as_py(), dtype=np.float32)
            samples.append(arr[3:9])   # left rot6d
            samples.append(arr[12:18]) # right rot6d
            if len(samples) >= num_frames:
                break
    arr = np.stack(samples[:num_frames], axis=0)  # (N, 6)
    return arr.reshape(-1, 6)  # (N, 6)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_quat_to_rot6d_to_quat(rot6d_samples: np.ndarray) -> dict:
    """Round-trip 1: quat(original rot6d) -> rot6d -> quat. Measures whether the
    obs path (quat stored as rot6d, then converted back to quat for the robot)
    loses information. Because the source rot6d may already be non-orthonormal,
    we first lift rot6d -> quat (which GS-orthonormalizes), then go back."""
    # Start from a clean quaternion derived from the data rot6d.
    quats = np.array([rot6d_to_quat_xyzw(r) for r in rot6d_samples])
    rot6d_again = np.array([quat_xyzw_to_rot6d(q) for q in quats])
    quats_back = np.array([rot6d_to_quat_xyzw(r) for r in rot6d_again])

    quat_dists = np.array([quat_distance(a, b) for a, b in zip(quats, quats_back)])
    rot6d_l2 = np.linalg.norm(quat_xyzw_to_rot6d_batch(quats) - rot6d_again, axis=1)

    return {
        "test": "quat -> rot6d -> quat (obs path)",
        "quat_geodesic_rad_mean": float(quat_dists.mean()),
        "quat_geodesic_rad_max": float(quat_dists.max()),
        "quat_geodesic_deg_max": float(np.degrees(quat_dists.max())),
        "rot6d_l2_mean": float(rot6d_l2.mean()),
        "rot6d_l2_max": float(rot6d_l2.max()),
    }


def quat_xyzw_to_rot6d_batch(quats: np.ndarray) -> np.ndarray:
    return np.array([quat_xyzw_to_rot6d(q) for q in quats])


def test_rot6d_to_quat_to_rot6d(rot6d_samples: np.ndarray) -> dict:
    """Round-trip 2: rot6d -> quat -> rot6d. This is the action deploy path:
    the model predicts rot6d, the robot converts to quat. We check whether
    converting back to rot6d recovers the input. Discrepancy comes from
    Gram-Schmidt orthonormalization in rot6d_to_quat_xyzw."""
    quats = np.array([rot6d_to_quat_xyzw(r) for r in rot6d_samples])
    rot6d_back = np.array([quat_xyzw_to_rot6d(q) for q in quats])

    l2 = np.linalg.norm(rot6d_samples.astype(np.float64) - rot6d_back.astype(np.float64), axis=1)
    # Angle between the rotations (geodesic) ignoring representation.
    q_in = np.array([rot6d_to_quat_xyzw(r) for r in rot6d_samples])
    q_out = np.array([rot6d_to_quat_xyzw(r) for r in rot6d_back])
    geod = np.array([quat_distance(a, b) for a, b in zip(q_in, q_out)])

    return {
        "test": "rot6d -> quat -> rot6d (act path)",
        "rot6d_l2_mean": float(l2.mean()),
        "rot6d_l2_max": float(l2.max()),
        "rot6d_geodesic_deg_mean": float(np.degrees(geod.mean())),
        "rot6d_geodesic_deg_max": float(np.degrees(geod.max())),
    }


def test_eef9_roundtrip(rot6d_samples: np.ndarray) -> dict:
    """Round-trip 3: eef9 -> pose7 -> eef9 with a fixed xyz (position is passthrough,
    so this isolates the rot6d portion)."""
    xyz = np.array([0.4, 0.2, 1.0], dtype=np.float32)
    eef9_in = np.stack([np.concatenate([xyz, r]) for r in rot6d_samples])
    pose7 = np.array([eef9_to_astribot_pose(e) for e in eef9_in])
    eef9_out = np.array([astribot_pose_to_eef9(p) for p in pose7])

    pos_l2 = np.linalg.norm(eef9_in[:, :3] - eef9_out[:, :3], axis=1)
    rot_l2 = np.linalg.norm(eef9_in[:, 3:] - eef9_out[:, 3:], axis=1)

    return {
        "test": "eef9 -> pose7 -> eef9 (full deploy)",
        "pos_l2_max": float(pos_l2.max()),
        "rot6d_l2_mean": float(rot_l2.mean()),
        "rot6d_l2_max": float(rot_l2.max()),
    }


def test_raw_orthonormality(rot6d_samples: np.ndarray) -> dict:
    """How orthonormal is the rot6d stored in the dataset? This is a baseline
    for H2: if the raw data rot6d is already slightly non-orthonormal, the
    Gram-Schmidt in rot6d_to_quat_xyzw changes orientation on every deploy."""
    errs = np.array([orthonormality_error(r) for r in rot6d_samples])
    # Also measure how much the GS step changes the rotation, in degrees.
    q_raw = np.array([rot6d_to_quat_xyzw(r) for r in rot6d_samples])
    # Build a "raw" quaternion directly from the non-GS matrix.
    raw_geod = []
    for r in rot6d_samples:
        rows = np.asarray(r, dtype=np.float64).reshape(2, 3)
        row2 = np.cross(rows[0], rows[1])
        R_raw = np.stack([rows[0], rows[1], row2], axis=0)
        q_raw_no_gs = _matrix_to_quat_xyzw(R_raw.astype(np.float32))
        raw_geod.append(quat_distance(q_raw_no_gs, rot6d_to_quat_xyzw(r)))
    raw_geod = np.array(raw_geod)

    return {
        "test": "raw data rot6d orthonormality + GS-induced rotation change",
        "orthonormality_err_mean": float(errs.mean()),
        "orthonormality_err_max": float(errs.max()),
        "gs_rotation_change_deg_mean": float(np.degrees(raw_geod.mean())),
        "gs_rotation_change_deg_max": float(np.degrees(raw_geod.max())),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        default="data/spray_water_rot6d_rosbag_ts_filter",
        help="Path to the spray_water LeRobot dataset root.",
    )
    parser.add_argument("--num-frames", type=int, default=100)
    parser.add_argument(
        "--quat-rt-tol-rad",
        type=float,
        default=1e-3,
        help="Tolerance (radians) for quaternion round-trip geodesic error.",
    )
    parser.add_argument(
        "--rot6d-orth-tol",
        type=float,
        default=1e-2,
        help="Tolerance for raw rot6d orthonormality error (Frobenius).",
    )
    args = parser.parse_args()

    print(f"Loading rot6d samples from {args.data_dir} (target {args.num_frames})...")
    samples = load_rot6d_samples(args.data_dir, args.num_frames)
    print(f"Loaded {len(samples)} rot6d samples (left+right arms combined).")

    results = [
        test_quat_to_rot6d_to_quat(samples),
        test_rot6d_to_quat_to_rot6d(samples),
        test_eef9_roundtrip(samples),
        test_raw_orthonormality(samples),
    ]

    print("\n" + "=" * 78)
    print("A1: deploy rotation conversion round-trip results")
    print("=" * 78)
    for r in results:
        print(f"\n[{r['test']}]")
        for k, v in r.items():
            if k == "test":
                continue
            print(f"  {k:35s} = {v:.6e}")

    # Verdict
    quat_rt = results[0]
    raw_orth = results[3]
    print("\n" + "-" * 78)
    print("VERDICT (H5: deploy rotation conversion bug):")
    h5_fail = quat_rt["quat_geodesic_rad_max"] > args.quat_rt_tol_rad
    print(
        f"  quat round-trip max geodesic = {quat_rt['quat_geodesic_deg_max']:.5f} deg "
        f"(tolerance {np.degrees(args.quat_rt_tol_rad):.5f} deg) -> "
        f"{'FAIL' if h5_fail else 'PASS'}"
    )
    print(
        f"  raw rot6d orthonormality err  = {raw_orth['orthonormality_err_max']:.5e} "
        f"(tolerance {args.rot6d_orth_tol:.0e})"
    )
    print(
        f"  GS changes rotation by up to {raw_orth['gs_rotation_change_deg_max']:.5f} deg "
        f"(mean {raw_orth['gs_rotation_change_deg_mean']:.5f} deg)"
    )
    if not h5_fail:
        print("  -> deploy rotation conversion is numerically correct (no H5 bug).")
        print("     BUT: if the model outputs non-orthonormal rot6d, GS will change the")
        print("     commanded orientation by the 'GS rotation change' magnitude above.")
        print("     This is relevant to H2 (normalization) rather than H5 (conversion bug).")
    return 0 if not h5_fail else 1


if __name__ == "__main__":
    sys.exit(main())
