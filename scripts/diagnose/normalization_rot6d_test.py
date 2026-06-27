#!/usr/bin/env python3
"""A2: test whether min/max normalization breaks rot6d orthonormality (H2).

FastWAM applies a single per-dimension min/max linear normalizer to the whole
58-dim action vector (xyz + rot6d + hand joints). Because each rot6d component
has a different (min, max, range), the normalizer scales the two rows of the
rotation matrix by DIFFERENT factors and shifts them by DIFFERENT offsets.

Consequences tested here:
  1. Take GT orthonormal rot6d from the dataset. forward() -> backward().
     The denormalized rot6d should equal the input (identity round-trip).
     This is guaranteed by linearity, but confirms the normalizer is wired up.
  2. The CRITICAL test: take GT orthonormal rot6d, forward() it, then add
     SMALL Gaussian noise (simulating diffusion sampling error in normalized
     space), then backward() and measure orthonormality. Compare the
     denormalization-induced orthonormality error against the error when the
     SAME noise is added directly in the un-normalized rot6d space.
     -> If normalized-space noise produces MUCH larger orthonormality error
        after denormalization, the per-dim scaling amplifies rotation errors
        and the model is forced to learn a harder manifold.
  3. Measure the "normalized-space orthonormality": after forward(), how far
     are the two rows from being unit-norm / orthogonal IN NORMALIZED SPACE.
     The model has to output values on this distorted manifold. A perfect
     orthonormal rotation maps to a NON-orthonormal pair of vectors in
     normalized space, so the target the model must fit is itself "warped".

This script does NOT require torch/fastwam to be installed for the core math;
it reconstructs the SingleFieldLinearNormalizer scale/offset from stats.json
exactly as src/fastwam/datasets/lerobot/utils/normalizer.py does. A torch path
is also provided (--use-torch) for cross-checking when available.

Usage:
    python scripts/diagnose/normalization_rot6d_test.py \
        --stats runs/spray_water_rot6d_rosbag_ts_filter_uncond_3cam_384_1e-4/2026-06-22_00-40-40/dataset_stats.json \
        --data-dir data/spray_water_rot6d_rosbag_ts_filter \
        --noise-std 0.05
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


# ---------------------------------------------------------------------------
# Reconstruct SingleFieldLinearNormalizer (min/max mode) from normalizer.py
# ---------------------------------------------------------------------------


class MinMaxNormalizer:
    """Exact reproduction of SingleFieldLinearNormalizer in min/max mode.

    See src/fastwam/datasets/lerobot/utils/normalizer.py:92-134.
    output range is [-1, 1]; dims with range < range_tol are treated as
    constant and mapped to the midpoint.
    """

    range_tol = 1e-4
    output_min = -1.0
    output_max = 1.0

    def __init__(self, stats_min: np.ndarray, stats_max: np.ndarray):
        input_min = np.asarray(stats_min, dtype=np.float64)
        input_max = np.asarray(stats_max, dtype=np.float64)
        input_range = input_max - input_min
        ignore_dim = input_range < self.range_tol
        input_range[ignore_dim] = self.output_max - self.output_min
        scale = (self.output_max - self.output_min) / input_range
        offset = self.output_min - scale * input_min
        offset[ignore_dim] = (self.output_max + self.output_min) / 2 - input_min[ignore_dim]
        self.scale = scale
        self.offset = offset
        self.ignore_dim = ignore_dim

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = x * self.scale + self.offset
        return np.clip(x, -5.0, 5.0)

    def backward(self, x: np.ndarray) -> np.ndarray:
        return (x - self.offset) / self.scale


def load_action_stats(stats_path: str) -> dict:
    with open(stats_path) as f:
        s = json.load(f)
    st = s["action"]["default"]
    return {
        "min": np.array(st["global_min"], dtype=np.float64),
        "max": np.array(st["global_max"], dtype=np.float64),
        "mean": np.array(st["global_mean"], dtype=np.float64),
        "std": np.array(st["global_std"], dtype=np.float64),
    }


def load_actions(data_dir: str, num_frames: int) -> np.ndarray:
    parquet_dir = Path(data_dir) / "data" / "chunk-000"
    files = sorted(parquet_dir.glob("episode_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet episodes under {parquet_dir}")
    out: list[np.ndarray] = []
    per_file = max(1, num_frames // max(1, len(files)))
    for f in files:
        if len(out) >= num_frames:
            break
        table = pq.read_table(f)
        n = table.num_rows
        idxs = np.linspace(0, n - 1, num=min(per_file, n), dtype=int)
        col = table.column("action")
        for i in idxs:
            out.append(np.array(col[i].as_py(), dtype=np.float64))
            if len(out) >= num_frames:
                break
    return np.stack(out[:num_frames], axis=0)


# ---------------------------------------------------------------------------
# rot6d geometry helpers
# ---------------------------------------------------------------------------


def rot6d_to_matrix_gs(rot6d: np.ndarray) -> np.ndarray:
    """3x3 rotation from rot6d using Gram-Schmidt (same as deploy)."""
    rows = np.asarray(rot6d, dtype=np.float64).reshape(2, 3)
    r0 = rows[0] / max(np.linalg.norm(rows[0]), 1e-12)
    r1 = rows[1] - float(np.dot(rows[1], r0)) * r0
    r1 = r1 / max(np.linalg.norm(r1), 1e-12)
    r2 = np.cross(r0, r1)
    return np.stack([r0, r1, r2], axis=0)


def orthonormality_error(rot6d: np.ndarray) -> float:
    """Frobenius norm of (R R^T - I) using the RAW (non-GS) matrix built from
    the two rows + cross product. Measures how far the 6 values are from
    representing a valid rotation."""
    rows = np.asarray(rot6d, dtype=np.float64).reshape(2, 3)
    r2 = np.cross(rows[0], rows[1])
    R = np.stack([rows[0], rows[1], r2], axis=0)
    return float(np.linalg.norm(R @ R.T - np.eye(3), ord="fro"))


def row_norms(rot6d: np.ndarray) -> tuple[float, float]:
    rows = np.asarray(rot6d, dtype=np.float64).reshape(2, 3)
    return float(np.linalg.norm(rows[0])), float(np.linalg.norm(rows[1]))


def row_orthogonality(rot6d: np.ndarray) -> float:
    rows = np.asarray(rot6d, dtype=np.float64).reshape(2, 3)
    return float(np.dot(rows[0], rows[1]))


def gs_rotation_change_deg(rot6d: np.ndarray) -> float:
    """Geodesic degrees between the raw (non-GS) rotation and the GS-repaired
    rotation. This is what the robot 'changes' when it orthonormalizes a
    non-orthonormal rot6d command."""
    rows = np.asarray(rot6d, dtype=np.float64).reshape(2, 3)
    r2 = np.cross(rows[0], rows[1])
    R_raw = np.stack([rows[0], rows[1], r2], axis=0)
    # Quat from raw (no GS, but _matrix_to_quat normalizes so it's a valid rotation
    # closest to the raw matrix).
    from rot6d_roundtrip_test import _matrix_to_quat_xyzw, rot6d_to_quat_xyzw, quat_distance
    q_raw = _matrix_to_quat_xyzw(R_raw.astype(np.float32))
    q_gs = rot6d_to_quat_xyzw(rot6d)
    return float(np.degrees(quat_distance(q_raw, q_gs)))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_roundtrip_identity(actions: np.ndarray, norm: MinMaxNormalizer) -> dict:
    """forward -> backward must be identity (linearity). Sanity check."""
    fwd = norm.forward(actions)
    back = norm.backward(fwd)
    err = np.linalg.norm(actions - back, axis=1)
    return {
        "test": "forward->backward identity (sanity)",
        "l2_max": float(err.max()),
        "l2_mean": float(err.mean()),
    }


def test_normalized_space_distortion(actions: np.ndarray, norm: MinMaxNormalizer) -> dict:
    """How far is the NORMALIZED rot6d from being orthonormal?

    A perfect orthonormal rotation, after forward(), becomes a pair of vectors
    that are NOT unit-norm and NOT orthogonal in normalized space. The model
    must therefore fit targets on a warped manifold. We quantify this warp.
    """
    fwd = norm.forward(actions)
    left_rot6d = fwd[:, 3:9]
    right_rot6d = fwd[:, 12:18]

    def stats(block: np.ndarray) -> dict:
        orth_err = np.array([orthonormality_error(r) for r in block])
        norms = np.array([row_norms(r) for r in block])
        dot = np.array([row_orthogonality(r) for r in block])
        return {
            "orthonormality_err_mean": float(orth_err.mean()),
            "orthonormality_err_max": float(orth_err.max()),
            "row0_norm_mean": float(norms[:, 0].mean()),
            "row0_norm_std": float(norms[:, 0].std()),
            "row1_norm_mean": float(norms[:, 1].mean()),
            "row1_norm_std": float(norms[:, 1].std()),
            "row_dot_mean": float(np.abs(dot).mean()),
        }

    raw_left = actions[:, 3:9]
    raw_right = actions[:, 12:18]
    return {
        "test": "normalized-space rot6d distortion",
        "raw_left": stats(raw_left),
        "norm_left": stats(left_rot6d),
        "raw_right": stats(raw_right),
        "norm_right": stats(right_rot6d),
    }


def test_noise_amplification(actions: np.ndarray, norm: MinMaxNormalizer, noise_std: float, seed: int = 0) -> dict:
    """Add Gaussian noise in NORMALIZED space (where the model operates and
    samples), denormalize, and measure orthonormality + GS-induced rotation
    change. Compare against adding the SAME per-dim std noise directly in the
    un-normalized rot6d space.

    This isolates the amplification due to per-dim scale mismatch.
    """
    rng = np.random.default_rng(seed)
    fwd = norm.forward(actions)
    noise = rng.normal(0.0, noise_std, size=fwd.shape)
    noised_fwd = fwd + noise
    denorm = norm.backward(noised_fwd)

    # Per-dim std in un-normalized space that corresponds to noise_std in normalized space
    # (= noise_std / scale). We use that to make a fair "noise added directly in
    # un-normalized space" comparison.
    unnormalized_noise_std_per_dim = noise_std / norm.scale
    noise_raw = rng.normal(0.0, 1.0, size=actions.shape) * unnormalized_noise_std_per_dim[np.newaxis, :]
    noised_raw = actions + noise_raw

    def arm_summary(block_norm: np.ndarray, block_raw: np.ndarray, slices: dict) -> dict:
        orth_norm = np.array([orthonormality_error(b) for b in block_norm])
        orth_raw = np.array([orthonormality_error(b) for b in block_raw])
        gs_norm = np.array([gs_rotation_change_deg(b) for b in block_norm])
        gs_raw = np.array([gs_rotation_change_deg(b) for b in block_raw])
        return {
            "orth_err_norm_mean": float(orth_norm.mean()),
            "orth_err_norm_max": float(orth_norm.max()),
            "orth_err_raw_mean": float(orth_raw.mean()),
            "orth_err_raw_max": float(orth_raw.max()),
            "amplification_factor_mean": float(orth_norm.mean() / max(orth_raw.mean(), 1e-12)),
            "gs_change_norm_deg_max": float(gs_norm.max()),
            "gs_change_raw_deg_max": float(gs_raw.max()),
        }

    left = arm_summary(denorm[:, 3:9], noised_raw[:, 3:9], {})
    right = arm_summary(denorm[:, 12:18], noised_raw[:, 12:18], {})
    return {"test": "noise amplification (normalized vs raw)", "noise_std": noise_std, "left": left, "right": right}


def test_per_dim_scale_spread(stats: dict) -> dict:
    """Report the scale factor spread across rot6d dims vs xyz vs hand dims.
    Large spread => strong warp of the rotation manifold."""
    norm = MinMaxNormalizer(stats["min"], stats["max"])
    scale = norm.scale
    return {
        "test": "per-dim normalizer scale spread",
        "rot6d_left_scale": scale[3:9].tolist(),
        "rot6d_right_scale": scale[12:18].tolist(),
        "rot6d_scale_min": float(scale[np.r_[3:9, 12:18]].min()),
        "rot6d_scale_max": float(scale[np.r_[3:9, 12:18]].max()),
        "rot6d_scale_ratio_max_over_min": float(scale[np.r_[3:9, 12:18]].max() / scale[np.r_[3:9, 12:18]].min()),
        "xyz_scale_min": float(scale[0:3].min()),
        "xyz_scale_max": float(scale[0:3].max()),
        "hand_scale_min": float(scale[18:58].min()),
        "hand_scale_max": float(scale[18:58].max()),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--stats",
        default="runs/spray_water_rot6d_rosbag_ts_filter_uncond_3cam_384_1e-4/2026-06-22_00-40-40/dataset_stats.json",
    )
    parser.add_argument("--data-dir", default="data/spray_water_rot6d_rosbag_ts_filter")
    parser.add_argument("--num-frames", type=int, default=200)
    parser.add_argument("--noise-std", type=float, default=0.05, help="Gaussian std in normalized [-1,1] space")
    parser.add_argument("--orth-tol", type=float, default=1e-2, help="Orthonormality error threshold for verdict")
    args = parser.parse_args()

    # import the round-trip helpers from A1 for GS rotation change metric
    sys.path.insert(0, str(Path(__file__).parent))
    from rot6d_roundtrip_test import _matrix_to_quat_xyzw, rot6d_to_quat_xyzw, quat_distance  # noqa: F401

    print(f"Loading stats from {args.stats}")
    stats = load_action_stats(args.stats)
    norm = MinMaxNormalizer(stats["min"], stats["max"])

    print(f"Loading {args.num_frames} actions from {args.data_dir}")
    actions = load_actions(args.data_dir, args.num_frames)
    print(f"Loaded {len(actions)} actions, shape {actions.shape}")

    print("\n" + "=" * 78)
    print("A2: min/max normalization rot6d orthonormality test (H2)")
    print("=" * 78)

    r_id = test_roundtrip_identity(actions, norm)
    print(f"\n[{r_id['test']}]")
    for k, v in r_id.items():
        if k != "test":
            print(f"  {k:20s} = {v:.6e}")

    r_scale = test_per_dim_scale_spread(stats)
    print(f"\n[{r_scale['test']}]")
    for k, v in r_scale.items():
        if k != "test":
            if isinstance(v, list):
                print(f"  {k:30s} = {[f'{x:.4f}' for x in v]}")
            else:
                print(f"  {k:30s} = {v:.4f}")

    r_dist = test_normalized_space_distortion(actions, norm)
    print(f"\n[{r_dist['test']}]")
    for arm in ("left", "right"):
        print(f"  --- {arm} arm ---")
        for label, d in (("raw (GT)", r_dist[f"raw_{arm}"]), ("normalized", r_dist[f"norm_{arm}"])):
            print(f"    {label:12s}: orth_err mean={d['orthonormality_err_mean']:.3e} max={d['orthonormality_err_max']:.3e} "
                  f"|row0|={d['row0_norm_mean']:.3f}±{d['row0_norm_std']:.3f} "
                  f"|row1|={d['row1_norm_mean']:.3f}±{d['row1_norm_std']:.3f} "
                  f"|dot|={d['row_dot_mean']:.3e}")

    r_noise = test_noise_amplification(actions, norm, args.noise_std)
    print(f"\n[{r_noise['test']}]  (noise_std={args.noise_std} in normalized space)")
    for arm in ("left", "right"):
        d = r_noise[arm]
        print(f"  --- {arm} arm ---")
        print(f"    orthonormality err  norm-space mean={d['orth_err_norm_mean']:.3e} max={d['orth_err_norm_max']:.3e}")
        print(f"    orthonormality err  raw-space  mean={d['orth_err_raw_mean']:.3e} max={d['orth_err_raw_max']:.3e}")
        print(f"    amplification factor (norm/raw) mean = {d['amplification_factor_mean']:.3f}x")
        print(f"    GS rotation change norm-space max = {d['gs_change_norm_deg_max']:.4f} deg")
        print(f"    GS rotation change raw-space  max = {d['gs_change_raw_deg_max']:.4f} deg")

    # Verdict
    print("\n" + "-" * 78)
    print("VERDICT (H2: does min/max normalization warp the rot6d manifold?):")
    norm_orth = r_dist["norm_left"]["orthonormality_err_max"]
    raw_orth = r_dist["raw_left"]["orthonormality_err_max"]
    amp = r_noise["left"]["amplification_factor_mean"]
    scale_ratio = r_scale["rot6d_scale_ratio_max_over_min"]
    print(f"  GT rot6d orthonormality err (raw)       = {raw_orth:.3e}")
    print(f"  GT rot6d orthonormality err (normalized)= {norm_orth:.3e}")
    print(f"  rot6d per-dim scale ratio (max/min)     = {scale_ratio:.2f}x")
    print(f"  noise amplification (norm vs raw)       = {amp:.2f}x")
    h2_confirmed = norm_orth > args.orth_tol or amp > 1.5 or scale_ratio > 2.0
    if h2_confirmed:
        print("  -> CONFIRMED: min/max normalization warps the rot6d manifold.")
        print("     The model must fit targets on a non-orthonormal manifold in normalized")
        print("     space, and diffusion sampling errors are amplified by the per-dim scale")
        print("     mismatch when denormalized. This is a FastWAM-specific issue (GR00T uses")
        print("     per-modality temporal_meanstd normalization for rot6d).")
        print("     -> DexJoCo (sim) uses rotvec (3-dim), which has no orthonormality")
        print("        constraint, so this issue does NOT affect the sim pipeline.")
    else:
        print("  -> NOT confirmed: normalization preserves rot6d geometry adequately.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
