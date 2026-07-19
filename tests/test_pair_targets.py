import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "fastwam"
    / "datasets"
    / "eve"
    / "pair_targets.py"
)
SPEC = importlib.util.spec_from_file_location("pair_targets_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
pair_targets = importlib.util.module_from_spec(SPEC)
sys.modules["pair_targets_test"] = pair_targets
SPEC.loader.exec_module(pair_targets)
PairTargetStore = pair_targets.PairTargetStore


HASH = "a" * 64


def write_targets(path: Path, **overrides) -> None:
    payload = {
        "pair_id": np.asarray(["pair-0", "pair-1"], dtype="<U16"),
        "success_event_id": np.asarray(["success-0", "success-1"], dtype="<U16"),
        "failure_event_id": np.asarray(["failure-0", "failure-1"], dtype="<U16"),
        "split": np.asarray(["train", "val"], dtype="<U8"),
        "pair_weight": np.asarray([0.75, 1.0], dtype=np.float32),
        "z_plus": np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        "z_minus": np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32),
        "teacher_sha256": np.asarray([HASH, HASH], dtype="<U64"),
    }
    payload.update(overrides)
    np.savez_compressed(path, **payload)


class PairTargetStoreTest(unittest.TestCase):
    def test_lookup_returns_validated_read_only_numpy_targets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pair_targets.npz"
            write_targets(path)
            with PairTargetStore(
                path, expected_teacher_sha256=HASH.upper()
            ) as store:
                self.assertEqual(len(store), 2)
                self.assertEqual(store.embedding_dim, 2)
                self.assertEqual(store.splits, frozenset({"train", "val"}))
                self.assertIn("pair-0", store)
                target = store.get("pair-0")

                self.assertEqual(target.success_event_id, "success-0")
                self.assertEqual(target.failure_event_id, "failure-0")
                self.assertEqual(target.split, "train")
                self.assertAlmostEqual(target.pair_weight, 0.75)
                self.assertFalse(target.z_plus.flags.writeable)
                self.assertFalse(target.z_minus.flags.writeable)
                np.testing.assert_array_equal(target.z_plus, [1.0, 0.0])
                with self.assertRaises(ValueError):
                    target.z_plus[0] = 4.0

            with self.assertRaisesRegex(RuntimeError, "closed"):
                len(store)

    def test_duplicate_pair_ids_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pair_targets.npz"
            write_targets(
                path, pair_id=np.asarray(["pair-0", "pair-0"], dtype="<U16")
            )
            with self.assertRaisesRegex(ValueError, "unique"):
                PairTargetStore(path)

    def test_embedding_shape_and_finite_values_are_strict(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mismatched = root / "mismatched.npz"
            write_targets(
                mismatched,
                z_minus=np.asarray(
                    [[0.0, 1.0, 2.0], [1.0, 0.0, 2.0]], dtype=np.float32
                ),
            )
            with self.assertRaisesRegex(ValueError, "identical"):
                PairTargetStore(mismatched)

            nonfinite = root / "nonfinite.npz"
            write_targets(
                nonfinite,
                z_plus=np.asarray([[np.nan, 0.0], [0.0, 1.0]], dtype=np.float32),
            )
            with self.assertRaisesRegex(ValueError, "non-finite"):
                PairTargetStore(nonfinite)

    def test_weights_splits_and_teacher_hash_are_validated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bad_weight = root / "bad_weight.npz"
            write_targets(
                bad_weight,
                pair_weight=np.asarray([0.5, 1.1], dtype=np.float32),
            )
            with self.assertRaisesRegex(ValueError, r"\[0, 1\]"):
                PairTargetStore(bad_weight)

            bad_split = root / "bad_split.npz"
            write_targets(
                bad_split, split=np.asarray(["train", ""], dtype="<U8")
            )
            with self.assertRaisesRegex(ValueError, "non-empty"):
                PairTargetStore(bad_split)

            inconsistent_hash = root / "inconsistent_hash.npz"
            write_targets(
                inconsistent_hash,
                teacher_sha256=np.asarray([HASH, "b" * 64], dtype="<U64"),
            )
            with self.assertRaisesRegex(ValueError, "consistent"):
                PairTargetStore(inconsistent_hash)

            valid = root / "valid.npz"
            write_targets(valid)
            with self.assertRaisesRegex(ValueError, "expected teacher"):
                PairTargetStore(valid, expected_teacher_sha256="c" * 64)

    def test_object_string_arrays_are_never_unpickled(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pair_targets.npz"
            write_targets(
                path,
                pair_id=np.asarray(["pair-0", "pair-1"], dtype=object),
            )
            with self.assertRaisesRegex(ValueError, "fixed-width Unicode"):
                PairTargetStore(path)

    def test_unknown_pair_and_backend_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pair_targets.npz"
            write_targets(path)
            with PairTargetStore(path) as store:
                with self.assertRaisesRegex(KeyError, "Unknown pair_id"):
                    store.get("missing")
                with self.assertRaisesRegex(ValueError, "backend"):
                    store.get("pair-0", backend="jax")

    def test_embedding_allocation_limit_is_checked_before_loading(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pair_targets.npz"
            write_targets(path)
            with self.assertRaisesRegex(ValueError, "require"):
                PairTargetStore(path, max_embedding_bytes=1)


if __name__ == "__main__":
    unittest.main()
