"""Read-only access to frozen Offline Steer pair targets."""

from __future__ import annotations

from dataclasses import dataclass
import hmac
import math
from pathlib import Path
import re
from typing import Any, Literal
import zipfile

import numpy as np


_REQUIRED_ARRAYS = frozenset(
    {
        "pair_id",
        "success_event_id",
        "failure_event_id",
        "split",
        "pair_weight",
        "z_plus",
        "z_minus",
        "teacher_sha256",
    }
)
_SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")


@dataclass(frozen=True)
class PairTarget:
    """One success/failure teacher-target pair."""

    pair_id: str
    success_event_id: str
    failure_event_id: str
    split: str
    pair_weight: float
    z_plus: Any
    z_minus: Any
    teacher_sha256: str


class PairTargetStore:
    """Strict, read-only pair-target store backed by a NumPy ``.npz`` file.

    The archive is opened with ``allow_pickle=False``. String columns must use
    NumPy's fixed-width Unicode dtype, and embeddings remain outside the
    EveRobot manifest. Arrays are loaded once and indexed by ``pair_id``;
    lookups return read-only NumPy views or independent Torch tensors.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        expected_teacher_sha256: str | None = None,
        max_rows: int = 10_000_000,
        max_embedding_dim: int = 65_536,
        max_embedding_bytes: int = 4 * 1024**3,
        max_metadata_bytes: int = 512 * 1024**2,
        validation_chunk_rows: int = 4096,
    ) -> None:
        self.path = Path(path).expanduser().resolve()
        if self.path.suffix.lower() != ".npz":
            raise ValueError("PairTargetStore currently supports only .npz files")
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        if (
            max_rows <= 0
            or max_embedding_dim <= 0
            or max_embedding_bytes <= 0
            or max_metadata_bytes <= 0
        ):
            raise ValueError("Pair-target safety limits must be positive")
        if validation_chunk_rows <= 0:
            raise ValueError("validation_chunk_rows must be positive")

        self._preflight_npz(
            max_rows=max_rows,
            max_embedding_dim=max_embedding_dim,
            max_embedding_bytes=max_embedding_bytes,
            max_metadata_bytes=max_metadata_bytes,
        )
        self._archive = np.load(self.path, allow_pickle=False)
        self._closed = False
        try:
            missing = sorted(_REQUIRED_ARRAYS - set(self._archive.files))
            extra = sorted(set(self._archive.files) - _REQUIRED_ARRAYS)
            if missing:
                raise ValueError(f"Pair-target archive is missing arrays: {missing}")
            if extra:
                raise ValueError(f"Pair-target archive has unexpected arrays: {extra}")

            self._pair_ids = self._unicode_column("pair_id")
            self._success_event_ids = self._unicode_column("success_event_id")
            self._failure_event_ids = self._unicode_column("failure_event_id")
            self._splits = self._unicode_column("split")
            self._teacher_hashes = self._unicode_column("teacher_sha256")
            row_count = len(self._pair_ids)
            if row_count == 0:
                raise ValueError("Pair-target archive must contain at least one row")
            if row_count > max_rows:
                raise ValueError(
                    f"Pair-target archive has {row_count} rows; limit is {max_rows}"
                )

            for label, column in (
                ("success_event_id", self._success_event_ids),
                ("failure_event_id", self._failure_event_ids),
                ("split", self._splits),
                ("teacher_sha256", self._teacher_hashes),
            ):
                if len(column) != row_count:
                    raise ValueError(
                        f"{label} has {len(column)} rows; expected {row_count}"
                    )

            pair_values = [str(value) for value in self._pair_ids]
            if len(set(pair_values)) != row_count:
                raise ValueError("pair_id values must be unique")
            self._index = {pair_id: index for index, pair_id in enumerate(pair_values)}

            self._pair_weights = self._numeric_column(
                "pair_weight", row_count=row_count, ndim=1
            )
            if not np.issubdtype(self._pair_weights.dtype, np.floating):
                raise ValueError("pair_weight must use a floating-point dtype")
            if not np.isfinite(self._pair_weights).all() or (
                (self._pair_weights < 0.0) | (self._pair_weights > 1.0)
            ).any():
                raise ValueError("pair_weight values must be finite and in [0, 1]")

            self._z_plus = self._embedding("z_plus", row_count=row_count)
            self._z_minus = self._embedding("z_minus", row_count=row_count)
            if self._z_plus.shape != self._z_minus.shape:
                raise ValueError(
                    "z_plus and z_minus must have identical [rows, embedding_dim] shapes"
                )
            self.embedding_dim = int(self._z_plus.shape[1])
            if self.embedding_dim > max_embedding_dim:
                raise ValueError(
                    f"Embedding dimension {self.embedding_dim} exceeds "
                    f"limit {max_embedding_dim}"
                )
            embedding_bytes = int(self._z_plus.nbytes + self._z_minus.nbytes)
            if embedding_bytes > max_embedding_bytes:
                raise ValueError(
                    f"Pair-target embeddings require {embedding_bytes} bytes; "
                    f"limit is {max_embedding_bytes}"
                )
            for start in range(0, row_count, validation_chunk_rows):
                end = min(row_count, start + validation_chunk_rows)
                if not np.isfinite(self._z_plus[start:end]).all():
                    raise ValueError("z_plus contains non-finite values")
                if not np.isfinite(self._z_minus[start:end]).all():
                    raise ValueError("z_minus contains non-finite values")

            hashes = {str(value).lower() for value in self._teacher_hashes}
            if len(hashes) != 1 or not all(
                _SHA256_PATTERN.fullmatch(str(value)) for value in self._teacher_hashes
            ):
                raise ValueError(
                    "teacher_sha256 must contain one consistent SHA-256 digest"
                )
            self.teacher_sha256 = next(iter(hashes))
            if expected_teacher_sha256 is not None:
                expected = self._validate_hash(
                    expected_teacher_sha256, "expected_teacher_sha256"
                )
                if not hmac.compare_digest(self.teacher_sha256, expected):
                    raise ValueError(
                        "Pair-target teacher_sha256 does not match the expected teacher"
                    )

            self.splits = frozenset(str(value) for value in self._splits)
            self._make_read_only()
        except Exception:
            self._archive.close()
            self._closed = True
            raise

    @staticmethod
    def _validate_hash(value: str, label: str) -> str:
        if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
            raise ValueError(f"{label} must be a SHA-256 hex digest")
        return value.lower()

    @staticmethod
    def _read_npy_header(stream: Any, name: str) -> tuple[tuple[int, ...], np.dtype]:
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
        elif version == (2, 0):
            shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
        else:
            raise ValueError(f"{name} uses unsupported NPY format version {version}")
        return tuple(int(size) for size in shape), np.dtype(dtype)

    def _preflight_npz(
        self,
        *,
        max_rows: int,
        max_embedding_dim: int,
        max_embedding_bytes: int,
        max_metadata_bytes: int,
    ) -> None:
        """Validate array headers before NumPy allocates decompressed arrays."""

        try:
            with zipfile.ZipFile(self.path, "r") as archive:
                members = archive.namelist()
                expected = {f"{name}.npy" for name in _REQUIRED_ARRAYS}
                if len(members) != len(set(members)):
                    raise ValueError("Pair-target archive has duplicate members")
                missing = sorted(expected - set(members))
                extra = sorted(set(members) - expected)
                if missing:
                    raise ValueError(
                        f"Pair-target archive is missing arrays: {missing}"
                    )
                if extra:
                    raise ValueError(
                        f"Pair-target archive has unexpected arrays: {extra}"
                    )
                headers: dict[str, tuple[tuple[int, ...], np.dtype]] = {}
                for name in _REQUIRED_ARRAYS:
                    with archive.open(f"{name}.npy", "r") as stream:
                        headers[name] = self._read_npy_header(stream, name)
        except zipfile.BadZipFile as error:
            raise ValueError(f"Invalid pair-target NPZ archive: {self.path}") from error

        pair_shape, _ = headers["pair_id"]
        if len(pair_shape) != 1:
            raise ValueError("pair_id must be a one-dimensional array")
        row_count = pair_shape[0]
        if row_count <= 0:
            raise ValueError("Pair-target archive must contain at least one row")
        if row_count > max_rows:
            raise ValueError(
                f"Pair-target archive has {row_count} rows; limit is {max_rows}"
            )

        metadata_bytes = 0
        for name in (
            "pair_id",
            "success_event_id",
            "failure_event_id",
            "split",
            "teacher_sha256",
        ):
            shape, dtype = headers[name]
            if len(shape) != 1 or shape[0] != row_count:
                raise ValueError(f"{name} must have shape [{row_count}]")
            if dtype.kind != "U" or dtype.hasobject:
                raise ValueError(f"{name} must use a fixed-width Unicode dtype")
            metadata_bytes += math.prod(shape) * dtype.itemsize

        weight_shape, weight_dtype = headers["pair_weight"]
        if weight_shape != (row_count,):
            raise ValueError(f"pair_weight must have shape [{row_count}]")
        if not np.issubdtype(weight_dtype, np.floating):
            raise ValueError("pair_weight must use a floating-point dtype")
        metadata_bytes += math.prod(weight_shape) * weight_dtype.itemsize
        if metadata_bytes > max_metadata_bytes:
            raise ValueError(
                f"Pair-target metadata requires {metadata_bytes} bytes; "
                f"limit is {max_metadata_bytes}"
            )

        plus_shape, plus_dtype = headers["z_plus"]
        minus_shape, minus_dtype = headers["z_minus"]
        if (
            len(plus_shape) != 2
            or plus_shape != minus_shape
            or plus_shape[0] != row_count
            or plus_shape[1] <= 0
        ):
            raise ValueError(
                "z_plus and z_minus must have identical [rows, embedding_dim] shapes"
            )
        if not np.issubdtype(plus_dtype, np.floating) or not np.issubdtype(
            minus_dtype, np.floating
        ):
            raise ValueError("z_plus and z_minus must use floating-point dtypes")
        if plus_shape[1] > max_embedding_dim:
            raise ValueError(
                f"Embedding dimension {plus_shape[1]} exceeds limit {max_embedding_dim}"
            )
        embedding_bytes = (
            math.prod(plus_shape) * plus_dtype.itemsize
            + math.prod(minus_shape) * minus_dtype.itemsize
        )
        if embedding_bytes > max_embedding_bytes:
            raise ValueError(
                f"Pair-target embeddings require {embedding_bytes} bytes; "
                f"limit is {max_embedding_bytes}"
            )

    def _unicode_column(self, name: str) -> np.ndarray:
        try:
            value = self._archive[name]
        except ValueError as error:
            raise ValueError(
                f"{name} cannot be loaded without pickle; use fixed Unicode arrays"
            ) from error
        if value.ndim != 1:
            raise ValueError(f"{name} must be a one-dimensional array")
        if value.dtype.kind != "U":
            raise ValueError(f"{name} must use a fixed-width Unicode dtype")
        if any(not str(item).strip() for item in value):
            raise ValueError(f"{name} values must be non-empty strings")
        return value

    def _numeric_column(
        self, name: str, *, row_count: int, ndim: int
    ) -> np.ndarray:
        value = self._archive[name]
        if value.ndim != ndim:
            raise ValueError(f"{name} must be a {ndim}-dimensional array")
        if value.shape[0] != row_count:
            raise ValueError(
                f"{name} has {value.shape[0]} rows; expected {row_count}"
            )
        if value.dtype.kind not in {"f", "i", "u"}:
            raise ValueError(f"{name} must use a numeric dtype")
        return value

    def _embedding(self, name: str, *, row_count: int) -> np.ndarray:
        value = self._numeric_column(name, row_count=row_count, ndim=2)
        if not np.issubdtype(value.dtype, np.floating):
            raise ValueError(f"{name} must use a floating-point dtype")
        if value.shape[1] <= 0:
            raise ValueError(f"{name} must have a positive embedding dimension")
        return value

    def _make_read_only(self) -> None:
        for value in (
            self._pair_ids,
            self._success_event_ids,
            self._failure_event_ids,
            self._splits,
            self._pair_weights,
            self._z_plus,
            self._z_minus,
            self._teacher_hashes,
        ):
            value.setflags(write=False)

    def __len__(self) -> int:
        self._require_open()
        return len(self._pair_ids)

    def __contains__(self, pair_id: object) -> bool:
        self._require_open()
        return isinstance(pair_id, str) and pair_id in self._index

    @property
    def pair_ids(self) -> tuple[str, ...]:
        """Return the immutable row-order pair identifiers."""

        self._require_open()
        return tuple(str(value) for value in self._pair_ids)

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("PairTargetStore is closed")

    def get(
        self,
        pair_id: str,
        *,
        backend: Literal["numpy", "torch"] = "numpy",
        device: Any = None,
        dtype: Any = None,
    ) -> PairTarget:
        """Return one target using NumPy views or independent Torch tensors."""

        self._require_open()
        try:
            index = self._index[pair_id]
        except KeyError as error:
            raise KeyError(f"Unknown pair_id: {pair_id}") from error

        if backend == "numpy":
            if device is not None:
                raise ValueError("device is only supported for the torch backend")
            z_plus = self._z_plus[index]
            z_minus = self._z_minus[index]
            if dtype is not None:
                z_plus = z_plus.astype(dtype, copy=True)
                z_minus = z_minus.astype(dtype, copy=True)
                z_plus.setflags(write=False)
                z_minus.setflags(write=False)
        elif backend == "torch":
            try:
                import torch
            except ImportError as error:
                raise RuntimeError(
                    "Torch backend requested but torch is unavailable"
                ) from error
            z_plus = torch.as_tensor(
                np.array(self._z_plus[index], copy=True), dtype=dtype, device=device
            )
            z_minus = torch.as_tensor(
                np.array(self._z_minus[index], copy=True), dtype=dtype, device=device
            )
        else:
            raise ValueError("backend must be 'numpy' or 'torch'")

        return PairTarget(
            pair_id=str(self._pair_ids[index]),
            success_event_id=str(self._success_event_ids[index]),
            failure_event_id=str(self._failure_event_ids[index]),
            split=str(self._splits[index]),
            pair_weight=float(self._pair_weights[index]),
            z_plus=z_plus,
            z_minus=z_minus,
            teacher_sha256=self.teacher_sha256,
        )

    def close(self) -> None:
        if not self._closed:
            self._archive.close()
            self._closed = True

    def __enter__(self) -> PairTargetStore:
        self._require_open()
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()
