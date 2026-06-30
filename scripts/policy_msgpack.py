"""Torch-free msgpack serialization for numpy payloads.

Shared by policy clients (dexjoco env, no torch) and by the torch-aware server
serializer. The server also handles ``torch.Tensor`` inputs; that path lives in
``fastwam_policy_server.py`` and reuses the numpy encoding below after converting
tensors to numpy.

Wire format (msgpack dict with ``__ndarray__`` marker):
    {"__ndarray__": True, "data": <bytes>, "dtype": "<str>", "shape": <tuple>}
"""

from __future__ import annotations

from typing import Any

import msgpack
import numpy as np


def _encode_numpy(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return {
            "__ndarray__": True,
            "data": obj.tobytes(),
            "dtype": str(obj.dtype),
            "shape": obj.shape,
        }
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"Unsupported type for msgpack: {type(obj)}")


def _decode_numpy(obj: Any) -> Any:
    if not isinstance(obj, dict):
        return obj
    # New format: __ndarray__ marker
    nd_key = "__ndarray__" if "__ndarray__" in obj else b"__ndarray__"
    if obj.get(nd_key):
        dtype_key = "dtype" if "dtype" in obj else b"dtype"
        data_key = "data" if "data" in obj else b"data"
        shape_key = "shape" if "shape" in obj else b"shape"
        dtype = obj[dtype_key]
        if isinstance(dtype, bytes):
            dtype = dtype.decode("utf-8")
        return np.frombuffer(obj[data_key], dtype=np.dtype(dtype)).reshape(obj[shape_key])
    # Legacy format: "nd" marker (older GR00T-compatible clients)
    nd_key = "nd" if "nd" in obj else b"nd"
    if obj.get(nd_key):
        type_key = "type" if "type" in obj else b"type"
        data_key = "data" if "data" in obj else b"data"
        shape_key = "shape" if "shape" in obj else b"shape"
        if type_key in obj and data_key in obj and shape_key in obj:
            dtype = obj[type_key]
            if isinstance(dtype, bytes):
                dtype = dtype.decode("utf-8")
            return np.frombuffer(obj[data_key], dtype=np.dtype(dtype)).reshape(obj[shape_key])
    # msgpack_numpy fallback
    try:
        import msgpack_numpy as mnp

        decoded = mnp.decode(obj)
        if isinstance(decoded, np.ndarray):
            return decoded
    except Exception:
        pass
    return obj


def to_bytes(data: Any) -> bytes:
    return msgpack.packb(data, default=_encode_numpy, use_bin_type=True)


def from_bytes(data: bytes) -> Any:
    return msgpack.unpackb(data, object_hook=_decode_numpy, raw=False)
