"""Torch-free ZMQ client for FastWAM policy server (DexJoCo eval env)."""

from __future__ import annotations

from typing import Any

import msgpack
import numpy as np
import zmq

DEFAULT_SERVER_PORT = 5560


class MsgSerializer:
    @staticmethod
    def to_bytes(data: Any) -> bytes:
        return msgpack.packb(data, default=MsgSerializer._encode_custom, use_bin_type=True)

    @staticmethod
    def from_bytes(data: bytes) -> Any:
        return msgpack.unpackb(data, object_hook=MsgSerializer._decode_custom, raw=False)

    @staticmethod
    def _encode_custom(obj: Any) -> Any:
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

    @staticmethod
    def _decode_custom(obj: Any) -> Any:
        if not isinstance(obj, dict):
            return obj
        if obj.get("__ndarray__") or obj.get(b"__ndarray__"):
            dtype_key = "dtype" if "dtype" in obj else b"dtype"
            data_key = "data" if "data" in obj else b"data"
            shape_key = "shape" if "shape" in obj else b"shape"
            return np.frombuffer(obj[data_key], dtype=np.dtype(obj[dtype_key])).reshape(obj[shape_key])
        return obj


class PolicyClient:
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = DEFAULT_SERVER_PORT,
        timeout_ms: int = 120000,
        api_token: str | None = None,
    ) -> None:
        self.context = zmq.Context()
        self.host = host
        self.port = port
        self.timeout_ms = timeout_ms
        self.api_token = api_token
        self._init_socket()

    def _init_socket(self) -> None:
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self.socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        self.socket.connect(f"tcp://{self.host}:{self.port}")

    def ping(self) -> bool:
        try:
            self.call_endpoint("ping", requires_input=False)
            return True
        except zmq.error.ZMQError:
            self._init_socket()
            return False

    def reset(self, options: dict | None = None) -> dict:
        return self.call_endpoint("reset", {"options": options})

    def get_action(
        self,
        observation: dict,
        options: dict | None = None,
    ) -> tuple[dict, dict]:
        response = self.call_endpoint(
            "get_action", {"observation": observation, "options": options}
        )
        if isinstance(response, (list, tuple)) and len(response) == 2:
            return response[0], response[1]
        if isinstance(response, dict) and "action" in response:
            return response, {}
        raise RuntimeError(f"Unexpected get_action response: {type(response)}")

    def call_endpoint(
        self,
        endpoint: str,
        data: dict | None = None,
        requires_input: bool = True,
    ) -> Any:
        request: dict = {"endpoint": endpoint}
        if requires_input:
            request["data"] = data
        if self.api_token:
            request["api_token"] = self.api_token

        try:
            self.socket.send(MsgSerializer.to_bytes(request))
            message = self.socket.recv()
        except zmq.error.Again:
            self._init_socket()
            raise

        response = MsgSerializer.from_bytes(message)
        if isinstance(response, dict) and "error" in response:
            tb = response.get("traceback", "")
            msg = f"Server error: {response['error']}"
            if tb:
                msg = f"{msg}\n{tb}"
            raise RuntimeError(msg)
        return response

    def close(self) -> None:
        self.socket.close(linger=0)
        self.context.term()
