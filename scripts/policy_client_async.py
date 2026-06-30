"""Torch-free async ZMQ policy client for DexJoCo eval.

This module contains ONLY the client side of the async protocol — no ``torch``
import, no server classes. It is safe to import from the ``dexjoco`` conda env
(which has MuJoCo + numpy + zmq but not torch).

Server-side code (``fastwam_policy_server_async.py``) re-exports
``PolicyClientAsync`` from here for backwards compatibility.
"""

from __future__ import annotations

import threading
import zmq

from policy_msgpack import from_bytes, to_bytes

DEFAULT_ASYNC_SERVER_PORT = 5561


class PolicyClientAsync:
    """DEALER client for PolicyServerAsync (thread-safe, one socket per instance)."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = DEFAULT_ASYNC_SERVER_PORT,
        timeout_ms: int = 300000,
        api_token: str | None = None,
        identity: bytes | str | None = None,
    ) -> None:
        self.context = zmq.Context()
        self.host = host
        self.port = port
        self.timeout_ms = timeout_ms
        self.api_token = api_token
        self.identity = self._normalize_identity(identity)
        self._send_lock = threading.Lock()
        self._init_socket()

    @staticmethod
    def _normalize_identity(identity: bytes | str | None) -> bytes:
        if identity is None:
            return f"client-{threading.get_ident()}".encode("utf-8")
        if isinstance(identity, str):
            return identity.encode("utf-8")
        return identity

    def _init_socket(self) -> None:
        self.socket = self.context.socket(zmq.DEALER)
        self.socket.setsockopt(zmq.IDENTITY, self.identity)
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

    def get_actions_batch(
        self,
        observations: list[dict],
        options: dict | None = None,
    ) -> dict:
        return self.call_endpoint(
            "get_actions_batch",
            {"observations": observations, "options": options},
        )

    def call_endpoint(
        self,
        endpoint: str,
        data: dict | None = None,
        requires_input: bool = True,
    ) -> object:
        request: dict = {"endpoint": endpoint}
        if requires_input:
            request["data"] = data
        if self.api_token:
            request["api_token"] = self.api_token

        with self._send_lock:
            try:
                self.socket.send(to_bytes(request))
                frames = self.socket.recv_multipart()
                message = frames[-1]
            except zmq.error.Again:
                self._init_socket()
                raise

        response = from_bytes(message)
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
