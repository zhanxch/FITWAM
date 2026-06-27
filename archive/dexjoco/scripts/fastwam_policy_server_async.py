"""Async ZMQ policy server for FastWAM (ROUTER/DEALER, concurrent clients)."""

from __future__ import annotations

import queue
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable

import msgpack
import numpy as np
import torch
import zmq

from fastwam_policy_server import MsgSerializer

DEFAULT_ASYNC_SERVER_PORT = 5561


@dataclass
class EndpointHandler:
    handler: Callable
    requires_input: bool = True
    use_gpu_lock: bool = False


class PolicyServerAsync:
    """Inference server over ZeroMQ ROUTER with threaded request handling.

    Multiple eval workers can connect concurrently. GPU inference is serialized
    via a lock because FastWAM ``infer_action`` currently requires batch size 1.
    """

    def __init__(
        self,
        policy: Any,
        host: str = "*",
        port: int = DEFAULT_ASYNC_SERVER_PORT,
        api_token: str | None = None,
        num_workers: int = 8,
    ) -> None:
        self.policy = policy
        self.running = True
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.ROUTER)
        self.socket.setsockopt(zmq.ROUTER_MANDATORY, 1)
        self.socket.bind(f"tcp://{host}:{port}")
        self.api_token = api_token
        self.gpu_lock = threading.Lock()
        self.reply_queue: queue.Queue[list[bytes]] = queue.Queue()
        self.executor = ThreadPoolExecutor(max_workers=max(1, int(num_workers)))
        self._endpoints: dict[str, EndpointHandler] = {}

        self.register_endpoint("ping", self._handle_ping, requires_input=False)
        self.register_endpoint("kill", self._kill_server, requires_input=False)
        self.register_endpoint(
            "get_action",
            self.policy.get_action,
            requires_input=True,
            use_gpu_lock=True,
        )
        self.register_endpoint(
            "get_actions_batch",
            self._get_actions_batch,
            requires_input=True,
            use_gpu_lock=True,
        )
        self.register_endpoint("reset", self.policy.reset, requires_input=True)
        self.register_endpoint(
            "get_modality_config",
            getattr(self.policy, "get_modality_config", lambda: {}),
            requires_input=False,
        )

    def _get_actions_batch(self, observations: list[dict], options: dict | None = None) -> dict:
        handler = getattr(self.policy, "get_actions_batch", None)
        if handler is not None:
            return handler(observations=observations, options=options)
        actions = []
        for observation in observations:
            result, _meta = self.policy.get_action(observation=observation, options=options)
            actions.append(result["action"])
        return {"actions": actions, "action_horizon": getattr(self.policy, "action_horizon", None)}

    def _kill_server(self) -> dict:
        self.running = False
        return {"status": "ok"}

    def _handle_ping(self) -> dict:
        return {"status": "ok", "message": "Async server is running"}

    def register_endpoint(
        self,
        name: str,
        handler: Callable,
        *,
        requires_input: bool = True,
        use_gpu_lock: bool = False,
    ) -> None:
        self._endpoints[name] = EndpointHandler(
            handler=handler,
            requires_input=requires_input,
            use_gpu_lock=use_gpu_lock,
        )

    def _validate_token(self, request: dict) -> bool:
        if self.api_token is None:
            return True
        return request.get("api_token") == self.api_token

    @staticmethod
    def _maybe_clear_cuda_cache() -> None:
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    def _pack_response(self, payload: Any) -> bytes:
        try:
            return MsgSerializer.to_bytes(payload)
        except Exception as exc:
            err = {
                "error": f"Failed to serialize server response: {exc}",
                "traceback": traceback.format_exc(),
            }
            return MsgSerializer.to_bytes(err)

    def _handle_request(self, request: dict) -> Any:
        if not self._validate_token(request):
            return {"error": "Unauthorized: Invalid API token"}

        endpoint = request.get("endpoint", "get_action")
        if endpoint not in self._endpoints:
            raise ValueError(f"Unknown endpoint: {endpoint}")

        handler = self._endpoints[endpoint]
        if handler.requires_input:
            return handler.handler(**request.get("data", {}))
        return handler.handler()

    def _process_frames(self, frames: list[bytes], request_id: int) -> None:
        identity = frames[0]
        empty = frames[1] if len(frames) > 1 else b""
        payload = frames[2] if len(frames) > 2 else frames[-1]

        response: Any
        try:
            request = MsgSerializer.from_bytes(payload)
            endpoint = request.get("endpoint", "get_action")
            handler = self._endpoints.get(endpoint)
            if handler is not None and handler.use_gpu_lock:
                with self.gpu_lock:
                    response = self._handle_request(request)
            else:
                response = self._handle_request(request)
        except Exception as exc:
            tb = traceback.format_exc()
            endpoint = None
            try:
                endpoint = request.get("endpoint")  # type: ignore[name-defined]
            except Exception:
                pass
            print(
                f"[policy-server-async] request #{request_id} failed (server stays up): {exc}",
                flush=True,
            )
            print(tb, flush=True)
            self._maybe_clear_cuda_cache()
            response = {
                "error": str(exc),
                "traceback": tb,
                "endpoint": endpoint,
                "request_id": request_id,
            }

        self.reply_queue.put([identity, empty, self._pack_response(response)])

    def run(self) -> None:
        addr = self.socket.getsockopt_string(zmq.LAST_ENDPOINT)
        print(f"Async server is ready and listening on {addr}", flush=True)
        request_id = 0
        poller = zmq.Poller()
        poller.register(self.socket, zmq.POLLIN)

        while self.running:
            events = dict(poller.poll(timeout=10))
            if self.socket in events:
                try:
                    frames = self.socket.recv_multipart()
                except zmq.ZMQError as exc:
                    print(f"[policy-server-async] ZMQ recv error (continuing): {exc}", flush=True)
                    continue
                request_id += 1
                self.executor.submit(self._process_frames, frames, request_id)

            while True:
                try:
                    reply_frames = self.reply_queue.get_nowait()
                except queue.Empty:
                    break
                try:
                    self.socket.send_multipart(reply_frames)
                except zmq.ZMQError as exc:
                    print(
                        f"[policy-server-async] ZMQ send failed: {exc}. "
                        "Server process continues.",
                        flush=True,
                    )

        self.executor.shutdown(wait=True, cancel_futures=False)


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
    ) -> Any:
        request: dict = {"endpoint": endpoint}
        if requires_input:
            request["data"] = data
        if self.api_token:
            request["api_token"] = self.api_token

        with self._send_lock:
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
