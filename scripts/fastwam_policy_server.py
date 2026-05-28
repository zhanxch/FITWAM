"""ZMQ policy server/client for FastWAM (GR00T PolicyServer-compatible protocol)."""

from __future__ import annotations

import socket
import subprocess
import tempfile
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import msgpack
import numpy as np
import torch
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
    def _tensor_to_numpy(obj: torch.Tensor) -> np.ndarray:
        tensor = obj.detach().cpu()
        if tensor.dtype in {torch.bfloat16, torch.float16}:
            tensor = tensor.float()
        elif tensor.dtype == torch.bool:
            return tensor.numpy()
        return tensor.numpy()

    @staticmethod
    def _encode_custom(obj: Any) -> Any:
        if torch.is_tensor(obj):
            arr = MsgSerializer._tensor_to_numpy(obj)
            return {
                "__ndarray__": True,
                "data": arr.tobytes(),
                "dtype": str(arr.dtype),
                "shape": arr.shape,
            }
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
        try:
            import msgpack_numpy as mnp

            decoded = mnp.decode(obj)
            if isinstance(decoded, np.ndarray):
                return decoded
        except Exception:
            pass
        if obj.get("__ndarray__") or obj.get(b"__ndarray__"):
            dtype_key = "dtype" if "dtype" in obj else b"dtype"
            data_key = "data" if "data" in obj else b"data"
            shape_key = "shape" if "shape" in obj else b"shape"
            return np.frombuffer(obj[data_key], dtype=np.dtype(obj[dtype_key])).reshape(obj[shape_key])
        return obj


@dataclass
class EndpointHandler:
    handler: Callable
    requires_input: bool = True


class PolicyServer:
    """Inference server over ZeroMQ REP (same endpoints as Isaac-GR00T PolicyServer)."""

    def __init__(
        self,
        policy: Any,
        host: str = "*",
        port: int = DEFAULT_SERVER_PORT,
        api_token: str | None = None,
    ):
        self.policy = policy
        self.running = True
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://{host}:{port}")
        self._endpoints: dict[str, EndpointHandler] = {}
        self.api_token = api_token

        self.register_endpoint("ping", self._handle_ping, requires_input=False)
        self.register_endpoint("kill", self._kill_server, requires_input=False)
        self.register_endpoint("get_action", self.policy.get_action)
        self.register_endpoint("reset", self.policy.reset)
        self.register_endpoint(
            "get_modality_config",
            getattr(self.policy, "get_modality_config", lambda: {}),
            requires_input=False,
        )

    def _kill_server(self) -> dict:
        self.running = False
        return {"status": "ok"}

    def _handle_ping(self) -> dict:
        return {"status": "ok", "message": "Server is running"}

    def register_endpoint(self, name: str, handler: Callable, requires_input: bool = True) -> None:
        self._endpoints[name] = EndpointHandler(handler, requires_input)

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

    def run(self) -> None:
        addr = self.socket.getsockopt_string(zmq.LAST_ENDPOINT)
        print(f"Server is ready and listening on {addr}", flush=True)
        request_id = 0
        while self.running:
            request_id += 1
            try:
                message = self.socket.recv()
            except zmq.ZMQError as exc:
                print(f"[policy-server] ZMQ recv error (continuing): {exc}", flush=True)
                continue

            response: Any
            try:
                request = MsgSerializer.from_bytes(message)
                response = self._handle_request(request)
            except Exception as exc:
                tb = traceback.format_exc()
                endpoint = None
                try:
                    endpoint = request.get("endpoint")  # type: ignore[name-defined]
                except Exception:
                    pass
                print(
                    f"[policy-server] request #{request_id} failed (server stays up): {exc}",
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

            try:
                self.socket.send(self._pack_response(response))
            except zmq.ZMQError as exc:
                print(
                    f"[policy-server] ZMQ send failed after request #{request_id}: {exc}. "
                    "Server process continues; retry the client request.",
                    flush=True,
                )
            except Exception as exc:
                print(
                    f"[policy-server] unexpected send error after request #{request_id}: {exc}",
                    flush=True,
                )


class PolicyClient:
    def __init__(
        self,
        host: str = "localhost",
        port: int = DEFAULT_SERVER_PORT,
        timeout_ms: int = 15000,
        api_token: str | None = None,
    ):
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

    def kill_server(self) -> None:
        self.call_endpoint("kill", requires_input=False)

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


def assert_port_available(host: str, port: int) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError as exc:
            raise AssertionError(
                f"Port {port} on {host} is already in use. "
                "Stop the conflicting process or choose another port."
            ) from exc


def start_server_process(
    server_cmd: str,
    *,
    cwd: Path,
    env: dict[str, str],
) -> tuple[subprocess.Popen, Path]:
    stderr_log = Path(tempfile.mktemp(prefix="fastwam_server_stderr_", suffix=".log"))
    stderr_fh = open(stderr_log, "w")  # noqa: SIM115
    proc = subprocess.Popen(
        ["bash", "-c", server_cmd],
        cwd=cwd,
        env=env,
        stdout=stderr_fh,
        stderr=stderr_fh,
    )
    return proc, stderr_log


def dump_server_log(log_path: Path, tail_chars: int = 8000) -> str:
    try:
        text = log_path.read_text()
        return text[-tail_chars:] if len(text) > tail_chars else text
    except OSError:
        return "<server log not available>"


def wait_for_server_ready(
    proc: subprocess.Popen,
    host: str,
    port: int,
    timeout_s: float,
    server_log: Path | None = None,
) -> None:
    deadline = time.monotonic() + timeout_s
    while True:
        if proc.poll() is not None:
            log_info = ""
            if server_log is not None:
                log_info = f"\nServer output:\n{dump_server_log(server_log)}"
            raise AssertionError(
                f"FastWAM model server failed to start.\nreturncode={proc.returncode}{log_info}"
            )
        try:
            with socket.create_connection((host, port), timeout=1.0):
                elapsed = timeout_s - (deadline - time.monotonic())
                print(f"FastWAM model server ready after {elapsed:.1f}s.", flush=True)
                return
        except OSError:
            if time.monotonic() >= deadline:
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait(timeout=15)
                log_info = ""
                if server_log is not None:
                    log_info = f"\nServer output:\n{dump_server_log(server_log)}"
                raise AssertionError(
                    "FastWAM model server did not become ready before timeout.\n"
                    f"timeout_seconds={timeout_s}{log_info}"
                )
            time.sleep(0.5)
