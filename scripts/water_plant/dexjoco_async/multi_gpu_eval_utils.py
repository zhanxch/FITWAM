"""Reusable utilities for multi-GPU sharded FastWAM / DexJoCo evaluation.

This module is intentionally decoupled from any specific eval business logic.
It only knows how to:

* shard ``(episodes, seeds)`` across workers,
* find free TCP ports,
* ping an async ``PolicyServerAsync`` endpoint until it is ready,
* launch a server / eval-client subprocess inside an arbitrary conda env.

It imports only the standard library + ``zmq`` + ``msgpack`` (for ping). It does
NOT import ``torch`` or ``mujoco``, so it can be imported from either the
``fastwam`` or ``dexjoco`` conda environments.
"""

from __future__ import annotations

import os
import shlex
import socket
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


@dataclass
class ServerSpec:
    """Where a single policy server lives and which GPU it owns."""

    gpu: int
    port: int
    bind_host: str = "0.0.0.0"
    connect_host: str = "127.0.0.1"
    # Logical device inside the process. With CUDA_VISIBLE_DEVICES=<gpu> set, the
    # visible device is always cuda:0, so each server uses the same string.
    device: str = "cuda:0"


@dataclass
class ShardSpec:
    """A slice of the total workload assigned to one server.

    ``num_episodes`` episodes run with seeds ``[base_seed, base_seed+num_episodes)``
    which are globally contiguous and non-overlapping across shards.
    ``global_episode_start`` is the 0-indexed offset of this shard's first
    episode within the full eval (used by the aggregator to re-index episodes).
    """

    shard_id: int
    server: ServerSpec
    num_episodes: int
    base_seed: int
    global_episode_start: int


def find_free_ports(num: int, *, host: str = "127.0.0.1") -> list[int]:
    """Allocate ``num`` ephemeral TCP ports by binding to port 0.

    There is an inherent TOCTOU race between releasing a port and the server
    binding it; for local orchestrator use this is acceptable and we launch
    servers immediately after.
    """
    ports: list[int] = []
    opened: list[socket.socket] = []
    try:
        for _ in range(max(1, int(num))):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((host, 0))
            opened.append(sock)
            ports.append(int(sock.getsockname()[1]))
    finally:
        for sock in opened:
            sock.close()
    return ports


def shard_episodes(
    servers: Sequence[ServerSpec],
    total_episodes: int,
    base_seed: int,
) -> list[ShardSpec]:
    """Distribute ``total_episodes`` across ``len(servers)`` shards.

    Remainder episodes go to the first shards so the distribution is as even as
    possible (e.g. 100 over 4 -> 25 each; 103 over 4 -> 26/26/26/25). Shards
    with zero episodes are still returned (the caller decides whether to skip
    launching a server/client for them).
    """
    n = len(servers)
    if n <= 0:
        raise ValueError("servers must be a non-empty list")
    if total_episodes < 0:
        raise ValueError("total_episodes must be non-negative")
    base, rem = divmod(max(0, int(total_episodes)), n)
    shards: list[ShardSpec] = []
    ep_cursor = 0
    seed_cursor = int(base_seed)
    for i, srv in enumerate(servers):
        count = base + (1 if i < rem else 0)
        shards.append(
            ShardSpec(
                shard_id=i,
                server=srv,
                num_episodes=count,
                base_seed=seed_cursor,
                global_episode_start=ep_cursor,
            )
        )
        ep_cursor += count
        seed_cursor += count
    return shards


def ping_server(host: str, port: int, *, timeout_ms: int = 5000) -> bool:
    """Return True if an async policy server answers ``ping`` on ``host:port``.

    Self-contained: builds the same wire format as ``PolicyClientAsync`` so it
    does not need the client class (and therefore not even numpy).
    """
    try:
        import msgpack
        import zmq
    except Exception:
        return False

    ctx = zmq.Context()
    sock = ctx.socket(zmq.DEALER)
    try:
        sock.setsockopt(zmq.IDENTITY, f"probe-{port}-{os.getpid()}".encode("utf-8"))
        sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
        sock.setsockopt(zmq.SNDTIMEO, timeout_ms)
        sock.connect(f"tcp://{host}:{port}")
        sock.send(msgpack.packb({"endpoint": "ping"}, use_bin_type=True))
        frames = sock.recv_multipart()
        resp = msgpack.unpackb(frames[-1], raw=False)
        return isinstance(resp, dict) and resp.get("status") == "ok"
    except Exception:
        return False
    finally:
        try:
            sock.close(linger=0)
        except Exception:
            pass
        try:
            ctx.term()
        except Exception:
            pass


def wait_for_server(
    host: str,
    port: int,
    *,
    timeout_s: float = 1200.0,
    poll_interval_s: float = 5.0,
) -> bool:
    """Block until ``ping_server`` succeeds or ``timeout_s`` elapses."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if ping_server(host, port):
            return True
        time.sleep(poll_interval_s)
    return False


def locate_conda_sh() -> Path:
    """Return the path to ``etc/profile.d/conda.sh`` for the active conda."""
    result = subprocess.run(
        ["conda", "info", "--base"], capture_output=True, text=True, check=False
    )
    base = result.stdout.strip()
    if not base:
        raise RuntimeError(
            "Could not locate conda base via `conda info --base`. "
            "Make sure conda is on PATH when running the orchestrator."
        )
    conda_sh = Path(base) / "etc" / "profile.d" / "conda.sh"
    if not conda_sh.exists():
        raise RuntimeError(f"conda.sh not found at {conda_sh}")
    return conda_sh


def build_conda_command(
    conda_sh: Path,
    env: str,
    exports: dict[str, str],
    python_argv: Sequence[str],
    *,
    raw_exports: dict[str, str] | None = None,
) -> list[str]:
    """Build a ``bash -c`` invocation that activates ``env`` then runs ``python_argv``.

    Mirrors the manual workflow: ``source conda.sh && conda activate <env> &&
    export K=V ... && exec python ...``. Streaming output is unbuffered because
    we use ``exec`` (no ``conda run`` wrapper).

    ``exports`` values are shell-quoted (literal, no expansion). ``raw_exports``
    values are emitted verbatim after ``export KEY=`` so shell variables such as
    ``${CONDA_PREFIX}`` are expanded by bash at activation time — use double
    quotes inside the value if spaces must be preserved.
    """
    parts: list[str] = [f'source "{conda_sh}"', f"conda activate {shlex.quote(env)}"]
    for key, value in exports.items():
        parts.append(f"export {key}={shlex.quote(value)}")
    for key, value in (raw_exports or {}).items():
        parts.append(f"export {key}={value}")
    parts.append("exec " + " ".join(shlex.quote(str(a)) for a in python_argv))
    return ["bash", "-c", " && ".join(parts)]


def launch_subprocess(
    cmd: Sequence[str],
    *,
    log_path: Path,
    cwd: Path,
    env: dict[str, str] | None = None,
) -> subprocess.Popen:
    """Start ``cmd`` with stdout/stderr teed to ``log_path``.

    The child inherits the orchestrator environment plus ``env`` overrides.
    Output is line-buffered to the log file so progress is visible in real time.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("w", encoding="utf-8", buffering=1)
    pass_env = dict(os.environ)
    if env:
        pass_env.update({k: str(v) for k, v in env.items()})
    return subprocess.Popen(
        list(cmd),
        cwd=str(cwd),
        stdout=log_file,
        stderr=subprocess.STDOUT,
        env=pass_env,
        start_new_session=True,
    )


def terminate_process(proc: subprocess.Popen, *, label: str, grace_s: float = 10.0) -> None:
    """Best-effort termination of a subprocess: SIGTERM then SIGKILL after grace."""
    if proc.poll() is not None:
        return
    try:
        proc.terminate()
    except Exception as exc:  # pragma: no cover - best effort
        print(f"[multi-gpu] terminate({label}) failed: {exc}", flush=True)
        return
    try:
        proc.wait(timeout=grace_s)
    except subprocess.TimeoutExpired:
        try:
            proc.kill()
        except Exception:
            pass
        try:
            proc.wait(timeout=grace_s)
        except Exception:
            pass


__all__ = [
    "ServerSpec",
    "ShardSpec",
    "find_free_ports",
    "shard_episodes",
    "ping_server",
    "wait_for_server",
    "locate_conda_sh",
    "build_conda_command",
    "launch_subprocess",
    "terminate_process",
]
