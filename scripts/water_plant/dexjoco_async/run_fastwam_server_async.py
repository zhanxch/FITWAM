#!/usr/bin/env python3
"""Launch an async FastWAM inference policy server (concurrent ZMQ clients)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
THIS_DIR = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_ROOT, SCRIPTS_ROOT, THIS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from fastwam_policy_server_async import DEFAULT_ASYNC_SERVER_PORT, PolicyServerAsync
from run_fastwam_server import FastWAMPolicy, MockFastWAMPolicy, _build_policy_from_run, _resolve_run_dir


class FastWAMPolicyAsync(FastWAMPolicy):
    """FastWAM policy with batched request helper (sequential GPU infer under the hood)."""

    def get_actions_batch(
        self,
        observations: list[dict[str, Any]],
        options: dict | None = None,
    ) -> dict[str, Any]:
        if not observations:
            raise ValueError("observations must be a non-empty list")
        actions = []
        for observation in observations:
            result, _meta = self.get_action(observation=observation, options=options)
            actions.append(result["action"])
        return {"actions": actions, "action_horizon": self.action_horizon}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run FastWAM async ZMQ policy server (supports concurrent eval workers)."
    )
    parser.add_argument("--mock", action="store_true", help="Start mock policy (no model load).")
    parser.add_argument("--run-dir", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--dataset-stats-path", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--action-horizon", type=int, default=None)
    parser.add_argument("--num-inference-steps", type=int, default=None)
    parser.add_argument(
        "--load-text-encoder",
        dest="load_text_encoder",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=DEFAULT_ASYNC_SERVER_PORT)
    parser.add_argument("--api-token", type=str, default=None)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Thread pool size for concurrent client request handling.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("Starting FastWAM async inference server...", flush=True)
    print(f"  Host: {args.host}", flush=True)
    print(f"  Port: {args.port}", flush=True)
    print(f"  Worker threads: {args.num_workers}", flush=True)

    if args.mock:
        policy = MockFastWAMPolicy()
        print("  Policy: mock (no checkpoint)", flush=True)
    else:
        if not args.run_dir or not args.checkpoint:
            raise ValueError("--run-dir and --checkpoint are required unless --mock is set.")
        run_dir = Path(args.run_dir).expanduser().resolve()
        base_policy = _build_policy_from_run(
            run_dir=run_dir,
            checkpoint=args.checkpoint,
            dataset_stats_path=args.dataset_stats_path,
            device=args.device,
            action_horizon=args.action_horizon,
            num_inference_steps=args.num_inference_steps,
            load_text_encoder=args.load_text_encoder,
        )
        policy = FastWAMPolicyAsync(
            model=base_policy.model,
            processor=base_policy.processor,
            device=base_policy.device,
            action_horizon=base_policy.action_horizon,
            num_inference_steps=base_policy.num_inference_steps,
            num_video_frames=base_policy.num_video_frames,
            text_cfg_scale=base_policy.text_cfg_scale,
            negative_prompt=base_policy.negative_prompt,
            sigma_shift=base_policy.sigma_shift,
            seed=base_policy.seed,
            rand_device=base_policy.rand_device,
            tiled=base_policy.tiled,
        )
        print(f"  Run dir: {_resolve_run_dir(run_dir)}", flush=True)
        print(f"  Device: {args.device}", flush=True)

    server = PolicyServerAsync(
        policy=policy,
        host=args.host,
        port=args.port,
        api_token=args.api_token,
        num_workers=args.num_workers,
    )
    print(f"\n✓ Async server ready — listening on {args.host}:{args.port}\n", flush=True)
    try:
        server.run()
    except KeyboardInterrupt:
        print("\nShutting down async server...", flush=True)


if __name__ == "__main__":
    main()
