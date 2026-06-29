# DexJoCo Async Inference

Historical DexJoCo-only async/LPF evaluation utilities for FastWAM.

These scripts evaluate a trained FastWAM policy in DexJoCo with two control modes:

- `blocking`: wait for a policy chunk before executing the next segment.
- `overlap`: submit the next policy request while executing the current chunk.

Main files:

| File | Role |
|------|------|
| `run_fastwam_server_async.py` | Starts an async ZMQ FastWAM policy server. |
| `fastwam_policy_server_async.py` | ROUTER/DEALER ZMQ server/client implementation. |
| `eval_dexjoco_fastwam_control.py` | DexJoCo closed-loop evaluator with overlap control and optional LPF. |
| `run_dexjoco_async_lpf_eval_clients.sh` | Multi-condition evaluation launcher. |
| `summarize_dexjoco_async_ablation.py` | Aggregates summary and video manifests. |

The committed result subset is in [`results/dexjoco_async_microwave`](../../results/dexjoco_async_microwave).
