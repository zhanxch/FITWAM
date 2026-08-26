#!/usr/bin/env python3
"""Write RUN_DIR/adaptive_cfg_tau.json from E+ / E0 execution-segment RMS.

tau = quantile(E+, 1-recall). Default recall=0.90 → q_0.10(E+).
FPR0 = P(E0 > tau) is a check: if it exceeds --max-fpr0, separable=false
and eval must skip adaptive (report w=1 only). Do not grid-search tau on
official seeds 0–49, and do not treat 0.05 as a v6 prior.

  python scripts/dewo_v2/calibrate_adaptive_cfg_tau.py \
    --e-plus e_plus.json --e-zero e_zero.json \
    --out runs/.../adaptive_cfg_tau.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fastwam.models.wan22.uncond_adapter import write_adaptive_cfg_tau_json


def _load_rms(path: str | None) -> list[float]:
    if not path:
        return []
    file = Path(path)
    if not file.is_file():
        raise FileNotFoundError(f"missing RMS file: {file}")
    if file.suffix == ".npy":
        import numpy as np

        values = np.load(file)
        return [float(v) for v in values.reshape(-1).tolist()]
    text = file.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if file.suffix == ".json":
        payload = json.loads(text)
        if isinstance(payload, dict):
            for key in ("e_plus", "e_zero", "values", "rms"):
                if key in payload:
                    payload = payload[key]
                    break
        if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes)):
            raise ValueError(f"{file} JSON must be a numeric list or {{values: [...]}}.")
        return [float(v) for v in payload]
    return [float(line) for line in text.splitlines() if line.strip() and not line.startswith("#")]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e-plus", required=True, help="D+ NFE0 exec RMS (.json/.npy/.txt)")
    parser.add_argument("--e-zero", default=None, help="ordinary-success prefix RMS (optional)")
    parser.add_argument("--recall", type=float, default=0.90)
    parser.add_argument("--max-fpr0", type=float, default=0.05)
    parser.add_argument("--recipe", default="v6")
    parser.add_argument(
        "--out",
        required=True,
        help="output JSON, typically RUN_DIR/adaptive_cfg_tau.json",
    )
    args = parser.parse_args()
    payload = write_adaptive_cfg_tau_json(
        args.out,
        _load_rms(args.e_plus),
        _load_rms(args.e_zero) if args.e_zero else None,
        recall=args.recall,
        max_fpr0=args.max_fpr0,
        recipe=args.recipe,
    )
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
