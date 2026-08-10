#!/usr/bin/env python3
"""Build analysis artifacts for the interaction-centric action-sensitivity figure.

Hypothesis
----------
Expert demonstrations teach *what* action to take; rollout experience teaches
*when* action precision matters (interaction-critical states).

This script does **not** claim a native predictive-variance head. Panel-A
uncertainty is the overlapping policy-chunk disagreement available in official
eval dumps (H=32, replan=25). Interaction stages are localized by the frozen
soft-event motif scorer (JS change of local action motifs).

Data
----
  Expert soft-event gallery : data/water_plant_soft_event_v1
  S0 official 4x50          : evaluate_results/.../S0/step_006500
  B1-remap-cfg official 4x50: evaluate_results/.../B1-remap-cfg
  Soft-event motif model    : data/water_plant_soft_event_v1/meta/eve/soft_event_motif_model.json

Outputs under --output (default results/interaction_sensitivity_evidence_20260808/):
  report.json
  panel_a_uncertainty.npz
  panel_b_action_deviation.npz
  panel_c_stage_failure.npz
  panel_d_latent_probe.npz
  episode_rows.jsonl

Example:
  conda activate web
  python scripts/analysis/build_interaction_sensitivity_evidence.py
"""

from __future__ import annotations

import argparse
import importlib.machinery
import importlib.util
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXPERT = ROOT / "data/water_plant_soft_event_v1"
DEFAULT_MOTIF = ROOT / "data/water_plant_soft_event_v1/meta/eve/soft_event_motif_model.json"
DEFAULT_S0 = (
    ROOT
    / "evaluate_results/dexjoco/official_4x50_remaining_skip_early_20260807_122616/S0/step_006500"
)
DEFAULT_B1 = (
    ROOT
    / "evaluate_results/dexjoco/official_4x50_B1_remap_video_cfg_20260807_100508/B1-remap-cfg"
)
DEFAULT_SOFT_EVENT_PYC = ROOT / "src/fastwam/__pycache__/soft_event.cpython-310.pyc"
DEFAULT_OUT = ROOT / "results/interaction_sensitivity_evidence_20260808"

STAGES = ("approach", "pre_contact", "interaction", "post_contact")
STAGE_LABELS = {
    "approach": "Approach",
    "pre_contact": "Pre-contact",
    "interaction": "Interaction",
    "post_contact": "Post-contact",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--expert", type=Path, default=DEFAULT_EXPERT)
    p.add_argument("--motif-model", type=Path, default=DEFAULT_MOTIF)
    p.add_argument("--s0-eval", type=Path, default=DEFAULT_S0)
    p.add_argument("--b1-eval", type=Path, default=DEFAULT_B1)
    p.add_argument("--soft-event-pyc", type=Path, default=DEFAULT_SOFT_EVENT_PYC)
    p.add_argument("--output", type=Path, default=DEFAULT_OUT)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--progress-bins", type=int, default=40)
    p.add_argument("--event-threshold-quantile", type=float, default=0.80)
    p.add_argument("--pre-contact-margin", type=float, default=0.08)
    p.add_argument("--interaction-half-width", type=float, default=0.12)
    p.add_argument("--trim-failure-seconds", type=float, default=8.0)
    p.add_argument("--trim-only-length", type=int, default=1000)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--seed", type=int, default=20260808)
    p.add_argument("--max-episodes-per-model", type=int, default=0, help="0 = all")
    return p.parse_args()


def json_dump(obj: Any, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True) + "\n")


def load_soft_event_module(pyc_path: Path):
    if not pyc_path.exists():
        raise FileNotFoundError(f"Missing soft_event pyc: {pyc_path}")
    tmp = Path("/tmp/fastwam_soft_event_iso")
    tmp.mkdir(parents=True, exist_ok=True)
    dst = tmp / "soft_event.pyc"
    shutil.copy(pyc_path, dst)
    name = "fastwam_soft_event_iso"
    loader = importlib.machinery.SourcelessFileLoader(name, str(dst))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    loader.exec_module(mod)
    return mod


class SoftEventScorer:
    def __init__(self, model_path: Path, soft_mod: Any):
        model = json.loads(model_path.read_text())
        self.mod = soft_mod
        self.location = np.asarray(model["normalization"]["location"], dtype=np.float32)
        self.scale = np.asarray(model["normalization"]["scale"], dtype=np.float32)
        self.pca_mean = np.asarray(model["descriptor"]["pca_mean"], dtype=np.float32)
        self.components = np.asarray(model["descriptor"]["pca_components"], dtype=np.float32)
        self.centers = np.asarray(model["motifs"]["centers"], dtype=np.float32)
        self.temperature = float(model["motifs"]["softmax_temperature"])
        self.radius = int(model["config"]["radius"])
        self.confidence_floor = float(model["config"]["confidence_floor"])
        self.smooth_window = int(model["config"]["smooth_window"])
        self.cal_low = float(model["score"]["calibration_low"])
        self.cal_high = float(model["score"]["calibration_high"])
        self.meta = {
            "model_path": str(model_path),
            "radius": self.radius,
            "num_prototypes": int(self.centers.shape[0]),
            "pca_dim": int(self.components.shape[0]),
            "calibration_low": self.cal_low,
            "calibration_high": self.cal_high,
        }

    def score(self, actions22: np.ndarray) -> dict[str, np.ndarray]:
        actions22 = np.asarray(actions22, dtype=np.float32)
        if actions22.ndim != 2 or actions22.shape[1] != 22:
            raise ValueError(f"expected (T,22) actions, got {actions22.shape}")
        norm = (actions22 - self.location) / np.maximum(self.scale, 1e-6)
        desc, valid = self.mod._local_descriptors(norm.astype(np.float32), radius=self.radius)
        projected = (desc - self.pca_mean) @ self.components.T
        probs = self.mod._soft_assign(projected, self.centers, self.temperature)
        raw = self.mod._event_curve(
            probs, radius=self.radius, confidence_floor=self.confidence_floor
        )
        raw = self.mod._centered_average(raw, self.smooth_window)
        score = np.clip((raw - self.cal_low) / max(self.cal_high - self.cal_low, 1e-6), 0.0, 1.0)
        score = score.astype(np.float32)
        return {
            "score": score,
            "raw": raw.astype(np.float32),
            "valid": np.asarray(valid, dtype=bool),
            "descriptor": desc.astype(np.float32),
            "projected": projected.astype(np.float32),
            "probs": probs.astype(np.float32),
        }


def parse_episode_id(path: Path) -> int:
    m = re.search(r"episode_(\d+)_", path.name)
    return int(m.group(1)) if m else -1


def detect_run(path: Path) -> str:
    for part in path.parts:
        if part.startswith("run"):
            return part
    return "unknown"


def trim_failure_actions(actions: np.ndarray, *, fps: int, trim_s: float, trim_only_length: int) -> np.ndarray:
    if len(actions) >= trim_only_length and trim_s > 0:
        drop = int(round(trim_s * fps))
        if drop > 0 and drop < len(actions):
            return actions[:-drop]
    return actions


def load_eval_episodes(root: Path, *, source: str, fps: int, trim_s: float, trim_only_length: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*_actions.npz")):
        name = path.name
        if name.endswith("_failure_actions.npz"):
            success = False
        elif name.endswith("_success_actions.npz"):
            success = True
        else:
            continue
        z = np.load(path)
        executed = np.asarray(z["executed_actions"], dtype=np.float32)
        if not success:
            executed = trim_failure_actions(
                executed, fps=fps, trim_s=trim_s, trim_only_length=trim_only_length
            )
        chunks = np.asarray(z["policy_chunks"], dtype=np.float32)
        query = np.asarray(z["policy_query_steps"], dtype=np.int64)
        # Align chunk coverage to possibly trimmed length.
        t_max = len(executed)
        keep_replans = query < t_max
        chunks = chunks[keep_replans]
        query = query[keep_replans]
        rows.append(
            {
                "source": source,
                "path": str(path),
                "run": detect_run(path),
                "episode_id": parse_episode_id(path),
                "success": bool(success),
                "executed": executed,
                "policy_chunks": chunks,
                "policy_query_steps": query,
                "replan_steps": int(z["replan_steps"]) if "replan_steps" in z.files else 25,
                "action_horizon": int(z["action_horizon"]) if "action_horizon" in z.files else 32,
                "length": int(len(executed)),
            }
        )
    if not rows:
        raise FileNotFoundError(f"No *_actions.npz under {root}")
    return rows


def chunk_disagreement_sigma(
    policy_chunks: np.ndarray,
    query_steps: np.ndarray,
    *,
    length: int,
    action_horizon: int,
) -> np.ndarray:
    """Per-timestep L2 std across overlapping replan chunk predictions."""
    sums = np.zeros((length, policy_chunks.shape[-1]), dtype=np.float64)
    sq = np.zeros_like(sums)
    counts = np.zeros(length, dtype=np.int32)
    for chunk, q in zip(policy_chunks, query_steps):
        q = int(q)
        for h in range(min(action_horizon, length - q)):
            t = q + h
            a = chunk[h]
            sums[t] += a
            sq[t] += a.astype(np.float64) ** 2
            counts[t] += 1
    sigma = np.zeros(length, dtype=np.float32)
    for t in range(length):
        c = counts[t]
        if c >= 2:
            mean = sums[t] / c
            var = np.maximum(sq[t] / c - mean**2, 0.0)
            sigma[t] = float(np.sqrt(var.sum()))
        elif c == 1:
            sigma[t] = 0.0
        else:
            sigma[t] = np.nan
    return sigma


def load_expert_actions(expert_root: Path) -> list[np.ndarray]:
    chunk = expert_root / "data" / "chunk-000"
    out: list[np.ndarray] = []
    for path in sorted(chunk.glob("episode_*.parquet")):
        table = pq.read_table(path, columns=["action"])
        action = np.asarray(table.column("action").to_pylist(), dtype=np.float32)
        # soft-event datasets prepend score in action[0]
        if action.shape[1] == 23:
            action = action[:, 1:]
        out.append(action)
    if not out:
        raise FileNotFoundError(f"No expert episodes under {chunk}")
    return out


def expert_progress_action_table(expert_actions: list[np.ndarray], n_bins: int) -> dict[str, np.ndarray]:
    """Progress-binned expert mean/std action used as Δa reference."""
    sums = np.zeros((n_bins, 22), dtype=np.float64)
    sq = np.zeros_like(sums)
    counts = np.zeros(n_bins, dtype=np.int64)
    for actions in expert_actions:
        tlen = len(actions)
        prog = np.arange(tlen, dtype=np.float64) / max(tlen - 1, 1)
        bins = np.clip((prog * n_bins).astype(np.int64), 0, n_bins - 1)
        for b in range(n_bins):
            mask = bins == b
            if not np.any(mask):
                continue
            a = actions[mask]
            sums[b] += a.sum(axis=0)
            sq[b] += (a.astype(np.float64) ** 2).sum(axis=0)
            counts[b] += len(a)
    mean = np.zeros((n_bins, 22), dtype=np.float32)
    std = np.zeros((n_bins, 22), dtype=np.float32)
    for b in range(n_bins):
        if counts[b] == 0:
            continue
        mean[b] = (sums[b] / counts[b]).astype(np.float32)
        var = np.maximum(sq[b] / counts[b] - mean[b].astype(np.float64) ** 2, 0.0)
        std[b] = np.sqrt(var).astype(np.float32)
    # fill empty bins from neighbors
    for b in range(n_bins):
        if counts[b] > 0:
            continue
        for d in range(1, n_bins):
            for nb in (b - d, b + d):
                if 0 <= nb < n_bins and counts[nb] > 0:
                    mean[b] = mean[nb]
                    std[b] = std[nb]
                    break
            else:
                continue
            break
    return {"mean": mean, "std": std, "counts": counts}


def calibrate_event_threshold(expert_actions: list[np.ndarray], scorer: SoftEventScorer, q: float) -> float:
    vals: list[float] = []
    for actions in expert_actions:
        score = scorer.score(actions)["score"]
        finite = score[np.isfinite(score)]
        if len(finite):
            vals.append(float(np.quantile(finite, q)))
    if not vals:
        raise RuntimeError("Could not calibrate soft-event threshold")
    return float(np.median(vals))


def segment_stages(
    score: np.ndarray,
    *,
    threshold: float,
    pre_margin: float,
    interaction_half_width: float = 0.12,
) -> tuple[np.ndarray, dict[str, float]]:
    """Peak-centered stage labels from soft-event score.

    Threshold-above-q masks are often too wide on long eval rollouts. We instead
    center the interaction window on the soft-event peak and carve Approach /
    Pre-contact / Post-contact around that window.
    """
    tlen = len(score)
    progress = np.arange(tlen, dtype=np.float64) / max(tlen - 1, 1)
    finite = np.isfinite(score)
    if np.any(finite):
        peak_i = int(np.nanargmax(score))
        peak_progress = float(progress[peak_i])
        peak_score = float(score[peak_i])
    else:
        peak_progress = 0.45
        peak_score = 0.0
    half = float(interaction_half_width)
    onset = max(0.0, peak_progress - half)
    offset = min(1.0, peak_progress + half)
    pre_start = max(0.0, onset - pre_margin)
    # Optional: shrink window to score>=threshold near the peak if available.
    if np.any(finite):
        near = finite & (np.abs(progress - peak_progress) <= half) & (score >= threshold)
        if np.any(near):
            idx = np.where(near)[0]
            onset = float(progress[idx[0]])
            offset = float(progress[idx[-1]])
            if offset <= onset:
                offset = min(1.0, onset + 0.02)
            pre_start = max(0.0, onset - pre_margin)
    labels = np.empty(tlen, dtype=object)
    for i, p in enumerate(progress):
        if p < pre_start:
            labels[i] = "approach"
        elif p < onset:
            labels[i] = "pre_contact"
        elif p <= offset:
            labels[i] = "interaction"
        else:
            labels[i] = "post_contact"
    meta = {
        "onset": float(onset),
        "offset": float(offset),
        "pre_start": float(pre_start),
        "event_mass": float(((labels == "interaction")).mean()) if tlen else 0.0,
        "peak_score": float(peak_score),
        "peak_progress": float(peak_progress),
    }
    return labels, meta


def progress_bin_curve(
    progress: np.ndarray,
    values: np.ndarray,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    mean = np.full(n_bins, np.nan, dtype=np.float64)
    std = np.full(n_bins, np.nan, dtype=np.float64)
    for b in range(n_bins):
        mask = (progress >= edges[b]) & (progress < edges[b + 1] if b < n_bins - 1 else progress <= edges[b + 1])
        mask &= np.isfinite(values)
        if not np.any(mask):
            continue
        mean[b] = float(np.mean(values[mask]))
        std[b] = float(np.std(values[mask]))
    return centers, mean, std


def safe_auc(y: np.ndarray, s: np.ndarray) -> float | None:
    y = np.asarray(y).astype(int)
    if len(np.unique(y)) < 2:
        return None
    try:
        return float(roc_auc_score(y, s))
    except ValueError:
        return None


def probe_metrics(x: np.ndarray, y_fail: np.ndarray, y_crit: np.ndarray, seed: int) -> dict[str, Any]:
    """Linear probes: failure risk (AUC) and criticality regression (R^2)."""
    out: dict[str, Any] = {"n": int(len(x))}
    if len(x) < 50:
        out["failure_auc"] = None
        out["criticality_r2"] = None
        return out
    scaler = StandardScaler()
    xs = scaler.fit_transform(x)
    # failure AUC with stratified CV
    y = y_fail.astype(int)
    if len(np.unique(y)) >= 2:
        cv = StratifiedKFold(n_splits=min(5, int(np.min(np.bincount(y)))), shuffle=True, random_state=seed)
        scores = np.zeros(len(y), dtype=np.float64)
        for tr, te in cv.split(xs, y):
            clf = LogisticRegression(max_iter=200, class_weight="balanced", random_state=seed)
            clf.fit(xs[tr], y[tr])
            scores[te] = clf.predict_proba(xs[te])[:, 1]
        out["failure_auc"] = safe_auc(y, scores)
    else:
        out["failure_auc"] = None
    # criticality ridge CV R2
    y2 = y_crit.astype(np.float64)
    if np.std(y2) > 1e-8:
        # simple holdout
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(y2))
        cut = int(0.8 * len(idx))
        tr, te = idx[:cut], idx[cut:]
        reg = Ridge(alpha=1.0)
        reg.fit(xs[tr], y2[tr])
        pred = reg.predict(xs[te])
        ss_res = float(np.sum((y2[te] - pred) ** 2))
        ss_tot = float(np.sum((y2[te] - y2[te].mean()) ** 2))
        out["criticality_r2"] = None if ss_tot <= 1e-12 else float(1.0 - ss_res / ss_tot)
    else:
        out["criticality_r2"] = None
    return out


def main() -> None:
    args = parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)

    soft_mod = load_soft_event_module(args.soft_event_pyc)
    scorer = SoftEventScorer(args.motif_model, soft_mod)
    expert_actions = load_expert_actions(args.expert)
    expert_ref = expert_progress_action_table(expert_actions, args.progress_bins)
    event_threshold = calibrate_event_threshold(
        expert_actions, scorer, args.event_threshold_quantile
    )

    models = {
        "S0": load_eval_episodes(
            args.s0_eval,
            source="S0",
            fps=args.fps,
            trim_s=args.trim_failure_seconds,
            trim_only_length=args.trim_only_length,
        ),
        "B1": load_eval_episodes(
            args.b1_eval,
            source="B1",
            fps=args.fps,
            trim_s=args.trim_failure_seconds,
            trim_only_length=args.trim_only_length,
        ),
    }
    if args.max_episodes_per_model > 0:
        for k in models:
            models[k] = models[k][: args.max_episodes_per_model]

    # Accumulators
    panel_a: dict[str, Any] = {}
    panel_b: dict[str, Any] = {}
    panel_c: dict[str, Any] = {}
    panel_d: dict[str, Any] = {}
    episode_rows: list[dict[str, Any]] = []

    # For panel B / D collect interaction-frame residuals and embeddings
    b_delta: dict[str, dict[str, list[float]]] = {
        "S0": {"success": [], "failure": []},
        "B1": {"success": [], "failure": []},
    }
    d_pack: dict[str, dict[str, list[np.ndarray]]] = {
        "S0": {"x": [], "fail": [], "crit": [], "prog": []},
        "B1": {"x": [], "fail": [], "crit": [], "prog": []},
    }

    # Stage failure attribution
    stage_fail_counts = {m: {s: 0 for s in STAGES} for m in ("S0", "B1")}
    stage_ep_counts = {m: {s: 0 for s in STAGES} for m in ("S0", "B1")}
    model_n = {m: {"success": 0, "failure": 0} for m in ("S0", "B1")}

    # Panel A curves: list of (progress, sigma, score) per episode then aggregate
    a_progress: dict[str, list[np.ndarray]] = {"S0": [], "B1": []}
    a_sigma: dict[str, list[np.ndarray]] = {"S0": [], "B1": []}
    a_score: dict[str, list[np.ndarray]] = {"S0": [], "B1": []}
    a_delta_succ: dict[str, list[np.ndarray]] = {"S0": [], "B1": []}
    a_prog_succ: dict[str, list[np.ndarray]] = {"S0": [], "B1": []}
    a_onset: dict[str, list[float]] = {"S0": [], "B1": []}
    a_offset: dict[str, list[float]] = {"S0": [], "B1": []}
    a_peak: dict[str, list[float]] = {"S0": [], "B1": []}

    # Stage-wise sigma means for diagnostics
    stage_sigma = {m: {s: [] for s in STAGES} for m in ("S0", "B1")}
    stage_delta = {m: {s: [] for s in STAGES} for m in ("S0", "B1")}

    rng = np.random.default_rng(args.seed)

    for model_name, episodes in models.items():
        for ep in episodes:
            actions = ep["executed"]
            scored = scorer.score(actions)
            score = scored["score"]
            stages, stage_meta = segment_stages(
                score,
                threshold=event_threshold,
                pre_margin=args.pre_contact_margin,
                interaction_half_width=args.interaction_half_width,
            )
            sigma = chunk_disagreement_sigma(
                ep["policy_chunks"],
                ep["policy_query_steps"],
                length=ep["length"],
                action_horizon=ep["action_horizon"],
            )
            tlen = ep["length"]
            progress = np.arange(tlen, dtype=np.float64) / max(tlen - 1, 1)
            bins = np.clip((progress * args.progress_bins).astype(np.int64), 0, args.progress_bins - 1)
            a_ref = expert_ref["mean"][bins]
            delta_vec = actions - a_ref
            delta_norm = np.linalg.norm(delta_vec, axis=1).astype(np.float32)

            idx = np.arange(0, tlen, args.stride)
            a_progress[model_name].append(progress[idx])
            a_sigma[model_name].append(sigma[idx])
            a_score[model_name].append(score[idx])
            a_onset[model_name].append(stage_meta["onset"])
            a_offset[model_name].append(stage_meta["offset"])
            a_peak[model_name].append(stage_meta["peak_progress"])
            if ep["success"]:
                a_delta_succ[model_name].append(delta_norm[idx])
                a_prog_succ[model_name].append(progress[idx])

            # stage stats
            for s in STAGES:
                mask = stages == s
                if np.any(mask & np.isfinite(sigma)):
                    stage_sigma[model_name][s].append(float(np.nanmean(sigma[mask])))
                if np.any(mask):
                    stage_delta[model_name][s].append(float(np.mean(delta_norm[mask])))

            # failure attribution: stage of maximum action deviation (precision breach)
            if np.any(np.isfinite(delta_norm)):
                peak_stage = str(stages[int(np.argmax(delta_norm))])
            else:
                peak_stage = str(stages[int(np.nanargmax(score))]) if np.any(np.isfinite(score)) else "interaction"
            model_n[model_name]["success" if ep["success"] else "failure"] += 1
            stage_ep_counts[model_name][peak_stage] += 1
            if not ep["success"]:
                stage_fail_counts[model_name][peak_stage] += 1

            # interaction Δa for panel B
            inter = stages == "interaction"
            if np.any(inter):
                inter_idx = np.where(inter)[0][:: max(args.stride, 1)]
                # subsample for storage
                if len(inter_idx) > 80:
                    inter_idx = rng.choice(inter_idx, size=80, replace=False)
                key = "success" if ep["success"] else "failure"
                b_delta[model_name][key].extend(delta_norm[inter_idx].tolist())

                # panel D features: projected motif descriptor + progress
                x = scored["projected"][inter_idx]
                d_pack[model_name]["x"].append(x)
                d_pack[model_name]["fail"].append(
                    np.full(len(inter_idx), 0 if ep["success"] else 1, dtype=np.int8)
                )
                d_pack[model_name]["crit"].append(score[inter_idx])
                d_pack[model_name]["prog"].append(progress[inter_idx])

            episode_rows.append(
                {
                    "model": model_name,
                    "run": ep["run"],
                    "episode_id": ep["episode_id"],
                    "success": ep["success"],
                    "length": ep["length"],
                    "onset": stage_meta["onset"],
                    "offset": stage_meta["offset"],
                    "peak_score": stage_meta["peak_score"],
                    "peak_progress": stage_meta["peak_progress"],
                    "peak_stage": peak_stage,
                    "mean_sigma": float(np.nanmean(sigma)),
                    "interaction_mean_sigma": float(np.nanmean(sigma[inter])) if np.any(inter) else None,
                    "interaction_mean_delta": float(np.mean(delta_norm[inter])) if np.any(inter) else None,
                }
            )

    # ---- Panel A aggregate ----
    for model_name in ("S0", "B1"):
        prog = np.concatenate(a_progress[model_name]) if a_progress[model_name] else np.array([])
        sig = np.concatenate(a_sigma[model_name]) if a_sigma[model_name] else np.array([])
        sco = np.concatenate(a_score[model_name]) if a_score[model_name] else np.array([])
        centers, mean_s, std_s = progress_bin_curve(prog, sig, args.progress_bins)
        _, mean_e, _ = progress_bin_curve(prog, sco, args.progress_bins)
        # success-only Δa quantiles (empirical action interval vs expert)
        if a_prog_succ[model_name]:
            ps = np.concatenate(a_prog_succ[model_name])
            ds = np.concatenate(a_delta_succ[model_name])
            edges = np.linspace(0.0, 1.0, args.progress_bins + 1)
            d_p10 = np.full(args.progress_bins, np.nan)
            d_p50 = np.full(args.progress_bins, np.nan)
            d_p90 = np.full(args.progress_bins, np.nan)
            for b in range(args.progress_bins):
                mask = (ps >= edges[b]) & (
                    ps < edges[b + 1] if b < args.progress_bins - 1 else ps <= edges[b + 1]
                )
                if np.any(mask):
                    d_p10[b], d_p50[b], d_p90[b] = np.quantile(ds[mask], [0.1, 0.5, 0.9])
        else:
            d_p10 = d_p50 = d_p90 = np.full(args.progress_bins, np.nan)
        onset_q = (
            np.quantile(a_onset[model_name], [0.25, 0.5, 0.75])
            if a_onset[model_name]
            else [0.4, 0.45, 0.5]
        )
        offset_q = (
            np.quantile(a_offset[model_name], [0.25, 0.5, 0.75])
            if a_offset[model_name]
            else [0.55, 0.6, 0.65]
        )
        peak_q = (
            np.quantile(a_peak[model_name], [0.25, 0.5, 0.75])
            if a_peak[model_name]
            else [0.4, 0.45, 0.5]
        )
        stage_sigma_summary = {
            s: {
                "mean": float(np.mean(stage_sigma[model_name][s])) if stage_sigma[model_name][s] else None,
                "std": float(np.std(stage_sigma[model_name][s])) if stage_sigma[model_name][s] else None,
                "n": int(len(stage_sigma[model_name][s])),
            }
            for s in STAGES
        }
        panel_a[model_name] = {
            "progress_centers": centers,
            "sigma_mean": mean_s,
            "sigma_std": std_s,
            "soft_event_mean": mean_e,
            "delta_p10": d_p10,
            "delta_p50": d_p50,
            "delta_p90": d_p90,
            "interaction_onset_q": np.asarray(onset_q, dtype=np.float64),
            "interaction_offset_q": np.asarray(offset_q, dtype=np.float64),
            "peak_progress_q": np.asarray(peak_q, dtype=np.float64),
            "stage_sigma": stage_sigma_summary,
        }

    # ---- Panel B ----
    # 1D action-deviation histograms + empirical failure boundary
    for model_name in ("S0", "B1"):
        succ = np.asarray(b_delta[model_name]["success"], dtype=np.float64)
        fail = np.asarray(b_delta[model_name]["failure"], dtype=np.float64)
        all_d = np.concatenate([succ, fail]) if len(succ) + len(fail) else np.array([0.0])
        hi = float(np.quantile(all_d, 0.99)) if len(all_d) else 1.0
        edges = np.linspace(0.0, max(hi, 1e-3), 41)
        centers = 0.5 * (edges[:-1] + edges[1:])
        hs, _ = np.histogram(succ, bins=edges, density=True) if len(succ) else (np.zeros(40), edges)
        hf, _ = np.histogram(fail, bins=edges, density=True) if len(fail) else (np.zeros(40), edges)
        # P(failure | Δa bin) with Laplace smoothing using episode-balanced weights
        # Use pooled counts
        cs, _ = np.histogram(succ, bins=edges)
        cf, _ = np.histogram(fail, bins=edges)
        p_fail = (cf + 1.0) / (cs + cf + 2.0)
        # failure boundary: smallest Δa where P(fail)>0.5 among bins with support
        support = (cs + cf) >= 5
        boundary = None
        if np.any(support & (p_fail >= 0.5)):
            boundary = float(centers[np.where(support & (p_fail >= 0.5))[0][0]])
        elif np.any(support):
            boundary = float(centers[support][np.argmax(p_fail[support])])
        panel_b[model_name] = {
            "delta_centers": centers,
            "density_success": np.asarray(hs, dtype=np.float64),
            "density_failure": np.asarray(hf, dtype=np.float64),
            "p_failure": p_fail.astype(np.float64),
            "failure_boundary": boundary,
            "n_success_frames": int(len(succ)),
            "n_failure_frames": int(len(fail)),
            "success_mean_delta": float(np.mean(succ)) if len(succ) else None,
            "failure_mean_delta": float(np.mean(fail)) if len(fail) else None,
            "success_std_delta": float(np.std(succ)) if len(succ) else None,
            "failure_std_delta": float(np.std(fail)) if len(fail) else None,
        }

    # ---- Panel C ----
    for model_name in ("S0", "B1"):
        n_fail = max(model_n[model_name]["failure"], 1)
        n_all = max(sum(model_n[model_name].values()), 1)
        rates = {}
        masses = {}
        for s in STAGES:
            # among failures, fraction attributed to stage
            masses[s] = stage_fail_counts[model_name][s] / n_fail
            # failure rate contribution: attributed failures / all episodes
            rates[s] = stage_fail_counts[model_name][s] / n_all
        panel_c[model_name] = {
            "stage_labels": list(STAGES),
            "failure_attribution_mass": [masses[s] for s in STAGES],
            "failure_rate_contribution": [rates[s] for s in STAGES],
            "stage_fail_counts": {s: stage_fail_counts[model_name][s] for s in STAGES},
            "n_success": model_n[model_name]["success"],
            "n_failure": model_n[model_name]["failure"],
            "overall_failure_rate": model_n[model_name]["failure"] / n_all,
            "stage_mean_delta": {
                s: float(np.mean(stage_delta[model_name][s])) if stage_delta[model_name][s] else None
                for s in STAGES
            },
        }

    # ---- Panel D latent probe + PCA viz samples ----
    for model_name in ("S0", "B1"):
        if not d_pack[model_name]["x"]:
            panel_d[model_name] = {"probe": {}, "pca": {}}
            continue
        x = np.concatenate(d_pack[model_name]["x"], axis=0)
        y_fail = np.concatenate(d_pack[model_name]["fail"], axis=0)
        y_crit = np.concatenate(d_pack[model_name]["crit"], axis=0)
        prog = np.concatenate(d_pack[model_name]["prog"], axis=0)
        probe = probe_metrics(x, y_fail, y_crit, args.seed)
        # PCA for scatter (subsample)
        pca = PCA(n_components=2, random_state=args.seed)
        z = pca.fit_transform(StandardScaler().fit_transform(x))
        if len(z) > 4000:
            keep = rng.choice(len(z), size=4000, replace=False)
            z_s, yf_s, yc_s, p_s = z[keep], y_fail[keep], y_crit[keep], prog[keep]
        else:
            z_s, yf_s, yc_s, p_s = z, y_fail, y_crit, prog
        # criticality high/low separation in PC1
        hi = y_crit >= np.nanquantile(y_crit, 0.8)
        lo = y_crit <= np.nanquantile(y_crit, 0.2)
        sep = None
        if np.any(hi) and np.any(lo):
            # mean distance between high/low criticality centroids
            sep = float(np.linalg.norm(z[hi].mean(axis=0) - z[lo].mean(axis=0)))
        panel_d[model_name] = {
            "probe": probe,
            "pca_explained_variance_ratio": pca.explained_variance_ratio_.astype(np.float64),
            "criticality_centroid_separation": sep,
            "scatter_z": z_s.astype(np.float32),
            "scatter_fail": yf_s.astype(np.int8),
            "scatter_crit": yc_s.astype(np.float32),
            "scatter_progress": p_s.astype(np.float32),
        }

    # Save artifacts
    np.savez_compressed(
        out / "panel_a_uncertainty.npz",
        **{
            f"{m}_{k}": np.asarray(v)
            for m, pack in panel_a.items()
            for k, v in pack.items()
            if k != "stage_sigma"
        },
    )
    # stage_sigma separately in report
    b_save: dict[str, np.ndarray] = {}
    for m, pack in panel_b.items():
        for k, v in pack.items():
            if isinstance(v, (list, np.ndarray)):
                b_save[f"{m}_{k}"] = np.asarray(v)
            elif isinstance(v, (int, float)) or v is None:
                b_save[f"{m}_{k}"] = np.asarray(np.nan if v is None else v)
    np.savez_compressed(out / "panel_b_action_deviation.npz", **b_save)

    c_save: dict[str, np.ndarray] = {}
    for m, pack in panel_c.items():
        c_save[f"{m}_failure_attribution_mass"] = np.asarray(pack["failure_attribution_mass"], dtype=np.float64)
        c_save[f"{m}_failure_rate_contribution"] = np.asarray(pack["failure_rate_contribution"], dtype=np.float64)
        c_save[f"{m}_overall_failure_rate"] = np.asarray(pack["overall_failure_rate"])
    np.savez_compressed(out / "panel_c_stage_failure.npz", stage_names=np.asarray(STAGES), **c_save)

    d_save: dict[str, np.ndarray] = {}
    for m, pack in panel_d.items():
        if not pack or "scatter_z" not in pack:
            continue
        d_save[f"{m}_scatter_z"] = pack["scatter_z"]
        d_save[f"{m}_scatter_fail"] = pack["scatter_fail"]
        d_save[f"{m}_scatter_crit"] = pack["scatter_crit"]
        d_save[f"{m}_scatter_progress"] = pack["scatter_progress"]
        d_save[f"{m}_pca_evr"] = pack["pca_explained_variance_ratio"]
        if pack.get("criticality_centroid_separation") is not None:
            d_save[f"{m}_crit_sep"] = np.asarray(pack["criticality_centroid_separation"])
        if pack.get("probe", {}).get("failure_auc") is not None:
            d_save[f"{m}_failure_auc"] = np.asarray(pack["probe"]["failure_auc"])
        if pack.get("probe", {}).get("criticality_r2") is not None:
            d_save[f"{m}_criticality_r2"] = np.asarray(pack["probe"]["criticality_r2"])
    np.savez_compressed(out / "panel_d_latent_probe.npz", **d_save)

    with (out / "episode_rows.jsonl").open("w") as f:
        for row in episode_rows:
            f.write(json.dumps(row) + "\n")

    # Summary metrics for the paper caption
    def stage_sigma_ratio(model: str) -> float | None:
        inter = panel_a[model]["stage_sigma"]["interaction"]["mean"]
        approach = panel_a[model]["stage_sigma"]["approach"]["mean"]
        if inter is None or approach is None or approach <= 1e-12:
            return None
        return float(inter / approach)

    report = {
        "hypothesis": (
            "Expert demonstrations teach what action to take, while rollout "
            "experience teaches when action precision matters."
        ),
        "protocol": {
            "expert": str(args.expert),
            "s0_eval": str(args.s0_eval),
            "b1_eval": str(args.b1_eval),
            "soft_event_threshold_quantile": args.event_threshold_quantile,
            "event_threshold": event_threshold,
            "pre_contact_margin": args.pre_contact_margin,
            "uncertainty_proxy": (
                "L2 std across overlapping policy_chunks at each t "
                "(action_horizon=32, replan_steps=25); not a native variance head"
            ),
            "delta_a_reference": "progress-binned expert mean action",
            "failure_trim_seconds": args.trim_failure_seconds,
            "seed": args.seed,
            "soft_event": scorer.meta,
        },
        "counts": model_n,
        "panel_a": {
            m: {
                "interaction_onset_median": float(panel_a[m]["interaction_onset_q"][1]),
                "interaction_offset_median": float(panel_a[m]["interaction_offset_q"][1]),
                "peak_progress_median": float(panel_a[m]["peak_progress_q"][1]),
                "stage_sigma": panel_a[m]["stage_sigma"],
                "interaction_over_approach_sigma": stage_sigma_ratio(m),
                "interaction_success_delta_p50": (
                    float(np.nanmean(panel_a[m]["delta_p50"][
                        (panel_a[m]["progress_centers"] >= panel_a[m]["interaction_onset_q"][1])
                        & (panel_a[m]["progress_centers"] <= panel_a[m]["interaction_offset_q"][1])
                    ]))
                    if np.any(np.isfinite(panel_a[m]["delta_p50"]))
                    else None
                ),
            }
            for m in ("S0", "B1")
        },
        "protocol_notes": {
            "panel_a_primary": (
                "soft-event criticality + success Δa quantile band; "
                "chunk-disagreement sigma is a secondary proxy (not native variance)"
            ),
            "stage_rule": "peak-centered soft-event window with optional threshold shrink",
            "failure_attribution": "stage of maximum ||a - a_expert(progress)||",
        },
        "panel_b": {
            m: {
                "failure_boundary": panel_b[m]["failure_boundary"],
                "success_mean_delta": panel_b[m]["success_mean_delta"],
                "failure_mean_delta": panel_b[m]["failure_mean_delta"],
                "success_std_delta": panel_b[m]["success_std_delta"],
                "failure_std_delta": panel_b[m]["failure_std_delta"],
                "n_success_frames": panel_b[m]["n_success_frames"],
                "n_failure_frames": panel_b[m]["n_failure_frames"],
            }
            for m in ("S0", "B1")
        },
        "panel_c": panel_c,
        "panel_d": {
            m: {
                "probe": panel_d[m].get("probe", {}),
                "criticality_centroid_separation": panel_d[m].get("criticality_centroid_separation"),
                "pca_explained_variance_ratio": (
                    panel_d[m]["pca_explained_variance_ratio"].tolist()
                    if panel_d[m].get("pca_explained_variance_ratio") is not None
                    else None
                ),
            }
            for m in ("S0", "B1")
        },
        "takeaway": (
            "Expert demonstrations provide successful actions, but rollout experience "
            "reveals action sensitivity and failure boundaries. Rollout retraining makes "
            "the model focus on interaction-critical states where action precision matters most."
        ),
    }
    json_dump(report, out / "report.json")
    print(json.dumps({"output": str(out), "counts": model_n, "event_threshold": event_threshold}, indent=2))
    print("Panel A interaction/approach sigma:", {m: stage_sigma_ratio(m) for m in ("S0", "B1")})
    print(
        "Panel C interaction failure contribution:",
        {m: panel_c[m]["failure_rate_contribution"][2] for m in ("S0", "B1")},
    )
    print(
        "Panel D failure AUC:",
        {m: panel_d[m].get("probe", {}).get("failure_auc") for m in ("S0", "B1")},
    )


if __name__ == "__main__":
    main()
