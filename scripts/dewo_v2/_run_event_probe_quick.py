#!/usr/bin/env python3
"""Quick event probe — run directly, writes probe_summary.json."""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "dewo_v2"))

from analyze_event_value_probe import (  # noqa: E402
    _build_pair_labeled_rows,
    _cluster_scan,
    _feat_window,
    _load_benti_rows,
    _load_pair_dump,
    _mlp_cv,
    _plot_cluster,
    _plot_pair_separation,
    _rows_to_xy,
)

OUT = Path(
    "/data_all/xiangchengzhan/FastWAM/evaluate_results/dexjoco/"
    "fold_glasses_v9_base_cfg_eval_step_005000_oracle_once_20260827_200336/event_probe"
)
BENTI = Path(
    "/data_all/xiangchengzhan/FastWAM/evaluate_results/dexjoco/"
    "fold_glasses_dewo_v9_step_005000_benti_cfg1_4x50_20260827_162324"
)
PAIR_INDEX = ROOT / "data/fold_glasses_dewo_v9_pair_full_lerobot/pair_index.json"


def main() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    replan = 24
    benti_rows = _load_benti_rows(BENTI, replan)
    print(f"load benti n={len(benti_rows)} dt={time.time()-t0:.1f}s", flush=True)

    X_all, y_fs, _ = _rows_to_xy(benti_rows)
    summary = {"benti_n_replans": len(benti_rows), "feature_dim": int(X_all.shape[1])}
    summary["mlp_fail_vs_succ"] = _mlp_cv(X_all, y_fs, n_splits=5, epochs=25)
    print("fail_vs_succ", summary["mlp_fail_vs_succ"], flush=True)

    fail_rows = [r for r in benti_rows if not r.success and r.replan_k >= 1]
    stall_x, stall_y = [], []
    for row in fail_rows:
        rel = float(row.rels[row.replan_k])
        stall_y.append(1 if (rel == rel and rel < 0.05) else 0)
        stall_x.append(_feat_window(row.values, row.rels, row.replan_k, row.step, max(row.step + 24, 1)))
    summary["mlp_fail_stall_rel_lt_0.05"] = _mlp_cv(np.stack(stall_x), np.asarray(stall_y), epochs=25)

    pairs = json.loads(PAIR_INDEX.read_text())
    pair_list = pairs["pairs"] if isinstance(pairs, dict) else pairs
    t_stars = sorted({int(p["t_star_last_recoverable"]) for p in pair_list})
    tol = replan // 2
    cal_x, cal_y = [], []
    for row in benti_rows:
        if row.replan_k < 1:
            continue
        cal_y.append(1 if any(abs(row.step - t) <= tol for t in t_stars) else 0)
        cal_x.append(_feat_window(row.values, row.rels, row.replan_k, row.step, max(row.step + 24, 1)))
    summary["oracle_tstar_calendar"] = {
        "n_tstar": len(t_stars),
        "t_star_median": float(__import__("numpy").median(t_stars)),
        "n_pos": int(sum(cal_y)),
        "n_neg": int(len(cal_y) - sum(cal_y)),
    }
    summary["mlp_calendar_tstar_vs_rest"] = _mlp_cv(np.stack(cal_x), np.asarray(cal_y), epochs=25)
    print("calendar", summary["mlp_calendar_tstar_vs_rest"], flush=True)

    fail_only = [r for r in benti_rows if not r.success and r.replan_k >= 2]
    Xf = np.stack(
        [_feat_window(r.values, r.rels, r.replan_k, r.step, max(r.step + 24, 1)) for r in fail_only]
    )
    cl = {"scan": [], "best": {"k": 2, "silhouette": -1.0, "labels": np.zeros(min(len(Xf), 1), dtype=np.int32)}}
    try:
        Xc = Xf
        if Xc.shape[0] > 500:
            Xc = Xc[np.random.default_rng(0).choice(Xc.shape[0], 500, replace=False)]
        cl = _cluster_scan(Xc, k_max=5)
        _plot_cluster(OUT, Xc, cl["best"]["labels"], f"KMeans k={cl['best']['k']} fail replans")
    except Exception as exc:
        summary["cluster_error"] = str(exc)
    summary["cluster_fail_replans"] = {
        "scan": cl["scan"],
        "best_k": cl["best"]["k"],
        "best_sil": cl["best"]["silhouette"],
    }

    dump_path = OUT / "pair_event_values.jsonl"
    dump_rows = _load_pair_dump(dump_path)
    if dump_rows:
        pair_rows = _build_pair_labeled_rows(dump_rows, replan)
        Xp, yp, _ = _rows_to_xy(pair_rows)
        summary["pair_dump_n"] = len(dump_rows)
        summary["mlp_pair_event_tstar"] = _mlp_cv(Xp, yp, epochs=25)
        ev_v = [float(r["cfg_value"]) for r in dump_rows if r.get("is_event")]
        neg_v = [float(r["cfg_value"]) for r in dump_rows if not r.get("is_event")]
        summary["pair_v_event_mean"] = float(np.mean(ev_v)) if ev_v else None
        summary["pair_v_neg_mean"] = float(np.mean(neg_v)) if neg_v else None
        _plot_pair_separation(OUT, dump_rows)
        print("pair event", summary["mlp_pair_event_tstar"], flush=True)
    else:
        summary["pair_dump"] = "missing"

    (OUT / "probe_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"done dt={time.time()-t0:.1f}s -> {OUT/'probe_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
