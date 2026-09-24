#!/usr/bin/env python3
"""Information gate: do geometry-oriented Thomson features explain out-of-fold residuals?

    uv run python tools/ts_gate.py runs/cv5/fold0

Reads `<fold>/residuals.npz` (tools/oof_residuals.py). For each of the seven derived scalars and
for q95/betaN, a gradient-boosted regressor predicts the residual (true - predicted) under
shot-grouped 5-fold cross-fitting, from

    A: the production raw features (90 columns) plus the model's own predicted values,
    B: A plus the geometry Thomson block of tools/ts_geometry_features.py.

The reported number is the out-of-fold explained fraction of residual variance. A alone measures
how much of the residual any flexible post-hoc learner finds in the inputs the model already has;
B - A is the information the new block adds.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from experiments import DEFAULT_LOCAL_DATA_DIR, HF_TRAIN_CONFIG
from my_experiments import baseline_model as B
from toolkit.parallel import pmap, resolve_jobs
from ts_geometry_features import frame_features


def _ts_task(path: Path) -> np.ndarray:
    return frame_features(pd.read_parquet(path).iloc[0])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("fold", type=Path)
    ap.add_argument("--jobs", type=int, default=24)
    ap.add_argument("--max-shots", type=int, default=0)
    args = ap.parse_args()
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import GroupKFold

    z = np.load(args.fold / "residuals.npz")
    data_dir = Path(DEFAULT_LOCAL_DATA_DIR) / "data" / HF_TRAIN_CONFIG
    files = [data_dir / n for n in z["files"]]
    lengths = z["lengths"]
    X, _, _, lens = B._read_shots(files, "reading inputs", 1.0, args.jobs)
    if not np.array_equal(lens, lengths):
        raise ValueError("frame counts differ between residuals.npz and the shot cache")
    ts = np.vstack(pmap(_ts_task, files, resolve_jobs(args.jobs, len(files))))
    if len(ts) != len(X):
        raise ValueError(f"Thomson block {len(ts)} rows, inputs {len(X)}")
    shot = z["shot"]
    names = [*map(str, z["cons_names"]), "q95", "betaN"]
    true = np.column_stack([z["cons_true"], z["qb_true"]])
    pred = np.column_stack([z["cons_pred"], z["qb_pred"]])
    ok = np.isfinite(true).all(1) & np.isfinite(pred).all(1)
    if args.max_shots:
        ok &= shot < args.max_shots
    base = np.column_stack([X, pred])[ok]
    full = np.column_stack([base, ts[ok]])
    resid = (true - pred)[ok]
    groups = shot[ok]
    print(f"frames {ok.sum()} of {len(ok)}, base features {base.shape[1]}, "
          f"with Thomson geometry {full.shape[1]}")
    report = {}
    for j, name in enumerate(names):
        y = resid[:, j]
        var = float(np.var(y))
        row = {"resid_var": var, "var_true": float(np.var(true[ok, j]))}
        for label, feats in (("A", base), ("B", full)):
            oof = np.zeros_like(y)
            for tr, te in GroupKFold(5).split(feats, y, groups):
                m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05,
                                                  max_leaf_nodes=31, min_samples_leaf=200,
                                                  l2_regularization=1.0, random_state=0)
                m.fit(feats[tr], y[tr])
                oof[te] = m.predict(feats[te])
            row[f"explained_{label}"] = float(1 - np.mean((y - oof) ** 2) / var)
        # What the explained share is worth in this scalar's R2 if all of it reached the score.
        row["r2_now"] = float(1 - var / row["var_true"] - np.mean(y) ** 2 / row["var_true"])
        report[name] = row
        print(f"{name:8s} R2 {row['r2_now']:.5f}  explained A {row['explained_A']:+.4f}  "
              f"B {row['explained_B']:+.4f}  B-A {row['explained_B'] - row['explained_A']:+.4f}",
              flush=True)
    (args.fold / "ts_gate.json").write_text(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
