#!/usr/bin/env python3
"""Second-stage gate: a gradient-boosted residual model on top of a fold's OOF blend.

    uv run python tools/stack_gate.py runs/cv5/fold0 --blend mlp*0.5+seq*0.5 --k 20

The stage-1 prediction is the blend's targets (PCA coefficients + q95/betaN). Stage 2 predicts
the residual (true - stage 1) of the first K coefficients and of q95/betaN from the raw inputs
plus the stage-1 targets, cross-fitted over shots INSIDE the scored fold (GroupKFold), so every
corrected frame comes from a stage-2 model that never saw its shot. The corrected targets are
written as `<fold>/oof_stack.npz` under member name `stack`, scorable with
`tools/cv_oof.py score <campaign> --blend stack --oof-name oof_stack.npz`.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cv_oof import _fold_files, parse_blend
from my_experiments import baseline_model as B


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("fold", type=Path)
    ap.add_argument("--blend", default="mlp*0.5+seq*0.5")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--jobs", type=int, default=24)
    ap.add_argument("--out", default="oof_stack.npz")
    args = ap.parse_args()
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import GroupKFold

    art = joblib.load(args.fold / "baseline.joblib")
    files = _fold_files(art)
    z = np.load(args.fold / "oof.npz")
    if list(z["files"]) != [f.name for f in files]:
        raise ValueError("oof.npz file order differs from the artifact holdout")
    n = int(art["n_pca"])
    w = parse_blend(args.blend)
    stage1 = sum(wt * z[f"targets_{m}"] for m, wt in w.items())
    X, Y, S, lengths = B._read_shots(files, "reading fold", 1.0, args.jobs)
    if not np.array_equal(lengths, z["lengths"]):
        raise ValueError("frame counts differ from oof.npz")
    true_coef = np.empty((len(X), n))
    for a in range(0, len(X), 20000):
        b = min(a + 20000, len(X))
        resid_map = Y[a:b].reshape(b - a, -1) - B.coil_flux(art["coil"], X[a:b]).reshape(b - a, -1)
        true_coef[a:b] = art["pca"].transform(resid_map)
    del Y
    cols = list(range(args.k)) + [n, n + 1]
    truth = np.column_stack([true_coef[:, :args.k], S])
    resid = truth - stage1[:, cols]
    feats = np.column_stack([X, stage1]).astype(np.float32)
    shot = np.repeat(np.arange(len(lengths)), lengths)
    corrected = stage1.copy()
    for j, c in enumerate(cols):
        y = resid[:, j]
        oof = np.zeros_like(y)
        for tr, te in GroupKFold(5).split(feats, y, shot):
            m = HistGradientBoostingRegressor(max_iter=args.iters, learning_rate=0.05,
                                              max_leaf_nodes=31, min_samples_leaf=200,
                                              l2_regularization=1.0, random_state=0)
            m.fit(feats[tr], y[tr])
            oof[te] = m.predict(feats[te])
        corrected[:, c] += oof
        expl = 1 - np.mean((y - oof) ** 2) / np.mean(y ** 2)
        print(f"target {c:3d}: residual MSE explained {expl:+.4f}", flush=True)
    np.savez(args.fold / args.out, files=z["files"], lengths=z["lengths"],
             true_scalars=z["true_scalars"], targets_stack=corrected, targets_base=stage1)
    print(f"wrote {args.fold / args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
