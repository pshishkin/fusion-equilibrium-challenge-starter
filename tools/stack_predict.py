#!/usr/bin/env python3
"""Public-test predictions of a K-fold member library with the cross-fold GBM second stage.

    uv run python tools/stack_predict.py runs/cv5 --members mlp,seq --stack runs/stackcv_base \
        --out runs/submit_x/diii_d_public_test.npz [--base-out runs/submit_x/base.npz]

For every fold k the fold's members predict each test shot; their decoded maps are projected on
the global basis exactly as in tools/stack_cv.py, `models_all.joblib` supplies the residual
correction, and the corrected fold predictions are averaged over the folds. `--base-out` writes
the same average without the correction, for a paired comparison. Shots are written as
shot_%04d in sorted filename order, the order of the existing public submissions.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from experiments import DEFAULT_LOCAL_DATA_DIR
from my_experiments import baseline_model as B
import json

from stack_cv import apply_stage2, features, linear_blend, member_targets, project
from toolkit.parallel import pmap, resolve_jobs


def _feats(path: Path) -> np.ndarray:
    row = pd.read_parquet(path).iloc[0]
    for col in row.index:
        if col.startswith("efit_") and col not in ("efit_times", "efit_grid_R", "efit_grid_Z"):
            raise ValueError(f"{path.name}: test row carries target column {col}")
    return B.features_for_row(row)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("campaign", type=Path)
    ap.add_argument("--members", default="mlp,seq")
    ap.add_argument("--weights", default="")
    ap.add_argument("--stack", type=Path, required=True)
    ap.add_argument("--config", default="diii_d_public_test")
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--base-out", type=Path)
    ap.add_argument("--max-shots", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=16)
    args = ap.parse_args()
    files = sorted((Path(DEFAULT_LOCAL_DATA_DIR) / "data" / args.config).glob("*.parquet"))
    if args.max_shots:
        files = files[:args.max_shots]
    feats = pmap(_feats, files, resolve_jobs(args.jobs, len(files)))
    lengths = np.array([len(f) for f in feats])
    X = np.concatenate(feats)
    print(f"{len(files)} shots, {len(X)} frames")
    members = args.members.split(",")
    weights = (np.array([float(x) for x in args.weights.split(",")]) if args.weights
               else np.full(len(members), 1.0 / len(members)))
    weights = weights / weights.sum()
    base_art = joblib.load(args.campaign / "fold0" / "baseline.joblib")
    basis = {"pca": base_art["pca"], "coil": base_art["coil"]}
    k = basis["pca"].n_components
    models = joblib.load(args.stack / "models_all.joblib")
    cfg_path = args.stack / "config.json"
    cfg = (json.loads(cfg_path.read_text()) if cfg_path.exists()
           else {"linear": False, "context": True, "disagreement": False})
    W = joblib.load(args.stack / "linear_all.joblib") if cfg["linear"] else None
    print(f"stage 2 config: {cfg}")
    bounds = np.concatenate([[0], np.cumsum(lengths)])
    folds = [int(x) for x in args.folds.split(",")]
    acc_maps = np.zeros((len(X), 65, 65), dtype=np.float64)
    acc_qb = np.zeros((len(X), 2))
    base_maps = np.zeros((len(X), 65, 65), dtype=np.float64) if args.base_out else None
    base_qb = np.zeros((len(X), 2))
    for fold in folds:
        coefs, qbs, maps_blend = [], [], np.zeros((len(X), 65, 65), dtype=np.float32)
        for w, m in zip(weights, members, strict=True):
            src, _, name = m.rpartition(":")
            art = joblib.load((Path(src) if src else args.campaign) / f"fold{fold}"
                              / "baseline.joblib")
            n = int(art["n_pca"])
            Xs = art["scaler"].transform(B.build_inputs(art["coil"], X))
            T = np.concatenate([B._predict_targets(art, name, Xs[bounds[i]:bounds[i + 1]])
                                for i in range(len(files))])
            maps = np.empty((len(X), 65, 65), dtype=np.float32)
            for a in range(0, len(X), 20000):
                b = min(a + 20000, len(X))
                maps[a:b] = (art["pca"].inverse_transform(T[a:b, :n]).reshape(-1, 65, 65)
                             + B.coil_flux(art["coil"], X[a:b]))
            coefs.append(project(maps, X, basis))
            qbs.append(T[:, n:n + 2])
            maps_blend += np.float32(w) * maps
            del maps
        t = {"X": X, "lengths": lengths, "member_coef": coefs, "member_qb": qbs}
        F, stage1 = features(t, weights, cfg["context"], cfg["disagreement"])
        if W is not None:
            t["members_T"] = member_targets(t)
            shift = linear_blend(t, W) - stage1
            delta = apply_stage2(models, np.column_stack([F, stage1 + shift])) + shift
        else:
            delta = apply_stage2(models, F)
        for a in range(0, len(X), 20000):
            b = min(a + 20000, len(X))
            c1 = stage1[a:b, :k]
            corr = (basis["pca"].inverse_transform(c1 + delta[a:b, :k])
                    - basis["pca"].inverse_transform(c1)).reshape(-1, 65, 65)
            acc_maps[a:b] += maps_blend[a:b] + corr
            if base_maps is not None:
                base_maps[a:b] += maps_blend[a:b]
        acc_qb += stage1[:, k:k + 2] + delta[:, k:k + 2]
        base_qb += stage1[:, k:k + 2]
        print(f"fold {fold}: mean |delta coef| {np.abs(delta[:, :k]).mean():.4g}, "
              f"q95/betaN delta {np.abs(delta[:, k:]).mean(0)}", flush=True)
    for out, mp, qb in ((args.out, acc_maps, acc_qb), (args.base_out, base_maps, base_qb)):
        if out is None:
            continue
        mp = mp / len(folds)
        qb = qb / len(folds)
        if not np.isfinite(mp).all() or not np.isfinite(qb).all():
            raise ValueError(f"{out}: non-finite predictions")
        payload = {}
        for i in range(len(files)):
            a, b = bounds[i], bounds[i + 1]
            payload[f"shot_{i:04d}_psirz"] = mp[a:b].astype(np.float32)
            payload[f"shot_{i:04d}_q95"] = qb[a:b, 0].astype(np.float32)
            payload[f"shot_{i:04d}_betaN"] = qb[a:b, 1].astype(np.float32)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out, **payload)
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
