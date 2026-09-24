#!/usr/bin/env python3
"""Cross-fold second stage: one GBM residual model over every K-fold OOF prediction.

    uv run python tools/stack_cv.py runs/cv5 --members mlp,seq --out runs/cv5/stackcv

Each fold has its own PCA basis, so coefficients are not commensurable across folds. Every
member's DECODED map is therefore projected onto one global basis (fold 0's PCA and coil field);
stage 2 predicts the residual of those global coefficients and of q95/betaN, and the correction
is added to the stage-1 map as a map-space delta, so whatever lies outside the basis is kept.

Honest evaluation: the stage-2 model that corrects fold k is trained on the OOF frames of the
other folds only. Each fold's corrected maps are scored natively (tools/cv_oof.score_fold) and
the per-fold S is written beside the models (`models_fold{k}.joblib`; `models_all.joblib` is
fitted on every fold and is what a submission applies).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cv_oof import _fold_files, score_fold
from my_experiments import baseline_model as B


def context_features(stage1: np.ndarray, lengths: np.ndarray) -> list[np.ndarray]:
    """Previous/next-frame differences, shot mean, relative position and shot length."""
    shot = np.repeat(np.arange(len(lengths)), lengths)
    starts = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    idx = np.arange(len(shot))
    pos = idx - starts[shot]
    prev = np.where(pos > 0, idx - 1, idx)
    nxt = np.where(pos < lengths[shot] - 1, idx + 1, idx)
    sums = np.add.reduceat(stage1, starts, axis=0)
    return [stage1 - stage1[prev], stage1[nxt] - stage1, (sums / lengths[:, None])[shot],
            (pos / np.maximum(lengths[shot] - 1, 1))[:, None], lengths[shot][:, None]]


def project(maps: np.ndarray, X: np.ndarray, basis: dict[str, Any]) -> np.ndarray:
    """Global-basis coefficients of decoded maps (coil field removed first)."""
    out = np.empty((len(maps), basis["pca"].n_components))
    for a in range(0, len(maps), 20000):
        b = min(a + 20000, len(maps))
        plasma = maps[a:b].reshape(b - a, -1) - B.coil_flux(basis["coil"], X[a:b]).reshape(b - a, -1)
        out[a:b] = basis["pca"].transform(plasma)
    return out


def fold_table(fold_dir: Path, members: list[str], basis: dict[str, Any],
               jobs: int) -> dict[str, Any]:
    """Everything stage 2 needs for one fold: inputs, per-member global coefs + qb, truth."""
    art = joblib.load(fold_dir / "baseline.joblib")
    files = _fold_files(art)
    z = np.load(fold_dir / "oof.npz")
    if list(z["files"]) != [f.name for f in files]:
        raise ValueError(f"{fold_dir}: oof.npz order differs from the artifact holdout")
    X, Y, S, lengths = B._read_shots(files, f"reading {fold_dir.name}", 1.0, jobs)
    n = int(art["n_pca"])
    member_coef, member_qb = [], []
    for m in members:
        T = z[f"targets_{m}"]
        maps = np.empty((len(X), 65, 65), dtype=np.float32)
        for a in range(0, len(X), 20000):
            b = min(a + 20000, len(X))
            maps[a:b] = (art["pca"].inverse_transform(T[a:b, :n]).reshape(-1, 65, 65)
                         + B.coil_flux(art["coil"], X[a:b]))
        member_coef.append(project(maps, X, basis))
        member_qb.append(T[:, n:n + 2])
        del maps
    decoders = [(art, z, m) for m in members]
    true_coef = project(Y, X, basis)
    return {"files": files, "X": X, "Y": Y, "S": S, "lengths": lengths,
            "member_coef": member_coef, "member_qb": member_qb, "true_coef": true_coef,
            "decoders": decoders}


def features(t: dict[str, Any], weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coef = sum(w * c for w, c in zip(weights, t["member_coef"], strict=True))
    qb = sum(w * q for w, q in zip(weights, t["member_qb"], strict=True))
    stage1 = np.column_stack([coef, qb])
    parts = [t["X"], stage1]
    for c, q in zip(t["member_coef"], t["member_qb"], strict=True):
        parts.append(np.column_stack([c, q]))
    parts += context_features(stage1, t["lengths"])
    return np.column_stack(parts).astype(np.float32), stage1


def fit_stage2(F: np.ndarray, R: np.ndarray, iters: int, every: int) -> list[Any]:
    from sklearn.ensemble import HistGradientBoostingRegressor
    rows = np.arange(0, len(F), every)
    models = []
    for j in range(R.shape[1]):
        m = HistGradientBoostingRegressor(max_iter=iters, learning_rate=0.05, max_leaf_nodes=31,
                                          min_samples_leaf=200, l2_regularization=1.0,
                                          random_state=j)
        m.fit(F[rows], R[rows, j])
        models.append(m)
    return models


def apply_stage2(models: list[Any], F: np.ndarray) -> np.ndarray:
    return np.column_stack([m.predict(F) for m in models])


def corrected_payload(t: dict[str, Any], stage1: np.ndarray, delta: np.ndarray,
                      weights: np.ndarray, basis: dict[str, Any]) -> dict[str, np.ndarray]:
    """Stage-1 blend map (each member decoded in its own basis) plus the global-basis delta."""
    X, lengths = t["X"], t["lengths"]
    k = basis["pca"].n_components
    bounds = np.concatenate([[0], np.cumsum(lengths)])
    maps = np.zeros((len(X), 65, 65), dtype=np.float32)
    for a in range(0, len(X), 20000):
        b = min(a + 20000, len(X))
        for w, (art, z, m) in zip(weights, t["decoders"], strict=True):
            n = int(art["n_pca"])
            maps[a:b] += np.float32(w) * (art["pca"].inverse_transform(z[f"targets_{m}"][a:b, :n])
                                          .reshape(-1, 65, 65) + B.coil_flux(art["coil"], X[a:b]))
        c1 = stage1[a:b, :k]
        maps[a:b] += (basis["pca"].inverse_transform(c1 + delta[a:b, :k])
                      - basis["pca"].inverse_transform(c1)).reshape(-1, 65, 65)
    qb = stage1[:, k:k + 2] + delta[:, k:k + 2]
    out: dict[str, np.ndarray] = {}
    for i in range(len(lengths)):
        a, b = bounds[i], bounds[i + 1]
        out[f"shot_{i:04d}_psirz"] = maps[a:b]
        out[f"shot_{i:04d}_q95"] = qb[a:b, 0].astype(np.float32)
        out[f"shot_{i:04d}_betaN"] = qb[a:b, 1].astype(np.float32)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("campaign", type=Path)
    ap.add_argument("--members", default="mlp,seq",
                    help="comma list; 'dir:name' takes a member from another campaign's fold")
    ap.add_argument("--weights", default="", help="comma list, default equal")
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--every", type=int, default=2, help="train on every Nth frame")
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--no-score", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    folds = [int(x) for x in args.folds.split(",")]
    members = args.members.split(",")
    weights = (np.array([float(x) for x in args.weights.split(",")]) if args.weights
               else np.full(len(members), 1.0 / len(members)))
    weights = weights / weights.sum()
    base_art = joblib.load(args.campaign / "fold0" / "baseline.joblib")
    basis = {"pca": base_art["pca"], "coil": base_art["coil"]}
    tables = {}
    for k in folds:
        # A member is 'name' (this campaign) or 'dir:name' (same fold of another campaign).
        t = None
        for m in members:
            src, _, name = m.rpartition(":")
            d = (Path(src) if src else args.campaign) / f"fold{k}"
            tt = fold_table(d, [name], basis, args.jobs)
            if t is None:
                t = tt
                continue
            if [f.name for f in tt["files"]] != [f.name for f in t["files"]]:
                raise ValueError(f"{d}: fold {k} files differ")
            t["member_coef"] += tt["member_coef"]
            t["member_qb"] += tt["member_qb"]
            t["decoders"] += tt["decoders"]
            del tt
        assert t is not None
        del t["Y"]
        F, stage1 = features(t, weights)
        t["F"], t["stage1"] = F, stage1
        t["R"] = np.column_stack([t["true_coef"], t["S"]]) - stage1
        tables[k] = t
        print(f"fold {k}: {len(F)} frames, {F.shape[1]} features", flush=True)
    report = {"members": members, "weights": weights.tolist(), "iters": args.iters,
              "every": args.every, "folds": {}}
    for k in folds:
        others = [j for j in folds if j != k]
        t0 = time.time()
        models = fit_stage2(np.concatenate([tables[j]["F"] for j in others]),
                            np.concatenate([tables[j]["R"] for j in others]),
                            args.iters, args.every)
        joblib.dump(models, args.out / f"models_fold{k}.joblib")
        t = tables[k]
        delta = apply_stage2(models, t["F"])
        expl = 1 - ((t["R"] - delta) ** 2).mean(0) / (t["R"] ** 2).mean(0)
        print(f"fold {k}: stage 2 fitted in {time.time() - t0:.0f} s; explained "
              f"median {np.median(expl):+.3f}", flush=True)
        entry: dict[str, Any] = {"explained": expl.tolist()}
        if not args.no_score:
            base = corrected_payload(t, t["stage1"], np.zeros_like(delta), weights, basis)
            mb = score_fold(t["files"], base, args.jobs)
            del base
            corr = corrected_payload(t, t["stage1"], delta, weights, basis)
            mc = score_fold(t["files"], corr, args.jobs)
            del corr
            (rb,), (rc,) = mb["results"].values(), mc["results"].values()
            entry |= {"base": rb, "stacked": rc}
            print(f"fold {k}: S base {rb['S']:.6f} stacked {rc['S']:.6f} "
                  f"delta {rc['S'] - rb['S']:+.6f}", flush=True)
        report["folds"][k] = entry
        (args.out / "report.json").write_text(json.dumps(report, indent=1, default=float))
    if len(folds) == 5:
        models = fit_stage2(np.concatenate([tables[j]["F"] for j in folds]),
                            np.concatenate([tables[j]["R"] for j in folds]),
                            args.iters, args.every)
        joblib.dump(models, args.out / "models_all.joblib")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
