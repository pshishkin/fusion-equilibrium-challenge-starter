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
from stage2_models import CatBoostStage2
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


def project_targets(T: np.ndarray, art: dict[str, Any], X: np.ndarray,
                    basis: dict[str, Any]) -> np.ndarray:
    """`project` of the maps a member's targets decode to, without decoding them.

    decode = T C_a + mu_a + coil_a(X); project = (map - coil_b(X) - mu_b) C_b'. Both PCAs are
    plain (unwhitened) sklearn PCAs and both coil fields are linear in the same signal columns, so
    the composition is (T C_a + mu_a - mu_b) C_b' + X[:, idx] (maps_a - maps_b) C_b': products
    with 50-wide matrices instead of 4225-pixel maps. Equal to the decoded path up to float32
    rounding of the maps, which that path stores.
    """
    pa, pb = art["pca"].pca, basis["pca"].pca
    if pa.whiten or pb.whiten or not np.array_equal(art["coil"]["index"], basis["coil"]["index"]):
        raise ValueError("fast projection needs unwhitened PCAs and one coil signal index")
    n = int(art["n_pca"])
    Cb = pb.components_.astype(np.float64)
    A = pa.components_.astype(np.float64) @ Cb.T
    off = (pa.mean_ - pb.mean_) @ Cb.T
    D = (art["coil"]["maps"].reshape(len(art["coil"]["index"]), -1).astype(np.float64)
         - basis["coil"]["maps"].reshape(len(basis["coil"]["index"]), -1)) @ Cb.T
    idx = art["coil"]["index"]
    return (T[:, :n].astype(np.float64) @ A + off
            + X[:, idx].astype(np.float64) @ D).astype(np.float32)


def fold_table(fold_dirs: list[Path], members: list[str], basis: dict[str, Any],
               jobs: int) -> dict[str, Any]:
    """Everything stage 2 needs for one fold: inputs, per-member global coefs + qb, truth.

    Member i is `members[i]` of the artifact in `fold_dirs[i]`; every artifact must score the
    same files. The fold is read and the truth projected once.
    """
    arts: dict[Path, tuple[dict[str, Any], Any]] = {}
    for d in dict.fromkeys(fold_dirs):
        art = joblib.load(d / "baseline.joblib")
        z = np.load(d / "oof.npz")
        if list(z["files"]) != [f.name for f in _fold_files(art)]:
            raise ValueError(f"{d}: oof.npz order differs from the artifact holdout")
        arts[d] = (art, z)
    files = _fold_files(arts[fold_dirs[0]][0])
    for d, (art, _) in arts.items():
        if [f.name for f in _fold_files(art)] != [f.name for f in files]:
            raise ValueError(f"{d}: scored files differ from {fold_dirs[0]}")
    X, Y, S, lengths = B._read_shots(files, f"reading {fold_dirs[0].name}", 1.0, jobs)
    member_coef, member_qb, decoders = [], [], []
    for d, m in zip(fold_dirs, members, strict=True):
        art, z = arts[d]
        n = int(art["n_pca"])
        T = z[f"targets_{m}"]
        member_coef.append(project_targets(T, art, X, basis))
        member_qb.append(T[:, n:n + 2].astype(np.float32))
        decoders.append((art, z, m))
    true_coef = project(Y, X, basis)
    return {"files": files, "X": X, "Y": Y, "S": S, "lengths": lengths,
            "member_coef": member_coef, "member_qb": member_qb, "true_coef": true_coef,
            "decoders": decoders}


def load_fold(campaign: Path, members: list[str], k: int, basis: dict[str, Any],
              jobs: int) -> dict[str, Any]:
    """fold_table of fold k over members given as 'name' (this campaign) or 'dir:name' (the same
    fold of another campaign, which must hold exactly the same files)."""
    dirs, names = [], []
    for m in members:
        src, _, name = m.rpartition(":")
        dirs.append((Path(src) if src else campaign) / f"fold{k}")
        names.append(name)
    return fold_table(dirs, names, basis, jobs)


def load_basis(campaign: Path, basis_from: Path | None) -> dict[str, Any]:
    """The common basis: fold 0's PCA and coil field of `basis_from`, else of the campaign."""
    art = joblib.load((basis_from or campaign) / "fold0" / "baseline.joblib")
    return {"pca": art["pca"], "coil": art["coil"]}


def member_targets(t: dict[str, Any]) -> list[np.ndarray]:
    """Each member's (global coefficients | q95, betaN), (N, 52)."""
    return [np.column_stack([c, q]) for c, q in zip(t["member_coef"], t["member_qb"], strict=True)]


def features(t: dict[str, Any], weights: np.ndarray, context: bool = True,
             disagreement: bool | str = False) -> tuple[np.ndarray, np.ndarray]:
    members = member_targets(t)
    stage1 = sum(w * m for w, m in zip(weights, members, strict=True))
    parts = [t["X"], stage1, *members]
    if disagreement == "mean":
        # Linear in the member count: each member's deviation from the blend, and the spread.
        for m in members:
            parts.append(m - stage1)
        parts.append(np.std(np.stack(members), axis=0))
    elif disagreement:
        # Trees cannot form a difference of two columns; an uncertainty proxy has to be given.
        # All pairs: quadratic in the member count (15 pairs x 104 columns for six members).
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                d = members[i] - members[j]
                parts += [d, np.abs(d)]
    if context:
        parts += context_features(stage1, t["lengths"])
    return np.column_stack([np.asarray(p, dtype=np.float32) for p in parts]), stage1


LINEAR_VARIANT = "free"


def fit_linear(tables: list[dict[str, Any]]) -> np.ndarray:
    """(M, 52) no-intercept member weights per target column, on OOF frames: least squares, or
    the named variant of tools/stack_linear_variants.py (e.g. ridgecv_block)."""
    if LINEAR_VARIANT != "free":
        from stack_linear_variants import fold_stats, variants
        k = tables[0]["truth"].shape[1] - 2
        fit = variants(k, len(tables[0]["members_T"]))[LINEAR_VARIANT]
        for t in tables:
            if "stats" not in t:
                t["stats"] = fold_stats(t["members_T"], t["truth"])
        return fit([t["stats"] for t in tables])
    M = len(tables[0]["members_T"])
    W = np.empty((M, tables[0]["truth"].shape[1]))
    for j in range(W.shape[1]):
        A = np.concatenate([np.column_stack([m[:, j] for m in t["members_T"]]) for t in tables])
        y = np.concatenate([t["truth"][:, j] for t in tables])
        W[:, j] = np.linalg.lstsq(A, y, rcond=None)[0]
    return W


def linear_blend(t: dict[str, Any], W: np.ndarray) -> np.ndarray:
    return sum(W[i] * m for i, m in enumerate(t["members_T"]))


GBM = "sklearn"


def target_scale(mode: str, truth: np.ndarray, R: np.ndarray) -> np.ndarray | None:
    """Per-column divisors of the stage-2 residual; None means each column's own spread.

    `metric` weighs the columns as the composite does, as TargetScaler does for the base models:
    one divisor for the whole psi block (orthonormal basis, so the summed coefficient error is
    the pixel error), each scalar on its own variance, the blocks as W_PSI against W_QB / 2. The
    psi total variance is the in-basis spread about the mean image, an under-estimate of the
    scorer's flat-mean SS_tot, so psi is weighted somewhat above the composite's ratio. `geo` is
    the geometric mean of `std` and `metric`: the high-order columns count, but less.
    """
    if mode == "std":
        return None
    from common import W_PSI, W_QB
    k = truth.shape[1] - 2
    metric = np.empty(truth.shape[1])
    metric[:k] = np.sqrt(truth[:, :k].var(0).sum() / W_PSI)
    metric[k:] = np.sqrt(truth[:, k:].var(0) / (W_QB / 2))
    if mode == "metric":
        return metric
    std = R.std(0)
    return np.sqrt(metric / np.sqrt((metric ** 2).mean()) * std / np.sqrt((std ** 2).mean()))


TARGET_SCALE = "std"
GBM_DEPTH, GBM_LR = 6, 0.08


def fit_stage2(F: np.ndarray, R: np.ndarray, iters: int, every: int,
               truth: np.ndarray | None = None) -> Any:
    from sklearn.ensemble import HistGradientBoostingRegressor
    rows = np.arange(0, len(F), every)
    if GBM == "catboost":
        scale = None if truth is None else target_scale(TARGET_SCALE, truth, R)
        return CatBoostStage2(iters, depth=GBM_DEPTH, lr=GBM_LR, scale=scale).fit(F[rows], R[rows])
    models = []
    for j in range(R.shape[1]):
        m = HistGradientBoostingRegressor(max_iter=iters, learning_rate=0.05, max_leaf_nodes=31,
                                          min_samples_leaf=200, l2_regularization=1.0,
                                          random_state=j)
        m.fit(F[rows], R[rows, j])
        models.append(m)
    return models


def apply_stage2(models: Any, F: np.ndarray) -> np.ndarray:
    if isinstance(models, CatBoostStage2):
        return models.predict(F)
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
    ap.add_argument("--depth", type=int, default=6, help="CatBoost tree depth")
    ap.add_argument("--lr", type=float, default=0.08, help="CatBoost learning rate")
    ap.add_argument("--every", type=int, default=2, help="train on every Nth frame")
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--no-score", action="store_true")
    ap.add_argument("--eval-folds", default="",
                    help="folds to fit and score a held-out stage for (default: all --folds)")
    ap.add_argument("--stacked-only", action="store_true",
                    help="score only the corrected blend, not the base and linear references")
    ap.add_argument("--no-context", action="store_true")
    ap.add_argument("--gbm", choices=("sklearn", "catboost"), default="sklearn")
    ap.add_argument("--disagreement", action="store_true")
    ap.add_argument("--disagreement-mean", action="store_true",
                    help="disagreement as each member's deviation from the blend plus the "
                         "spread: linear in the member count instead of all pairs")
    ap.add_argument("--target-scale", default="std",
                    help="std, metric or geo (see target_scale); a comma list fits each in turn "
                         "on the same tables, the first being the stage's own")
    ap.add_argument("--basis-from", type=Path,
                    help="campaign whose fold 0 PCA is the common basis (default: the campaign)")
    ap.add_argument("--linear-variant", default="free",
                    help="weight fit of the linear stage, a tools/stack_linear_variants.py name")
    ap.add_argument("--linear", action="store_true",
                    help="fit per-coefficient member weights on the other folds' OOF first; the "
                         "GBM then learns the residual of that linear blend")
    args = ap.parse_args()
    global GBM, TARGET_SCALE
    scales = args.target_scale.split(",")
    if set(scales) - {"std", "metric", "geo"}:
        raise ValueError(f"--target-scale {args.target_scale!r}")
    GBM, TARGET_SCALE = args.gbm, scales[0]
    global LINEAR_VARIANT, GBM_DEPTH, GBM_LR
    LINEAR_VARIANT, GBM_DEPTH, GBM_LR = args.linear_variant, args.depth, args.lr
    if args.disagreement_mean:
        args.disagreement = "mean"
    args.out.mkdir(parents=True, exist_ok=True)
    folds = [int(x) for x in args.folds.split(",")]
    members = args.members.split(",")
    weights = (np.array([float(x) for x in args.weights.split(",")]) if args.weights
               else np.full(len(members), 1.0 / len(members)))
    weights = weights / weights.sum()
    basis = load_basis(args.campaign, args.basis_from)
    tables = {}
    for k in folds:
        t = load_fold(args.campaign, members, k, basis, args.jobs)
        del t["Y"]
        F, stage1 = features(t, weights, not args.no_context, args.disagreement)
        t["F"], t["stage1"] = F, stage1
        t["truth"] = np.column_stack([t["true_coef"], t["S"]])
        t["members_T"] = member_targets(t)
        t["R"] = t["truth"] - stage1
        tables[k] = t
        print(f"fold {k}: {len(F)} frames, {F.shape[1]} features", flush=True)
    report = {"members": members, "weights": weights.tolist(), "iters": args.iters,
              "every": args.every, "folds": {}}
    eval_folds = [int(x) for x in args.eval_folds.split(",")] if args.eval_folds else folds
    for k in eval_folds:
        others = [j for j in folds if j != k]
        t0 = time.time()
        # `shift` moves the equal-weight stage 1 to the linear blend; zero without --linear.
        shifts = {}
        if args.linear:
            W = fit_linear([tables[j] for j in others])
            joblib.dump(W, args.out / f"linear_fold{k}.joblib")
            for j in folds:
                shifts[j] = linear_blend(tables[j], W) - tables[j]["stage1"]
        else:
            shifts = {j: 0.0 for j in folds}
        # Rows are thinned per fold BEFORE concatenation: the full four-fold table is never built.
        ev = args.every
        Fo = np.concatenate([np.column_stack([tables[j]["F"][::ev],
                                              (tables[j]["stage1"] + shifts[j])[::ev]])
                             if args.linear else tables[j]["F"][::ev] for j in others])
        Ro = np.concatenate([(tables[j]["R"] - shifts[j])[::ev] for j in others])
        To = np.concatenate([tables[j]["truth"][::ev] for j in others])
        t = tables[k]
        Fk = (np.column_stack([t["F"], t["stage1"] + shifts[k]]) if args.linear else t["F"])
        entry: dict[str, Any] = {}
        # Several target scalings share one table load; the first is the stage's own.
        for i, scale in enumerate(scales):
            TARGET_SCALE = scale
            tag = "" if i == 0 else f"_{scale}"
            models = fit_stage2(Fo, Ro, args.iters, 1, To)
            joblib.dump(models, args.out / f"models{tag}_fold{k}.joblib")
            delta = apply_stage2(models, Fk) + shifts[k]
            expl = 1 - ((t["R"] - delta) ** 2).mean(0) / (t["R"] ** 2).mean(0)
            print(f"fold {k} [{scale}]: stage 2 fitted in {time.time() - t0:.0f} s; explained "
                  f"median {np.median(expl):+.3f}", flush=True)
            entry[f"explained{tag}"] = expl.tolist()
            if not args.no_score:
                corr = corrected_payload(t, t["stage1"], delta, weights, basis)
                (rc,) = score_fold(t["files"], corr, args.jobs)["results"].values()
                del corr
                entry[f"stacked{tag}"] = rc
                print(f"fold {k} [{scale}]: S stacked {rc['S']:.6f}", flush=True)
            report["folds"][k] = entry
            (args.out / "report.json").write_text(json.dumps(report, indent=1, default=float))
        del Fo, Ro, To
        if not args.no_score and not args.stacked_only:
            rc = entry["stacked"]
            base = corrected_payload(t, t["stage1"], np.zeros_like(delta), weights, basis)
            (rb,) = score_fold(t["files"], base, args.jobs)["results"].values()
            del base
            entry["base"] = rb
            if args.linear:
                lin = corrected_payload(t, t["stage1"], np.broadcast_to(shifts[k], delta.shape),
                                        weights, basis)
                (rl,) = score_fold(t["files"], lin, args.jobs)["results"].values()
                del lin
                entry["linear_only"] = rl
                print(f"fold {k}: S linear-only {rl['S']:.6f}", flush=True)
            print(f"fold {k}: S base {rb['S']:.6f} stacked {rc['S']:.6f} "
                  f"delta {rc['S'] - rb['S']:+.6f}", flush=True)
        report["folds"][k] = entry
        (args.out / "report.json").write_text(json.dumps(report, indent=1, default=float))
    if len(folds) == 5 and not args.eval_folds:
        TARGET_SCALE = scales[0]
        if args.linear:
            W = fit_linear([tables[j] for j in folds])
            joblib.dump(W, args.out / "linear_all.joblib")
            shifts = {j: linear_blend(tables[j], W) - tables[j]["stage1"] for j in folds}
            ev = args.every
            Fa = np.concatenate([np.column_stack([tables[j]["F"][::ev],
                                                  (tables[j]["stage1"] + shifts[j])[::ev]])
                                 for j in folds])
            Ra = np.concatenate([(tables[j]["R"] - shifts[j])[::ev] for j in folds])
        else:
            Fa = np.concatenate([tables[j]["F"][::args.every] for j in folds])
            Ra = np.concatenate([tables[j]["R"][::args.every] for j in folds])
        models = fit_stage2(Fa, Ra, args.iters, 1,
                            np.concatenate([tables[j]["truth"][::args.every] for j in folds]))
        joblib.dump(models, args.out / "models_all.joblib")
        (args.out / "config.json").write_text(json.dumps(
            {"members": members, "linear": args.linear, "context": not args.no_context,
             "disagreement": args.disagreement, "gbm": args.gbm,
             "target_scale": scales[0], "linear_variant": args.linear_variant,
             "basis_from": str(args.basis_from) if args.basis_from else None}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
