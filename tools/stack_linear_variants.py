#!/usr/bin/env python3
"""Linear member-weight variants for the cross-fold stack, compared on held-out folds.

    uv run python tools/stack_linear_variants.py runs/cv5_fixcos --members mlp,seq,... \
        --out runs/linvar_fc10 [--basis-from runs/fc_p70] [--score free,block10]

Every variant predicts target column j as sum_m W[m, j] * member_m[:, j] (no intercept), so its
squared error on a fold depends only on that fold's per-column sufficient statistics
G_j = A_j'A_j, b_j = A_j'y_j and y_j'y_j. They are computed once per fold; every variant is then
fitted on four folds' statistics and its held-out error on the fifth is exact, in seconds.

Variants: `free` (per-column least squares, what stack_cv --linear fits), `equal`, `block<B>`
(one weight per member per block of B psi coefficients, q95 and betaN free), `poly<D>` (each
member's weight a degree-D polynomial of the coefficient index), `ridge<L>` (towards equal
weights, penalty L * trace(G_j) / M), `smooth<L>` (free plus L * trace-scaled squared
differences of neighbouring psi columns' weights), `nnls`, `factor` (w_m * s_j), and
`ridgecv_{col,block,psi}` (the ridge penalty chosen by inner leave-one-fold-out per column, per
block of ten psi columns, or once for all psi columns; q95 and betaN always on their own).
The statistics are saved (`stats.npz`, `--stats-from`) so new variants need no table load.

The psi error is reported in basis units: PCA components are orthonormal, so the summed
coefficient MSE is the in-basis pixel MSE, the part of R2_psi a variant can change. `--score`
additionally scores the named variants natively on every fold and writes, per variant, the
all-fold weights as a stack directory `stack_predict --linear-only` can apply.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
from scipy.optimize import nnls

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cv_oof import score_fold
from stack_cv import corrected_payload, linear_blend, load_basis, load_fold, member_targets

Stats = tuple[np.ndarray, np.ndarray, np.ndarray, int]  # G (J,M,M), b (J,M), yy (J,), n


def fold_stats(members_T: list[np.ndarray], truth: np.ndarray) -> Stats:
    A = np.stack(members_T, axis=2).astype(np.float64)  # (N, J, M)
    y = truth.astype(np.float64)
    G = np.einsum("njm,njk->jmk", A, A)
    b = np.einsum("njm,nj->jm", A, y)
    return G, b, (y * y).sum(0), len(y)


def add(stats: list[Stats]) -> Stats:
    return (sum(s[0] for s in stats), sum(s[1] for s in stats), sum(s[2] for s in stats),
            sum(s[3] for s in stats))


def mse(W: np.ndarray, s: Stats) -> np.ndarray:
    """Per-column MSE of weights W (M, J) on a fold's statistics."""
    G, b, yy, n = s
    return (np.einsum("mj,jmk,kj->j", W, G, W) - 2 * np.einsum("mj,jm->j", W, b) + yy) / n


def fit_free(s: Stats, cols: np.ndarray | None = None) -> np.ndarray:
    G, b = s[0], s[1]
    return np.stack([np.linalg.solve(G[j], b[j]) for j in range(len(b))], axis=1)


def fit_grouped(s: Stats, groups: list[np.ndarray]) -> np.ndarray:
    G, b = s[0], s[1]
    W = np.empty((G.shape[1], len(b)))
    for g in groups:
        W[:, g] = np.linalg.solve(G[g].sum(0), b[g].sum(0))[:, None]
    return W


def fit_basis_fn(s: Stats, psi: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """W[m, j] = sum_p theta[m, p] phi[j, p] over the psi columns; the rest free."""
    G, b = s[0], s[1]
    M, P = G.shape[1], phi.shape[1]
    H = np.zeros((M * P, M * P))
    r = np.zeros(M * P)
    for j, f in zip(psi, phi, strict=True):
        H += np.kron(G[j], np.outer(f, f))
        r += np.kron(b[j], f)
    theta = np.linalg.solve(H, r).reshape(M, P)
    W = fit_free(s)
    W[:, psi] = theta @ phi.T
    return W


def fit_ridge(s: Stats, lam: float) -> np.ndarray:
    G, b = s[0], s[1]
    M = G.shape[1]
    W = np.empty((M, len(b)))
    for j in range(len(b)):
        a = lam * np.trace(G[j]) / M
        W[:, j] = np.linalg.solve(G[j] + a * np.eye(M), b[j] + a / M)
    return W


def fit_smooth(s: Stats, psi: np.ndarray, lam: float) -> np.ndarray:
    """Joint solve over the psi columns with a penalty on neighbouring columns' weight change."""
    G, b = s[0], s[1]
    M, J = G.shape[1], len(psi)
    H = np.zeros((M * J, M * J))
    r = np.zeros(M * J)
    for i, j in enumerate(psi):
        H[i * M:(i + 1) * M, i * M:(i + 1) * M] += G[j]
        r[i * M:(i + 1) * M] = b[j]
    for i in range(J - 1):
        a = lam * (np.trace(G[psi[i]]) + np.trace(G[psi[i + 1]])) / (2 * M)
        for x, y, sign in ((i, i, 1), (i + 1, i + 1, 1), (i, i + 1, -1), (i + 1, i, -1)):
            H[x * M:(x + 1) * M, y * M:(y + 1) * M] += sign * a * np.eye(M)
    W = fit_free(s)
    W[:, psi] = np.linalg.solve(H, r).reshape(J, M).T
    return W


def fit_nnls(s: Stats) -> np.ndarray:
    G, b = s[0], s[1]
    W = np.empty((G.shape[1], len(b)))
    for j in range(len(b)):
        L = np.linalg.cholesky(G[j])
        W[:, j] = nnls(L.T, np.linalg.solve(L, b[j]))[0]
    return W


def fit_factor(s: Stats, psi: np.ndarray) -> np.ndarray:
    """W[m, j] = w_m * s_j over the psi columns (alternating least squares); the rest free."""
    G, b = s[0], s[1]
    M = G.shape[1]
    sc = np.ones(len(psi))
    w = np.full(M, 1.0 / M)
    for _ in range(100):
        w = np.linalg.solve(sum(c * c * G[j] for c, j in zip(sc, psi)),
                            sum(c * b[j] for c, j in zip(sc, psi)))
        sc = np.array([w @ b[j] / (w @ G[j] @ w) for j in psi])
    W = fit_free(s)
    W[:, psi] = np.outer(w, sc)
    return W


RIDGE_GRID = (0.0, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1)


def fit_ridge_cv(train: list[Stats], groups: list[np.ndarray]) -> np.ndarray:
    """Ridge towards equal weights, the penalty chosen per column group by leave-one-fold-out
    over the training folds (statistics add, so every inner fit is exact and cheap)."""
    inner = np.zeros((len(RIDGE_GRID), len(train[0][1])))
    for i in range(len(train)):
        rest = add([s for j, s in enumerate(train) if j != i])
        for g, lam in enumerate(RIDGE_GRID):
            inner[g] += mse(fit_ridge(rest, lam), train[i])
    full = add(train)
    W = np.empty((full[0].shape[1], len(full[1])))
    for cols in groups:
        best = int(np.argmin(inner[:, cols].sum(1)))
        W[:, cols] = fit_ridge(full, RIDGE_GRID[best])[:, cols]
    return W


def variants(k: int, M: int) -> dict[str, Callable[[list[Stats]], np.ndarray]]:
    """Fitters over the training folds' statistics."""
    single = _single_variants(k, M)
    out: dict[str, Callable[[list[Stats]], np.ndarray]] = {
        name: (lambda L, f=f: f(add(L))) for name, f in single.items()}
    cols = np.arange(k + 2)
    out["ridgecv_col"] = lambda L: fit_ridge_cv(L, [np.array([j]) for j in cols])
    out["ridgecv_block"] = lambda L: fit_ridge_cv(
        L, [cols[a:min(a + 10, k)] for a in range(0, k, 10)] + [np.array([k]), np.array([k + 1])])
    out["ridgecv_psi"] = lambda L: fit_ridge_cv(L, [cols[:k], np.array([k]), np.array([k + 1])])
    return out


def _single_variants(k: int, M: int) -> dict[str, Callable[[Stats], np.ndarray]]:
    psi = np.arange(k)
    qb = [np.array([k]), np.array([k + 1])]
    x = 2 * psi / (k - 1) - 1
    out: dict[str, Callable[[Stats], np.ndarray]] = {
        "free": fit_free,
        "equal": lambda s: np.full((M, len(s[1])), 1.0 / M),
        "nnls": fit_nnls,
        "factor": lambda s: fit_factor(s, psi),
    }
    for B in (5, 10, 25, 50):
        groups = [psi[a:a + B] for a in range(0, k, B)] + qb
        out[f"block{B}"] = lambda s, g=groups: fit_grouped(s, g)
    for D in (1, 2, 3, 5, 8):
        phi = np.polynomial.legendre.legvander(x, D)
        out[f"poly{D}"] = lambda s, f=phi: fit_basis_fn(s, psi, f)
    for lam in (1e-4, 1e-3, 1e-2, 1e-1):
        out[f"ridge{lam:g}"] = lambda s, a=lam: fit_ridge(s, a)
        out[f"smooth{lam:g}"] = lambda s, a=lam: fit_smooth(s, psi, a)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("campaign", type=Path)
    ap.add_argument("--members", required=True)
    ap.add_argument("--basis-from", type=Path)
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--score", default="", help="comma list of variants to score natively")
    ap.add_argument("--jobs", type=int, default=30)
    ap.add_argument("--stats-from", type=Path,
                    help="stats.npz of an earlier run: skip the table load (no --score)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    folds = [int(x) for x in args.folds.split(",")]
    members = args.members.split(",")
    M = len(members)
    equal = np.full(M, 1.0 / M)
    basis = load_basis(args.campaign, args.basis_from)
    k = basis["pca"].n_components
    to_score = [v for v in args.score.split(",") if v]
    tables: dict[int, dict[str, Any]] = {}
    stats: dict[int, Stats] = {}
    if args.stats_from:
        if to_score:
            raise ValueError("--stats-from cannot --score: the scorer needs the tables")
        z = np.load(args.stats_from)
        stats = {f: (z[f"G_{f}"], z[f"b_{f}"], z[f"yy_{f}"], int(z[f"n_{f}"])) for f in folds}
    for f in ([] if args.stats_from else folds):
        t = load_fold(args.campaign, members, f, basis, args.jobs)
        del t["Y"]
        t["members_T"] = member_targets(t)
        t["truth"] = np.column_stack([t["true_coef"], t["S"]])
        stats[f] = fold_stats(t["members_T"], t["truth"])
        if to_score:
            tables[f] = t
        print(f"fold {f}: {stats[f][3]} frames, {M} members, {k} basis components", flush=True)
    np.savez(args.out / "stats.npz", **{f"{n}_{f}": a for f, st in stats.items()
                                        for n, a in zip(("G", "b", "yy", "n"), st)})
    fits = variants(k, M)
    unknown = set(to_score) - set(fits)
    if unknown:
        raise ValueError(f"unknown variants {sorted(unknown)}; known {sorted(fits)}")
    report: dict[str, Any] = {"members": members, "basis_components": k,
                              "basis_from": str(args.basis_from) if args.basis_from else None,
                              "variants": {}}
    held: dict[str, np.ndarray] = {}
    for name, fit in fits.items():
        per = np.stack([mse(fit([stats[j] for j in folds if j != f]), stats[f])
                        for f in folds])  # (folds, J)
        held[name] = per
    ref = held["free"]
    print(f"{'variant':>12} {'psi MSE/free':>13} {'q95/free':>9} {'betaN/free':>11} "
          f"{'psi folds better':>17}")
    for name, per in held.items():
        psi_ratio = per[:, :k].sum(1) / ref[:, :k].sum(1)
        q = per[:, k:] / ref[:, k:]
        report["variants"][name] = {"psi_mse_ratio_per_fold": psi_ratio.tolist(),
                                    "q95_ratio": q[:, 0].mean(), "betaN_ratio": q[:, 1].mean(),
                                    "mse_per_fold_column": per.tolist()}
        print(f"{name:>12} {psi_ratio.mean():13.5f} {q[:, 0].mean():9.5f} {q[:, 1].mean():11.5f} "
              f"{int((psi_ratio < 1).sum()):>10}/{len(folds)}")
    (args.out / "report.json").write_text(json.dumps(report, indent=1, default=float))
    for name in to_score:
        d = args.out / name
        d.mkdir(exist_ok=True)
        entry: dict[str, Any] = {}
        for f in folds:
            t = tables[f]
            W = fits[name]([stats[j] for j in folds if j != f])
            stage1 = sum(w * m for w, m in zip(equal, t["members_T"], strict=True))
            shift = linear_blend(t, W) - stage1
            payload = corrected_payload(t, stage1, shift, equal, basis)
            (r,) = score_fold(t["files"], payload, args.jobs)["results"].values()
            del payload
            entry[f] = r
            print(f"{name} fold {f}: S {r['S']:.6f}", flush=True)
            report["variants"][name]["native"] = entry
            (args.out / "report.json").write_text(json.dumps(report, indent=1, default=float))
        mean = np.mean([r["S"] for r in entry.values()])
        print(f"{name}: mean S over {len(entry)} folds {mean:.6f}", flush=True)
        joblib.dump(fits[name]([stats[j] for j in folds]), d / "linear_all.joblib")
        (d / "config.json").write_text(json.dumps(
            {"members": members, "linear": True, "context": False, "disagreement": False,
             "gbm": None, "variant": name,
             "basis_from": str(args.basis_from) if args.basis_from else None}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
