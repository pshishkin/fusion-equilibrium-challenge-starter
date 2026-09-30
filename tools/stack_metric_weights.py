#!/usr/bin/env python3
"""Stack member weights fitted under the competition's own geometry, not per coefficient.

    uv run python tools/stack_metric_weights.py runs/cv5_fixcos --members mlp,seq,... \
        --basis-from runs/fc_p70 --out runs/metricw_fc24 [--score]

The linear stage (`stack_linear_variants.py`) fits each coefficient's member weights on its own,
i.e. it minimises the SUM of per-coefficient squared errors. The base models are trained on a
different geometry: `target_metric.metric_form`, `M = I + (SS_tot / W_PSI) * M_cons`, where
`M_cons` linearises the seven scorer-derived scalars (`jacobian_form`) and is NOT diagonal — two
coefficient errors can cancel in the flux and add up in the axis position, or the reverse. This
fits the member weights of all psi coefficients JOINTLY under that metric:

    minimise  sum_n e_n' M e_n,   e_nj = sum_m w_mj A_nmj - y_nj,

a quadratic in all M*K weights whose normal equations need only per-fold sufficient statistics
(`S = sum_n a_n a_n'` over the M*K member-coefficient values, `A'y`, `y'y`). So every variant is
fitted on four folds and its held-out metric loss on the fifth is exact, as in the per-column
tool. q95 and betaN are outside M and keep the per-column ridgecv_block weights.

Variants: `metric_free` (a weight per member and coefficient) and `metric_block` (one per member
per block of ten coefficients), each with a ridge penalty towards equal weights whose strength
is chosen by inner leave-one-fold-out on the metric loss; against `ridgecv_block` and `equal`.
`--score` scores the reference and the best metric variant natively on every fold.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import stack_cv
from cv_oof import _fold_files, score_fold
from my_experiments import baseline_model as B
from my_experiments.target_metric import jacobian_form, metric_form, scorer_context
from stack_cv import corrected_payload, linear_blend, load_basis, load_fold, member_targets
from stack_linear_variants import fit_ridge_cv

LAMBDAS = (0.0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1)


def metric_matrix(basis: dict[str, Any], probe_dir: Path, frames: int, delta_rel: float,
                  jobs: int) -> tuple[np.ndarray, dict[str, Any]]:
    """`M` in the common basis, from truth maps of `probe_dir`'s scored shots (the same
    construction the base models train with, `baseline_model` jacobian branch)."""
    art = joblib.load(probe_dir / "baseline.joblib")
    files = _fold_files(art)[::7]
    X, Y, _, _ = B._read_shots(files, "reading probe shots", 1.0, jobs)
    Yd = Y.astype(np.float64)
    ss_tot = float(((Yd - Yd.mean()) ** 2).sum() / len(Yd))
    pick = np.linspace(0, len(Y) - 1, frames).astype(int)
    k = basis["pca"].n_components
    images = np.asarray(basis["pca"].pca.components_).reshape(k, 65, 65)
    delta = delta_rel * float(np.sqrt(ss_tot / k))
    grid = art["coil"]
    ctx = scorer_context(grid["grid_R"], grid["grid_Z"], grid["machine"])
    m_cons, var, ratio, n_used, _ = jacobian_form(images, Yd[pick], delta, ctx, jobs)
    M = metric_form(m_cons, ss_tot)
    info = {"psi_ss_tot": ss_tot, "probe_frames": int(n_used), "delta": delta,
            "linear_over_actual": ratio.tolist(),
            "cons_share_of_trace": float(np.trace(M - np.eye(k)) / np.trace(M))}
    return M, info


def fold_stats(members_T: list[np.ndarray], truth: np.ndarray, k: int) -> dict[str, np.ndarray]:
    """S (MK x MK), A'y (MK x K), y'y (K x K) over the psi columns, index m*K + j."""
    M = len(members_T)
    S = np.zeros((M * k, M * k))
    Ay = np.zeros((M * k, k))
    yy = np.zeros((k, k))
    for a in range(0, len(truth), 40000):
        b = min(a + 40000, len(truth))
        A = np.concatenate([m[a:b, :k] for m in members_T], axis=1).astype(np.float64)
        y = truth[a:b, :k].astype(np.float64)
        S += A.T @ A
        Ay += A.T @ y
        yy += y.T @ y
    return {"S": S, "Ay": Ay, "yy": yy, "n": len(truth)}


def add(stats: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {key: sum(s[key] for s in stats) for key in ("S", "Ay", "yy", "n")}


def system(s: dict[str, np.ndarray], Mt: np.ndarray, m: int) -> tuple[np.ndarray, np.ndarray]:
    k = len(Mt)
    H = s["S"] * np.kron(np.ones((m, m)), Mt)
    b = np.einsum("mjl,jl->mj", s["Ay"].reshape(m, k, k), Mt).reshape(-1)
    return H, b


def loss(w: np.ndarray, s: dict[str, np.ndarray], Mt: np.ndarray, m: int) -> float:
    H, b = system(s, Mt, m)
    return float((w @ H @ w - 2 * w @ b + np.trace(Mt @ s["yy"])) / s["n"])


def psi_mse(w: np.ndarray, s: dict[str, np.ndarray], m: int) -> float:
    return loss(w, s, np.eye(len(s["yy"])), m)


def expansion(m: int, k: int, block: int | None) -> np.ndarray:
    if block is None:
        return np.eye(m * k)
    groups = np.arange(k) // block
    g = groups.max() + 1
    E = np.zeros((m * k, m * g))
    for mm in range(m):
        E[mm * k + np.arange(k), mm * g + groups] = 1.0
    return E


def fit(s: dict[str, np.ndarray], Mt: np.ndarray, m: int, lam: float, E: np.ndarray) -> np.ndarray:
    """Weights (M*K,) minimising the metric loss plus lam * trace-scaled ridge towards 1/M."""
    H, b = system(s, Mt, m)
    Hr, br = E.T @ H @ E, E.T @ b
    scale = np.diag(Hr).copy()
    w0 = np.full(len(br), 1.0 / m)
    P = lam * np.diag(scale)
    theta = np.linalg.solve(Hr + P + 1e-12 * np.eye(len(br)) * scale.mean(), br + P @ w0)
    return E @ theta


def fit_cv(train: list[dict[str, np.ndarray]], Mt: np.ndarray, m: int, E: np.ndarray
           ) -> tuple[np.ndarray, float]:
    inner = np.zeros(len(LAMBDAS))
    for i in range(len(train)):
        rest = add([s for j, s in enumerate(train) if j != i])
        for g, lam in enumerate(LAMBDAS):
            inner[g] += loss(fit(rest, Mt, m, lam, E), train[i], Mt, m)
    lam = LAMBDAS[int(np.argmin(inner))]
    return fit(add(train), Mt, m, lam, E), lam


def column_stats(s: dict[str, np.ndarray], m: int, k: int):
    """Per-column (G, b, yy, n) for the psi columns, the form `fit_ridge_cv` takes."""
    idx = [np.arange(m) * k + j for j in range(k)]
    G = np.stack([s["S"][np.ix_(i, i)] for i in idx])
    b = np.stack([s["Ay"][i, j] for j, i in enumerate(idx)])
    return G, b, np.diag(s["yy"]).copy(), s["n"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("campaign", type=Path)
    ap.add_argument("--members", required=True)
    ap.add_argument("--basis-from", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--probe-frames", type=int, default=300)
    ap.add_argument("--delta", type=float, default=0.0332, help="jacobian_delta, as in training")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--jobs", type=int, default=30)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    folds = [0, 1, 2, 3, 4]
    members = args.members.split(",")
    m = len(members)
    basis = load_basis(args.campaign, args.basis_from)
    k = basis["pca"].n_components
    mpath = args.out / "metric.npz"
    if mpath.exists():
        z = np.load(mpath)
        Mt, info = z["M"], json.loads(str(z["info"]))
    else:
        Mt, info = metric_matrix(basis, args.campaign / "fold0", args.probe_frames, args.delta,
                                 args.jobs)
        np.savez(mpath, M=Mt, info=json.dumps(info))
    print(f"metric: {info}", flush=True)
    stack_cv.LINEAR_VARIANT = "ridgecv_block"
    tables, stats = {}, {}
    for f in folds:
        t = load_fold(args.campaign, members, f, basis, args.jobs)
        del t["Y"]
        t["members_T"] = member_targets(t)
        del t["member_coef"], t["member_qb"]
        t["truth"] = np.column_stack([t["true_coef"], t["S"]])
        stats[f] = fold_stats(t["members_T"], t["truth"], k)
        if args.score:
            tables[f] = t
        print(f"fold {f}: {stats[f]['n']} frames", flush=True)
    E_free, E_block = expansion(m, k, None), expansion(m, k, 10)
    report: dict[str, Any] = {"members": members, "metric": info, "folds": {}}
    weights: dict[int, dict[str, np.ndarray]] = {}
    for f in folds:
        train = [stats[j] for j in folds if j != f]
        cols = [column_stats(s, m, k) for s in train]
        groups = [np.arange(a, min(a + 10, k)) for a in range(0, k, 10)]
        W_ridge = fit_ridge_cv(cols, groups)          # (M, K) psi columns
        ws = {"equal": np.full(m * k, 1.0 / m), "ridgecv_block": W_ridge.T.reshape(-1)}
        ws["metric_free"], lam_f = fit_cv(train, Mt, m, E_free)
        ws["metric_block"], lam_b = fit_cv(train, Mt, m, E_block)
        weights[f] = ws
        held = stats[f]
        entry = {name: {"metric_loss": loss(w, held, Mt, m), "psi_mse": psi_mse(w, held, m)}
                 for name, w in ws.items()}
        entry["lambda"] = {"metric_free": lam_f, "metric_block": lam_b}
        ref = entry["ridgecv_block"]["metric_loss"]
        print(f"fold {f}: metric loss / ridgecv_block " + "  ".join(
            f"{n} {e['metric_loss'] / ref:.4f}" for n, e in entry.items() if n != "lambda")
            + f"  (lambda free {lam_f:g}, block {lam_b:g})", flush=True)
        report["folds"][f] = entry
        (args.out / "report.json").write_text(json.dumps(report, indent=1, default=float))
    if not args.score:
        return 0
    best = min(("metric_free", "metric_block"), key=lambda n: np.mean(
        [report["folds"][f][n]["metric_loss"] / report["folds"][f]["ridgecv_block"]["metric_loss"]
         for f in folds]))
    equal = np.full(m, 1.0 / m)
    for f in folds:
        t = tables[f]
        cols = [column_stats(stats[j], m, k) for j in folds if j != f]
        qb = stack_cv.fit_linear([tables[j] for j in folds if j != f])[:, k:]  # q95/betaN
        stage1 = sum(w * mt for w, mt in zip(equal, t["members_T"], strict=True))
        for name in ("ridgecv_block", best):
            W = np.column_stack([weights[f][name].reshape(m, k), qb])
            payload = corrected_payload(t, stage1, linear_blend(t, W) - stage1, equal, basis)
            (r,) = score_fold(t["files"], payload, args.jobs)["results"].values()
            del payload
            report["folds"][f].setdefault("native", {})[name] = r
            print(f"fold {f} {name}: S {r['S']:.6f} Cons {r['consistency']:.6f} "
                  f"D_LCFS {r['dlcfs']:.6f} R2psi {r['r2_psi']:.6f}", flush=True)
            (args.out / "report.json").write_text(json.dumps(report, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
