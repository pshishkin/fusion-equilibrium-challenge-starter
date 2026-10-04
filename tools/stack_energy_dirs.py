#!/usr/bin/env python3
"""Recalibrate the stack's coefficients along its worst poloidal-field-ERROR directions (Astra, 2026-10-04).

    uv run python tools/stack_energy_dirs.py runs/cv5_fixcos --members ... \
        --basis-from runs/cap_mlp_p100_3x512 --out runs/edirs_s15 [--ms 5,10,20]

li is <B_p^2>, B_p = |grad psi| / R. In coefficient space the squared poloidal-field error over
the scorer's plasma envelope is a quadratic form, e' G e (`target_metric.poloidal_field_form`,
restricted to the envelope). PCA orders directions by flux variance; this looks at the linear
stack's OOF errors in G-whitened coordinates (z = c L, G = L L'), where every unit of error is a
unit of field-error energy:

1. on the training folds, the eigen-directions of the error covariance (most field-error energy
   first) and, per direction u, the least-squares scale s_u of the prediction onto the truth;
2. on the held-out fold, the m most expensive directions rescaled by their s_u (a 1-parameter
   shrink per direction; s_u < 1 where a direction is predicted with more amplitude than it earns);
3. native score for m in --ms against m = 0.

A pure rotation changes nothing: the only freedom is the per-direction scale on the chosen
directions. Diagnostics: per direction, its share of the field-error energy, s_u and held-out R^2.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cons_residual_gate import geometry
from cv_oof import score_fold
from my_experiments.target_metric import _grad
from stack_cv import corrected_payload, linear_blend, load_basis, load_fold, member_targets
from stack_linear_variants import fit_ridge_cv, fold_stats

FOLDS = [0, 1, 2, 3, 4]


def groups(k: int) -> list[np.ndarray]:
    cols = np.arange(k + 2)
    return [cols[a:min(a + 10, k)] for a in range(0, k, 10)] + [np.array([k]), np.array([k + 1])]


def field_form(components: np.ndarray, R: np.ndarray, Z: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """e' G e = integral over the envelope of |grad e|^2 / R (constant factors dropped)."""
    gr, gz = _grad(components, R, Z)
    w = (np.sqrt(1.0 / R)[None, :] * mask)[None]
    a = (gr * w).reshape(len(components), -1)
    b = (gz * w).reshape(len(components), -1)
    return np.asarray(a @ a.T + b @ b.T, dtype=np.float64)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("campaign", type=Path)
    ap.add_argument("--members", required=True)
    ap.add_argument("--basis-from", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ms", default="5,10,20")
    ap.add_argument("--jobs", type=int, default=20)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    members = args.members.split(",")
    ms = [int(x) for x in args.ms.split(",")]
    basis = load_basis(args.campaign, args.basis_from)
    k = basis["pca"].n_components
    equal = np.full(len(members), 1.0 / len(members))
    stats, C, Y = {}, {}, {}
    files0 = None
    for f in FOLDS:
        t = load_fold(args.campaign, members, f, basis, args.jobs)
        files0 = files0 or t["files"]
        stats[f] = fold_stats(member_targets(t), np.column_stack([t["true_coef"], t["S"]]))
        del t
        print(f"fold {f}: statistics", flush=True)
    W = {f: fit_ridge_cv([stats[j] for j in FOLDS if j != f], groups(k)) for f in FOLDS}
    for f in FOLDS:
        t = load_fold(args.campaign, members, f, basis, args.jobs)
        C[f] = linear_blend({"members_T": member_targets(t)}, W[f])[:, :k].astype(np.float64)
        Y[f] = np.asarray(t["true_coef"], dtype=np.float64)
        del t
        print(f"fold {f}: OOF coefficients", flush=True)
    _, R, Z, mask, _ = geometry(files0)
    comps = np.asarray(basis["pca"].pca.components_, dtype=np.float64).reshape(k, 65, 65)
    G = field_form(comps, R, Z, mask.astype(np.float64))
    G += 1e-9 * np.trace(G) / k * np.eye(k)
    L = np.linalg.cholesky(G)          # G = L L'; energy of e = ||e L||^2
    Linv = np.linalg.inv(L)
    report: dict = {"members": members, "ms": ms, "folds": {}}
    for f in FOLDS:
        tr = [j for j in FOLDS if j != f]
        z = np.concatenate([C[j] for j in tr]) @ L
        y = np.concatenate([Y[j] for j in tr]) @ L
        e = y - z
        evals, U = np.linalg.eigh(e.T @ e / len(e))
        order = np.argsort(evals)[::-1]
        evals, U = evals[order], U[:, order]
        p, q = z @ U, y @ U
        s = (p * q).sum(0) / (p * p).sum(0)
        zh, yh = C[f] @ L @ U, Y[f] @ L @ U
        r2 = 1 - ((yh - zh) ** 2).sum(0) / ((yh - yh.mean(0)) ** 2).sum(0)
        share = evals / evals.sum()
        diag = [{"share": float(share[i]), "s": float(s[i]), "r2_heldout": float(r2[i])}
                for i in range(20)]
        print(f"fold {f}: top directions (share, s, held-out R2): " + "  ".join(
            f"{d['share']:.3f}/{d['s']:.3f}/{d['r2_heldout']:.3f}" for d in diag[:10]), flush=True)
        entry: dict = {"directions": diag}
        t = load_fold(args.campaign, members, f, basis, args.jobs)
        del t["Y"]
        t["members_T"] = member_targets(t)
        stage1 = sum(w * m for w, m in zip(equal, t["members_T"], strict=True))
        blend = linear_blend(t, W[f])
        for m in [0, *ms]:
            b2 = blend.copy()
            if m:
                zf = C[f] @ L
                proj = zf @ U[:, :m]
                zf = zf + (proj * (s[:m] - 1.0)) @ U[:, :m].T
                b2[:, :k] = zf @ Linv
            payload = corrected_payload(t, stage1, b2 - stage1, equal, basis)
            (r,) = score_fold(t["files"], payload, args.jobs)["results"].values()
            del payload
            entry[f"m{m}"] = r
            c = r["r2_cons_each"]
            print(f"fold {f} m{m}: S {r['S']:.6f} li {c['li']:.6f} kappa {c['kappa']:.6f} "
                  f"dlcfs {r['dlcfs']:.6f} cons {r['consistency']:.6f}", flush=True)
            report["folds"][str(f)] = entry
            (args.out / "report.json").write_text(json.dumps(report, indent=1, default=float))
        del t
    for m in [0, *ms]:
        sv = [report["folds"][str(f)][f"m{m}"]["S"] for f in FOLDS]
        li = [report["folds"][str(f)][f"m{m}"]["r2_cons_each"]["li"] for f in FOLDS]
        print(f"m{m:<3d} mean S {np.mean(sv):.6f}  li {np.mean(li):.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
