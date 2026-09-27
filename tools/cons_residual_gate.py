#!/usr/bin/env python3
"""Gate: can the stack's error in the seven psi-derived scalars be predicted from its inputs?

    uv run python tools/cons_residual_gate.py runs/cv5_fixcos --members mlp,seq,... \
        --basis-from runs/fc_p70 --out runs/consgate_fc11

Consistency is 68% of 1 - S (li, kappa and R_axis alone about 45%), yet the stack is fitted on
flux coefficients, not on the scalars scored from the map. A third stage could predict each
scalar's residual and move the map (in the basis) until its scalars match. This measures the
ceiling of that idea before any of it is built:

1. Per fold, the linear (basis-70 ridgecv_block) stack map, weights fitted on the other folds.
2. The seven scalars of that map, frame by frame, exactly as the scorer derives them; the truth
   side comes from the scorer's reference cache (same code, same files).
3. A gradient-boosted regressor per scalar on the stack features predicts the residual
   `truth - map scalar`, trained on four folds and evaluated on the fifth.

Reported per scalar: R2_cons of the map, and R2_cons if the predicted residual were applied
exactly. The difference, weighted as the composite weighs it (W_CONS / 7), bounds the gain of a
map correction that realises the scalar predictions perfectly.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import local_score
import stack_cv
from cv_oof import REF_CACHE, _read_scored_shot
from stack_cv import corrected_payload, features, linear_blend, load_basis, load_fold
from toolkit.parallel import pmap, resolve_jobs

def _scalars_task(args: tuple) -> np.ndarray:
    """(T, 7) scalars of one predicted shot, as score_shot derives them (psi sign +1). The pool
    spawns its workers, so the grid travels with every task rather than in a module global."""
    psi, machine, R, Z, mask_coarse, mask_f = args
    axis_sign = local_score.AXIS_SIGN[machine]
    out = np.full((len(psi), local_score.N_CONS), np.nan)
    for k in range(len(psi)):
        p = psi[k].astype(np.float64)
        ex, _ = local_score.extract_lcfs_with_sign(p, R, Z, machine, mask_coarse, mask_f,
                                                   n_iter=local_score.N_ITER, axis_sign=axis_sign)
        vals = local_score.derive_frame(p, R, Z, machine, mask_coarse, mask_f, contour=ex,
                                        axis_sign=axis_sign)
        out[k] = [vals[n] for n in local_score.CONS_SCALARS]
    return out


def geometry(files: list[Path]) -> tuple:
    """(machine, R, Z, mask_coarse, mask_f) as the scorer reads them off the shots."""
    return local_score._machine_of([_read_scored_shot(files[0])])


def true_scalars(files: list[Path]) -> np.ndarray:
    key = hashlib.sha256(json.dumps([local_score.SCORING_VERSION,
                                     [f.name for f in files]]).encode()).hexdigest()[:24]
    path = REF_CACHE / f"refs_{key}.joblib"
    if not path.exists():
        raise ValueError(f"{path}: no reference cache for this fold; score it once first")
    return np.concatenate([ref[1] for ref in joblib.load(path)])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("campaign", type=Path)
    ap.add_argument("--members", required=True)
    ap.add_argument("--basis-from", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--every", type=int, default=4, help="train the regressors on every Nth frame")
    ap.add_argument("--jobs", type=int, default=30)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    folds = [0, 1, 2, 3, 4]
    members = args.members.split(",")
    M = len(members)
    equal = np.full(M, 1.0 / M)
    basis = load_basis(args.campaign, args.basis_from)
    stack_cv.LINEAR_VARIANT = "ridgecv_block"
    tables = {}
    for k in folds:
        t = load_fold(args.campaign, members, k, basis, args.jobs)
        del t["Y"]
        t["members_T"] = stack_cv.member_targets(t)
        t["truth"] = np.column_stack([t["true_coef"], t["S"]])
        tables[k] = t
        print(f"fold {k}: {len(t['X'])} frames", flush=True)
    per_fold = {}
    for k in folds:
        t = tables[k]
        cache = args.out / f"scalars_fold{k}.npz"
        W = stack_cv.fit_linear([tables[j] for j in folds if j != k])
        F, stage1 = features(t, equal, False, "mean")
        shift = linear_blend(t, W) - stage1
        t["F"] = np.column_stack([F, stage1 + shift]).astype(np.float32)
        if cache.exists():
            z = np.load(cache)
            pred, true = z["pred"], z["true"]
        else:
            geo = geometry(t["files"])
            payload = corrected_payload(t, stage1, shift, equal, basis)
            shots = [(payload[f"shot_{i:04d}_psirz"], *geo) for i in range(len(t["files"]))]
            del payload
            pred = np.concatenate(pmap(_scalars_task, shots, resolve_jobs(args.jobs, len(shots))))
            del shots
            true = true_scalars(t["files"])
            np.savez(cache, pred=pred, true=true)
        t["cons_pred"], t["cons_true"] = pred, true
        ok = np.isfinite(pred) & np.isfinite(true)
        r2 = [1 - ((true[ok[:, j], j] - pred[ok[:, j], j]) ** 2).sum()
              / ((true[ok[:, j], j] - true[ok[:, j], j].mean()) ** 2).sum() for j in range(7)]
        per_fold[k] = r2
        print(f"fold {k}: map R2_cons " + " ".join(
            f"{n} {v:.5f}" for n, v in zip(local_score.CONS_SCALARS, r2)), flush=True)
    from sklearn.ensemble import HistGradientBoostingRegressor
    names = local_score.CONS_SCALARS
    report: dict[str, Any] = {"members": members, "scalars": names, "folds": {}}
    for k in folds:
        others = [j for j in folds if j != k]
        t = tables[k]
        entry = {}
        for j, name in enumerate(names):
            Xtr, ytr = [], []
            for o in others:
                u = tables[o]
                ok = np.isfinite(u["cons_pred"][:, j]) & np.isfinite(u["cons_true"][:, j])
                rows = np.flatnonzero(ok)[::args.every]
                Xtr.append(u["F"][rows])
                ytr.append(u["cons_true"][rows, j] - u["cons_pred"][rows, j])
            model = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.08,
                                                  max_leaf_nodes=63, min_samples_leaf=100,
                                                  l2_regularization=1.0, random_state=j)
            model.fit(np.concatenate(Xtr), np.concatenate(ytr))
            ok = np.isfinite(t["cons_pred"][:, j]) & np.isfinite(t["cons_true"][:, j])
            y = t["cons_true"][ok, j]
            p = t["cons_pred"][ok, j]
            corr = model.predict(t["F"][ok])
            sst = ((y - y.mean()) ** 2).sum()
            before = 1 - ((y - p) ** 2).sum() / sst
            after = 1 - ((y - p - corr) ** 2).sum() / sst
            entry[name] = {"before": before, "after": after}
            print(f"fold {k} {name:>8}: R2_cons {before:.5f} -> {after:.5f} "
                  f"({after - before:+.5f})", flush=True)
        gain = np.mean([e["after"] - e["before"] for e in entry.values()])
        entry["dS_bound"] = 0.20 * gain
        print(f"fold {k}: mean dR2_cons {gain:+.5f}, dS bound {0.20 * gain:+.6f}", flush=True)
        report["folds"][k] = entry
        (args.out / "report.json").write_text(json.dumps(report, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
