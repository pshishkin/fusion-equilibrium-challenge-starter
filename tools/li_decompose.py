#!/usr/bin/env python3
"""Oracle diagnostic: is the li error the CONTOUR (integration region and boundary loop) or the
FIELD inside it?

For a sample of fold-0 shots (the worst li shots of the stacked map plus random ones) the
ensemble-mean OOF map is compared with the truth through the scorer's own `internal_inductance`
with the four pairings of (field, contour): (pred, pred) is what is scored, (true, true) the
reference, (pred field, true contour) the error left if the boundary were perfect, (true field,
pred contour) the error the boundary alone causes. Reported as shares of the scored SSE. Oracle
only — the truth contour is not available at inference.
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import local_score
from cv_oof import _fold_files, _read_scored_shot
from derive import _bilinear, _points_in_polygon, _poloidal_field, internal_inductance
from my_experiments import baseline_model as B
from toolkit.parallel import pmap


def li_parts(psi, R, Z, machine, C):
    """(<Bp^2>_V, (loop Bp dl / lp)^2) — li's numerator and denominator, as the scorer builds them."""
    Bp = _poloidal_field(psi, R, Z, machine)
    RR, ZZ = np.meshgrid(R, Z)
    inside = _points_in_polygon(RR.ravel(), ZZ.ravel(), C).reshape(RR.shape)
    dV = RR * float(np.mean(np.diff(R))) * float(np.mean(np.diff(Z)))
    Bp2 = np.where(np.isfinite(Bp), Bp, 0.0) ** 2
    num = float(np.sum(Bp2[inside] * dV[inside]) / np.sum(dV[inside]))
    Cc = np.vstack([C, C[0]]); seg = np.sqrt(np.sum(np.diff(Cc, axis=0) ** 2, axis=1))
    on = _bilinear(Bp, R, Z, C[:, 0], C[:, 1])
    den = (float(np.sum(0.5 * (on + np.roll(on, -1)) * seg)) / float(np.sum(seg))) ** 2
    return num, den


def _task(args):
    pred, true, machine, R, Z, mc, mf = args
    axis = local_score.AXIS_SIGN[machine]
    out = np.full((len(pred), 8), np.nan)
    for i in range(len(pred)):
        p, t = pred[i].astype(np.float64), true[i].astype(np.float64)
        cp, _ = local_score.extract_lcfs_with_sign(p, R, Z, machine, mc, mf,
                                                   n_iter=local_score.N_ITER, axis_sign=axis)
        ct = local_score.extract_lcfs(t, R, Z, machine, mc, mf, n_points=local_score.N_POINTS)
        if cp is None or ct is None:
            continue
        out[i, :4] = [internal_inductance(p, R, Z, machine, cp), internal_inductance(t, R, Z, machine, ct),
                      internal_inductance(p, R, Z, machine, ct), internal_inductance(t, R, Z, machine, cp)]
        out[i, 4:6] = li_parts(p, R, Z, machine, ct)
        out[i, 6:8] = li_parts(t, R, Z, machine, ct)
    return out


def main() -> int:
    members = sys.argv[1].split(",")
    campaign = Path("runs/cv5_fixcos")
    art0 = joblib.load(campaign / "fold0" / "baseline.joblib")
    files = _fold_files(art0)
    z = np.load("runs/consgate_fc11/scalars_fold0.npz")
    lens = np.load(campaign / "fold0" / "oof.npz")["lengths"]
    b = np.concatenate([[0], np.cumsum(lens)])
    e = np.where(np.isfinite(z["pred"][:, 6]) & np.isfinite(z["true"][:, 6]),
                 (z["true"][:, 6] - z["pred"][:, 6]) ** 2, 0.0)
    per = np.add.reduceat(e, b[:-1])
    rng = np.random.default_rng(0)
    worst = np.argsort(per)[::-1][:50]
    rand = rng.choice(np.setdiff1d(np.arange(len(lens)), worst), 100, replace=False)
    pick = np.sort(np.concatenate([worst, rand]))
    X, Y, _, lp = B._read_shots([files[i] for i in pick], "reading", 1.0, 24)
    maps = np.zeros(Y.shape, np.float64)
    rows = np.concatenate([np.arange(b[i], b[i + 1]) for i in pick])
    for m in members:
        src, _, name = m.rpartition(":")
        d = (Path(src) if src else campaign) / "fold0"
        art = joblib.load(d / "baseline.joblib")
        T = np.load(d / "oof.npz")[f"targets_{name}"][rows]
        n = int(art["n_pca"])
        maps += art["pca"].inverse_transform(T[:, :n]).reshape(-1, 65, 65) + B.coil_flux(art["coil"], X)
    maps /= len(members)
    geo = local_score._machine_of([_read_scored_shot(files[0])])
    bb = np.concatenate([[0], np.cumsum(lp)])
    tasks = [(maps[bb[i]:bb[i + 1]], Y[bb[i]:bb[i + 1]], *geo) for i in range(len(pick))]
    res = np.concatenate(pmap(_task, tasks, 30))
    ok = np.isfinite(res).all(1)
    lp_, lt, l_pt, l_tp, num_p, den_p, num_t, den_t = res[ok].T
    # With the true contour: log li error = log-num error - log-den error; split its variance.
    dn, dd = np.log(num_p / num_t), np.log(den_p / den_t)
    tot = dn - dd
    print(f"li log-error with TRUE contour: var {tot.var():.3g}; interior <Bp^2> part var {dn.var():.3g}, "
          f"boundary (loop Bp/lp)^2 part var {dd.var():.3g}, correlation {np.corrcoef(dn, dd)[0, 1]:+.2f}")
    worst_rows = np.concatenate([np.full(lp[i], pick[i] in set(worst)) for i in range(len(pick))])[ok]
    for label, sel in (("all sampled", np.ones(ok.sum(), bool)), ("worst-50 shots", worst_rows),
                       ("random-100 shots", ~worst_rows)):
        sse = ((lp_ - lt)[sel] ** 2).sum()
        print(f"{label:>17}: frames {sel.sum():6d} | scored SSE 1.00 | pred field + TRUE contour "
              f"{((l_pt - lt)[sel] ** 2).sum() / sse:.2f} | true field + PRED contour "
              f"{((l_tp - lt)[sel] ** 2).sum() / sse:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
