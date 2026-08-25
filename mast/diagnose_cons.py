#!/usr/bin/env python3
"""Why each Consistency scalar scores what it scores -- bias, scale, or the wrong sign entirely.

    uv run python mast/diagnose_cons.py --updown 0.0 -0.01

The composite floors each of the seven pooled R2 at zero, so a scalar is worth nothing until it
crosses, and "how negative" is NOT how far it has to travel. This separates the three ways a
scalar can be below zero, because they have completely different prices:

* **offset** -- our value is displaced but tracks the truth. One constant in the map fixes it.
* **scale**  -- our value swings too far or too little. Still reachable by a global correction.
* **sign**   -- our value moves the OPPOSITE way to the truth. No static correction reaches this;
  the map's dynamics are wrong and only the physics fixes it.

What it found on 2026-08-19, and the reason it exists: five of the seven are the third kind.
R_axis correlates **-0.75** with the truth, tri_top -0.74, tri_bot -0.65, and each holds inside
every demo shot separately (R_axis -0.82 / -0.71 / -0.73), so it is not a per-shot offset. Against
the discharge current the split is explicit -- the truth's R_axis rises with Ip at +0.79 while ours
FALLS at -0.47. Only li moves the right way, at +0.82, and li is the one quantity of the seven that
reads the current profile rather than the boundary.

**And the scariest number in the table is worth nothing.** Z_axis reads -4894 because the truth's
Z_axis has a standard deviation of 0.0012 m -- one twenty-fifth of a grid cell. Predicting its mean
exactly scores R2 = 0, which is what the floor already gives us. Chasing it is chasing zero.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "fusion_scoring"))

from common import CONS_SCALARS  # noqa: E402

import local_score as ls  # noqa: E402
from mast.calibration import SHIPPED  # noqa: E402
from mast.predict import predict_row  # noqa: E402
from mast.sweep import DEMO, Fold  # noqa: E402


def measure(fold: Fold, cals: tuple) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`(truth, ours, shot_id)` for the seven derived scalars, frame by frame."""
    truth, ours, ids = [], [], []
    for i, s in enumerate(fold.shots):
        p = predict_row(s["row"], cals)
        cons_gt, cmask = fold.refs[i][1], fold.refs[i][2]
        r = ls.build_reference(p["psirz"], fold.R, fold.Z, fold.mask, fold.mask_f, fold.machine)
        truth.append(np.where(cmask, cons_gt, np.nan))
        ours.append(np.where(r[2], r[1], np.nan))
        ids.append(np.full(len(cons_gt), i))
    return np.concatenate(truth), np.concatenate(ours), np.concatenate(ids)


def table(T: np.ndarray, P: np.ndarray, ids: np.ndarray) -> None:
    print(f"{'scalar':<9}{'R2':>10}{'corr':>8}{'within-shot corr':>24}"
          f"{'our sd / truth sd':>20}{'R2 if affine':>14}")
    for j, name in enumerate(CONS_SCALARS):
        t, o = T[:, j], P[:, j]
        ok = np.isfinite(t) & np.isfinite(o)
        if ok.sum() < 4 or t[ok].std() == 0:
            print(f"{name:<9}  not scorable on this fold")
            continue
        tt, oo = t[ok], o[ok]
        sst = ((tt - tt.mean()) ** 2).sum()
        r2 = 1 - ((tt - oo) ** 2).sum() / sst
        A = np.column_stack([oo, np.ones_like(oo)])
        w = np.linalg.lstsq(A, tt, rcond=None)[0]
        per = []
        for s in np.unique(ids):
            m = ok & (ids == s)
            per.append(np.corrcoef(t[m], o[m])[0, 1] if m.sum() > 3 and t[m].std() > 0
                       else np.nan)
        print(f"{name:<9}{r2:>10.2f}{np.corrcoef(tt, oo)[0, 1]:>8.3f}   "
              + " ".join(f"{v:+6.2f}" for v in per)
              + f"{oo.std() / tt.std():>16.2f}{1 - ((tt - A @ w) ** 2).sum() / sst:>14.3f}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--updown", type=float, nargs="+", default=[0.0])
    ap.add_argument("--edge", type=float, nargs="+", default=[None])
    ap.add_argument("--coil", default=None,
                    help="coil family to rescale, e.g. p6 (applies to both u and l)")
    ap.add_argument("--coil-scale", type=float, nargs="+", default=[1.0])
    ap.add_argument("--jobs", type=int, default=3)
    a = ap.parse_args(argv)
    fold = Fold(sorted(DEMO.glob("mast_shot_*.parquet")), a.jobs)
    for u in a.updown:
        for e in a.edge:
            for g in a.coil_scale:
                cals = []
                for c in SHIPPED:
                    gains = dict(c.gains)
                    if a.coil is not None:
                        for k in gains:
                            if f"_{a.coil}" in k:
                                gains[k] *= g
                    cals.append(replace(c, updown=u, gains=gains,
                                        profile=(c.profile if e is None
                                                 else replace(c.profile, edge=e))))
                label = f"updown={u:+.4f}"
                if e is not None:
                    label += f" edge={e:.3f}"
                if a.coil is not None:
                    label += f" {a.coil}x{g:.3f}"
                print(f"\n=== {label} ===")
                truth, pred, ids = measure(fold, tuple(cals))
                table(truth, pred, ids)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
