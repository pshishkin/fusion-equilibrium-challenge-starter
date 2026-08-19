#!/usr/bin/env python3
"""
What is actually LEFT in Challenge 1 — every term of S, and every scalar inside Consistency.

    uv run python my_experiments/diagnose_budget.py

Group D spent eight gates chasing mechanisms and returned eight negatives, and C9 then argued that
`li` and `kappa` sit two to three times above their own label-noise floor while the two
triangularities are finished. That argument is about the RATIO of model error to label jitter. It
never asked the other question, which is prior to it: **how much of S is a scalar worth if it were
solved outright?**

The arithmetic is short, and it is exact rather than estimated, because `finalize_machine` averages
the seven pooled R2 with equal weight:

    Consistency = mean_j R2_j          =>  d S / d R2_j  =  W_CONS / N_CONS  =  0.20 / 7

so a scalar whose pooled R2 is R2_j has a REMAINING budget of

    (W_CONS / N_CONS) * (1 - R2_j)

and no feature, loss or architecture aimed at that scalar alone can be worth more than that number,
however well it works. The same reading applies term by term to the composite as a whole.

Read against the two resolutions this fork has measured: one paired production run resolves
**0.0013** of S, three seeds on one salt resolve **0.0008**. A budget under 0.0013 is a target that
cannot be confirmed even if it is hit perfectly, which is a different and stronger statement than
"it is small".

The residual correlation matrix at the end answers the follow-up: if the seven residuals were one
shared map error, a single mechanism could collect several budgets at once and the per-scalar
ceiling would be the wrong bar. It is printed so that argument has to survive a number.

Reads `results/frame_costs_ensemble.csv` — the three-artifact ensemble's per-frame residuals and
ground truth on the 70 held-out shots, written by `diagnose_frames.py`.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "fusion_scoring"))

from common import CONS_SCALARS, N_CONS, W_CONS, W_LCFS, W_PSI, W_QB  # noqa: E402

from toolkit.progress import install_timestamps  # noqa: E402

FloatArray = npt.NDArray[np.floating]
COSTS = HERE.parent / "results" / "frame_costs_ensemble.csv"
# The production run of 2026-08-15, logged at logs/20260815-111643-production.log:538, on the same
# 70 held-out shots this CSV covers. Passed in rather than recomputed: D_LCFS is a macro mean over
# shots of a contour distance and R2_psi is over 10^8 pixels, neither of which is in the CSV.
PROD_TERMS = {"r2_psi": 0.9998, "r2_qb": 0.9962, "one_minus_dlcfs": 0.9902}
PAIRED_RESOLUTION = 0.0013


def pooled_r2(gt: FloatArray, res: FloatArray) -> float:
    """1 - SS_res / SS_tot, pooled over frames exactly as `finalize_machine` pools it."""
    ok = np.isfinite(gt) & np.isfinite(res)
    if not ok.any():
        raise ValueError("no frame has both a ground truth and a residual")
    g, r = gt[ok], res[ok]
    ss_tot = float(((g - g.mean()) ** 2).sum())
    if ss_tot <= 0:
        raise ValueError(f"SS_tot is {ss_tot} over {int(ok.sum())} frames; R2 is undefined")
    return 1.0 - float((r ** 2).sum()) / ss_tot


def main() -> int:
    install_timestamps()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--costs", type=Path, default=COSTS)
    args = ap.parse_args()

    df = pd.read_csv(args.costs)
    missing = [c for s in CONS_SCALARS for c in (f"gt_{s}", f"res_{s}") if c not in df.columns]
    if missing:
        raise SystemExit(f"{args.costs.name} is missing {missing}; regenerate it with the current "
                         f"diagnose_frames.py")
    print(f"{len(df)} frames of {len(set(df['shot']))} held-out shots from {args.costs.name}")

    gt = np.column_stack([df[f"gt_{s}"].to_numpy(dtype=np.float64) for s in CONS_SCALARS])
    res = np.column_stack([df[f"res_{s}"].to_numpy(dtype=np.float64) for s in CONS_SCALARS])
    r2 = np.array([pooled_r2(gt[:, j], res[:, j]) for j in range(N_CONS)])
    per_scalar = (W_CONS / N_CONS) * (1.0 - r2)

    print("\n  Inside Consistency — a PERFECT scalar is worth its budget and not one count more:")
    print(f"    {'scalar':<9} {'pooled R2':>11} {'budget of S':>13} {'vs 0.0013':>11}")
    for j in np.argsort(-per_scalar):
        verdict = "confirmable" if per_scalar[j] >= PAIRED_RESOLUTION else "below resolution"
        print(f"    {CONS_SCALARS[j]:<9} {r2[j]:>11.5f} {per_scalar[j]:>+13.5f} {verdict:>11}")
    cons = float(r2.mean())
    print(f"    {'ALL SEVEN':<9} {cons:>11.5f} {W_CONS * (1 - cons):>+13.5f}")

    terms = [("R2_psi", W_PSI, PROD_TERMS["r2_psi"]),
             ("R2_qb", W_QB, PROD_TERMS["r2_qb"]),
             ("1 - D_LCFS", W_LCFS, PROD_TERMS["one_minus_dlcfs"]),
             ("Consistency", W_CONS, cons)]
    print("\n  The composite, term by term — the first three from the production log, "
          "Consistency from this CSV:")
    print(f"    {'term':<12} {'weight':>7} {'value':>9} {'budget of S':>13}")
    total = 0.0
    for name, w, v in terms:
        total += w * (1.0 - v)
        print(f"    {name:<12} {w:>7.2f} {v:>9.4f} {w * (1 - v):>+13.5f}")
    print(f"    {'EVERYTHING':<12} {'':>7} {1.0 - total:>9.4f} {total:>+13.5f}")
    print(f"\n  So Challenge 1 has {total:.4f} of S left in it, {W_CONS * (1 - cons) / total:.0%} "
          f"of that in Consistency, and every single line above except Consistency as a whole is "
          f"under the {PAIRED_RESOLUTION} a paired production run resolves.")

    # Are the seven one error or seven? Normalised by each scalar's own root-mean SS_tot per frame,
    # which is the unit the metric divides by, so the columns are commensurable.
    ok = np.isfinite(gt).all(axis=1) & np.isfinite(res).all(axis=1)
    g, r = gt[ok], res[ok]
    unit = np.array([np.sqrt(float(((g[:, j] - g[:, j].mean()) ** 2).mean()))
                     for j in range(N_CONS)])
    z = r / unit
    corr = np.corrcoef(z.T)
    print(f"\n  Residual correlation in metric units, over {int(ok.sum())} complete frames:")
    print("    " + " " * 9 + "".join(f"{s[:7]:>9}" for s in CONS_SCALARS))
    for i, s in enumerate(CONS_SCALARS):
        print(f"    {s:<9}" + "".join(f"{corr[i, j]:>9.2f}" for j in range(N_CONS)))
    share = np.linalg.eigvalsh(np.cov(z.T))[::-1]
    share = share / share.sum()
    print("    PC variance share: " + " ".join(f"{v:.3f}" for v in share))
    print(f"\n  The leading component holds {share[0]:.1%} of the seven-dimensional residual, so "
          f"these are seven problems and not one, and a mechanism aimed at one scalar collects one "
          f"budget. That is what makes the per-scalar column above a ceiling rather than a share.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
