#!/usr/bin/env python3
"""
D5 — the ORACLE ceiling on every per-shot confidence idea in ideas.md.

    uv run python my_experiments/diagnose_shrinkage.py

A23 and its descendants all propose the same operation: where the model is unsure, pull its answer
toward the fold mean. Which factor to pull by is the hard half, and every version of it differs only
in how well that factor is estimated. So the question worth asking first is not "how well can it be
estimated" but "how much would a PERFECT estimate be worth" — and that has a closed form on data
already on disk, because `diagnose_frames.py` now writes the ground-truth scalars beside the
residual.

An oracle is given the answer and asked only to choose the shrinkage. Three of them, increasingly
generous, and each is a ceiling on the one above:

  * one factor per SHOT, shared by the seven scalars — what a per-shot confidence signal could buy;
  * one factor per shot and per SCALAR — a confidence signal that knows which functional is wrong;
  * one factor per FRAME and per scalar, which is degenerate and must return exactly 1.0000 — it is
    printed as an arithmetic check on the two above, not as a bound anything could approach.

Every factor is chosen with each scalar in units of its own SS_tot, because the score is: R2 divides
each functional by its own spread, so a factor SHARED across scalars has to be chosen in that space.
Chosen in raw units instead, the shared arm comes out worse than doing nothing — an impossible
answer for an oracle, and the one that caught the mistake.

A real estimator recovers a fraction of an oracle, never the whole of it, so a ceiling below the
threshold refutes every member of the family at once. **Pre-registered kill: dead below +0.0065 of
the pooled mean R2 over the seven scalars at the LOOSEST oracle**, which is +0.0013 of S once the
0.20 Consistency weight is applied — what one paired production run resolves.

Read against the algebra this is meant to check rather than replace: optimal shrinkage of an
estimator at R2 = r buys (1-r)^2/(2-r), which summed over the seven measured R2 is +0.0001 of S.
If the oracle lands there, the algebra was right and the family is closed on both grounds.
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

from common import CONS_SCALARS, N_CONS, W_CONS  # noqa: E402

from my_experiments.progress import install_timestamps  # noqa: E402

FloatArray = npt.NDArray[np.floating]
COSTS = HERE.parent / "results" / "frame_costs_ensemble.csv"


def r2(truth: FloatArray, est: FloatArray, ok: npt.NDArray[np.bool_]) -> FloatArray:
    """Per scalar, 1 - SS_res/SS_tot against the fold mean, the form `metrics.py` uses."""
    out = np.full(N_CONS, np.nan)
    for j in range(N_CONS):
        m = ok[:, j]
        t, e = truth[m, j], est[m, j]
        tot = float(((t - t.mean()) ** 2).sum())
        out[j] = 1.0 - float(((t - e) ** 2).sum()) / tot if tot > 0 else np.nan
    return out


def best_alpha(t: FloatArray, d: FloatArray) -> float:
    """The factor minimising ||t - alpha*d||^2 — the oracle's whole freedom, in one number.

    `t` is the truth measured from the fold mean and `d` the prediction measured from the same
    point, so alpha = 1 leaves the prediction alone and alpha = 0 replaces it with the fold mean.
    """
    den = float((d * d).sum())
    return float((t * d).sum() / den) if den > 0 else 1.0


def main() -> int:
    install_timestamps()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--costs", type=Path, default=COSTS)
    args = ap.parse_args()

    df = pd.read_csv(args.costs)
    missing = [f"gt_{s}" for s in CONS_SCALARS if f"gt_{s}" not in df.columns]
    if missing:
        raise SystemExit(f"{args.costs.name} has no {missing[0]} — regenerate it with the current "
                         f"diagnose_frames.py, which writes the ground-truth scalars")
    truth = df[[f"gt_{s}" for s in CONS_SCALARS]].to_numpy(dtype=np.float64)
    pred = truth - df[[f"res_{s}" for s in CONS_SCALARS]].to_numpy(dtype=np.float64)
    ok = np.isfinite(truth) & np.isfinite(pred)
    shots = df["shot"].to_numpy()
    print(f"{len(df)} frames of {len(set(shots))} shots from {args.costs.name}")

    # The fold mean is the point the scorer's R2 measures everything from, so it is also the point
    # shrinkage pulls toward. Per scalar, over the frames where that scalar is defined.
    mean = np.array([truth[ok[:, j], j].mean() for j in range(N_CONS)])
    t0, d0 = truth - mean, pred - mean
    # Each scalar in units of its own SS_tot, because the score does: R2 divides every functional by
    # its own spread, so a factor shared ACROSS scalars has to be chosen in that space too. Without
    # this the shared-factor arm minimises the raw sum of squares, which `volume` in cubic metres
    # dominates over `kappa` near 1.8 — and it came out WORSE than doing nothing, which is the
    # impossible answer that exposed the mistake.
    unit = np.array([np.sqrt(float((t0[ok[:, j], j] ** 2).sum())) for j in range(N_CONS)])
    tn, dn = t0 / unit, d0 / unit

    base = r2(truth, pred, ok)
    print("\n  baseline pooled R2 per scalar: "
          + ", ".join(f"{s} {v:.4f}" for s, v in zip(CONS_SCALARS, base, strict=True)))
    print(f"  pooled mean {np.nanmean(base):.5f}")

    arms: dict[str, FloatArray] = {}

    # One factor per shot, shared by the seven — a per-shot confidence signal's ceiling.
    a = np.ones(len(df))
    for s in dict.fromkeys(shots):
        m = shots == s
        a[m] = best_alpha(tn[m][ok[m]], dn[m][ok[m]])
    arms["one factor per shot"] = mean + a[:, None] * d0
    per_shot = a.copy()

    # One factor per shot AND per scalar.
    a7 = np.ones((len(df), N_CONS))
    for s in dict.fromkeys(shots):
        m = shots == s
        for j in range(N_CONS):
            sel = m & ok[:, j]
            if sel.any():
                a7[m, j] = best_alpha(t0[sel, j], d0[sel, j])
    arms["per shot and scalar"] = mean + a7 * d0

    # One factor per frame and per scalar. This one is DEGENERATE and is printed as a check, not as
    # a bound: a factor free to vary per frame reconstructs the truth exactly, so it must come out
    # at R2 = 1.0000. If it does not, the arithmetic above is wrong somewhere.
    with np.errstate(divide="ignore", invalid="ignore"):
        af = np.where(np.abs(d0) > 0, t0 / d0, 1.0)
    arms["per frame and scalar"] = mean + af * d0

    print("\n  oracle shrinkage, pooled mean R2 over the seven scalars:")
    for name, est in arms.items():
        got = np.nanmean(r2(truth, est, ok))
        print(f"    {name:<22} {got:.5f}   {got - np.nanmean(base):+.5f}   "
              f"= {W_CONS * (got - np.nanmean(base)):+.5f} of S")

    lo, hi = np.percentile(per_shot, [5, 95])
    print(f"\n  the per-shot factor itself: median {np.median(per_shot):.4f}, "
          f"5-95% {lo:.4f}..{hi:.4f}, and {float((per_shot < 1).mean()):.0%} of shots want "
          f"shrinking at all")
    print("  A factor spread narrowly about 1.0 is a second statement of the same result: there is "
          "little for any confidence signal to do, whatever its quality.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
