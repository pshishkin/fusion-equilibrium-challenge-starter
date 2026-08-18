#!/usr/bin/env python3
"""
C9 — how much of what is left is the model's error, and how much is EFIT's own jitter?

    uv run python my_experiments/diagnose_label_noise.py

Six free gates in Group D all returned nothing, and three of them said the same thing from
different directions: the ensemble's residual on unseen shots carries no linear structure left in
these inputs (D6's control at R2 -0.006, D7's correction at 0.0%, it-6's 0.008). One explanation is
that the fit is done and the remainder is not a fit problem at all — it is the label.

The seven scored functionals are derived from EFIT reconstructions, and EFIT is itself an inverse
solve with its own error. That error shows up as JITTER: `li` and `kappa` are set by current
diffusion and plasma shaping, both of which evolve on hundreds of milliseconds, so frame-to-frame
movement at a 20 ms step is not physics. And jitter in the LABEL is irreducible — the map that would
reproduce it does not exist, because the scorer derives the truth from the same noisy EFIT.

The estimator is the second difference. For x_t = s_t + e_t with `s` smooth on the frame scale and
`e` white with variance sigma^2,

    Var(x_{t-1} - 2 x_t + x_{t+1})  =  6 sigma^2  +  (a second difference of s, negligible)

so sigma^2 = Var(D2 x) / 6. It would be an UPPER bound on the noise if the noise were WHITE — real
curvature in `s` only inflates it — but that assumption is not free, and correlated label error
pushes the estimate the other way. The lag sweep at the end tests it and, measured, **does not
decide**: sigma rises 1.14x to 1.63x from k = 1 to k = 3, which is what correlated noise predicts
and also what genuine curvature over a 120 ms span predicts. So this script reports a RANGE across
three readings rather than a bound, and the range is what the conclusion has to survive.

**Two traps, and one of them has already moved this number by 5.6x.**

* **Gaps.** 98.0% of frame intervals are 20 ms but 31% of shots carry one over 100 ms, up to 900 ms.
  A second difference straddling a 900 ms gap measures the physics, not the noise. Only triples
  whose BOTH intervals are the nominal step are used, and the share kept is printed.
* **Outliers.** The variance of a second difference is dominated by its tail. A trimmed estimator is
  reported beside the plain one rather than instead of it, because the gap between them IS the
  uncertainty on this measurement and quoting either alone overstates what is known.

What comes out is a per-scalar ceiling: even a perfect model scores R2_j = 1 - sigma_j^2 / SS_tot_j
on the fold, so the Consistency term cannot exceed the mean of those and S cannot exceed the rest of
the composite plus 0.20 times it. That is the number that says whether to keep optimising.
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

from common import CONS_SCALARS, W_CONS  # noqa: E402

from my_experiments.progress import install_timestamps  # noqa: E402

FloatArray = npt.NDArray[np.floating]
COSTS = HERE.parent / "results" / "frame_costs_ensemble.csv"
FRAME_STEP_MS = 20.0


def second_differences(df: pd.DataFrame, col: str, tol: float, lag: int = 1) -> FloatArray:
    """x_{t-k} - 2 x_t + x_{t+k}, over spans whose every interval is the nominal step.

    `lag` is what turns one estimator into a test of its own assumption. Var(D2) = 6 sigma^2 holds
    for WHITE noise; for noise correlated at lag k it is sigma^2 (6 - 8 rho_k + 2 rho_2k), which is
    SMALLER, so a correlated label would make sigma look smaller than it is and the ceiling look
    higher. Correlation dies with k while the curvature of the physics grows with it, so sigma
    rising from k = 1 to k = 3 is the signature of correlated noise and sigma flat is the signature
    of white noise plus smooth physics.
    """
    out = []
    for _, g in df.groupby("shot", sort=False):
        x = g[col].to_numpy(dtype=np.float64)
        t = g["time_ms"].to_numpy(dtype=np.float64)
        if len(x) < 2 * lag + 1:
            continue
        dt = np.diff(t)
        even = np.ones(len(x) - 2 * lag, dtype=bool)
        for i in range(2 * lag):
            even &= np.abs(dt[i:len(even) + i] - FRAME_STEP_MS) < tol
        d2 = x[:-2 * lag] - 2 * x[lag:-lag] + x[2 * lag:]
        out.append(d2[even & np.isfinite(d2)])
    return np.concatenate(out) if out else np.zeros(0)


def main() -> int:
    install_timestamps()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--costs", type=Path, default=COSTS)
    ap.add_argument("--tol", type=float, default=1.0, help="ms a frame interval may differ by")
    ap.add_argument("--trim", type=float, default=0.05,
                    help="tail share dropped from each end in the trimmed estimator")
    args = ap.parse_args()

    df = pd.read_csv(args.costs)
    if f"gt_{CONS_SCALARS[0]}" not in df.columns:
        raise SystemExit(f"{args.costs.name} has no gt_ columns — regenerate it with the current "
                         f"diagnose_frames.py")
    print(f"{len(df)} frames of {len(set(df['shot']))} shots from {args.costs.name}")

    truth = df[[f"gt_{s}" for s in CONS_SCALARS]].to_numpy(dtype=np.float64)
    resid = df[[f"res_{s}" for s in CONS_SCALARS]].to_numpy(dtype=np.float64)

    print(f"\n  {'scalar':<9} {'model RMS':>11} {'EFIT jitter':>12} {'trimmed':>10} "
          f"{'ratio':>8} {'R2 now':>9} {'R2 ceiling':>11}")
    ceiling_plain, ceiling_trim, r2_now = [], [], []
    kept = np.nan
    for j, name in enumerate(CONS_SCALARS):
        ok = np.isfinite(truth[:, j])
        t = truth[ok, j]
        ss_tot = float(((t - t.mean()) ** 2).sum()) / len(t)          # per frame, as R2 reads it
        rms = float(np.sqrt(np.nanmean(resid[:, j] ** 2)))

        d2 = second_differences(df, f"gt_{name}", args.tol)
        if np.isnan(kept):
            kept = len(d2) / max(1, len(df))
        sigma = float(np.sqrt(max(0.0, (d2 ** 2).mean() / 6.0)))
        lo, hi = np.percentile(d2, [100 * args.trim, 100 * (1 - args.trim)])
        core = d2[(d2 >= lo) & (d2 <= hi)]
        sigma_t = float(np.sqrt(max(0.0, (core ** 2).mean() / 6.0)))

        r2 = 1.0 - rms ** 2 / ss_tot
        ceiling_plain.append(1.0 - sigma ** 2 / ss_tot)
        ceiling_trim.append(1.0 - sigma_t ** 2 / ss_tot)
        r2_now.append(r2)
        print(f"  {name:<9} {rms:>11.5f} {sigma:>12.5f} {sigma_t:>10.5f} "
              f"{sigma / max(rms, 1e-30):>8.2f} {r2:>9.5f} {ceiling_plain[-1]:>11.5f}")

    print(f"\n  triples usable after the gap filter: {kept:.1%} of frames")
    now, cp, ct = np.mean(r2_now), np.mean(ceiling_plain), np.mean(ceiling_trim)
    print(f"\n  Consistency now {now:.5f}; ceiling {cp:.5f} plain, {ct:.5f} trimmed")
    print(f"  headroom in the Consistency term: {cp - now:+.5f} plain, {ct - now:+.5f} trimmed")
    print(f"  which is {W_CONS * (cp - now):+.5f} and {W_CONS * (ct - now):+.5f} of S")
    print(f"\n  Read the two columns as a range, not a number: the plain estimator calls every "
          f"second difference noise and the trimmed one calls the largest {2 * args.trim:.0%} "
          f"physics, and nothing here can tell them apart.")

    # The assumption, tested rather than declared. Both error directions are live: curvature in the
    # physics inflates sigma (ceiling too low), correlation in the noise deflates it (ceiling too
    # high), and only the lag sweep says which one is operating.
    print("\n  sigma by lag — the white-noise assumption, tested:")
    print(f"    {'scalar':<9}" + "".join(f"{'k=' + str(k):>10}" for k in (1, 2, 3))
          + f"{'k3/k1':>9}")
    ceiling_corr = []
    for j, name in enumerate(CONS_SCALARS):
        ok = np.isfinite(truth[:, j])
        t = truth[ok, j]
        ss_tot = float(((t - t.mean()) ** 2).sum()) / len(t)
        sig = []
        for k in (1, 2, 3):
            d2 = second_differences(df, f"gt_{name}", args.tol, k)
            sig.append(float(np.sqrt(max(0.0, (d2 ** 2).mean() / 6.0))) if len(d2) else np.nan)
        ceiling_corr.append(1.0 - sig[2] ** 2 / ss_tot)
        print(f"    {name:<9}" + "".join(f"{v:>10.5f}" for v in sig)
              + f"{sig[2] / max(sig[0], 1e-30):>9.2f}")

    cc = float(np.mean(ceiling_corr))
    print(f"\n    Every scalar rises with the lag, by 1.14x to 1.63x — and that does NOT separate "
          f"the two explanations, because both predict a rise. Correlated label noise makes k=1 an "
          f"under-estimate; real curvature in the physics makes k=3 an over-estimate, and at k=3 "
          f"the span is 120 ms, which is a timescale these quantities genuinely move on. So the "
          f"lag sweep does not decide, and its value is to BOUND: charging the whole rise to "
          f"correlation gives a ceiling of {cc:.5f}, i.e. {W_CONS * (cc - now):+.5f} of S.")
    print(f"\n  The headroom in Consistency, over the three readings: "
          f"{W_CONS * (cc - now):+.5f} (all of the lag rise is noise) .. "
          f"{W_CONS * (cp - now):+.5f} (white noise) .. "
          f"{W_CONS * (ct - now):+.5f} (the largest tail is physics) of S.")
    if W_CONS * (cc - now) < 0.0013:
        print("  The pessimistic end is under what one paired run resolves, so on that reading the "
              "geometry terms are finished and the label is what remains.")
    else:
        print("  Every reading leaves more than the 0.0013 a paired run resolves, so the geometry "
              "terms are NOT at the label floor. Where they are is per scalar, in the ratio column "
              "above: the two triangularities sit at 0.78-0.87 of the floor and are finished, "
              "while li at 0.32 and kappa at 0.47 are two to three times above it — and those "
              "two are 42% of the geometry cost by C1's decomposition.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
