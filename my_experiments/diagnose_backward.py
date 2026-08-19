#!/usr/bin/env python3
"""
D9's remaining half — does the FUTURE of a shot say anything the past does not?

    uv run python my_experiments/diagnose_backward.py --jobs 20

Every column in `features_for_row` is instantaneous, centred, or strictly causal: `_vessel_currents`
runs exactly one FORWARD `lfilter`, and after D7 there is no clock column either. Yet the scored
shot is complete and offline — the whole discharge is on disk at inference — and the sequence model
exploits that: it is bidirectional, and C6 measured its BACKWARD channels as its long ones, median
tau 122.0 frames against 90.7 forward, with influence correlating +0.254 backward against +0.045.

So the untested question is whether an anti-causal feature carries something. The construction is
the accepted vessel bank run the other way round: the same leaky integral of dI/dt at the same four
timescales, on the time-REVERSED signal. Where the forward bank says "what the coils have been
doing", the backward one says "what they are about to do" — and a plasma whose current is about to
be ramped down is in a different state now than one that is not.

Narrow, deliberately: eight columns, ECOILA and plasma_current at the four taus. A19 settled that
axis — its narrow arm moved all four seeds and its broad one (80 columns) was flat, and `frame_gaps`
cost 0.0012 for two nearly-constant columns.

**Two arms, because D7 taught what one arm is worth here.** A19 arm 1 moved `ridge` by +0.0216, the
largest move any change has made to the linear baseline in this fork, and the MLP did not move at
all; then D7 reproduced exactly that pattern for the clock. So this reports both:

  * against the TRUTH — can a linear model predict li and kappa better when told the future? This
    has power, because li is predictable; it is the mechanism test.
  * against the ENSEMBLE's own residual on held-out shots — is there anything left for it to fix?
    This is the one that decides, because it is what ships.

Pre-registered kill, from D9: **+0.02 of held-out R2 on the RESIDUAL arm, for li and kappa both.**
The truth arm is reported for the mechanism and cannot on its own license a feature.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.linear_model import Ridge

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "fusion_scoring"))

from common import CONS_SCALARS  # noqa: E402

from experiments import DEFAULT_LOCAL_DATA_DIR, HF_TRAIN_CONFIG  # noqa: E402
from my_experiments.baseline_model import (  # noqa: E402
    ARTIFACT,
    RAW_DERIV_HALF_MS,
    VESSEL_TAUS_MS,
    _read_shots,
    build_inputs,
    inputs_only_shot,
    sorted_shots,
    take_share,
)
from toolkit.parallel import pimap, resolve_jobs  # noqa: E402
from toolkit.progress import install_timestamps  # noqa: E402

FloatArray = npt.NDArray[np.floating]
COSTS = HERE.parent / "results" / "frame_costs_ensemble.csv"
# The two signals A19's narrow arm settled on: the solenoid, which C3 put at 0.4432 of permutation
# reliance, and the plasma current itself.
BACKWARD_SIGNALS = ("ECOILA", "plasma_current")
ALPHAS = (1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6)


def _task(path: Path) -> FloatArray:
    """(T, 8) the vessel bank run BACKWARD, on this shot's own sampling base."""
    from scipy.signal import lfilter, lfilter_zi

    row = pd.read_parquet(path).iloc[0]
    shot = inputs_only_shot(row)
    efit_times = shot["efit_times"]
    out = []
    for sig in BACKWARD_SIGNALS:
        if sig not in shot["magnetics"]:
            out.append(np.zeros((len(efit_times), len(VESSEL_TAUS_MS))))
            continue
        mag = shot["magnetics"][sig]
        t = np.asarray(mag["times"], dtype=np.float64)
        v = np.asarray(mag["values"], dtype=np.float64)
        step = float(np.median(np.diff(t)))
        k = max(1, round(RAW_DERIV_HALF_MS / step))
        lo = np.clip(np.arange(len(t)) - k, 0, len(t) - 1)
        hi = np.clip(np.arange(len(t)) + k, 0, len(t) - 1)
        dt = t[hi] - t[lo]
        x = np.where(dt > 0, (v[hi] - v[lo]) / np.maximum(dt, 1e-30), 0.0)
        # The only change from `_vessel_currents`: the signal is reversed before the recursion and
        # the result reversed back, so sample i integrates what happens AFTER it. Everything else —
        # the exact-solution coefficient, the steady-state initialisation, the interpolation onto
        # the EFIT frames — is the accepted construction untouched, so the contrast is direction
        # and nothing else.
        xr = x[::-1]
        cols = []
        for tau in VESSEL_TAUS_MS:
            a = float(np.exp(-step / tau))
            b, denom = np.array([1.0 - a]), np.array([1.0, -a])
            y, _ = lfilter(b, denom, xr, zi=lfilter_zi(b, denom) * xr[0])
            cols.append(np.interp(efit_times, t, y[::-1]))
        out.append(np.column_stack(cols))
    return np.hstack(out)


def explained(X: FloatArray, y: FloatArray, A: npt.NDArray[np.bool_],
              B: npt.NDArray[np.bool_], groups: npt.NDArray) -> float:
    """Out-of-sample R2 on `B` of a ridge fitted on `A`, standardised and alpha chosen inside `A`.

    Both are load-bearing and both were learned in D6, where a fixed alpha on an unstandardised
    design produced control R2 of -0.49 to -1.14 and increments that swung by 0.7.
    """
    ok_a, ok_b = A & np.isfinite(y), B & np.isfinite(y)
    mu, sd = X[ok_a].mean(axis=0), X[ok_a].std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    Xa, Xb = (X[ok_a] - mu) / sd, (X[ok_b] - mu) / sd
    inner_shots = list(dict.fromkeys(groups[ok_a]))
    inner = np.isin(groups[ok_a], inner_shots[:len(inner_shots) // 2])

    best, best_alpha = -np.inf, ALPHAS[0]
    for alpha in ALPHAS:
        est = Ridge(alpha=alpha).fit(Xa[inner], y[ok_a][inner]).predict(Xa[~inner])
        held = y[ok_a][~inner]
        tot = float(((held - held.mean()) ** 2).sum())
        got = 1.0 - float(((held - est) ** 2).sum()) / tot if tot > 0 else -np.inf
        if got > best:
            best, best_alpha = got, alpha

    est = Ridge(alpha=best_alpha).fit(Xa, y[ok_a]).predict(Xb)
    tot = float(((y[ok_b] - y[ok_b].mean()) ** 2).sum())
    return 1.0 - float(((y[ok_b] - est) ** 2).sum()) / tot if tot > 0 else np.nan


def main() -> int:
    install_timestamps()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--share", type=float, default=0.01)
    ap.add_argument("--jobs", type=int, default=20)
    ap.add_argument("--costs", type=Path, default=COSTS)
    ap.add_argument("--artifact", type=Path, default=ARTIFACT)
    ap.add_argument("--local-data-dir", type=Path, default=DEFAULT_LOCAL_DATA_DIR)
    ap.add_argument("--config", default=HF_TRAIN_CONFIG)
    args = ap.parse_args()

    art = joblib.load(args.artifact)
    files = take_share(sorted_shots(args.local_data_dir, args.config, int(art["split_salt"])),
                       args.share, "tail")
    df = pd.read_csv(args.costs)
    if list(dict.fromkeys(df["shot"])) != [p.stem for p in files]:
        raise SystemExit(f"{args.costs.name} covers different shots than --share {args.share}")
    print(f"{len(files)} held-out shots, {len(df)} frames; backward bank on "
          f"{', '.join(BACKWARD_SIGNALS)} at {VESSEL_TAUS_MS} ms")

    jobs = resolve_jobs(args.jobs, len(files))
    from tqdm import tqdm
    back = np.concatenate(list(tqdm(pimap(_task, files, jobs), total=len(files),
                                    unit="shot", desc="  backward")))

    feats, _psi, _s, _n = _read_shots(files, "features", 1.0, args.jobs)
    X = np.asarray(art["scaler"].transform(build_inputs(art["coil"], feats)), dtype=np.float64)
    if len(X) != len(df) or len(back) != len(df):
        raise ValueError(f"{len(X)} features and {len(back)} backward rows against {len(df)} costs")
    Xt = np.hstack([X, back])
    print(f"  control {X.shape[1]} columns (forward vessel bank included), "
          f"treatment adds {back.shape[1]} backward")

    shots = df["shot"].to_numpy()
    order = list(dict.fromkeys(shots))
    A = np.isin(shots, order[:len(order) // 2])
    B = ~A

    truth = {s: df[f"gt_{s}"].to_numpy(dtype=np.float64) for s in CONS_SCALARS}
    resid = {s: df[f"res_{s}"].to_numpy(dtype=np.float64) for s in CONS_SCALARS}
    print(f"\n  MECHANISM — predicting the TRUTH, fitted on {int(A.sum())} frames, "
          f"read on {int(B.sum())}:")
    print(f"    {'scalar':<9} {'control':>10} {'+ backward':>12} {'increment':>12}")
    for s in ("li", "kappa", "R_axis", "volume"):
        c = explained(X, truth[s], A, B, shots)
        t = explained(Xt, truth[s], A, B, shots)
        print(f"    {s:<9} {c:>10.4f} {t:>12.4f} {t - c:>+12.4f}")

    print("\n  DECIDES — predicting the ENSEMBLE's residual on the same held-out shots:")
    print(f"    {'scalar':<9} {'control':>10} {'+ backward':>12} {'increment':>12}")
    verdict = {}
    for s in ("li", "kappa", "R_axis", "volume"):
        c = explained(X, resid[s], A, B, shots)
        t = explained(Xt, resid[s], A, B, shots)
        verdict[s] = t - c
        print(f"    res_{s:<6} {c:>10.4f} {t:>12.4f} {t - c:>+12.4f}")

    print(f"\n  the gate is +0.02 on res_li AND res_kappa; measured {verdict['li']:+.4f} "
          f"and {verdict['kappa']:+.4f}.")
    if verdict["li"] < 0.02 or verdict["kappa"] < 0.02:
        print("  Refuted. The anti-causal construction is the last untested mechanism in the file, "
              "and the ensemble carries no residual it can reach. Note what the mechanism arm says "
              "separately: a difference there with none here is A19's pattern for the third time, "
              "and it means the fitted model already holds what the linear one is missing.")
    else:
        print("  It clears. Build the NARROW arm only, as a replacement for one of the three "
              "feature sets — never a fourth member, which measured +0.0001 — and read the paired "
              "delta on salt 0 plus the last decile in diagnose_frames.py before spending a "
              "confirmation salt. The cache key includes VESSEL_TAUS_MS, so the rebuild is forced "
              "and it invalidates every run in flight.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
