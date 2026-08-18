#!/usr/bin/env python3
"""
D6 — is Thomson summarised in the wrong coordinate?

    uv run python my_experiments/diagnose_thomson_flux.py --jobs 20

The core Thomson system sits at a fixed R = 1.940 and varies only in Z, so its thirteen summaries
describe the profile along a line fixed in the MACHINE. But a tokamak profile is a function of the
flux surface, not of the machine: as the axis moves — and R_axis and Z_axis are scored precisely
because they move — a fixed channel samples a different surface every frame, and the summaries mix
profile shape with plasma position. The physics says the right axis is the normalised flux
psi_N = (psi - psi_axis) / (psi_boundary - psi_axis).

**This is an ORACLE and cannot be submitted.** psi_N here is computed from the TRUE `efit_psirz`,
which does not exist at inference; a real version would need a two-stage model, an out-of-fold
first-stage prediction over 1.19M rows, and a decode plus o-point plus contour per frame at
inference. The oracle exists to price that before any of it is written: if the truth-coordinate
summaries do not explain the residual the machine-coordinate ones leave, then no attainable version
of them will.

It is bounded above before it starts. The whole Thomson block is worth **0.0023** of S (measured
08-17, the block removed and the model refitted), so re-summarising it cannot be worth more than
that and will realise a fraction. What the measurement can do cheaply is CLOSE the family.

Pre-registered kill: **build nothing unless the flux-coordinate summaries explain 10% more of
`res_li`'s variance than the machine-coordinate ones do**, out of sample. li is the target because
it is 18.7% of the geometry cost and is a profile quantity; R_axis is reported beside it because it
is what the coordinate change is supposed to remove.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from sklearn.linear_model import Ridge

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "fusion_scoring"))

from common import AXIS_SIGN  # noqa: E402
from o_point import find_o_point  # noqa: E402

import local_score  # noqa: E402
from experiments import DEFAULT_LOCAL_DATA_DIR, HF_TRAIN_CONFIG, _as_psirz_stack  # noqa: E402
from my_experiments.baseline_model import (  # noqa: E402
    ARTIFACT,
    N_THOMSON_BLOCK,
    _read_shots,
    build_inputs,
    sorted_shots,
    take_share,
)
from my_experiments.parallel import pimap, resolve_jobs  # noqa: E402
from my_experiments.progress import install_timestamps  # noqa: E402
from my_experiments.target_metric import scorer_context  # noqa: E402

FloatArray = npt.NDArray[np.floating]
COSTS = HERE.parent / "results" / "frame_costs_ensemble.csv"
N_STATS = 13


def _stats(values: FloatArray, dens: FloatArray, coord: FloatArray) -> FloatArray:
    """(n_times, 13) profile summaries with a coordinate that varies PER FRAME.

    The same thirteen numbers `_profile_stats` produces, and in the same order, but `coord` is
    (n_times, n_channels) rather than one fixed axis — which is the whole point: in flux
    coordinates a channel's position moves with the equilibrium.
    """
    live = values != 0
    n = live.sum(axis=1)
    enough = n >= 3
    span = np.maximum(coord.max(axis=1) - coord.min(axis=1), 1e-9)
    press = values * dens

    def masked(a: FloatArray) -> FloatArray:
        return np.where(live, a, np.nan)

    def slope(a: FloatArray) -> FloatArray:
        """Least-squares gradient of `a` against the per-frame coordinate, live channels only."""
        w = live.astype(np.float64)
        cnt = np.maximum(w.sum(axis=1), 1.0)
        xm = (w * coord).sum(axis=1) / cnt
        ym = (w * a).sum(axis=1) / cnt
        dx = np.where(live, coord - xm[:, None], 0.0)
        dy = np.where(live, a - ym[:, None], 0.0)
        den = (dx * dx).sum(axis=1)
        return np.where(den > 0, (dx * dy).sum(axis=1) / np.where(den > 0, den, 1.0), np.nan)

    with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
        warnings.filterwarnings("ignore", ".*empty slice.*", RuntimeWarning)
        warnings.filterwarnings("ignore", "All-NaN slice encountered", RuntimeWarning)
        te, ne_, pr = masked(values), masked(dens), masked(press)
        peak_te, peak_ne, peak_p = (np.nanmax(v, axis=1) for v in (te, ne_, pr))
        rows = np.arange(len(coord))
        at_te = coord[rows, np.nanargmax(np.where(np.isnan(te), -np.inf, te), axis=1)]
        at_p = coord[rows, np.nanargmax(np.where(np.isnan(pr), -np.inf, pr), axis=1)]
        mean_te, mean_ne, mean_p = (np.nanmean(v, axis=1) for v in (te, ne_, pr))
        peaking = peak_te / np.where(mean_te > 0, mean_te, np.nan)
        w = np.where(live & (values > 0), values, 0.0)
        wsum = np.maximum(w.sum(axis=1), 1e-30)
        cbar = (w * coord).sum(axis=1) / wsum
        width = np.sqrt(np.maximum((w * (coord - cbar[:, None]) ** 2).sum(axis=1) / wsum, 0.0))
        out = np.column_stack([peak_te, at_te, mean_te * span, slope(values),
                               peak_ne, mean_ne * span, slope(dens),
                               peak_p, at_p, mean_p * span, slope(press),
                               peaking, width])
    out[~enough] = np.nan
    return out


def _task(args: tuple) -> FloatArray:
    """(T, 27) flux-coordinate Thomson summaries for one shot, on its own `efit_times`."""
    path, ctx = args
    row = pd.read_parquet(path).iloc[0]
    psi = _as_psirz_stack(row["efit_psirz"]).astype(np.float64)
    R, Z = ctx["grid_R"], ctx["grid_Z"]
    t_efit = np.asarray(row["efit_times"], dtype=np.float64).ravel()

    chord_R = np.asarray(row["thomson_chord_R"], dtype=np.float64)
    chord_Z = np.asarray(row["thomson_chord_Z"], dtype=np.float64)
    out = np.full((len(psi), 2 * N_STATS + 1), np.nan)

    # psi_N per frame at every chord position. psi_axis from the o-point the scorer's own routine
    # finds; psi_boundary from the SHIPPED LCFS polygon, averaged over its vertices — using the
    # shipped contour rather than re-extracting keeps this a statement about the coordinate and not
    # about the extractor.
    n_lcfs = np.asarray(row["efit_lcfs_n"])
    lcfs_r = np.asarray(row["efit_lcfs_r"], dtype=object)
    lcfs_z = np.asarray(row["efit_lcfs_z"], dtype=object)
    psi_n_all = np.full((len(psi), len(chord_R)), np.nan)
    for k in range(len(psi)):
        interp = RegularGridInterpolator((Z, R), psi[k], bounds_error=False, fill_value=np.nan)
        # `find_o_point` wants the axis at a MAXIMUM, and DIII-D ships psi with AXIS_SIGN = -1, so
        # it is handed the signed map. psi_N itself is invariant to that flip — it divides one
        # difference of psi by another — so everything after this line uses the raw values.
        o = find_o_point(AXIS_SIGN[local_score.MACHINE] * psi[k], R, Z, ctx["mask_coarse"])
        m = int(n_lcfs[k]) if np.isfinite(n_lcfs[k]) else 0
        if o is None or m < 3:
            continue
        # o is (R_axis, Z_axis, iz, ir, used_local): the third element is a grid INDEX, not the
        # flux there. Interpolate rather than index, so the sub-pixel axis is used as found.
        psi_axis = float(interp([[o[1], o[0]]])[0])
        bdry = interp(np.column_stack([np.asarray(lcfs_z[k], dtype=np.float64)[:m],
                                       np.asarray(lcfs_r[k], dtype=np.float64)[:m]]))
        psi_b = float(np.nanmean(bdry))
        den = psi_b - psi_axis
        if not np.isfinite(den) or den == 0:
            continue
        psi_n_all[k] = (interp(np.column_stack([chord_Z, chord_R])) - psi_axis) / den

    blocks = []
    for prefix, key in (("core", "thomson_core"), ("edge", "thomson_edge")):
        te = np.asarray([np.asarray(a, dtype=np.float64) for a in row[f"{key}_Te"]])
        ne = np.asarray([np.asarray(a, dtype=np.float64) for a in row[f"{key}_ne"]])
        times = np.asarray(row[f"{key}_times"], dtype=np.float64).ravel()
        start = 0 if prefix == "core" else len(np.asarray(row["thomson_core_Te"][0]))
        cols = slice(start, start + te.shape[1])
        # The channels on the EFIT clock, so the profile and the flux surface it sits on are read
        # at the same instant. Production interpolates the SUMMARIES instead; both arms here use
        # the same interpolation, so the contrast is the coordinate and nothing else.
        te_e = np.column_stack([np.interp(t_efit, times, te[:, i]) for i in range(te.shape[1])])
        ne_e = np.column_stack([np.interp(t_efit, times, ne[:, i]) for i in range(ne.shape[1])])
        coord = psi_n_all[:, cols]
        order = np.argsort(np.nan_to_num(coord, nan=1e9), axis=1)
        rows_ix = np.arange(len(coord))[:, None]
        blocks.append(_stats(te_e[rows_ix, order], ne_e[rows_ix, order],
                             np.nan_to_num(coord[rows_ix, order], nan=1.5)))
    out[:, :2 * N_STATS] = np.hstack(blocks)
    out[:, -1] = np.isfinite(psi_n_all).any(axis=1).astype(np.float64)
    return out


ALPHAS = (1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6)


def explained(X: FloatArray, y: FloatArray, A: npt.NDArray[np.bool_],
              B: npt.NDArray[np.bool_], groups: npt.NDArray[np.str_]) -> tuple[float, float]:
    """Out-of-sample R2 on `B` of a ridge fitted on `A`, with alpha chosen inside `A`.

    Two things this has to get right, both learned by getting them wrong first.

    **Alpha is selected, not assumed.** At a fixed alpha = 1 every arm came out at R2 between
    -0.49 and -1.14 — far worse than predicting the mean, on a design ridge itself flagged at
    rcond 1e-51 — and increments swung from -0.35 to +0.38, which is noise wearing a number's
    clothes. The selection is by a SHOT-grouped split inside A, because frames of one shot are not
    independent: it-5c measured 45.9% of the residual as a per-shot constant, so a split that puts
    a shot on both sides of the fold picks an alpha for a problem nobody is being asked.

    **Both blocks are standardised on A.** The control arrives already scaled by the artifact's own
    `StandardScaler` and the flux summaries do not, so an unstandardised treatment block is
    penalised on a different scale from the columns it is competing with — the arm would then lose
    or win on units rather than on information.
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
    return (1.0 - float(((y[ok_b] - est) ** 2).sum()) / tot if tot > 0 else np.nan), best_alpha


def main() -> int:
    install_timestamps()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--share", type=float, default=0.01)
    ap.add_argument("--alpha", type=float, default=1.0)
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
    print(f"{len(files)} held-out shots, {len(df)} frames, artifact {args.artifact.name}")

    mask = np.load(HERE.parent / "fusion_scoring" / "masks" / "d3d_envelope.npz")
    ctx = scorer_context(mask["grid_R"], mask["grid_Z"], local_score.MACHINE)
    jobs = resolve_jobs(args.jobs, len(files))
    from tqdm import tqdm
    parts = list(tqdm(pimap(_task, [(p, ctx) for p in files], jobs), total=len(files),
                      unit="shot", desc="  flux coords"))
    flux = np.concatenate(parts)
    good = float(np.isfinite(flux[:, :2 * N_STATS]).all(axis=1).mean())
    print(f"  flux-coordinate summaries defined on {good:.1%} of frames")
    flux = np.nan_to_num(flux, nan=0.0)

    feats, _psi, _s, _n = _read_shots(files, "features", 1.0, args.jobs)
    X = np.asarray(art["scaler"].transform(build_inputs(art["coil"], feats)), dtype=np.float64)
    if len(X) != len(df):
        raise ValueError(f"{len(X)} feature rows against {len(df)} rows of {args.costs.name}")
    # The control is the model's OWN inputs, Thomson block included: the question is what the flux
    # coordinate adds over the machine coordinate, not what Thomson adds over nothing.
    print(f"  control: {X.shape[1]} columns, of which {N_THOMSON_BLOCK} are Thomson in machine "
          f"coordinates; treatment adds {flux.shape[1]} in flux coordinates")

    shots = df["shot"].to_numpy()
    order = list(dict.fromkeys(shots))
    A = np.isin(shots, order[:len(order) // 2])
    B = ~A
    Xt = np.hstack([X, flux])
    print(f"\n  out-of-sample R2 of the ensemble's own residual, fitted on {int(A.sum())} frames "
          f"and read on {int(B.sum())}, alpha chosen by a shot-grouped split inside the fit half:")
    print(f"    {'residual':<10} {'control':>10} {'+ flux':>10} {'increment':>12}   alphas")
    verdict = 0.0
    for name in ("li", "R_axis", "kappa", "volume"):
        y = df[f"res_{name}"].to_numpy(dtype=np.float64)
        c, ac = explained(X, y, A, B, shots)
        t, at = explained(Xt, y, A, B, shots)
        print(f"    res_{name:<6} {c:>10.4f} {t:>10.4f} {t - c:>+12.4f}   {ac:.0e} / {at:.0e}")
        if name == "li":
            verdict = t - c

    print(f"\n  the pre-registered gate is +0.10 on res_li, and this is {verdict:+.4f}.")
    if verdict < 0.10:
        print("  Refuted. The flux coordinate is the right one physically and the machine "
              "coordinate is already carrying what it would carry — measured with the TRUE psi, "
              "so no attainable two-stage version can do better. Close the family.")
    else:
        print("  It clears. Price the build honestly before writing it: out-of-fold stage-1 "
              "predictions over 1.19M rows, and a decode -> o-point -> contour -> renormalise -> "
              "27 stats -> second forward pass on every inference frame.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
