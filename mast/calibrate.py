#!/usr/bin/env python3
"""Fit the eleven coil gains and the three profile numbers, on the only MAST truth that exists.

    uv run python mast/calibrate.py --rounds 4

`parquet_data/` ships three MAST demo shots, and unlike the 1206 in `mast_public_test` they carry
`efit_psirz` and every EFIT scalar — **115 frames of ground truth**, which is the whole of what
Challenge 2 can be developed against locally. This script is where they are spent.

**What is being fitted, and why it is a calibration rather than a model.** Two things in the solve
are not determined by the dataset:

* the map from a current column to a flux, because MAST ships conductor rows whose count is not the
  coil's electrical turn count (the solenoid ships 656 rows; the fit wants ~250 turns' worth);
* the shape of `p'(psi)` and `FF'(psi)`, which no free-boundary solve without magnetic probes can
  avoid assuming.

Eleven gains and three exponents, against 115 x 4225 pixels. The honest risk is that three shots of
one machine on one day do not represent 1206, and the mitigation is built in: **the gains of the
three coils whose geometry the dataset ships completely (P2, P4, P5) must come out near 1.0**, and
if they do not, the fit is absorbing something that is not a turn count. That is a check the number
of free parameters cannot fake.

**Why it alternates.** A gain fitted against the raw truth launders the plasma's own field into the
coils — the same failure the DIII-D calibration hit, where omitting a plasma filament returned
gains scattered from -0.15 to -0.57 and both signs. So the loop is: solve the equilibrium with the
current gains, subtract the plasma flux the solve produced, refit the gains on what is left, repeat.
The bootstrap round stands in for the solve with a single filament at the shipped axis position.
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mast import predict
from mast import shot as shot_mod
from mast.calibration import CALIBRATION, Calibration
from solver import scalars
from solver.greens import CoilBasis, build_basis, grid_green, vacuum_residual
from solver.gs import Profile, operator, solve
from solver.machine import MAST

FloatArray = npt.NDArray[np.floating]

DEMO = Path(__file__).resolve().parent.parent / "parquet_data"
# The surface `f_per_ka` is read off: the one q95 is defined on, so that the constant stays a
# toroidal field and can be checked against the machine. `scalars.Q_LEVEL` is a different choice.
F_LEVEL = 0.95
# The coils whose conductor geometry the dataset ships in full, and whose gains are therefore the
# check on the whole calibration rather than free parameters.
WELL_POSED = ("magnetics_p2l_current", "magnetics_p2u_current", "magnetics_p4l_current",
              "magnetics_p4u_current", "magnetics_p5l_current", "magnetics_p5u_current")
GRID = {"alpha": (0.5, 1.0, 1.5, 2.0), "gamma": (0.75, 1.0, 1.25),
        "p_scale": (1.5, 2.0, 2.5)}


def psi_stack(cell: Any) -> FloatArray:
    """`efit_psirz` ships as an object array of rows; make it (T, nZ, nR)."""
    return np.stack([np.stack([np.asarray(c, dtype=np.float64) for c in np.asarray(f)])
                     for f in np.asarray(cell)])


def load(paths: list[Path]) -> tuple[CoilBasis, list[dict]]:
    """The basis (one geometry for all MAST) and one dict of arrays per demo shot."""
    basis = None
    shots = []
    for path in paths:
        row = pd.read_parquet(path).iloc[0]
        if basis is None:
            basis = build_basis(row, MAST)
        s = shot_mod.read(row, basis.columns)
        truth = psi_stack(row["efit_psirz"])
        if truth.shape[0] != s.times.size:
            raise ValueError(f"{path.name}: {truth.shape[0]} flux maps against {s.times.size} "
                             f"EFIT frames")
        shots.append({"name": path.stem, "shot": s, "psi": truth,
                      "r_axis": np.asarray(row["efit_r_axis"], dtype=np.float64),
                      "z_axis": np.asarray(row["efit_z_axis"], dtype=np.float64),
                      "q95": np.asarray(row["efit_q95"], dtype=np.float64),
                      "beta_n": np.asarray(row["efit_beta_n"], dtype=np.float64)})
    if basis is None:
        raise SystemExit("no demo shots given")
    return basis, shots


def pooled_r2(truth: list[FloatArray], pred: list[FloatArray]) -> float:
    """The scorer's own reduction: one ratio of sums over every pixel of every frame."""
    t = np.concatenate([x.ravel() for x in truth])
    p = np.concatenate([x.ravel() for x in pred])
    return 1.0 - float(((t - p) ** 2).sum()) / float(((t - t.mean()) ** 2).sum())


def families(columns: list[str]) -> list[list[int]]:
    """Group the current columns by coil family, so P4L and P4U share one gain.

    A turn count is a property of the coil, not of which way its current happens to run, and MAST's
    P-coils are built as up/down pairs: `coil_R`, `coil_width` and `coil_height` are identical
    between the two members of every pair. Tying them halves the free parameters and, more
    usefully, puts a physical constraint into what was an unconstrained fit — the first version
    left them free and returned P6L = 0.45 against P6U = -2.35, which is not two turn counts.
    """
    keys = [c.replace("magnetics_", "").replace("_current", "").rstrip("lu") for c in columns]
    order = list(dict.fromkeys(keys))
    return [[i for i, k in enumerate(keys) if k == fam] for fam in order]


def fit_gains(basis: CoilBasis, shots: list[dict], plasma: list[FloatArray],
              prior: FloatArray, damp: float) -> FloatArray:
    """One gain per coil FAMILY on the truth minus the plasma's flux, damped toward `prior`.

    The damping is not cosmetic. The plasma flux subtracted here came out of a solve that used the
    PREVIOUS gains, so the two are coupled and an undamped alternation diverges — measured, with
    per-column gains: R2_psi went 0.639 -> 0.291 -> -0.428 over three rounds while the well-posed
    gains swung from 0.65 to 1.44 to -0.51.
    """
    groups = families(basis.columns)
    cols = []
    for group in groups:
        block = [sum(s["shot"].currents[:, i][:, None, None] * basis.maps[i] for i in group)
                 for s in shots]
        cols.append(np.concatenate([b.ravel() for b in block]))
    a = np.column_stack(cols)
    y = np.concatenate([(s["psi"] - p).ravel() for s, p in zip(shots, plasma, strict=True)])
    g, *_ = np.linalg.lstsq(a, y, rcond=None)
    old = np.array([prior[group[0]] for group in groups], dtype=np.float64)
    per_family = (1.0 - damp) * old + damp * np.asarray(g, dtype=np.float64)
    out = np.empty(len(basis.columns), dtype=np.float64)
    for value, group in zip(per_family, groups, strict=True):
        out[group] = value
    return out


def bootstrap_plasma(basis: CoilBasis, shots: list[dict]) -> list[FloatArray]:
    """Round zero's stand-in for the solve: one filament at the shipped axis, carrying Ip.

    Crude on purpose — it exists only so the first gain fit is not asked to explain the plasma's
    field with coil currents. Its own amplitude is fitted alongside the gains and then discarded.
    """
    out = []
    for s in shots:
        sh = s["shot"]
        out.append(np.stack([grid_green(np.array([[s["r_axis"][k], s["z_axis"][k]]]),
                                        basis.grid_R, basis.grid_Z) * sh.ip[k]
                             for k in range(sh.times.size)]))
    return out


def solve_all(basis: CoilBasis, shots: list[dict], gains: FloatArray, profile: Profile,
              n_iter: int, relax: float) -> tuple[list[FloatArray], list[FloatArray]]:
    """(total flux, plasma flux) per shot, at these gains and this profile.

    The converged `Solved` objects are stashed on each shot dict, so the value-scalar calibration
    below can read the same equilibria the flux R2 was computed from rather than re-solving.
    """
    op = operator(MAST, basis.grid_R, basis.grid_Z)
    total: list[FloatArray] = []
    plasma: list[FloatArray] = []
    for s in shots:
        sh = s["shot"]
        coil = basis.flux(sh.currents * gains[None, :])
        got = [solve(coil[k], float(sh.ip[k]), op, profile=profile, n_iter=n_iter, relax=relax,
                     thomson=(sh.ts_R, sh.ts_te[k], sh.ts_ne[k], predict.PSI_N_NODES))
               for k in range(sh.times.size)]
        s["solved"] = got
        total.append(np.stack([g.psi for g in got]))
        plasma.append(total[-1] - coil)
    return total, plasma


def fit_q95(basis: CoilBasis, shots: list[dict], f_per_ka: float) -> tuple[float, float, float]:
    """The affine on top of the contour integral. Returns (scale, offset, its R2)."""
    op = operator(MAST, basis.grid_R, basis.grid_Z)
    x, y = [], []
    for s in shots:
        for k, g in enumerate(s["solved"]):
            v = scalars.q95(g, op, float(s["shot"].tf[k]), f_per_ka)
            if np.isfinite(v) and np.isfinite(s["q95"][k]):
                x.append(v)
                y.append(float(s["q95"][k]))
    xa, ya = np.array(x), np.array(y)
    if xa.size == 0:
        raise ValueError("no frame produced a proxy surface; q95 cannot be calibrated")
    a = np.column_stack([xa, np.ones(xa.size)])
    coef, *_ = np.linalg.lstsq(a, ya, rcond=None)
    r2 = 1.0 - float(((ya - a @ coef) ** 2).sum()) / float(((ya - ya.mean()) ** 2).sum())
    return float(coef[0]), float(coef[1]), r2


def fit_f_per_ka(basis: CoilBasis, shots: list[dict],
                 level: float = F_LEVEL) -> tuple[float, float, int]:
    """`F = R B_phi` per kA of TF feed, from the demo shots' own q95. Returns (constant, R2, n).

    Fitted at psi_N = **0.95**, not at the `Q_LEVEL` the shipped q95 uses. The two are different
    questions and conflating them cost a physical constant its meaning: `f_per_ka` is a property of
    the toroidal field coil, so it has to be read off the surface q95 actually NAMES, while
    `Q_LEVEL` is a proxy chosen for correlation and its offset belongs in the affine. Fitting both
    at 0.80 sent `f_per_ka` to 0.0096 T m/kA, which is a toroidal field of 0.96 T on a machine that
    runs at half of that — the number stopped being checkable, and being checkable against an
    independent physical estimate is the whole reason it is fitted through the origin.

    Everything in `q = (F / 2pi) * contour_integral dl / (R^2 |grad psi|)` except `F` is geometry
    the solve already produced, and `F` is a property of the toroidal field coil that the dataset
    does not ship — the turn count again. So one constant, fitted through the origin, and the
    SHAPE of q95 in time is then a prediction rather than a fit: with 115 frames against one free
    number, the R2 printed beside it is what says whether the contour integral is right.
    """
    op = operator(MAST, basis.grid_R, basis.grid_Z)
    x, y = [], []
    for s in shots:
        for k, g in enumerate(s["solved"]):
            shape = scalars.q_shape(g, op, level)
            if np.isfinite(shape) and np.isfinite(s["q95"][k]):
                x.append(abs(float(s["shot"].tf[k])) * shape)
                y.append(float(s["q95"][k]))
    xa, ya = np.array(x), np.array(y)
    if xa.size == 0:
        raise ValueError("no frame produced a 95% flux surface; q95 cannot be calibrated")
    f = float((xa * ya).sum() / (xa * xa).sum())
    r2 = 1.0 - float(((ya - f * xa) ** 2).sum()) / float(((ya - ya.mean()) ** 2).sum())
    return f, r2, xa.size


def fit_beta_n(basis: CoilBasis, shots: list[dict], f_per_ka: float) -> tuple[float, float, float]:
    """The affine calibration of Thomson-derived betaN. Returns (scale, offset, its R2).

    Measured on the demo shots: the raw physical betaN correlates **+0.258** with the shipped one
    and sits 0.703 against 0.981 on average, so its LEVEL is wrong by more than a scale — a
    through-the-origin fit still scores -0.69. The affine reaches +0.066, which is `rho^2`, and its
    slope of 0.31 is the honest statement that most of what it emits is the mean.
    """
    op = operator(MAST, basis.grid_R, basis.grid_Z)
    x, y = [], []
    for s in shots:
        sh = s["shot"]
        for k, g in enumerate(s["solved"]):
            try:
                p = scalars.pressure_profile(g, op, sh.ts_R, sh.ts_te[k], sh.ts_ne[k],
                                             predict.PSI_N_NODES)
                v = scalars.beta_n(g, op, float(sh.tf[k]), f_per_ka, float(sh.ip[k]),
                                   p, predict.PSI_N_NODES)
            except ValueError:
                continue
            if np.isfinite(v) and np.isfinite(s["beta_n"][k]):
                x.append(v)
                y.append(float(s["beta_n"][k]))
    xa, ya = np.array(x), np.array(y)
    if xa.size == 0:
        raise ValueError("no frame produced a pressure profile; betaN cannot be calibrated")
    a = np.column_stack([xa, np.ones(xa.size)])
    coef, *_ = np.linalg.lstsq(a, ya, rcond=None)
    r2 = 1.0 - float(((ya - a @ coef) ** 2).sum()) / float(((ya - ya.mean()) ** 2).sum())
    return float(coef[0]), float(coef[1]), r2


def score_value_scalars(basis: CoilBasis, shots: list[dict], cal: Calibration) -> dict:
    """Pooled R2 of q95 and betaN at this calibration, the way `finalize_machine` pools them."""
    op = operator(MAST, basis.grid_R, basis.grid_Z)
    out = {}
    for name, key in (("q95", "q95"), ("betaN", "beta_n")):
        pred, true = [], []
        for s in shots:
            q, b = predict.value_scalars(s["shot"], op, s["solved"], cal)
            pred.append(q if name == "q95" else b)
            true.append(s[key])
        p, t = np.concatenate(pred), np.concatenate(true)
        out[name] = 1.0 - float(((t - p) ** 2).sum()) / float(((t - t.mean()) ** 2).sum())
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rounds", type=int, default=4, help="gain/solve alternations after bootstrap")
    ap.add_argument("--demo", type=Path, default=DEMO)
    ap.add_argument("--n-iter", type=int, default=CALIBRATION.n_iter)
    ap.add_argument("--relax", type=float, default=CALIBRATION.relax)
    ap.add_argument("--damp", type=float, default=0.4,
                    help="share of each round's gain fit that is adopted; 1.0 diverges")
    ap.add_argument("--profile-sweep", action="store_true",
                    help="re-run the alpha/gamma/beta grid at the current gains (slow)")
    ap.add_argument("--keep-gains", action="store_true",
                    help="start from the gains already in calibration.py instead of refitting")
    args = ap.parse_args()

    paths = sorted(args.demo.glob("mast_shot_*.parquet"))
    if not paths:
        raise SystemExit(f"no mast_shot_*.parquet in {args.demo}")
    basis, shots = load(paths)
    frames = sum(s["shot"].times.size for s in shots)
    rows = "; ".join(f"{n} x {c[10:-8]}"
                     for n, c in zip(basis.turns.astype(int), basis.columns, strict=True))
    print(f"{len(shots)} demo shots, {frames} frames, {len(basis.columns)} current columns over "
          f"{int(basis.turns.sum())} conductor rows ({rows}); closest filament to a grid node "
          f"{basis.min_gap * 1e3:.2f} mm")
    # The Green's function's own self-check, run rather than described: the Grad-Shafranov operator
    # annihilates any vacuum flux, so a formula that does not zero it is wrong however plausible
    # the picture looks. On MAST's grid — 0.030 m in R against 0.062 in Z — this lands at 2.2e-2,
    # which is the five-point stencil's truncation and not the formula.
    print(f"  Delta* annihilation of one filament's field: "
          f"{vacuum_residual(basis.grid_R, basis.grid_Z):.2e} of its peak")

    truth = [s["psi"] for s in shots]
    profile = CALIBRATION.profile
    if args.keep_gains:
        gains: FloatArray = np.array([CALIBRATION.gains[c] for c in basis.columns],
                                     dtype=np.float64)
        print(f"\n  keeping the shipped calibration: {_report(basis, gains)}")
    else:
        gains = fit_gains(basis, shots, bootstrap_plasma(basis, shots),
                          np.ones(len(basis.columns)), 1.0)
        print(f"\n  round 0 (bootstrap filament): {_report(basis, gains)}")

    for rnd in range(1, args.rounds + 1):
        total, plasma = solve_all(basis, shots, gains, profile, args.n_iter, args.relax)
        r2 = pooled_r2(truth, total)
        gains = fit_gains(basis, shots, plasma, gains, args.damp)
        print(f"  round {rnd}: solved R2_psi {r2:+.5f}   {_report(basis, gains)}")

    if args.profile_sweep:
        best = (-np.inf, profile)
        for a, g, ps in itertools.product(GRID["alpha"], GRID["gamma"], GRID["p_scale"]):
            trial = Profile(alpha=a, gamma=g, beta=0.0, p_scale=ps)
            try:
                total, _ = solve_all(basis, shots, gains, trial, args.n_iter, args.relax)
            except ValueError as exc:
                print(f"    alpha={a} gamma={g} p_scale={ps}: failed — {exc}")
                continue
            r2 = pooled_r2(truth, total)
            print(f"    alpha={a} gamma={g} p_scale={ps}: R2_psi {r2:+.5f}")
            if r2 > best[0]:
                best = (r2, trial)
        profile = best[1]
        print(f"  best profile {profile} at R2_psi {best[0]:+.5f}")

    total, _ = solve_all(basis, shots, gains, profile, args.n_iter, args.relax)
    print(f"\n  FINAL R2_psi {pooled_r2(truth, total):+.5f} with {profile}")
    q95 = float(np.concatenate([s["q95"] for s in shots]).mean())
    beta_n = float(np.concatenate([s["beta_n"] for s in shots]).mean())
    print(f"  demo-shot means, the fallback for a frame with no 95% surface: "
          f"q95 {q95:.4f}, betaN {beta_n:.4f}")

    f_per_ka, f_r2, n_frames = fit_f_per_ka(basis, shots)
    b0 = f_per_ka * 85.0 / 0.85
    print(f"  F per kA of TF feed: {f_per_ka:.6f} T m at psi_N = {F_LEVEL}, from {n_frames} "
          f"frames; through the origin it explains R2 {f_r2:+.4f} of q95, and it implies "
          f"B0 = {b0:.3f} T at 85 kA and R0 = 0.85 m (MAST runs at 0.4-0.55)")
    q_scale, q_offset, q_r2 = fit_q95(basis, shots, f_per_ka)
    print(f"  q95 affine: {q_scale:.4f} x + {q_offset:.4f}, in-sample R2 {q_r2:+.4f}")
    b_scale, b_offset, b_r2 = fit_beta_n(basis, shots, f_per_ka)
    print(f"  betaN affine: {b_scale:.4f} x + {b_offset:.4f}, in-sample R2 {b_r2:+.4f}")
    cal = Calibration(gains=dict(zip(basis.columns, gains, strict=True)), profile=profile,
                      q95_const=q95, beta_n_const=beta_n, f_per_ka=f_per_ka,
                      q95_scale=q_scale, q95_offset=q_offset,
                      beta_n_scale=b_scale, beta_n_offset=b_offset,
                      n_iter=args.n_iter, relax=args.relax)
    got = score_value_scalars(basis, shots, cal)
    print(f"  pooled R2 as the scorer reads them: q95 {got['q95']:+.4f}, "
          f"betaN {got['betaN']:+.4f}, mean {0.5 * (got['q95'] + got['betaN']):+.4f}")
    print("\n  paste into mast/calibration.py:\n")
    print("    gains={")
    for c, v in zip(basis.columns, gains, strict=True):
        print(f'        "{c}": {v:.6f},')
    print("    },")
    print(f"    profile=Profile(alpha={profile.alpha}, gamma={profile.gamma}, "
          f"beta={profile.beta}, p_scale={profile.p_scale}),")
    print(f"    q95_const={q95:.4f},\n    beta_n_const={beta_n:.4f},"
          f"\n    f_per_ka={f_per_ka:.6f},"
          f"\n    q95_scale={q_scale:.6f},\n    q95_offset={q_offset:.6f},"
          f"\n    beta_n_scale={b_scale:.6f},\n    beta_n_offset={b_offset:.6f},")
    return 0


def _report(basis: CoilBasis, gains: FloatArray) -> str:
    """The check that matters: what the fully-shipped coils came out at."""
    idx = [basis.columns.index(c) for c in WELL_POSED if c in basis.columns]
    vals = gains[idx]
    return (f"P2/P4/P5 gains {np.array2string(vals, precision=2)} "
            f"(want ~1.0), solenoid "
            f"{gains[basis.columns.index('magnetics_sol_current')]:.1f}")


if __name__ == "__main__":
    raise SystemExit(main())
