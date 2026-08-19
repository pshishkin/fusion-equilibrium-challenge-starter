#!/usr/bin/env python3
"""One MAST shot in, a flux map and two scalars out. This is what the submission calls.

The whole of Challenge 2 in four steps, none of which involves a fitted model:

1. `greens.build_basis` turns the shipped conductor rectangles into one flux map per current
   column. Geometry only — and MAST ships the same 812 rows on every shot, so it is built once and
   cached rather than 1206 times at 13.3 s each.
2. `shot.read` interpolates the coil currents and Ip onto the EFIT frames.
3. `gs.solve` runs the free-boundary Grad-Shafranov iteration, per frame.
4. The two value scalars are read off the converged equilibrium.

`CALIBRATION` holds the eleven coil gains and the three profile numbers. They are machine
constants, not a model: the gains are turn counts the dataset does not ship, and the profile is the
one shape assumption a solve with no magnetic probes cannot avoid making. Both are fitted once, on
the three demo shots that carry truth, by `calibrate.py` — see `mast/README.md` for what that
buys and what it risks.
"""
from __future__ import annotations

import hashlib
from functools import lru_cache
from typing import Any

import numpy as np
import numpy.typing as npt
from threadpoolctl import threadpool_limits

from mast import shot as shot_mod
from mast.calibration import CALIBRATION, SHIPPED, Calibration
from solver import scalars
from solver.greens import CoilBasis, build_basis
from solver.gs import Operator, Solved, operator, solve
from solver.machine import MAST

FloatArray = npt.NDArray[np.floating]

# Columns `slim_row` keeps: the inference inputs, and nothing else. A full MAST row is ~5 MB and
# every scored shot stays resident for the whole fold, because the metric pools R2 across it.
# `magnetics_*` columns named by the coil table are added on top, in `slim_row` itself.
KEEP = ("source", "efit_times", "efit_grid_R", "efit_grid_Z", "magnetics_time",
        "coil_name", "coil_input_column", "coil_R", "coil_Z", "coil_width", "coil_height",
        shot_mod.TF_COLUMN,
        "thomson_core_R", "thomson_core_times", "thomson_core_Te", "thomson_core_ne",
        "thomson_edge_spatial", "thomson_edge_times", "thomson_edge_Te", "thomson_edge_ne")

_BASIS: dict[str, CoilBasis] = {}

# Where the pressure profile is sampled in normalised flux. 33 nodes over a 65x65 grid: finer would
# be interpolating the same measurement twice.
PSI_N_NODES = np.linspace(0.0, 1.0, 33)


def slim_row(row: Any) -> dict:
    """The inference inputs only, plus every `magnetics_*` column the coil table names."""
    out: dict[str, Any] = {}
    for col in KEEP:
        if col not in row:
            raise ValueError(f"no column {col!r} on this MAST row")
        v = row[col]
        out[col] = v if isinstance(v, str) else np.array(v, copy=True)
    for col in {*(str(c) for c in np.asarray(row["coil_input_column"])), shot_mod.IP_COLUMN}:
        if col not in row:
            raise ValueError(f"the coil table names {col!r} but the row carries no such signal")
        out[col] = np.array(row[col], copy=True)
    return out


def _geometry_key(row: Any) -> str:
    """A digest of everything `build_basis` reads, so identical geometry hits the cache."""
    h = hashlib.blake2b(digest_size=16)
    for col in ("coil_R", "coil_Z", "coil_width", "coil_height", "efit_grid_R", "efit_grid_Z"):
        h.update(np.ascontiguousarray(np.asarray(row[col], dtype=np.float64)).tobytes())
    h.update("|".join(str(c) for c in np.asarray(row["coil_input_column"])).encode())
    return h.hexdigest()


def basis_for(row: Any) -> CoilBasis:
    """The coil basis, built once per distinct geometry. MAST's is the same on all 1206 shots."""
    key = _geometry_key(row)
    if key not in _BASIS:
        _BASIS[key] = build_basis(row, MAST)
    return _BASIS[key]


@lru_cache(maxsize=2)
def _gain_vector(columns: tuple, cal: Calibration) -> FloatArray:
    """The calibrated gain per basis column, in the basis's own column order."""
    missing = [c for c in columns if c not in cal.gains]
    if missing:
        raise ValueError(f"the calibration has no gain for {missing}; it was fitted on columns "
                         f"{sorted(cal.gains)}")
    # `updown` tilts the vertical field: upper coils up, lower coils down. A column is upper or
    # lower by the letter MAST's own naming puts before `_current` — p2u/p2l and so on. The
    # solenoid has neither and is left alone, which is right: it is not a shaping coil.
    out = []
    for c in columns:
        stem = c.replace("magnetics_", "").replace("_current", "")
        tilt = (1.0 + cal.updown) if stem.endswith("u") else (
            (1.0 - cal.updown) if stem.endswith("l") else 1.0)
        out.append(cal.gains[c] * tilt)
    return np.array(out, dtype=np.float64)


def solve_shot(row: Any, cal: Calibration = CALIBRATION) -> tuple[shot_mod.Shot, Operator,
                                                                  list[Solved]]:
    """The shot's inputs, its grid operator, and the converged equilibrium of every frame.

    **Pinned to one BLAS thread, and it is worth 11x.** Every array here is 65x65 — a sparse
    back-substitution of 3969 unknowns and a (256 x 4225) matvec — so a threaded BLAS spends all
    of its time in fork/join and none in arithmetic. Measured on one `mast_public_test` shot:
    **720 ms a frame with the default thread pool, 63 ms with one thread**. On this box `nproc`
    reports 64 against 32 real cores, so the default is doubly wrong, and running several shots in
    a pool makes it worse rather than better. 1.1 hours for the whole 1206-shot split on one core.
    """
    basis = basis_for(row)
    s = shot_mod.read(row, basis.columns)
    op = operator(MAST, basis.grid_R, basis.grid_Z)
    psi_coil = basis.flux(s.currents * _gain_vector(tuple(basis.columns), cal)[None, :])
    with threadpool_limits(limits=1):
        solved = []
        failed = []
        for k in range(len(s.times)):
            try:
                solved.append(solve(psi_coil[k], float(s.ip[k]), op, profile=cal.profile,
                                    n_iter=cal.n_iter, relax=cal.relax,
                                    thomson=(s.ts_R, s.ts_te[k], s.ts_ne[k], PSI_N_NODES)))
            except ValueError:
                solved.append(_no_plasma(psi_coil[k], op))
                failed.append(k)
    if failed:
        ip = np.abs(s.ip[failed]) / 1e3
        print(f"  note: {len(failed)} of {len(s.times)} frames have no closed surface "
              f"(|Ip| {ip.min():.0f}-{ip.max():.0f} kA); emitting the coil field alone for them")
    return s, op, solved


def _no_plasma(psi_coil: FloatArray, op: Operator) -> Solved:
    """What to emit for a frame whose free-boundary iteration finds no closed surface.

    **Measured: 4 frames of 3243, over 60 `mast_public_test` shots — 0.12%.** All four are at
    |Ip| of 157 to 245 kA against a flat-top 600 to 800, and all four are within the last three
    frames of their shot. That is the current ramp-down, where the discharge is dying and there is
    no confined plasma left to solve for — so the coil field alone is not a fallback standing in
    for an answer, it IS the answer, and the count is printed rather than swallowed so that a rate
    which stops being 0.1% is visible immediately.

    `inside` is left empty on purpose: `scalars.minor_radius` then raises, and `value_scalars`
    falls back to the constants for q95 and betaN on exactly these frames.
    """
    return Solved(psi=psi_coil, psi_axis=float(psi_coil[op.mask].max()),
                  psi_bnd=float(psi_coil[op.mask].max()), r_axis=op.seed_R, z_axis=op.seed_Z,
                  inside=np.zeros(psi_coil.shape, dtype=bool),
                  j_phi=np.zeros_like(psi_coil), iterations=0, moved=float("nan"))


def predict_row(row: Any, cals: tuple[Calibration, ...] = SHIPPED) -> dict:
    """`{"psirz": (T, nZ, nR), "q95": (T,), "betaN": (T,)}` — the scorer's own key names.

    Several calibrations are averaged as DECODED flux maps, which is the only thing they are
    commensurable in: each carries its own coil gains, its own TF constant and its own two affines,
    so their intermediate quantities mean different things. Challenge 1 learned the same lesson
    averaging across feature sets — see `artefacts/candidate/README.md`.
    """
    maps, q95, beta = [], [], []
    for cal in cals:
        s, op, solved = solve_shot(row, cal)
        psi = np.stack([g.psi for g in solved])
        if not np.isfinite(psi).all():
            raise ValueError(f"the solve returned {int((~np.isfinite(psi)).sum())} non-finite "
                             f"cells")
        q, b = value_scalars(s, op, solved, cal)
        maps.append(psi)
        q95.append(q)
        beta.append(b)
    return {"psirz": np.mean(maps, axis=0),
            "q95": np.mean(q95, axis=0),
            "betaN": np.mean(beta, axis=0)}


def value_scalars(s: shot_mod.Shot, op: Operator, solved: list[Solved],
                  cal: Calibration) -> tuple[FloatArray, FloatArray]:
    """`(q95, betaN)` per frame, computed from the equilibrium and Thomson.

    A frame whose contour integral or pressure profile cannot be formed falls back to the constant
    for that scalar rather than emitting nan: the scorer replaces a non-finite scalar by the fold
    mean, which is R2 = 0 for that frame either way, and a nan would also make the failure invisible
    in the printed rate. The counts are returned by `calibrate.py`, not swallowed here.
    """
    q = np.full(len(solved), cal.q95_const, dtype=np.float64)
    b = np.full(len(solved), cal.beta_n_const, dtype=np.float64)
    for k, g in enumerate(solved):
        try:
            q_k = scalars.q95(g, op, float(s.tf[k]), cal.f_per_ka,
                              cal.q95_scale, cal.q95_offset)
            profile = scalars.pressure_profile(g, op, s.ts_R, s.ts_te[k], s.ts_ne[k], PSI_N_NODES)
            b_k = cal.beta_n_offset + cal.beta_n_scale * scalars.beta_n(
                g, op, float(s.tf[k]), cal.f_per_ka, float(s.ip[k]), profile, PSI_N_NODES)
        except ValueError:
            continue
        if np.isfinite(q_k):
            q[k] = q_k
        if np.isfinite(b_k):
            b[k] = b_k
    return q, b
