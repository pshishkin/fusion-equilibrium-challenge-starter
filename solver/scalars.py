#!/usr/bin/env python3
"""q95 and betaN, computed from the converged equilibrium rather than guessed.

They are 0.15 of the composite and there is no model to regress them with, so they have to come out
of the physics. Both do, and each needs exactly one number the dataset does not ship:

* **q95.** The safety factor at psi_N = 0.95 is a contour integral of the flux map,

      q(psi) = (F / 2pi) * contour_integral  dl / (R |grad psi|),        F = R B_phi

  — note the SINGLE power of R. The textbook form is
  `q = (1/2pi) * integral dl B_phi / (R B_p)`, and with psi per radian `B_p = |grad psi| / R`, so
  one R cancels. Writing `R^2 |grad psi|` (the R^2 belongs with `B_p`, not with `grad psi`) makes
  the integral too large by a factor of order `<1/R>`, which on a machine whose plasma spans
  R = 0.2 to 1.4 m is a factor of about four — measured: the TF constant it implied was 0.00117
  T m/kA against the 0.005 that puts B0 at MAST's actual half tesla.

  in which everything but `F` is geometry the solve already produced. `F` is a constant of the
  toroidal field coil, and `magnetics_tf_current` ships — so one calibration constant `f_per_ka`
  turns the shipped current into `F`, and the shape of q95 in time comes entirely from the map.

* **betaN.** The ITER normalisation is `betaN = beta_t[%] * a[m] * B0[T] / Ip[MA]` with
  `beta_t = 2 mu0 <p> / B0^2`, so it needs the volume-averaged pressure. MAST's Thomson is a
  **midplane** laser with a per-channel major radius, which is the one piece of luck in this
  challenge: `p(R) = 2 n_e k T_e` along Z = Z_axis is a direct measurement, and the flux map maps
  R to psi_N so it can be averaged over the plasma volume rather than along the chord.

Both are calibrated in `calibrate.py` against the demo shots and stored in `calibration.py`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import numpy.typing as npt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "fusion_scoring"))

from contour import encloses_point, find_contours, is_closed

from solver.gs import Operator, Solved, midplane_pressure

FloatArray = npt.NDArray[np.floating]

MU0 = 4e-7 * np.pi
# Which flux surface stands in for q95, and it belongs to the SOURCE rather than to the metric.
# With the parametric source the contour integral's correlation with `efit_q95` over the 115 demo
# frames is +0.316 at psi_N = 0.80, **+0.453 at 0.90**, +0.429 at 0.95, +0.262 at 0.99; with the
# Thomson-driven source the whole curve lifts to +0.75-0.78 and the best level moves to 0.80. So
# the level was retuned when that source shipped and has to be retuned back with it — a proxy
# surface is not a constant of the machine.
Q_LEVEL = 0.90


def _bilinear(field: FloatArray, rc: FloatArray) -> FloatArray:
    """Sample `field` at fractional (row, col) coordinates, as `find_contours` returns them."""
    r0 = np.clip(np.floor(rc[:, 0]).astype(int), 0, field.shape[0] - 2)
    c0 = np.clip(np.floor(rc[:, 1]).astype(int), 0, field.shape[1] - 2)
    fr, fc = rc[:, 0] - r0, rc[:, 1] - c0
    return (field[r0, c0] * (1 - fr) * (1 - fc) + field[r0 + 1, c0] * fr * (1 - fc)
            + field[r0, c0 + 1] * (1 - fr) * fc + field[r0 + 1, c0 + 1] * fr * fc)


def q_shape(s: Solved, op: Operator, level: float = Q_LEVEL) -> float:
    """`(1 / 2pi) * contour_integral dl / (R |grad psi|)` at psi_N = `level`, in 1/(Wb/rad).

    Multiply by `F = R B_phi` to get q. Returns nan when no closed surface exists at that level,
    which the caller must handle rather than paper over — a frame with no 95% surface is a frame
    whose equilibrium did not converge.
    """
    # `Solved` stores the axis and boundary flux in the machine's own sign. Contour finding wants
    # the orientation in which the axis is the MAXIMUM, and which one that is follows from the two
    # numbers themselves rather than from the sign of Ip, which is not carried on the result.
    up = float(np.sign(s.psi_axis - s.psi_bnd))
    look = up * s.psi
    axis, bnd = up * s.psi_axis, up * s.psi_bnd
    target = axis - level * (axis - bnd)
    iz = int(np.argmin(np.abs(op.grid_Z - s.z_axis)))
    ir = int(np.argmin(np.abs(op.grid_R - s.r_axis)))

    dR = float(op.grid_R[1] - op.grid_R[0])
    dZ = float(op.grid_Z[1] - op.grid_Z[0])
    gz, gr = np.gradient(look, dZ, dR)
    grad = np.hypot(gr, gz)

    for c in find_contours(look, float(target)):
        if len(c) < 8 or not is_closed(c) or not encloses_point(c, iz, ir):
            continue
        r = op.grid_R[0] + c[:, 1] * dR
        z = op.grid_Z[0] + c[:, 0] * dZ
        dl = np.hypot(np.diff(r), np.diff(z))
        g = _bilinear(grad, c)
        mid_r = 0.5 * (r[1:] + r[:-1])
        mid_g = 0.5 * (g[1:] + g[:-1])
        if not (mid_g > 0).all():
            continue
        return float((dl / (mid_r * mid_g)).sum() / (2.0 * np.pi))
    return float("nan")


def q95(s: Solved, op: Operator, tf_current_ka: float, f_per_ka: float,
        scale: float = 1.0, offset: float = 0.0) -> float:
    """The safety factor, from the contour integral through an AFFINE calibration.

    The affine is not a convenience. Measured on the TRUE flux maps of the demo shots, the same
    integral scores R2 **0.551 through the origin and 0.809 with an intercept** — so a fixed
    offset between the proxy surface and the real 95% one is a large part of the error, and fitting
    through zero throws it away. With the Thomson-driven source the same split shows on OUR maps:
    -0.279 through the origin at psi_N = 0.90 against +0.473 affine.
    """
    return offset + scale * abs(f_per_ka * tf_current_ka) * q_shape(s, op)


def minor_radius(s: Solved, op: Operator) -> float:
    """Half the plasma's radial extent at the height of the magnetic axis."""
    iz = int(np.argmin(np.abs(op.grid_Z - s.z_axis)))
    row = np.flatnonzero(s.inside[iz])
    if row.size < 2:
        raise ValueError(f"the boundary spans {row.size} cells at the axis height; "
                         f"a minor radius is not defined")
    dR = float(op.grid_R[1] - op.grid_R[0])
    return 0.5 * float(op.grid_R[row[-1]] - op.grid_R[row[0]] + dR)


def volume_average(s: Solved, op: Operator, psi_n_to_p: FloatArray, nodes: FloatArray) -> float:
    """The volume average over the plasma of a quantity given as p(psi_N) on `nodes`."""
    span = s.psi_axis - s.psi_bnd
    if span == 0:
        raise ValueError("the axis and boundary flux are equal; psi_N is undefined")
    psi_n = np.clip((s.psi_axis - s.psi) / span, 0.0, 1.0)
    rr, _zz = np.meshgrid(op.grid_R, op.grid_Z)
    dv = np.where(s.inside, 2.0 * np.pi * rr * op.dA, 0.0)
    total = float(dv.sum())
    if total <= 0:
        raise ValueError("the boundary encloses no volume")
    return float((np.interp(psi_n, nodes, psi_n_to_p) * dv).sum() / total)


def beta_n(s: Solved, op: Operator, tf_current_ka: float, f_per_ka: float,
           ip_amps: float, psi_n_to_p: FloatArray, nodes: FloatArray) -> float:
    """`betaN = beta_t[%] a B0 / Ip[MA]`, with the pressure profile measured by Thomson."""
    f = abs(f_per_ka * tf_current_ka)
    b0 = f / s.r_axis
    if b0 <= 0 or ip_amps == 0:
        return float("nan")
    beta_t = 2.0 * MU0 * volume_average(s, op, psi_n_to_p, nodes) / b0**2
    return 100.0 * beta_t * minor_radius(s, op) * b0 / abs(ip_amps / 1e6)


def pressure_profile(s: Solved, op: Operator, chord_R: FloatArray, te_ev: FloatArray,
                     ne_m3: FloatArray, nodes: FloatArray) -> FloatArray:
    """Thomson's midplane p(R) resampled onto psi_N, ready for `volume_average`.

    A thin wrapper over `gs.midplane_pressure`, which the SOLVER also calls — the pressure profile
    is now a source term in the equation as well as an input to betaN, and having two versions of
    it drift apart is exactly the kind of bug this fork has caught twice.
    """
    span = s.psi_axis - s.psi_bnd
    iz = int(np.argmin(np.abs(op.grid_Z - s.z_axis)))
    return midplane_pressure(s.psi[iz], op.grid_R, s.psi_axis, span,
                             chord_R, te_ev, ne_m3, nodes)
