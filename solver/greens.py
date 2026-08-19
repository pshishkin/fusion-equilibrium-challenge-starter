#!/usr/bin/env python3
"""The vacuum field: what the coils put on the grid, before any plasma.

Everything here is a Green's function times a current — no fitting, no data, and nothing that
needs a training set, which is the whole reason Challenge 2 is approachable at all. MAST ships
`coil_R/Z/width/height` and `coil_input_column` on every row of the test split, so the same
calculation that needs 7041 labelled shots to learn on DIII-D is exact here for free.

The flux per radian of a circular filament of radius `a` at height `z0` carrying current `I`,
evaluated at `(R, Z)`:

    psi = (mu0 I / pi) sqrt(a R) / k [ (1 - k^2/2) K(k) - E(k) ],   k^2 = 4 a R / ((a+R)^2 + dz^2)

The `1/pi` is the trap: dropping it puts every fitted gain at exactly 1/pi and the picture still
looks right. `vacuum_residual` is the second, independent check — the Grad-Shafranov operator
annihilates any vacuum flux, so a Green's function that does not zero it is wrong however
plausible the plot.

**MAST ships one row per conductor turn**, 812 of them, the solenoid alone being 656. DIII-D ships
one row per coil. So the join from rows to current columns is many-to-one here and the weighting
inside it is a question with a measured answer — see `COIL_TURNS` below.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.special import ellipe, ellipk

from solver.machine import Machine

FloatArray = npt.NDArray[np.floating]

MU0 = 4e-7 * np.pi

# Filaments per conductor rectangle, radial x vertical. MAST's P-coil rectangles are centimetres
# across and sit far from the grid nodes that matter, but the solenoid's inner edge is at
# R = 0.119 against a grid starting at R = 0.060, so subdividing is not optional there.
N_FIL_R, N_FIL_Z = 3, 5


@dataclass(frozen=True)
class CoilBasis:
    """One flux map per current column, in Wb/rad per shipped unit, in the machine's own sign."""

    columns: list[str]
    maps: FloatArray          # (C, nZ, nR)
    grid_R: FloatArray        # (nR,)
    grid_Z: FloatArray        # (nZ,)
    turns: FloatArray         # (C,) rows that shared each column
    min_gap: float            # closest approach of any filament to any grid node, metres
    machine: Machine

    def flux(self, currents: FloatArray) -> FloatArray:
        """(T, C) currents in column order -> (T, nZ, nR)."""
        if currents.ndim != 2 or currents.shape[1] != len(self.columns):
            raise ValueError(f"currents {currents.shape} against {len(self.columns)} basis columns")
        return np.tensordot(currents, self.maps, axes=(1, 0))


def rectangle_filaments(r: float, z: float, width: float, height: float) -> FloatArray:
    """(F, 2) filament centres tiling one conductor rectangle, equally weighted.

    `coil_angle1`/`coil_angle2` are structurally 0.0 on every MAST row — IMAS rectangles carry no
    skew — so unlike DIII-D there is no shear convention to guess at here.
    """
    du = (np.arange(N_FIL_R) + 0.5) / N_FIL_R - 0.5
    dv = (np.arange(N_FIL_Z) + 0.5) / N_FIL_Z - 0.5
    rr, zz = r + du * width, z + dv * height
    return np.stack(np.broadcast_arrays(rr[:, None], zz[None, :]), axis=-1).reshape(-1, 2)


def green_psi(fil: FloatArray, at_R: FloatArray, at_Z: FloatArray) -> FloatArray:
    """Flux per radian in Wb/rad per ampere, summed over `fil`, at points broadcast from R and Z.

    `at_R` and `at_Z` are used as given rather than meshed, so this serves both the full grid
    (pass the outer product) and a list of boundary points (pass two 1-D arrays of equal length).
    """
    if at_R.shape != at_Z.shape:
        raise ValueError(f"evaluation points disagree: R {at_R.shape}, Z {at_Z.shape}")
    a = fil[:, 0].reshape((-1,) + (1,) * at_R.ndim)
    z0 = fil[:, 1].reshape((-1,) + (1,) * at_R.ndim)
    d2 = (a + at_R) ** 2 + (at_Z - z0) ** 2
    k2 = 4.0 * a * at_R / d2
    if not np.isfinite(k2).all() or (k2 >= 1.0).any():
        raise ValueError("a filament coincides with an evaluation point: k^2 reached 1")
    psi = (MU0 / np.pi) * np.sqrt(a * at_R) / np.sqrt(k2) * (
        (1.0 - k2 / 2.0) * ellipk(k2) - ellipe(k2))
    return np.asarray(psi.sum(axis=0), dtype=np.float64)


def grid_green(fil: FloatArray, grid_R: FloatArray, grid_Z: FloatArray) -> FloatArray:
    """(nZ, nR) — `green_psi` over the full grid."""
    rr, zz = np.meshgrid(grid_R, grid_Z)
    return green_psi(fil, rr, zz)


def vacuum_residual(grid_R: FloatArray, grid_Z: FloatArray,
                    r: float = 0.9, z: float = 0.0, keep_out: float = 0.4) -> float:
    """max |Delta* psi| / max |psi| for one filament, away from it — the formula's self-check."""
    g = grid_green(np.array([[r, z]], dtype=np.float64), grid_R, grid_Z)
    dr, dz = float(grid_R[1] - grid_R[0]), float(grid_Z[1] - grid_Z[0])
    d2r = (g[:, 2:] - 2 * g[:, 1:-1] + g[:, :-2]) / dr**2
    d1r = (g[:, 2:] - g[:, :-2]) / (2 * dr)
    d2z = (g[2:, 1:-1] - 2 * g[1:-1, 1:-1] + g[:-2, 1:-1]) / dz**2
    star = d2r[1:-1] - d1r[1:-1] / grid_R[None, 1:-1] + d2z
    rr, zz = np.meshgrid(grid_R[1:-1], grid_Z[1:-1])
    return float(np.abs(star[np.hypot(rr - r, zz - z) > keep_out]).max() / np.abs(g).max())


def build_basis(row: Any, machine: Machine) -> CoilBasis:
    """The per-column flux basis for MAST's geometry, read from the row rather than hardcoded.

    Rows sharing a current column are AVERAGED, not summed. That is a measured choice and not an
    obvious one: summing is what the IMAS picture suggests, since each row is one conductor turn
    carrying the column's current, and it overstates the field by the turn count. Fitted per-column
    gains against the three demo shots' true flux, with a plasma filament free to take the plasma's
    own field, come out at **1.16 / 1.11 (P2), 1.01 / 1.07 (P4), 0.80 / 0.80 (P5)** times the
    averaged basis — i.e. the shipped current is already the coil's total ampere-turns. The badly
    conditioned columns disagree loudly (P3 at 28-31x, the solenoid at 246x), which is why the
    gains are calibrated rather than assumed; see `calibrate.py`.
    """
    for col in ("source", "coil_input_column", "coil_R", "coil_Z", "coil_width", "coil_height",
                "efit_grid_R", "efit_grid_Z"):
        if col not in row:
            raise ValueError(f"no column {col!r} on this row: MAST coil geometry is required")
    source = str(row["source"])
    if source != machine.name:
        raise ValueError(f"row is from {source!r} but the basis was asked for {machine.name!r}")

    grid_R = np.asarray(row["efit_grid_R"], dtype=np.float64)
    grid_Z = np.asarray(row["efit_grid_Z"], dtype=np.float64)
    join = [str(c) for c in np.asarray(row["coil_input_column"])]
    cR = np.asarray(row["coil_R"], dtype=np.float64)
    cZ = np.asarray(row["coil_Z"], dtype=np.float64)
    cW = np.asarray(row["coil_width"], dtype=np.float64)
    cH = np.asarray(row["coil_height"], dtype=np.float64)
    if not (len(join) == len(cR) == len(cZ) == len(cW) == len(cH)):
        raise ValueError(f"coil table is ragged: {len(join)} columns, {len(cR)} R, {len(cZ)} Z, "
                         f"{len(cW)} width, {len(cH)} height")

    columns = sorted(set(join))
    maps = np.zeros((len(columns), grid_Z.size, grid_R.size), dtype=np.float64)
    turns = np.zeros(len(columns), dtype=np.float64)
    min_gap = np.inf
    for i, column in enumerate(columns):
        rows = [j for j, c in enumerate(join) if c == column]
        turns[i] = len(rows)
        for j in rows:
            fil = rectangle_filaments(cR[j], cZ[j], cW[j], cH[j])
            min_gap = min(min_gap, _closest_node(fil, grid_R, grid_Z))
            maps[i] += grid_green(fil, grid_R, grid_Z) / len(fil)
        maps[i] /= len(rows)
    maps *= machine.current_unit * machine.sign
    return CoilBasis(columns, maps, grid_R, grid_Z, turns, float(min_gap), machine)


def _closest_node(fil: FloatArray, grid_R: FloatArray, grid_Z: FloatArray) -> float:
    """Metres from the nearest filament to the nearest grid node — the singularity's margin."""
    dr = np.abs(fil[:, 0][:, None] - grid_R[None, :]).min(axis=1)
    dz = np.abs(fil[:, 1][:, None] - grid_Z[None, :]).min(axis=1)
    return float(np.hypot(dr, dz).min())
