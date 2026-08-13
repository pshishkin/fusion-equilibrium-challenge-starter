#!/usr/bin/env python3
"""
The vacuum field: psi from the coil currents, computed rather than learned.

    psi(R,Z) = psi_coil(R,Z) + psi_plasma(R,Z)

The first term is not a modelling problem, it is a Green's function times a current, and every
ingredient ships on every row of every split: `coil_R/Z/width/height` place the conductors,
`coil_input_column` joins them to the `magnetics_*` current columns, and `efit_grid_R/Z` gives the
grid to evaluate on. Nothing here reads a label, so it works unchanged on the test splits.

For an axisymmetric circular filament of radius `a` at height `z` carrying current `I`, the
poloidal flux per radian at (R, Z) is

    psi = mu0 * I * sqrt(a R) / (pi k) * [ (1 - k^2/2) K(k) - E(k) ],
    k^2 = 4 a R / ((a + R)^2 + (Z - z)^2)

with K and E the complete elliptic integrals. Superposition does the rest: the basis is one map
per current column, and psi_coil is a matrix-vector product against that column's time series.

TWO GRANULARITIES, ONE CODE PATH. DIII-D ships 19 lumped rectangles with the turn count already
folded into the current (`magnetics_F*` is ampere-turns), so a rectangle's filaments AVERAGE --
they share out one conductor's current over its cross-section. MAST ships 812 individual turns,
several rows per current column, so rows of one group SUM. Averaging within a row and summing
across rows of a group is the rule that gives both machines the right answer.

WHAT IS MISSING, deliberately: `magnetics_bcoil` (DIII-D) and MAST's `tf`/`efps` have no
poloidal-plane rectangle -- the toroidal field coil drives no poloidal flux, so it belongs in no
Green's function. `ECOILB`, a second solenoid group co-located with `ECOILA`, is not shipped at
all, and induced vessel currents are not shipped either. So psi_coil is the vacuum field of the
coils we were given, not of the machine; whatever it misses stays in the residual for a model to
learn, which is exactly where it can still be learned from the currents it is fed.

STORAGE SIGN. The formula above puts a MAXIMUM at a positive-current loop, which is the sign
convention MAST stores and the OPPOSITE of DIII-D's -- the same one bit `fusion_scoring/common.py`
already measured as `AXIS_SIGN`. The basis is multiplied by it, so `CoilBasis.flux` comes out
directly comparable to the machine's own stored `efit_psirz`, with no fitted sign anywhere.

CALIBRATED, NOT ASSUMED. With the coils and one filament at the magnetic axis carrying Ip, a
free-gain fit outside the plasma boundary over 200 frames returns 1.00 for the F-coils as a group
and 0.94 for that filament (measured; `eda_coil_field.py` reprints it). Two zero-parameter
predictions landing on 1.0 is what says the shipped ampere-turns, the shipped rectangles, the
elliptic integrals and this sign convention all agree with the stored flux. The 0.94 is physics,
not error: a 1 MA plasma is not a filament, and its external field is that of a slightly weaker
one.

`ECOILA` is the exception and is expected to be off: it ships in kA, NOT kA-turn (README_ORIGINAL
-- EFIT models that group as 48 single-turn elements, `ECOILB` is a second co-located group that
is not shipped at all, so its gain absorbs both). Measured, it comes out near 140.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import ellipe, ellipk

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "fusion_scoring"))

from common import AXIS_SIGN

from my_experiments.models import FloatArray

MU0 = 4e-7 * np.pi

# Currents ship in kA (or kA-turn); the Green's function wants amperes.
CURRENT_UNIT = 1e3

# Filaments per rectangle, radial x vertical. The point is not accuracy of the far field -- one
# filament is already excellent there -- but that four DIII-D F-coils sit INSIDE the flux grid
# (R = 0.861 against a grid starting at 0.840), where a single filament would put a 1/r-ish
# singularity a few millimetres from a grid node. Subdividing spreads it over the real conductor.
N_FIL_R, N_FIL_Z = 3, 7


@dataclass(frozen=True)
class CoilBasis:
    """One flux map per current column, in Wb/rad per unit of the column's own current."""

    columns: list[str]        # `magnetics_*`, the join key from `coil_input_column`
    maps: FloatArray          # (C, nZ, nR), already in the machine's storage sign
    grid_R: FloatArray        # (nR,)
    grid_Z: FloatArray        # (nZ,)
    machine: str              # "DIII-D" or "MAST", from the row's `source`
    min_gap: float            # closest approach of any filament to any grid node, metres

    def flux(self, currents: FloatArray) -> FloatArray:
        """(T, C) currents in column order -> (T, nZ, nR) coil flux."""
        if currents.ndim != 2 or currents.shape[1] != len(self.columns):
            raise ValueError(
                f"currents {currents.shape} does not match {len(self.columns)} basis columns"
            )
        return np.tensordot(currents, self.maps, axes=(1, 0))


def rectangle_filaments(r: float, z: float, width: float, height: float) -> FloatArray:
    """(F, 2) filament centres tiling one conductor rectangle, equally weighted.

    The six sheared DIII-D coils (`coil_angle1`/`angle2` on F5/F6/F7) are tiled as upright
    rectangles of the same centre and extent: EFIT's two shear conventions are not documented in
    this dataset, and inventing one would move conductors by centimetres in an unverifiable
    direction. The cost is a redistribution inside one coil, which `fit_currents` can partly
    absorb into that coil's coefficient; the residual maps in `eda_coil_field.py` are where to
    look if it turns out to matter.
    """
    du = (np.arange(N_FIL_R) + 0.5) / N_FIL_R - 0.5
    dv = (np.arange(N_FIL_Z) + 0.5) / N_FIL_Z - 0.5
    rr = r + du * width
    zz = z + dv * height
    return np.stack(np.broadcast_arrays(rr[:, None], zz[None, :]), axis=-1).reshape(-1, 2)


def green_psi(fil: FloatArray, grid_R: FloatArray, grid_Z: FloatArray) -> FloatArray:
    """(nZ, nR) poloidal flux per radian, in Wb/rad per ampere, summed over the filaments."""
    rr, zz = np.meshgrid(grid_R, grid_Z)                      # (nZ, nR)
    a = fil[:, 0][:, None, None]
    z0 = fil[:, 1][:, None, None]
    d2 = (a + rr) ** 2 + (zz - z0) ** 2
    k2 = 4.0 * a * rr / d2
    if not np.isfinite(k2).all() or (k2 >= 1.0).any():
        raise ValueError("a filament coincides with a grid node: k^2 reached 1, psi is infinite")
    psi = (MU0 / np.pi) * np.sqrt(a * rr) / np.sqrt(k2) * (
        (1.0 - k2 / 2.0) * ellipk(k2) - ellipe(k2)
    )
    return np.asarray(psi.sum(axis=0), dtype=np.float64)


def vacuum_residual(grid_R: FloatArray, grid_Z: FloatArray,
                    r: float = 1.7, z: float = 0.0, keep_out: float = 0.5) -> float:
    """max |Delta* psi| / max |psi| for one filament, away from it — a self-check on the formula.

    The Grad-Shafranov operator `psi_RR - psi_R/R + psi_ZZ` annihilates any vacuum flux, so a
    Green's function that does not zero it is wrong no matter how plausible its picture looks.
    This catches the errors a fitted gain would otherwise absorb into itself; the dropped `1/pi`
    would not have been caught here, which is exactly why both checks are worth having.
    """
    g = green_psi(np.array([[r, z]], dtype=np.float64), grid_R, grid_Z)
    dr = float(grid_R[1] - grid_R[0])
    dz = float(grid_Z[1] - grid_Z[0])
    d2r = (g[:, 2:] - 2 * g[:, 1:-1] + g[:, :-2]) / dr**2
    d1r = (g[:, 2:] - g[:, :-2]) / (2 * dr)
    d2z = (g[2:, 1:-1] - 2 * g[1:-1, 1:-1] + g[:-2, 1:-1]) / dz**2
    star = d2r[1:-1] - d1r[1:-1] / grid_R[None, 1:-1] + d2z
    rr, zz = np.meshgrid(grid_R[1:-1], grid_Z[1:-1])
    far = np.hypot(rr - r, zz - z) > keep_out
    return float(np.abs(star[far]).max() / np.abs(g).max())


def build_basis(row: Any) -> CoilBasis:
    """The per-column flux basis for one shot's machine geometry.

    Read from the row rather than hardcoded: DIII-D's rectangles are the same on every shot, but
    MAST's are a different machine entirely and the same code has to serve both.
    """
    for col in ("source", "coil_input_column", "coil_R", "coil_Z", "coil_width", "coil_height",
                "efit_grid_R", "efit_grid_Z"):
        if col not in row:
            raise ValueError(f"no column {col!r} on this row: coil geometry is required")

    machine = str(row["source"])
    if machine not in AXIS_SIGN:
        raise ValueError(f"source {machine!r} has no flux sign convention; "
                         f"known: {sorted(AXIS_SIGN)}")

    grid_R = np.asarray(row["efit_grid_R"], dtype=np.float64)
    grid_Z = np.asarray(row["efit_grid_Z"], dtype=np.float64)
    join = [str(c) for c in np.asarray(row["coil_input_column"])]
    cR = np.asarray(row["coil_R"], dtype=np.float64)
    cZ = np.asarray(row["coil_Z"], dtype=np.float64)
    cW = np.asarray(row["coil_width"], dtype=np.float64)
    cH = np.asarray(row["coil_height"], dtype=np.float64)
    if not (len(join) == len(cR) == len(cZ) == len(cW) == len(cH)):
        raise ValueError(
            f"coil table is ragged: {len(join)} columns, {len(cR)} R, {len(cZ)} Z, "
            f"{len(cW)} width, {len(cH)} height"
        )

    columns = sorted(set(join))
    maps = np.zeros((len(columns), grid_Z.size, grid_R.size), dtype=np.float64)
    min_gap = np.inf
    for i, column in enumerate(columns):
        rows = [j for j, c in enumerate(join) if c == column]
        for j in rows:
            fil = rectangle_filaments(cR[j], cZ[j], cW[j], cH[j])
            min_gap = min(min_gap, _closest_node(fil, grid_R, grid_Z))
            # Average within a rectangle (its turns are folded into the current), sum across the
            # rows that share a current column (MAST ships one row per turn).
            maps[i] += green_psi(fil, grid_R, grid_Z) / len(fil)
    maps *= CURRENT_UNIT * AXIS_SIGN[machine]
    return CoilBasis(columns, maps, grid_R, grid_Z, machine, float(min_gap))


def filament_flux(r: float, z: float, current: float, basis: CoilBasis) -> FloatArray:
    """(nZ, nR) flux of one filament at (r, z), in the same units and sign as `basis.maps`.

    Calibration only: the leading term of the plasma's OWN field outside the boundary is a
    filament at the magnetic axis carrying Ip, and it has to be in the design matrix or the coil
    gains absorb it. With 1 MA of plasma against ~140 kA-turn per shaping coil, that absorption is
    not a small correction -- fitting without it returns coil gains scattered over half an order of
    magnitude and both signs. This reads the axis position, which is a LABEL: it belongs to the
    calibration check, never to a feature.
    """
    g = green_psi(np.array([[r, z]], dtype=np.float64), basis.grid_R, basis.grid_Z)
    return g * current * CURRENT_UNIT * AXIS_SIGN[basis.machine]


def fit_flux_gains(maps: FloatArray, currents: FloatArray, psi: FloatArray) -> FloatArray:
    """(C,) gains minimising ||psi - sum_c g_c I_c G_c||^2 over every frame and every grid node.

    This is the gain the PIPELINE uses, and it is a different question from the calibration in
    `fit_gains`. That one asks "is the Green's function right", answers it away from the plasma,
    and needs the plasma in the design to answer it honestly. This one asks "what linear-in-current
    field leaves the model the least to learn", which is a modelling choice with no true value to
    recover: whatever is subtracted is added back unchanged at inference, so the split is exact for
    any gains at all.

    Fitting rather than hardcoding also disposes of `ECOILA`, whose turn count the dataset does not
    ship and which is not cleanly identifiable anyway -- allow a per-frame constant in the
    calibration and its gain moves from +142 to -10, because a solenoid's field over this grid is
    nearly degenerate with an offset.

    Never assembled as a matrix: the design is an outer product, so A^T A is the elementwise
    product of the current second-moment matrix with the maps' Gram matrix, and A^T b needs only
    each frame's projection onto each map. Both are (C, C) and (T, C), not (T*nodes, C).
    """
    flat = maps.reshape(len(maps), -1)
    gram = flat @ flat.T                                     # (C, C)
    cur2 = currents.T @ currents                             # (C, C)
    proj = psi.reshape(len(psi), -1) @ flat.T                # (T, C)
    ata = cur2 * gram
    atb = np.einsum("tc,tc->c", currents, proj)
    gains, *_ = np.linalg.lstsq(ata, atb, rcond=None)
    return np.asarray(gains, dtype=np.float64)


def coil_pca_transform(
    maps: FloatArray, currents: FloatArray, n_components: int
) -> tuple[FloatArray, FloatArray]:
    """(mean, W) such that `(I - mean) @ W` is the coil flux in its own principal directions.

    The coil flux lives in the C-dimensional span of the maps, so its PCA has at most C components
    and each one is a fixed linear combination of the currents. That is worth stating plainly:
    this feature set carries exactly the information the currents carry, rotated into the order of
    how much FLUX each direction actually produces and truncated there. For a linear model it is
    the same features; for a tree, which splits on one coordinate at a time, it is not.

    So no flux map is ever materialised. With `S` the symmetric square root of the current
    covariance, the principal directions are the right singular vectors of `S @ maps`, and
    projecting onto them collapses to the (C, k) matrix `maps @ V_k^T`.
    """
    n_cols = len(maps)
    if not 0 < n_components <= n_cols:
        raise ValueError(
            f"n_coil_pca must be in 1..{n_cols} — the coil flux spans {n_cols} dimensions, one per "
            f"current column, so asking for {n_components} components asks for directions that do "
            f"not exist"
        )
    flat = maps.reshape(n_cols, -1)
    mean = currents.mean(axis=0)
    cov = np.cov((currents - mean).T)
    eigval, eigvec = np.linalg.eigh(cov)
    root = eigvec @ np.diag(np.sqrt(np.clip(eigval, 0.0, None))) @ eigvec.T
    _, _, vt = np.linalg.svd(root @ flat, full_matrices=False)
    return mean, np.asarray(flat @ vt[:n_components].T, dtype=np.float64)


def _closest_node(fil: FloatArray, grid_R: FloatArray, grid_Z: FloatArray) -> float:
    """Distance from the nearest grid node to the nearest filament, metres.

    Reported rather than acted on: it is the scale at which the log singularity of the Green's
    function is being resolved, so a run where it collapses to millimetres is a run whose coil
    flux near that node is a subdivision artefact, not physics.
    """
    dr = fil[:, 0][:, None] - grid_R[None, :]
    dz = fil[:, 1][:, None] - grid_Z[None, :]
    return float(np.sqrt((dr**2).min(axis=1)[:, None] + (dz**2).min(axis=1)[None, :]).min())


class Calibration:
    """Per-coil gains against the stored flux, with the plasma given room to be wrong.

    This answers "is the Green's function right", and it is NOT what the pipeline uses -- see
    `fit_flux_gains` for that. A gain of 1.0 says the shipped ampere-turns, the shipped rectangle,
    the elliptic integrals and the storage sign all agree with the data, with nothing fitted.

    Three things have to be handled or the answer is wrong, and each one was measured:

    * **Fit outside the plasma boundary.** Inside it, the flux is the plasma's.
    * **Put the plasma in the design.** A 1 MA plasma against ~140 kA-turn per shaping coil reaches
      well past the boundary; omitting it returned gains from -0.15 to -0.57, both signs.
    * **Give it a per-frame amplitude and a per-frame constant, not one global gain.** A single
      filament is a crude plasma, and pooling its amplitude over frames pushes its misfit back into
      the coils: the F-coil group reads 0.997 over 10 shots and 0.911 over 40 with a pooled
      amplitude, but 0.87 either way once each frame gets its own. The larger sample is the
      trustworthy one, so treat 0.9 and not 1.00 as the measured agreement.

    Accumulated frame by frame, because the nuisance columns are per-frame: the normal equations
    are (C, C) however many frames go in.
    """

    def __init__(self, n_coils: int) -> None:
        self.ata = np.zeros((n_coils, n_coils))
        self.atb = np.zeros(n_coils)
        self.sse = 0.0
        self.n_nodes = 0

    def add(self, maps: FloatArray, currents: FloatArray, psi: FloatArray,
            mask: FloatArray, nuisance: FloatArray) -> None:
        """One frame: (C,nZ,nR) maps, (C,) currents, (nZ,nR) psi and mask, (nZ,nR,K) nuisance."""
        sel = mask.reshape(-1)
        a = (currents[:, None] * maps.reshape(len(maps), -1)).T[sel]
        b = psi.reshape(-1)[sel]
        q, _ = np.linalg.qr(nuisance.reshape(-1, nuisance.shape[-1])[sel])
        a = a - q @ (q.T @ a)                 # project the per-frame nuisances out of both sides
        b = b - q @ (q.T @ b)
        self.ata += a.T @ a
        self.atb += a.T @ b
        self.sse += float(b @ b)
        self.n_nodes += int(sel.sum())

    def solve(self, groups: FloatArray | None = None) -> tuple[FloatArray, float]:
        """Gains and the R^2 they reach. `groups` (C, k) ties coils together, e.g. all F-coils."""
        ata, atb = self.ata, self.atb
        if groups is not None:
            ata, atb = groups.T @ ata @ groups, groups.T @ atb
        gains, *_ = np.linalg.lstsq(ata, atb, rcond=None)
        explained = 2.0 * float(gains @ atb) - float(gains @ ata @ gains)
        return np.asarray(gains, dtype=np.float64), explained / self.sse
