#!/usr/bin/env python3
"""A free-boundary Grad-Shafranov solve, which is what replaces a training set on MAST.

Challenge 1 learns the map from 7041 labelled shots. There is no `mast_train` config, by design,
so on MAST the only thing that knows what an equilibrium looks like is the equation itself:

    Delta* psi  =  -mu0 R J_phi,      Delta* = d2/dR2 - (1/R) d/dR + d2/dZ2
    J_phi       =  R p'(psi) + F F'(psi) / (mu0 R)

with `psi = psi_coil + psi_plasma`, the coil part exact from `greens.py`. Both source profiles are
unknown functions of psi and both are parameterised the standard way — one shape, two exponents,
one radial mix — because nothing in the test split constrains them further:

    J_phi(R, psi_N)  =  lam [ beta R/R0 + (1 - beta) R0/R ] (1 - psi_N^alpha)^gamma

`lam` is not free: it is set every iteration so that the current enclosed by the boundary equals
the measured `Ip`, which is the one plasma quantity MAST does ship. So the solve has **three**
numbers in it, not a fitted model, and the same three serve every shot.

The iteration is Picard: guess the plasma flux, find the axis and the boundary it implies,
rebuild the source from that, solve again. Two pieces make it fast enough to run on 1206 shots:

* **The operator is factorised once.** The grid is a machine constant, so the sparse LU of Delta*
  on the interior is built at import time for a given grid and reused for every frame of every
  shot. What changes per frame is only the right-hand side and the Dirichlet rim.
* **The rim comes from the Green's function, not from a guess.** psi_plasma must decay to zero at
  infinity, and truncating the domain at the grid edge with psi = 0 there would put an image
  current just outside it. Instead the rim values are the exact free-space flux of the current
  distribution being solved for — one precomputed (rim x cells) matrix, one matvec per iteration.

**The boundary is found without a flood fill**, which the profiling would not have allowed: the
last closed surface is the higher of the flux where the plasma touches the limiter envelope and
the flux at the highest interior saddle, both read directly off the grid. See `_boundary_flux`.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.ndimage import label, shift
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "fusion_scoring"))

from o_point import find_o_point

from solver.greens import MU0, green_psi
from solver.machine import MACHINES, Machine

# Two electrons' worth: the ion pressure is not measured, and T_i = T_e with n_i = n_e is the
# standard stand-in. It is a factor on the pressure term and `Profile.p_scale` can absorb it, but
# it is named here rather than hidden inside that constant.
PRESSURE_SPECIES = 2.0
ELEMENTARY_CHARGE = 1.602176634e-19

FloatArray = npt.NDArray[np.floating]
BoolArray = npt.NDArray[np.bool_]

# The machine's stored flux convention, `machine.sign`, is +1 when the axis is a MAXIMUM of
# `efit_psirz` — which is also the sign a positive toroidal current gives in `green_psi`. On MAST
# the two agree and nothing is flipped; on DIII-D they do not, and every place that used to read a
# module constant now reads `op.machine.sign` instead. That is the single change that lets one
# solver serve both machines, and getting it wrong looks like a converged solve with the plasma in
# the wrong place rather than like an error.


@dataclass(frozen=True)
class Profile:
    """What stands in for p'(psi) and FF'(psi).

    Two modes, and `p_scale` picks between them.

    **Parametric (`p_scale = 0`).** Both source functions share one shape and are mixed by `beta`:
    `J = lam [beta R/R0 + (1 - beta) R0/R] (1 - psi_N^alpha)^gamma`. Three global numbers for a
    whole machine — and measured, that is what caps R2_psi at **0.954** even when the TRUE vacuum
    field is substituted in, against a leaderboard best of 0.9732.

    **Thomson-driven (`p_scale > 0`).** The pressure term stops being a fitted fraction and becomes
    a measurement: MAST's Thomson is a midplane laser with a per-channel major radius, so `p(R)` is
    observed, the flux map maps R to psi_N, and `p'(psi)` follows by differentiation —

        J = p_scale * R p'(psi)  +  lam (1 - psi_N^alpha)^gamma / R

    with `lam` still set by the measured Ip, so the pressure term carries whatever it carries and
    the poloidal-current term takes the remainder. `beta` is unused in this mode; `p_scale` is one
    global number covering the ion pressure that is not measured and Thomson's own calibration,
    where `beta` was one global number covering the entire pressure profile of every frame.
    """
    alpha: float = 2.0     # how fast the source falls off in psi_N
    gamma: float = 1.0     # how hard it is switched off at the boundary
    beta: float = 0.5      # parametric mode only: mix of the R and 1/R terms
    p_scale: float = 0.0   # 0 = parametric; > 0 = drive the R term from Thomson's pressure
    # How far past the last closed surface the plasma is taken to extend, as a fraction of the
    # axis-to-boundary flux span. Zero is the textbook boundary. It exists because the boundary
    # this solve produces is measurably too SMALL: on the demo shots our LCFS is **0.929x** the
    # true one on average, and D_LCFS decomposes as roughly a third position, a third size and a
    # third shape (`mast/diagnose_boundary.py`). A size error that systematic is what one number
    # can move, and D_LCFS plus Consistency are 30% of the composite — a share that could not be
    # optimised at all until the scorer stopped grading MAST on DIII-D's grid.
    edge: float = 0.0


@dataclass(frozen=True)
class Solved:
    psi: FloatArray            # (nZ, nR) total flux, MAST's storage sign
    psi_axis: float
    psi_bnd: float
    r_axis: float
    z_axis: float
    inside: BoolArray          # (nZ, nR) cells the solve placed inside the boundary
    j_phi: FloatArray          # (nZ, nR) A/m2
    iterations: int
    moved: float               # last relative change of psi_plasma; the convergence witness


@dataclass(frozen=True)
class Operator:
    """Everything about the grid that does not change between frames."""
    grid_R: FloatArray
    grid_Z: FloatArray
    mask: BoolArray            # the vessel envelope the scorer itself uses
    limiter: tuple             # (column index, fraction) of the inboard wall on the grid
    wall_rc: FloatArray        # (n, 2) the first wall resampled onto fractional grid coordinates
    outside: BoolArray         # cells no closed plasma may reach: past the centre column, or rim
    axis_mask: BoolArray       # where the O-point search may look — the envelope, or a band of it
    lu: Any                    # splu of Delta* on the interior nodes
    rim_index: tuple           # (iz, ir) of the Dirichlet rim, in the order `rim_green` expects
    rim_green: FloatArray      # (n_rim, nZ * nR) free-space flux at the rim per amp in each cell
    dA: float
    seed_R: float              # where iteration 1 puts its blob: the envelope's own centroid
    seed_Z: float
    seed_a: float              # and how wide, as a third of the envelope's half-extent
    seed_b: float
    machine: Machine


def _load_mask(machine: Machine, grid_R: FloatArray, grid_Z: FloatArray) -> BoolArray:
    d = np.load(machine.mask_path)
    if not (np.allclose(d["grid_R"], grid_R) and np.allclose(d["grid_Z"], grid_Z)):
        raise ValueError(f"{machine.envelope} is on a different grid than this row: "
                         f"R {d['grid_R'][0]}..{d['grid_R'][-1]} against "
                         f"{grid_R[0]}..{grid_R[-1]}")
    mask: BoolArray = np.asarray(d["mask_coarse"], dtype=bool)
    if not mask.any():
        raise ValueError(f"{machine.envelope} carries an empty envelope")
    return mask


def _wall_on_grid(machine: Machine, grid_R: FloatArray, grid_Z: FloatArray) -> FloatArray:
    """The first wall resampled to about one point per grid cell, in fractional (row, col).

    Empty when the machine ships no wall, in which case the boundary search falls back to the
    inboard vertical line alone. With a wall, the limiter candidate becomes what EFIT means by it:
    the highest flux anywhere the plasma could touch the machine, not only on the centre column.
    """
    if machine.wall is None:
        return np.zeros((0, 2), dtype=np.float64)
    poly = np.asarray(machine.wall, dtype=np.float64)
    dR, dZ = float(grid_R[1] - grid_R[0]), float(grid_Z[1] - grid_Z[0])
    out = []
    for (r1, z1), (r2, z2) in zip(poly, np.roll(poly, -1, axis=0), strict=True):
        n = max(2, int(np.hypot((r2 - r1) / dR, (z2 - z1) / dZ)) + 1)
        s = np.linspace(0.0, 1.0, n, endpoint=False)
        out.append(np.column_stack([(z1 + s * (z2 - z1) - grid_Z[0]) / dZ,
                                    (r1 + s * (r2 - r1) - grid_R[0]) / dR]))
    return np.concatenate(out)


def _limiter_column(machine: Machine, grid_R: FloatArray) -> tuple[int, float]:
    """Where MAST's centre column cuts the grid: the index below `R_LIMITER` and the fraction past.

    MAST is limited on the inboard side by the centre column, and that is a hard wall rather than
    a modelling choice — the shipped `efit_lcfs_r` of the three demo shots has a **minimum radius of
    0.1963 m on every one of the 115 frames**, to four decimals, which is the machine's own inner
    limiter surface. The flux grid does not have a node there (its columns are 0.1812 and 0.2115),
    so the limiter flux is interpolated between the two; taking the nearer node instead costs
    0.0009 Wb/rad on a boundary flux of 0.027, which is 3% of the whole axis-to-boundary span.
    """
    r = machine.r_limiter
    if grid_R[0] > r or grid_R[-1] < r:
        raise ValueError(f"the grid spans R {grid_R[0]:.3f}..{grid_R[-1]:.3f} and does not reach "
                         f"{machine.name}'s inner wall at {r}")
    j = int(np.searchsorted(grid_R, r) - 1)
    return j, float((r - grid_R[j]) / (grid_R[j + 1] - grid_R[j]))


@lru_cache(maxsize=4)
def _operator(rkey: tuple, zkey: tuple, name: str) -> Operator:
    """Factorise Delta* and precompute the rim Green's matrix for one grid. Cached per grid."""
    machine = MACHINES[name]
    grid_R = np.asarray(rkey, dtype=np.float64)
    grid_Z = np.asarray(zkey, dtype=np.float64)
    nR, nZ = grid_R.size, grid_Z.size
    dR = float(grid_R[1] - grid_R[0])
    dZ = float(grid_Z[1] - grid_Z[0])
    if not (np.allclose(np.diff(grid_R), dR) and np.allclose(np.diff(grid_Z), dZ)):
        raise ValueError("the flux grid is not uniform; the five-point stencil assumes it is")

    # Delta* on the interior, Dirichlet on the rim. Unknowns are numbered (iz - 1) * (nR - 2) +
    # (ir - 1) over iz, ir in 1..n-2.
    nr_i, nz_i = nR - 2, nZ - 2
    rows, cols, vals = [], [], []

    def add(a: int, b: int, v: float) -> None:
        rows.append(a)
        cols.append(b)
        vals.append(v)

    for iz in range(nz_i):
        for ir in range(nr_i):
            k = iz * nr_i + ir
            r = grid_R[ir + 1]
            add(k, k, -2.0 / dR**2 - 2.0 / dZ**2)
            for d, kk in ((+1, ir + 1 < nr_i), (-1, ir - 1 >= 0)):
                v = 1.0 / dR**2 - d / (2.0 * dR * r)
                if kk:
                    add(k, k + d, v)
            for d, kk in ((+1, iz + 1 < nz_i), (-1, iz - 1 >= 0)):
                if kk:
                    add(k, k + d * nr_i, 1.0 / dZ**2)
    lu = splu(csc_matrix((vals, (rows, cols)), shape=(nr_i * nz_i, nr_i * nz_i)))

    # The Dirichlet rim: every node on the grid edge.
    gz, gr = np.meshgrid(np.arange(nZ), np.arange(nR), indexing="ij")
    on_rim = (gz == 0) | (gz == nZ - 1) | (gr == 0) | (gr == nR - 1)
    rim_iz, rim_ir = gz[on_rim], gr[on_rim]
    rr, zz = np.meshgrid(grid_R, grid_Z)
    # (n_rim, nZ * nR): the flux one amp in each cell would put on each rim node. Cells ON the rim
    # are zeroed — they would be coincident with their own rim node, and the source never reaches
    # there anyway since the envelope stops well inside.
    cells = np.column_stack([rr.ravel(), zz.ravel()])
    keep = ~on_rim.ravel()
    g = np.zeros((rim_iz.size, cells.shape[0]), dtype=np.float64)
    at_R = grid_R[rim_ir]
    at_Z = grid_Z[rim_iz]
    for c in np.flatnonzero(keep):
        g[:, c] = green_psi(cells[c][None, :], at_R, at_Z)
    mask = _load_mask(machine, grid_R, grid_Z)
    mz, mr = np.nonzero(mask)
    seed_R = float(grid_R[mr].mean())
    seed_Z = float(grid_Z[mz].mean())
    seed_a = float(np.ptp(grid_R[mr])) / 6.0
    seed_b = float(np.ptp(grid_Z[mz])) / 6.0
    # Where a closed plasma may not reach: the grid's own edge. The centre column is handled
    # separately in `_boundary_flux`, because MAST's wall at R = 0.1963 falls BETWEEN two grid
    # columns and a cell-level test there is a whole grid step coarse — 0.03 m on a 0.5 m plasma.
    outside = np.zeros((nZ, nR), dtype=bool)
    outside[0], outside[-1], outside[:, 0], outside[:, -1] = True, True, True, True
    # The vertical pin, if this machine has one — see `Machine.z_pin` for why an elongated plasma
    # needs it and what it costs.
    axis_mask = mask
    if machine.z_pin is not None:
        axis_mask = mask & (np.abs(grid_Z - machine.z_pin) <= machine.z_band)[:, None]
        if not axis_mask.any():
            raise ValueError(f"{machine.name}'s axis band of +-{machine.z_band} m about "
                             f"{machine.z_pin} contains no envelope cell")
    return Operator(grid_R, grid_Z, mask, _limiter_column(machine, grid_R),
                    _wall_on_grid(machine, grid_R, grid_Z), outside, axis_mask, lu,
                    (rim_iz, rim_ir), g, dR * dZ, seed_R, seed_Z, seed_a, seed_b, machine)


def operator(machine: Machine, grid_R: FloatArray, grid_Z: FloatArray) -> Operator:
    """The cached grid operator. Building it costs a few seconds; every frame after that is free."""
    return _operator(tuple(np.asarray(grid_R, dtype=np.float64)),
                     tuple(np.asarray(grid_Z, dtype=np.float64)), machine.name)


def _saddles(psi: FloatArray) -> BoolArray:
    """Cells where psi has a saddle: the X-points, found by counting sign changes on the 8-ring.

    A discrete critical point is a maximum if every neighbour is below it, a minimum if every one
    is above, and a saddle if the difference changes sign exactly four times going round. Counting
    rather than testing a Hessian keeps this vectorised, and the grid is 65x65 — a fitted Hessian
    on three points per direction would be noise.
    """
    ring = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
    sign = np.stack([np.roll(np.roll(psi, -dz, axis=0), -dr, axis=1) > psi for dz, dr in ring])
    changes = np.zeros(psi.shape, dtype=np.int64)
    for k in range(len(ring)):
        changes += (sign[k] != sign[(k + 1) % len(ring)]).astype(np.int64)
    interior = np.zeros(psi.shape, dtype=bool)
    interior[1:-1, 1:-1] = True
    return (changes == 4) & interior


def _boundary_flux(psi: FloatArray, op: Operator, iz: int, ir: int) -> float:
    """The last closed surface's flux, by testing the finitely many levels it can be at.

    With the axis at a maximum the plasma is `{psi >= psi_bnd}`, so lowering the level grows it and
    `psi_bnd` is the highest level at which it stops being contained. That level is not arbitrary:
    it is either the flux where the plasma reaches MAST's centre column, or the flux at an X-point.
    Both are read straight off the grid, so the search is over **a handful of candidates rather
    than a continuum** — no bisection, and no accuracy lost between bisection steps.

    Each candidate is tested by connectivity, and that is the part a max-of-candidates version gets
    wrong. The centre column carries its own flux hill from the solenoid, and on a diverted frame
    that hill sits ABOVE the true boundary while belonging to no plasma at all — taking the maximum
    without asking what is connected to the axis put the boundary 19% of the axis-to-boundary span
    too high on the median demo frame, and shrank the plasma to nothing on the worst.

    Measured against the shipped `efit_psi_boundary` on all 115 demo frames: see `mast/README.md`.
    """
    span = float(psi[iz, ir]) - float(psi.min())
    if span <= 0:
        raise ValueError(f"the flux map is flat at the axis: span {span:.5g}")
    eps = 1e-3 * span
    axis = float(psi[iz, ir])
    j, f = op.limiter
    column = psi[:, j] * (1.0 - f) + psi[:, j + 1] * f
    iz_lim = int(np.argmax(column))

    sad = _saddles(psi)
    rr, zz = np.meshgrid(op.grid_R, op.grid_Z)
    near = np.hypot(rr - op.grid_R[ir], zz - op.grid_Z[iz]) < op.machine.x_reach
    cand = [(float(v), False) for v in psi[sad & near & (psi < axis)]]
    lim = float(column[iz_lim])
    if lim < axis:
        cand.append((lim, True))

    # Descending, and return the FIRST that binds. A higher level is a smaller plasma, so the
    # highest binding candidate IS the boundary and everything below it is irrelevant — which
    # turns a fixed nine connectivity tests per iteration into one or two. Measured on a
    # `mast_public_test` shot: 2.0 s a frame before, and the whole 1206-shot split would have been
    # 35 hours on one core.
    for level, is_limiter in sorted(cand, reverse=True)[:N_CANDIDATES]:
        blob = _component(psi, iz, ir, level - eps)
        # The centre column binds only if the plasma actually REACHES it: that column carries the
        # solenoid's own flux hill, which on a diverted frame sits above the true boundary while
        # belonging to no plasma at all. An X-point binds if the plasma spills past it, which shows
        # up as the component running out along the divertor legs to the grid edge.
        if blob[iz_lim, j + 1] if is_limiter else bool((blob & op.outside).any()):
            return level

    # Neither wall is reached and no X-point binds: this trial plasma simply fills the vessel. It
    # happens in the first Picard iterations, where the flux is still mostly the seed blob's and no
    # magnetic well has formed — and it is not an error, because the vessel is a wall as much as
    # the centre column is. It needs a bisection rather than a candidate list, since the envelope
    # is a closed curve with no distinguished level, so it is done separately: measured on the demo
    # shots, the two candidate types bind on 115 frames out of 115 once the iteration has settled.
    return _envelope_flux(psi, op, iz, ir, axis)


def _bilinear(field: FloatArray, rc: FloatArray) -> FloatArray:
    """Sample `field` at fractional (row, col) coordinates."""
    r0 = np.clip(np.floor(rc[:, 0]).astype(int), 0, field.shape[0] - 2)
    c0 = np.clip(np.floor(rc[:, 1]).astype(int), 0, field.shape[1] - 2)
    fr, fc = rc[:, 0] - r0, rc[:, 1] - c0
    return (field[r0, c0] * (1 - fr) * (1 - fc) + field[r0 + 1, c0] * fr * (1 - fc)
            + field[r0, c0 + 1] * (1 - fr) * fc + field[r0 + 1, c0 + 1] * fr * fc)


def _component(psi: FloatArray, iz: int, ir: int, level: float) -> BoolArray:
    """The connected component of `psi >= level` that contains the axis."""
    lab, _n = label(psi >= level)
    return np.asarray(lab == lab[iz, ir], dtype=bool)


def _envelope_flux(psi: FloatArray, op: Operator, iz: int, ir: int, axis: float) -> float:
    """The highest level at which the plasma still fits inside the vessel envelope, by bisection."""
    lo, hi = float(psi.min()), axis
    for _ in range(BISECT_STEPS):
        mid = 0.5 * (lo + hi)
        if (_component(psi, iz, ir, mid) & ~op.mask).any():
            lo = mid
        else:
            hi = mid
    return hi


# How many candidate levels the boundary search tries before giving up.
N_CANDIDATES = 8

# Bisection steps for the vessel fallback; 24 halvings resolve a flux span to 6e-8 of itself.
BISECT_STEPS = 24


DEFAULT_PROFILE = Profile()


def midplane_pressure(psi_row: FloatArray, grid_R: FloatArray, psi_axis: float, span: float,
                      chord_R: FloatArray, te_ev: FloatArray, ne_m3: FloatArray,
                      nodes: FloatArray) -> FloatArray:
    """Thomson's `p(R)` resampled onto psi_N, from one row of the flux map.

    The laser runs horizontally through the machine, so it crosses the plasma twice and the same
    psi_N is measured on the inboard and the outboard side. Both are kept: that is what makes this
    a profile measurement rather than a chord one.

    A dead channel is `nan` on MAST — the opposite of DIII-D, where a missing sample is an exact
    zero — so the mask is `isfinite` and not `> 0`.
    """
    if not (chord_R.shape == te_ev.shape == ne_m3.shape):
        raise ValueError(f"Thomson arrays disagree: R {chord_R.shape}, Te {te_ev.shape}, "
                         f"ne {ne_m3.shape}")
    if span == 0:
        raise ValueError("the axis and boundary flux are equal; psi_N is undefined")
    psi_n = np.clip((psi_axis - np.interp(chord_R, grid_R, psi_row)) / span, 0.0, 1.0)
    live = np.isfinite(te_ev) & np.isfinite(ne_m3) & (te_ev > 0) & (ne_m3 > 0)
    if not live.any():
        return np.zeros_like(nodes)
    p = PRESSURE_SPECIES * ne_m3 * te_ev * ELEMENTARY_CHARGE
    order = np.argsort(psi_n[live])
    x, y = psi_n[live][order], p[live][order]
    out: FloatArray = np.interp(nodes, x, y, left=float(y[0]), right=0.0)
    return out


def solve(psi_coil: FloatArray, ip_amps: float, op: Operator,
          profile: Profile = DEFAULT_PROFILE,
          n_iter: int = 25, relax: float = 0.5, tol: float = 1e-4,
          thomson: tuple | None = None) -> Solved:
    """One frame: the coil flux and the plasma current in, the total flux out.

    `ip_amps` is signed. A negative Ip is a plasma driven the other way round and everything below
    is linear in it except the boundary search, which reads psi through the machine's own sign
    times sign(Ip) — so the axis stays a maximum of what is searched whichever way it runs.

    **The seed is a blob and stays one.** Warm-starting each frame from the previous frame's
    converged current is the obvious improvement — a discharge evolves on tens of milliseconds and
    the frames are 5 ms apart — and it was built and measured: **R2_psi 0.916 -> 0.443 and the
    composite 0.609 -> 0.342**. A converged current is sharply concentrated, and starting the
    boundary search inside it keeps the plasma there; the broad blob lets the solution find its own
    extent. Recorded here so it is not rebuilt on the same reasoning.
    """
    if psi_coil.shape != (op.grid_Z.size, op.grid_R.size):
        raise ValueError(f"coil flux {psi_coil.shape} against grid "
                         f"{(op.grid_Z.size, op.grid_R.size)}")
    if not np.isfinite(psi_coil).all():
        raise ValueError(f"coil flux carries {int((~np.isfinite(psi_coil)).sum())} "
                         f"non-finite cells")
    sign = op.machine.sign
    flip = float(np.sign(ip_amps)) or 1.0
    rr, zz = np.meshgrid(op.grid_R, op.grid_Z)
    nr_i, nz_i = op.grid_R.size - 2, op.grid_Z.size - 2

    psi_p = np.zeros_like(psi_coil)
    moved = np.inf
    it = 0
    j_phi = np.zeros_like(psi_coil)
    inside = op.mask.copy()
    psi = psi_coil.copy()
    psi_axis = psi_bnd = 0.0
    r_axis, z_axis = op.seed_R, op.seed_Z
    for it in range(1, n_iter + 1):
        psi = psi_coil + psi_p
        # Search on the quantity whose axis is a maximum: MAST's stored sign times the current's.
        look = flip * sign * psi
        if it == 1:
            # The vacuum field alone has no O-point inside the vessel — its extremum sits ON the
            # envelope rim, where axis and boundary are the same number and psi_N is 0/0. There is
            # nothing wrong with that; a magnetic well is made BY the plasma current, so the first
            # source cannot be read off psi. Failing that, it is a broad blob on the envelope
            # carrying the measured Ip, which opens a well the iteration then moves to where it
            # belongs — deliberately crude, since if the answer depended on it the Picard loop
            # would not have converged, and `moved` is what says whether it did.
            j_unit = np.where(op.mask, np.exp(-(((rr - op.seed_R) / op.seed_a) ** 2
                                                + ((zz - op.seed_Z) / op.seed_b) ** 2)), 0.0)
        else:
            r_axis, z_axis, iz, ir, _local = find_o_point(look, op.grid_R, op.grid_Z,
                                                          op.axis_mask)
            psi_axis = float(look[iz, ir])
            psi_bnd = _boundary_flux(look, op, int(iz), int(ir))
            span = psi_axis - psi_bnd
            if not np.isfinite(span) or span <= 0:
                raise ValueError(f"iteration {it}: the axis is not above the boundary "
                                 f"(axis {psi_axis:.5g}, boundary {psi_bnd:.5g})")
            # `edge` moves the surface the source is switched off at, and psi_N is normalised to
            # THAT surface — otherwise the knob is a no-op, because psi_N already reaches 1 at the
            # last closed surface and the shape factor is zero there however far `inside` extends.
            # Measured as exactly that: S identical to four decimals for edge 0.00 to 0.20.
            edge = psi_bnd - profile.edge * span
            psi_n = np.clip((psi_axis - look) / (psi_axis - edge), 0.0, 1.0)
            inside = (look >= edge) & op.mask
            shape = np.where(inside, (1.0 - psi_n ** profile.alpha) ** profile.gamma, 0.0)
            if profile.p_scale > 0.0:
                if thomson is None:
                    raise ValueError("profile.p_scale > 0 but no Thomson profile was passed; the "
                                     "pressure term has nothing to be driven by")
                chord_R, te_ev, ne_m3, nodes = thomson
                p_of_n = profile.p_scale * midplane_pressure(
                    look[iz], op.grid_R, psi_axis, span, chord_R, te_ev, ne_m3, nodes)
                # dp/dpsi from dp/dpsi_N. psi_N falls as psi rises, so the minus sign is what makes
                # p' positive — a pressure that peaks on the axis drives current in the same
                # direction as Ip, which is the sign check this term has to pass.
                p_prime = np.interp(psi_n, nodes, -np.gradient(p_of_n, nodes) / span)
                j_press = np.where(inside, rr * p_prime, 0.0)
                j_para = np.where(inside, shape / rr, 0.0)
                # The pressure term carries what it carries; the poloidal-current term takes the
                # remainder of the measured Ip. That is the whole point: nothing about the pressure
                # is fitted per frame any more.
                left = ip_amps - float(j_press.sum()) * op.dA
                norm_para = float(j_para.sum()) * op.dA
                if norm_para == 0:
                    raise ValueError(f"iteration {it}: the parametric term integrates to zero")
                j_unit = j_press + j_para * (left / norm_para)
            else:
                radial = profile.beta * rr / r_axis + (1.0 - profile.beta) * r_axis / rr
                j_unit = shape * radial
        if op.machine.z_pin is not None and op.machine.z_pin_current and it > 1 \
                and float(j_unit.sum()) > 0:
            # Put the CURRENT's own centroid on the pin. Bounding where the O-point may be found
            # does not bound where the current sits, and on DIII-D the two part company: with a
            # band of +-0.15 m the axis still comes out at dZ = -0.173 +- 0.055 m, pressed against
            # the band's lower edge. A real tokamak holds the plasma with feedback on exactly this
            # quantity, and EFIT pins it with magnetics this challenge withholds.
            zc = float((j_unit * zz).sum() / float(j_unit.sum()))
            rows = (op.machine.z_pin - zc) / (op.grid_Z[1] - op.grid_Z[0])
            j_unit = shift(j_unit, (rows, 0.0), order=1, mode="constant", cval=0.0)
        norm = float(j_unit.sum()) * op.dA
        if norm <= 0:
            raise ValueError(f"iteration {it}: the boundary encloses no current-carrying cell")
        trial = j_unit * (ip_amps / norm)

        # Relax the SOURCE, not the flux. The boundary is chosen from a discrete candidate list
        # every iteration — a centre-column touch or one of a handful of X-points — so the trial
        # current can JUMP between iterations even when psi has barely moved. Damping psi after
        # the solve leaves that jump in the loop and it oscillates; damping the source removes it
        # before the solve, and is the standard fix. Measured on the demo shots: the fraction of
        # frames converging to 1e-4 within 60 iterations goes from under a half to essentially all.
        w = 1.0 if it == 1 else relax
        j_phi = (1.0 - w) * j_phi + w * trial

        # Delta* psi_p = -mu0 R J, with the rim held at the free-space flux of this same J.
        rim = op.rim_green @ (j_phi.ravel() * op.dA)
        rhs = (-MU0 * rr * j_phi * sign)[1:-1, 1:-1].copy()
        rim_map = np.zeros_like(psi_coil)
        rim_map[op.rim_index] = rim * sign
        dZ = op.grid_Z[1] - op.grid_Z[0]
        dR = op.grid_R[1] - op.grid_R[0]
        r_in = op.grid_R[1:-1]
        rhs[0, :] -= rim_map[0, 1:-1] / dZ**2
        rhs[-1, :] -= rim_map[-1, 1:-1] / dZ**2
        rhs[:, 0] -= rim_map[1:-1, 0] * (1.0 / dR**2 + 1.0 / (2.0 * dR * r_in))
        rhs[:, -1] -= rim_map[1:-1, -1] * (1.0 / dR**2 - 1.0 / (2.0 * dR * r_in))

        new = np.zeros_like(psi_coil)
        new[1:-1, 1:-1] = op.lu.solve(rhs.ravel()).reshape(nz_i, nr_i)
        new[op.rim_index] = rim_map[op.rim_index]
        scale = max(float(np.abs(new).max()), 1e-30)
        moved = float(np.abs(new - psi_p).max()) / scale
        psi_p = new
        if it > 1 and moved < tol:
            break

    psi = psi_coil + psi_p
    return Solved(psi, sign * flip * psi_axis, sign * flip * psi_bnd, r_axis, z_axis,
                  inside, j_phi, it, moved)
