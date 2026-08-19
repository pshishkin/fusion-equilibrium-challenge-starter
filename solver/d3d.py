#!/usr/bin/env python3
"""DIII-D, read for the SOLVER — the machine with 7041 labelled shots to test choices on.

This is not Challenge 1. Challenge 1 is won by a fitted model in `my_experiments/`, which this file
neither imports nor competes with. DIII-D is here for one reason: three MAST demo shots have now
refuted three modelling decisions that were fitted on them, the last one with the sign reversed, so
the only way to choose anything for MAST is to test it where the truth exists in quantity.

What differs from `mast/shot.py`, and it is all bookkeeping:

* **The plasma current has its own clock.** `magnetics_plasma_current_times`, 30719 samples,
  against `magnetics_time`'s 480256 for every coil. This fork has already been bitten by that once
  — the three-second offset of the plasma-current axis — so the two are interpolated separately
  and never assumed to share a base.
* **One conductor row per coil**, 19 of them, against MAST's 812 turn-by-turn rows.
* **The stored flux has the opposite sign**: `AXIS_SIGN["DIII-D"] = -1`, so the axis is a MINIMUM
  of `efit_psirz`. Nothing here handles that — `solver/gs.py` reads it off `machine.sign`, and
  running this file is the test of whether it does so correctly.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from solver.machine import D3D

FloatArray = npt.NDArray[np.floating]


@dataclass(frozen=True)
class D3DShot:
    times: FloatArray          # (T,) ms, the EFIT frame clock
    currents: FloatArray       # (T, C) coil currents, in `columns` order
    columns: list[str]
    ip: FloatArray             # (T,) amperes, signed
    tf: FloatArray             # (T,) toroidal field coil feed
    grid_R: FloatArray
    grid_Z: FloatArray
    psi: FloatArray | None     # (T, nZ, nR) truth, when the config ships it
    lcfs: list | None          # (T,) of (N, 2) true boundary polygons, likewise


def psi_stack(cell: Any) -> FloatArray:
    return np.stack([np.stack([np.asarray(c, dtype=np.float64) for c in np.asarray(f)])
                     for f in np.asarray(cell)])


def _onto(times: FloatArray, base: FloatArray, values: FloatArray, what: str) -> FloatArray:
    """One signal on the EFIT frames, over its own live samples, with coverage required."""
    if values.shape != base.shape:
        raise ValueError(f"{what} has {values.shape} samples against {base.shape} times")
    live = np.isfinite(values)
    if not live.any():
        raise ValueError(f"{what} is nan on every one of its {values.size} samples")
    lo, hi = base[live][0], base[live][-1]
    if times[0] < lo or times[-1] > hi:
        raise ValueError(f"EFIT frames span {times[0]:.1f}..{times[-1]:.1f} ms but {what} is live "
                         f"only over {lo:.1f}..{hi:.1f} ms")
    out: FloatArray = np.interp(times, base[live], values[live])
    return out


def read(row: Any, columns: list[str], with_truth: bool = True) -> D3DShot:
    """Interpolate the named coil columns, Ip and the TF feed onto this shot's own EFIT frames."""
    source = str(row["source"])
    if source != D3D.name:
        raise ValueError(f"row is from {source!r}, not {D3D.name!r}")
    times = np.asarray(row["efit_times"], dtype=np.float64)
    base = np.asarray(row["magnetics_time"], dtype=np.float64)
    if not np.all(np.diff(base) > 0):
        raise ValueError("magnetics_time is not strictly increasing")

    cur = np.column_stack([_onto(times, base, np.asarray(row[c], dtype=np.float64), c)
                           for c in columns])
    ip_base = np.asarray(row[D3D.ip_times], dtype=np.float64)
    ip = _onto(times, ip_base, np.asarray(row[D3D.ip_column], dtype=np.float64),
               D3D.ip_column) * D3D.current_unit
    tf = _onto(times, base, np.asarray(row[D3D.tf_column], dtype=np.float64), D3D.tf_column)
    psi = psi_stack(row["efit_psirz"]) if with_truth and "efit_psirz" in row else None
    if psi is not None and psi.shape[0] != times.size:
        raise ValueError(f"{psi.shape[0]} flux maps against {times.size} EFIT frames")
    lcfs = None
    if psi is not None:
        n = np.asarray(row["efit_lcfs_n"], dtype=np.int64)
        lr, lz = np.asarray(row["efit_lcfs_r"]), np.asarray(row["efit_lcfs_z"])
        lcfs = [np.column_stack([np.asarray(lr[k], dtype=np.float64)[:int(n[k])],
                                 np.asarray(lz[k], dtype=np.float64)[:int(n[k])]])
                for k in range(times.size)]
    return D3DShot(times, cur, list(columns), ip, tf,
                   np.asarray(row["efit_grid_R"], dtype=np.float64),
                   np.asarray(row["efit_grid_Z"], dtype=np.float64), psi, lcfs)


def outside_lcfs(shot: D3DShot, keep: npt.NDArray[np.bool_]) -> npt.NDArray[np.bool_]:
    """(n_live, nZ, nR) — grid cells OUTSIDE the shipped plasma boundary, frame by frame.

    Where the coil gains have to be fitted, and this fork already paid to learn it: inside the
    boundary the flux is dominated by a distributed plasma current that a single filament cannot
    represent, so a fit there launders the plasma into the coils. Outside it the plasma looks like
    one filament and the coils are identifiable — which is why `my_experiments/coil_field.py` gets
    0.87-0.89 for DIII-D's F-coils and a fit over the whole grid returns -1.4 to +2.8.

    Even-odd ray casting, vectorised over the grid, so a 65x65 frame costs one pass per polygon
    edge and no dependency.
    """
    if shot.lcfs is None:
        raise ValueError("this shot has no efit_lcfs; the calibration needs the training config")
    rr, zz = np.meshgrid(shot.grid_R, shot.grid_Z)
    out = np.empty((int(keep.sum()), *rr.shape), dtype=bool)
    for i, k in enumerate(np.flatnonzero(keep)):
        poly = shot.lcfs[k]
        r1, z1 = poly[:, 0], poly[:, 1]
        r2, z2 = np.roll(r1, -1), np.roll(z1, -1)
        inside = np.zeros(rr.shape, dtype=bool)
        for a, b, c, d in zip(r1, z1, r2, z2, strict=True):
            straddles = (b > zz) != (d > zz)
            if not straddles.any():
                continue
            cut = a + (zz - b) * (c - a) / (d - b)
            inside ^= straddles & (rr < cut)
        out[i] = ~inside
    return out
