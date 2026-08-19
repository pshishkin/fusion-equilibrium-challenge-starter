#!/usr/bin/env python3
"""One MAST row, read into the handful of arrays the solver needs.

The DIII-D pipeline builds an 84-column feature vector. There is nothing to fit on MAST, so there
are no features here at all — only the physical inputs of a free-boundary solve:

* the coil currents, on the EFIT frame clock;
* the plasma current, likewise;
* the flux grid, which is the machine's own and comes off the row.

**The two time bases do not agree and nothing in the dataset says so.** MAST's magnetics run from
-2000 ms to +4000 ms at 0.2 ms, while `efit_times` covers a few hundred milliseconds in the middle
of that at 5 ms. Every signal is therefore interpolated onto `efit_times` here and nowhere else, so
there is exactly one place where a frame can end up reading the wrong instant.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from solver.machine import MAST

FloatArray = npt.NDArray[np.floating]

# The machine's own names and units, so that this file and the solver cannot drift apart about
# which column is the plasma current. kA on every MAST current column, `magnetics_plasma_current`
# included — confirmed by the check that settled the units for good: integrating
# `-Delta* psi / (mu0 R)` over the cells inside the shipped LCFS of the demo shots returns
# 532 / 633 / 660 / 635 kA against a shipped Ip of 496 / 609 / 637 / 618 kA, the same quantity to
# 5-7%. That test fixes the units of psi (Wb/rad), of Ip (kA) and the sign convention at once.
MACHINE = MAST.name
IP_COLUMN = MAST.ip_column
TF_COLUMN = MAST.tf_column
CURRENT_UNIT = MAST.current_unit


@dataclass(frozen=True)
class Shot:
    times: FloatArray          # (T,) ms, the EFIT frame clock
    currents: FloatArray       # (T, C) coil currents in kA, in `columns` order
    columns: list[str]
    ip: FloatArray             # (T,) amperes, signed
    grid_R: FloatArray
    grid_Z: FloatArray
    tf: FloatArray             # (T,) toroidal field coil feed, kA
    ts_R: FloatArray           # (K,) Thomson channel major radius, metres — MAST's laser is
    ts_te: FloatArray          # (T, K) eV, nan on a dead channel        midplane, so R is all
    ts_ne: FloatArray          # (T, K) m^-3, nan on a dead channel      the geometry there is


def read(row: Any, columns: list[str]) -> Shot:
    """Interpolate the named current columns and Ip onto this shot's own EFIT frames."""
    source = str(row["source"])
    if source != MACHINE:
        raise ValueError(f"source is {source!r}, not {MACHINE!r} — mast/ is MAST only")
    for col in ("efit_times", "magnetics_time", IP_COLUMN, TF_COLUMN,
                "efit_grid_R", "efit_grid_Z", "thomson_core_R", "thomson_edge_spatial",
                "thomson_core_times", "thomson_core_Te", "thomson_core_ne",
                "thomson_edge_times", "thomson_edge_Te", "thomson_edge_ne"):
        if col not in row:
            raise ValueError(f"no column {col!r} on this MAST row")

    times = np.asarray(row["efit_times"], dtype=np.float64)
    base: FloatArray = np.asarray(row["magnetics_time"], dtype=np.float64)
    if times.size == 0:
        raise ValueError("this row carries no EFIT frames")
    base, keep = _monotone(base)

    cur = np.empty((times.size, len(columns)), dtype=np.float64)
    for i, col in enumerate(columns):
        if col not in row:
            raise ValueError(f"no column {col!r}: the coil table names it but the row\n"
                             f"carries no such signal")
        cur[:, i] = _on_frames(times, base, row, col, keep)

    ip = _on_frames(times, base, row, IP_COLUMN, keep) * CURRENT_UNIT
    tf = _on_frames(times, base, row, TF_COLUMN, keep)
    ts_R, ts_te, ts_ne = thomson(row, times)
    return Shot(times, cur, list(columns), ip,
                np.asarray(row["efit_grid_R"], dtype=np.float64),
                np.asarray(row["efit_grid_Z"], dtype=np.float64),
                tf, ts_R, ts_te, ts_ne)


def _on_frames(times: FloatArray, base: FloatArray, row: Any, col: str,
               keep: npt.NDArray[np.bool_]) -> FloatArray:
    """One magnetics signal on the EFIT frames, interpolated over its own live samples.

    **The second shipped defect, and unlike the repeated timestamps it is per SIGNAL.** MAST's
    magnetics come in two populations — 30000 samples and 15482 — and in the second, a signal is
    padded with `nan` outside the window it was actually acquired in. On
    `mast_shot_018ab61eba` the coil currents are finite only from -150.0 to 1349.8 ms of a record
    that spans -2500 to 5499, while `magnetics_tf_current` is finite across the whole record but
    with nan scattered THROUGH it — a different sampling rate padded to a common length.

    So `nan` here means "not sampled", and interpolating between the samples that exist is the
    correct reading rather than a repair. What stays an exception is coverage: if the EFIT frames
    fall outside the signal's live span, interpolation would clamp to an endpoint and report a
    constant with no warning, and that is the failure this check exists for.
    """
    v = np.asarray(row[col], dtype=np.float64)
    if v.shape != keep.shape:
        raise ValueError(f"{col} has {v.shape} samples against {keep.shape} times")
    live = np.isfinite(v[keep])
    if not live.any():
        raise ValueError(f"{col} is nan on every one of its {int(keep.sum())} samples")
    lo, hi = base[live][0], base[live][-1]
    if times[0] < lo or times[-1] > hi:
        raise ValueError(f"EFIT frames span {times[0]:.1f}..{times[-1]:.1f} ms but {col} is live "
                         f"only over {lo:.1f}..{hi:.1f} ms")
    out: FloatArray = np.interp(times, base[live], v[keep][live])
    return out


def _monotone(base: FloatArray) -> tuple[FloatArray, npt.NDArray[np.bool_]]:
    """The magnetics clock with its repeated samples dropped, and the mask that did it.

    **A real defect in the shipped data, and a narrow one.** 9 of the first 120 `mast_public_test`
    shots carry a `magnetics_time` that is not strictly increasing — all of them in the
    15482-sample population (the other has 30000), all with exactly **five repeated timestamps at
    the same five indices** (5102, 9760, 9898, 10510, 10648), and every one an exact duplicate:
    `dt = 0`, never negative. So the clock is not scrambled and the samples are not out of order;
    a handful are written twice.

    Dropping the repeats is therefore a repair with a name, not a defensive `np.unique` over
    whatever arrives: a genuinely DECREASING step would mean two spliced records and is still an
    exception, because interpolating across one would read the wrong instant and say nothing.
    """
    d = np.diff(base)
    if (d < 0).any():
        k = int(np.flatnonzero(d < 0)[0])
        raise ValueError(f"magnetics_time goes BACKWARDS at sample {k}: {base[k]:.4f} -> "
                         f"{base[k + 1]:.4f}; that is two spliced records, not a repeated sample")
    keep = np.ones(base.size, dtype=bool)
    keep[1:] = d > 0
    return base[keep], keep


def _profiles(cell: Any) -> FloatArray:
    """An object array of per-pulse channel vectors, as (n_pulses, n_channels)."""
    return np.stack([np.asarray(a, dtype=np.float64) for a in np.asarray(cell)])


def thomson(row: Any, times: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
    """The midplane Te and ne profiles, on the EFIT frames. `(K,)`, `(T, K)`, `(T, K)`.

    Three things about MAST's Thomson that DIII-D's does not have, and each of them bites:

    * **The core arrays carry one channel more than the core radii do** — 131 against 130 — and it
      is the FIRST, not the last: channel 0 is nan on 100% of the 191 demo frames while every other
      channel is live 19% to 95% of the time, and the busiest frame of each shot has exactly 130
      finite values. So the profiles are read from index 1 on. Asserted rather than trusted, since
      a silent one-channel shift would tilt every pressure profile.
    * **A dead channel is `nan`, not `0.0`.** The opposite of DIII-D, where missing samples are
      exact zeros, so a mask written for one machine reads the other backwards.
    * **The laser fires on its own clock**, ~2 ms against EFIT's 5 ms, and covers 6.7-254 ms. Each
      EFIT frame takes the NEAREST laser pulse rather than an interpolation between two: the
      profile is a set of channels that are independently alive or dead, and interpolating between
      two different sets of live channels invents measurements.
    """
    r_core = np.asarray(row["thomson_core_R"], dtype=np.float64)
    r_edge = np.asarray(row["thomson_edge_spatial"], dtype=np.float64)
    out_te, out_ne = [], []
    for prefix, radii, skip in (("core", r_core, 1), ("edge", r_edge, 0)):
        t_ts = np.asarray(row[f"thomson_{prefix}_times"], dtype=np.float64)
        te = _profiles(row[f"thomson_{prefix}_Te"])
        ne = _profiles(row[f"thomson_{prefix}_ne"])
        if te.shape != ne.shape:
            raise ValueError(f"{prefix} Te {te.shape} and ne {ne.shape} disagree")
        if te.shape[1] != radii.size + skip:
            raise ValueError(f"{prefix} carries {te.shape[1]} channels against {radii.size} radii "
                             f"and an expected offset of {skip}")
        if skip and np.isfinite(te[:, 0]).any():
            raise ValueError(f"{prefix} channel 0 is supposed to be the unmatched one and is nan "
                             f"throughout, but {int(np.isfinite(te[:, 0]).sum())} frames have it")
        if te.shape[0] != t_ts.size:
            raise ValueError(f"{prefix} has {te.shape[0]} profiles against {t_ts.size} times")
        nearest = np.abs(times[:, None] - t_ts[None, :]).argmin(axis=1)
        out_te.append(te[nearest, skip:])
        out_ne.append(ne[nearest, skip:])
    return (np.concatenate([r_core, r_edge]),
            np.hstack(out_te), np.hstack(out_ne))
