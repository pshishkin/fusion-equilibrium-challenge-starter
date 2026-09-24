"""Thomson features aimed at plasma GEOMETRY rather than at pressure, per EFIT frame.

The core system is a vertical chord at R = 1.94 m whose channels span roughly Z 0.03..0.83 m, so
the height at which Te and ne collapse is where the upper LCFS crosses that chord. The tangential
system crosses the outboard midplane. The production summaries (peak, peak position, integrals,
slope, width) do not locate that collapse. Inputs only: no `efit_*` column is read here.
"""
from __future__ import annotations

from typing import Any

import numpy as np

TE_THR = (30.0, 50.0, 100.0, 200.0, 500.0)
NE_THR = (0.5e19, 1e19, 2e19, 4e19)
Z_GRID = np.arange(0.0, 0.851, 0.05)
MAX_DT_MS = 10.0


def _nearest(times: np.ndarray, efit_times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    idx = np.clip(np.searchsorted(times, efit_times), 1, len(times) - 1)
    left = times[idx - 1]
    right = times[idx]
    pick = np.where(np.abs(efit_times - left) <= np.abs(right - efit_times), idx - 1, idx)
    return pick, np.abs(times[pick] - efit_times)


def _top_crossing(values: np.ndarray, coord: np.ndarray, thr: float) -> float:
    above = values > thr
    return float(coord[above].max()) if above.any() else np.nan


def frame_features(row: Any) -> np.ndarray:
    efit_times = np.asarray(row["efit_times"], dtype=np.float64)
    names = [str(x) for x in np.asarray(row["thomson_chord_name"])]
    chord_R = np.asarray(row["thomson_chord_R"], dtype=np.float64)
    chord_Z = np.asarray(row["thomson_chord_Z"], dtype=np.float64)
    out = []
    for key, prefix, coords, grid in (("thomson_core", "TS_core", chord_Z, Z_GRID),
                                      ("thomson_edge", "TS_tangential", chord_R, None)):
        te = np.stack([np.asarray(x, dtype=np.float64) for x in row[f"{key}_Te"]])
        ne = np.stack([np.asarray(x, dtype=np.float64) for x in row[f"{key}_ne"]])
        times = np.asarray(row[f"{key}_times"], dtype=np.float64)
        start = 0 if prefix == "TS_core" else len(np.asarray(row["thomson_core_Te"][0]))
        labels = names[start:start + te.shape[1]]
        if not all(n.startswith(prefix) for n in labels):
            raise ValueError(f"chords {start}.. are {labels[:3]}, expected {prefix}*")
        c = coords[start:start + te.shape[1]]
        order = np.argsort(c)
        c, te, ne = c[order], te[:, order], ne[:, order]
        pick, dt = _nearest(times, efit_times)
        n_grid = len(grid) if grid is not None else te.shape[1]
        block = np.full((len(efit_times), 3 + len(TE_THR) + len(NE_THR) + 2 * n_grid), np.nan)
        for k, (i, d) in enumerate(zip(pick, dt, strict=True)):
            live = te[i] > 0
            block[k, 0] = d
            block[k, 1] = live.sum()
            if d > MAX_DT_MS or live.sum() < 3:
                continue
            cl, tl, nl = c[live], te[i, live], ne[i, live]
            block[k, 2] = cl.max()
            col = 3
            for thr in TE_THR:
                block[k, col] = _top_crossing(tl, cl, thr)
                col += 1
            for thr in NE_THR:
                block[k, col] = _top_crossing(nl, cl, thr)
                col += 1
            lt, ln = np.log10(np.maximum(tl, 1.0)), np.log10(np.maximum(nl, 1e17))
            if grid is not None:
                inside = (grid >= cl.min()) & (grid <= cl.max())
                block[k, col:col + n_grid][inside] = np.interp(grid[inside], cl, lt)
                block[k, col + n_grid:col + 2 * n_grid][inside] = np.interp(grid[inside], cl, ln)
            else:
                full_t = np.full(n_grid, np.nan)
                full_n = np.full(n_grid, np.nan)
                full_t[live] = lt
                full_n[live] = ln
                block[k, col:col + n_grid] = full_t
                block[k, col + n_grid:col + 2 * n_grid] = full_n
        out.append(block)
    return np.hstack(out)
