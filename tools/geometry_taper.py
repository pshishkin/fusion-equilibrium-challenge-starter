"""Boundary-zero C1 taper for the separately preregistered contour-warp probe."""

from __future__ import annotations

import numpy as np


def contour_margin(contour: np.ndarray, R: np.ndarray, Z: np.ndarray) -> float:
    """Half the closest physical contour-to-grid-edge distance, in metres."""
    if contour.ndim != 2 or contour.shape[1] != 2 or not np.isfinite(contour).all():
        raise ValueError(f"contour shape/finiteness {contour.shape}, expected finite (N,2)")
    distances = np.column_stack(
        [
            contour[:, 0] - R[0],
            R[-1] - contour[:, 0],
            contour[:, 1] - Z[0],
            Z[-1] - contour[:, 1],
        ]
    )
    smallest = float(np.min(distances))
    if not np.isfinite(smallest) or smallest <= 0:
        raise ValueError(f"predicted contour meets/leaves the grid, min edge distance {smallest}")
    return 0.5 * smallest


def boundary_taper(
    points_unit: np.ndarray, lo: np.ndarray, span: np.ndarray, margin: float
) -> np.ndarray:
    """Product of four fixed smoothsteps: zero at every grid edge, one past margin."""
    if points_unit.ndim != 2 or points_unit.shape[1] != 2:
        raise ValueError(f"unit points shape {points_unit.shape}, expected (N,2)")
    if not np.isfinite(points_unit).all() or not np.isfinite(margin) or margin <= 0:
        raise ValueError(f"unit points/margin not finite and positive; margin {margin}")
    physical = lo + points_unit * span
    distance = np.column_stack(
        [
            physical[:, 0] - lo[0],
            lo[0] + span[0] - physical[:, 0],
            physical[:, 1] - lo[1],
            lo[1] + span[1] - physical[:, 1],
        ]
    )
    t = np.clip(distance / margin, 0.0, 1.0)
    h = t * t * (3.0 - 2.0 * t)
    return np.prod(h, axis=1)
