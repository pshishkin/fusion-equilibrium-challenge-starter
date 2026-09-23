"""Input-only components for the reviewed contour-displacement screen.

This module contains no competition-data training entry point. The target encoder
accepts a true contour for a future supervised fit; inference uses only the source
contour, permitted shot inputs, and the fitted coefficient predictor.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.spatial import cKDTree
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from my_experiments.baseline_model import FEATURE_WIDTH, features_for_row, slim_row
from tools.geometry_map_smoke import (
    NumericalInfeasibility,
    _cyclic_match,
    _edge_anchors,
    _simple_closed,
)
from tools.geometry_taper import boundary_taper, contour_margin

N_CONTOUR = 512
N_MODES = 8
N_COEFFICIENTS = 1 + 2 * N_MODES
N_SUMMARIES = 7
N_INPUTS = FEATURE_WIDTH + N_SUMMARIES + 1
RIDGE_ALPHA = 35.0


def canonical_contour(contour: np.ndarray) -> tuple[np.ndarray, int]:
    """Start the physical contour at its outboard point, without a radial model."""
    c = np.asarray(contour, dtype=np.float64)
    if c.shape != (N_CONTOUR, 2) or not np.isfinite(c).all():
        raise ValueError(f"predicted contour shape/finiteness {c.shape}/"
                         f"{int((~np.isfinite(c)).sum())}, expected (512,2)/0")
    if not _simple_closed(c):
        raise ValueError("predicted contour is non-simple or degenerate")
    start = int(np.argmax(c[:, 0]))
    return np.roll(c, -start, axis=0), start


def outward_normals(contour: np.ndarray) -> np.ndarray:
    """Unit physical normals using polygon orientation and central arc-length steps."""
    c = np.asarray(contour, dtype=np.float64)
    if c.shape != (N_CONTOUR, 2) or not np.isfinite(c).all():
        raise ValueError(f"contour shape/finiteness {c.shape}/"
                         f"{int((~np.isfinite(c)).sum())}, expected (512,2)/0")
    nxt = np.roll(c, -1, axis=0)
    prv = np.roll(c, 1, axis=0)
    area2 = float(np.sum(c[:, 0] * nxt[:, 1] - nxt[:, 0] * c[:, 1]))
    tangent = nxt - prv
    length = np.linalg.norm(tangent, axis=1)
    if not np.isfinite(area2) or area2 == 0.0 or np.any(length <= 0.0):
        raise ValueError("predicted contour has zero area or degenerate central tangent")
    sign = np.sign(area2)
    normals = sign * np.column_stack([tangent[:, 1], -tangent[:, 0]]) / length[:, None]
    if not np.isfinite(normals).all():
        raise ValueError("predicted contour has nonfinite outward normals")
    return normals


def encode_normal_target(pred: np.ndarray, truth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Encode the matched normal displacement; truth is used only in target building."""
    c, _ = canonical_contour(pred)
    matched, _ = _cyclic_match(c, truth)
    normal = outward_normals(c)
    displacement = np.sum((matched - c) * normal, axis=1)
    spectrum = np.fft.rfft(displacement) / N_CONTOUR
    coefficients = np.empty(N_COEFFICIENTS, dtype=np.float64)
    coefficients[0] = spectrum[0].real
    coefficients[1::2] = spectrum[1:N_MODES + 1].real
    coefficients[2::2] = spectrum[1:N_MODES + 1].imag
    return coefficients, displacement


def decode_normal_coefficients(coefficients: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Return a 512-by-2 physical displacement in original scorer indexing."""
    coeff = np.asarray(coefficients, dtype=np.float64)
    if coeff.shape != (N_COEFFICIENTS,) or not np.isfinite(coeff).all():
        raise ValueError(f"normal coefficients shape/finiteness {coeff.shape}/"
                         f"{int((~np.isfinite(coeff)).sum())}, expected (17,)/0")
    canonical, start = canonical_contour(pred)
    normal = outward_normals(canonical)
    spectrum = np.zeros(N_CONTOUR // 2 + 1, dtype=np.complex128)
    spectrum[0] = coeff[0] * N_CONTOUR
    spectrum[1:N_MODES + 1] = (
        coeff[1::2] + 1j * coeff[2::2]
    ) * N_CONTOUR
    scalar = np.fft.irfft(spectrum, n=N_CONTOUR)
    displacement = np.roll(scalar[:, None] * normal, start, axis=0)
    if not np.isfinite(displacement).all():
        raise ValueError("decoded normal displacement is nonfinite")
    return displacement


def contour_summaries(contour: np.ndarray) -> np.ndarray:
    """Seven cheap geometric features from the predicted polygon only."""
    c = np.asarray(contour, dtype=np.float64)
    if c.shape != (N_CONTOUR, 2) or not np.isfinite(c).all():
        raise ValueError(f"predicted contour shape/finiteness {c.shape}/"
                         f"{int((~np.isfinite(c)).sum())}, expected (512,2)/0")
    if not _simple_closed(c):
        raise ValueError("predicted contour is non-simple or degenerate")
    nxt = np.roll(c, -1, axis=0)
    cross = c[:, 0] * nxt[:, 1] - nxt[:, 0] * c[:, 1]
    area2 = float(cross.sum())
    if not np.isfinite(area2) or area2 == 0.0:
        raise ValueError("predicted contour polygon has zero area")
    centroid_r = float(np.sum((c[:, 0] + nxt[:, 0]) * cross) / (3.0 * area2))
    out = np.array([
        c[:, 0].min(), c[:, 0].max(), c[:, 1].min(), c[:, 1].max(),
        abs(area2) / 2.0, np.linalg.norm(nxt - c, axis=1).sum(), centroid_r,
    ], dtype=np.float64)
    if not np.isfinite(out).all():
        raise ValueError("predicted contour summaries are nonfinite")
    return out


def allowed_input_features(row: Any, contours: np.ndarray) -> np.ndarray:
    """Build 216 columns via the existing target-stripping inference input path."""
    slim = slim_row(row)
    forbidden = [name for name in slim if name.startswith("efit_")
                 and name not in ("efit_times", "efit_grid_R", "efit_grid_Z")]
    if forbidden:
        raise ValueError(f"slim input unexpectedly contains EFIT targets {forbidden}")
    times = np.asarray(slim["efit_times"], dtype=np.float64)
    c = np.asarray(contours, dtype=np.float64)
    if times.ndim != 1 or c.shape != (len(times), N_CONTOUR, 2):
        raise ValueError(f"shot times/contours shape {times.shape}/{c.shape}, expected "
                         f"(T,)/(T,512,2)")
    if not np.isfinite(times).all() or (len(times) > 1 and np.any(np.diff(times) <= 0)):
        raise ValueError("shot efit_times must be finite and strictly increasing")
    phase = np.zeros(len(times), dtype=np.float64) if len(times) == 1 else (
        (times - times[0]) / (times[-1] - times[0])
    )
    base = np.asarray(features_for_row(slim), dtype=np.float64)
    if base.shape != (len(times), FEATURE_WIDTH) or not np.isfinite(base).all():
        raise ValueError(f"permitted base features shape/finiteness {base.shape}/"
                         f"{int((~np.isfinite(base)).sum())}, expected "
                         f"({len(times)},{FEATURE_WIDTH})/0")
    summaries = np.stack([contour_summaries(one) for one in c])
    result = np.column_stack([base, summaries, phase])
    if result.shape != (len(times), N_INPUTS) or not np.isfinite(result).all():
        raise ValueError(f"permitted inputs shape/finiteness {result.shape}/"
                         f"{int((~np.isfinite(result)).sum())}, expected "
                         f"({len(times)},{N_INPUTS})/0")
    return result


def per_shot_weights(lengths: list[int]) -> np.ndarray:
    """Give every training shot total weight one, regardless of its frame count."""
    if not lengths or any(length <= 0 for length in lengths):
        raise ValueError(f"shot lengths must be positive and nonempty, got {lengths}")
    return np.concatenate([np.full(length, 1.0 / length) for length in lengths])


class ContourRidge:
    """One fixed-alpha, weighted multioutput linear screen; no parameter search."""

    def __init__(self) -> None:
        self.scaler = StandardScaler()
        self.model = Ridge(alpha=RIDGE_ALPHA)
        self.fitted = False

    def fit(self, x: np.ndarray, y: np.ndarray, lengths: list[int]) -> ContourRidge:
        X = np.asarray(x, dtype=np.float64)
        Y = np.asarray(y, dtype=np.float64)
        if X.shape != (sum(lengths), N_INPUTS) or Y.shape != (len(X), N_COEFFICIENTS):
            raise ValueError(f"Ridge input/target shapes {X.shape}/{Y.shape}, expected "
                             f"({sum(lengths)},{N_INPUTS})/({sum(lengths)},{N_COEFFICIENTS})")
        if not np.isfinite(X).all() or not np.isfinite(Y).all():
            raise ValueError("Ridge input or supervised target is nonfinite")
        weight = per_shot_weights(lengths)
        self.scaler.fit(X, sample_weight=weight)
        scaled = self.scaler.transform(X)
        self.model.fit(scaled, Y, sample_weight=weight)
        self.fitted = True
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise ValueError("ContourRidge has not been fitted")
        X = np.asarray(x, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != N_INPUTS or not np.isfinite(X).all():
            raise ValueError(f"Ridge prediction input shape/finiteness {X.shape}/"
                             f"{int((~np.isfinite(X)).sum())}, expected (T,{N_INPUTS})/0")
        prediction = np.asarray(self.model.predict(self.scaler.transform(X)))
        if prediction.shape != (len(X), N_COEFFICIENTS) or not np.isfinite(prediction).all():
            raise ValueError("Ridge predicted coefficients have wrong shape or nonfinite values")
        return prediction


def correct_map_from_displacement(
    psi: np.ndarray,
    c_pred: np.ndarray,
    displacement: np.ndarray,
    axis: tuple[float, float],
    R: np.ndarray,
    Z: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    """Apply the old fixed additive formula to an input-only contour displacement."""
    if psi.shape != (65, 65) or psi.dtype != np.float32 or not np.isfinite(psi).all():
        raise NumericalInfeasibility(f"source psi shape/dtype/finiteness {psi.shape}/"
                                     f"{psi.dtype}/{int((~np.isfinite(psi)).sum())}, "
                                     "expected (65,65)/float32/0")
    source_contour = np.asarray(c_pred, dtype=np.float64)
    if source_contour.shape != (N_CONTOUR, 2) or not np.isfinite(source_contour).all():
        raise NumericalInfeasibility(f"source contour shape/finiteness "
                                     f"{source_contour.shape}/"
                                     f"{int((~np.isfinite(source_contour)).sum())}, "
                                     "expected (512,2)/0")
    R = np.asarray(R, dtype=np.float64)
    Z = np.asarray(Z, dtype=np.float64)
    if (R.shape != (65,) or Z.shape != (65,) or not np.isfinite(R).all()
            or not np.isfinite(Z).all() or np.any(np.diff(R) <= 0)
            or np.any(np.diff(Z) <= 0)):
        raise NumericalInfeasibility(f"physical grid shape/finiteness/order "
                                     f"{R.shape}/{Z.shape}, expected increasing (65,)/(65,)")
    movement_xy = np.asarray(displacement, dtype=np.float64)
    if movement_xy.shape != (N_CONTOUR, 2) or not np.isfinite(movement_xy).all():
        raise NumericalInfeasibility(f"contour displacement shape/finiteness "
                                     f"{movement_xy.shape}/"
                                     f"{int((~np.isfinite(movement_xy)).sum())}, "
                                     "expected (512,2)/0")
    if np.count_nonzero(movement_xy) == 0:
        return psi.copy(), {"zero_displacement_exact": 1.0}
    movement = np.linalg.norm(movement_xy, axis=1)
    lo = np.array([R[0], Z[0]])
    span = np.array([R[-1] - R[0], Z[-1] - Z[0]])
    selected = np.arange(0, N_CONTOUR, 8)
    controls = (source_contour[selected] - lo) / span
    target_displacement = movement_xy[selected] / span
    anchors = np.vstack([controls, (np.asarray(axis) - lo) / span, _edge_anchors()])
    displacements = np.vstack([target_displacement, np.zeros((17, 2))])
    if len(np.unique(anchors, axis=0)) != len(anchors):
        raise NumericalInfeasibility("TPS control points are not unique")
    try:
        rbf = RBFInterpolator(
            anchors, displacements, kernel="thin_plate_spline", smoothing=0.0001, degree=1
        )
    except (ValueError, np.linalg.LinAlgError) as exc:
        raise NumericalInfeasibility(f"TPS solve failed: {type(exc).__name__}") from exc
    try:
        margin = contour_margin(source_contour, R, Z)
        contour_taper = boundary_taper(controls, lo, span, margin)
        axis_taper = boundary_taper(((np.asarray(axis) - lo) / span)[None, :], lo, span, margin)
    except ValueError as exc:
        raise NumericalInfeasibility(f"boundary taper invalid: {exc}") from exc
    if not np.all(contour_taper == 1.0) or not np.all(axis_taper == 1.0):
        raise NumericalInfeasibility("contour/axis taper is not full strength")

    grid_r, grid_z = np.meshgrid(R, Z)
    physical = np.column_stack([grid_r.ravel(), grid_z.ravel()])
    unit = (physical - lo) / span
    taper = boundary_taper(unit, lo, span, margin)
    edge = np.isclose(unit[:, 0], 0) | np.isclose(unit[:, 0], 1)
    edge |= np.isclose(unit[:, 1], 0) | np.isclose(unit[:, 1], 1)
    if not np.all(taper[edge] == 0.0):
        raise NumericalInfeasibility("edge taper is nonzero on grid boundary")
    displacement_physical = rbf(unit) * taper[:, None] * span
    if not np.isfinite(displacement_physical).all():
        raise NumericalInfeasibility("TPS displacement is nonfinite")
    width = 2.0 * float(min(np.diff(R).min(), np.diff(Z).min()))
    if width <= 0:
        raise NumericalInfeasibility(f"contour shell width {width} must be positive")
    distance, _ = cKDTree(source_contour).query(physical, k=1)
    shell = np.exp(-0.5 * (distance / width) ** 2)
    dpsi_dz, dpsi_dr = np.gradient(psi.astype(np.float64), Z, R)
    if not np.isfinite(dpsi_dz).all() or not np.isfinite(dpsi_dr).all():
        raise NumericalInfeasibility("physical psi gradient is nonfinite")
    change = -shell * (
        displacement_physical[:, 0] * dpsi_dr.ravel()
        + displacement_physical[:, 1] * dpsi_dz.ravel()
    )
    result = (psi.astype(np.float64).ravel() + change).reshape(65, 65).astype(np.float32)
    if result.shape != psi.shape or not np.isfinite(result).all():
        raise NumericalInfeasibility("additive map has wrong shape or nonfinite values")
    if result.ravel()[edge].tobytes() != psi.ravel()[edge].tobytes():
        raise NumericalInfeasibility("additive map changed grid-edge psi bits")
    return result, {
        "zero_displacement_exact": 0.0,
        "max_matched_contour_move_m": float(movement.max()),
        "p95_matched_contour_move_m": float(np.quantile(movement, 0.95)),
        "boundary_taper_margin_m": margin,
        "shell_width_m": width,
        "max_abs_displacement_m": float(np.linalg.norm(displacement_physical, axis=1).max()),
        "max_abs_delta_psi": float(np.abs(change).max()),
    }
