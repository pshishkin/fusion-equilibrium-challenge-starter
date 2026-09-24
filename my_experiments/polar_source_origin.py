"""Physical polar LCFS charts around a source-predicted, input-only centroid."""

from __future__ import annotations

from typing import Any

import numpy as np

N_RAYS = 512
ANGLES = 2 * np.pi * (np.arange(N_RAYS, dtype=np.float64) + 0.5) / N_RAYS
DIRECTIONS = np.column_stack([np.cos(ANGLES), np.sin(ANGLES)])


def polygon_centroid(contour: np.ndarray, location: str) -> np.ndarray:
    """Return the physical shoelace centroid of a finite CCW 512-vertex polygon."""
    c = np.asarray(contour, dtype=np.float64)
    if c.shape != (512, 2) or not np.isfinite(c).all():
        raise ValueError(f"{location}: contour {c.shape}/finite={np.isfinite(c).all()}, "
                         "expected (512,2)/true")
    nxt = np.roll(c, -1, axis=0)
    cross = c[:, 0] * nxt[:, 1] - nxt[:, 0] * c[:, 1]
    area2 = float(np.sum(cross))
    if not np.isfinite(area2) or area2 <= 0:
        raise ValueError(f"{location}: signed doubled polygon area {area2}, expected positive CCW")
    center = np.array([np.sum((c[:, 0] + nxt[:, 0]) * cross),
                       np.sum((c[:, 1] + nxt[:, 1]) * cross)]) / (3 * area2)
    if not np.isfinite(center).all():
        raise ValueError(f"{location}: physical polygon centroid is nonfinite")
    return center


def _cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def origin_edge_distance(contour: np.ndarray, origin: np.ndarray,
                         location: str) -> float:
    """Minimum physical distance from origin to each closed polygon segment."""
    c = np.asarray(contour, dtype=np.float64)
    center = np.asarray(origin, dtype=np.float64)
    if c.shape != (512, 2) or center.shape != (2,):
        raise ValueError(f"{location}: contour/origin {c.shape}/{center.shape}, "
                         "expected (512,2)/(2,)")
    seg = np.roll(c, -1, axis=0) - c
    length_sq = np.sum(seg * seg, axis=1)
    if np.any(length_sq <= 0) or not np.isfinite(length_sq).all():
        raise ValueError(f"{location}: zero/nonfinite contour segment length")
    along = np.clip(np.sum((center[None] - c) * seg, axis=1) / length_sq, 0.0, 1.0)
    nearest = c + along[:, None] * seg
    return float(np.min(np.linalg.norm(center[None] - nearest, axis=1)))


def ray_outer_radii(contour: np.ndarray, origin: np.ndarray, spacing: float,
                    origin_inside_true: bool, location: str
                    ) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Choose the outermost positive hit on every fixed ray, counting real notches."""
    c = np.asarray(contour, dtype=np.float64)
    center = np.asarray(origin, dtype=np.float64)
    if (c.shape != (512, 2) or center.shape != (2,) or not np.isfinite(c).all()
            or not np.isfinite(center).all() or not np.isfinite(spacing)
            or spacing <= 0):
        raise ValueError(f"{location}: contour/origin/spacing {c.shape}/"
                         f"{center.shape}/{spacing}, expected finite 512x2/2/positive")
    tau = 1e-4 * spacing
    edge_distance = origin_edge_distance(c, center, location)
    if not origin_inside_true or edge_distance <= tau:
        return None, {
            "hard_reason": "source_origin_outside_or_on_true_contour",
            "origin_inside_true": bool(origin_inside_true),
            "origin_true_edge_distance_m": edge_distance,
            "multi_hit_rays": 0, "zero_hit_rays": 0,
            "ambiguous_rays": 0, "first_pathology": None,
        }
    start = c
    seg = np.roll(c, -1, axis=0) - c
    lengths = np.linalg.norm(seg, axis=1)
    if np.any(lengths <= 0) or not np.isfinite(lengths).all():
        raise ValueError(f"{location}: zero/nonfinite true contour segment length")
    offset = start - center
    denominator = _cross(DIRECTIONS[:, None, :], seg[None, :, :])
    parallel = np.abs(denominator) <= 1e-12 * lengths[None, :]
    collinear = parallel & (np.abs(_cross(offset[None, :, :],
                                            DIRECTIONS[:, None, :])) <= tau)
    safe_denominator = np.where(parallel, 1.0, denominator)
    radius = _cross(offset, seg)[None, :] / safe_denominator
    parameter = _cross(offset[None, :, :], DIRECTIONS[:, None, :]) / safe_denominator
    allowed_parameter = ((parameter >= -tau / lengths[None, :])
                         & (parameter <= 1.0 + tau / lengths[None, :]))
    hits = (~parallel) & allowed_parameter & (radius > tau)
    outer = np.empty(N_RAYS, dtype=np.float64)
    multi_rays = 0
    zero_rays = 0
    ambiguous_rays = int(np.count_nonzero(np.any(collinear, axis=1)))
    max_inner_gap = 0.0
    first_pathology: dict[str, Any] | None = None
    for angle in range(N_RAYS):
        values = np.sort(radius[angle, hits[angle]])
        unique: list[float] = []
        for value in values:
            if not unique or value - unique[-1] > tau:
                unique.append(float(value))
        if not unique:
            zero_rays += 1
            outer[angle] = np.nan
        else:
            outer[angle] = unique[-1]
            if len(unique) > 1:
                multi_rays += 1
                max_inner_gap = max(max_inner_gap, unique[-1] - unique[0])
        if first_pathology is None and (len(unique) != 1 or np.any(collinear[angle])):
            first_pathology = {"angle_index": angle,
                               "angle_radians": float(ANGLES[angle]),
                               "unique_positive_hit_count": len(unique),
                               "first_five_positive_radii_m": unique[:5],
                               "collinear_segment_count": int(np.count_nonzero(collinear[angle]))}
    hard_reason = ("true_collinear_ray" if ambiguous_rays else
                   "true_zero_hit_ray" if zero_rays else None)
    detail = {
        "hard_reason": hard_reason,
        "origin_inside_true": True,
        "origin_true_edge_distance_m": edge_distance,
        "multi_hit_rays": multi_rays,
        "zero_hit_rays": zero_rays,
        "ambiguous_rays": ambiguous_rays,
        "max_inner_to_outer_hit_gap_m": max_inner_gap,
        "first_pathology": first_pathology,
    }
    if hard_reason is not None or not np.isfinite(outer).all() or np.any(outer <= tau):
        return None, detail
    return outer, detail


def polar_contour(origin: np.ndarray, radius: np.ndarray,
                  location: str) -> np.ndarray:
    """Convert strictly positive half-bin ray lengths to a physical polygon."""
    center = np.asarray(origin, dtype=np.float64)
    r = np.asarray(radius, dtype=np.float64)
    if (center.shape != (2,) or r.shape != (512,)
            or not np.isfinite(center).all() or not np.isfinite(r).all()
            or np.any(r <= 0)):
        raise ValueError(f"{location}: polar origin/radius {center.shape}/{r.shape}, "
                         "expected finite (2,)/(512,) with positive radii")
    return center[None, :] + r[:, None] * DIRECTIONS
