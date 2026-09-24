"""Synthetic polar rays keep true notches visible without truth-conditioned fallback."""

from __future__ import annotations

import unittest

import numpy as np

from my_experiments.polar_source_origin import (
    DIRECTIONS,
    polar_contour,
    polygon_centroid,
    ray_outer_radii,
)
from tools.geometry_map_smoke import _simple_closed


def _resample_polygon(vertices: np.ndarray) -> np.ndarray:
    end = np.roll(vertices, -1, axis=0)
    lengths = np.linalg.norm(end - vertices, axis=1)
    cumulative = np.r_[0, np.cumsum(lengths)]
    arclength = np.linspace(0, cumulative[-1], 513)[:-1]
    segment = np.searchsorted(cumulative[1:], arclength, side="right")
    fraction = (arclength - cumulative[segment]) / lengths[segment]
    return vertices[segment] + fraction[:, None] * (end[segment] - vertices[segment])


class PolarSourceOriginTest(unittest.TestCase):
    def test_convex_circle_one_hit_and_positive_simple_reconstruction(self) -> None:
        vertices = np.column_stack([np.cos(2 * np.pi * np.arange(512) / 512),
                                    np.sin(2 * np.pi * np.arange(512) / 512)])
        center = polygon_centroid(vertices, "synthetic-circle")
        radii, detail = ray_outer_radii(vertices, center, 0.1, True, "synthetic-circle")
        self.assertIsNone(detail["hard_reason"])
        self.assertEqual((detail["multi_hit_rays"], detail["zero_hit_rays"],
                          detail["ambiguous_rays"]), (0, 0, 0))
        assert radii is not None
        reconstructed = polar_contour(center, radii, "synthetic-circle")
        self.assertTrue(_simple_closed(reconstructed))
        self.assertTrue(np.all(radii > 0))

    def test_outermost_radius_fills_notch_and_records_three_hits(self) -> None:
        vertices = np.array([[-2, -2], [2, -2], [2, -1], [-1, -1],
                             [-1, 1], [2, 1], [2, 2], [-2, 2]], dtype=np.float64)
        contour = _resample_polygon(vertices)
        source_origin = np.array([-1.5, 0.0])
        outer, detail = ray_outer_radii(contour, source_origin, 0.1, True,
                                         "synthetic-concave")
        self.assertIsNone(detail["hard_reason"])
        self.assertGreater(detail["multi_hit_rays"], 0)
        self.assertEqual((detail["zero_hit_rays"], detail["ambiguous_rays"]), (0, 0))
        self.assertEqual(detail["first_pathology"]["unique_positive_hit_count"], 3)
        assert outer is not None
        self.assertGreater(outer[23], detail["first_pathology"]["first_five_positive_radii_m"][0])
        reconstructed = polar_contour(source_origin, outer, "synthetic-concave")
        self.assertTrue(_simple_closed(reconstructed))
        self.assertTrue(np.isfinite(reconstructed).all())

    def test_outside_true_origin_is_hard_failure(self) -> None:
        contour = DIRECTIONS.copy()
        radii, detail = ray_outer_radii(contour, np.array([2.0, 0.0]), 0.1,
                                         False, "synthetic-outside")
        self.assertIsNone(radii)
        self.assertEqual(detail["hard_reason"], "source_origin_outside_or_on_true_contour")


if __name__ == "__main__":
    unittest.main()
