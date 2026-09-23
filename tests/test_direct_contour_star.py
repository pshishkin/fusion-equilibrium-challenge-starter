"""Synthetic checks of fixed centroid-ray intersection accounting."""

from __future__ import annotations

import unittest

import numpy as np

from fusion_scoring.contour import resample_closed
from tools.diagnose_direct_contour_star import _polygon_centroid, _ray_uniqueness


class DirectContourStarTests(unittest.TestCase):
    def test_ellipse_has_one_positive_hit_for_every_shifted_ray(self) -> None:
        theta = 2 * np.pi * np.arange(512) / 512
        contour = np.column_stack([1.7 + 0.4 * np.cos(theta),
                                   0.8 * np.sin(theta)])
        center = _polygon_centroid(contour)
        np.testing.assert_allclose(center, [1.7, 0.0], atol=1e-12)
        result = _ray_uniqueness(contour, center, 0.0265625, True)
        self.assertTrue(result["centroid_strictly_inside"])
        self.assertEqual(result["exactly_one_positive_intersection_angles"], 512)
        self.assertEqual(result["zero_intersection_angles"], 0)
        self.assertEqual(result["multiple_intersection_angles"], 0)
        self.assertEqual(result["collinear_ambiguous_angles"], 0)

    def test_concave_C_shape_records_multiple_rays(self) -> None:
        vertices = np.array([
            [-3.0, -3.0], [3.0, -3.0], [3.0, -2.0], [0.0, -2.0],
            [0.0, 2.0], [3.0, 2.0], [3.0, 3.0], [-3.0, 3.0],
        ])
        contour = resample_closed(vertices)
        center = _polygon_centroid(contour)
        result = _ray_uniqueness(contour, center, 0.0265625, True)
        self.assertGreater(result["multiple_intersection_angles"], 0)
        self.assertLess(result["exactly_one_positive_intersection_angles"], 512)
        self.assertIsNotNone(result["first_pathology"])


if __name__ == "__main__":
    unittest.main()
