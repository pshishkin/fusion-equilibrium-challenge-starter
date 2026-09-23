"""Numerical polygon-validity checks for the validation-only contour oracle."""

from __future__ import annotations

import unittest

import numpy as np

from tools.run_direct_full_contour_oracle import _valid_polygon


class DirectFullContourOracleTests(unittest.TestCase):
    def test_simple_finite_in_grid_polygon_and_invalid_variants(self) -> None:
        R = np.linspace(0.84, 2.54, 65)
        Z = np.linspace(-1.6, 1.6, 65)
        theta = 2 * np.pi * np.arange(512) / 512
        ellipse = np.column_stack([1.7 + 0.4 * np.cos(theta), 0.8 * np.sin(theta)])
        self.assertTrue(_valid_polygon(ellipse, R, Z))
        self.assertFalse(_valid_polygon(ellipse * np.array([2.0, 1.0]), R, Z))
        self.assertFalse(_valid_polygon(np.column_stack([
            1.7 + 0.4 * np.sin(theta), 0.8 * np.sin(2 * theta)
        ]), R, Z))
        broken = ellipse.copy()
        broken[7, 0] = np.nan
        self.assertFalse(_valid_polygon(broken, R, Z))


if __name__ == "__main__":
    unittest.main()
