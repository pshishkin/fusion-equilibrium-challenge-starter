"""The approved contour taper is zero on the grid edge and full at the contour."""

from __future__ import annotations

import unittest

import numpy as np

from tools.geometry_taper import boundary_taper, contour_margin


class GeometryTaperTests(unittest.TestCase):
    def test_edges_zero_contour_and_axis_full(self) -> None:
        R = np.linspace(0.84, 2.54, 65)
        Z = np.linspace(-1.6, 1.6, 65)
        theta = np.linspace(0, 2 * np.pi, 512, endpoint=False)
        contour = np.column_stack([1.69 + 0.5 * np.cos(theta), 0.8 * np.sin(theta)])
        margin = contour_margin(contour, R, Z)
        lo = np.array([R[0], Z[0]])
        span = np.array([R[-1] - R[0], Z[-1] - Z[0]])
        cc = (contour - lo) / span
        axis = (np.array([[1.69, 0.0]]) - lo) / span
        grid_r, grid_z = np.meshgrid((R - R[0]) / span[0], (Z - Z[0]) / span[1])
        grid = np.column_stack([grid_r.ravel(), grid_z.ravel()])
        edge = np.isclose(grid[:, 0], 0) | np.isclose(grid[:, 0], 1)
        edge |= np.isclose(grid[:, 1], 0) | np.isclose(grid[:, 1], 1)
        np.testing.assert_array_equal(boundary_taper(cc, lo, span, margin), 1.0)
        np.testing.assert_array_equal(boundary_taper(axis, lo, span, margin), 1.0)
        np.testing.assert_array_equal(boundary_taper(grid[edge], lo, span, margin), 0.0)
        self.assertGreater(margin, 0)


if __name__ == "__main__":
    unittest.main()
