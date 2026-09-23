"""The additive contour probe changes finite psi only near its fixed shell."""

from __future__ import annotations

import unittest

import numpy as np
from scipy.spatial import cKDTree

from tools.geometry_additive_probe import _correct_map
from tools.geometry_map_smoke import NumericalInfeasibility


class GeometryAdditiveProbeTests(unittest.TestCase):
    def test_fixed_contour_shift_keeps_edge_bits_and_localizes_change(self) -> None:
        R = np.linspace(0.84, 2.54, 65)
        Z = np.linspace(-1.6, 1.6, 65)
        rr, zz = np.meshgrid(R, Z)
        psi = ((rr - 1.7) ** 2 + zz**2).astype(np.float32)
        theta = np.linspace(0, 2 * np.pi, 512, endpoint=False)
        source_contour = np.column_stack([1.7 + 0.4 * np.cos(theta), 0.8 * np.sin(theta)])
        true_contour = source_contour + np.array([0.02, 0.0])
        corrected, detail = _correct_map(psi, source_contour, true_contour, (1.7, 0.0), R, Z)
        self.assertEqual(corrected.dtype, np.float32)
        self.assertTrue(np.isfinite(corrected).all())
        np.testing.assert_array_equal(corrected[0], psi[0])
        np.testing.assert_array_equal(corrected[-1], psi[-1])
        np.testing.assert_array_equal(corrected[:, 0], psi[:, 0])
        np.testing.assert_array_equal(corrected[:, -1], psi[:, -1])
        self.assertAlmostEqual(detail["shell_width_m"], 2 * min(np.diff(R).min(), np.diff(Z).min()))
        distance, _ = cKDTree(source_contour).query(np.column_stack([rr.ravel(), zz.ravel()]))
        delta = np.abs(corrected.astype(np.float64) - psi.astype(np.float64)).ravel()
        self.assertGreater(
            float(delta[distance < detail["shell_width_m"]].mean()),
            float(delta[distance > 3 * detail["shell_width_m"]].mean()),
        )

    def test_nonfinite_source_raises(self) -> None:
        R = np.linspace(0.84, 2.54, 65)
        Z = np.linspace(-1.6, 1.6, 65)
        theta = np.linspace(0, 2 * np.pi, 512, endpoint=False)
        contour = np.column_stack([1.7 + 0.4 * np.cos(theta), 0.8 * np.sin(theta)])
        psi = np.zeros((65, 65), dtype=np.float32)
        psi[10, 10] = np.nan
        with self.assertRaisesRegex(NumericalInfeasibility, r"expected \(65,65\)/float32/0"):
            _correct_map(psi, contour, contour, (1.7, 0.0), R, Z)

    def test_large_contour_displacement_is_scored_as_a_finite_map(self) -> None:
        R = np.linspace(0.84, 2.54, 65)
        Z = np.linspace(-1.6, 1.6, 65)
        rr, zz = np.meshgrid(R, Z)
        psi = ((rr - 1.7) ** 2 + zz**2).astype(np.float32)
        theta = np.linspace(0, 2 * np.pi, 512, endpoint=False)
        source_contour = np.column_stack([1.7 + 0.4 * np.cos(theta), 0.8 * np.sin(theta)])
        true_contour = source_contour + np.array([0.25, 0.0])
        corrected, detail = _correct_map(psi, source_contour, true_contour, (1.7, 0.0), R, Z)
        self.assertGreater(
            detail["max_matched_contour_move_m"], 4 * np.hypot(np.diff(R).mean(), np.diff(Z).mean())
        )
        self.assertTrue(np.isfinite(corrected).all())
        self.assertEqual(corrected[0].tobytes(), psi[0].tobytes())


if __name__ == "__main__":
    unittest.main()
