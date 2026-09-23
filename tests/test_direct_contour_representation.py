"""Synthetic absolute LCFS projection and fixed sample-index checks."""

from __future__ import annotations

import unittest

import numpy as np

from tools.diagnose_direct_contour_representation import (
    _canonical_true,
    _indices,
    _project,
)


class DirectContourRepresentationTests(unittest.TestCase):
    def test_absolute_contour_canonicalization_ignores_orientation_and_start(self) -> None:
        theta = np.arange(512) * 2.0 * np.pi / 512
        contour = np.column_stack([
            1.7 + 0.4 * np.cos(theta) + 0.02 * np.cos(3 * theta),
            0.8 * np.sin(theta),
        ])
        original = _canonical_true(contour)
        reverse_roll = _canonical_true(np.roll(contour[::-1], 31, axis=0))
        np.testing.assert_allclose(original, reverse_roll, atol=0, rtol=0)
        reconstructed, energy = _project(original, 8)
        np.testing.assert_allclose(reconstructed, original, atol=1e-14)
        self.assertAlmostEqual(energy, 1.0, places=12)

    def test_K16_retains_a_shape_mode_K8_discards(self) -> None:
        theta = np.arange(512) * 2.0 * np.pi / 512
        contour = _canonical_true(np.column_stack([
            1.7 + 0.4 * np.cos(theta) + 0.01 * np.cos(12 * theta),
            0.8 * np.sin(theta),
        ]))
        k8, e8 = _project(contour, 8)
        k16, e16 = _project(contour, 16)
        self.assertGreater(e16, e8)
        self.assertGreater(float(np.max(np.abs(k8 - contour))), 0.005)
        np.testing.assert_allclose(k16, contour, atol=1e-14)

    def test_fixed_three_positions_and_short_shot_rejection(self) -> None:
        self.assertEqual(_indices(3), [0, 1, 2])
        self.assertEqual(_indices(101), [10, 50, 90])
        with self.assertRaisesRegex(ValueError, "expected at least 3"):
            _indices(2)


if __name__ == "__main__":
    unittest.main()
