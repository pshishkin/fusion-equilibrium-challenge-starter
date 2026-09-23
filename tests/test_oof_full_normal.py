"""Full normal diagnostic changes Fourier truncation only."""

from __future__ import annotations

import unittest

import numpy as np

from tools.contour_learnability import decode_normal_coefficients, encode_normal_target
from tools.oof_normal_full_diagnostic import full_normal_displacement


class FullNormalDiagnosticTests(unittest.TestCase):
    def test_full_normal_keeps_high_frequency_without_tangent_motion(self) -> None:
        theta = np.arange(512) * (2 * np.pi / 512)
        radius = 0.4
        source = np.column_stack([1.7 + radius * np.cos(theta),
                                  radius * np.sin(theta)])
        true_radius = radius + 0.015 * np.sin(12 * theta) + 0.005 * np.cos(3 * theta)
        truth = np.column_stack([1.7 + true_radius * np.cos(theta),
                                 true_radius * np.sin(theta)])
        coefficients, _ = encode_normal_target(source, truth)
        full = full_normal_displacement(source, truth)
        lowpass = decode_normal_coefficients(coefficients, source)
        self.assertEqual(full.shape, (512, 2))
        self.assertTrue(np.isfinite(full).all())
        self.assertGreater(float(np.linalg.norm(full - lowpass)), 0.15)
        tangent = np.roll(source, -1, axis=0) - np.roll(source, 1, axis=0)
        tangent /= np.linalg.norm(tangent, axis=1)[:, None]
        np.testing.assert_allclose(np.sum(full * tangent, axis=1), 0.0, atol=1e-14)

    def test_zero_truth_displacement_is_exact_zero(self) -> None:
        theta = np.arange(512) * (2 * np.pi / 512)
        source = np.column_stack([1.7 + 0.4 * np.cos(theta), 0.4 * np.sin(theta)])
        self.assertEqual(np.count_nonzero(full_normal_displacement(source, source)), 0)


if __name__ == "__main__":
    unittest.main()
