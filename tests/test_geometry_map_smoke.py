"""Numerical invariants for the two constrained truth-assisted map corrections."""

from __future__ import annotations

import unittest

import numpy as np

from tools.geometry_map_smoke import (
    NumericalInfeasibility,
    _cyclic_match,
    _dense_forward_safety,
    _edge_anchors,
    _invert_damped_newton,
    _invert_fixed_point,
    _remap_map,
    _simple_closed,
    _warp_map,
)


class GeometryMapSmokeTests(unittest.TestCase):
    def test_closed_curve_alignment_handles_cycle_and_orientation(self) -> None:
        theta = np.linspace(0, 2 * np.pi, 512, endpoint=False)
        curve = np.column_stack([1.7 + 0.4 * np.cos(theta), 0.8 * np.sin(theta)])
        self.assertTrue(_simple_closed(curve))
        self.assertEqual(_edge_anchors().shape, (16, 2))
        for shifted in (np.roll(curve, 73, axis=0), np.roll(curve[::-1], 73, axis=0)):
            matched, _ = _cyclic_match(curve, shifted)
            np.testing.assert_array_equal(matched, curve)

    def test_identity_contour_warp_preserves_map(self) -> None:
        r = np.linspace(0.84, 2.54, 65)
        z = np.linspace(-1.6, 1.6, 65)
        rr, zz = np.meshgrid(r, z)
        psi = ((rr - 1.7) ** 2 + zz**2).astype(np.float32)
        theta = np.linspace(0, 2 * np.pi, 512, endpoint=False)
        contour = np.column_stack([1.7 + 0.4 * np.cos(theta), 0.8 * np.sin(theta)])
        corrected, detail = _warp_map(psi, contour, contour, (1.7, 0.0), r, z)
        np.testing.assert_array_equal(corrected, psi)
        self.assertGreater(detail["min_jacobian"], 0.2)
        tapered, detail_tapered = _warp_map(
            psi, contour, contour, (1.7, 0.0), r, z, boundary_zero=True
        )
        np.testing.assert_array_equal(tapered, psi)
        self.assertEqual(detail_tapered["tapered_boundary_displacement_max_m"], 0.0)

    def test_zero_profile_remap_is_bit_exact_and_family_monotone(self) -> None:
        psi = np.linspace(-1.0, 1.0, 65 * 65, dtype=np.float32).reshape(65, 65)
        self.assertEqual(_remap_map(psi, -1, 1.0, 0.0, 0.0).tobytes(), psi.tobytes())
        u = np.linspace(0, 1, 10001)
        for a in np.linspace(-2, 2, 9):
            g = u + a * u**2 * (1 - u) ** 2
            self.assertTrue(np.all(np.diff(g) > 0))
            self.assertEqual(g[0], 0.0)
            self.assertEqual(g[-1], 1.0)

    def test_adaptive_inverse_contracts_past_forty_without_relaxing_tolerance(self) -> None:
        fixed = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])

        def slow_field(x: np.ndarray) -> np.ndarray:
            return 0.8 * (x - 0.5)

        inverse, iterations, residual = _invert_fixed_point(
            fixed, slow_field, np.ones(2), 1e-7, 160, adaptive_monotone=True
        )
        self.assertGreater(iterations, 40)
        self.assertLessEqual(iterations, 160)
        self.assertLessEqual(residual, 1e-7)
        self.assertTrue(np.all((inverse >= 0) & (inverse <= 1)))
        with self.assertRaisesRegex(ValueError, "exceeds tolerance"):
            _invert_fixed_point(fixed, slow_field, np.ones(2), 1e-7, 40, adaptive_monotone=True)

    def test_direct_newton_inverts_affine_map_and_keeps_boundary_exact(self) -> None:
        fixed = np.array([[0.0, 0.0], [0.4, 0.4], [0.5, 0.5], [0.6, 0.6], [1.0, 1.0]])
        none_boundary = np.zeros(len(fixed), dtype=bool)

        def affine_field(x: np.ndarray) -> np.ndarray:
            return 0.8 * (x - 0.5)

        alphas = [0.5**i for i in range(10)]
        inverse, iterations, residual, min_det, _ = _invert_damped_newton(
            fixed, affine_field, none_boundary, 1e-7, 12, 1e-5, alphas
        )
        self.assertLessEqual(iterations, 2)
        self.assertLessEqual(residual, 1e-7)
        self.assertGreater(min_det, 0.2)
        np.testing.assert_allclose(inverse, (fixed + 0.4) / 1.8)
        with self.assertRaisesRegex(NumericalInfeasibility, "after 0 steps"):
            _invert_damped_newton(fixed, affine_field, none_boundary, 1e-7, 0, 1e-5, alphas)

        boundary = np.array([True, False, False, False, True])

        def tapered_field(x: np.ndarray) -> np.ndarray:
            return 0.1 * x * (1 - x)

        tapered_inverse, _, tapered_residual, _, _ = _invert_damped_newton(
            fixed, tapered_field, boundary, 1e-7, 12, 1e-5, alphas
        )
        self.assertEqual(tapered_inverse[boundary].tobytes(), fixed[boundary].tobytes())
        self.assertLessEqual(tapered_residual, 1e-7)

    def test_dense_forward_safety_accepts_identity_and_rejects_fold(self) -> None:
        axis = np.linspace(0, 1, 65)
        rr, zz = np.meshgrid(axis, axis)
        fixed = np.column_stack([rr.ravel(), zz.ravel()])

        def zero_field(x: np.ndarray) -> np.ndarray:
            return np.zeros_like(x)

        min_area, min_det, max_condition = _dense_forward_safety(
            fixed, fixed.copy(), zero_field, np.array([1.7, 3.2]), 257, 10.0
        )
        self.assertGreater(min_area, 0)
        self.assertAlmostEqual(min_det, 1.0)
        self.assertAlmostEqual(max_condition, 1.0)

        def folded_field(x: np.ndarray) -> np.ndarray:
            return np.column_stack([-2 * x[:, 0], np.zeros(len(x))])

        with self.assertRaisesRegex(NumericalInfeasibility, "signed area"):
            _dense_forward_safety(
                fixed,
                fixed + folded_field(fixed),
                folded_field,
                np.array([1.7, 3.2]),
                257,
                10.0,
            )


if __name__ == "__main__":
    unittest.main()
