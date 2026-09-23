"""Numerical invariants for the two constrained truth-assisted map corrections."""

from __future__ import annotations

import unittest

import numpy as np

from tools.geometry_map_smoke import (
    _cyclic_match,
    _edge_anchors,
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

    def test_zero_profile_remap_is_bit_exact_and_family_monotone(self) -> None:
        psi = np.linspace(-1.0, 1.0, 65 * 65, dtype=np.float32).reshape(65, 65)
        self.assertEqual(_remap_map(psi, -1, 1.0, 0.0, 0.0).tobytes(), psi.tobytes())
        u = np.linspace(0, 1, 10001)
        for a in np.linspace(-2, 2, 9):
            g = u + a * u**2 * (1 - u) ** 2
            self.assertTrue(np.all(np.diff(g) > 0))
            self.assertEqual(g[0], 0.0)
            self.assertEqual(g[-1], 1.0)


if __name__ == "__main__":
    unittest.main()
