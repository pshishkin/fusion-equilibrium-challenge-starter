"""Synthetic checks for the untrained contour-normal learner components."""

from __future__ import annotations

import unittest

import numpy as np

from my_experiments.baseline_model import FEATURE_WIDTH
from tools.contour_learnability import (
    N_COEFFICIENTS,
    N_CONTOUR,
    N_INPUTS,
    ContourRidge,
    allowed_input_features,
    canonical_contour,
    contour_summaries,
    correct_map_from_displacement,
    decode_normal_coefficients,
    encode_normal_target,
    outward_normals,
    per_shot_weights,
)
from tools.geometry_additive_probe import _correct_map
from tools.geometry_map_smoke import _cyclic_match


def _ellipse() -> np.ndarray:
    theta = np.linspace(0.0, 2.0 * np.pi, N_CONTOUR, endpoint=False)
    return np.column_stack([1.7 + 0.4 * np.cos(theta), 0.8 * np.sin(theta)])


def _grid_and_psi() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    R = np.linspace(0.84, 2.54, 65)
    Z = np.linspace(-1.6, 1.6, 65)
    rr, zz = np.meshgrid(R, Z)
    psi = ((rr - 1.7) ** 2 + zz**2).astype(np.float32)
    return R, Z, psi


class _PoisonTruth:
    def __array__(self, dtype: object = None) -> np.ndarray:
        raise AssertionError(f"EFIT truth was accessed with dtype {dtype}")


class ContourLearnabilityTests(unittest.TestCase):
    def test_canonical_normals_and_fourier_roundtrip(self) -> None:
        original = np.roll(_ellipse(), 43, axis=0)
        canonical, start = canonical_contour(original)
        self.assertEqual(start, 43)
        normals = outward_normals(canonical)
        self.assertGreater(normals[0, 0], 0.999)
        np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-12)
        reversed_canonical, _ = canonical_contour(original[::-1])
        self.assertGreater(outward_normals(reversed_canonical)[0, 0], 0.999)
        phase = np.arange(N_CONTOUR) * 2.0 * np.pi / N_CONTOUR
        d = 0.02 + 0.003 * np.cos(2 * phase) - 0.001 * np.sin(4 * phase)
        truth = canonical + d[:, None] * normals
        coefficients, target = encode_normal_target(original, truth)
        self.assertEqual(coefficients.shape, (N_COEFFICIENTS,))
        decoded = decode_normal_coefficients(coefficients, original)
        canonical_decoded = np.roll(decoded, -start, axis=0)
        expected = np.fft.irfft(
            np.r_[np.fft.rfft(target)[:9], np.zeros(248)], n=N_CONTOUR
        )
        np.testing.assert_allclose(
            np.sum(canonical_decoded * normals, axis=1), expected, atol=1e-14
        )
        np.testing.assert_allclose(
            np.sum(canonical_decoded * np.column_stack([-normals[:, 1], normals[:, 0]]),
                   axis=1), 0.0, atol=1e-14
        )

    def test_allowed_features_strip_truth_and_use_only_predicted_contour(self) -> None:
        contour = _ellipse()
        ready = np.arange(2 * FEATURE_WIDTH, dtype=np.float32).reshape(2, FEATURE_WIDTH)
        row: dict[str, object] = {
            "source": "DIII-D",
            "efit_times": np.array([3.0, 5.0]),
            "efit_grid_R": np.linspace(0.84, 2.54, 65),
            "efit_grid_Z": np.linspace(-1.6, 1.6, 65),
            "coil_input_column": "coil_current",
            "coil_R": np.array([1.0]),
            "coil_Z": np.array([0.0]),
            "coil_width": np.array([0.1]),
            "coil_height": np.array([0.1]),
            "magnetics_features": ready,
            "efit_psirz": _PoisonTruth(),
            "efit_li": _PoisonTruth(),
        }
        X = allowed_input_features(row, np.stack([contour, contour]))
        self.assertEqual(X.shape, (2, N_INPUTS))
        np.testing.assert_array_equal(X[:, :FEATURE_WIDTH], ready)
        np.testing.assert_allclose(X[:, -1], [0.0, 1.0])
        np.testing.assert_array_equal(X[:, FEATURE_WIDTH:-1],
                                      np.stack([contour_summaries(contour)] * 2))
        stripped = {k: v for k, v in row.items() if k not in ("efit_psirz", "efit_li")}
        np.testing.assert_array_equal(X, allowed_input_features(stripped, np.stack([contour] * 2)))

    def test_weighted_fixed_alpha_ridge_is_finite_and_shot_balanced(self) -> None:
        lengths = [2, 4]
        weights = per_shot_weights(lengths)
        np.testing.assert_allclose([weights[:2].sum(), weights[2:].sum()], [1.0, 1.0])
        X = np.zeros((6, N_INPUTS))
        X[2:, 0] = 1.0
        X[:, -1] = np.linspace(0.0, 1.0, 6)
        y = np.zeros((6, N_COEFFICIENTS))
        y[:, 0] = np.arange(6) * 0.01
        model = ContourRidge().fit(X, y, lengths)
        self.assertAlmostEqual(float(model.scaler.mean_[0]), 0.5)
        self.assertEqual(model.model.alpha, 35.0)
        prediction = model.predict(X)
        self.assertEqual(prediction.shape, y.shape)
        self.assertTrue(np.isfinite(prediction).all())

    def test_pure_additive_helper_bit_parity_and_zero_control(self) -> None:
        R, Z, psi = _grid_and_psi()
        pred = _ellipse()
        truth = np.roll(pred + np.array([0.02, -0.005]), 31, axis=0)
        matched, _ = _cyclic_match(pred, truth)
        legacy, _ = _correct_map(psi, pred, truth, (1.7, 0.0), R, Z)
        direct, _ = correct_map_from_displacement(
            psi, pred, matched - pred, (1.7, 0.0), R, Z
        )
        self.assertEqual(legacy.tobytes(), direct.tobytes())
        psi[0, 0] = np.float32(-0.0)
        zero, detail = correct_map_from_displacement(
            psi, pred, np.zeros_like(pred), (1.7, 0.0), R, Z
        )
        self.assertEqual(psi.tobytes(), zero.tobytes())
        self.assertEqual(detail["zero_displacement_exact"], 1.0)


if __name__ == "__main__":
    unittest.main()
