"""Synthetic input/target isolation and fixed full-polygon MLP arithmetic."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import torch

from toolkit.records import file_hash
from tools.direct_full_contour_cache import _save_atomic, allowed_input_features
from tools.direct_full_contour_predict import predict_polygons
from tools.direct_full_contour_train import DirectContourMLP, contour_loss


class _PoisonTruth:
    def __array__(self, dtype: object = None) -> np.ndarray:
        raise AssertionError(f"true EFIT target was accessed with {dtype}")


class DirectFullContourTests(unittest.TestCase):
    def test_209_inputs_ignore_target_fields_and_keep_phase(self) -> None:
        ready = np.arange(416, dtype=np.float32).reshape(2, 208)
        row: dict[str, object] = {
            "source": "DIII-D",
            "efit_times": np.array([200.0, 225.0]),
            "efit_grid_R": np.linspace(0.84, 2.54, 65),
            "efit_grid_Z": np.linspace(-1.6, 1.6, 65),
            "coil_input_column": "coil_current",
            "coil_R": np.array([1.0]), "coil_Z": np.array([0.0]),
            "coil_width": np.array([0.1]), "coil_height": np.array([0.1]),
            "magnetics_features": ready,
            "efit_psirz": _PoisonTruth(), "efit_li": _PoisonTruth(),
        }
        X = allowed_input_features(row)
        self.assertEqual(X.shape, (2, 209))
        np.testing.assert_array_equal(X[:, :208], ready)
        np.testing.assert_array_equal(X[:, 208], [0.0, 1.0])

    def test_fixed_worst_vertex_loss_and_gradients(self) -> None:
        prediction = torch.zeros((1, 1024), requires_grad=True)
        target = torch.zeros((1, 1024))
        target[0, 0] = 1.0
        measured = contour_loss(prediction, target).mean()
        self.assertAlmostEqual(float(measured.item()), 1 / 512 + 1 / 26, places=7)
        measured.backward()
        self.assertIsNotNone(prediction.grad)
        self.assertTrue(torch.isfinite(prediction.grad).all())
        network = DirectContourMLP()
        self.assertEqual(network(torch.zeros(3, 209)).shape, (3, 1024))

    def test_cpu_inference_uses_saved_scaler_and_physical_mean(self) -> None:
        theta = 2 * np.pi * np.arange(512) / 512
        expected = np.column_stack([1.7 + 0.4 * np.cos(theta),
                                    0.8 * np.sin(theta)]).astype(np.float32)
        network = DirectContourMLP()
        for parameter in network.parameters():
            torch.nn.init.zeros_(parameter)
        checkpoint = {
            "state_dict": network.state_dict(),
            "feature_mean": torch.zeros(209),
            "feature_std": torch.ones(209),
            "target_mean": torch.from_numpy(expected.reshape(1024)),
            "target_scale": torch.tensor(0.1),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "direct_contour.joblib"
            joblib.dump(checkpoint, path)
            predicted = predict_polygons(np.zeros((2, 209), dtype=np.float32),
                                         path, file_hash(path))
        np.testing.assert_array_equal(predicted, np.stack([expected, expected]))

    def test_per_shot_cache_keeps_selected_indices_and_full_count(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "one.npz"
            X = np.zeros((2, 209), dtype=np.float32)
            C = np.zeros((2, 512, 2), dtype=np.float32)
            _save_atomic(path, X, C, np.array([0, 4], dtype=np.int32), 5)
            with np.load(path, allow_pickle=False) as saved:
                self.assertEqual(int(saved["full_frames"]), 5)
                np.testing.assert_array_equal(saved["indices"], [0, 4])
                self.assertEqual(saved["features"].shape, (2, 209))
                self.assertEqual(saved["contours"].shape, (2, 512, 2))


if __name__ == "__main__":
    unittest.main()
