"""Synthetic checks for physical-axis loss and exact zero-source behavior."""

from __future__ import annotations

import unittest
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from my_experiments.map_residual_unet import (
    MapResidualUNet,
    apply_delta,
    frame_energies,
    grid_channels,
    torch_loss,
)


class MapResidualUNetTest(unittest.TestCase):
    def setUp(self) -> None:
        self.params: dict[str, Any] = yaml.safe_load(
            Path("baselines/map-residual-unet.params.yaml").read_text())
        self.r = np.linspace(1.0, 2.0, 65, dtype=np.float64)
        self.z = np.linspace(-1.0, 1.0, 65, dtype=np.float64)
        self.mask = np.ones((65, 65), dtype=np.bool_)

    def test_affine_field_physical_axis_stencil(self) -> None:
        rr, zz = np.meshgrid(self.r, self.z)
        field = np.asarray(3 * rr + 5 * zz, dtype=np.float32)[None]
        _, gradient = frame_energies(field, np.ones((65, 65)), self.r, self.z)
        self.assertAlmostEqual(float(gradient[0]), 17.0, places=4)
        prediction = torch.from_numpy(field[:, None].copy())
        target = torch.zeros_like(prediction)
        weight = torch.ones((65, 65))
        loss = torch_loss(prediction, target, weight,
                          torch.from_numpy(self.r.copy()), torch.from_numpy(self.z.copy()),
                          1.0, 1.0)
        value, grad = frame_energies(field, np.ones((65, 65)), self.r, self.z)
        self.assertAlmostEqual(float(loss), float(value[0] + grad[0]), places=3)

    def test_zero_output_and_source_bits(self) -> None:
        torch.manual_seed(0)
        model = MapResidualUNet(self.params["model"], 86)
        source = np.broadcast_to(np.linspace(0, 1, 65, dtype=np.float32)[None, None, :],
                                 (2, 65, 65)).copy()
        source[0, 0, 0] = -0.0
        channels, extra, scale = grid_channels(source, self.r, self.z, self.mask, 1024,
                                                 "synthetic-shot")
        self.assertEqual(channels.shape, (2, 4, 65, 65))
        self.assertEqual(extra.shape, (2, 2))
        result = model(torch.from_numpy(channels), torch.zeros((2, 86)))
        self.assertEqual(result.shape, (2, 1, 65, 65))
        self.assertEqual(torch.count_nonzero(result).item(), 0)
        out = apply_delta(source, result.detach().numpy()[:, 0], scale, "synthetic-shot")
        self.assertEqual(out.tobytes(), source.tobytes())

    def test_range_guard_reports_frame_and_numbers(self) -> None:
        source = np.zeros((1, 65, 65), dtype=np.float32)
        with self.assertRaisesRegex(ValueError,
                                    r"bad-shot: frame 0 source flux range .*required >"):
            grid_channels(source, self.r, self.z, self.mask, 1024, "bad-shot")

    def test_loss_gradient_reaches_zero_initialized_head(self) -> None:
        torch.manual_seed(0)
        model = MapResidualUNet(self.params["model"], 86)
        x = torch.randn((2, 4, 65, 65))
        features = torch.randn((2, 86))
        target = torch.randn((2, 1, 65, 65)) * 0.01
        predicted = model(x, features)
        loss = torch_loss(predicted, target, torch.ones((65, 65)),
                          torch.from_numpy(self.r.copy()), torch.from_numpy(self.z.copy()),
                          1.0, 1.0)
        loss.backward()
        self.assertIsNotNone(model.output.weight.grad)
        assert model.output.weight.grad is not None
        self.assertGreater(float(model.output.weight.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
