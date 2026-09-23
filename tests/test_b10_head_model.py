"""Matched B10 heads have exact zero-residual source identity."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import torch
import yaml

from tools.b10_head_model import WaveformResidualNet, decode_residual


class B10HeadTests(unittest.TestCase):
    def test_matched_forward_and_gradient_are_finite(self) -> None:
        params = yaml.safe_load((Path(__file__).resolve().parent.parent /
                                 "baselines/b10-two-rate-info-gate.params.yaml").read_text())
        model = WaveformResidualNet(params["model"])
        batch = 3
        out = model(torch.zeros(batch, 84), torch.zeros(batch, 52),
                    torch.zeros(batch, 41, 21), torch.ones(batch, 41),
                    torch.zeros(batch, 21))
        self.assertEqual(tuple(out.shape), (batch, 52))
        self.assertEqual(torch.count_nonzero(out).item(), 0)
        torch.mean((out - 1.0) ** 2).backward()
        assert model.output.weight.grad is not None
        self.assertTrue(torch.isfinite(model.output.weight.grad).all())

    def test_zero_residual_preserves_float32_bits(self) -> None:
        psi = np.zeros((2, 65, 65), dtype=np.float32)
        psi[0, 0, 0] = np.float32(-0.0)
        q95 = np.array([1.0, -0.0], dtype=np.float32)
        beta = np.array([-0.0, 2.0], dtype=np.float32)
        result = decode_residual(psi, q95, beta, np.zeros((2, 52)),
                                 np.zeros((2, 52)), {})
        self.assertEqual(result["psirz"].tobytes(), psi.tobytes())
        self.assertEqual(result["q95"].tobytes(), q95.tobytes())
        self.assertEqual(result["betaN"].tobytes(), beta.tobytes())


if __name__ == "__main__":
    unittest.main()
