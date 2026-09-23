"""The raw anti-alias and matched cadence control preserve their clock semantics."""

from __future__ import annotations

import unittest

import numpy as np

from tools.b10_waveforms import cadence_control, trapezoid_bins


class B10WaveformTests(unittest.TestCase):
    def test_linear_raw_trace_integrates_to_bin_centre_value(self) -> None:
        centres = np.array([10.0, 20.0, 40.0])
        for step in (0.05, 0.5):
            times = np.arange(0.0, 60.0 + step / 2, step)
            values = np.column_stack([2.0 * times + 3.0, -0.5 * times + 10.0])
            measured = trapezoid_bins(times, values, centres, 0.25)
            expected = np.column_stack([2.0 * centres + 3.0, -0.5 * centres + 10.0])
            np.testing.assert_allclose(measured, expected, atol=2e-5, rtol=0)
        with self.assertRaisesRegex(ValueError, "raw support"):
            trapezoid_bins(np.arange(0.0, 40.5, 0.5),
                           np.ones((81, 2)), np.array([40.0]), 0.25)

    def test_EFIT_reconstruction_pads_edges_with_identical_mask(self) -> None:
        times = np.array([0.0, 20.0, 40.0])
        values = np.tile(times[:, None], (1, 21))
        query = np.array([[-10.0, 0.0, 10.0], [10.0, 20.0, 30.0],
                          [30.0, 40.0, 50.0]])
        control, mask = cadence_control(times, values, query)
        np.testing.assert_array_equal(control[:, :, 0],
                                      np.array([[0.0, 0.0, 10.0],
                                                [10.0, 20.0, 30.0],
                                                [30.0, 40.0, 40.0]]))
        np.testing.assert_array_equal(mask,
                                      np.array([[0.0, 1.0, 1.0],
                                                [1.0, 1.0, 1.0],
                                                [1.0, 1.0, 0.0]]))


if __name__ == "__main__":
    unittest.main()
