"""Raw clock classes and centred-bin support are deterministic and strict."""

from __future__ import annotations

import unittest

import numpy as np

from tools.b10_raw_support import classify_clock, raw_bin_support


class B10RawSupportTests(unittest.TestCase):
    def test_both_measured_clock_classes(self) -> None:
        self.assertEqual(classify_clock(np.arange(100) * 0.05, "fast", 0.01, 1.5)[0], 0.05)
        self.assertEqual(classify_clock(np.arange(100) * 0.5, "slow", 0.01, 1.5)[0], 0.5)
        with self.assertRaisesRegex(ValueError, "expected 0.05 or 0.5"):
            classify_clock(np.arange(100) * 0.2, "unknown", 0.01, 1.5)
        jumped = np.arange(100) * 0.5
        jumped[50:] += 1.0
        with self.assertRaisesRegex(ValueError, "maximum"):
            classify_clock(jumped, "gap", 0.01, 1.5)

    def test_centered_bins_require_raw_margin(self) -> None:
        frames = np.array([20.0, 40.0])
        centres = np.arange(-10.0, 10.0 + 0.25, 0.5)
        okay, lo, hi = raw_bin_support(frames, np.arange(0.0, 60.5, 0.5), centres, 0.25)
        self.assertEqual(okay, 0)
        self.assertEqual((lo, hi), (9.75, 50.25))
        missing, _, _ = raw_bin_support(frames, np.arange(10.0, 50.5, 0.5), centres, 0.25)
        self.assertGreater(missing, 0)


if __name__ == "__main__":
    unittest.main()
