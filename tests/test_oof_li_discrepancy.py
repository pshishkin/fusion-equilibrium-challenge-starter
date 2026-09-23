"""The read-only li audit reports absolute differences without altering maps."""

from __future__ import annotations

import unittest

import numpy as np

from tools.oof_li_discrepancy import summarize_absolute


class LiDiscrepancyTests(unittest.TestCase):
    def test_distribution_and_thresholds(self) -> None:
        report = summarize_absolute(np.array([0.0, 1e-4, 1e-3, 0.1]))
        self.assertAlmostEqual(report["mean"], 0.025275)
        self.assertAlmostEqual(report["max"], 0.1)
        self.assertAlmostEqual(report["fraction_at_most_1e_minus_4"], 0.5)
        self.assertAlmostEqual(report["fraction_at_most_1e_minus_3"], 0.75)

    def test_rejects_invalid_discrepancies(self) -> None:
        for values in (np.array([]), np.array([np.nan]), np.array([-0.1])):
            with self.assertRaisesRegex(ValueError, "expected finite nonnegative"):
                summarize_absolute(values)


if __name__ == "__main__":
    unittest.main()
