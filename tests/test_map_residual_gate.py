"""Fixed physical-source fallback and decoded two-base arithmetic."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from tools.direct_contour_compact_val_predict import INPUT_COLUMNS
from tools.map_residual_data import _atomic_npz
from tools.map_residual_gate import _mean_one, guard_maps


class MapResidualGateTest(unittest.TestCase):
    def test_target_stripped_column_allowlist(self) -> None:
        forbidden = [name for name in INPUT_COLUMNS if name.startswith("efit_")
                     and name not in ("efit_times", "efit_grid_R", "efit_grid_Z")]
        self.assertEqual(forbidden, [])

    def test_li_guard_counts_explicit_reasons_and_copies_source_bits(self) -> None:
        source = np.zeros((3, 65, 65), dtype=np.float32)
        source[:, 0, 0] = np.array([-0.0, 2.0, 3.0], dtype=np.float32)
        candidate = source.copy()
        candidate[:, 2, 2] = 1.0
        values = [(1.0, None), (1.01, None),
                  (1.0, None), (1.2, None),
                  (None, "li_invalid")]
        grid = np.arange(65, dtype=np.float64)
        with patch("tools.map_residual_gate._li", side_effect=values):
            guarded, reasons, difference = guard_maps(
                source, candidate, grid, grid, np.ones((65, 65), dtype=np.bool_), 0.05)
        self.assertEqual(reasons["abs_li_change_over_threshold"], 1)
        self.assertEqual(reasons["source_li_invalid"], 1)
        self.assertEqual(sum(reasons.values()), 2)
        self.assertEqual(guarded[0].tobytes(), candidate[0].tobytes())
        self.assertEqual(guarded[1].tobytes(), source[1].tobytes())
        self.assertEqual(guarded[2].tobytes(), source[2].tobytes())
        self.assertAlmostEqual(float(difference[0]), 0.01)

    def test_mean2_averages_decoded_float32_once_and_keeps_features(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first, second = root / "a.npz", root / "b.npz"
            features = np.full((2, 84), 3.0, dtype=np.float32)
            for path, value in ((first, 0.1), (second, 0.2)):
                arrays = {
                    "source_psi": np.full((2, 65, 65), value, dtype=np.float32),
                    "selected84": features,
                    "q95": np.full(2, value, dtype=np.float32),
                    "betaN": np.full(2, value, dtype=np.float32),
                }
                _atomic_npz(path, arrays)
            output = root / "out"
            output.mkdir()
            result = _mean_one(("shot.parquet", str(first), str(second), str(output)))
            with np.load(result["output_path"], allow_pickle=False) as saved:
                expected = np.float32((float(np.float32(0.1))
                                       + float(np.float32(0.2))) / 2)
                self.assertEqual(saved["source_psi"][0, 0, 0], expected)
                self.assertEqual(saved["q95"][0], expected)
                self.assertEqual(saved["selected84"].tobytes(), features.tobytes())


if __name__ == "__main__":
    unittest.main()
