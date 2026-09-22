"""Float16 export audit changes only flux maps and rejects overflow."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from toolkit.records import file_hash, write_json
from tools.historical_compare import _save
from tools.precision_audit import quantize_flux


class PrecisionAuditChecks(unittest.TestCase):
    def test_flux_cast_keeps_scalar_bits_and_fails_on_overflow(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            source = folder / "source.npz"
            target = folder / "half.npz"
            psi = np.full((2, 65, 65), 0.123456, dtype=np.float32)
            q95 = np.array([2.123456, 3.987654], dtype=np.float32)
            beta = np.array([0.333333, 0.777777], dtype=np.float32)
            arrays = {"shot_0000_psirz": psi, "shot_0000_q95": q95,
                      "shot_0000_betaN": beta}
            _save(source, arrays)
            write_json(source.with_suffix(".json"), {
                "files": ["shot.parquet"], "prediction_sha256": file_hash(source),
            })
            result = quantize_flux(source, target)
            self.assertEqual(result["flux_values_changed"], psi.size)
            self.assertTrue(result["scalar_bits_unchanged"])
            with np.load(target, allow_pickle=False) as packed:
                self.assertEqual(packed["shot_0000_psirz"].dtype, np.float16)
                self.assertEqual(packed["shot_0000_q95"].tobytes(), q95.tobytes())
                self.assertEqual(packed["shot_0000_betaN"].tobytes(), beta.tobytes())
            overflow = folder / "overflow.npz"
            arrays["shot_0000_psirz"][0, 0, 0] = 1e10
            _save(overflow, arrays)
            write_json(overflow.with_suffix(".json"), {
                "files": ["shot.parquet"], "prediction_sha256": file_hash(overflow),
            })
            rejected = folder / "rejected.npz"
            with self.assertRaisesRegex(ValueError, "produced 1 nonfinite"):
                quantize_flux(overflow, rejected)
            self.assertFalse(rejected.exists())


if __name__ == "__main__":
    unittest.main()
