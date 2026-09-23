"""The public flux precision probe may change only psi dtype and values."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from toolkit.records import file_hash
from tools import precision_probe
from tools.precision_probe import build


class PrecisionProbeTests(unittest.TestCase):
    def _case(self, folder: Path, psi_value: float = 0.123456) -> tuple[Path, Path, Path]:
        source = folder / "source.npz"
        target = folder / "half.npz"
        arrays = {
            "shot_0000_psirz": np.full((2, 65, 65), psi_value, dtype=np.float32),
            "shot_0000_q95": np.array([2.123456, 3.987654], dtype=np.float32),
            "shot_0000_betaN": np.array([0.333333, 0.777777], dtype=np.float32),
        }
        np.savez_compressed(source, **arrays)  # type: ignore[arg-type]
        manifest = folder / "manifest.json"
        manifest.write_text(json.dumps({"files": ["shot.parquet"], "frames": [2]}))
        source_record = folder / "source.json"
        source_record.write_text(
            json.dumps(
                {
                    "files": ["shot.parquet"],
                    "prediction_sha256": file_hash(source),
                }
            )
        )
        recipe = folder / "recipe.json"
        recipe.write_text("{}\n")
        spec = folder / "spec.json"
        spec.write_text(
            json.dumps(
                {
                    "transform": "psirz_float32_to_float16_only",
                    "producer_sha256": file_hash(Path(precision_probe.__file__)),
                    "recipe": str(recipe),
                    "recipe_sha256": file_hash(recipe),
                    "held70": {
                        "source": str(source),
                        "source_sha256": file_hash(source),
                        "target": str(target),
                        "manifest": str(manifest),
                        "manifest_sha256": file_hash(manifest),
                        "source_record": str(source_record),
                        "source_record_sha256": file_hash(source_record),
                        "shots": 1,
                        "frames": 2,
                    },
                }
            )
        )
        return spec, source, target

    def test_only_flux_cast_and_scalar_bits_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            spec, source, target = self._case(Path(folder))
            result = build(spec, "held70")
            self.assertTrue(result["scalar_bits_unchanged"])
            self.assertEqual(result["flux_values_total"], 2 * 65 * 65)
            self.assertEqual(
                result["dtypes"],
                {
                    "psirz": "float16",
                    "q95": "float32",
                    "betaN": "float32",
                },
            )
            with (
                np.load(source, allow_pickle=False) as before,
                np.load(target, allow_pickle=False) as after,
            ):
                self.assertEqual(list(before.files), list(after.files))
                np.testing.assert_array_equal(
                    after["shot_0000_psirz"], before["shot_0000_psirz"].astype(np.float16)
                )
                for field in ("q95", "betaN"):
                    key = f"shot_0000_{field}"
                    self.assertEqual(after[key].tobytes(), before[key].tobytes())

    def test_overflow_fails_without_publishing(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            spec, _, target = self._case(Path(folder), psi_value=1e10)
            with self.assertRaisesRegex(ValueError, "float16 cast produced"):
                build(spec, "held70")
            self.assertFalse(target.exists())

    def test_wrong_input_hash_fails_without_publishing(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            spec, source, target = self._case(Path(folder))
            with source.open("ab") as stream:
                stream.write(b"changed")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                build(spec, "held70")
            self.assertFalse(target.exists())


if __name__ == "__main__":
    unittest.main()
