"""Decoded historical blends keep the registered shot order and fixed member weights."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.historical_compare import _blend_arrays, _blend_many_arrays, _save, blend, blend_many


class HistoricalBlendChecks(unittest.TestCase):
    def test_five_to_one_blend_and_shot_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            left, right, out = (folder / name for name in ("old.npz", "candidate.npz", "mix.npz"))
            names = ["shot_a.parquet", "shot_b.parquet"]
            for path, value in ((left, 2.0), (right, 8.0)):
                arrays: dict[str, np.ndarray] = {}
                for i in range(len(names)):
                    arrays[f"shot_{i:04d}_psirz"] = np.full((2, 65, 65), value, np.float32)
                    arrays[f"shot_{i:04d}_q95"] = np.full(2, value, np.float32)
                    arrays[f"shot_{i:04d}_betaN"] = np.full(2, value, np.float32)
                _save(path, arrays)
                path.with_suffix(".json").write_text(json.dumps({"files": names}))
            blend(left, right, out, 5, 1)
            with np.load(out, allow_pickle=False) as packed:
                self.assertEqual(len(packed.files), 6)
                for key in packed.files:
                    self.assertTrue(np.array_equal(packed[key], np.full_like(packed[key], 3.0)))
            record = json.loads(out.with_suffix(".json").read_text())
            self.assertEqual(record["weights"], {"baseline_members": 5,
                                                  "candidate_members": 1})
            self.assertEqual(record["files"], names)
            wrong = json.loads(right.with_suffix(".json").read_text())
            wrong["files"] = list(reversed(names))
            right.with_suffix(".json").write_text(json.dumps(wrong))
            with self.assertRaisesRegex(ValueError, "scored filenames differ"):
                blend(left, right, folder / "bad.npz", 5, 1)

    def test_blend_rejects_bad_weights_and_nonfinite_maps(self) -> None:
        x = np.array([1.0, 2.0], dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "shapes/finiteness"):
            _blend_arrays(x, np.array([np.nan, 2.0], np.float32), 5, 1)
        with self.assertRaisesRegex(ValueError, "shapes/finiteness"):
            _blend_arrays(x, np.array([1.0], np.float32), 5, 1)

    def test_single_cast_multi_blend_and_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            files = [folder / f"input_{i}.npz" for i in range(4)]
            values = [2.0, 8.0, 10.0, 12.0]
            for path, value in zip(files, values, strict=True):
                _save(path, {"shot_0000_psirz": np.full((1, 65, 65), value, np.float32),
                             "shot_0000_q95": np.array([value], np.float32),
                             "shot_0000_betaN": np.array([value], np.float32)})
                path.with_suffix(".json").write_text(json.dumps({
                    "files": ["shot.parquet"],
                    "prediction_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }))
            output = folder / "stack.npz"
            blend_many(files, [5, 1, 1, 1], output)
            with np.load(output, allow_pickle=False) as packed:
                for key in packed.files:
                    self.assertTrue(np.array_equal(packed[key], np.full_like(packed[key], 5.0)))
            x = np.array([0.992149, 1.0, -3.0], dtype=np.float32)
            self.assertEqual(_blend_many_arrays([x] * 4, [5, 1, 1, 1]).tobytes(),
                             x.tobytes())


if __name__ == "__main__":
    unittest.main()
