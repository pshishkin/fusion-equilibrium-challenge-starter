"""Fixed old/new block arithmetic uses one float64 accumulation per output."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.cross_salt_blend import _decoded_block
from tools.cross_salt_report import _fit_stages


class CrossSaltBlendTest(unittest.TestCase):
    def test_half_blocks_have_fixed_weights_and_scalar_anchor(self) -> None:
        old = [np.array([float(index)], dtype=np.float32) for index in range(6)]
        new = [np.array([float(index + 6)], dtype=np.float32) for index in range(8)]
        old_block, new_block, mixed = _decoded_block(old, new, None)
        expected_old = sum(float(array[0]) * weight for array, weight in
                           zip(old, (5, 1, 1, 1, 2, 1), strict=True)) / 11
        expected_new = sum(float(array[0]) for array in new) / 8
        self.assertEqual(old_block[0], np.float32(expected_old))
        self.assertEqual(new_block[0], np.float32(expected_new))
        self.assertEqual(mixed[0], np.float32((expected_old + expected_new) / 2))
        prior_scalar = np.array([0.12345679], dtype=np.float32)
        scalar_old, scalar_new, scalar_mixed = _decoded_block(old, new, prior_scalar)
        self.assertEqual(scalar_old.tobytes(), prior_scalar.tobytes())
        self.assertEqual(scalar_new.tobytes(), new_block.tobytes())
        self.assertEqual(scalar_mixed[0],
                         np.float32((float(prior_scalar[0]) + expected_new) / 2))

    def test_shape_mismatch_fails(self) -> None:
        arrays = [np.ones(2, dtype=np.float32) for _ in range(6)]
        candidates = [np.ones(2, dtype=np.float32) for _ in range(7)]
        candidates.append(np.ones(3, dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "shapes differ"):
            _decoded_block(arrays, candidates, None)

    def test_worker_stage_times_are_read_from_native_log(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            log = folder / "run.log"
            log.write_text("mlp: fitted in 12.5 s, best step 3\n"
                           "seq: fitted in 30.0 s, best step 4\n")
            (folder / "metrics.json").write_text(json.dumps({
                "timing_seconds": {"train": 60.0, "score": 15.0}}))
            (folder / "status.json").write_text(json.dumps({
                "state": "scored", "wall_s": 76.0, "log": str(log)}))
            result = _fit_stages(folder)
            self.assertEqual(result["mlp_fit"], 12.5)
            self.assertEqual(result["seq_fit"], 30.0)
            self.assertEqual(result["other_train_read_preparation"], 17.5)


if __name__ == "__main__":
    unittest.main()
