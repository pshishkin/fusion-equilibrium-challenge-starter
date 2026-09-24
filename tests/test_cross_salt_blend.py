"""Fixed old/new block arithmetic uses one float64 accumulation per output."""

from __future__ import annotations

import unittest

import numpy as np

from tools.cross_salt_blend import _decoded_block


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


if __name__ == "__main__":
    unittest.main()
