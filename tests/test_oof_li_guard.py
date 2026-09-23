"""The fixed li guard copies complete frame bytes and has a strict threshold."""

from __future__ import annotations

import unittest

import numpy as np

from tools.oof_li_guard import guarded_maps


class OofLiGuardTests(unittest.TestCase):
    def test_only_strictly_excessive_frames_use_source_bytes(self) -> None:
        source = np.zeros((3, 65, 65), dtype=np.float32)
        candidate = np.ones_like(source)
        source[0, 0, 0] = np.float32(-0.0)
        before = np.array([0.0, 0.0, 0.0])
        after = np.array([0.06, 0.05, 0.01])
        guarded, changed = guarded_maps(source, candidate, before, after, 0.05)
        self.assertEqual(changed.tolist(), [True, False, False])
        self.assertEqual(guarded[0].tobytes(), source[0].tobytes())
        self.assertEqual(guarded[1].tobytes(), candidate[1].tobytes())
        self.assertEqual(guarded[2].tobytes(), candidate[2].tobytes())

    def test_nonfinite_li_is_a_failure(self) -> None:
        maps = np.zeros((1, 65, 65), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "expected matched finite"):
            guarded_maps(maps, maps, np.array([np.nan]), np.array([1.0]), 0.05)


if __name__ == "__main__":
    unittest.main()
