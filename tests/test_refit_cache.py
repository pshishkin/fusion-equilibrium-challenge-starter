"""Preserve full-shot auxiliary rows and provenance when composing a refit cache."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from my_experiments.refit_cache import (
    _check_name_identity,
    _compose,
    _load_block,
    _publish,
    _verify_producers,
)


class RefitCacheChecks(unittest.TestCase):
    def test_order_lengths_nan_bits_and_existing_target_are_verified(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_a, source_b, target = [root / name for name in ("a.npz", "b.npz", "all.npz")]
            a = np.arange(21, dtype=np.float32).reshape(3, 7)
            a[1, 2] = np.nan
            b = (np.arange(14, dtype=np.float32) + 100).reshape(2, 7)
            b[0, 4] = np.nan
            np.savez_compressed(source_a, cons=a, lengths=np.array([2, 1], dtype=np.int64))
            np.savez_compressed(source_b, cons=b, lengths=np.array([2], dtype=np.int64))
            first, second = _load_block(source_a, 2), _load_block(source_b, 1)
            composed = _compose(first, second)
            self.assertEqual(composed[1].tolist(), [2, 1, 2])
            expected_shots = (a[:2], a[2:3], b[:2])
            offsets = np.r_[0, np.cumsum(composed[1])]
            for i, expected in enumerate(expected_shots):
                shot = composed[0][offsets[i]:offsets[i + 1]]
                self.assertEqual(shot.tobytes(), expected.tobytes())
                self.assertTrue(np.array_equal(shot, expected, equal_nan=True))
            self.assertEqual(_publish(target, composed, 3), "published")
            self.assertEqual(_publish(target, composed, 3), "already_present_verified")
            self.assertEqual(_load_block(target, 3)[0].tobytes(), composed[0].tobytes())
            changed = composed[0].copy()
            changed[0, 0] += 1
            with self.assertRaisesRegex(ValueError, "existing cons differs"):
                _publish(target, (changed, composed[1]), 3)

    def test_bad_block_shapes_and_file_identity_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bad = root / "bad.npz"
            np.savez_compressed(bad, cons=np.zeros((2, 7), dtype=np.float64),
                                lengths=np.array([2], dtype=np.int64))
            with self.assertRaisesRegex(ValueError, "float32"):
                _load_block(bad, 1)
            np.savez_compressed(bad, cons=np.zeros((2, 7), dtype=np.float32),
                                lengths=np.array([1], dtype=np.int64))
            with self.assertRaisesRegex(ValueError, "lengths sum"):
                _load_block(bad, 1)
        _check_name_identity(["a", "b"], ["c"], ["a", "b", "c"], ["d"])
        with self.assertRaisesRegex(ValueError, "ordered files"):
            _check_name_identity(["a", "b"], ["c"], ["a", "c", "b"], ["d"])
        with self.assertRaisesRegex(ValueError, "overlap"):
            _check_name_identity(["a", "b"], ["c"], ["a", "b", "c"], ["c"])
        with self.assertRaisesRegex(ValueError, "distinct"):
            _check_name_identity(["a", "b"], ["b"], ["a", "b", "b"], ["d"])

    def test_producer_identity_checks_every_code_and_mask_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            roots = (Path(tmp) / "control", Path(tmp) / "refit", Path(tmp) / "working")
            rels = (
                "my_experiments/aux_targets.py", "my_experiments/target_metric.py",
                "local_score.py", "experiments.py", "uv.lock",
                "fusion_scoring/derive.py", "fusion_scoring/masks/d3d_envelope.npz",
            )
            for root in roots:
                for rel in rels:
                    path = root / rel
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(rel.encode())
            self.assertEqual(len(_verify_producers(roots)), len(rels))
            (roots[1] / "fusion_scoring/derive.py").write_text("changed")
            with self.assertRaisesRegex(ValueError, "derive.py differs"):
                _verify_producers(roots)


if __name__ == "__main__":
    unittest.main()
