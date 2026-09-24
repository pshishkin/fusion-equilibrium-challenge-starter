"""D8 shot summaries and role split contain no target-dependent choices."""

from __future__ import annotations

import unittest

import numpy as np

from tools.d8_nonlinear_gate import _roles, shot_summary


class D8NonlinearGateTests(unittest.TestCase):
    def test_population_mean_std_and_one_frame_duration(self) -> None:
        frame = np.arange(84, dtype=np.float32)
        single = shot_summary(frame[None, :], 0.0)
        self.assertEqual(single.shape, (169,))
        np.testing.assert_array_equal(single[:84], frame)
        self.assertEqual(np.count_nonzero(single[84:]), 0)
        two = shot_summary(np.stack([frame, frame + 2]), 20.0)
        np.testing.assert_array_equal(two[:84], frame + 1)
        np.testing.assert_array_equal(two[84:168], np.ones(84))
        self.assertEqual(two[-1], 20.0)

    def test_fixed_shot_hash_roles_do_not_depend_on_input_order(self) -> None:
        names = [f"shot_{index:04d}.parquet" for index in range(1355)]
        fit, held = _roles(names)
        other_fit, other_held = _roles(names[::-1])
        self.assertEqual((fit, held), (other_fit, other_held))
        self.assertEqual((len(fit), len(held)), (1100, 255))
        self.assertFalse(set(fit) & set(held))
        with self.assertRaisesRegex(ValueError, "expected 1355"):
            _roles(names[:-1])


if __name__ == "__main__":
    unittest.main()
