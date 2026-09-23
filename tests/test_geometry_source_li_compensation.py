"""Input-only li target, exact zero control, and explicit invalid-a accounting."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pandas as pd

from tools.geometry_source_li_compensation import (
    _compensate_frame,
    _grid_for_shot,
    _interior_taper,
    _spatial_profile,
)


class FakeDerive:
    def magnetic_axis(self, *_args: Any) -> tuple[float, float, int, int]:
        return 1.7, 0.0, 32, 32

    def _bilinear(
        self, _phi: np.ndarray, _R: np.ndarray, _Z: np.ndarray, rq: np.ndarray, _zq: np.ndarray
    ) -> np.ndarray:
        return np.full(len(rq), 0.5)

    def internal_inductance(self, *_args: Any) -> float:
        return 1.0

    def _points_in_polygon(
        self, rq: np.ndarray, zq: np.ndarray, _contour: np.ndarray
    ) -> np.ndarray:
        return ((rq - 1.7) / 0.4) ** 2 + (zq / 0.8) ** 2 < 1


class FakeLCFS:
    def extract_lcfs_with_sign(self, *_args: Any, **_kwargs: Any) -> tuple[np.ndarray | None, int]:
        theta = np.linspace(0, 2 * np.pi, 512, endpoint=False)
        return np.column_stack([1.7 + 0.4 * np.cos(theta), 0.8 * np.sin(theta)]), -1


class RejectChangesLCFS(FakeLCFS):
    def __init__(self, original: np.ndarray) -> None:
        self.original = original

    def extract_lcfs_with_sign(
        self, candidate: np.ndarray, *_args: Any, **_kwargs: Any
    ) -> tuple[np.ndarray | None, int]:
        if candidate.tobytes() != self.original.tobytes():
            return None, -1
        return super().extract_lcfs_with_sign(candidate)


class FakeContour:
    def symmetric_hausdorff(self, _left: np.ndarray, _right: np.ndarray) -> float:
        return 0.0


class SourceLiCompensationTests(unittest.TestCase):
    def test_grid_loader_requests_input_geometry_only(self) -> None:
        R = np.linspace(0.84, 2.54, 65)
        Z = np.linspace(-1.6, 1.6, 65)
        frame = pd.DataFrame({"efit_grid_R": [R], "efit_grid_Z": [Z]})
        with patch(
            "tools.geometry_source_li_compensation.pd.read_parquet", return_value=frame
        ) as read:
            got_r, got_z = _grid_for_shot(Path("input-only.parquet"))
        self.assertEqual(read.call_args.kwargs["columns"], ["efit_grid_R", "efit_grid_Z"])
        np.testing.assert_array_equal(got_r, R)
        np.testing.assert_array_equal(got_z, Z)

    def test_zero_control_and_all_invalid_nonzero_candidates_are_counted(self) -> None:
        R = np.linspace(0.84, 2.54, 65)
        Z = np.linspace(-1.6, 1.6, 65)
        source = np.full((65, 65), -0.75, dtype=np.float32)
        source[32, 32] = -1.0
        additive = source.copy()
        a_grid = json.loads(
            Path("experiments/geometry-additive-source-li-compensation.json").read_text()
        )["family"]["candidate_a_grid"]
        chosen, detail = _compensate_frame(
            source,
            additive,
            R,
            Z,
            np.ones((65, 65), dtype=bool),
            -1,
            a_grid,
            FakeDerive(),
            RejectChangesLCFS(additive),
            FakeContour(),
        )
        self.assertEqual(chosen.tobytes(), additive.tobytes())
        self.assertEqual(detail["selected_a"], 0.0)
        self.assertEqual(detail["valid_a_count"], 1)
        self.assertEqual(len(detail["rejected"]), 8)
        self.assertEqual({item["reason"] for item in detail["rejected"]}, {"lcfs_missing"})

    def test_fixed_family_is_monotone_and_zero_map_is_bit_exact(self) -> None:
        psi = np.linspace(-1.0, 1.0, 65 * 65, dtype=np.float32).reshape(65, 65)
        self.assertEqual(
            _spatial_profile(psi, -1, 1.0, 0.0, 0.0, np.ones_like(psi)).tobytes(),
            psi.tobytes(),
        )
        u = np.linspace(0, 1, 10001)
        for a in np.linspace(-2, 2, 9):
            derivative = 1 + 2 * a * u * (1 - u) * (1 - 2 * u)
            self.assertGreater(float(derivative.min()), 0.615)

    def test_inside_taper_zero_outside_edge_and_contour(self) -> None:
        R = np.linspace(0.84, 2.54, 65)
        Z = np.linspace(-1.6, 1.6, 65)
        theta = np.linspace(0, 2 * np.pi, 512, endpoint=False)
        contour = np.column_stack([1.7 + 0.4 * np.cos(theta), 0.8 * np.sin(theta)])
        taper, width, inside, seconds = _interior_taper(contour, R, Z, FakeDerive())
        self.assertEqual(taper.shape, (65, 65))
        self.assertEqual(taper[0].tobytes(), np.zeros(65).tobytes())
        self.assertEqual(taper[-1].tobytes(), np.zeros(65).tobytes())
        self.assertEqual(taper[:, 0].tobytes(), np.zeros(65).tobytes())
        self.assertGreater(taper[32, 32], 0.99)
        self.assertGreater(inside, 0)
        self.assertGreater(width, 0)
        self.assertGreater(seconds, 0)


if __name__ == "__main__":
    unittest.main()
