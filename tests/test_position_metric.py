"""B11 matched metric algebra, frame mapping, preflight, and tiny MLP training."""

from __future__ import annotations

import copy
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from my_experiments.models import TorchMLPModel, load_params
from my_experiments.position_metric import (
    _assemble_metrics,
    _selector,
    frame_groups,
    normalize_config,
    validate_preflight,
)

PARAMS = Path("baselines/position-metric.params.yaml")


class PositionMetricChecks(unittest.TestCase):
    def test_physical_quadratic_and_weighted_global_equivalence(self) -> None:
        rng = np.random.default_rng(7)
        matrices = rng.normal(size=(3, 5, 5))
        raw = np.stack([matrix @ matrix.T for matrix in matrices])
        fractions = np.asarray([0.2, 0.5, 0.3])
        result = _assemble_metrics(raw, fractions, 0.5, 1.7, 1.0)
        equalized = np.asarray(result["equalized_consistency_forms"])
        traces = np.trace(equalized, axis1=1, axis2=2)
        np.testing.assert_allclose(traces, np.full(3, result["common_trace"]),
                                   rtol=1e-12, atol=1e-12)
        global_metric = np.asarray(result["global_metric"])
        group_metrics = np.asarray(result["group_metrics"])
        scaled = np.asarray(result["scaled_group_metrics"])
        factors = np.asarray(result["scaled_loss_factors"])
        np.testing.assert_allclose(np.einsum("g,gij->ij", fractions, group_metrics),
                                   global_metric, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(np.einsum("g,gij->ij", fractions, scaled), np.eye(5),
                                   rtol=1e-12, atol=1e-12)
        for group in range(3):
            error_coeff = rng.normal(size=5)
            scaled_error = error_coeff @ np.asarray(result["global_factor"])
            physical = error_coeff @ group_metrics[group] @ error_coeff
            rotated = np.sum((scaled_error @ factors[group]) ** 2)
            self.assertAlmostEqual(float(physical), float(rotated), places=10)

    def test_kept_frame_indices_and_singleton(self) -> None:
        groups = frame_groups(np.asarray([1, 2, 3, 10], dtype=np.int64), (0.3, 0.7))
        self.assertEqual(groups.tolist()[:6], [1, 0, 2, 0, 1, 2])
        keep, full_counts, kept, raw = _selector(np.asarray([0.1, 0.5, 0.9]),
                                                  10, 0.5, 10, "shot.parquet")
        selected = keep(20)
        self.assertEqual(full_counts, [20])
        self.assertEqual(kept[0].tolist(), [1, 4, 8])
        np.testing.assert_array_equal(selected, raw[0])
        self.assertEqual(len(selected), 3)
        mismatch, _, _, _ = _selector(np.asarray([0.1, 0.5, 0.9]),
                                       10, 0.5, 9, "bad.parquet")
        with self.assertRaisesRegex(ValueError, "bad.parquet.*10 frames.*9"):
            mismatch(20)

    def test_explicit_config_and_unsupported_preflight(self) -> None:
        params = load_params(PARAMS, 0, ["mlp"])
        assert params.position_metric is not None
        self.assertEqual(params.position_metric["mode"], "pooled")
        validate_preflight(params, "validation")
        with self.assertRaisesRegex(ValueError, "validation mode"):
            validate_preflight(params, "fixed_steps")
        wrong = copy.deepcopy(params.position_metric)
        wrong["boundaries"] = [0.2, 0.7]
        with self.assertRaisesRegex(ValueError, "midpoints"):
            normalize_config(wrong, "test")
        template = params.models["mlp"]
        assert isinstance(template, TorchMLPModel)
        huber = replace(template, loss="huber")
        with self.assertRaisesRegex(ValueError, "loss=mse"):
            validate_preflight(replace(params, models={"mlp": huber}), "validation")
        adaptive = replace(template, head_cons_weights="adaptive")
        with self.assertRaisesRegex(ValueError, "trainable static functional head"):
            validate_preflight(replace(params, models={"mlp": adaptive}), "validation")
        with self.assertRaisesRegex(ValueError, "only the torch MLP"):
            validate_preflight(replace(params, models={"ridge": params.models["mlp"]}),
                               "validation")

    def test_global_none_path_and_conditioned_cpu_head(self) -> None:
        params = load_params(PARAMS, 0, ["mlp"])
        template = params.models["mlp"]
        assert isinstance(template, TorchMLPModel)
        rng = np.random.default_rng(13)
        x = rng.normal(size=(24, 8)).astype(np.float32)
        y = rng.normal(size=(24, 5)).astype(np.float32)
        xv = rng.normal(size=(12, 8)).astype(np.float32)
        yv = rng.normal(size=(12, 5)).astype(np.float32)
        ordinary = replace(template, hidden_sizes=[12], max_steps=2, patience_steps=2,
                           eval_every_steps=1, batch_size=8, device="cpu", threads=1,
                           head_weight=0.0, ema_decay=0.0)
        explicit_none = replace(ordinary)
        ordinary.fit(x, y, xv, yv, head_n_pca=3)
        explicit_none.fit(x, y, xv, yv, head_n_pca=3,
                          position_factors=None, position_groups=None)
        self.assertEqual(ordinary._best_step, explicit_none._best_step)
        assert ordinary._state is not None and explicit_none._state is not None
        for key, value in ordinary._state.items():
            np.testing.assert_array_equal(value, explicit_none._state[key])

        conditioned = replace(template, hidden_sizes=[12], head_hidden=8,
                              max_steps=2, patience_steps=2, eval_every_steps=1,
                              batch_size=8, device="cpu", threads=1,
                              head_weight=0.1, ema_decay=0.0)
        head = rng.normal(size=(24, 7)).astype(np.float32)
        head_val = rng.normal(size=(12, 7)).astype(np.float32)
        factors = np.stack([np.eye(3), np.diag([1.1, 0.9, 1.0]),
                            np.diag([0.9, 1.1, 1.0])])
        groups = np.tile(np.asarray([0, 1, 2], dtype=np.int64), 8)
        conditioned.fit(x, y, xv, yv, head_y=head, head_y_val=head_val,
                        head_n_pca=3, position_factors=factors, position_groups=groups)
        self.assertEqual(conditioned._ran_steps, 2)
        self.assertGreaterEqual(conditioned._best_step, 1)
        self.assertEqual(conditioned.predict(x[:3]).shape, (3, 5))
        with self.assertRaisesRegex(ValueError, "supplied together"):
            replace(conditioned).fit(x, y, xv, yv, head_n_pca=3,
                                     position_factors=factors)


if __name__ == "__main__":
    unittest.main()
