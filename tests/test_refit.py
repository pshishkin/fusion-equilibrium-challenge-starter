"""Fixed-step refit must keep the last update without reading a validation block."""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from my_experiments import baseline_model
from my_experiments.baseline_model import parse_share, split_train_val
from my_experiments.models import TorchMLPModel, load_params
from my_experiments.seq_model import TorchSeqModel

PARAMS = Path(__file__).resolve().parent.parent / "baselines/iteration.params.yaml"


class RefitSplitChecks(unittest.TestCase):
    def test_zero_validation_requires_explicit_mode_and_preserves_tail(self) -> None:
        with self.assertRaisesRegex(SystemExit, "shot share"):
            parse_share("0/1.0")
        self.assertEqual(parse_share("0/1.0", allow_zero_shots=True), (0.0, 1.0))
        files = [Path(f"shot_{i:04d}.parquet") for i in range(7041)]
        with self.assertRaisesRegex(SystemExit, "explicitly selected"):
            split_train_val(files, 0.99, 0.0)
        train, val = split_train_val(files, 0.99, 0.0, allow_empty_validation=True)
        self.assertEqual((len(train), len(val)), (6971, 0))
        self.assertEqual(len(set(train) & set(files[-70:])), 0)

    def test_configuration_fails_before_reading_shots(self) -> None:
        with (
            patch.object(baseline_model, "require_artifact_path",
                         return_value=Path("/tmp/test.joblib")),
            patch("my_experiments.data_check.check_dataset", return_value={"files": 7041}),
            patch.object(baseline_model, "sorted_shots", side_effect=AssertionError("read shots")),
            self.assertRaisesRegex(ValueError, "patience_steps"),
        ):
            baseline_model.train("0.99/1.0", "0/1.0", Path("/tmp"), "diii_d_train",
                                 PARAMS, only=["mlp"], fit_mode="fixed_steps")


class RefitModelChecks(unittest.TestCase):
    def test_mlp_keeps_last_step_and_rejects_implicit_empty_validation(self) -> None:
        model = load_params(PARAMS, only=["mlp"], sets=[
            "models.mlp.device=cpu", "models.mlp.threads=1",
            "models.mlp.hidden_sizes=[8, 8]", "models.mlp.batch_size=4",
            "models.mlp.max_steps=3", "models.mlp.eval_every_steps=2",
            "models.mlp.patience_steps=0", "models.mlp.head_weight=0.0",
        ]).models["mlp"]
        self.assertIsInstance(model, TorchMLPModel)
        assert isinstance(model, TorchMLPModel)
        x = np.arange(24, dtype=np.float64).reshape(8, 3) / 24
        y = np.sin(np.arange(32, dtype=np.float64).reshape(8, 4))
        xv, yv = np.empty((0, 3)), np.empty((0, 4))
        with self.assertRaisesRegex(ValueError, "validation mode requires"):
            model.fit(x, y, xv, yv)
        model.fit(x, y, xv, yv, fixed_steps=True)
        self.assertEqual(model._best_step, 3)
        self.assertEqual(model._ran_steps, 3)
        self.assertIn("kept last step 3", model.fit_report())
        self.assertTrue(np.isfinite(model.predict(x)).all())

    def test_mlp_fixed_steps_with_static_functional_head(self) -> None:
        model = load_params(PARAMS, only=["mlp"], sets=[
            "models.mlp.device=cpu", "models.mlp.threads=1",
            "models.mlp.hidden_sizes=[8, 8]", "models.mlp.batch_size=4",
            "models.mlp.max_steps=2", "models.mlp.eval_every_steps=2",
            "models.mlp.patience_steps=0", "models.mlp.head_weight=0.1",
            "models.mlp.head_hidden=8", "models.mlp.head_frozen=false",
        ]).models["mlp"]
        self.assertIsInstance(model, TorchMLPModel)
        assert isinstance(model, TorchMLPModel)
        x = np.arange(24, dtype=np.float64).reshape(8, 3) / 24
        y = np.sin(np.arange(32, dtype=np.float64).reshape(8, 4))
        head = np.column_stack([np.sin(np.arange(8) + j) for j in range(7)])
        model.fit(x, y, np.empty((0, 3)), np.empty((0, 4)), head_y=head,
                  head_n_pca=2, fixed_steps=True)
        self.assertEqual(model._best_step, 2)
        self.assertTrue(np.isfinite(model.predict(x)).all())

    def test_sequence_keeps_last_step_without_validation(self) -> None:
        model = load_params(PARAMS, only=["seq"], sets=[
            "models.seq.device=cpu", "models.seq.threads=1",
            "models.seq.hidden_sizes=[8, 8]", "models.seq.seq_hidden=8",
            "models.seq.batch_shots=2", "models.seq.max_steps=3",
            "models.seq.eval_every_steps=2", "models.seq.patience_steps=0",
        ]).models["seq"]
        self.assertIsInstance(model, TorchSeqModel)
        assert isinstance(model, TorchSeqModel)
        x = np.arange(18, dtype=np.float64).reshape(6, 3) / 18
        y = np.sin(np.arange(24, dtype=np.float64).reshape(6, 4))
        xv, yv = np.empty((0, 3)), np.empty((0, 4))
        with self.assertRaisesRegex(ValueError, "validation mode requires"):
            model.fit(x, y, xv, yv, shots=np.array([3, 3]), shots_val=np.array([]))
        model.fit(x, y, xv, yv, shots=np.array([3, 3]),
                  shots_val=np.array([], dtype=np.int64), fixed_steps=True)
        self.assertEqual(model._best_step, 3)
        self.assertEqual(model._ran_steps, 3)
        self.assertIn("on training shots", model.fit_report())
        self.assertTrue(np.isfinite(model.predict(x[:3])).all())


if __name__ == "__main__":
    unittest.main()
