"""Second-stage model classes, importable so pickled stage-2 models load in any script."""
from __future__ import annotations

import os
from typing import Any

import numpy as np


class CatBoostStage2:
    """One GPU CatBoost MultiRMSE model over every residual column, columns standardised so the
    large low-order coefficients do not outweigh the rest; predictions are scaled back."""

    def __init__(self, iters: int, depth: int = 6, lr: float = 0.08,
                 scale: np.ndarray | None = None) -> None:
        """`scale`, if given, divides the residual columns instead of their own spread; it is
        renormalised so the scaled residuals average unit variance."""
        self.iters, self.depth, self.lr = iters, depth, lr
        self.fixed_scale = scale
        self.scale: np.ndarray | None = None
        self.model: Any = None

    def fit(self, F: np.ndarray, R: np.ndarray) -> CatBoostStage2:
        from catboost import CatBoostRegressor
        if self.fixed_scale is None:
            self.scale = R.std(0)
        else:
            self.scale = self.fixed_scale * np.sqrt(((R / self.fixed_scale) ** 2).mean())
        if not (self.scale > 0).all():
            raise ValueError(f"residual column with zero spread: {np.flatnonzero(self.scale <= 0)}")
        self.model = CatBoostRegressor(loss_function="MultiRMSE", iterations=self.iters,
                                       learning_rate=self.lr, depth=self.depth, task_type="GPU",
                                       devices="0", verbose=0, boosting_type="Plain", gpu_ram_part=float(os.environ.get("FUSION_GPU_RAM_PART", "0.3")),
                                       random_seed=0)
        self.model.fit(F, R / self.scale)
        return self

    def predict(self, F: np.ndarray) -> np.ndarray:
        assert self.scale is not None
        # CatBoost silently accepts EXTRA columns and reads the leading ones (2026-10-19: full39e's
        # stage, fitted on 242-wide inputs, got 377-wide ones after thomson_chord was added and its
        # private-test corrections came out 4-9x too large). A width mismatch is an error.
        want = len(self.model.feature_names_)
        if F.shape[1] != want:
            raise ValueError(f"stage-2 features: got {F.shape[1]} columns, the trees were fitted on {want}")
        return np.asarray(self.model.predict(F)) * self.scale
