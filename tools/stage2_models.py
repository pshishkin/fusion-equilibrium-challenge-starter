"""Second-stage model classes, importable so pickled stage-2 models load in any script."""
from __future__ import annotations

from typing import Any

import numpy as np


class CatBoostStage2:
    """One GPU CatBoost MultiRMSE model over every residual column, columns standardised so the
    large low-order coefficients do not outweigh the rest; predictions are scaled back."""

    def __init__(self, iters: int, depth: int = 6, lr: float = 0.08) -> None:
        self.iters, self.depth, self.lr = iters, depth, lr
        self.scale: np.ndarray | None = None
        self.model: Any = None

    def fit(self, F: np.ndarray, R: np.ndarray) -> CatBoostStage2:
        from catboost import CatBoostRegressor
        self.scale = R.std(0)
        if not (self.scale > 0).all():
            raise ValueError(f"residual column with zero spread: {np.flatnonzero(self.scale <= 0)}")
        self.model = CatBoostRegressor(loss_function="MultiRMSE", iterations=self.iters,
                                       learning_rate=self.lr, depth=self.depth, task_type="GPU",
                                       devices="0", verbose=0, boosting_type="Plain",
                                       random_seed=0)
        self.model.fit(F, R / self.scale)
        return self

    def predict(self, F: np.ndarray) -> np.ndarray:
        assert self.scale is not None
        return np.asarray(self.model.predict(F)) * self.scale
