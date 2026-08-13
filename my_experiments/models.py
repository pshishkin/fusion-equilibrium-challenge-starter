#!/usr/bin/env python3
"""
The model zoo and the params.yaml that configures it.

Every model here solves the same problem: 21 magnetics features on the EFIT time base ->
the stacked target vector

    [ pca_0 ... pca_{n_pca-1}, q95, betaN ]

so they are interchangeable, comparable, and averageable. Nothing here knows about psi(R,Z) —
`baseline_model.py` owns the PCA that turns those coefficients back into a flux map. A model
receives scaled features and scaled targets and returns predictions in that same scaled space;
`TargetScaler` below defines it, once, for all of them.

Three families, chosen to be genuinely different rather than three flavours of the same
assumption: a linear map (`ridge`), gradient-boosted trees (`catboost`), and a small neural net
(`torch_mlp`). Their average is the `ensemble`.

Hyper-parameters come from params.yaml and only from there — no defaults in the code, so there is
never a second value to keep in sync. An unknown key is a TypeError, not a shrug.
"""
from __future__ import annotations

import abc
import itertools
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import yaml
from sklearn.linear_model import Ridge
from tqdm import tqdm

# The composite's own weights, read from the vendored scorer rather than copied here: the
# metric_aligned scaling below is only as correct as these numbers, and a hard-coded 0.55 would go
# stale the day the organizers reweight the leaderboard.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "fusion_scoring"))
from common import N_SCALARS, W_PSI, W_QB

FloatArray = npt.NDArray[np.floating[Any]]

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PARAMS_PATH = REPO_ROOT / "params.yaml"

# How often the MLP writes a train/val line above its progress bar. A logging cadence, not a
# hyper-parameter, so it stays in code — CatBoost's own `verbose` is set the same way.
MLP_LOG_EVERY = 100


# --------------------------------------------------------------------------- target scaling

@dataclass
class TargetScaler:
    """Centres and scales the target vector once, for every model alike.

    Not configurable, and deliberately so — this is the one scaling the metric asks for:

    * **One shared divisor for the whole PCA block.** The basis is orthonormal, so by Parseval the
      pixel error of a flux map is the UNWEIGHTED sum of the coefficient errors,
      ||dpsi||^2 = sum_k dc_k^2. A single divisor preserves that geometry exactly, so minimizing
      the loss is minimizing R2_psi. Standardizing each component separately would instead
      minimize sum_k dc_k^2 / lambda_k — a whitened loss that hands component 50, worth ~1e-5 of
      component 1, the same weight as component 1.
    * **Each of the two scalars on its own std.** Their term of the metric is an R2 per scalar,
      each against its own variance, so per-target scaling is what matches it.
    * **Both blocks weighted as the composite weights them**, W_PSI against W_QB / 2, imported
      from the scorer rather than copied. The summed squared error over the target vector is then

          W_PSI * SS_res_psi/SS_tot_psi  +  (W_QB/2) * sum_j SS_res_j/SS_tot_j

      — the differentiable half of the composite itself, term for term. It lands near 71/29
      between psi and the scalars, not 79/21, because SS_tot_psi is measured against a single FLAT
      mean rather than against the mean image.

    Finally the whole vector is rescaled by one constant to average unit variance: that changes no
    ratio above and keeps the numbers where an optimizer is comfortable.

    The alternatives were measured before being discarded; see README, "Target scaling".
    """

    n_pca: int
    _center: FloatArray | None = field(default=None, init=False, repr=False)
    _scale: FloatArray | None = field(default=None, init=False, repr=False)

    def fit(self, Y: FloatArray, psi_ss_tot_per_frame: float) -> TargetScaler:
        """`psi_ss_tot_per_frame` is the metric's own denominator, per frame: the mean over frames
        of sum_pixels (psi - m)^2 with m the single flat mean of the training flux."""
        if Y.ndim != 2 or Y.shape[1] <= self.n_pca:
            raise ValueError(f"targets {Y.shape}, expected (n_frames, {self.n_pca} + 2)")
        if not psi_ss_tot_per_frame > 0:
            raise ValueError(f"psi_ss_tot_per_frame must be positive, got {psi_ss_tot_per_frame!r}")
        center = Y.mean(axis=0)
        std = Y.std(axis=0)
        if not (std > 0).all():
            flat = np.nonzero(std <= 0)[0].tolist()
            raise ValueError(f"target column(s) {flat} are constant over the training frames — "
                             f"nothing to fit and nothing to scale by")

        scale = np.concatenate([
            np.full(self.n_pca, float(np.sqrt(psi_ss_tot_per_frame / W_PSI))),
            std[self.n_pca:] / np.sqrt(W_QB / N_SCALARS),
        ])
        # One constant over the whole vector: it moves every dimension by the same factor, so the
        # block ratio above survives, and the targets land at average unit variance.
        scale = scale * float(np.sqrt(((std / scale) ** 2).mean()))

        self._center, self._scale = center, scale
        return self

    def _fitted(self) -> tuple[FloatArray, FloatArray]:
        if self._center is None or self._scale is None:
            raise RuntimeError("TargetScaler used before fit()")
        return self._center, self._scale

    def transform(self, Y: FloatArray) -> FloatArray:
        center, scale = self._fitted()
        return (Y - center) / scale

    def inverse_transform(self, Y: FloatArray) -> FloatArray:
        center, scale = self._fitted()
        return Y * scale + center


# --------------------------------------------------------------------------- the interface

class TargetModel(abc.ABC):
    """One regressor from scaled features (n_frames, 21) to targets (n_frames, n_pca + 2)."""

    @abc.abstractmethod
    def fit(self, X: FloatArray, Y: FloatArray, X_val: FloatArray, Y_val: FloatArray) -> None:
        """Fit on the training frames, stopping on the validation ones.

        The validation set is mandatory, not optional: a model that quietly trains to its full
        iteration count because nobody passed one is exactly the silent failure AGENTS.md bans.
        A model with nothing to stop on (ridge) says so in its docstring and ignores it.
        """

    @abc.abstractmethod
    def predict(self, X: FloatArray) -> FloatArray:
        """(n_frames, n_targets) predictions, float64."""

    @property
    @abc.abstractmethod
    def kind(self) -> str:
        """The `type:` string in params.yaml that builds this model."""

    def fit_report(self) -> str:
        """Where early stopping landed, appended to the training line. Empty when there is none."""
        return ""

    def _check_fit_input(self, X: FloatArray, Y: FloatArray,
                         X_val: FloatArray | None = None,
                         Y_val: FloatArray | None = None) -> None:
        if X_val is not None and Y_val is not None:
            if X_val.shape[1] != X.shape[1] or Y_val.shape[1] != Y.shape[1]:
                raise ValueError(f"{self.kind}: validation set is {X_val.shape}/{Y_val.shape}, "
                                 f"training set is {X.shape}/{Y.shape}")
            self._check_fit_input(X_val, Y_val)
        if X.ndim != 2 or Y.ndim != 2:
            raise ValueError(f"{self.kind}: expected 2-D X and Y, got {X.shape} and {Y.shape}")
        if len(X) != len(Y):
            raise ValueError(f"{self.kind}: {len(X)} feature rows against {len(Y)} target rows")
        if not np.isfinite(X).all():
            raise ValueError(f"{self.kind}: X has {int((~np.isfinite(X)).sum())} non-finite values")
        if not np.isfinite(Y).all():
            raise ValueError(f"{self.kind}: Y has {int((~np.isfinite(Y)).sum())} non-finite values")

    def _check_predict_output(self, X: FloatArray, P: FloatArray) -> FloatArray:
        if P.ndim != 2 or len(P) != len(X):
            raise ValueError(f"{self.kind}: predicted {P.shape} for {len(X)} input rows")
        if not np.isfinite(P).all():
            raise ValueError(f"{self.kind}: prediction has "
                             f"{int((~np.isfinite(P)).sum())} non-finite values")
        return P


# --------------------------------------------------------------------------- ridge

@dataclass
class RidgeModel(TargetModel):
    """Linear least squares with an L2 penalty — the baseline this fork started from.

    Multi-output Ridge is exactly one independent ridge per target with a shared alpha, so fitting
    the PCA coefficients and the two scalars in one call is not a shortcut: it is the same fit.

    It has no iterations, so there is nothing to stop early: the validation set is accepted and
    ignored. Tuning alpha on it would be a different model (RidgeCV), not early stopping.
    """

    alpha: float
    _est: Ridge | None = field(default=None, init=False, repr=False)

    @property
    def kind(self) -> str:
        return "ridge"

    def fit(self, X: FloatArray, Y: FloatArray, X_val: FloatArray, Y_val: FloatArray) -> None:
        self._check_fit_input(X, Y, X_val, Y_val)
        self._est = Ridge(alpha=self.alpha).fit(X, Y)

    def predict(self, X: FloatArray) -> FloatArray:
        if self._est is None:
            raise RuntimeError("ridge: predict() before fit()")
        return self._check_predict_output(X, np.asarray(self._est.predict(X), dtype=np.float64))


# --------------------------------------------------------------------------- catboost

@dataclass
class CatBoostModel(TargetModel):
    """Gradient-boosted trees, one MultiRMSE model over all outputs at once.

    MultiRMSE sums the squared error across output dimensions, so the targets have to arrive
    already scaled — see TargetScaler, which the pipeline applies for every model alike.

    `iterations` is now an upper bound: the fit stops after `early_stopping_rounds` without an
    improvement on the validation set and keeps the best iteration, not the last one.
    `early_stopping_rounds: 0` turns both off — trains the full budget and keeps its LAST
    iteration, which is the control the feature has to beat.
    """

    iterations: int
    depth: int
    learning_rate: float
    l2_leaf_reg: float
    early_stopping_rounds: int
    random_seed: int
    thread_count: int
    _est: Any = field(default=None, init=False, repr=False)

    @property
    def kind(self) -> str:
        return "catboost"

    def fit(self, X: FloatArray, Y: FloatArray, X_val: FloatArray, Y_val: FloatArray) -> None:
        self._check_fit_input(X, Y, X_val, Y_val)
        from catboost import CatBoostRegressor

        self._est = CatBoostRegressor(
            loss_function="MultiRMSE",
            iterations=self.iterations,
            depth=self.depth,
            learning_rate=self.learning_rate,
            l2_leaf_reg=self.l2_leaf_reg,
            random_seed=self.random_seed,
            thread_count=self.thread_count,
            allow_writing_files=False,      # no catboost_info/ droppings in the repo
            verbose=max(1, self.iterations // 10),
        )
        # eval_set is passed either way, so the validation curve is in the log even when nothing
        # acts on it; what `early_stopping_rounds: 0` switches off is the selection itself.
        self._est.fit(X, Y, eval_set=(X_val, Y_val),
                      early_stopping_rounds=self.early_stopping_rounds or None,
                      use_best_model=bool(self.early_stopping_rounds))

    def fit_report(self) -> str:
        if self._est is None:
            return ""
        if not self.early_stopping_rounds:
            return f", early stopping off, kept iteration {self._est.tree_count_ - 1}"
        best = int(self._est.get_best_iteration())
        return (f", best iteration {best} of {self.iterations}"
                f"{' (ran out, raise iterations)' if best >= self.iterations - 1 else ''}")

    def predict(self, X: FloatArray) -> FloatArray:
        if self._est is None:
            raise RuntimeError("catboost: predict() before fit()")
        return self._check_predict_output(X, np.asarray(self._est.predict(X), dtype=np.float64))


# --------------------------------------------------------------------------- torch MLP

@dataclass
class TorchMLPModel(TargetModel):
    """A fully-connected net on the same 21 features, trained on CPU.

    The fitted weights are kept as numpy arrays rather than as an `nn.Module`, so the artifact
    unpickles on a machine without torch — the module is rebuilt from `hidden_sizes` on first use.
    Like CatBoostModel it expects targets that TargetScaler has already scaled, and like it,
    `patience: 0` turns early stopping off — trains all `epochs` and keeps the LAST one.
    """

    hidden_sizes: list[int]
    epochs: int
    patience: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    seed: int
    threads: int
    _state: dict[str, np.ndarray] | None = field(default=None, init=False, repr=False)
    _n_in: int = field(default=0, init=False, repr=False)
    _n_out: int = field(default=0, init=False, repr=False)
    _best_epoch: int = field(default=-1, init=False, repr=False)
    _ran_epochs: int = field(default=0, init=False, repr=False)

    @property
    def kind(self) -> str:
        return "torch_mlp"

    def _build(self, n_in: int, n_out: int) -> Any:
        """The architecture, as a plain Sequential: Linear/ReLU stack, linear head."""
        import torch

        sizes = [n_in, *self.hidden_sizes]
        layers: list[Any] = []
        for a, b in itertools.pairwise(sizes):
            layers += [torch.nn.Linear(a, b), torch.nn.ReLU()]
        layers.append(torch.nn.Linear(sizes[-1], n_out))
        return torch.nn.Sequential(*layers)

    def fit(self, X: FloatArray, Y: FloatArray, X_val: FloatArray, Y_val: FloatArray) -> None:
        self._check_fit_input(X, Y, X_val, Y_val)
        import torch

        # Not one thread per core: a batch of this net is too small to feed 20 threads and the
        # synchronization dominates. See params.yaml for the measurement.
        torch.set_num_threads(self.threads)
        torch.manual_seed(self.seed)
        self._n_in, self._n_out = X.shape[1], Y.shape[1]
        net = self._build(self._n_in, self._n_out)

        Xt = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))
        Yt = torch.from_numpy(np.ascontiguousarray(Y, dtype=np.float32))
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(Xt, Yt),
            batch_size=self.batch_size, shuffle=True,
            generator=torch.Generator().manual_seed(self.seed),
        )
        Xvt = torch.from_numpy(np.ascontiguousarray(X_val, dtype=np.float32))
        Yvt = torch.from_numpy(np.ascontiguousarray(Y_val, dtype=np.float32))
        opt = torch.optim.Adam(net.parameters(), lr=self.learning_rate,
                               weight_decay=self.weight_decay)
        loss_fn = torch.nn.MSELoss()

        # Early stopping keeps the BEST epoch, not the last one: without it the saved weights are
        # whichever epoch the counter happened to end on, which is a coin flip against overfitting.
        best_loss, best_state, since_best = float("inf"), None, 0
        bar = tqdm(range(self.epochs), desc="    mlp", unit="epoch")
        for epoch in bar:
            total, n = 0.0, 0
            for xb, yb in loader:
                opt.zero_grad()
                loss = loss_fn(net(xb), yb)
                loss.backward()
                opt.step()
                total += float(loss) * len(xb)
                n += len(xb)

            net.eval()
            with torch.no_grad():
                val_loss = float(loss_fn(net(Xvt), Yvt))
            net.train()

            self._ran_epochs = epoch + 1
            if val_loss < best_loss or not self.patience:
                best_loss, since_best, self._best_epoch = val_loss, 0, epoch
                best_state = {k: v.detach().cpu().numpy().copy()
                              for k, v in net.state_dict().items()}
            else:
                since_best += 1
            bar.set_postfix(mse=f"{total / n:.4f}", val=f"{val_loss:.4f}",
                            best=f"{best_loss:.4f}@{self._best_epoch}")
            # The bar shows the last epoch; these lines are the history — a val loss that turned
            # around 400 epochs ago is invisible in a postfix that only ever shows "now".
            # MultiRMSE alongside MSE so the line can be read against CatBoost's own log, which
            # reports sqrt of the SUM over the 52 outputs where torch reports their MEAN:
            # MultiRMSE = sqrt(n_targets * MSE). Without it the two models' validation curves look
            # like they are measuring different things — they are not.
            multi_rmse = float(np.sqrt(self._n_out * val_loss))
            if epoch % MLP_LOG_EVERY == 0:
                bar.write(f"    mlp epoch {epoch:5d}: train {total / n:.6f}  val {val_loss:.6f}"
                          f"  (MultiRMSE {multi_rmse:.4f})  best {best_loss:.6f} @ "
                          f"{self._best_epoch}")
            if self.patience and since_best >= self.patience:
                bar.write(f"    mlp epoch {epoch:5d}: train {total / n:.6f}  val {val_loss:.6f}"
                          f"  (MultiRMSE {multi_rmse:.4f})  -> stopping, {self.patience} epochs "
                          f"since best {best_loss:.6f} @ {self._best_epoch}")
                break

        if best_state is None:
            raise RuntimeError("torch_mlp: no epoch completed — epochs must be at least 1")
        self._state = best_state

    def fit_report(self) -> str:
        if self._best_epoch < 0:
            return ""
        if not self.patience:
            return f", early stopping off, kept epoch {self._best_epoch}"
        return (f", best epoch {self._best_epoch} of {self._ran_epochs} run"
                f"{f' (of {self.epochs} allowed)' if self._ran_epochs < self.epochs else ''}")

    def predict(self, X: FloatArray) -> FloatArray:
        if self._state is None:
            raise RuntimeError("torch_mlp: predict() before fit()")
        import torch

        torch.set_num_threads(self.threads)
        net = self._build(self._n_in, self._n_out)
        net.load_state_dict({k: torch.from_numpy(v) for k, v in self._state.items()})
        net.eval()
        with torch.no_grad():
            xb = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))
            out = net(xb).numpy().astype(np.float64)
        return self._check_predict_output(X, out)


# --------------------------------------------------------------------------- params.yaml

MODEL_TYPES: dict[str, type[TargetModel]] = {
    "ridge": RidgeModel,
    "catboost": CatBoostModel,
    "torch_mlp": TorchMLPModel,
}


@dataclass(frozen=True)
class Params:
    """params.yaml, parsed and validated."""

    n_pca: int
    pca_seed: int
    subtract_coil_field: bool
    inputs: str                         # currents | coil_pca | both
    n_coil_pca: int
    models: dict[str, TargetModel]      # name -> unfitted model, enabled ones only
    ensemble: dict[str, float]          # name -> weight, already normalized to sum to 1
    path: Path

    @property
    def n_targets(self) -> int:
        return self.n_pca + 2           # + q95, betaN


# What `features.inputs` may say. The pipeline builds the feature matrix from this and nothing
# else, so a typo here is a ValueError naming the file rather than a silently different experiment.
INPUT_MODES = ("currents", "coil_pca", "both")


def _require_keys(got: dict[str, Any], want: set[str], where: str, path: Path) -> None:
    missing, extra = want - set(got), set(got) - want
    if missing or extra:
        raise ValueError(
            f"{path}: {where} expects exactly {sorted(want)}; "
            f"missing {sorted(missing)}, unexpected {sorted(extra)}"
        )


def _build_model(name: str, cfg: dict[str, Any], path: Path) -> TargetModel:
    kind = cfg.pop("type")
    if kind not in MODEL_TYPES:
        raise ValueError(f"{path}: model '{name}' has type '{kind}', "
                         f"known types are {sorted(MODEL_TYPES)}")
    try:
        return MODEL_TYPES[kind](**cfg)
    except TypeError as exc:
        # A hyper-parameter that does not exist is the error it looks like, not a no-op.
        raise ValueError(f"{path}: model '{name}' (type {kind}): {exc}") from exc


def load_params(path: Path = DEFAULT_PARAMS_PATH) -> Params:
    """Read params.yaml, or raise with the file and the offending key."""
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — it holds every hyper-parameter of the zoo")
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(doc, dict):
        raise ValueError(f"{path}: expected a mapping at the top level, got {type(doc).__name__}")
    _require_keys(doc, {"features", "models", "ensemble"}, "the top level", path)
    _require_keys(doc["features"],
                  {"n_pca", "pca_seed", "subtract_coil_field", "inputs", "n_coil_pca"},
                  "features", path)
    inputs = str(doc["features"]["inputs"])
    if inputs not in INPUT_MODES:
        raise ValueError(f"{path}: features.inputs is {inputs!r}, expected one of "
                         f"{sorted(INPUT_MODES)}")
    _require_keys(doc["ensemble"], {"members"}, "ensemble", path)

    models: dict[str, TargetModel] = {}
    for name, cfg in doc["models"].items():
        cfg = dict(cfg)
        if "enabled" not in cfg or "type" not in cfg:
            raise ValueError(f"{path}: model '{name}' needs both 'type' and 'enabled'")
        if cfg.pop("enabled"):
            models[name] = _build_model(name, cfg, path)
    if not models:
        raise ValueError(f"{path}: every model is disabled — nothing to train")

    weights = dict(doc["ensemble"]["members"])
    unknown = set(weights) - set(models)
    if unknown:
        raise ValueError(f"{path}: ensemble member(s) {sorted(unknown)} are not enabled models "
                         f"(enabled: {sorted(models)})")
    if not weights:
        raise ValueError(f"{path}: the ensemble has no members")
    total = float(sum(weights.values()))
    if total <= 0:
        raise ValueError(f"{path}: ensemble weights sum to {total}, must be positive")

    return Params(n_pca=int(doc["features"]["n_pca"]),
                  pca_seed=int(doc["features"]["pca_seed"]),
                  subtract_coil_field=bool(doc["features"]["subtract_coil_field"]),
                  inputs=inputs,
                  n_coil_pca=int(doc["features"]["n_coil_pca"]),
                  models=models,
                  ensemble={k: float(v) / total for k, v in weights.items()}, path=path)
