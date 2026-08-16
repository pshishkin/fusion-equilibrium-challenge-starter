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
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import yaml
from sklearn.linear_model import Ridge
from tqdm import tqdm

from my_experiments.progress import bar_kwargs

# The composite's own weights, read from the vendored scorer rather than copied here: the
# metric_aligned scaling below is only as correct as these numbers, and a hard-coded 0.55 would go
# stale the day the organizers reweight the leaderboard.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "fusion_scoring"))
from common import CONS_SCALARS, N_SCALARS, W_PSI, W_QB

FloatArray = npt.NDArray[np.floating[Any]]

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PARAMS_PATH = REPO_ROOT / "params.yaml"

# How often the MLP writes a train/val line, in optimizer steps. A logging cadence, not a
# hyper-parameter, so it stays in code — CatBoost's own `verbose` is set the same way. In a log
# file this line is the ONLY training output: the bar is switched off there (progress.bar_kwargs),
# because the line already carries everything the bar showed and a timestamp besides.
# Rounded DOWN to the evaluation cadence, since a line can only be written where there is a
# validation number to put in it.
MLP_LOG_EVERY_STEPS = 2000


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
    # A9. Auxiliary outputs appended after q95 and betaN: the seven scored functionals, taught to
    # the net directly and thrown away at inference. `aux_weight` is their share of the loss, set
    # the same way the two submitted scalars get theirs — divided among them and turned into a
    # divisor — so the three blocks' ratio is stated once, here, and not tuned by hand.
    n_aux: int = 0
    aux_weight: float = 0.0
    _center: FloatArray | None = field(default=None, init=False, repr=False)
    _scale: FloatArray | None = field(default=None, init=False, repr=False)
    # (n_pca, n_pca) `L` with `M = L L^T`, applied to the psi block after the scalar divisor above.
    # The divisor is one number over that whole block, so the two commute and the block ratio the
    # metric asks for survives untouched. Identity — today's Parseval loss — unless set.
    _psi_L: FloatArray | None = field(default=None, init=False, repr=False)
    _psi_L_inv: FloatArray | None = field(default=None, init=False, repr=False)

    def with_psi_metric(self, factor: FloatArray | None) -> TargetScaler:
        """Set the psi block's loss metric to `M = L L^T` by supplying `L`; None means Parseval.

        `L` carries its own absolute scale, deliberately: `metric_form` adds a term of the
        composite the loss did not represent before, so the psi block's share of the loss is meant
        to GROW — the map now carries W_PSI + W_CONS instead of W_PSI alone. Nothing here
        renormalises that away.
        """
        if factor is not None:
            if factor.shape != (self.n_pca, self.n_pca):
                raise ValueError(f"psi metric factor {factor.shape}, expected "
                                 f"{(self.n_pca, self.n_pca)}")
            if np.allclose(factor, np.eye(self.n_pca)):
                factor = None
        self._psi_L = factor
        self._psi_L_inv = None if factor is None else np.linalg.inv(factor)
        return self

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

        n_scalars = len(std) - self.n_pca - self.n_aux
        if n_scalars != N_SCALARS:
            raise ValueError(f"targets {Y.shape} hold {n_scalars} submitted scalars between the "
                             f"{self.n_pca} components and the {self.n_aux} auxiliary outputs, "
                             f"expected {N_SCALARS}")
        blocks = [np.full(self.n_pca, float(np.sqrt(psi_ss_tot_per_frame / W_PSI))),
                  std[self.n_pca:self.n_pca + N_SCALARS] / np.sqrt(W_QB / N_SCALARS)]
        if self.n_aux:
            if not self.aux_weight > 0:
                raise ValueError(f"{self.n_aux} auxiliary outputs with aux_weight "
                                 f"{self.aux_weight} — a block that carries no weight is a block "
                                 f"that should not be there")
            blocks.append(std[self.n_pca + N_SCALARS:]
                          / np.sqrt(self.aux_weight / self.n_aux))
        scale = np.concatenate(blocks)
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
        out = (Y - center) / scale
        if self._psi_L is not None:
            out = np.hstack([out[:, :self.n_pca] @ self._psi_L, out[:, self.n_pca:]])
        return out

    def inverse_transform(self, Y: FloatArray) -> FloatArray:
        center, scale = self._fitted()
        if self._psi_L_inv is not None:
            Y = np.hstack([Y[:, :self.n_pca] @ self._psi_L_inv, Y[:, self.n_pca:]])
        return Y * scale + center


# --------------------------------------------------------------------------- the interface

class TargetModel(abc.ABC):
    """One regressor from scaled features (n_frames, 21) to targets (n_frames, n_pca + 2)."""

    @abc.abstractmethod
    def fit(self, X: FloatArray, Y: FloatArray, X_val: FloatArray, Y_val: FloatArray,
            shots: npt.NDArray[np.int64] | None = None,
            shots_val: npt.NDArray[np.int64] | None = None,
            monitor: Callable[[FloatArray], float] | None = None,
            monitor_idx: npt.NDArray[np.intp] | None = None) -> None:
        """Fit on the training frames, stopping on the validation ones.

        The validation set is mandatory, not optional: a model that quietly trains to its full
        iteration count because nobody passed one is exactly the silent failure AGENTS.md bans.
        A model with nothing to stop on (ridge) says so in its docstring and ignores it.

        `shots` and `shots_val` are the per-shot frame COUNTS of the two blocks, in the order the
        frames are concatenated in — the only thing the flat arrays cannot say. Every model that
        scores frames independently ignores them; a model whose unit is the shot cannot be fitted
        without them and says so rather than inventing boundaries.
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
    ignored. Tuning alpha on it would be a different model (RidgeCV), not early stopping. The shot
    boundaries are ignored for the stronger reason — every row is an independent example here, and
    which shot it came from is not in the model's hypothesis class at all.
    """

    alpha: float
    _est: Ridge | None = field(default=None, init=False, repr=False)

    @property
    def kind(self) -> str:
        return "ridge"

    def fit(self, X: FloatArray, Y: FloatArray, X_val: FloatArray, Y_val: FloatArray,
            shots: npt.NDArray[np.int64] | None = None,
            shots_val: npt.NDArray[np.int64] | None = None,
            monitor: Callable[[FloatArray], float] | None = None,
            monitor_idx: npt.NDArray[np.intp] | None = None) -> None:
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
    # `CPU` or `GPU`. MultiRMSE is supported on both by catboost 1.2.10 — checked, because it has
    # not always been. `thread_count` still applies on GPU: it governs the data preparation, not
    # the boosting.
    task_type: str
    _est: Any = field(default=None, init=False, repr=False)

    @property
    def kind(self) -> str:
        return "catboost"

    def fit(self, X: FloatArray, Y: FloatArray, X_val: FloatArray, Y_val: FloatArray,
            shots: npt.NDArray[np.int64] | None = None,
            shots_val: npt.NDArray[np.int64] | None = None,
            monitor: Callable[[FloatArray], float] | None = None,
            monitor_idx: npt.NDArray[np.intp] | None = None) -> None:
        self._check_fit_input(X, Y, X_val, Y_val)
        if self.task_type not in CATBOOST_DEVICES:
            raise ValueError(f"catboost: unknown task_type {self.task_type!r}; known "
                             f"{sorted(CATBOOST_DEVICES)} — CatBoost says 'gpu' where torch says "
                             f"'cuda', and this setting is CatBoost's")
        from catboost import CatBoostRegressor

        self._est = CatBoostRegressor(
            loss_function="MultiRMSE",
            iterations=self.iterations,
            depth=self.depth,
            learning_rate=self.learning_rate,
            l2_leaf_reg=self.l2_leaf_reg,
            random_seed=self.random_seed,
            thread_count=self.thread_count,
            task_type=self.task_type.upper(),
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

# What `device` may say in params.yaml. `auto` is deliberately NOT offered: a run whose arithmetic
# depends on which hardware happened to be free is not reproducible, and pinning the split salt and
# the PCA seed was only worth doing because everything else about a configuration is fixed too.
TORCH_DEVICES = ("cpu", "cuda")

# CatBoost's own vocabulary, which is NOT torch's: it says GPU where torch says cuda. Keeping the
# two lists apart rather than translating between them is deliberate — a shared name would suggest
# a shared meaning, and `resolve_device` checks `torch.cuda.is_available()`, which says nothing
# about whether CatBoost was built with CUDA.
CATBOOST_DEVICES = ("cpu", "gpu")


# Nonlinearities offered for the hidden layers. ReLU is the baseline every number before
# 2026-08-14 was measured on; the rest have a nonzero derivative on the negative side, which is the
# property in question — a dead unit in a three-layer net never comes back.
#
# There is deliberately no choice for the OUTPUT: it stays linear. The targets are scaled PCA
# coefficients and scalars with unbounded range, so a squashing head (tanh, sigmoid) would cap the
# tails of the very distribution being regressed.
ACTIVATIONS = ("relu", "gelu", "silu", "tanh")

# What `lr_schedule` may say. `none` is the constant rate every number before 2026-08-14 was
# measured on. `cosine` decays from `learning_rate` to `learning_rate * lr_final_factor` over
# `lr_t_max_steps` steps, then HOLDS at that floor.
#
# The horizon is its own setting, and that was learned the hard way. It was first driven by the
# training ceiling, on the reasoning that using the point early stopping happens to reach would
# make the schedule's shape depend on its own outcome — true, but the ceiling is set far above any
# real run, so a fit that stops near a quarter of it only ever traverses the cosine's first flat
# quarter: the rate fell from 1e-3 to 8.4e-4 and never went below 84% of its start. That measured
# a 16% rate cut, not an annealing schedule. Set `lr_t_max_steps` near where the fit actually ends.
LR_SCHEDULES = ("none", "cosine")

# What `loss` may say. `mse` is what every number recorded so far was fitted on. `huber` is the
# same curve near zero and linear past `huber_delta`, so a frame the model reconstructs badly stops
# dominating the gradient of the batch it lands in.
LOSSES = ("mse", "huber")

# What the early-stopping rule reads. `loss` is the validation MSE every fit in this fork has used;
# `composite` is the competition score itself, over a fixed sample of validation frames decoded and
# run through the vendored scorer. See monitor.py for why they disagree.
STOP_ON = ("loss", "composite")


def build_mlp(n_in: int, hidden_sizes: list[int], n_out: int, activation: str = "relu",
              dropout: float = 0.0, norm: str = "none", residual: bool = False,
              n_scalars: int = 0) -> Any:
    """The architecture, in one place.

    The net is DECLARED once here and nothing downstream restates its shape; `predict` rebuilds
    from this same function, so the artifact and the fit cannot drift apart.

    Block order is `Linear -> norm -> activation -> Dropout`: a normaliser sees the linear
    pre-activations it is meant to normalise, and dropout does not feed zeroed units into the
    running statistics.

    `norm` is `none`, `batch` or `layer`. They are not two settings of one idea — batch norm
    couples the examples in a minibatch and layer norm does not — which is why the refutation of
    one says nothing about the other. **`batch` is 2-D only**: it takes (N, C), so a caller that
    passes (B, T, C), as the sequence model does, must use `none` or `layer`.

    `residual` wraps every block whose input and output widths match in a skip connection, which
    needs the hidden sizes to be equal. The first block never is — it maps `n_in` to the width —
    so a 3-block trunk gets two skips.

    `n_scalars`, when nonzero, gives the LAST `n_scalars` outputs their own head. The target
    vector is `[pca coefficients..., q95, betaN]`, and those two blocks are weighted 71/29 by the
    metric and have quite different statistics; one shared final layer makes them share a
    representation whether or not that helps.
    """
    import torch

    acts = {"relu": torch.nn.ReLU, "gelu": torch.nn.GELU, "silu": torch.nn.SiLU,
            "tanh": torch.nn.Tanh}
    norms = {"none": None, "batch": torch.nn.BatchNorm1d, "layer": torch.nn.LayerNorm}
    if activation not in acts:
        raise ValueError(f"unknown activation {activation!r}; known {sorted(acts)}")
    if norm not in norms:
        raise ValueError(f"unknown norm {norm!r}; known {sorted(norms)}")
    if not 0.0 <= dropout < 1.0:
        raise ValueError(f"dropout must be a share in [0, 1), got {dropout}")
    if residual and len(set(hidden_sizes)) > 1:
        raise ValueError(f"residual needs one width for the whole trunk, so a skip has matching "
                         f"ends; hidden_sizes is {hidden_sizes}")
    if n_scalars >= n_out:
        raise ValueError(f"n_scalars {n_scalars} leaves nothing for the other head of {n_out}")

    class Skip(torch.nn.Module):
        """`x + block(x)`, for a block whose ends match."""

        def __init__(self, block: Any) -> None:
            super().__init__()
            self.block = block

        def forward(self, x: Any) -> Any:
            return x + self.block(x)

    class Split(torch.nn.Module):
        """Two heads on one trunk, concatenated back into the target vector's own order."""

        def __init__(self, width: int, n_first: int, n_last: int) -> None:
            super().__init__()
            self.first = torch.nn.Linear(width, n_first)
            self.last = torch.nn.Linear(width, n_last)

        def forward(self, x: Any) -> Any:
            return torch.cat([self.first(x), self.last(x)], dim=-1)

    sizes = [n_in, *hidden_sizes]
    blocks: list[Any] = []
    make_norm = norms[norm]
    for a, b in itertools.pairwise(sizes):
        layers: list[Any] = [torch.nn.Linear(a, b)]
        if make_norm is not None:
            layers.append(make_norm(b))
        layers.append(acts[activation]())
        if dropout:
            layers.append(torch.nn.Dropout(dropout))
        block = torch.nn.Sequential(*layers)
        blocks.append(Skip(block) if residual and a == b else block)
    head = (Split(sizes[-1], n_out - n_scalars, n_scalars) if n_scalars
            else torch.nn.Linear(sizes[-1], n_out))
    return torch.nn.Sequential(*blocks, head)


def resolve_device(name: str, where: str) -> str:
    """The torch device to fit on, or an exception naming exactly what is missing."""
    import torch

    if name not in TORCH_DEVICES:
        raise ValueError(f"{where}: unknown device {name!r}; known {sorted(TORCH_DEVICES)}")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"{where}: device 'cuda', but torch.cuda.is_available() is False. This torch is "
            f"{torch.__version__}, built against CUDA {torch.version.cuda}. Set device: cpu in "
            f"params.yaml to fit on the CPU instead."
        )
    return name


@dataclass
class TorchMLPModel(TargetModel):
    """A fully-connected net on the same 21 features.

    The fitted weights are kept as numpy arrays rather than as an `nn.Module`, so the artifact
    unpickles on a machine without torch — the module is rebuilt from `hidden_sizes` on first use.
    Like CatBoostModel it expects targets that TargetScaler has already scaled, and like it,
    `patience_steps: 0` turns early stopping off — trains all `max_steps` and keeps the LAST
    evaluation.

    **Everything here is counted in optimizer STEPS, not epochs.** An epoch is `rows / batch_size`
    steps, so an epoch-counted leash silently means something different the moment the data share
    or the batch size moves — and this fork changes both. Counted in steps, `patience_steps` is
    the same amount of optimization whatever it is measured on, and a share that quintuples the
    rows no longer quintuples the budget behind the experimenter's back.

    `device` selects where the FIT runs. Inference is always on the CPU: the artifact has to
    unpickle and predict where a submission is scored, which is a machine this fork does not
    control, and a forward pass over one shot's ~300 frames is microseconds either way.

    Shot boundaries are accepted and ignored: this model scores every frame from that frame alone.
    `seq_model.TorchSeqModel` is the one that reads them.
    """

    hidden_sizes: list[int]
    # The ceiling, in optimizer steps. Batches keep coming from reshuffled passes over the data
    # until this is reached or the patience runs out; there is no epoch boundary in the loop.
    max_steps: int
    # Steps without a better validation loss before the fit stops. 0 turns early stopping off.
    # Only checked where validation is computed, so it is effectively rounded up to a multiple of
    # `eval_every_steps`.
    patience_steps: int
    # How often the validation loss is computed — and therefore the resolution at which the best
    # weights are picked and the patience is judged. The final step is always evaluated, so a
    # budget that is not a multiple of this still ends on a real measurement.
    eval_every_steps: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    seed: int
    threads: int
    device: str
    # The architecture knobs, all of them handed straight to build_mlp. See ACTIVATIONS on why the
    # output layer is not among them.
    activation: str
    dropout: float
    # `none`, `batch` or `layer` on the hidden pre-activations. Batch norm was measured at four
    # learning rates and gave parity for 38% more time; layer norm is a different mechanism (no
    # coupling between the examples of a minibatch), so that result is not evidence about it.
    norm: str
    # Skip connections around every hidden block. Needs one width for the whole trunk. The model is
    # bias-limited rather than variance-limited, so it wants effective depth — and plain depth 3
    # already scored slightly below depth 2, which is what depth that will not train looks like.
    residual: bool
    # Give q95 and betaN their own output head instead of sharing the final layer with the 50 flux
    # coefficients.
    split_heads: bool
    # `mse`, or `huber` with the transition at `huber_delta` standard deviations of the scaled
    # target. MSE lets one badly reconstructed frame contribute as much gradient as thirty ordinary
    # ones; Huber keeps the quadratic centre and bounds the tail.
    loss: str
    huber_delta: float
    # Rescale the gradient when its norm exceeds this; 0.0 is off. Measured 2026-08-15: at the
    # production shares and the rate tuned for a twenty-fifth of them, 15 of 28 fits BLEW UP —
    # training loss to 67.9 against unit-variance targets — and then recovered into a different
    # place. The pipeline is deterministic, so this is not run-to-run noise: it is a fit sitting on
    # the edge of stability, where a difference in the fourth decimal of the input decides whether
    # it survives. Clipping bounds the step that does it without giving up the rate.
    grad_clip: float
    # Constant `none`, or `cosine` decaying to lr * lr_final_factor over `lr_t_max_steps`. The
    # schedule is driven by a declared horizon rather than by the step early stopping happens to
    # reach, so it does not depend on when the fit ends — a schedule whose shape moves with its own
    # outcome is not a setting, it is a feedback loop.
    lr_schedule: str
    lr_final_factor: float
    # Steps the schedule is spread over; 0 means "use `max_steps`". Past it the rate HOLDS at
    # learning_rate * lr_final_factor rather than cycling back up, which is what
    # CosineAnnealingLR does on its own if it is stepped past T_max.
    lr_t_max_steps: int
    # A20. Decay of an exponential moving average of the WEIGHTS, kept alongside the live ones and
    # evaluated beside them at every evaluation; the fit keeps whichever scored better, so this can
    # only help or tie by the same rule that picks the best step. 0.0 turns it off.
    #
    # Why it should do anything at all: seed-to-seed sigma at production is 0.0009 of S and
    # averaging four seeds is worth 0.0042, so a large part of what a fit produces is optimisation
    # noise rather than a better function. Some of that noise is BETWEEN seeds and only more seeds
    # remove it; some is the last stretch of one trajectory rattling inside a basin, and an average
    # over that stretch removes it for free.
    ema_decay: float = 0.0
    # A13. Steps the validation loss may go without improving before the composite is allowed to
    # decide anything. Until then the fit keeps the best-LOSS weights, because a small frame sample
    # can flatter an untrained net and the loss cannot. Ignored when `stop_on: loss`.
    composite_after_steps: int = 10000
    # A13. `loss` keeps the weights at the lowest validation MSE, which is what every fit in this
    # fork has done. `composite` keeps them at the highest COMPETITION SCORE over a fixed sample of
    # validation frames, decoded and pushed through the vendored scorer — see monitor.py for why
    # the two disagree and for what the sample costs. The loss the gradient descends is unchanged
    # either way; only where the fit stops and which weights it keeps move.
    stop_on: str = "loss"
    # The composite is decoded and contour-extracted, so it costs milliseconds per frame where the
    # validation loss costs one matrix multiply. It gets its own, much coarser cadence, and the
    # patience is still counted in steps.
    composite_every_steps: int = 2000
    _state: dict[str, np.ndarray] | None = field(default=None, init=False, repr=False)
    _n_in: int = field(default=0, init=False, repr=False)
    _n_out: int = field(default=0, init=False, repr=False)
    _best_step: int = field(default=-1, init=False, repr=False)
    # A20: how many evaluations the averaged weights beat the live ones at. 0 with
    # `ema_decay` on is a result, not a bug — it says the trajectory is not rattling.
    _ema_wins: int = field(default=0, init=False, repr=False)
    _ran_steps: int = field(default=0, init=False, repr=False)
    _clipped: float = field(default=float("nan"), init=False, repr=False)
    _norm_median: float = field(default=float("nan"), init=False, repr=False)
    _norm_max: float = field(default=float("nan"), init=False, repr=False)

    @property
    def kind(self) -> str:
        return "torch_mlp"

    def _build(self, n_in: int, n_out: int) -> Any:
        return build_mlp(n_in, self.hidden_sizes, n_out, self.activation, self.dropout,
                         self.norm, self.residual, N_SCALARS if self.split_heads else 0)

    def fit(self, X: FloatArray, Y: FloatArray, X_val: FloatArray, Y_val: FloatArray,
            shots: npt.NDArray[np.int64] | None = None,
            shots_val: npt.NDArray[np.int64] | None = None,
            monitor: Callable[[FloatArray], float] | None = None,
            monitor_idx: npt.NDArray[np.intp] | None = None) -> None:
        self._check_fit_input(X, Y, X_val, Y_val)
        import torch

        dev = resolve_device(self.device, f"torch_mlp(seed={self.seed})")
        # BatchNorm1d cannot normalise a batch of one — it raises "Expected more than 1 value per
        # channel" in training mode — and the last batch of an epoch is whatever is left over. Said
        # here, with both numbers, rather than a thousand epochs into a run.
        if self.norm == "batch" and len(X) % self.batch_size == 1:
            raise ValueError(
                f"torch_mlp(seed={self.seed}): {len(X)} training rows at batch_size "
                f"{self.batch_size} leaves a final batch of ONE row, and batch norm cannot "
                f"normalise it. Change batch_size."
            )
        # Not one thread per core: a batch of this net is too small to feed 20 threads and the
        # synchronization dominates. See params.yaml for the measurement.
        torch.set_num_threads(self.threads)
        torch.manual_seed(self.seed)
        self._n_in, self._n_out = X.shape[1], Y.shape[1]
        # Built on the CPU and then moved, so the initial weights are the ones a CPU fit would
        # have drawn — the two runs start from the same point and only the arithmetic differs.
        net = self._build(self._n_in, self._n_out).to(dev)

        Xt_cpu = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))
        Yt_cpu = torch.from_numpy(np.ascontiguousarray(Y, dtype=np.float32))
        Xt, Yt = Xt_cpu.to(dev), Yt_cpu.to(dev)
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(Xt_cpu, Yt_cpu),
            batch_size=self.batch_size, shuffle=True,
            generator=torch.Generator().manual_seed(self.seed),
        )

        def batch_indices() -> Iterator[list[int]]:
            """One pass over the data, as index lists, from the DataLoader's OWN sampler.

            The rows are then gathered in one indexing operation instead of item by item through
            the loader's fetch-and-collate path. Measured on the production shapes — 70414 rows,
            138 batches of 512 — that path costs 0.270 s per pass against 0.128 s for the same
            batches by indexing: 53% of every pass was bookkeeping, and the shuffle itself is 4%.

            Going through a real iterator is the part that matters. `DataLoader` draws a base seed
            from `generator` every time one is created, so a hand-rolled `randperm` loop agrees
            with it on the first pass and diverges from the second on — checked, not assumed.
            Driving its sampler keeps the batches bit-for-bit what they were, so every number
            measured before this change still compares.
            """
            it = iter(loader)
            sampler_iter = getattr(it, "_sampler_iter", None)
            if sampler_iter is None:
                raise RuntimeError(
                    "this torch no longer exposes DataLoader._sampler_iter, so the batch order "
                    "cannot be reproduced. Iterate `loader` directly again — it is 2x slower per "
                    "epoch but correct — and re-measure the baseline, since the shuffle changes."
                )
            return sampler_iter
        Xvt = torch.from_numpy(np.ascontiguousarray(X_val, dtype=np.float32)).to(dev)
        Yvt = torch.from_numpy(np.ascontiguousarray(Y_val, dtype=np.float32)).to(dev)
        # fused Adam is one kernel for the whole step rather than a handful per tensor, which
        # matters only because this loop is bound by kernel launches. Measured equivalent to the
        # default implementation to 3.5e-7 relative over 20 steps.
        opt = torch.optim.Adam(net.parameters(), lr=self.learning_rate,
                               weight_decay=self.weight_decay, fused=(dev == "cuda"))
        if self.lr_schedule not in LR_SCHEDULES:
            raise ValueError(f"torch_mlp(seed={self.seed}): unknown lr_schedule "
                             f"{self.lr_schedule!r}; known {sorted(LR_SCHEDULES)}")
        sched = None
        t_max = self.lr_t_max_steps or self.max_steps
        if self.lr_schedule == "cosine":
            if t_max < 1:
                raise ValueError(f"torch_mlp(seed={self.seed}): lr_t_max_steps resolves to {t_max}")
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=t_max, eta_min=self.learning_rate * self.lr_final_factor)
        if self.loss not in LOSSES:
            raise ValueError(f"torch_mlp(seed={self.seed}): unknown loss {self.loss!r}; "
                             f"known {sorted(LOSSES)}")
        loss_fn = (torch.nn.MSELoss() if self.loss == "mse"
                   else torch.nn.HuberLoss(delta=self.huber_delta))
        for name, value in (("max_steps", self.max_steps),
                            ("eval_every_steps", self.eval_every_steps)):
            if value < 1:
                raise ValueError(f"torch_mlp(seed={self.seed}): {name} is {value}, must be >= 1")
        # The step loop restarts an exhausted pass, so a dataset with no rows would spin forever
        # instead of failing. It cannot happen through train.py, which is exactly why it is worth
        # one line here rather than a debugging session if it ever does.
        if not len(X):
            raise ValueError(f"torch_mlp(seed={self.seed}): no training rows")

        # Early stopping keeps the BEST evaluation, not the last one: without it the saved weights
        # are whichever step the counter happened to end on, a coin flip against overfitting.
        # One slot per step, filled only where clipping ran. 368000 steps is 1.5 MB, and keeping
        # every norm rather than a running mean is what makes the MEDIAN available at the end —
        # the mean of a distribution with rare spikes says nothing about the typical step.
        norms = torch.zeros(self.max_steps + 1, device=dev)
        # A13: what the stopping rule reads. `watch` is None unless the caller passed a monitor AND
        # this model was configured to use it, so the default path is bit-for-bit what it was.
        if self.stop_on not in STOP_ON:
            raise ValueError(f"torch_mlp(seed={self.seed}): unknown stop_on {self.stop_on!r}; "
                             f"known {sorted(STOP_ON)}")
        if self.stop_on == "composite" and (monitor is None or monitor_idx is None):
            raise ValueError(f"torch_mlp(seed={self.seed}): stop_on='composite' needs a monitor, "
                             f"and train.py builds one only when some model asks for it — this is "
                             f"a wiring bug, not a configuration one")
        watch = monitor if self.stop_on == "composite" else None
        watch_rows = (torch.as_tensor(np.asarray(monitor_idx), device=dev)
                      if watch is not None else None)
        if watch is not None and self.composite_every_steps < 1:
            raise ValueError(f"torch_mlp(seed={self.seed}): composite_every_steps is "
                             f"{self.composite_every_steps}, must be >= 1")
        composite, judged, scored = float("nan"), float("inf"), False
        best_loss, best_state, since_best = float("inf"), None, 0
        # The loss phase of the A13 rule: `armed` flips once, and only forward.
        best_val, since_val_best, armed = float("inf"), 0, False
        if not 0.0 <= self.ema_decay < 1.0:
            raise ValueError(f"torch_mlp(seed={self.seed}): ema_decay is {self.ema_decay}, "
                             f"expected 0 (off) or a decay in [0, 1)")
        # The shadow copy, and a place to park the live weights while the average is evaluated in
        # the same net. Cloned rather than referenced: `ema` must not follow `opt.step()`.
        ema = ({k: v.detach().clone() for k, v in net.state_dict().items()}
               if self.ema_decay else None)
        # Since the last evaluation, not since the start: the reported training loss is then a
        # window comparable to the validation number printed beside it, whatever the window is.
        total_t, n, stop = torch.zeros((), device=dev), 0, False
        step = 0
        bar = tqdm(total=self.max_steps, desc="    mlp", unit="step",
                   **bar_kwargs(off_in_log=True))
        while not stop:
            # Each call is one reshuffled pass over the rows. Passes are exhausted and restarted
            # rather than counted: the loop's clock is `step`, and where a pass happens to end
            # carries no meaning worth branching on.
            for idx in batch_indices():
                ib = torch.as_tensor(idx, device=dev)
                xb, yb = Xt[ib], Yt[ib]
                opt.zero_grad()
                loss = loss_fn(net(xb), yb)
                loss.backward()
                if self.grad_clip:
                    # The return value is the norm BEFORE clipping, which is the only way to tell a
                    # safety valve from a disguised learning-rate cut: if this fires on most steps
                    # the rate has effectively been divided by the median ratio, and the threshold
                    # was chosen by convention rather than from the data.
                    norms[step] = torch.nn.utils.clip_grad_norm_(net.parameters(), self.grad_clip)
                opt.step()
                # Accumulated ON the device and read once per evaluation. `float(loss)` inside the
                # batch loop is a full device sync every batch, which serialises exactly the
                # overlap a GPU exists to provide; on the CPU it costs nothing either way.
                total_t += loss.detach() * len(idx)
                n += len(idx)
                step += 1
                if ema is not None:
                    with torch.no_grad():
                        for k, v in net.state_dict().items():
                            # Integer buffers (none today, but batch norm's num_batches_tracked is
                            # one the moment `norm: batch` is used) cannot hold a running mean and
                            # are carried across rather than averaged.
                            if v.dtype.is_floating_point:
                                ema[k].mul_(self.ema_decay).add_(v, alpha=1 - self.ema_decay)
                            else:
                                ema[k].copy_(v)

                # Stepped only up to the horizon: past it CosineAnnealingLR would carry the rate
                # back UP toward its start, which is a warm restart nobody asked for.
                if sched is not None and step <= t_max:
                    sched.step()

                self._ran_steps = step
                last = step >= self.max_steps
                if step % self.eval_every_steps and not last:
                    continue

                # The composite runs on its own coarse grid, and always on the last step so the
                # fit cannot end on a measurement it never took. `judged` is what the best weights
                # and the patience are decided by — lower is better either way, so the composite
                # enters NEGATED and the comparison below stays one line.
                scored = (watch is None
                          or last or step % self.composite_every_steps == 0)

                def measure(scored: bool = scored) -> tuple[float, float, float]:
                    """(validation loss, composite, judged) for whatever weights `net` holds.

                    `scored` is bound as a default rather than closed over: the closure is rebuilt
                    every evaluation and reading a loop variable from it is the classic late-
                    binding bug even where, as here, it happens to be called immediately.
                    """
                    vl = float(loss_fn(net(Xvt), Yvt))
                    if watch is None:
                        return vl, float("nan"), vl
                    if not scored:
                        return vl, float("nan"), float("inf")
                    s = watch(net(Xvt[watch_rows]).detach().cpu().numpy())
                    return vl, s, -s

                net.eval()
                with torch.no_grad():
                    val_loss, composite, judged = measure()
                    # A20: the running average of the weights is a second candidate, judged by the
                    # same rule at the same moment. Whichever is better is what gets kept, so this
                    # cannot lose to not having it — the comparison is on validation data either
                    # way, and the average is never forced.
                    from_ema = False
                    if ema is not None and scored:
                        live = {k: v.detach().clone() for k, v in net.state_dict().items()}
                        net.load_state_dict(ema)
                        ema_loss, ema_comp, ema_judged = measure()
                        if ema_judged < judged:
                            val_loss, composite, judged, from_ema = (ema_loss, ema_comp,
                                                                     ema_judged, True)
                            self._ema_wins += 1
                        net.load_state_dict(live)
                net.train()
                train_loss = float(total_t) / n
                total_t, n = torch.zeros((), device=dev), 0
                # A13, and the reason the composite does not simply replace the loss. The
                # validation loss is reliably good at ONE thing — telling an undertrained fit from
                # a trained one — and the composite is uniquely good at another, choosing among
                # fits that are all trained. Letting the composite decide from step one throws the
                # first away: measured, `mlp2` recorded its best composite at the very first
                # evaluation, step 2000, and eighteen thousand steps later had only matched it,
                # because a 150-frame sample can flatter a barely-started net. The loss cannot make
                # that mistake, so it holds the gate.
                #
                # So: track the loss always; hand the decision to the composite only once the loss
                # has gone `composite_after_steps` without improving; and until then keep the
                # best-LOSS weights, so a fit that ends early still returns something trained.
                if val_loss < best_val:
                    best_val, since_val_best = val_loss, 0
                else:
                    since_val_best += self.eval_every_steps
                if watch is not None and not armed and since_val_best >= self.composite_after_steps:
                    armed = True
                    # The composite's own contest starts here, with nothing carried over from the
                    # loss phase — `best_loss` held a loss and would never be beaten by a negated
                    # score, and the patience for the fit's END restarts with the new criterion.
                    best_loss, since_best = float("inf"), 0
                    bar.write(f"    mlp step {step:7d}: validation loss flat for "
                              f"{since_val_best} steps -> the composite decides from here")
                decides = scored and (watch is None or armed)
                if decides and (judged < best_loss or not self.patience_steps):
                    best_loss, since_best, self._best_step = judged, 0, step
                    keep = ema if (from_ema and ema is not None) else net.state_dict()
                    best_state = {k: v.detach().cpu().numpy().copy() for k, v in keep.items()}
                elif decides:
                    since_best += (self.composite_every_steps if watch is not None
                                   else self.eval_every_steps)
                elif watch is not None and not armed and since_val_best == 0:
                    # Still in the loss phase and the loss just improved: keep those weights, so an
                    # early stop or a max_steps ceiling never returns an arbitrary step.
                    self._best_step = step
                    keep = net.state_dict()
                    best_state = {k: v.detach().cpu().numpy().copy() for k, v in keep.items()}
                # refresh=False, or the postfix forces a redraw at EVERY evaluation and quietly
                # defeats the `miniters` that keeps a teed log readable. The values still ride
                # along on the next redraw the bar makes on its own.
                bar.update(step - bar.n)
                # `best_loss` holds a negated score once the composite is armed, a loss before
                # that, and infinity in the gap — so what a human reads is assembled here rather
                # than printed raw.
                if watch is None:
                    shown = best_loss
                elif armed:
                    shown = -best_loss
                else:
                    shown = best_val
                bar.set_postfix(mse=f"{train_loss:.4f}", val=f"{val_loss:.4f}",
                                best=f"{shown:.4f}@{self._best_step}", refresh=False)
                # The bar shows the last step; these lines are the history — a val loss that turned
                # around 20000 steps ago is invisible in a postfix that only ever shows "now".
                # MultiRMSE alongside MSE so the line can be read against CatBoost's own log, which
                # reports sqrt of the SUM over the 52 outputs where torch reports their MEAN:
                # MultiRMSE = sqrt(n_targets * MSE). Without it the two models' validation curves
                # look like they are measuring different things — they are not.
                multi_rmse = float(np.sqrt(self._n_out * val_loss))
                head = (f"    mlp step {step:7d}: train {train_loss:.6f}  val {val_loss:.6f}"
                        f"  (MultiRMSE {multi_rmse:.4f})")
                if watch is not None and scored:
                    head += f"  S {composite:.6f}"
                if step % MLP_LOG_EVERY_STEPS < self.eval_every_steps:
                    bar.write(f"{head}  best {shown:.6f} @ {self._best_step}")
                if last or (self.patience_steps and since_best >= self.patience_steps):
                    if not last:
                        bar.write(f"{head}  -> stopping, {since_best} steps since best "
                                  f"{shown:.6f} @ {self._best_step}")
                    stop = True
                    break

        if best_state is None:
            raise RuntimeError("torch_mlp: no evaluation completed — the fit ran no steps")
        self._state = best_state
        if self.grad_clip:
            seen = norms[1:self._ran_steps + 1]
            self._clipped = float((seen > self.grad_clip).float().mean())
            self._norm_median = float(seen.median())
            self._norm_max = float(seen.max())

    def fit_report(self) -> str:
        if self._best_step < 0:
            return ""
        if not self.patience_steps:
            return f", early stopping off, kept step {self._best_step}"
        clip = ("" if not self.grad_clip else
                f", clipped {self._clipped:.1%} of steps (median grad norm "
                f"{self._norm_median:.3g}, max {self._norm_max:.3g}, threshold {self.grad_clip:g})")
        ema = (f", averaged weights won {self._ema_wins} evaluation(s)"
               if self.ema_decay else "")
        return (f", best step {self._best_step} of {self._ran_steps} run"
                f"{f' (of {self.max_steps} allowed)' if self._ran_steps < self.max_steps else ''}"
                f"{clip}{ema}")

    def predict(self, X: FloatArray) -> FloatArray:
        if self._state is None:
            raise RuntimeError("torch_mlp: predict() before fit()")
        import torch

        # On the CPU whatever `device` said — see the class docstring: a submission is scored on a
        # machine this fork does not control, so inference must not require a GPU.
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


def _register_sequence_model() -> None:
    """Add `torch_seq` to the registry, from the module that defines it.

    It lives in `seq_model.py` rather than here because it is a different KIND of model — its unit
    is the shot, not the frame — and because it needs `build_mlp` and `TargetModel` from this
    module, which is a cycle if the import goes the other way. Imported at call time, so the cycle
    never exists: by the time anything asks for a model, this module is fully defined.
    """
    from my_experiments.seq_model import TorchSeqModel
    MODEL_TYPES.setdefault("torch_seq", TorchSeqModel)


@dataclass(frozen=True)
class Params:
    """params.yaml, parsed and validated."""

    n_pca: int
    pca_seed: int
    pca_frame_share: float
    derivatives: str                    # none | raw | interp | both
    derivative_signals: str             # driving | poloidal
    # A19/A22: which signals get the leaky-integral block of dI/dt. 'none' is off.
    vessel: str                         # none | driving | poloidal
    thomson: str                        # "" for off, else comma-separated group names
    frame_gaps: bool                    # the distance to the neighbouring frames, as input
    split_salt: int
    holdout_share: float
    loss_metric: str
    calibrate_scalars: bool
    jacobian_frames: int
    # A13: validation frames the composite monitor scores. See monitor.py.
    monitor_frames: int
    # A9: which of the seven scored functionals become auxiliary outputs, and what share of the
    # loss they carry. `none` is off and is what every number before 2026-08-16 was measured on.
    aux_scalars: str
    aux_weight: float
    jacobian_delta: float
    boundary: bool
    subtract_coil_field: bool
    inputs: str                         # currents | coil_pca | both
    n_coil_pca: int
    models: dict[str, TargetModel]      # name -> unfitted model, enabled ones only
    ensemble: dict[str, float]          # name -> weight, already normalized to sum to 1
    path: Path
    # params.yaml AFTER --salt / --only were applied. The artifact stores this rather than the
    # file's text, so a screening run records the configuration it actually fitted instead of the
    # one that happens to be on disk. Identical to the file when nothing was overridden.
    effective_yaml: str

    @property
    def aux_names(self) -> list[str]:
        """The functionals taught as auxiliary outputs, in the scorer's own order."""
        if self.aux_scalars == "none":
            return []
        want = [s.strip() for s in self.aux_scalars.split(",")]
        return [s for s in CONS_SCALARS if s in want]

    @property
    def n_targets(self) -> int:
        return self.n_pca + 2 + len(self.aux_names)      # + q95, betaN, + A9's auxiliaries


# What `features.inputs` may say. The pipeline builds the feature matrix from this and nothing
# else, so a typo here is a ValueError naming the file rather than a silently different experiment.
INPUT_MODES = ("currents", "coil_pca", "both", "flux_probe")

# What `features.derivatives` may say. `raw` differentiates each signal on its own
# 0.05 ms time base before interpolating onto the EFIT frames; `interp` differentiates
# the interpolated series at the 20 ms frame spacing. They are different physical
# timescales, not a right and a wrong way, which is why both are offered.
DERIV_MODES = ("none", "raw", "interp", "both")

# Which signals are differentiated. Defined here so params.yaml can be validated without
# importing the pipeline; the membership itself lives in baseline_model.
DERIV_SIGNAL_SET_NAMES = ("driving", "poloidal")

# What the psi block's loss metric is. `parseval` is the pixel error of the map, which is exactly
# R2_psi and nothing else — the control, and the fallback where the probe cannot run. `jacobian`
# measures how the seven scored functionals actually respond to each coefficient and assembles the
# metric from the competition's own weights, with no knob anywhere.
LOSS_METRICS = ("parseval", "jacobian")


def _require_keys(got: dict[str, Any], want: set[str], where: str, path: Path) -> None:
    missing, extra = want - set(got), set(got) - want
    if missing or extra:
        raise ValueError(
            f"{path}: {where} expects exactly {sorted(want)}; "
            f"missing {sorted(missing)}, unexpected {sorted(extra)}"
        )


def _build_model(name: str, cfg: dict[str, Any], path: Path) -> TargetModel:
    kind = cfg.pop("type")
    _register_sequence_model()
    if kind not in MODEL_TYPES:
        raise ValueError(f"{path}: model '{name}' has type '{kind}', "
                         f"known types are {sorted(MODEL_TYPES)}")
    try:
        return MODEL_TYPES[kind](**cfg)
    except TypeError as exc:
        # A hyper-parameter that does not exist is the error it looks like, not a no-op.
        raise ValueError(f"{path}: model '{name}' (type {kind}): {exc}") from exc


def _set_by_path(doc: dict[str, Any], dotted: str, raw: str, path: Path) -> None:
    """`models.mlp.learning_rate=0.003` -> doc["models"]["mlp"]["learning_rate"] = 0.003.

    The key must ALREADY exist. A sweep that can invent keys is a sweep that can silently vary
    nothing at all, which looks exactly like a change that did not help — the same reason
    params.yaml rejects an unknown key rather than ignoring it.
    """
    parts = dotted.split(".")
    node: Any = doc
    for i, part in enumerate(parts[:-1]):
        if not isinstance(node, dict) or part not in node:
            raise ValueError(f"{path}: --set {dotted}= has no key {'.'.join(parts[:i + 1])!r}")
        node = node[part]
    leaf = parts[-1]
    if not isinstance(node, dict) or leaf not in node:
        raise ValueError(f"{path}: --set {dotted}= has no key {dotted!r}; "
                         f"{'.'.join(parts[:-1]) or 'the top level'} holds "
                         f"{sorted(node) if isinstance(node, dict) else type(node).__name__}")
    # YAML rather than a hand-rolled cast, so `true`, `[512, 512]` and `null` mean here exactly
    # what they mean in the file this is overriding.
    value = yaml.safe_load(raw)
    old = node[leaf]
    # ...with one trap YAML 1.1 lays: `3e-3` is a STRING to it, because a float needs a dot or a
    # signed exponent (`3.0e-3`, `3e+3`). Left alone it reaches torch as "3e-3" and dies deep in
    # the optimizer with `'<=' not supported between float and str`, a thousand lines after the
    # cause. Measured the hard way: it killed a sweep at run 3 of 9.
    if isinstance(value, str) and isinstance(old, (int, float)) and not isinstance(old, bool):
        try:
            value = float(value)
        except ValueError:
            raise ValueError(
                f"{path}: --set {dotted}={raw!r} — {dotted} holds {type(old).__name__} "
                f"{old!r}, and {raw!r} is not a number"
            ) from None
    if isinstance(old, bool) != isinstance(value, bool):
        raise ValueError(f"{path}: --set {dotted}={raw!r} would change {dotted} from "
                         f"{type(old).__name__} {old!r} to {type(value).__name__} {value!r}")
    node[leaf] = value


def apply_overrides(doc: dict[str, Any], path: Path, salt: int | None,
                    only: list[str] | None, sets: list[str] | None = None) -> dict[str, Any]:
    """The two things a screening run wants to vary without editing the file.

    Hyper-parameters live in params.yaml and only there — see AGENTS.md — and this does not
    contradict that. What the rule forbids is a DEFAULT hiding in argparse, a second value to keep
    in sync. These are explicit, they have no defaults of their own, they are printed, and the
    EFFECTIVE document is what the artifact stores, so a run still says exactly what produced it.

    * `salt` reshuffles which shots train, validate and score. It is the replicate that matters,
      and sweeping it meant editing the file between runs, which is how a sweep ends up comparing
      two configurations by accident.
    * `only` narrows the zoo to the named models and rebuilds the ensemble from whatever of them
      survives — one net instead of four is a quarter of the fit, which is what makes a screen a
      screen. Everything not named is switched off, `ridge` included: it costs 0.0 s and is the
      deterministic control, so name it when you want it.
    """
    if salt is not None:
        doc["split"]["salt"] = salt
    if only is not None:
        if not only:
            raise ValueError(f"{path}: --only was given no model names")
        unknown = set(only) - set(doc["models"])
        if unknown:
            raise ValueError(f"{path}: --only names {sorted(unknown)}, which are not in the file; "
                             f"it holds {sorted(doc['models'])}")
        for name, cfg in doc["models"].items():
            cfg["enabled"] = name in only
        members = {k: v for k, v in doc["ensemble"]["members"].items() if k in only}
        # Every ensemble member was switched off, so the ensemble has to be redefined rather than
        # left dangling. The named models, equally weighted, is the only reading that is not a
        # guess — and with one name it is that model.
        doc["ensemble"]["members"] = members or dict.fromkeys(only, 1.0)
    for item in sets or []:
        if "=" not in item:
            raise ValueError(f"--set expects key=value, got {item!r}")
        dotted, raw = item.split("=", 1)
        _set_by_path(doc, dotted.strip(), raw.strip(), path)
    return doc


def load_params(path: Path = DEFAULT_PARAMS_PATH, salt: int | None = None,
                only: list[str] | None = None, sets: list[str] | None = None) -> Params:
    """Read params.yaml, or raise with the file and the offending key.

    `salt` and `only` override the file for one run; see `apply_overrides`.
    """
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — it holds every hyper-parameter of the zoo")
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(doc, dict):
        raise ValueError(f"{path}: expected a mapping at the top level, got {type(doc).__name__}")
    doc = apply_overrides(doc, path, salt, only, sets)
    _require_keys(doc, {"features", "split", "loss", "models", "ensemble"},
                  "the top level", path)
    _require_keys(doc["split"], {"salt", "holdout_share"}, "split", path)
    _require_keys(doc["loss"],
                  {"metric", "calibrate_scalars", "jacobian_frames", "jacobian_delta", "boundary",
                   "monitor_frames", "aux_scalars", "aux_weight"},
                  "loss", path)
    loss_metric = str(doc["loss"]["metric"])
    if loss_metric not in LOSS_METRICS:
        raise ValueError(f"{path}: loss.metric is {loss_metric!r}, expected one of "
                         f"{sorted(LOSS_METRICS)}")
    _require_keys(doc["features"],
                  {"n_pca", "pca_seed", "pca_frame_share", "subtract_coil_field", "inputs",
                   "n_coil_pca", "derivatives", "derivative_signals", "thomson",
                   "frame_gaps", "vessel"},
                  "features", path)
    pca_frame_share = float(doc["features"]["pca_frame_share"])
    if not 0.0 < pca_frame_share <= 1.0:
        raise ValueError(f"{path}: features.pca_frame_share is {pca_frame_share}, expected a "
                         f"share in (0, 1]")
    derivatives = str(doc["features"]["derivatives"])
    if derivatives not in DERIV_MODES:
        raise ValueError(f"{path}: features.derivatives is {derivatives!r}, expected "
                         f"one of {sorted(DERIV_MODES)}")
    deriv_signals = str(doc["features"]["derivative_signals"])
    if deriv_signals not in DERIV_SIGNAL_SET_NAMES:
        raise ValueError(f"{path}: features.derivative_signals is {deriv_signals!r}, expected one "
                         f"of {sorted(DERIV_SIGNAL_SET_NAMES)}")
    aux_scalars = str(doc["loss"]["aux_scalars"])
    if aux_scalars != "none":
        bad_aux = [s for s in aux_scalars.split(",") if s.strip() not in CONS_SCALARS]
        if bad_aux:
            raise ValueError(f"{path}: loss.aux_scalars names {bad_aux}, which are not scored "
                             f"functionals; known {CONS_SCALARS}")
    vessel = str(doc["features"]["vessel"])
    if vessel not in ("none", *DERIV_SIGNAL_SET_NAMES):
        raise ValueError(f"{path}: features.vessel is {vessel!r}, expected 'none' or one of "
                         f"{sorted(DERIV_SIGNAL_SET_NAMES)}")
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
                  pca_frame_share=pca_frame_share,
                  derivatives=derivatives,
                  derivative_signals=deriv_signals,
                  vessel=vessel,
                  thomson=str(doc["features"]["thomson"] or ""),
                  frame_gaps=bool(doc["features"]["frame_gaps"]),
                  split_salt=int(doc["split"]["salt"]),
                  holdout_share=float(doc["split"]["holdout_share"]),
                  loss_metric=loss_metric,
                  calibrate_scalars=bool(doc["loss"]["calibrate_scalars"]),
                  jacobian_frames=int(doc["loss"]["jacobian_frames"]),
                  monitor_frames=int(doc["loss"]["monitor_frames"]),
                  aux_scalars=aux_scalars,
                  aux_weight=float(doc["loss"]["aux_weight"]),
                  jacobian_delta=float(doc["loss"]["jacobian_delta"]),
                  boundary=bool(doc["loss"]["boundary"]),
                  subtract_coil_field=bool(doc["features"]["subtract_coil_field"]),
                  inputs=inputs,
                  n_coil_pca=int(doc["features"]["n_coil_pca"]),
                  models=models,
                  ensemble={k: float(v) / total for k, v in weights.items()}, path=path,
                  effective_yaml=yaml.safe_dump(doc, sort_keys=False))
