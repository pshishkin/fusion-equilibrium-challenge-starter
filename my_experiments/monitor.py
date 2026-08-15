#!/usr/bin/env python3
"""
A13 — stop on the composite the competition scores, not on the validation MSE.

The two disagree, and it has been measured twice. The sequence model reached a LOWER validation
loss than the MLP and a WORSE composite (0.9914 against 0.9931, and worse on every term). C2 then
found the same gap in the fitted model itself: R2_psi is 0.9998 of a 1.0000 ceiling and finished,
while the geometry terms it does not see are where all 0.0045 of the reachable budget sits. A loss
that is nearly perfect on the quantity that no longer matters is not a good place to stop.

`build_monitor` returns a callable from predicted SCALED targets on the validation block to the
composite over a small fixed subsample of validation frames — higher is better. The models take it
as `fit(..., monitor=...)` and use it for the best-weights and patience decisions instead of the
loss. Nothing about the loss itself changes: the gradient is still the metric-weighted MSE, and
only the stopping rule moves.

WHAT IS AND IS NOT THE REAL METRIC. The four terms and their weights are the scorer's own, and
`extract_lcfs`/`derive_frame` are the vendored functions the platform runs, so this is the same
arithmetic. Three deliberate differences, all in the direction of cost:

  * a subsample of frames, not the fold — a few hundred, and it is the SAME frames at every
    evaluation, so the signal is comparable across steps even where it is a biased estimate of the
    fold;
  * the frames are drawn from the validation block, which is what stopping is allowed to read;
  * `D_LCFS` is averaged over the sampled FRAMES rather than over shots, because a subsample does
    not carry whole shots. The scorer's own D_LCFS is a mean over shots (`metrics.py:114`); over a
    frame sample the two differ by 0.000034 of S, measured, which is far below anything a stopping
    rule resolves.
"""
from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "fusion_scoring"))

from common import (  # noqa: E402
    AXIS_SIGN,
    CONS_SCALARS,
    N_CONS,
    W_CONS,
    W_LCFS,
    W_PSI,
    W_QB,
)
from contour import symmetric_hausdorff  # noqa: E402
from derive import derive_frame  # noqa: E402
from lcfs import extract_lcfs, extract_lcfs_with_sign, major_radius  # noqa: E402

FloatArray = npt.NDArray[np.floating[Any]]

# Matching local_score's own settings, so the contour this compares against is the one the scorer
# would have built.
N_POINTS = 512
N_ITER = 22
LI_IDX = CONS_SCALARS.index("li")


def build_monitor(pca: Any, target_scaler: Any, ctx: dict[str, Any], machine: str,
                  psi_true: FloatArray, scalars_true: FloatArray, n_pca: int,
                  ) -> Callable[[FloatArray], float]:
    """Compile the reference once; return `composite(pred_scaled_rows) -> float`, higher better.

    `psi_true` is (n, 65, 65) TOTAL flux for the sampled frames — the coil field already added
    back, because the functionals read the whole map and the basis holds only the residual.
    `scalars_true` is (n, 2), the q95 and betaN of those frames.

    The returned callable takes the predictions for exactly those frames, in that order, in the
    SCALED target space the models work in — so it inverts the target scaler itself and the caller
    hands over `net(X_val)[idx]` untouched.
    """
    axis_sign = AXIS_SIGN[machine]
    n = len(psi_true)
    if scalars_true.shape != (n, 2):
        raise ValueError(f"scalars_true {scalars_true.shape}, expected {(n, 2)}")

    contours: list[Any] = []
    rgeos: list[float] = []
    cons_true = np.full((n, N_CONS), np.nan)
    for k in range(n):
        c = extract_lcfs(psi_true[k], ctx["grid_R"], ctx["grid_Z"], machine, ctx["mask_coarse"],
                         ctx["mask_f"], n_points=N_POINTS)
        contours.append(c)
        if c is not None:
            rgeos.append(major_radius(c))
        vals = derive_frame(psi_true[k], ctx["grid_R"], ctx["grid_Z"], machine, ctx["mask_coarse"],
                            ctx["mask_f"], contour=c)
        for j, name in enumerate(CONS_SCALARS):
            cons_true[k, j] = vals[name]
    cmask = np.isfinite(cons_true)
    rgeo = float(np.mean(rgeos)) if rgeos else float("nan")
    if not np.isfinite(rgeo) or not rgeo > 0:
        raise ValueError("no reference contour survived on the monitored frames — the sample is "
                         "unusable, raise it or check the validation block")

    # Denominators, fixed for the life of the monitor: R2 needs the variance of the TRUTH, and it
    # must not move between evaluations or the signal would be measuring its own denominator.
    psi_ss_tot = float(((psi_true - psi_true.mean()) ** 2).sum())
    scal_ss_tot = ((scalars_true - scalars_true.mean(axis=0)) ** 2).sum(axis=0)
    cons_ss_tot = np.zeros(N_CONS)
    for j in range(N_CONS):
        if cmask[:, j].any():
            v = cons_true[cmask[:, j], j]
            cons_ss_tot[j] = float(((v - v.mean()) ** 2).sum())

    def composite(pred_scaled: FloatArray) -> float:
        if len(pred_scaled) != n:
            raise ValueError(f"monitor got {len(pred_scaled)} rows, expected {n} — the caller must "
                             f"pass the predictions for exactly the monitored frames, in order")
        tgt = np.asarray(target_scaler.inverse_transform(np.asarray(pred_scaled, dtype=np.float64)))
        psi_pred = np.asarray(pca.inverse_transform(tgt[:, :n_pca]), dtype=np.float64)
        psi_pred = psi_pred + ctx["coil"]

        r2_psi = 1.0 - float(((psi_true - psi_pred) ** 2).sum()) / psi_ss_tot
        r2_qb = float(np.mean([
            1.0 - float(((scalars_true[:, j] - tgt[:, n_pca + j]) ** 2).sum()) / scal_ss_tot[j]
            if scal_ss_tot[j] > 0 else 0.0 for j in range(2)]))

        ds: list[float] = []
        ss_res = np.zeros(N_CONS)
        for k in range(n):
            need = cmask[k]
            if contours[k] is None and not need.any():
                continue
            ex, _ = extract_lcfs_with_sign(psi_pred[k], ctx["grid_R"], ctx["grid_Z"], machine,
                                           ctx["mask_coarse"], ctx["mask_f"], n_iter=N_ITER,
                                           axis_sign=axis_sign)
            if contours[k] is not None:
                if ex is None:
                    ds.append(1.0)
                else:
                    d = symmetric_hausdorff(ex, np.asarray(contours[k], dtype=np.float64))
                    ds.append(min(1.0, d / rgeo))
            if need.any():
                vals = derive_frame(psi_pred[k], ctx["grid_R"], ctx["grid_Z"], machine,
                                    ctx["mask_coarse"], ctx["mask_f"], contour=ex,
                                    with_li=bool(need[LI_IDX]), axis_sign=axis_sign)
                for j in np.nonzero(need)[0].tolist():
                    v = vals[CONS_SCALARS[j]]
                    # A functional that came out undefined is scored at the fold mean, which is
                    # what the scorer does — the residual is then the full deviation of the truth.
                    fallback = float(cons_true[cmask[:, j], j].mean())
                    dv = cons_true[k, j] - (v if np.isfinite(v) else fallback)
                    ss_res[j] += dv * dv

        scored = cons_ss_tot > 0
        cons = float(np.mean(np.maximum(0.0, 1.0 - ss_res[scored] / cons_ss_tot[scored])))
        d_lcfs = float(np.mean(ds)) if ds else 1.0
        return (W_PSI * max(0.0, r2_psi) + W_QB * max(0.0, r2_qb)
                + W_LCFS * (1.0 - min(1.0, d_lcfs)) + W_CONS * cons)

    return composite
