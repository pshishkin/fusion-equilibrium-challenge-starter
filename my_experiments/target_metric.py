#!/usr/bin/env python3
"""
Which errors of the flux map the loss is allowed to care about.

The models regress PCA coefficients, and the PCA decoder is AFFINE. So every quadratic functional
of the map error is a quadratic form in the coefficient error, with a fixed matrix that can be
computed once from the basis:

    psi_pred - psi_gt = dc . V        (the mean image cancels)
    ||L(psi_pred - psi_gt)||^2 = dc^T M dc        for any linear operator L,  M = (LV)(LV)^T

`M` is (n_pca, n_pca) — 50x50 here — so this costs nothing at training time and needs no
autograd, no decoding to pixels, and no change to any model. A different `M` is simply a
different linear transform of the target block, which is where `TargetScaler` already lives.

Today's loss is `M = I`: by Parseval, on an orthonormal basis the unweighted sum of coefficient
errors IS the pixel error of the map, which is exactly what R2_psi measures. That is the right
metric for R2_psi and the wrong one for the rest of the composite, because Consistency and D_LCFS
do not read the map's values — they read its geometry. The magnetic axis is where grad psi
vanishes, so psi is flat there by definition and a pixel error that R2_psi cannot see moves the
axis a long way. Hence the two operators worth adding:

    grad psi   the POLOIDAL FIELD, B_p = |grad psi| / R. Weighting the error by it is weighting
               it by how wrong the field is rather than how wrong the flux is. `li` depends on psi
               only through <B_p^2>, and the boundary shape follows the field.
    Delta* psi the GRAD-SHAFRANOV operator, Delta* psi = -mu0 R j_phi: up to a constant, the
               TOROIDAL CURRENT DENSITY. Outside the plasma there is no current, so this is zero
               there, and penalising it says "do not invent plasma current where there is none".
               NOT `grad psi = 0`, which is false — the field extends well past the boundary.

Both operators are linear, so both are one more fixed 50x50 matrix, and mixing them is adding
matrices. What the mixing weights should be was once a hand-set knob; it is now measured —
`jacobian_form` perturbs each coefficient and watches the seven scored functionals through the
scorer's own code, and `metric_form` assembles `M` from the competition's own weights. The
poloidal-field form below is kept because `li` is exactly <B_p^2> and deserves its exact form
rather than a linearisation.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "fusion_scoring"))

from common import CONS_SCALARS, N_CONS, W_CONS, W_LCFS, W_PSI
from derive import _bilinear, _poloidal_field, derive_frame
from lcfs import extract_lcfs

from my_experiments.models import FloatArray
from my_experiments.parallel import pimap, resolve_jobs


def _grad(components: FloatArray, grid_R: FloatArray, grid_Z: FloatArray) -> tuple[
        FloatArray, FloatArray]:
    """d/dR and d/dZ of each basis image, on the (nZ, nR) grid."""
    dr = float(grid_R[1] - grid_R[0])
    dz = float(grid_Z[1] - grid_Z[0])
    return np.gradient(components, dr, axis=2), np.gradient(components, dz, axis=1)


def poloidal_field_form(components: FloatArray, grid_R: FloatArray,
                        grid_Z: FloatArray) -> FloatArray:
    """(n, n) `G` with `dc^T G dc` = the squared poloidal-field error of the map error.

    B_p = |grad psi| / R, and the volume element is dV = 2 pi R dR dZ, so

        integral |B_p error|^2 dV  =  2 pi integral |grad dpsi|^2 / R  dR dZ

    — the 1/R is the physics, not a detail: without it this weights an error at the inboard side
    the same as one at twice the radius, where the same flux gradient is half the field. Constant
    factors are dropped: only the SHAPE of this matrix matters, never its overall scale.
    """
    gr, gz = _grad(components, grid_R, grid_Z)
    w = np.sqrt(1.0 / grid_R)[None, None, :]
    a = (gr * w).reshape(len(components), -1)
    b = (gz * w).reshape(len(components), -1)
    return np.asarray(a @ a.T + b @ b.T, dtype=np.float64)


def grad_shafranov_form(components: FloatArray, grid_R: FloatArray, grid_Z: FloatArray,
                        weight: FloatArray | None = None) -> FloatArray:
    """(n, n) `Q` with `dc^T Q dc` = the squared toroidal-current error implied by the map error.

    Delta* psi = psi_RR - psi_R / R + psi_ZZ = -mu0 R j_phi. `weight` (nZ, nR) selects where it is
    penalised — the region with no plasma current in it. Note this is the error against the ground
    truth's own Delta*, never |Delta* psi_pred|^2: on a 65x65 grid the stored map has a finite
    Delta* of its own, and penalising the prediction for not being smoother than the truth would
    push it away from the truth.
    """
    dr = float(grid_R[1] - grid_R[0])
    dz = float(grid_Z[1] - grid_Z[0])
    d1r = np.gradient(components, dr, axis=2)
    d2r = np.gradient(d1r, dr, axis=2)
    d2z = np.gradient(np.gradient(components, dz, axis=1), dz, axis=1)
    star = d2r - d1r / grid_R[None, None, :] + d2z
    if weight is not None:
        if weight.shape != components.shape[1:]:
            raise ValueError(f"weight {weight.shape} does not match the grid "
                             f"{components.shape[1:]}")
        star = star * np.sqrt(weight)[None, :, :]
    flat = star.reshape(len(components), -1)
    return np.asarray(flat @ flat.T, dtype=np.float64)


MASKS = Path(__file__).resolve().parent.parent / "fusion_scoring" / "masks"


def scorer_context(grid_R: FloatArray, grid_Z: FloatArray, machine: str) -> dict[str, Any]:
    """Everything `derive_frame` needs besides the map, taken from the scorer's own mask bundle.

    The grids come from the bundle rather than from the row: the masks are defined on them, and a
    mask applied to a different grid is silently wrong rather than loudly so. They are checked
    against the row's grid here so that "silently" cannot happen.
    """
    tag = {"DIII-D": "d3d"}.get(machine)
    if tag is None:
        raise NotImplementedError(f"no envelope mask shipped for {machine!r}")
    bundle = np.load(MASKS / f"{tag}_envelope.npz")
    r, z = bundle["grid_R"], bundle["grid_Z"]
    if not (np.allclose(r, grid_R) and np.allclose(z, grid_Z)):
        raise ValueError(f"the {tag} envelope mask is defined on "
                         f"{r.min():.3f}..{r.max():.3f} x {z.min():.3f}..{z.max():.3f}, but this "
                         f"run's flux grid is {grid_R.min():.3f}..{grid_R.max():.3f} x "
                         f"{grid_Z.min():.3f}..{grid_Z.max():.3f}")
    coarse = bundle["mask_coarse"].astype(bool)
    return {"grid_R": r, "grid_Z": z, "machine": machine,
            "mask_coarse": coarse, "mask_f": coarse.astype(np.float64)}


def _derive(psi: FloatArray, ctx: dict[str, Any]) -> FloatArray:
    """The seven consistency scalars of one frame, through the scorer's own code."""
    vals = derive_frame(psi, ctx["grid_R"], ctx["grid_Z"], ctx["machine"],
                        ctx["mask_coarse"], ctx["mask_f"])
    return np.array([vals[name] for name in CONS_SCALARS], dtype=np.float64)


# Random full-vector probes per frame, used to check the linearisation against itself. Three is
# enough: they are averaged over hundreds of frames, and each one costs one `derive_frame`.
N_CALIBRATION_PROBES = 3


def _jacobian_task(args: tuple[Any, ...]) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Central-difference d(scalar)/d(coefficient) for a chunk of frames, and a linearity check.

    Returns the baseline scalar values (T, N_CONS), the Jacobians (T, N_CONS, n), and (T, 3,
    N_CONS) holding the squared ACTUAL change, the squared LINEAR-PREDICTED change, and the number
    of probes that produced a number at all, under full-vector perturbations of realistic size. A
    frame whose extraction fails on the baseline or on either side of any single-component
    perturbation contributes nan and is dropped by the caller.
    """
    psi_chunk, components, delta, ctx, seed = args
    n = len(components)
    base = np.full((len(psi_chunk), N_CONS), np.nan)
    jac = np.full((len(psi_chunk), N_CONS, n), np.nan)
    # Three planes, not two: squared actual change, squared linear prediction, and how many probes
    # actually produced a number. A full-vector perturbation is large enough to push a frame into a
    # state where a functional is undefined — `li` needs a closed contour to integrate B_p over —
    # and one such probe would otherwise turn the whole fold's sum into nan.
    check = np.zeros((len(psi_chunk), 3, N_CONS))
    rng = np.random.default_rng(seed)
    for i, psi in enumerate(psi_chunk):
        base[i] = _derive(psi, ctx)
        for k in range(n):
            step = delta * components[k]
            plus = _derive(psi + step, ctx)
            minus = _derive(psi - step, ctx)
            jac[i, :, k] = (plus - minus) / (2.0 * delta)
        # The per-component differences say how each direction acts ALONE and at that exact step.
        # Whether the sum of those still describes reality when every coefficient is wrong at once
        # — which is the regime the model is actually in — is a different question, and this is it.
        for _ in range(N_CALIBRATION_PROBES):
            dc = rng.normal(size=n) * delta
            actual = _derive(psi + np.tensordot(dc, components, axes=(0, 0)), ctx) - base[i]
            ok = np.isfinite(actual)
            check[i, 0] += np.where(ok, actual ** 2, 0.0)
            check[i, 1] += np.where(ok, (jac[i] @ dc) ** 2, 0.0)
            check[i, 2] += ok
    return base, jac, check


def jacobian_form(components: FloatArray, psi: FloatArray, delta: float, ctx: dict[str, Any],
                  jobs: int = 0,
                  calibrate: bool = False,
                  ) -> tuple[FloatArray, FloatArray, FloatArray, int, FloatArray]:
    """Measure how the seven scored functionals respond to each PCA coefficient.

    Returns `(M_cons, variances, linearity_ratio, n_used, probe_ok)` where `M_cons` is the (n, n)
    form with

        dc^T M_cons dc  =  sum_j (W_CONS / N_CONS) * (df_j)^2 / var(f_j)

    — the Consistency term of the composite, linearised around the ground truth. That is the piece
    the loss has never contained: the models minimise flux error, and Consistency reads geometry.

    ISOTROPIC PROBE. Every coefficient is perturbed by the same `delta`, not by a multiple of its
    own spread. The basis is orthonormal, so `delta * V_k` is the same size of FLUX error whichever
    k it is, and flux error is the unit the metric and the loss are both written in. Perturbing by
    each component's own variance would instead ask "what if the model were wrong in proportion to
    how much this component is used", which is a question about the data, not about the loss.

    `delta` must sit near the errors the model actually makes. These functionals are not linear —
    `extract_lcfs` finds a contour, `magnetic_axis` a sub-pixel maximum — so the derivative at an
    infinitesimal step describes a regime the model never occupies.

    D_LCFS is deliberately absent. It is a distance, so it is zero at the ground truth and its
    derivative is a one-sided thing that a central difference cannot see; it also carries only
    W_LCFS = 0.10 and, as the scorer's own docstring says, deliberately overlaps the shape scalars
    that ARE here.

    `calibrate` divides each scalar's block by how far its linearisation misses. Measured on a
    trained model, the Jacobian over-states `li` by 1.65x and `tri_bot` by 1.51x while `R_axis`,
    `Z_axis` and `volume` are within 10% — and an over-stated sensitivity is an over-weighted
    scalar, since M is JᵀJ / var. The ratio is measured here from random full-vector probes rather
    than from a model, so it needs nothing that does not exist before training.
    """
    n_jobs = resolve_jobs(jobs, max(1, len(psi)))
    chunks = np.array_split(np.arange(len(psi)), max(1, min(n_jobs * 2, len(psi))))
    tasks = [(psi[c], components, delta, ctx, i) for i, c in enumerate(chunks) if len(c)]
    base_parts, jac_parts, chk_parts = [], [], []
    for base, jac, chk in pimap(_jacobian_task, tasks, n_jobs):
        base_parts.append(base)
        jac_parts.append(jac)
        chk_parts.append(chk)
    base = np.concatenate(base_parts)
    jac = np.concatenate(jac_parts)
    chk = np.concatenate(chk_parts)

    good = np.isfinite(jac).all(axis=(1, 2)) & np.isfinite(base).all(axis=1)
    if good.sum() < 10:
        raise ValueError(f"only {int(good.sum())} of {len(base)} probe frames gave a finite "
                         f"Jacobian for all {N_CONS} scalars — too few to average")
    base, jac = base[good], jac[good]
    var = base.var(axis=0)
    if not (var > 0).all():
        flat = [CONS_SCALARS[j] for j in np.nonzero(var <= 0)[0]]
        raise ValueError(f"derived scalar(s) {flat} are constant over the probe frames, so the "
                         f"metric would divide by zero")

    # How far the linearisation is from the truth when EVERY coefficient is wrong at once, per
    # scalar. 1.0 means the Jacobian describes that functional correctly at this error size; above
    # 1.0 means it over-states the response and the scalar would be over-weighted in M.
    # Only over probes that produced a number. A perturbation big enough to be realistic is also
    # big enough to leave a frame with no closed contour, and those probes say nothing about the
    # linearisation — but how often it happens is worth knowing, so it is returned rather than
    # swallowed.
    chk = chk[good]
    finite = chk[:, 2, :].sum(axis=0)
    attempted = float(N_CALIBRATION_PROBES * len(chk))
    if not (finite > 0).all():
        dead = [CONS_SCALARS[j] for j in np.nonzero(finite <= 0)[0]]
        raise ValueError(f"every calibration probe left {dead} undefined over {len(chk)} frames, "
                         f"so the linearisation cannot be checked. The probe step {delta:.4g} is "
                         f"almost certainly too large for this scale.")
    ratio = np.sqrt(chk[:, 1, :].sum(axis=0) / np.maximum(chk[:, 0, :].sum(axis=0), 1e-300))
    probe_ok = finite / attempted
    cal = np.ones(N_CONS) if not calibrate else 1.0 / np.maximum(ratio, 1e-6)

    m = np.zeros((len(components), len(components)))
    for j in range(N_CONS):
        jj = jac[:, j, :]
        m += (W_CONS / N_CONS) * cal[j] ** 2 / var[j] * (jj.T @ jj) / len(jj)
    return m, var, ratio, int(good.sum()), probe_ok


# Where the boundary form stops trusting itself. The normal displacement of a level set is
# dn = dpsi / |grad psi|, so the weight 1/|grad psi|^2 blows up at the X-point, where |grad psi| is
# zero by definition. Measured over 320 contours: WITHOUT a clip the form listens to 16 of 512
# points — the worst 1% of them carry 55% of the weight — which is a max dressed up as an integral.
#
# The threshold is derived, not chosen. The linearisation holds while the displacement stays small
# against the scale on which |grad psi| itself varies, i.e. roughly one grid cell: at the model's
# own flux error (~0.0065 Wb/rad per pixel) and a 26 mm cell, that is |grad psi| above ~0.25 of the
# contour's median. The 10th percentile sits at 0.34 of the median — just inside that — and leaves
# 226 of 512 points effective while still weighting the X-point neighbourhood 8.7x above typical.
# So the largest weights are exactly where our own approximation is least valid, and this is where
# it stops. Not a hyper-parameter: a validity limit, measured once.
BOUNDARY_CLIP_QUANTILE = 0.10

# Contour resolution, matching the scorer's own default in local_score.
BOUNDARY_POINTS = 512


def _boundary_task(args: tuple[Any, ...]) -> tuple[FloatArray, float, int, int, int]:
    """Boundary form of a chunk of frames: (B, trace, n_used, n_clipped, n_points)."""
    psi_chunk, components, ctx = args
    n = len(components)
    grid_R, grid_Z = ctx["grid_R"], ctx["grid_Z"]
    acc = np.zeros((n, n))
    used = clipped = total = 0
    for psi in psi_chunk:
        contour = extract_lcfs(psi, grid_R, grid_Z, ctx["machine"], ctx["mask_coarse"],
                               ctx["mask_f"], n_points=BOUNDARY_POINTS)
        if contour is None or len(contour) < 8:
            continue
        rq, zq = contour[:, 0], contour[:, 1]
        # |grad psi| = R * B_p, and _poloidal_field is the scorer's own B_p — same code as li.
        grad = np.abs(_bilinear(_poloidal_field(psi, grid_R, grid_Z, ctx["machine"])
                                * grid_R[None, :], grid_R, grid_Z, rq, zq))
        arc = np.hypot(np.gradient(rq), np.gradient(zq))
        ok = np.isfinite(grad) & (grad > 0) & np.isfinite(arc)
        if ok.sum() < 8:
            continue
        rq, zq, grad, arc = rq[ok], zq[ok], grad[ok], arc[ok]
        floor = float(np.quantile(grad, BOUNDARY_CLIP_QUANTILE))
        clipped += int((grad < floor).sum())
        total += grad.size
        rgeo = 0.5 * (float(rq.max()) + float(rq.min()))
        w = arc / np.maximum(grad, floor) ** 2 / (arc.sum() * rgeo ** 2)
        sampled = np.stack([_bilinear(v, grid_R, grid_Z, rq, zq) for v in components])
        acc += (sampled * w) @ sampled.T
        used += 1
    return acc, float(np.trace(acc)), used, clipped, total


def boundary_form(components: FloatArray, psi: FloatArray, delta: float, ctx: dict[str, Any],
                  jobs: int = 0) -> tuple[FloatArray, float, int, float]:
    """How far the plasma boundary moves when the coefficients are wrong.

    Returns `(B, d_probe, n_used, clipped_fraction)` with

        dc^T B dc  =  mean over the contour of (normal displacement / rgeo)^2

    A small flux error moves a level set along its normal by `dpsi / |grad psi|`, which is LINEAR
    in the error and therefore one more fixed quadratic form — the same machinery as everything
    else here, evaluated on the boundary instead of over the grid.

    This is what `jacobian_form` structurally cannot supply. `D_LCFS` is a distance: it is zero at
    the ground truth, so a central difference sees no derivative at all, and the loss has never
    contained it. `kappa`, `volume` and the two triangularities ARE in the Jacobian, but they are
    four scalar summaries of a 512-point curve — they cannot express a rigid shift or a local
    dimple. This term sees the curve.

    `d_probe = delta * sqrt(tr B)` is the RMS displacement the isotropic probe itself produces, and
    it is what converts this mean-square into the metric's linear-in-distance `D_LCFS` term without
    inventing a constant — see `metric_form`.

    HONEST LIMIT: `D_LCFS` is a symmetric HAUSDORFF distance (`fusion_scoring/contour.py`), a
    maximum over the contour, and this is a mean square. It is a surrogate, not the term itself.
    """
    n_jobs = resolve_jobs(jobs, max(1, len(psi)))
    chunks = np.array_split(np.arange(len(psi)), max(1, min(n_jobs * 2, len(psi))))
    tasks = [(psi[c], components, ctx) for c in chunks if len(c)]
    acc = np.zeros((len(components), len(components)))
    used = clipped = total = 0
    for part, _, n_used, n_clip, n_pts in pimap(_boundary_task, tasks, n_jobs):
        acc += part
        used += n_used
        clipped += n_clip
        total += n_pts
    if used < 10:
        raise ValueError(f"only {used} of {len(psi)} probe frames yielded a usable LCFS contour — "
                         f"too few to average a boundary metric over")
    acc /= used
    return acc, delta * float(np.sqrt(np.trace(acc))), used, clipped / max(1, total)


def metric_form(m_cons: FloatArray, psi_ss_tot: float,
                m_bnd: FloatArray | None = None, d_probe: float = 0.0) -> FloatArray:
    """The full target metric, relative to the flux term the scaler already applies.

    `TargetScaler` divides the psi block by sqrt(psi_ss_tot / W_PSI), which by itself realises
    `W_PSI * SS_res / SS_tot` — the R2_psi term of the composite, and nothing else. This adds the
    Consistency term in the same units, so the loss becomes

        W_PSI * SS_res_psi/SS_tot_psi  +  (W_CONS/7) * sum_j SS_res_j/SS_tot_j

    with no free parameter anywhere: every weight comes from `fusion_scoring/common.py` and every
    denominator from the data. The psi block's share of the loss GROWS, and it should — the map is
    now carrying 0.55 + 0.20 of the composite instead of 0.55 — so nothing renormalises it back.

    `m_bnd` adds the boundary term the same way, with one wrinkle: the composite's `D_LCFS` term is
    LINEAR in distance while `m_bnd` is a mean square, so they are matched by linearising around
    the error the model actually makes,

        d(W_LCFS * sqrt(Q)) / dQ  =  W_LCFS / (2 * d_probe)

    with `d_probe` the RMS displacement the isotropic probe produces. That keeps the promise of no
    knobs: the operating point comes from `jacobian_delta`, which is already pinned to the model's
    own flux error, and not from a number chosen to make the answer come out.
    """
    m = np.eye(len(m_cons)) + (psi_ss_tot / W_PSI) * m_cons
    if m_bnd is not None:
        if not d_probe > 0:
            raise ValueError(f"d_probe must be positive to weight the boundary term, got "
                             f"{d_probe!r} — the probe moved the boundary nowhere at all")
        m = m + (psi_ss_tot / W_PSI) * (W_LCFS / (2.0 * d_probe)) * m_bnd
    return m


def factor(m: FloatArray) -> FloatArray:
    """`L` with `M = L L^T`, so that `dc @ L` has plain squared error `dc^T M dc`.

    A Cholesky factor, which exists because every `M` here is a Gram matrix plus a positive
    multiple of the identity. The identity comes back as the identity, so `metric: parseval` leaves
    the targets bit-identical to what they were before this module existed — which is what makes it
    usable as the control every measurement here is read against.
    """
    if np.allclose(m, np.eye(len(m))):
        return np.eye(len(m))
    try:
        return np.asarray(np.linalg.cholesky(m), dtype=np.float64)
    except np.linalg.LinAlgError as exc:
        w = np.linalg.eigvalsh(m)
        raise ValueError(f"the target metric is not positive definite (smallest eigenvalue "
                         f"{w.min():.3e}) and cannot be factored: {exc}") from exc
