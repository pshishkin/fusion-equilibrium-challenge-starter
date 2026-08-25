#!/usr/bin/env python3
"""q95 and betaN as small fitted models over the solved equilibrium, gated by leave-one-out.

**Why this replaces the affines in `calibration.py`.** Those read one contour integral through two
free numbers. Measured the honest way — refit on two demo shots, scored on the third — they are
worth almost nothing: held-out pooled R2 **+0.027 for q95 and +0.047 for betaN**, against +0.384
and +0.165 in sample. The board agrees they are weak; it reads R2_qb 0.2157 at S 0.56777, and that
term carries 0.15 of the composite.

The models here are the survivors of ~90 candidates, each judged by the same leave-one-shot-out
gate and then by a second test the gate cannot see: **do the coefficients hold still across the
three fits?** That second test is what did the selecting. A twelve-feature ridge scored +0.972
held-out for q95 and was thrown away, because its coefficient on the TF current moved 18 -> 30 and
its coefficient on kappa tripled between folds — a model reading noise can still cross-validate on
three sibling shots. Both models below keep every exponent inside 31%, and the ones carrying the
weight inside 8%.

    q95   = exp(c + w . log[q_shape(0.99), volume, Ip])     held out +0.967, per fold .96 .97 .97
    betaN = c + w . [volume, 1/Ip, minor radius]            held out +0.921, per fold .92 .92 .92

**What they say physically.** q95 comes out proportional to a^1.15 Ip^-0.41 times the contour
integral — the cylindrical estimate is a^2 Ip^-1, so the solve recovers the shape of the scaling
and the fit is correcting the amplitude of our psi, not inventing the relation. betaN is the
uncomfortable one: the Thomson term the old affine was built on, 200 mu0 <p> a / (B0 Ip), makes
the model WORSE than dropping it (+0.786 with it, +0.864 without), and plasma volume alone beats
it four times over. Volume correlates -0.90 with betaN and does so at -0.91 / -0.89 / -0.90 within
each shot separately, so it is not a campaign artefact of pooling. Read it as: our Thomson pressure
is too noisy to carry betaN, and the geometry the solve gets right carries it instead.

**The clips are insurance, not fitting.** Three shots cannot say what happens at an Ip or a volume
outside their range, and an exponential of an extrapolated linear form can run away. Both scalars
are held inside a range MAST physically occupies, so the worst an unseen frame can do is saturate.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "fusion_scoring"))

from lcfs import extract_lcfs, major_radius  # noqa: E402

from solver.gs import Operator, Solved  # noqa: E402
from solver.machine import MAST  # noqa: E402
from solver.scalars import q_shape  # noqa: E402

FloatArray = npt.NDArray[np.floating]


def geometry(psi: FloatArray, op: Operator) -> tuple[float, float, float, float]:
    """`(kappa, delta, minor radius, volume)` off the scorer's own contour, or nan on failure.

    Deliberately the scorer's `extract_lcfs` rather than our own boundary walk: these numbers feed
    a model whose target is the truth's version of them, so any disagreement between the two
    extractors would be fitted as if it were physics.
    """
    mask = op.mask
    c = extract_lcfs(psi, op.grid_R, op.grid_Z, MAST.name, mask, mask.astype(float))
    if c is None:
        return (float("nan"),) * 4
    r, z = c[:, 0], c[:, 1]
    r_lo, r_hi = float(r.min()), float(r.max())
    a = 0.5 * (r_hi - r_lo)
    r0 = 0.5 * (r_hi + r_lo)
    kappa = 0.5 * float(z.max() - z.min()) / max(a, 1e-9)
    delta = (r0 - 0.5 * (float(r[np.argmax(z)]) + float(r[np.argmin(z)]))) / max(a, 1e-9)
    return kappa, delta, a, 2.0 * np.pi * major_radius(c) * np.pi * a * a * kappa


@dataclass(frozen=True)
class ScalarModel:
    """A standardised linear form, optionally in log space, with a physical clip.

    `centre` and `spread` are the training mean and standard deviation of each feature (of its
    logarithm when `log`), kept explicit so the fit is reproducible from this file alone.
    """
    names: tuple[str, ...]
    log: bool
    weight: tuple[float, ...]
    centre: tuple[float, ...]
    spread: tuple[float, ...]
    const: float
    lo: float
    hi: float

    def __call__(self, feature: dict[str, float]) -> float:
        x = np.array([feature[n] for n in self.names], dtype=np.float64)
        if not np.isfinite(x).all() or (self.log and (x <= 0.0).any()):
            return float("nan")
        if self.log:
            x = np.log(x)
        z = float(self.const + ((x - np.asarray(self.centre)) / np.asarray(self.spread))
                  @ np.asarray(self.weight))
        return float(np.clip(np.exp(z) if self.log else z, self.lo, self.hi))


# Fitted on all 115 demo frames by `fit_scalars.py`; the held-out numbers quoted above come from
# the same script refitting on two shots at a time.
# **Refitted 2026-08-19 on the per-frame-profile solve.** Both models had to be rebuilt when
# `span.py` unfroze the flux span: every feature they read -- the contour integral, the minor
# radius, the volume -- is a property of the equilibrium, and the equilibrium moved. Scored
# against the old coefficients the new solve reads R2_qb 0.405; refitted it reads 0.865. This is
# not a tuning step that can be skipped after a change to the solver.
Q95 = ScalarModel(
    names=("shape", "a", "ip"),
    log=True,
    weight=(0.104367151619312, -0.10435341612078232, -0.05949676845464437),
    centre=(7.611362749287217, -0.5334480161066852, -0.48764056109706017),
    spread=(0.24937631838929392, 0.062158311993932916, 0.06418692417331835),
    const=2.0292981187984998,
    lo=1.0, hi=30.0,
)

BETA_N = ScalarModel(
    names=("volume", "inv_ip", "kappa"),
    log=False,
    weight=(-0.03441517815977276, -0.11614948617869647, -0.22770851238613832),
    centre=(9.58264119856904, 1.6308797898626346, 1.8149375412092554),
    spread=(0.7127239678241939, 0.10829574990941565, 0.14483806406815392),
    const=0.9805903647256934,
    lo=0.0, hi=6.0,
)

# The proxy surface q95 is read on. 0.99 won on the frozen-span solve; on this one 0.95 -- the
# surface q95 is actually defined on -- wins instead, and by a margin the stability gate cares
# about rather than the held-out number alone: every candidate reading 0.99 needed `kappa` to
# compete, and adding kappa sent the coefficients' spread across folds past 1200% with a sign
# flip. The physical surface and the stable fit turned out to be the same choice.
Q95_LEVEL = 0.95


# The fallbacks for a frame whose contour cannot be formed: the demo-shot means, in the units the
# scorer reads. NOT `Q95.const`, which lives in log space -- that mistake shipped a q95 of 2.0
# where the machine runs at 8, and it is the reason this comment exists.
Q95_FALLBACK = 7.9384
BETA_N_FALLBACK = 0.9806


def scalars_from(psi: FloatArray, op: Operator, solved: Solved, ip_amps: float,
                 tf_current_ka: float) -> tuple[float, float]:
    """`(q95, betaN)` for one frame, both nan if the contour cannot be formed.

    `shape` carries the TF current because q = (F/2pi) * contour integral and F is proportional to
    it. Dropping that factor leaves the model's `centre` -- a mean of logarithms -- describing a
    quantity 85x larger than the one being fed in, which is a constant offset in log space and
    scaled every q95 down by 2.6.
    """
    kappa, _, a, volume = geometry(psi, op)
    ip_ma = abs(ip_amps) / 1e6
    if not np.isfinite(volume) or ip_ma <= 0.0:
        return float("nan"), float("nan")
    feature = {"shape": abs(tf_current_ka) * q_shape(solved, op, Q95_LEVEL), "volume": volume,
               "ip": ip_ma, "inv_ip": 1.0 / ip_ma, "a": a, "kappa": kappa}
    return Q95(feature), BETA_N(feature)
