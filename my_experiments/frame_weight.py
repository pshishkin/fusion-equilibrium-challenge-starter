#!/usr/bin/env python3
"""A per-frame loss weight by position in the shot — the measurement, and the arithmetic it implies.

**Written 2026-08-20 and deliberately NOT wired in yet.** Every sweep entry re-imports the training
path and so do its pool workers, so an edit to `models.py` lands in every queued arm. This module
is inert until one line in `TorchMLP.fit` calls it; keeping it separate means the function can be
written and tested while a queue is running.

**What the measurement says.** Per-frame costs on the clean tail, decomposed onto the composite:

* the cost is **U-shaped**, not tail-heavy. The first decile costs 1.5x a flat shot and the last
  2.4x, with the middle eight between 0.6x and 0.9x. A22 only ever looked at the last one.
* the U lives in **Consistency** (0.00396 of the 0.00493 lost). D_LCFS is nearly flat in position,
  1.27x and 1.34x at the ends, so a position weight cannot be expected to move it.
* it is **not** ramp-down: |Ip| in the last decile is 1010 against a flat-top 1090, and falls below
  80% of flat-top on only 22% of those frames.
* it is **not** the current ramp rate either: cost correlates with |dIp/dt| at +0.105 frame by
  frame, and the most expensive decile has an unremarkable rate. So a weight by |dIp/dt| would put
  its mass on the FIRST decile and miss the most expensive one — measured, and rejected.
* high li is expensive only AT the ends: within the middle its top quintile costs 0.94x its bottom,
  while at the same top quintile the last decile costs 5.07x the middle. An interaction, so a
  position weight already captures it and an li term would add a parameter for nothing.

**The ceiling this implies, and it is smaller than A22 suggests.** Of the 0.00493 lost, 0.00396 is
Consistency and only the part ABOVE a flat profile is what concentration could recover — about
**0.0013**, not the 0.0025 the raw end-decile cost suggests.

**Mean-normalising is not cosmetic.** Without it the average gradient scales with the weight, and
the arm measures a learning-rate change wearing a frame-weight costume — the same confound that
made a cosine schedule look like a score result when it was a 16% rate cut.
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.floating]

# Set so the weight share on the ends equals their measured cost share — see `ends_weight`.
PRE_REGISTERED_STRENGTH = 1.5


def position_in_shot(shots: npt.NDArray[np.integer]) -> FloatArray:
    """0.0 at each shot's first frame, 1.0 at its last, one value per FRAME.

    `shots` is what `Model.fit` already receives: the per-shot frame COUNTS, in the order the frames
    are concatenated in. `models.py:196` states that contract and `seq_model.py:202` relies on it
    (`np.cumsum(shots)` gives the shot starts, and the counts must sum to the row count).

    An earlier version of this function read `shots` as a per-frame shot ID and took `np.diff` of
    it. That is a different array of a different length — on counts it would have returned one value
    per SHOT where one per FRAME is needed, so the weight would have been silently misaligned with
    the rows it multiplies. Caught 2026-08-21 before the arm ran, by reading the caller instead of
    the docstring next door.
    """
    lens = np.asarray(shots, dtype=np.int64)
    if lens.ndim != 1:
        raise ValueError(f"shots must be one frame count per shot, got shape {lens.shape}")
    if lens.size and lens.min() < 1:
        raise ValueError("a shot with no frames has no position; counts must be >= 1")
    out = np.zeros(int(lens.sum()), dtype=np.float64)
    at = 0
    for n in lens.tolist():
        # A one-frame shot is at position 0.0, not 0/0. It is both ends of itself, and calling it
        # the start is the choice that keeps `ends_weight` from up-weighting it twice.
        out[at:at + n] = 0.0 if n == 1 else np.arange(n) / (n - 1)
        at += n
    return out


def ends_weight(shots: npt.NDArray[np.integer], strength: float, edge: float = 0.1) -> FloatArray:
    """`1 + strength` on the first and last `edge` of every shot, 1 elsewhere, mean-normalised.

    A step rather than a curve because the evidence is a per-decile cost density; a smooth ramp
    would add a second free parameter nothing has measured.

    **`strength` is derived, not chosen.** On the clean tail the ends are 20.4% of the frames and
    carry 39.1% of the cost, so the strength that makes the weight share equal the cost share is

        (1 + s) * 0.204 / [(1 + s) * 0.204 + 0.796] = 0.391   ->   s = 1.5

    which is what `PRE_REGISTERED_STRENGTH` holds. Measured shares at other values: 27.8% of the
    weight at s = 0.5, 33.9% at 1.0, 43.5% at 2.0.
    """
    if strength < 0:
        raise ValueError(f"strength is {strength}, must be >= 0")
    if not 0.0 < edge < 0.5:
        raise ValueError(f"edge is {edge}, must be in (0, 0.5)")
    p = position_in_shot(shots)
    w = np.where((p <= edge) | (p >= 1.0 - edge), 1.0 + strength, 1.0)
    return w / w.mean()


def rampdown_mask(ip: FloatArray, shots: npt.NDArray[np.integer], level: float = 0.9) -> FloatArray:
    """1.0 on a shot's current ramp-down frames, 0.0 elsewhere, one value per frame.

    Ramp-down = after the last kept frame whose |Ip| reaches `level` of the shot's maximum |Ip|
    (`tools/error_regimes.py` uses the same definition). Measured 2026-10-06 on the 11-member stack:
    3% of frames, ~60% of li SSE, ~35% of Z_axis SSE; and on long-ramp-down shots the vessel
    members, which cut the flux SSE 15-18% everywhere else, leave the pooled SSE unchanged.
    """
    lens = np.asarray(shots, dtype=np.int64)
    a = np.abs(np.asarray(ip, dtype=np.float64))
    if lens.sum() != len(a):
        raise ValueError(f"shot counts sum to {lens.sum()}, Ip has {len(a)} frames")
    out = np.zeros(len(a))
    start = 0
    for n in lens:
        seg = a[start:start + n]
        hi = np.flatnonzero(seg >= level * seg.max()) if seg.max() > 0 else np.array([], int)
        if len(hi):
            out[start + hi[-1] + 1:start + n] = 1.0
        start += n
    return out
