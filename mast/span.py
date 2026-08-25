#!/usr/bin/env python3
"""How much poloidal flux the plasma contains, per frame, from the solenoid that put it there.

**The bug this exists to fix.** The shipped solve uses one profile shape for every frame
(`alpha = gamma = 1`) and normalises the source to Ip. At flat-top Ip barely moves, so the flux
span `psi_axis - psi_bnd` cannot move either -- ours sits at 0.055 +- 0.006 across a whole
discharge. The real discharge's span **grows 2.4x**, 0.036 -> 0.089, because its current profile
peaks as li rises. Frozen against growing is an ANTI-correlation, measured at -0.75 / -0.66 /
-0.55 on the three demo shots, and it propagates into every geometric scalar the scorer derives:
R_axis correlates -0.75 with the truth, tri_top -0.74, tri_bot -0.65. See `diagnose_cons.py`.

**Why the solenoid.** The plasma's poloidal flux is the flux the solenoid has delivered, so this is
close to a restatement rather than a fit: `corr(truth span, solenoid current) = -0.983`, and
-0.98 / -0.99 / -0.98 **within each shot separately**. Leave-one-shot-out R2 is **+0.9698**
against +0.9720 in sample, and the solenoid coefficient holds to **0.3%** across the three fits
(Ip's to 12%) -- the stability test that threw out a twelve-feature q95 model still passes here
comfortably.

Both inputs are read at inference from `Shot` and nothing is withheld: solenoid current in kA and
|Ip| in MA, on the EFIT frame clock.

**What it is worth.** Driving the per-frame profile peaking to hit this target takes Consistency
from 0.069 to **0.268** and R2_psi from 0.936 to **0.960**, with D_LCFS 0.4125 -> 0.3128 -- and
the held-out version captures 90% of what an oracle handed the truth's own span achieves. Three of
the seven derived scalars cross from negative into positive.
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from mast.scalar_model import ScalarModel

FloatArray = npt.NDArray[np.floating]

# The one input column this reads, named here so `predict.solve_shot` does not hard-code it.
SOL_COLUMN = "magnetics_sol_current"

# Fitted on all 115 demo frames; the held-out numbers above come from refitting on two shots.
SPAN = ScalarModel(
    names=("sol", "ip"),
    log=False,
    weight=(-0.01837977859476067, -0.001588415589633159),
    centre=(-13.416021585723199, 0.6154458275448667),
    spread=(16.196256640411857, 0.03448291584798319),
    const=0.06582356703128545,
    lo=0.010, hi=0.250,
)

# The bisection's bracket on the profile peaking. Measured on shot 28348, mean span runs 0.039 at
# gamma 0.30 to 0.215 at gamma 3.0, so this brackets the whole range the truth occupies (0.034 to
# 0.100) with room at both ends. Seven halvings in log space leave gamma known to 1.8%, which is
# far finer than the span target's own error.
GAMMA_LO, GAMMA_HI, GAMMA_STEPS = 0.30, 3.0, 7


def target(sol_ka: FloatArray, ip_amps: FloatArray) -> FloatArray:
    """The flux span each frame should come out at, in Wb per radian."""
    sol = np.asarray(sol_ka, dtype=np.float64)
    ip = np.abs(np.asarray(ip_amps, dtype=np.float64)) / 1e6
    return np.array([SPAN({"sol": float(s), "ip": float(i)}) for s, i in zip(sol, ip, strict=True)])
