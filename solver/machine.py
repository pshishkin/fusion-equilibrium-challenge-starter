#!/usr/bin/env python3
"""What the solver needs to know about a machine, and nothing else.

A handful of numbers and four column names. Everything else in `solver/` is physics that does not
care which tokamak it is pointed at — which is the point, because the only way to choose a
modelling decision for MAST is to test it on the machine that has 7041 labelled shots.

The DIII-D entry exists for validation, not for Challenge 1: that challenge is won by a fitted
model in `my_experiments/`, and nothing here competes with it.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "fusion_scoring"))

from common import AXIS_SIGN

MASKS = Path(__file__).resolve().parent.parent / "fusion_scoring" / "masks"

# MAST's first wall, as EFIT itself carries it: `efm/limiterr` / `efm/limiterz` from the public
# MAST archive at STFC Echo (https://s3.echo.stfc.ac.uk/mast/level1/shots/{shot}.zarr, CC BY 4.0),
# a closed polygon of 37 vertices. It is a MACHINE CONSTANT and not an equilibrium: byte-identical
# on shots sampled from all five campaigns M5 through M9, which is what makes taking it the same
# kind of act as reading the coil rectangles the competition ships.
#
# It corrects a guess. `mast/` inferred an inboard wall at R = 0.1963 from where `efit_lcfs_r`
# bottomed out on the three demo shots; the machine says **0.1952**, 1.1 mm further in — and,
# more usefully, supplies the whole shape, so the boundary search can ask where the plasma touches
# the wall rather than only where it crosses one vertical line at the centre column.
MAST_LIMITER = (
    (1.899999976158142, 0.4050000011920929),
    (1.5551042556762695, 0.4050000011920929),
    (1.5551042556762695, 0.8225002288818359),
    (1.407930612564087, 0.8225002288818359),
    (1.407930612564087, 1.0330003499984741),
    (1.039931058883667, 1.0330003499984741),
    (1.039931058883667, 1.1950000524520874),
    (1.899999976158142, 1.1950000524520874),
    (1.899999976158142, 1.8250000476837158),
    (0.5649306774139404, 1.8250000476837158),
    (0.5649306774139404, 1.7280815839767456),
    (0.7835000157356262, 1.7280815839767456),
    (0.7835000157356262, 1.7155816555023193),
    (0.5825902819633484, 1.5470000505447388),
    (0.4165000021457672, 1.5470000505447388),
    (0.2800000011920929, 1.683500051498413),
    (0.2800000011920929, 1.229088544845581),
    (0.19524440169334412, 1.0835000276565552),
    (0.19524440169334412, -1.0835000276565552),
    (0.2800000011920929, -1.229088544845581),
    (0.2800000011920929, -1.683500051498413),
    (0.4165000021457672, -1.5470000505447388),
    (0.5825902819633484, -1.5470000505447388),
    (0.7835000157356262, -1.7155816555023193),
    (0.7835000157356262, -1.7280815839767456),
    (0.5649306774139404, -1.7280815839767456),
    (0.5649306774139404, -1.8250000476837158),
    (1.899999976158142, -1.8250000476837158),
    (1.899999976158142, -1.1950000524520874),
    (1.039931058883667, -1.1950000524520874),
    (1.039931058883667, -1.0330003499984741),
    (1.407930612564087, -1.0330003499984741),
    (1.407930612564087, -0.8225002288818359),
    (1.5551042556762695, -0.8225002288818359),
    (1.5551042556762695, -0.4050000011920929),
    (1.899999976158142, -0.4050000011920929),
    (1.899999976158142, 0.4050000011920929),
)


@dataclass(frozen=True)
class Machine:
    name: str
    envelope: str          # the scorer's own vessel mask, in fusion_scoring/masks
    r_limiter: float       # inboard wall radius, metres
    x_reach: float         # how far from the axis a saddle is still this plasma's X-point
    ip_column: str
    tf_column: str
    ip_times: str | None   # a signal with its OWN time base names it here; None = the shared one
    current_unit: float    # multiply a shipped current by this to get amperes
    # Where the magnetic axis is held vertically, and how far it may wander. An elongated plasma
    # has NO stable vertical equilibrium — real tokamaks hold it with feedback, and EFIT pins it
    # with magnetic measurements this challenge withholds — so a free-boundary Picard iteration
    # slides the plasma to wherever the vacuum field happens to be stable. On DIII-D, measured, it
    # slides a WHOLE METRE: the solve puts the axis at Z = -1.0 against a true -0.03.
    #
    # Pinning it costs almost nothing, because the real axis barely moves: over 5231 live frames of
    # 40 DIII-D shots, Z_axis is -0.0172 +/- 0.0413 m, inside one 0.050 m grid cell on 89% of them.
    # `None` leaves the search free, which is what MAST shipped with and what its leaderboard entry
    # was scored under.
    # The first wall as a closed polygon, when the machine's own is known. `None` falls back to
    # `r_limiter` alone, which is a vertical line at the inboard wall and nothing else.
    wall: tuple | None = None
    z_pin: float | None = None
    z_band: float = 0.15
    # Whether the pin also moves the CURRENT, not just the O-point search. Restricting where the
    # search may look bounds the reported axis and leaves the source free to sit elsewhere, and on
    # DIII-D that is exactly what happens: with a band of +-0.15 m the axis presses against its
    # lower edge at dZ = -0.173 +- 0.055 m. Shifting the current distribution so its centroid lands
    # on `z_pin` is the constraint a feedback system actually applies.
    z_pin_current: bool = False

    @property
    def sign(self) -> float:
        """The machine's stored flux convention: +1 if the axis is a maximum of `efit_psirz`."""
        return float(AXIS_SIGN[self.name])

    @property
    def mask_path(self) -> Path:
        return MASKS / self.envelope


# MAST's inboard limiter is the centre column, and 0.1963 is not a guess: `efit_lcfs_r` bottoms out
# at 0.1963 on all 115 demo frames — a guess that the machine's own wall, taken from the public
# archive, corrects to 0.19524 (`MAST_LIMITER` above, and see `solver/mast_archive.py` for why
# only the geometry is taken from there).
MAST = Machine(
    name="MAST", envelope="mast_envelope.npz", r_limiter=0.19524440169334412,
    x_reach=1.2, wall=MAST_LIMITER,
    ip_column="magnetics_plasma_current", tf_column="magnetics_tf_current",
    ip_times=None, current_unit=1e3,
)

# DIII-D's inner wall sits at R = 1.016 m. Unlike MAST's, this one is NOT read off the data: the
# shipped `efit_lcfs_r` bottoms out at 1.047-1.095 on the demo shots, i.e. the plasma is diverted
# and never touches it, so the limiter branch of the boundary search rarely binds here.
D3D = Machine(
    name="DIII-D", envelope="d3d_envelope.npz", r_limiter=1.016, x_reach=1.2,
    ip_column="magnetics_plasma_current", tf_column="magnetics_bcoil",
    ip_times="magnetics_plasma_current_times", current_unit=1e3,
    z_pin=-0.0172, z_band=0.15, z_pin_current=True,
)

MACHINES = {m.name: m for m in (MAST, D3D)}
