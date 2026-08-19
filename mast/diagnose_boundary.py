#!/usr/bin/env python3
"""D_LCFS is a Hausdorff distance — so WHERE on the boundary is the worst point?

    uv run python mast/diagnose_boundary.py

The leaderboard put D_LCFS at **0.4121 against the leader's 0.2109**, and `symmetric_hausdorff`
divided by the major radius is a MAXIMUM over the contour, not an average. A boundary that is
accurate everywhere except its divertor legs scores the same as one that is wrong all over, and the
two need completely different fixes. This separates them, on the 115 demo frames:

* **where** the worst point sits, as a poloidal angle about the magnetic axis and as a share of the
  contour that lies within a given distance;
* **what kind** of error it is — the contour's own size, its position, or its shape — by removing
  each in turn and re-measuring what is left:

      raw          our contour against the true one, as the scorer takes it
      -centroid    both centred on their own centroid: what remains is size and shape
      -size        both centred AND scaled to the same mean radius: what remains is shape alone

A large drop at `-centroid` means the plasma is in the wrong place and the fix is the solve's
equilibrium position; a large drop at `-size` means the boundary is the wrong size, which is the
source amplitude; a small drop at both means the shape is wrong, and only the source's spatial
profile can move that.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "fusion_scoring"))

from contour import resample_closed, symmetric_hausdorff  # noqa: E402
from lcfs import extract_lcfs, major_radius  # noqa: E402

from mast import shot as shot_mod  # noqa: E402
from mast.calibration import CALIBRATION  # noqa: E402
from solver.greens import build_basis  # noqa: E402
from solver.gs import operator, solve  # noqa: E402
from solver.machine import MAST  # noqa: E402

FloatArray = npt.NDArray[np.floating]
DEMO = HERE.parent / "parquet_data"
MACHINE = "MAST"
N_POINTS = 512


def psi_stack(cell: Any) -> FloatArray:
    return np.stack([np.stack([np.asarray(c, dtype=np.float64) for c in np.asarray(f)])
                     for f in np.asarray(cell)])


def centred(c: FloatArray) -> tuple[FloatArray, FloatArray]:
    """The contour about its own centroid, and the centroid."""
    mid = c.mean(axis=0)
    return c - mid, mid


def normalised(c: FloatArray) -> FloatArray:
    """Centred and scaled to unit mean radius — shape with size and position removed."""
    d, _ = centred(c)
    return d / max(float(np.hypot(d[:, 0], d[:, 1]).mean()), 1e-12)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demo", type=Path, default=DEMO)
    args = ap.parse_args()

    paths = sorted(args.demo.glob("mast_shot_*.parquet"))
    basis = None
    raw, no_pos, no_size, angles, frac_close, sizes = [], [], [], [], [], []
    for path in paths:
        row = pd.read_parquet(path).iloc[0]
        if basis is None:
            basis = build_basis(row, MAST)
            op = operator(MAST, basis.grid_R, basis.grid_Z)
            mask_f = op.mask.astype(float)
        s = shot_mod.read(row, basis.columns)
        gains = np.array([CALIBRATION.gains[c] for c in basis.columns], dtype=np.float64)
        coil = basis.flux(s.currents * gains[None, :])
        truth = psi_stack(row["efit_psirz"])

        for k in range(s.times.size):
            got = solve(coil[k], float(s.ip[k]), op, profile=CALIBRATION.profile,
                        n_iter=CALIBRATION.n_iter, relax=CALIBRATION.relax)
            ours = extract_lcfs(got.psi, op.grid_R, op.grid_Z, MACHINE, op.mask, mask_f)
            ref = extract_lcfs(truth[k], op.grid_R, op.grid_Z, MACHINE, op.mask, mask_f)
            if ours is None or ref is None:
                continue
            a = resample_closed(np.asarray(ours, dtype=np.float64), N_POINTS)
            b = resample_closed(np.asarray(ref, dtype=np.float64), N_POINTS)
            rgeo = major_radius(b)
            raw.append(symmetric_hausdorff(a, b) / rgeo)
            no_pos.append(symmetric_hausdorff(centred(a)[0], centred(b)[0]) / rgeo)
            no_size.append(symmetric_hausdorff(normalised(a), normalised(b))
                           * float(np.hypot(*centred(b)[0].T).mean()) / rgeo)

            # Where the worst point is, as an angle about the TRUE contour's centroid, and how
            # much of our contour is already within a tenth of rgeo of the truth.
            mid = b.mean(axis=0)
            d = np.hypot(a[:, None, 0] - b[None, :, 0], a[:, None, 1] - b[None, :, 1])
            near = d.min(axis=1)
            worst = int(np.argmax(near))
            angles.append(np.degrees(np.arctan2(a[worst, 1] - mid[1], a[worst, 0] - mid[0])))
            frac_close.append(float((near < 0.10 * rgeo).mean()))
            sizes.append(float(np.hypot(*centred(a)[0].T).mean())
                         / float(np.hypot(*centred(b)[0].T).mean()))
        print(f"  {path.stem}: {s.times.size} frames")

    n = len(raw)
    print(f"\n{n} frames with a contour on both sides\n")
    print(f"  {'D_LCFS, as the scorer takes it':<34} {np.mean(raw):>8.4f}")
    print(f"  {'  with the position removed':<34} {np.mean(no_pos):>8.4f}")
    print(f"  {'  with position and size removed':<34} {np.mean(no_size):>8.4f}")
    print(f"\n  our contour is {np.mean(sizes):.3f}x the true one on average "
          f"(5-95%: {np.percentile(sizes, 5):.3f}-{np.percentile(sizes, 95):.3f})")
    print(f"  {np.mean(frac_close):.1%} of our contour is already within 0.10 rgeo of the truth")
    a = np.asarray(angles)
    print("\n  where the worst point sits, by poloidal angle about the true centroid "
          "(0 deg = outboard midplane, +90 = top):")
    hist, edges = np.histogram(a, bins=8, range=(-180, 180))
    for h, lo, hi in zip(hist, edges[:-1], edges[1:], strict=True):
        print(f"    {lo:>+5.0f}..{hi:>+5.0f} deg  {h:>4}  {'#' * int(40 * h / max(hist.max(), 1))}")
    print("\n  Read the three D_LCFS lines as a decomposition. Whichever removal drops it most is "
          "the error that dominates, and each points at a different part of the solve.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
