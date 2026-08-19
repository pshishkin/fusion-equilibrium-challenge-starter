#!/usr/bin/env python3
"""Where does q95's error actually come from — the formula, or the map underneath it?

    uv run python mast/diagnose_scalars.py

The leaderboard put `R2_{q95, betaN}` at **0.0574 against the leader's 0.5079**, which is 40% of
the whole deficit and the cheapest term in the composite to move. Before building anything, this
decides WHICH of two very different fixes is the one:

* if the contour integral is accurate **given a good map**, then q95 is downstream of the flux
  solve and there is nothing to fix in it separately — the work is all in the map;
* if it is inaccurate even on the TRUE map, the method is wrong and can be replaced on its own,
  which is a far cheaper piece of work.

So every estimator is run twice, once on our solved flux and once on the shipped `efit_psirz` of
the same frame, and each gets its own least-squares constant so that what is compared is SHAPE and
not level. Four estimators of q95:

  contour     (F / 2pi) * integral dl / (R |grad psi|) at psi_N = 0.90 — what ships
  uckan       5 a^2 B0 / (R0 Ip) * (1 + kappa^2 (1 + 2 delta^2 - 1.2 delta^3)) / 2

and betaN from Thomson's midplane pressure, likewise on both maps. `a`, `R0`, `kappa` and `delta`
come from the scorer's own `derive_frame` and `extract_lcfs`, so the geometry is measured the same
way the metric measures it.

Reads the three demo shots, which carry `efit_psirz` and every EFIT scalar — 115 frames, and the
only MAST truth in the released dataset.
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

from derive import derive_frame  # noqa: E402
from lcfs import extract_lcfs  # noqa: E402

from mast import shot as shot_mod  # noqa: E402
from mast.calibration import CALIBRATION  # noqa: E402
from solver import scalars  # noqa: E402
from solver.greens import build_basis  # noqa: E402
from solver.gs import Solved, operator, solve  # noqa: E402
from solver.machine import MAST  # noqa: E402

FloatArray = npt.NDArray[np.floating]
DEMO = HERE.parent / "parquet_data"
MACHINE = "MAST"


def psi_stack(cell: Any) -> FloatArray:
    return np.stack([np.stack([np.asarray(c, dtype=np.float64) for c in np.asarray(f)])
                     for f in np.asarray(cell)])


def geometry(psi: FloatArray, op: Any) -> dict:
    """`a`, `R0`, `kappa`, `delta` of one frame, via the scorer's own contour and derivations."""
    mask = op.mask
    c = extract_lcfs(psi, op.grid_R, op.grid_Z, MACHINE, mask, mask.astype(float))
    if c is None:
        return {}
    d = derive_frame(psi, op.grid_R, op.grid_Z, MACHINE, mask, mask.astype(float), contour=c)
    r_lo, r_hi = float(c[:, 0].min()), float(c[:, 0].max())
    return {"a": 0.5 * (r_hi - r_lo), "R0": 0.5 * (r_hi + r_lo),
            "kappa": float(d["kappa"]),
            "delta": 0.5 * (float(d["tri_top"]) + float(d["tri_bot"]))}


def uckan(g: dict, b0: float, ip_ma: float) -> float:
    """The standard cylindrical q95 with its shape factor. `b0` and the 5 carry the units."""
    if not g or ip_ma == 0 or g["R0"] <= 0:
        return float("nan")
    shape = (1.0 + g["kappa"] ** 2 * (1.0 + 2.0 * g["delta"] ** 2 - 1.2 * g["delta"] ** 3)) / 2.0
    return 5.0 * g["a"] ** 2 * b0 / (g["R0"] * abs(ip_ma)) * shape


def scored(name: str, pred: list[float], true: list[float]) -> None:
    """Print the best-constant R2 and the correlation — the constant is free, the shape is not."""
    p = np.asarray(pred, dtype=np.float64)
    t = np.asarray(true, dtype=np.float64)
    ok = np.isfinite(p) & np.isfinite(t)
    p, t = p[ok], t[ok]
    if p.size < 8:
        print(f"  {name:<34} only {p.size} usable frames")
        return
    k = float((p * t).sum() / (p * p).sum())
    r2_scale = 1.0 - float(((t - k * p) ** 2).sum()) / float(((t - t.mean()) ** 2).sum())
    a = np.column_stack([p, np.ones(p.size)])
    coef, *_ = np.linalg.lstsq(a, t, rcond=None)
    r2_aff = 1.0 - float(((t - a @ coef) ** 2).sum()) / float(((t - t.mean()) ** 2).sum())
    print(f"  {name:<34} n={p.size:>4}  corr {np.corrcoef(p, t)[0, 1]:+.3f}  "
          f"R2(scale) {r2_scale:+.4f}  R2(affine) {r2_aff:+.4f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demo", type=Path, default=DEMO)
    args = ap.parse_args()

    paths = sorted(args.demo.glob("mast_shot_*.parquet"))
    if not paths:
        raise SystemExit(f"no mast_shot_*.parquet in {args.demo}")

    got: dict[str, list[float]] = {k: [] for k in
                                   ("q_true", "q_ours_contour", "q_true_contour",
                                    "q_ours_uckan", "q_true_uckan",
                                    "b_true", "b_ours", "b_truegeom")}
    basis = None
    nodes = np.linspace(0.0, 1.0, 33)
    for path in paths:
        row = pd.read_parquet(path).iloc[0]
        if basis is None:
            basis = build_basis(row, MAST)
            op = operator(MAST, basis.grid_R, basis.grid_Z)
        s = shot_mod.read(row, basis.columns)
        gains = np.array([CALIBRATION.gains[c] for c in basis.columns], dtype=np.float64)
        coil = basis.flux(s.currents * gains[None, :])
        truth = psi_stack(row["efit_psirz"])
        q95_t = np.asarray(row["efit_q95"], dtype=np.float64)
        bn_t = np.asarray(row["efit_beta_n"], dtype=np.float64)

        for k in range(s.times.size):
            ours = solve(coil[k], float(s.ip[k]), op, profile=CALIBRATION.profile,
                         n_iter=CALIBRATION.n_iter, relax=CALIBRATION.relax)
            ref = _reference(truth[k], op)
            g_ours, g_true = geometry(ours.psi, op), geometry(truth[k], op)
            b0 = abs(CALIBRATION.f_per_ka * float(s.tf[k])) / max(ours.r_axis, 1e-9)
            ip_ma = float(s.ip[k]) / 1e6

            got["q_true"].append(float(q95_t[k]))
            got["q_ours_contour"].append(scalars.q_shape(ours, op))
            got["q_true_contour"].append(scalars.q_shape(ref, op))
            got["q_ours_uckan"].append(uckan(g_ours, b0, ip_ma))
            got["q_true_uckan"].append(uckan(g_true, b0, ip_ma))

            got["b_true"].append(float(bn_t[k]))
            got["b_ours"].append(_beta(ours, op, s, k, nodes))
            got["b_truegeom"].append(_beta(ref, op, s, k, nodes))
        print(f"  {path.stem}: {s.times.size} frames")

    print(f"\nq95, {len(got['q_true'])} frames — each estimator with its own free constant:")
    for key, label in (("q_ours_contour", "contour integral, OUR map"),
                       ("q_true_contour", "contour integral, TRUE map"),
                       ("q_ours_uckan", "Uckan formula, OUR geometry"),
                       ("q_true_uckan", "Uckan formula, TRUE geometry")):
        scored(label, got[key], got["q_true"])

    print(f"\nbetaN, {len(got['b_true'])} frames:")
    for key, label in (("b_ours", "Thomson pressure, OUR map"),
                       ("b_truegeom", "Thomson pressure, TRUE map")):
        scored(label, got[key], got["b_true"])

    print("\n  Read the OUR/TRUE pair for each estimator. A large gap means the flux map is what "
          "limits that scalar and it cannot be fixed on its own; a small one means the method is "
          "the limit and can be replaced without touching the solve.")
    return 0


def _reference(psi: FloatArray, op: Any) -> Solved:
    """A `Solved` carrying the TRUE flux map, so the same estimators can be run on it."""
    from o_point import find_o_point

    from solver.gs import _boundary_flux
    ra, za, iz, ir, _ = find_o_point(psi, op.grid_R, op.grid_Z, op.mask)
    bnd = _boundary_flux(psi, op, int(iz), int(ir))
    return Solved(psi=psi, psi_axis=float(psi[iz, ir]), psi_bnd=bnd, r_axis=ra, z_axis=za,
                  inside=(psi >= bnd) & op.mask, j_phi=np.zeros_like(psi),
                  iterations=0, moved=0.0)


def _beta(s: Solved, op: Any, sh: Any, k: int, nodes: FloatArray) -> float:
    try:
        p = scalars.pressure_profile(s, op, sh.ts_R, sh.ts_te[k], sh.ts_ne[k], nodes)
        return scalars.beta_n(s, op, float(sh.tf[k]), CALIBRATION.f_per_ka, float(sh.ip[k]),
                              p, nodes)
    except ValueError:
        return float("nan")


if __name__ == "__main__":
    raise SystemExit(main())
