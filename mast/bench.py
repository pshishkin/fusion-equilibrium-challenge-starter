#!/usr/bin/env python3
"""Solve the demo shots once, then evaluate scalar models in milliseconds instead of minutes.

The leaderboard has now measured which half of this pipeline generalises. Between the entry that
scored 0.5566 and the one that scored 0.56777, **R2_qb went 0.0574 -> 0.2157 while R2_psi went
0.8895 -> 0.8805**: q95's affine calibration transferred at 91% of its local gain, and every change
aimed at the flux map lost on the fold what it gained on three shots. So the thing worth iterating
on is the scalars, and iterating needs to be cheap.

Solving 115 frames three times over (one per ensemble member) costs about 80 s. Every scalar model
downstream of that reads the same equilibria. This module does the solve once, caches what the
scalars need, and exposes:

    frames()   the per-frame quantities any q95 or betaN model could want, plus the truth
    loo(fit)   leave-one-shot-out over the three demo shots, returning HELD-OUT pooled R2

`loo` is the gate, not the in-sample fit. The three shots are consecutive numbers from one campaign,
so a held-out shot here is a sibling and this measures fitting bias only — but that is exactly the
failure mode a two-parameter affine on 115 correlated frames is exposed to, and leave-one-out
already caught it once: R2_qb reads 0.2771 in sample against 0.1663 held out.
"""
from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "fusion_scoring"))

from lcfs import extract_lcfs, major_radius  # noqa: E402

from mast.calibration import SHIPPED  # noqa: E402
from mast.predict import solve_shot  # noqa: E402
from solver import scalars  # noqa: E402
from solver.gs import Operator, Solved  # noqa: E402
from solver.machine import MAST  # noqa: E402

FloatArray = npt.NDArray[np.floating]
DEMO = HERE.parent / "parquet_data"
PSI_N = np.linspace(0.0, 1.0, 33)
LEVELS = (0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99)


@dataclass(frozen=True)
class Frames:
    """One row per frame, with the shot each belongs to so leave-one-out can split on it."""
    shot: npt.NDArray[np.int64]
    q95: FloatArray            # truth
    beta_n: FloatArray         # truth
    ip: FloatArray             # amperes
    tf: FloatArray             # kA
    shape: FloatArray          # (n, len(LEVELS)) the contour integral at each proxy surface
    a: FloatArray              # minor radius from the solve
    r_axis: FloatArray
    z_axis: FloatArray
    kappa: FloatArray          # from the solved contour, via the scorer's own extractor
    delta: FloatArray
    volume: FloatArray
    p_vol: FloatArray          # volume-averaged Thomson pressure, Pa
    p_axis: FloatArray         # on-axis Thomson pressure
    span: FloatArray           # psi_axis - psi_bnd


def _geometry(psi: FloatArray, op: Operator) -> tuple[float, float, float, float]:
    """kappa, delta, minor radius and volume from the scorer's own contour, or nan."""
    mask = op.mask
    c = extract_lcfs(psi, op.grid_R, op.grid_Z, MAST.name, mask, mask.astype(float))
    if c is None:
        return (np.nan,) * 4
    r, z = c[:, 0], c[:, 1]
    r_lo, r_hi = float(r.min()), float(r.max())
    a = 0.5 * (r_hi - r_lo)
    r0 = 0.5 * (r_hi + r_lo)
    kappa = 0.5 * float(z.max() - z.min()) / max(a, 1e-9)
    delta = (r0 - 0.5 * (float(r[np.argmax(z)]) + float(r[np.argmin(z)]))) / max(a, 1e-9)
    return kappa, delta, a, 2.0 * np.pi * major_radius(c) * np.pi * a * a * kappa


def build(demo: Path = DEMO, cals: tuple = SHIPPED) -> Frames:
    """Solve every demo frame once and reduce it to what a scalar model can read."""
    cols: dict[str, list] = {k: [] for k in Frames.__annotations__}
    for i, path in enumerate(sorted(demo.glob("mast_shot_*.parquet"))):
        row = pd.read_parquet(path).iloc[0]
        per_member: list[list[Solved]] = []
        shot = None
        for cal in cals:
            s, op, solved = solve_shot(row, cal)
            per_member.append(solved)
            shot = s
        assert shot is not None
        q95 = np.asarray(row["efit_q95"], dtype=np.float64)
        bn = np.asarray(row["efit_beta_n"], dtype=np.float64)
        for k in range(shot.times.size):
            # The ensemble's flux map is the average; every geometric quantity is read off that,
            # exactly as the scorer will read it.
            psi = np.mean([m[k].psi for m in per_member], axis=0)
            ref = per_member[0][k]
            avg = Solved(psi=psi, psi_axis=float(np.mean([m[k].psi_axis for m in per_member])),
                         psi_bnd=float(np.mean([m[k].psi_bnd for m in per_member])),
                         r_axis=float(np.mean([m[k].r_axis for m in per_member])),
                         z_axis=float(np.mean([m[k].z_axis for m in per_member])),
                         inside=ref.inside, j_phi=ref.j_phi, iterations=0, moved=0.0)
            kappa, delta, a, vol = _geometry(psi, op)
            try:
                prof = scalars.pressure_profile(avg, op, shot.ts_R, shot.ts_te[k], shot.ts_ne[k],
                                                PSI_N)
                p_vol = scalars.volume_average(avg, op, prof, PSI_N)
                p_axis = float(prof[0])
            except ValueError:
                p_vol = p_axis = np.nan
            cols["shot"].append(i)
            cols["q95"].append(q95[k])
            cols["beta_n"].append(bn[k])
            cols["ip"].append(float(shot.ip[k]))
            cols["tf"].append(float(shot.tf[k]))
            cols["shape"].append([scalars.q_shape(avg, op, lv) for lv in LEVELS])
            cols["a"].append(a)
            cols["r_axis"].append(avg.r_axis)
            cols["z_axis"].append(avg.z_axis)
            cols["kappa"].append(kappa)
            cols["delta"].append(delta)
            cols["volume"].append(vol)
            cols["p_vol"].append(p_vol)
            cols["p_axis"].append(p_axis)
            cols["span"].append(avg.psi_axis - avg.psi_bnd)
        print(f"  {path.stem}: {shot.times.size} frames", flush=True)
    out: dict[str, Any] = {k: np.asarray(v, dtype=np.int64 if k == "shot" else np.float64)
                           for k, v in cols.items()}
    return Frames(**out)


def pooled_r2(truth: FloatArray, pred: FloatArray) -> float:
    ok = np.isfinite(truth) & np.isfinite(pred)
    t, p = truth[ok], pred[ok]
    if t.size < 4:
        return float("nan")
    return 1.0 - float(((t - p) ** 2).sum()) / float(((t - t.mean()) ** 2).sum())


def loo(f: Frames, target: str, fit: Callable[[Frames, npt.NDArray[np.bool_]], Any],
        apply: Callable[[Any, Frames, npt.NDArray[np.bool_]], FloatArray]) -> tuple[float, float]:
    """(held-out pooled R2, in-sample pooled R2) over leave-one-shot-out on the demo shots."""
    truth = getattr(f, target)
    held = np.full(truth.shape, np.nan)
    ins = np.full(truth.shape, np.nan)
    for s in np.unique(f.shot):
        out = f.shot == s
        model = fit(f, ~out)
        held[out] = apply(model, f, out)
        ins[out] = apply(fit(f, np.ones_like(out)), f, out)
    return pooled_r2(truth, held), pooled_r2(truth, ins)
