#!/usr/bin/env python3
"""Leave one demo shot out: the first honest held-out measurement MAST has ever had.

    uv run python mast/loo.py --rounds 6

Every MAST number in this repository — R2_psi 0.9317, S 0.6296 — is measured on the three demo
shots that the eleven coil gains, the profile and the two affine calibrations were all fitted on.
The leaderboard is the only held-out reading, and it comes back once per submission and only as a
total. So the size of the in-sample bias is unknown, and three modelling decisions have already
been refuted by the board after looking good locally.

This closes that gap the cheap way. For each shot in turn: refit the whole calibration on the OTHER
two, score the held-out one through `local_score.py`, rotate. Three fits, three honest numbers.

**What it can and cannot say.** The three shots are 28348 / 28350 / 28351 — consecutive numbers,
one campaign, one configuration — so a held-out shot here is a sibling, not a stranger. This
measures the bias from fitting on a sample at all; it cannot measure transfer to a discharge of a
different kind, and the 1206-shot fold contains plenty. Read it as an UPPER bound on how well the
calibration travels.

The base rate to read it against is `solver/how_many_shots.py`, which asked the same question where
the shots are plentiful: on DIII-D, a calibration fitted on three shots scores **-0.601 +- 0.120**
of R2_psi on shots it has not seen. If MAST does the same, everything here is sample noise. If it
does not, the difference is worth naming — MAST ships 812 conductor rows against DIII-D's one
rectangle per coil, so its solenoid has a shape the fit can separate from a constant.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from mast import calibrate  # noqa: E402
from mast.calibration import CALIBRATION, Calibration  # noqa: E402
from solver.greens import CoilBasis  # noqa: E402
from solver.gs import operator  # noqa: E402
from solver.machine import MAST  # noqa: E402

DEMO = HERE.parent / "parquet_data"


def shrink(scale: float, offset: float, mean: float, keep: float) -> tuple[float, float]:
    """Pull an affine toward the constant it degenerates into, keeping the fold mean fixed.

    `keep = 1` is the raw least-squares fit and `keep = 0` is the constant. The reason to have the
    knob is measured, not assumed: leave-one-out puts R2_qb at **0.2771 in sample and 0.1663 held
    out**, the largest in-sample bias of any term, while the flux map — eleven gains fitted the
    same way — shows none at all. Four numbers on 115 correlated frames is where this pipeline
    overfits, and it is worth 0.017 of S.
    """
    return keep * scale, mean * (1.0 - keep) + keep * offset


def fit_on(basis: CoilBasis, shots: list[dict], args: argparse.Namespace) -> Calibration:
    """The whole calibration — gains, then the TF constant, then both affines — on these shots."""
    gains = calibrate.fit_gains(basis, shots, calibrate.bootstrap_plasma(basis, shots),
                                np.ones(len(basis.columns)), 1.0)
    for _ in range(args.rounds):
        _total, plasma = calibrate.solve_all(basis, shots, gains, CALIBRATION.profile,
                                             args.n_iter, args.relax)
        gains = calibrate.fit_gains(basis, shots, plasma, gains, args.damp)
    calibrate.solve_all(basis, shots, gains, CALIBRATION.profile, args.n_iter, args.relax)
    f_per_ka, _r2, _n = calibrate.fit_f_per_ka(basis, shots)
    q_scale, q_offset, _ = calibrate.fit_q95(basis, shots, f_per_ka)
    b_scale, b_offset, _ = calibrate.fit_beta_n(basis, shots, f_per_ka)
    q_mean = float(np.concatenate([s["q95"] for s in shots]).mean())
    b_mean = float(np.concatenate([s["beta_n"] for s in shots]).mean())
    q_scale, q_offset = shrink(q_scale, q_offset, q_mean, args.keep)
    b_scale, b_offset = shrink(b_scale, b_offset, b_mean, args.keep)
    return Calibration(gains=dict(zip(basis.columns, gains, strict=True)),
                       profile=CALIBRATION.profile,
                       q95_const=q_mean, beta_n_const=b_mean,
                       f_per_ka=f_per_ka, q95_scale=q_scale, q95_offset=q_offset,
                       beta_n_scale=b_scale, beta_n_offset=b_offset,
                       n_iter=args.n_iter, relax=args.relax)


def score(path: Path, cal: Calibration, out: Path) -> dict:
    """Predict one shot with `cal` and hand it to the real scorer. One metric, always."""
    import pandas as pd

    from mast.predict import solve_shot, value_scalars

    row = pd.read_parquet(path).iloc[0]
    s, op, solved = solve_shot(row, cal)
    q, b = value_scalars(s, op, solved, cal)
    np.savez_compressed(
        out,
        shot_0000_psirz=np.stack([g.psi for g in solved]).astype(np.float16),
        shot_0000_q95=q.astype(np.float32),
        shot_0000_betaN=b.astype(np.float32))
    p = subprocess.run([sys.executable, "local_score.py", "--source", "local", "--files",
                        str(path), "--pred", str(out), "--jobs", "1"],
                       capture_output=True, text=True, cwd=HERE.parent)
    if "COMPOSITE" not in p.stdout:
        raise SystemExit(f"the scorer failed on {path.name}:\n{p.stdout[-2000:]}\n"
                         f"{p.stderr[-2000:]}")

    def grab(pat: str) -> float:
        m = re.search(pat, p.stdout)
        if m is None:
            raise SystemExit(f"no {pat!r} in the scorer's output for {path.name}")
        return float(m.group(1))

    return {"S": grab(r"COMPOSITE S = ([\d.]+)"), "psi": grab(r"R2_psi\s+([\d.-]+)"),
            "qb": grab(r"R2_\{q95,betaN\}\s+([\d.-]+)"),
            "lcfs": grab(r"1 - D_LCFS\s+([\d.-]+)"), "cons": grab(r"Consistency\s+([\d.-]+)")}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--demo", type=Path, default=DEMO)
    ap.add_argument("--out", type=Path, default=Path("/tmp/mast_loo.npz"))
    ap.add_argument("--n-iter", type=int, default=CALIBRATION.n_iter)
    ap.add_argument("--relax", type=float, default=CALIBRATION.relax)
    ap.add_argument("--damp", type=float, default=0.4)
    ap.add_argument("--keep", type=float, default=1.0,
                    help="how much of the fitted affine to keep; 0 falls back to the constant")
    args = ap.parse_args()

    paths = sorted(args.demo.glob("mast_shot_*.parquet"))
    if len(paths) < 3:
        raise SystemExit(f"leave-one-out needs at least three shots; {args.demo} has {len(paths)}")
    basis, shots = calibrate.load(paths)
    operator(MAST, basis.grid_R, basis.grid_Z)
    print(f"{len(paths)} demo shots, {sum(s['shot'].times.size for s in shots)} frames; "
          f"{CALIBRATION.profile}; affine keep={args.keep}")
    print(f"\n{'held out':<18}{'S':>9}{'R2_psi':>9}{'R2_qb':>8}{'1-D':>8}{'Cons':>8}"
          f"{'   P2/P4/P5 gains'}")

    held, in_sample = [], []
    for i, path in enumerate(paths):
        rest = [s for j, s in enumerate(shots) if j != i]
        cal = fit_on(basis, rest, args)
        out = score(path, cal, args.out)
        held.append(out)
        g = [cal.gains[c] for c in ("magnetics_p2u_current", "magnetics_p4u_current",
                                    "magnetics_p5u_current")]
        print(f"{path.stem:<18}{out['S']:>9.4f}{out['psi']:>9.4f}{out['qb']:>8.4f}"
              f"{out['lcfs']:>8.4f}{out['cons']:>8.4f}   "
              f"{np.array2string(np.array(g), precision=2)}")
        in_sample.append(score(path, CALIBRATION, args.out))

    for label, rows in (("HELD OUT", held), ("in sample", in_sample)):
        m = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
        sd = float(np.std([r["psi"] for r in rows]))
        print(f"\n  {label:<10} mean S {m['S']:.4f}   R2_psi {m['psi']:.4f} +- {sd:.4f}   "
              f"R2_qb {m['qb']:.4f}   1-D {m['lcfs']:.4f}   Cons {m['cons']:.4f}")
    drop = float(np.mean([a["psi"] for a in in_sample]) - np.mean([h["psi"] for h in held]))
    print(f"\n  in-sample bias on R2_psi: {drop:+.4f}. DIII-D's answer to the same question, where "
          f"shots are plentiful,\n  is that a three-shot calibration scores -0.601 +- 0.120 on "
          f"shots it has not seen (solver/how_many_shots.py).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
