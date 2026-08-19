#!/usr/bin/env python3
"""How many shots does a coil calibration need before it stops chasing the sample?

    uv run python solver/how_many_shots.py --draws 3 --eval 20 --jobs 20

MAST's eleven coil gains are fitted on **three** demo shots, and three MAST decisions taken on
those shots have since been refuted by the leaderboard. The question underneath all of them is not
about any one knob: it is whether a calibration fitted on three shots generalises at all. DIII-D
can answer it, because there the same estimator can be given 1, 3, 6, 12 or 24 shots and scored on
shots it has never seen — 7041 are available.

The estimator is the two-stage physics fit, which is the one that generalises here (the free
19-gain alternation scores 0.698 in sample and 0.304 out of it):

  stage 1  F-coils, with a per-frame plasma filament at the shipped axis AND a per-frame constant
           projected out, over the cells outside the shipped LCFS
  stage 2  ECOILA alone, no constant, F held fixed — because a solenoid's field over this grid is
           nearly degenerate with an offset and something has to carry the absolute level

Each sample size is drawn several times with different shots, so what comes out is a mean AND a
spread. The spread is the answer: a calibration whose held-out score swings by more than the
differences we are trying to measure is one that cannot be trusted at that sample size, however
good its average.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from solver import d3d  # noqa: E402
from solver.greens import CoilBasis, grid_green  # noqa: E402
from solver.gs import Profile  # noqa: E402
from solver.validate_d3d import (  # noqa: E402
    DATA,
    live,
    load,
    pooled_r2,
    solve_all,
    truth_of,
)
from toolkit.progress import install_timestamps  # noqa: E402

FloatArray = npt.NDArray[np.floating]
SIZES = (1, 3, 6, 12, 24)


def two_stage(basis: CoilBasis, shots: list, rows: list) -> FloatArray:
    """The physics fit: F-coils with a constant projected out, then ECOILA with none."""
    cols = list(basis.columns)
    i_e = cols.index("magnetics_ECOILA")
    i_f = [i for i in range(len(cols)) if i != i_e]
    blocks = []
    for s, row in zip(shots, rows, strict=True):
        keep = live(s)
        if not keep.any():
            continue
        n = int(keep.sum())
        outside = d3d.outside_lcfs(s, keep).reshape(n, -1)
        truth = truth_of(s).reshape(n, -1)
        ra = np.asarray(row["efit_r_axis"], dtype=np.float64)[keep]
        za = np.asarray(row["efit_z_axis"], dtype=np.float64)[keep]
        fil = np.stack([grid_green(np.array([[ra[k], za[k]]]), basis.grid_R, basis.grid_Z).ravel()
                        for k in range(n)]) * s.ip[keep][:, None]
        cur = s.currents[keep]
        des = np.stack([cur[:, c][:, None] * basis.maps[c].ravel()[None, :]
                        for c in range(len(cols))], axis=-1)
        blocks.append((des, truth, fil, outside))
    if not blocks:
        raise ValueError("no live frame in this draw")

    def stage(idx: list[int], with_const: bool, fixed: tuple | None = None) -> FloatArray:
        xs, ys = [], []
        for des, truth, fil, outside in blocks:
            for k in range(des.shape[0]):
                w = outside[k]
                a = des[k][w][:, idx]
                y = truth[k][w].copy()
                if fixed is not None:
                    y = y - des[k][w][:, fixed[0]] @ fixed[1]
                nui = [fil[k][w]] + ([np.ones(int(w.sum()))] if with_const else [])
                q, _ = np.linalg.qr(np.column_stack(nui))
                xs.append(a - q @ (q.T @ a))
                ys.append(y - q @ (q.T @ y))
        got, *_ = np.linalg.lstsq(np.concatenate(xs), np.concatenate(ys), rcond=None)
        return np.asarray(got, dtype=np.float64)

    g_f = stage(i_f, True)
    g_e = stage([i_e], False, fixed=(i_f, g_f))
    gains: FloatArray = np.empty(len(cols), dtype=np.float64)
    gains[i_e] = g_e[0]
    for slot, i in enumerate(i_f):
        gains[i] = g_f[slot]
    return gains


def main() -> int:
    install_timestamps()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--draws", type=int, default=3, help="independent shot draws per sample size")
    ap.add_argument("--eval", type=int, default=20, help="held-out shots, the SAME for every draw")
    ap.add_argument("--jobs", type=int, default=20)
    ap.add_argument("--pool", type=int, default=40,
                    help="shots read once and drawn from; they are held in memory, so this is the "
                         "knob that trades breadth of draw against RAM")
    ap.add_argument("--data", type=Path, default=DATA)
    args = ap.parse_args()

    files = sorted(args.data.glob("*.parquet"))
    rng = np.random.default_rng(7)
    order = rng.permutation(len(files))
    eval_paths = [files[i] for i in order[:args.eval]]
    pool_paths = [files[i] for i in order[args.eval:args.eval + args.pool]]
    basis, eval_shots = load(eval_paths)
    # A shot whose current never reaches the threshold contributes no frame, and a pool worker
    # asked to stack an empty list raises rather than returning nothing — drop them here.
    eval_shots = [s for s in eval_shots if live(s).any()]
    eval_truth = [truth_of(s) for s in eval_shots]
    print(f"held-out set: {len(eval_shots)} shots, "
          f"{sum(int(live(s).sum()) for s in eval_shots)} live frames — the same for every draw")
    # Read the pool ONCE and keep only shots that carry a plasma. A shot with no live frame is not
    # a hard draw, it is an empty one — and MAST's three demo shots all carry plasma, so a fair
    # comparison draws from shots that do.
    pool: list[tuple] = []
    for path in pool_paths:
        row = pd.read_parquet(path).iloc[0]
        shot = d3d.read(row, basis.columns)
        if live(shot).any():
            pool.append((shot, row))
    print(f"pool: {len(pool)} shots with a plasma, of {len(pool_paths)} read")
    print(f"\n{'shots':>6}{'draw':>6}{'held-out R2_psi':>18}{'F median':>10}{'ECOILA':>9}")

    profile = Profile(alpha=1.0, gamma=1.0, beta=0.3)
    for n in SIZES:
        got = []
        for d in range(args.draws):
            take = rng.choice(len(pool), size=n, replace=False)
            shots = [pool[i][0] for i in take]
            rows = [pool[i][1] for i in take]
            gains = two_stage(basis, shots, rows)
            pred = solve_all(basis, eval_shots, gains, profile, 40, 0.4, args.jobs)
            r2 = pooled_r2(eval_truth, pred)
            got.append(r2)
            f = [v for c, v in zip(basis.columns, gains, strict=True) if "ECOILA" not in c]
            print(f"{n:>6}{d:>6}{r2:>18.5f}{np.median(f):>10.4f}"
                  f"{gains[basis.columns.index('magnetics_ECOILA')]:>9.1f}", flush=True)
        print(f"{n:>6}{'all':>6}{np.mean(got):>11.5f} +- {np.std(got):.5f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
