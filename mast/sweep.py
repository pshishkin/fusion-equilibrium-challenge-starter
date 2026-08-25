#!/usr/bin/env python3
"""Score calibration variants against the real composite, reusing one reference pass.

`bench.py` made SCALAR models cheap by caching the solve. This does the same for anything that
changes the solve itself -- a coil tilt, a profile constant, a boundary knob -- where the flux map
moves and so `bench`'s cache is invalid. What it caches instead is the other expensive half: LCFS
extraction and the seven functionals on the GROUND TRUTH, which cost ~15 s and do not depend on
the candidate at all.

    uv run python mast/sweep.py --updown -0.02 0.0 0.02

Each variant then costs only its own solve. The number printed is `local_score.py`'s composite,
computed by the vendored scorer through the same `finalize_machine` -- not a proxy for it.

**Read the composite, never one term.** The leave-one-out ensemble bought R2_qb 0.183 -> 0.197 on
the board and paid D_LCFS 0.472 -> 0.487 for it; a sweep that watched either term alone would have
drawn the opposite conclusion from the one the board drew.
"""
from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "fusion_scoring"))

from common import N_CONS, N_SCALARS, SCALARS  # noqa: E402
from metrics import Accum, finalize_machine  # noqa: E402

import local_score as ls  # noqa: E402
from mast.calibration import SHIPPED  # noqa: E402
from mast.predict import predict_row  # noqa: E402

DEMO = HERE.parent / "parquet_data"


class Fold:
    """The demo shots with their reference targets built once."""

    def __init__(self, paths: list[Path], jobs: int = 3) -> None:
        import pandas as pd
        self.shots = [ls._shot_from_row(pd.read_parquet(p).iloc[0]) for p in paths]
        self.machine, self.R, self.Z, self.mask, self.mask_f = ls._machine_of(self.shots)
        self.jobs = jobs
        self.refs = ls._map(ls._ref_task,
                            [(s["psi"], self.R, self.Z, self.mask, self.mask_f, self.machine)
                             for s in self.shots], jobs, "  references")
        psi_sum = psi_n = 0.0
        scal_sum, scal_sumsq, scal_n = (np.zeros(N_SCALARS) for _ in range(3))
        cons_sum, cons_sumsq, cons_n = (np.zeros(N_CONS) for _ in range(3))
        psi_sumsq = 0.0
        for s, ref in zip(self.shots, self.refs, strict=True):
            g = s["psi"].astype(np.float64)
            fin = np.isfinite(g)
            psi_sum += float(g[fin].sum())
            psi_sumsq += float((g[fin] ** 2).sum())
            psi_n += int(fin.sum())
            for j, name in enumerate(SCALARS):
                v = np.asarray(s[name], dtype=np.float64)
                v = v[np.isfinite(v)]
                scal_sum[j] += v.sum()
                scal_sumsq[j] += (v ** 2).sum()
                scal_n[j] += v.size
            cons_gt, cmask = ref[1], ref[2]
            for j in range(N_CONS):
                v = cons_gt[cmask[:, j], j]
                cons_sum[j] += v.sum()
                cons_sumsq[j] += (v ** 2).sum()
                cons_n[j] += v.size
        self.ref_stats = {"psi_sum": psi_sum, "psi_sumsq": psi_sumsq, "psi_n": psi_n,
                          "scal_sum": scal_sum, "scal_sumsq": scal_sumsq, "scal_n": scal_n,
                          "cons_sum": cons_sum, "cons_sumsq": cons_sumsq, "cons_n": cons_n}
        mean_psi = psi_sum / psi_n if psi_n else 0.0
        self.means = (mean_psi,
                      np.where(scal_n > 0, scal_sum / np.where(scal_n > 0, scal_n, 1), 0.0),
                      np.where(cons_n > 0, cons_sum / np.where(cons_n > 0, cons_n, 1), 0.0))

    def score(self, predict: Callable[[object], dict]) -> dict:
        preds = [predict(s["row"]) for s in self.shots]
        sign = ls.choose_psi_sign(self.shots, preds, self.means[0])
        tasks = [({"psi": s["psi"], "q95": s["q95"], "betaN": s["betaN"]}, ref, p,
                  self.R, self.Z, self.mask, self.mask_f, sign, self.means, self.machine)
                 for s, ref, p in zip(self.shots, self.refs, preds, strict=True)]
        acc = Accum()
        acc.psi_sign = sign
        for part in ls._map(ls._score_task, tasks, self.jobs, "  scoring"):
            acc.add(part)
        return finalize_machine(acc, self.ref_stats)


def line(label: str, res: dict) -> str:
    out = (f"{label:<30}S {res['S']:.5f}  psi {res['r2_psi']:.5f}  qb {res['r2_qb']:+.4f}  "
           f"dlcfs {res['dlcfs']:.4f}  cons {res['consistency']:.5f}")
    each = res.get("r2_cons_each") or {}
    if each:
        out += "  [" + " ".join(f"{k}{v:+.2f}" if v is not None else f"{k}--"
                                for k, v in each.items()) + "]"
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--updown", type=float, nargs="+", default=[0.0],
                    help="up/down coil-gain asymmetry values to try")
    ap.add_argument("--edge", type=float, nargs="+", default=None,
                    help="profile `edge` values to try (default: leave the shipped one alone)")
    ap.add_argument("--per-shot", action="store_true",
                    help="score each demo shot as its own fold. A knob fitted on three shots is "
                         "only worth shipping if all three want it in the same direction.")
    ap.add_argument("--jobs", type=int, default=3)
    a = ap.parse_args(argv)
    paths = sorted(DEMO.glob("mast_shot_*.parquet"))
    folds = ([( p.stem[-5:], Fold([p], a.jobs)) for p in paths] if a.per_shot
             else [("all", Fold(paths, a.jobs))])
    grid = [(u, e) for u in a.updown for e in (a.edge if a.edge is not None else [None])]
    for name, fold in folds:
        print(f"\n--- fold {name} ---")
        base = None
        for u, e in grid:
            cals = tuple(replace(c, updown=u,
                                 profile=(c.profile if e is None
                                          else replace(c.profile, edge=e)))
                         for c in SHIPPED)

            def go(row: object, c: tuple = cals) -> dict:
                return predict_row(row, c)

            res = fold.score(go)
            if base is None:
                base = res["S"]
            label = f"updown={u:+.4f}" + ("" if e is None else f" edge={e:.3f}")
            print(line(label, res) + f"   dS {res['S']-base:+.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
