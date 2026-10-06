#!/usr/bin/env python3
"""Where does the stack lose R_axis, Z_axis, volume and li? By discharge phase and by beta_N.

    uv run python tools/error_regimes.py --cache runs/consgate_fc11 --out runs/error_regimes

The public leaders beat us mostly on R_axis, Z_axis, volume and li (2026-10-06), i.e. on the
interior geometry, which pressure (Shafranov shift) and the current profile set. If our error sits
on the flat-top at high beta, the inputs lack pressure information (only 12 Thomson summaries go
in); if it sits in the current ramps, it is the vessel / current-diffusion dynamics.

Per frame, from the cached scorer scalars of a cross-fold linear stack (`cons_residual_gate`):
phase (ramp-up before |Ip| first reaches 90% of its maximum, flat-top while above, ramp-down
after it last is), beta_N tertile (EFIT truth, a label: fine for a diagnosis), and the share of
each scalar's SSE against the share of frames.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cv_oof import _fold_files
from my_experiments import baseline_model as B
from toolkit.parallel import pmap

NAMES = ["R_axis", "Z_axis", "kappa", "tri_top", "tri_bot", "volume", "li"]
FOCUS = ["R_axis", "Z_axis", "volume", "li", "kappa"]


def shot_regimes(path: Path) -> np.ndarray:
    """(T, 3): phase code (0 ramp-up, 1 flat-top, 2 ramp-down), beta_N, |Ip| / max|Ip|."""
    r = pd.read_parquet(path, columns=["efit_times", "efit_beta_n", "magnetics_plasma_current",
                                       "magnetics_plasma_current_times", "magnetics_time"]).iloc[0]
    et = np.asarray(r["efit_times"], dtype=np.float64)
    ip = np.asarray(r["magnetics_plasma_current"], dtype=np.float64)
    it = np.asarray(B.align_ip_times(et, np.asarray(r["magnetics_plasma_current_times"], dtype=np.float64),
                                     ip, np.asarray(r["magnetics_time"], dtype=np.float64)), dtype=np.float64)
    a = np.abs(np.interp(et, it, ip))
    rel = a / max(a.max(), 1e-9)
    hi = np.flatnonzero(rel >= 0.9)
    phase = np.ones(len(et))
    if len(hi):
        phase[: hi[0]] = 0
        phase[hi[-1] + 1:] = 2
    return np.column_stack([phase, np.asarray(r["efit_beta_n"], dtype=np.float64), rel])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", type=Path, default=Path("runs/consgate_fc11"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--jobs", type=int, default=20)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    reg, err, tru = [], [], []
    for k in range(5):
        files = _fold_files(joblib.load(f"runs/cv5_fixcos/fold{k}/baseline.joblib"))
        z = np.load(args.cache / f"scalars_fold{k}.npz")
        g = np.concatenate(pmap(shot_regimes, files, args.jobs))
        if len(g) != len(z["pred"]):
            raise ValueError(f"fold {k}: {len(g)} regime rows for {len(z['pred'])} frames")
        reg.append(g)
        err.append(z["true"] - z["pred"])
        tru.append(z["true"])
        print(f"fold {k}: {len(g)} frames", flush=True)
    G, E, T = (np.concatenate(a) for a in (reg, err, tru))
    ok = np.isfinite(E).all(1) & np.isfinite(G).all(1)
    G, E, T = G[ok], E[ok], T[ok]
    sse = (E ** 2).sum(0)
    sst = ((T - T.mean(0)) ** 2).sum(0)
    report: dict = {"frames": int(len(E)), "r2_cons": dict(zip(NAMES, (1 - sse / sst).tolist()))}
    bt = np.quantile(G[:, 1], [1 / 3, 2 / 3])
    groups = {
        "ramp-up": G[:, 0] == 0, "flat-top": G[:, 0] == 1, "ramp-down": G[:, 0] == 2,
        "betaN low": G[:, 1] < bt[0], "betaN mid": (G[:, 1] >= bt[0]) & (G[:, 1] < bt[1]),
        "betaN high": G[:, 1] >= bt[1],
        "flat-top, betaN high": (G[:, 0] == 1) & (G[:, 1] >= bt[1]),
        "flat-top, betaN low": (G[:, 0] == 1) & (G[:, 1] < bt[0]),
    }
    print(f"\nbeta_N tertile edges {bt[0]:.2f} / {bt[1]:.2f}; R2_cons " + " ".join(
        f"{n} {v:.4f}" for n, v in report["r2_cons"].items()))
    print(f"{'group':24s} {'frames':>7s} " + " ".join(f"{n:>8s}" for n in FOCUS) + "   (SSE share / frame share)")
    rows = {}
    for name, m in groups.items():
        fs = m.mean()
        shares = {n: float((E[m, NAMES.index(n)] ** 2).sum() / sse[NAMES.index(n)]) for n in FOCUS}
        rows[name] = {"frame_share": float(fs), "sse_share": shares}
        print(f"{name:24s} {fs:7.3f} " + " ".join(f"{shares[n] / fs:8.2f}" for n in FOCUS))
    report["groups"] = rows
    (args.out / "report.json").write_text(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
