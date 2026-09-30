#!/usr/bin/env python3
"""Gate (Astra, round 2): is the tail of the Consistency error a REPRESENTATION limit?

Fold-0 shots are split, from the cached 11-member map errors (runs/consgate_fc11), into the
worst 5% by the summed normalised SSE of li, Z_axis and volume, and 150 random others. Their TRUE
maps are projected on a basis (plasma part, coil field added back) and reconstructed; the seven
scorer-derived scalars of the reconstruction are compared with those of the truth. Reported, per
scalar and group: the reconstruction's SSE as a share of the stacked prediction's SSE. A share
near 1 means the basis itself cannot hold these maps; near 0 means it can and the error is the
regression's.
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cons_residual_gate import _scalars_task, geometry, true_scalars
from cv_oof import _fold_files
from my_experiments import baseline_model as B
from toolkit.parallel import pmap

NAMES = ["R_axis", "Z_axis", "kappa", "tri_top", "tri_bot", "volume", "li"]


def main() -> int:
    art0 = joblib.load("runs/cv5_fixcos/fold0/baseline.joblib")
    files = _fold_files(art0)
    z = np.load("runs/consgate_fc11/scalars_fold0.npz"); P, T = z["pred"], z["true"]
    lens = np.load("runs/cv5_fixcos/fold0/oof.npz")["lengths"]; b = np.concatenate([[0], np.cumsum(lens)])
    var = np.nanvar(T, 0)
    e = np.where(np.isfinite(P) & np.isfinite(T), (T - P) ** 2, 0.0) / var
    per = np.add.reduceat(e[:, [1, 5, 6]].sum(1), b[:-1])
    worst = np.argsort(per)[::-1][: len(lens) // 20]
    rng = np.random.default_rng(1)
    rand = rng.choice(np.setdiff1d(np.arange(len(lens)), worst), 150, replace=False)
    geo = geometry(files)
    bases = {"pca50 (fold0 members)": art0, "pca70 (stack basis)": joblib.load("runs/fc_p70/fold0/baseline.joblib")}
    for group, pick in (("tail 5%", worst), ("random", rand)):
        pick = np.sort(pick)
        X, Y, _, lp = B._read_shots([files[i] for i in pick], "reading", 1.0, 24)
        rows = np.concatenate([np.arange(b[i], b[i + 1]) for i in pick])
        Tg, Pg = T[rows], P[rows]
        bb = np.concatenate([[0], np.cumsum(lp)])
        for label, art in bases.items():
            coil = B.coil_flux(art["coil"], X)
            plasma = (Y - coil).reshape(len(Y), -1)
            rec = art["pca"].inverse_transform(art["pca"].transform(plasma)).reshape(-1, 65, 65) + coil
            tasks = [(rec[bb[i]:bb[i + 1]].astype(np.float32), *geo) for i in range(len(pick))]
            R = np.concatenate(pmap(_scalars_task, tasks, 30))
            ok = np.isfinite(R) & np.isfinite(Tg) & np.isfinite(Pg)
            share = [((Tg[ok[:, j], j] - R[ok[:, j], j]) ** 2).sum() / ((Tg[ok[:, j], j] - Pg[ok[:, j], j]) ** 2).sum()
                     for j in range(7)]
            print(f"{group:>8} | {label:>22} | recon SSE / stacked SSE: " +
                  "  ".join(f"{n} {s:.2f}" for n, s in zip(NAMES, share)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
