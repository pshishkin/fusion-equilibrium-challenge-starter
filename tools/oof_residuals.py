#!/usr/bin/env python3
"""Per-frame out-of-fold residual table of one K-fold fold, for information gates.

    uv run python tools/oof_residuals.py runs/cv5/fold0 --blend mlp*0.5+seq*0.5

Writes `<fold>/residuals.npz`: for every scored frame the seven scorer-native derived scalars of
the TRUE map and of the predicted blend map, the submitted q95/betaN (true and predicted), the
shot index and the frame index. Derived scalars come from `local_score.build_reference`, the same
extraction the scorer runs, applied once to each map.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import local_score
from cv_oof import decode, parse_blend
from my_experiments import baseline_model as B
from toolkit.parallel import pimap, resolve_jobs

MACHINE = "DIII-D"


def _derive(args: tuple) -> tuple[np.ndarray, np.ndarray]:
    true, pred, R, Z, mask_coarse, mask_f = args
    _, cons_true, _, _ = local_score.build_reference(true, R, Z, mask_coarse, mask_f, MACHINE)
    _, cons_pred, _, _ = local_score.build_reference(pred, R, Z, mask_coarse, mask_f, MACHINE)
    return cons_true, cons_pred


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("fold", type=Path)
    ap.add_argument("--blend", default="mlp*0.5+seq*0.5")
    ap.add_argument("--jobs", type=int, default=24)
    args = ap.parse_args()
    files, payload = decode(args.fold, parse_blend(args.blend), args.jobs)
    _, Y, S, lengths = B._read_shots(files, "reading truth", 1.0, args.jobs)
    import pandas as pd
    row = pd.read_parquet(files[0]).iloc[0]
    R = np.asarray(row["efit_grid_R"], dtype=np.float64)
    Z = np.asarray(row["efit_grid_Z"], dtype=np.float64)
    d = np.load(Path(local_score.__file__).parent / "fusion_scoring" / "masks"
                / local_score.ENVELOPES[MACHINE])
    mask_coarse = d["mask_coarse"].astype(bool)
    mask_f = mask_coarse.astype(np.float64)
    bounds = np.concatenate([[0], np.cumsum(lengths)])
    tasks = []
    for i in range(len(files)):
        true = Y[bounds[i]:bounds[i + 1]].reshape(-1, 65, 65).astype(np.float64)
        pred = payload[f"shot_{i:04d}_psirz"].astype(np.float64)
        if true.shape != pred.shape:
            raise ValueError(f"{files[i].name}: true {true.shape}, predicted {pred.shape}")
        tasks.append((true, pred, R, Z, mask_coarse, mask_f))
    del Y
    out_true, out_pred = [], []
    for ct, cp in pimap(_derive, tasks, resolve_jobs(args.jobs, len(tasks))):
        out_true.append(ct)
        out_pred.append(cp)
    q_pred = np.concatenate([np.column_stack([payload[f"shot_{i:04d}_q95"],
                                              payload[f"shot_{i:04d}_betaN"]])
                             for i in range(len(files))])
    np.savez(args.fold / "residuals.npz",
             files=np.array([f.name for f in files]), lengths=lengths,
             shot=np.repeat(np.arange(len(files)), lengths),
             frame=np.concatenate([np.arange(n) for n in lengths]),
             cons_true=np.concatenate(out_true), cons_pred=np.concatenate(out_pred),
             qb_true=S, qb_pred=q_pred, cons_names=np.array(local_score.CONS_SCALARS))
    print(f"wrote {args.fold / 'residuals.npz'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
