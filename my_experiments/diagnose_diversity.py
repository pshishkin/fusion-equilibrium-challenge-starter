#!/usr/bin/env python3
"""Do the arms err in DIFFERENT PLACES? The measurement the whole shot-mixer case rests on.

    uv run python my_experiments/diagnose_diversity.py \
        --artifacts artefacts/lanes/baseline_y_seq_*_s0.joblib

`seq` is not in the ensemble because it is accurate — it LOSES to the MLP, 0.9914 against 0.9931.
It is there because it disagrees, and that is measured: `mlp+seq` = 0.9943 against 0.9940 for `mlp`
plus three extra seeds, so one extra ARCHITECTURE beats three extra seeds, and 0.0024 of the whole
score is member disagreement alone.

Every arm of ladder rung 1 was nonetheless read by its own composite, which is the wrong number for
the criterion. An arm that scores +0.0007 and errs where the control errs adds nothing to a blend;
one that scores level and errs elsewhere may add more than a fourth MLP seed. This computes the
thing itself — the correlation between two arms' ERROR VECTORS on the same held-out frames — and it
costs no training at all, because every arm's artifact is already on disk.

Reported in the PSI block, where the basis is orthonormal so a coefficient error is exactly a pixel
error by Parseval, and separately in the q95/betaN block, which is where rung 1 found the mixers
actually moved.

Refuses to compare artifacts whose PCA components differ: two errors expressed in different bases
are not the same vector, and correlating them would produce a number that looks fine and means
nothing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from my_experiments.baseline_model import (
    DEFAULT_LOCAL_DATA_DIR,
    HF_TRAIN_CONFIG,
    _predict_targets,
    _read_shots,
    build_inputs,
    coil_flux,
    sorted_shots,
)
from toolkit.progress import install_timestamps


def main() -> int:
    install_timestamps()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--artifacts", nargs="+", type=Path, required=True)
    ap.add_argument("--share", type=float, default=0.05)
    ap.add_argument("--model", default="ensemble")
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--local-data-dir", type=Path, default=DEFAULT_LOCAL_DATA_DIR)
    ap.add_argument("--config", default=HF_TRAIN_CONFIG)
    args = ap.parse_args()

    arts = {p.stem.removeprefix("baseline_"): joblib.load(p) for p in args.artifacts}
    names = list(arts)
    first = arts[names[0]]
    salt = int(first["split_salt"])
    if any(int(a["split_salt"]) != salt for a in arts.values()):
        raise SystemExit("the artifacts were fitted under different salts, so their held-out "
                         "shots differ and their errors are not on the same frames")
    ref = np.asarray(first["pca"].pca.components_)
    for n, a in arts.items():
        d = float(np.abs(np.asarray(a["pca"].pca.components_) - ref).max())
        if d > 1e-9:
            raise SystemExit(f"{n}'s PCA basis differs from {names[0]}'s by {d:.2e}; two errors in "
                             f"different bases are not the same vector")

    ordered = sorted_shots(args.local_data_dir, args.config, salt)
    files = ordered[-max(1, round(len(ordered) * args.share)):]
    seen = set(first["train_files"]) | set(first["val_files"])
    if {p.name for p in files} & seen:
        raise SystemExit("the measured shots overlap the fit")
    print(f"{len(files)} tail shots of salt {salt}, {len(names)} arms, basis identical across all")

    feats, psi, _, lengths = _read_shots(files, "reading shots", 1.0, args.jobs)
    plan = first["coil"]
    target = psi.astype(np.float64)
    if plan["subtract"]:
        target = target - coil_flux(plan, feats).astype(np.float64)
    truth = np.asarray(first["pca"].transform(target), dtype=np.float64)
    n_pca = int(first["n_pca"])

    # SHOT BY SHOT, because `seq` refuses a concatenated corpus rather than reading it as one long
    # sequence — the error it raises is the one thing standing between this diagnostic and a
    # silently wrong number, so it is obeyed rather than worked around.
    bounds = np.cumsum(np.concatenate([[0], np.asarray(lengths, dtype=np.int64)]))
    err = {}
    for n, a in arts.items():
        xs = np.asarray(a["scaler"].transform(build_inputs(a["coil"], feats)), dtype=np.float64)
        pred = np.empty((len(xs), n_pca), dtype=np.float64)
        for lo, hi in zip(bounds[:-1], bounds[1:]):
            pred[lo:hi] = _predict_targets(a, args.model, xs[lo:hi])[:, :n_pca]
        err[n] = pred - truth
        print(f"  {n}: rms coefficient error {float(np.sqrt((err[n] ** 2).mean())):.5f}",
              flush=True)

    for label, block in (("psi", slice(0, n_pca)),):
        e = {n: v[:, block].reshape(-1) for n, v in err.items()}
        e = {n: v - v.mean() for n, v in e.items()}
        print(f"\n  error correlation, {label} block "
              f"(1.00 = errs in exactly the same places as the control)")
        print(f"  {'arm':>8s} " + " ".join(f"{n.removeprefix('y_seq_').removesuffix('_s0'):>7s}"
                                           for n in names))
        for a_ in names:
            row = [float(e[a_] @ e[b_] / (np.linalg.norm(e[a_]) * np.linalg.norm(e[b_])))
                   for b_ in names]
            print(f"  {a_.removeprefix('y_seq_').removesuffix('_s0'):>8s} "
                  + " ".join(f"{v:7.3f}" for v in row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
