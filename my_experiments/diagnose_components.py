#!/usr/bin/env python3
"""Per-component predictability of the PCA coefficients — the free gate on raising `n_pca`.

    uv run python my_experiments/diagnose_components.py \
        --artifact artefacts/lanes/baseline_x_base_s0.joblib

**Why this exists.** `n_pca` upward is on the record as REFUTED — 75 gave -0.0005 and 100 gave
0.0000 in production — and the reason recorded with it is a mechanism rather than a verdict:
components 38-49 arrive with error at or above their own sd, i.e. as noise, and zeroing them
outright moves R2_psi by 0.000005. More global modes cannot help when the model cannot predict the
ones it already has.

**But that was measured at the OLD data scale, and there are now five times as many frames.** More
components need more frames both to ESTIMATE and to FIT, which is exactly the reason on file for
re-asking. This re-measures the profile on the current scale, and it costs one read of shots the
artifact never saw instead of a training lane.

**What decides it.** Per component `j`, the error sd against the truth's own sd. Below 1.0 the model
is beating the mean and the component carries information; at or above 1.0 predicting it is worse
than predicting zero, and every further component is a coordinate the regression will fill with
noise. Where that crossing sits is the whole answer: if it is still near 38 of 50, then 70 and 100
components are 32 and 62 coordinates of noise, and the ceiling they unlock is unreachable by
construction. If five times the data has pushed it outward, the entry reopens on its own terms.

Reported against the SCORER's weighting as well as raw, because a component's contribution to
R2_psi is its error in Wb/rad and not its error relative to its own spread: the leading components
carry orders of magnitude more variance, so a poor relative score on a tiny component costs nothing.
Both columns are needed to say anything — one without the other has been misread before.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import numpy.typing as npt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from my_experiments.baseline_model import (
    ARTIFACT,
    DEFAULT_LOCAL_DATA_DIR,
    ENSEMBLE,
    HF_TRAIN_CONFIG,
    _predict_targets,
    _read_shots,
    build_inputs,
    coil_flux,
    sorted_shots,
)
from toolkit.progress import install_timestamps

FloatArray = npt.NDArray[np.floating]


def main() -> int:
    install_timestamps()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--artifact", type=Path, default=ARTIFACT)
    ap.add_argument("--share", type=float, default=0.03,
                    help="share of shots to measure on, taken from the TAIL of this artifact's "
                         "own salt order so nothing in it was fitted or stopped on (default 0.03)")
    ap.add_argument("--models", nargs="+", default=[ENSEMBLE],
                    help=f"which members to profile (default {ENSEMBLE})")
    ap.add_argument("--jobs", type=int, default=0)
    ap.add_argument("--local-data-dir", type=Path, default=DEFAULT_LOCAL_DATA_DIR)
    ap.add_argument("--config", default=HF_TRAIN_CONFIG)
    args = ap.parse_args()

    art = joblib.load(args.artifact)
    ordered = sorted_shots(args.local_data_dir, args.config, int(art["split_salt"]))
    n = max(1, round(len(ordered) * args.share))
    files = ordered[-n:]
    # The tail, and then CHECKED against the artifact's own lists rather than trusted: a component
    # profile measured on shots the fit had seen would read as predictability that is memory.
    seen = set(art["train_files"]) | set(art["val_files"])
    overlap = {p.name for p in files} & seen
    if overlap:
        raise SystemExit(f"{len(overlap)} of the {len(files)} measured shots were in the fit "
                         f"(e.g. {sorted(overlap)[:3]}) — the profile would be memory, not skill")
    print(f"Profiling on {len(files)} tail shots of salt {art['split_salt']}, "
          f"none of them fitted or stopped on")

    feats, psi, _, _ = _read_shots(files, "reading shots", 1.0, args.jobs)
    plan = art["coil"]
    target = psi.astype(np.float64)
    if plan["subtract"]:
        target = target - coil_flux(plan, feats).astype(np.float64)
    truth = np.asarray(art["pca"].transform(target), dtype=np.float64)
    Xs = np.asarray(art["scaler"].transform(build_inputs(plan, feats)), dtype=np.float64)
    n_pca = int(art["n_pca"])
    print(f"  {len(truth)} frames, {n_pca} components")

    for name in args.models:
        pred = _predict_targets(art, name, Xs)[:, :n_pca]
        err = pred - truth
        # The truth's own spread per component, which for a PCA is the singular value: the model has
        # to beat THIS to be worth having, since predicting the component's mean is free.
        sd = truth.std(axis=0)
        rel = err.std(axis=0) / sd
        # And the same error in Wb/rad, which is what R2_psi actually integrates. The basis is
        # orthonormal, so by Parseval the squared coefficient error IS the squared pixel error.
        ss = (err ** 2).sum(axis=0)
        share = ss / ss.sum()
        print(f"\n  {name}: error sd relative to each component's own sd")
        print(f"  {'component':>12s}  {'rel err sd':>10s}  {'R2_j':>8s}  "
              f"{'share of psi err':>16s}  {'cumulative':>10s}")
        edges = [(0, 4), (4, 10), (10, 20), (20, 30), (30, 38), (38, 50)]
        edges = [(a, b) for a, b in edges if a < n_pca] + \
                ([(50, n_pca)] if n_pca > 50 else [])
        cum = 0.0
        for a, b in edges:
            b = min(b, n_pca)
            block = slice(a, b)
            cum += float(share[block].sum())
            r2 = 1.0 - float((rel[block] ** 2).mean())
            print(f"  {f'{a}-{b - 1}':>12s}  {rel[block].mean():10.3f}  {r2:8.3f}  "
                  f"{share[block].sum():16.4f}  {cum:10.4f}")
        crossing = np.argmax(rel >= 1.0) if (rel >= 1.0).any() else n_pca
        print(f"  -> first component whose error reaches its own sd: {crossing} of {n_pca}"
              f"{' (none does)' if crossing == n_pca else ''}")
        print(f"  -> components past it hold {share[crossing:].sum():.4f} of the psi error "
              f"and {(sd[crossing:] ** 2).sum() / (sd ** 2).sum():.6f} of its variance")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
