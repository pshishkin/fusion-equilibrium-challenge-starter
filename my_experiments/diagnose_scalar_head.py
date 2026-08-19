#!/usr/bin/env python3
"""
D1 — can the seven scored functionals be estimated better than by reading them off the map?

    uv run python my_experiments/diagnose_scalar_head.py --jobs 20

Consistency is 0.0034 of the 0.0045 this pipeline has left, and the scorer does not compute it from
the flux: it extracts a contour from the submitted map and derives seven numbers from that. So there
are two ways to be right about kappa. The pipeline uses one of them — get the map right and let
`derive_frame` read it — and has never measured the other, which is to predict the seven numbers
directly from the same inputs.

Six independent readings of this file arrived at the same object from four directions (predict and
transport, the ensemble's Jensen gap, averaging the functionals, combining in functional space), and
they are all the same correction: undoing the inward bias of an averaged contour's extreme points.
This script measures the one number they all depend on, which is whether a direct head is any better
than `f_j(psi_ens)` at all. **Nothing here transports anything back into a flux map.** If the head
cannot beat the readout on held-out shots, the transport question never arises — and if it can, the
size measured here is the CEILING on what any transport could realise, before the linearisation
takes its half.

Three controls, because without them the number is not attributable:

* **the affine-only arm**, `a + b * f_j(psi_ens)` fitted on the same held-out half. This is A7/it-3
  wearing a hat — a perfect affine correction of the seven scalars was measured at +0.0004 in 08-13
  — and if the blend cannot beat it, what the blend found is a bias and not information;
* **the Jensen arm**, the mean over the four seeds of `f_j(psi_m)` as an extra column. The
  functionals are nonlinear, so the mean of the members' scalars is not the scalars of the mean
  map, and that gap is free to exploit at inference. Built from the per-member tables on disk;
* **fitted on one half of the tail shots and read on the other**, split at a shot boundary. Fitting
  and reading on the same frames is how a +0.0004 in-sample number gets mistaken for a result.

The head is the PRODUCTION MLP — the same architecture, schedule and stopping rule, built from the
artifact's own `params_yaml` — fitted on the same 208 features to the seven scalars, with the
ground-truth values from `aux_targets`' cache so no derive pass is repeated.

Pre-registered kill: the blend must reach **+0.0065** of the pooled mean R2 over the seven scalars,
on the unseen half. Below that it is under the 0.0013 of S a paired run resolves once the
linearisation has taken its cut, and no transport is written.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import numpy.typing as npt
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "fusion_scoring"))

from common import CONS_SCALARS, N_CONS  # noqa: E402

from experiments import DEFAULT_LOCAL_DATA_DIR, HF_TRAIN_CONFIG  # noqa: E402
from my_experiments.baseline_model import (  # noqa: E402
    ARTIFACT,
    _read_shots,
    build_inputs,
    sorted_shots,
    take_share,
)
from my_experiments.models import _build_model  # noqa: E402
from toolkit.progress import install_timestamps  # noqa: E402

FloatArray = npt.NDArray[np.floating]
COSTS = HERE.parent / "results" / "frame_costs_ensemble.csv"
MEMBERS = ("mlp", "mlp1", "mlp2", "mlp3")


def features(files: list[Path], art: dict, desc: str, jobs: int, block: int = 400) -> FloatArray:
    """The 208 model inputs for every frame, read in blocks so the flux never all lives at once.

    `_read_shots` returns psi beside the features and psi is 500x the size of what this needs; at
    the full 7041 shots that is the difference between 3 GB and the 90 GB that OOMed a run twice.
    """
    out = []
    for lo in range(0, len(files), block):
        feats, _psi, _s, _n = _read_shots(files[lo:lo + block],
                                          f"{desc} {lo // block + 1}", 1.0, jobs)
        out.append(np.asarray(art["scaler"].transform(build_inputs(art["coil"], feats)),
                              dtype=np.float32))
        del feats, _psi
    return np.concatenate(out)


def cons_targets(files: list[Path], n_rows: int, label: str) -> FloatArray:
    """The seven ground-truth scalars per frame, from the aux_targets cache, mean-imputed."""
    from my_experiments import aux_targets
    cons = aux_targets.load(files)
    if cons is None:
        raise SystemExit(f"the {label} functionals are not cached for this split; build them with "
                         f"my_experiments/aux_targets.py")
    if len(cons) != n_rows:
        raise ValueError(f"{label} functionals hold {len(cons)} frames against {n_rows} rows of "
                         f"features — the cache is for a different split")
    a = cons.astype(np.float64)
    for j in range(a.shape[1]):
        bad = ~np.isfinite(a[:, j])
        if bad.any():
            a[bad, j] = np.nanmean(a[:, j])
    return a


def r2_pooled(truth: FloatArray, est: FloatArray, ok: npt.NDArray[np.bool_]) -> FloatArray:
    """Per scalar, 1 - SS_res/SS_tot against the fold mean — the form `metrics.py` uses."""
    out = np.full(N_CONS, np.nan)
    for j in range(N_CONS):
        m = ok[:, j]
        t, e = truth[m, j], est[m, j]
        tot = float(((t - t.mean()) ** 2).sum())
        out[j] = 1.0 - float(((t - e) ** 2).sum()) / tot if tot > 0 else np.nan
    return out


def main() -> int:
    install_timestamps()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--share", type=float, default=0.01, help="tail share, matching frame_costs")
    ap.add_argument("--jobs", type=int, default=20)
    ap.add_argument("--costs", type=Path, default=COSTS)
    ap.add_argument("--artifact", type=Path, default=ARTIFACT)
    ap.add_argument("--local-data-dir", type=Path, default=DEFAULT_LOCAL_DATA_DIR)
    ap.add_argument("--config", default=HF_TRAIN_CONFIG)
    args = ap.parse_args()

    art = joblib.load(args.artifact)
    ordered = sorted_shots(args.local_data_dir, args.config, int(art["split_salt"]))
    # The artifact stores bare NAMES; the shots live wherever `sorted_shots` finds them, which is a
    # download directory and not `--local-data-dir` itself. Resolve through the ordered list so the
    # aux_targets cache key — built from the same names in the same order — still matches.
    where = {p.name: p for p in ordered}
    train_files = [where[n] for n in art["train_files"]]
    val_files = [where[n] for n in art["val_files"]]
    tail_files = take_share(ordered, args.share, "tail")
    print(f"head fitted on {len(train_files)} shots, stopped on {len(val_files)}, "
          f"read on the {len(tail_files)} tail shots")

    df = pd.read_csv(args.costs)
    missing = [c for c in (f"gt_{s}" for s in CONS_SCALARS) if c not in df.columns]
    if missing:
        raise SystemExit(f"{args.costs.name} has no {missing[0]} column — regenerate it with the "
                         f"current diagnose_frames.py, which writes the ground-truth scalars")
    if list(dict.fromkeys(df["shot"])) != [p.stem for p in tail_files]:
        raise SystemExit(f"{args.costs.name} covers different shots than --share {args.share}")

    Xt = features(train_files, art, "train", args.jobs)
    Yt = cons_targets(train_files, len(Xt), "training")
    Xv = features(val_files, art, "val", args.jobs)
    Yv = cons_targets(val_files, len(Xv), "validation")
    centre, scale = Yt.mean(axis=0), Yt.std(axis=0)
    print(f"\n  {len(Xt)} training rows, {Xt.shape[1]} features -> {N_CONS} scalars")

    cfg = dict(yaml.safe_load(art["params_yaml"])["models"]["mlp"])
    cfg.pop("enabled", None)
    head = _build_model("head", cfg, args.artifact)
    head.fit(Xt.astype(np.float64), (Yt - centre) / scale,
             Xv.astype(np.float64), (Yv - centre) / scale)
    print(f"  head: {head.fit_report()}")
    del Xt, Yt, Xv, Yv

    Xe = features(tail_files, art, "tail", args.jobs)
    if len(Xe) != len(df):
        raise ValueError(f"{len(Xe)} tail frames against {len(df)} rows of {args.costs.name}")
    f_head = np.asarray(head.predict(Xe.astype(np.float64)))[:, :N_CONS] * scale + centre

    truth = df[[f"gt_{s}" for s in CONS_SCALARS]].to_numpy(dtype=np.float64)
    f_ens = truth - df[[f"res_{s}" for s in CONS_SCALARS]].to_numpy(dtype=np.float64)
    # The Jensen column: the mean over the four seeds of each member's own derived scalars, which
    # is NOT the scalars of the averaged map because the functionals are nonlinear. Built from the
    # per-member tables already on disk — same shots, same frames, same ground truth, so each
    # member's f_j is its own gt minus its own residual.
    jensen, have = np.zeros_like(f_ens), 0
    for name in MEMBERS:
        p = args.costs.parent / f"frame_costs_{name}.csv"
        if not p.exists():
            continue
        d = pd.read_csv(p)
        if len(d) != len(df) or not (d["shot"].to_numpy() == df["shot"].to_numpy()).all():
            print(f"  skipping {p.name}: it covers different frames")
            continue
        jensen += truth - d[[f"res_{s}" for s in CONS_SCALARS]].to_numpy(dtype=np.float64)
        have += 1
    if have:
        jensen /= have
        print(f"  Jensen column built from {have} members")

    ok = np.isfinite(truth) & np.isfinite(f_ens) & np.isfinite(f_head)
    if have:
        ok &= np.isfinite(jensen)
    shots = df["shot"].to_numpy()
    order = list(dict.fromkeys(shots))
    A = np.isin(shots, order[:len(order) // 2])
    B = ~A
    print(f"  {int(A.sum())} frames to fit the blend, {int(B.sum())} to read it on, "
          f"{len(order)} shots split in half")

    # Per scalar, a small least squares on the fitting half and applied to the reading half. The
    # arms are nested, so each line prices exactly what its extra column adds.
    arms: dict[str, list[FloatArray]] = {"affine on the readout": [f_ens],
                                         "readout + head": [f_ens, f_head]}
    if have:
        arms["readout + Jensen"] = [f_ens, jensen]
        arms["readout + head + Jensen"] = [f_ens, f_head, jensen]
    arms["head alone"] = [f_head]

    base = np.nanmean(r2_pooled(truth, f_ens, ok & B[:, None]))
    print("\n  pooled mean R2 over the seven scalars, on the half nothing was fitted on:")
    print(f"    {'readout f(psi_ens)':<26} {base:.5f}   (the baseline)")
    for name, cols in arms.items():
        est = np.empty_like(f_ens)
        for j in range(N_CONS):
            m = ok[:, j] & A
            design = np.column_stack([c[:, j] for c in cols] + [np.ones(len(f_ens))])
            beta, *_ = np.linalg.lstsq(design[m], truth[m, j], rcond=None)
            est[:, j] = design @ beta
        got = np.nanmean(r2_pooled(truth, est, ok & B[:, None]))
        print(f"    {name:<26} {got:.5f}   {got - base:+.5f}")

    print("\n  per scalar, the readout against the full blend:")
    full = arms.get("readout + head + Jensen", arms["readout + head"])
    est = np.empty_like(f_ens)
    for j in range(N_CONS):
        m = ok[:, j] & A
        design = np.column_stack([c[:, j] for c in full] + [np.ones(len(f_ens))])
        beta, *_ = np.linalg.lstsq(design[m], truth[m, j], rcond=None)
        est[:, j] = design @ beta
    read = ok & B[:, None]
    r_base, r_full = r2_pooled(truth, f_ens, read), r2_pooled(truth, est, read)
    for j, s in enumerate(CONS_SCALARS):
        print(f"    {s:<9} {r_base[j]:.5f} -> {r_full[j]:.5f}   {r_full[j] - r_base[j]:+.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
