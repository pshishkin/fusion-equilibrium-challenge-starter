#!/usr/bin/env python3
"""
C10 — does the poloidal-field form know WHERE the geometry error is, or only how big it is?

    uv run python my_experiments/diagnose_field_form.py --share 0.01

C9 measured that `li` is the scalar furthest above its own label-noise floor: the model errs 3.1x
more than EFIT jitters, against 1.15x for tri_top. So li is where the reachable budget is, and
`target_metric.poloidal_field_form` is a quadratic form written specifically for it —

    dc^T G dc  =  integral |grad d_psi|^2 / R  dR dZ,   which is the squared poloidal-field error,

with the 1/R that makes it a field rather than a flux gradient. **It is defined and called from
nowhere.** params.yaml says "`target_metric.poloidal_field_form` stays: li is exactly <B_p^2> and
deserves that exact form rather than a linearisation", and no code path reaches it.

Before reviving it, two things have to be true, and only the second has ever been checked.

**First, the claim in that comment is loose, and it matters.** li is a normalised <B_p^2>, so an
ERROR in li involves 2*integral B_p . dB_p + integral |dB_p|^2 — and `G` is only the second term.
It is the ENERGY OF THE FIELD ERROR, not the error in the field energy. So `G` is a proxy whose
first-order part is missing, and calling it exact is wrong on the page.

**Second, and this is what the script measures**: a form is only worth putting in the loss if it
discriminates. Any large coefficient error produces a large li error, so a form that merely tracks
`|dc|^2` adds nothing the Parseval loss does not already have. The test is therefore against that
null and not against zero:

    does dc^T G dc rank the frames by their ACTUAL li error better than |dc|^2 does?

Everything is measured on the held-out tail, per frame, with the artifact's own basis and the
residuals `diagnose_frames.py` already wrote. Nothing trains.

Pre-registered: **G must beat the |dc|^2 null by +0.05 of Spearman on li**, the same margin S5's
gate uses. Below that the dead code stays dead and the comment gets corrected instead.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "fusion_scoring"))

from common import CONS_SCALARS  # noqa: E402

from experiments import DEFAULT_LOCAL_DATA_DIR, HF_TRAIN_CONFIG, _as_psirz_stack  # noqa: E402
from my_experiments.baseline_model import (  # noqa: E402
    ARTIFACT,
    ENSEMBLE,
    _predict_targets,
    build_inputs,
    coil_flux,
    features_for_row,
    sorted_shots,
    take_share,
)
from my_experiments.progress import install_timestamps  # noqa: E402
from my_experiments.target_metric import (  # noqa: E402
    grad_shafranov_form,
    jacobian_form,
    metric_form,
    poloidal_field_form,
    scorer_context,
)

FloatArray = npt.NDArray[np.floating]
COSTS = HERE.parent / "results" / "frame_costs_ensemble.csv"


def main() -> int:
    install_timestamps()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--share", type=float, default=0.01)
    ap.add_argument("--probe-frames", type=int, default=300,
                    help="frames the jacobian null is built from, as params.yaml sets it")
    ap.add_argument("--jobs", type=int, default=20)
    # Both measured by diagnose_probe_step.py on this artifact and this tail, so the rebuilt M is
    # the one the fit actually used rather than one assembled from a different operating point.
    ap.add_argument("--delta", type=float, default=0.062539,
                    help="probe step in Wb/rad = jacobian_delta * sqrt(psi_ss_tot / n_pca)")
    ap.add_argument("--psi-ss-tot", type=float, default=177.4,
                    help="the metric's own per-frame denominator")
    ap.add_argument("--model", default=ENSEMBLE)
    ap.add_argument("--costs", type=Path, default=COSTS)
    ap.add_argument("--artifact", type=Path, default=ARTIFACT)
    ap.add_argument("--local-data-dir", type=Path, default=DEFAULT_LOCAL_DATA_DIR)
    ap.add_argument("--config", default=HF_TRAIN_CONFIG)
    args = ap.parse_args()

    art = joblib.load(args.artifact)
    n_pca = int(art["n_pca"])
    plan = art["coil"]
    files = take_share(sorted_shots(args.local_data_dir, args.config, int(art["split_salt"])),
                       args.share, "tail")
    df = pd.read_csv(args.costs)
    if list(dict.fromkeys(df["shot"])) != [p.stem for p in files]:
        raise SystemExit(f"{args.costs.name} covers different shots than --share {args.share}")

    mask = np.load(HERE.parent / "fusion_scoring" / "masks" / "d3d_envelope.npz")
    grid_R, grid_Z = mask["grid_R"], mask["grid_Z"]
    comps = np.asarray(art["pca"].pca.components_, dtype=np.float64).reshape(n_pca, len(grid_Z),
                                                                            len(grid_R))
    G = poloidal_field_form(comps, grid_R, grid_Z)
    Q = grad_shafranov_form(comps, grid_R, grid_Z)
    print(f"{len(files)} held-out shots, {len(df)} frames; G and Q built on {n_pca} components")

    dcs = []
    for path in files:
        row = pd.read_parquet(path).iloc[0]
        psi = _as_psirz_stack(row["efit_psirz"]).astype(np.float64)
        feats = features_for_row(row)
        target = psi - coil_flux(plan, feats).astype(np.float64) if plan["subtract"] else psi
        c_true = np.asarray(art["pca"].transform(target), dtype=np.float64)
        c_pred = _predict_targets(art, args.model,
                                  art["scaler"].transform(build_inputs(plan, feats)))[:, :n_pca]
        dcs.append(c_pred - c_true)
    dc = np.concatenate(dcs)
    if len(dc) != len(df):
        raise ValueError(f"{len(dc)} coefficient errors against {len(df)} cost rows")

    # THE NULL THAT MATTERS. |dc|^2 is Parseval, which is the loss this fork used until 08-13 and
    # is not the loss it uses now: `loss.metric: jacobian` has been default since, worth +0.0097 at
    # production, and M already carries a linearised block for every scalar including li. So a form
    # that beats Parseval has beaten last year's loss. The question is whether it beats THIS one,
    # and answering it needs M rebuilt — the artifact does not store it.
    ctx = scorer_context(grid_R, grid_Z, "DIII-D")
    probe = np.concatenate([_as_psirz_stack(pd.read_parquet(p).iloc[0]["efit_psirz"])
                            for p in files[:8]]).astype(np.float64)
    stride = max(1, len(probe) // args.probe_frames)
    probe = probe[::stride][:args.probe_frames]
    delta = args.delta
    print(f"  rebuilding M from {len(probe)} probe frames at delta {delta:.5f} ...")
    m_cons, _var, ratio, n_used, _ok = jacobian_form(comps, probe, delta, ctx, args.jobs)
    M = metric_form(m_cons, args.psi_ss_tot, None, 0.0)
    print(f"  M from {n_used} usable frames; linear/actual ratios "
          + ", ".join(f"{s} {r:.2f}" for s, r in zip(CONS_SCALARS, ratio, strict=True)))

    preds = {"|dc|^2 (Parseval)": (dc * dc).sum(axis=1),
             "dc^T M dc (the loss)": np.einsum("ij,jk,ik->i", dc, M, dc),
             "dc^T G dc (field)": np.einsum("ij,jk,ik->i", dc, G, dc),
             "dc^T Q dc (GS)": np.einsum("ij,jk,ik->i", dc, Q, dc)}

    print("\n  Spearman of each per-frame predictor against each scalar's SQUARED error:")
    print(f"    {'scalar':<9}" + "".join(f"{k:>22}" for k in preds))
    margins, over_parseval = {}, {}
    for s in CONS_SCALARS:
        err = df[f"res_{s}"].to_numpy(dtype=np.float64) ** 2
        ok = np.isfinite(err)
        rho = {k: float(spearmanr(v[ok], err[ok]).statistic) for k, v in preds.items()}
        margins[s] = rho["dc^T G dc (field)"] - rho["dc^T M dc (the loss)"]
        over_parseval[s] = rho["dc^T G dc (field)"] - rho["|dc|^2 (Parseval)"]
        print(f"    {s:<9}" + "".join(f"{rho[k]:>22.4f}" for k in preds))

    print("\n  margin of the field form, per scalar — over Parseval, and over the LOSS:")
    for s in CONS_SCALARS:
        print(f"    {s:<9} {over_parseval[s]:+.4f}   {margins[s]:+.4f}")

    print(f"\n  the gate is +0.05 on li, and the margin is {margins['li']:+.4f}.")
    if margins["li"] < 0.05:
        print("  Refuted AGAINST THE LOSS THAT SHIPS, whatever it does against Parseval. The "
              "jacobian metric already carries a linearised li block, and the field form ranks "
              "frames by their li error no better than that block does — so adding it would "
              "re-weight directions the metric already weights. The dead code stays dead, and "
              "params.yaml's claim that li 'deserves that exact form' is corrected instead: G is "
              "the energy of the field error, not the error in the field energy, and the "
              "first-order term it is missing is the one that would carry li.")
    else:
        print("  It clears. The next step is NOT a training run: it is to check that the frames G "
              "ranks highest are not the ones C7 showed carry no cost, and to set its weight from "
              "li's own block in M rather than by hand — the 08-13 `field` loss was removed "
              "precisely because a hand-set lambda could not be located.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
