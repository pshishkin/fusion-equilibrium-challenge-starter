#!/usr/bin/env python3
"""
D3 — is the Jacobian probe step still the size of the errors the model actually makes?

    uv run python my_experiments/diagnose_probe_step.py --share 0.01

`baseline_model.py` sets the probe step to `jacobian_delta * sqrt(psi_ss_tot / n_pca)`, and
`psi_ss_tot` is the metric's own denominator, built against a single flat mean exactly as `ss_tot`
is in `metrics.py`. Write that out and `jacobian_delta` **is** sqrt(1 - R2_psi) whenever the model's
flux error is spread evenly over the components: the per-frame residual energy is
(1 - R2_psi) * psi_ss_tot, and dividing it equally over `n_pca` orthonormal directions gives a
per-component error of `sqrt((1 - R2_psi) * psi_ss_tot / n_pca)`. So the knob is not a free
hyper-parameter — it is a measurement of the model, and params.yaml says so in its own comment
("it has to sit near the errors the model actually makes ... 0.05 is ~ the model's own relative flux
error at R2_psi = 0.997"). README:425 and :558 both say to revisit it when the scale moves.

It has moved. The value in params.yaml is 0.0332, fitted when R2_psi was 0.9989; R2_psi is now
0.9998, and four accepted changes have landed in between.

**But the equal-spread assumption is the part worth checking rather than asserting**, and that is
what this script measures. It takes the fitted artifact, predicts held-out frames, projects the
ground truth through the artifact's own basis, and reports the per-component residual RMS `r_k`.
Three readings come out of it:

  * `sqrt(mean_k r_k^2)` over all 50 components, which is the equal-spread answer and must agree
    with sqrt(1 - R2_psi) to within the fold — that agreement is the control that says the
    arithmetic above is right rather than merely plausible;
  * the same restricted to the components that carry 95% of the residual energy, which is the
    honest step for a probe meant to describe the directions the error actually lives in;
  * the profile itself, so "one isotropic step for fifty components" can be judged rather than
    assumed.

Nothing here trains and nothing here scores. It decides whether a production run is worth spending.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import numpy.typing as npt
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from experiments import DEFAULT_LOCAL_DATA_DIR, HF_TRAIN_CONFIG, _as_psirz_stack  # noqa: E402
from my_experiments.baseline_model import (  # noqa: E402
    ARTIFACT,
    ENSEMBLE,
    _predict_targets,
    build_inputs,
    check_grid,
    coil_flux,
    features_for_row,
    sorted_shots,
    take_share,
)
from my_experiments.progress import install_timestamps  # noqa: E402

FloatArray = npt.NDArray[np.floating]


def main() -> int:
    install_timestamps()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--share", type=float, default=0.01,
                    help="share of shots to read, from the tail of the list (default 0.01)")
    ap.add_argument("--model", default=ENSEMBLE)
    ap.add_argument("--artifact", type=Path, default=ARTIFACT)
    ap.add_argument("--local-data-dir", type=Path, default=DEFAULT_LOCAL_DATA_DIR)
    ap.add_argument("--config", default=HF_TRAIN_CONFIG)
    args = ap.parse_args()

    art = joblib.load(args.artifact)
    n_pca = int(art["n_pca"])
    plan = art["coil"]
    files = take_share(sorted_shots(args.local_data_dir, args.config, int(art["split_salt"])),
                       args.share, "tail")
    fitted = set(art["train_files"]) | set(art["val_files"])
    if {p.name for p in files} & fitted:
        raise SystemExit("the tail overlaps the fit; lower --share")
    print(f"{len(files)} held-out shots, artifact {args.artifact.name}, model {args.model!r}")

    res_parts: list[FloatArray] = []
    psi_total, psi_sumsq, n_frames = 0.0, 0.0, 0
    for path in files:
        row = pd.read_parquet(path).iloc[0]
        psi = _as_psirz_stack(row["efit_psirz"]).astype(np.float64)
        feats = features_for_row(row)
        if plan["subtract"] or plan["inputs"] != "currents":
            check_grid(plan, row)

        # psi_ss_tot exactly as `train` computes it (baseline_model.py:1101-1103): sums over the
        # flux AS STORED, before any decomposition, against one flat mean. Accumulated over shots
        # so the flat mean is the fold's, not each shot's.
        psi_total += float(psi.sum(dtype=np.float64))
        psi_sumsq += float(np.einsum("ijk,ijk->", psi, psi, dtype=np.float64))
        n_frames += len(psi)

        # Truth and prediction in the SAME coordinates: the basis was fitted on the residual after
        # the coil field, so the truth has to be pushed through the same subtraction. Projecting
        # the raw map would measure a different basis.
        target = psi - coil_flux(plan, feats).astype(np.float64) if plan["subtract"] else psi
        c_true = np.asarray(art["pca"].transform(target), dtype=np.float64)
        c_pred = _predict_targets(art, args.model,
                                  art["scaler"].transform(build_inputs(plan, feats)))[:, :n_pca]
        res_parts.append(c_pred - c_true)

    res = np.concatenate(res_parts)
    psi_ss_tot = (psi_sumsq - psi_total ** 2 / (n_frames * psi.shape[1] * psi.shape[2])) / n_frames
    unit = float(np.sqrt(psi_ss_tot / n_pca))
    r = np.sqrt((res ** 2).mean(axis=0))                 # per-component residual RMS, flux units
    energy = r ** 2
    print(f"\n{len(res)} frames, {n_pca} components. psi_ss_tot {psi_ss_tot:.4g} per frame, "
          f"so one unit of `jacobian_delta` is {unit:.6f} Wb/rad.")

    # From the ARTIFACT's own params, not from params.yaml on disk: the question is what this fit
    # used, and the file has moved since fits older than it.
    import yaml
    configured = float(yaml.safe_load(art["params_yaml"])["loss"]["jacobian_delta"])
    print(f"\n  configured  jacobian_delta = {configured:.4f}  "
          f"(probe step {configured * unit:.6f} Wb/rad)")

    equal = float(np.sqrt(energy.mean())) / unit
    print(f"  equal-spread over all {n_pca}          = {equal:.4f}  "
          f"-> the configured value is {configured / equal:.2f}x this")

    order = np.argsort(energy)[::-1]
    cum = np.cumsum(energy[order]) / energy.sum()
    for q in (0.90, 0.95, 0.99):
        k = int(np.searchsorted(cum, q)) + 1
        step = float(np.sqrt(energy[order[:k]].mean())) / unit
        print(f"  over the {k:2d} components carrying {q:.0%} of the residual energy = {step:.4f}  "
              f"-> configured is {configured / step:.2f}x this")

    # The control the whole argument rests on: if `jacobian_delta` really is sqrt(1 - R2_psi), then
    # the equal-spread number above must equal the flux R2 this fold reports, computed here from
    # the same residual. Anything else means the identity is wrong and the refit is not licensed.
    r2_in_basis = 1.0 - float(energy.sum()) / psi_ss_tot
    print(f"\n  control: R2_psi implied by these coefficients is {r2_in_basis:.6f}, "
          f"and sqrt(1 - R2_psi) = {np.sqrt(max(0.0, 1 - r2_in_basis)):.4f} against the "
          f"equal-spread {equal:.4f} above — these two are the same number by construction, so "
          f"agreement is the check that the arithmetic is right, not a result.")

    print("\n  per-component residual RMS, in units of `jacobian_delta`:")
    for lo in range(0, n_pca, 10):
        block = r[lo:lo + 10] / unit
        print(f"    {lo:2d}-{min(lo + 9, n_pca - 1):2d}: "
              + " ".join(f"{v:7.4f}" for v in block))
    spread = float(r.max() / max(r.min(), 1e-30))
    print(f"\n  largest / smallest per-component step: {spread:.1f}x — one isotropic probe step "
          f"describes every component equally well only if this is near 1.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
