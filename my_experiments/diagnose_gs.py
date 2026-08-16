#!/usr/bin/env python3
"""
B7 — does imposing "no toroidal current where there is no plasma" at inference buy anything?

    uv run python my_experiments/diagnose_gs.py --share 0.01 --jobs 8

Grad-Shafranov says `Delta* psi = -mu0 R j_phi`, so wherever there is no plasma current the flux
must satisfy `Delta* psi = 0`. The model never had that imposed on it: the loss charges for flux
error and for the linearised functionals, and nothing tells it the vacuum region is vacuum.

`grad_shafranov_form` already builds the quadratic form `Q` with `dc^T Q dc` equal to the squared
toroidal-current error the coefficient error implies. Damping along it is one linear solve per
frame, no retraining and no new fit:

    c'  =  (I + lam * Q_hat)^-1 c        Q_hat = Q / (trace(Q) / n)

`lam` is swept, and `lam = 0` is the model untouched, so the sweep contains its own control.

THE MASK IS THE WHOLE QUESTION, and this runs the assumption-free version of it. Penalising
`Delta*` OUTSIDE THE VESSEL ENVELOPE needs nothing from the prediction — there is certainly no
plasma current there — where the sharper version, "outside the predicted boundary", needs the
boundary extracted first and so is a different and much more expensive experiment. ideas.md says
the per-frame version is worth building only if the fixed mask shows a sign, and this is that test.

The prior is weak and worth stating before the numbers: C2 measured R2_psi at 0.9998 against a
ceiling of 1.0000, so the maps are already essentially as clean as the basis allows, and there may
simply be no vacuum-region error left to remove.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import numpy.typing as npt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import local_score  # noqa: E402
from experiments import DEFAULT_LOCAL_DATA_DIR, EFIT_GRID_SIZE, HF_TRAIN_CONFIG  # noqa: E402
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
from my_experiments.target_metric import grad_shafranov_form  # noqa: E402

FloatArray = npt.NDArray[np.floating]


def main() -> int:
    install_timestamps()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--share", type=float, default=0.01)
    ap.add_argument("--model", default=ENSEMBLE)
    ap.add_argument("--lambdas", nargs="+", type=float,
                    default=[0.0, 0.01, 0.03, 0.1, 0.3, 1.0])
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--local-data-dir", type=Path, default=DEFAULT_LOCAL_DATA_DIR)
    ap.add_argument("--config", default=HF_TRAIN_CONFIG)
    ap.add_argument("--work", type=Path, default=Path("/tmp/gs_sweep"))
    args = ap.parse_args()

    art = joblib.load(ARTIFACT)
    plan = art["coil"]
    n_pca = int(art["n_pca"])
    files = take_share(sorted_shots(args.local_data_dir, args.config, int(art["split_salt"])),
                       args.share, "tail")
    seen = set(art["train_files"]) | set(art["val_files"])
    if {p.name for p in files} & seen:
        raise SystemExit("the scoring tail overlaps the fit; lower --share")

    mask = np.load(HERE.parent / "fusion_scoring" / "masks" / "d3d_envelope.npz")
    outside = (~mask["mask_coarse"].astype(bool)).astype(np.float64)
    print(f"Damping toward Delta* = 0 outside the vessel envelope: "
          f"{outside.sum():.0f} of {outside.size} grid points carry the penalty")

    comps = np.asarray(art["pca"].pca.components_).reshape(n_pca, EFIT_GRID_SIZE, EFIT_GRID_SIZE)
    q = grad_shafranov_form(comps, plan["grid_R"], plan["grid_Z"], outside)
    q_hat = q / (float(np.trace(q)) / n_pca)
    print(f"  Q built on the {n_pca} components, condition number "
          f"{np.linalg.cond(q_hat + np.eye(n_pca)):.1f}")

    # Predict ONCE. Everything the sweep changes is a linear solve on the coefficients afterwards.
    shots = local_score.load_shots(0, 0, "local", args.local_data_dir, args.config, files)
    coeffs, scalars, coils = [], [], []
    for s in shots:
        feats = features_for_row(s["row"])
        if plan["subtract"] or plan["inputs"] != "currents":
            check_grid(plan, s["row"])
        tgt = _predict_targets(art, args.model, art["scaler"].transform(build_inputs(plan, feats)))
        coeffs.append(tgt[:, :n_pca])
        scalars.append(tgt[:, n_pca:])
        coils.append(coil_flux(plan, feats).astype(np.float64) if plan["subtract"] else None)
    print(f"  predicted {sum(len(c) for c in coeffs)} frames once; "
          f"sweeping {len(args.lambdas)} damping strengths")

    args.work.mkdir(parents=True, exist_ok=True)
    for lam in args.lambdas:
        a = np.eye(n_pca) + lam * q_hat
        preds = {}
        for i, (c, sc, coil) in enumerate(zip(coeffs, scalars, coils, strict=True)):
            damped = c if lam == 0.0 else np.linalg.solve(a, c.T).T
            psi = np.asarray(art["pca"].inverse_transform(damped), dtype=np.float64)
            if coil is not None:
                psi = psi + coil
            preds[f"shot_{i:04d}_psirz"] = psi.astype(np.float16)
            preds[f"shot_{i:04d}_q95"] = sc[:, 0].astype(np.float32)
            preds[f"shot_{i:04d}_betaN"] = sc[:, 1].astype(np.float32)
        out = args.work / f"lam_{lam:g}.npz"
        # The numpy stubs type the second parameter as a positional flag; see
        # submission_skeleton.py, which carries the same annotation for the same reason.
        np.savez_compressed(out, **preds)  # type: ignore[arg-type]
        print(f"\n=== lambda = {lam:g} ===")
        local_score.main(["--source", "local", "--local-data-dir", str(args.local_data_dir),
                          "--config", args.config, "--jobs", str(args.jobs),
                          "--pred", str(out), "--files", *[str(p) for p in files]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
