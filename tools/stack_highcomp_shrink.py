#!/usr/bin/env python3
"""Shrink the high-order basis components of the stacked map: does li come back?

    uv run python tools/stack_highcomp_shrink.py runs/cv5_fixcos --members ... \
        --basis-from runs/cap_mlp_p100_3x512 --out runs/shrink_s15 [--cuts 70,50] [--alphas 0.7,0.4]

Public 2026-10-04: moving the stack to a PCA-100 basis improved D_LCFS, kappa and triangularity as
CV said, but li fell -0.0016 (full18 -> full26) on the test set. li is <B_p^2>, the integral of the
squared flux GRADIENT, and the components the wide basis adds are the fine-scale ones: their noisy
predictions cost little flux MSE (what the ridge stage minimises) and a lot of gradient. This
scales the global-basis components from `cut` on of the final linear-stage map by `alpha` (towards
the basis mean), one parameter, and scores every fold natively against alpha = 1.

The map is `corrected_payload`'s (members decoded in their own bases plus the global-basis
delta); the shrink subtracts (1 - alpha) * c[:, cut:] @ components[cut:], where c are the map's
global-basis coefficients (stage 1 projected + delta).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import stack_cv
from cv_oof import score_fold
from stack_cv import corrected_payload, linear_blend, load_basis, load_fold, member_targets

FOLDS = [0, 1, 2, 3, 4]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("campaign", type=Path)
    ap.add_argument("--members", required=True)
    ap.add_argument("--basis-from", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cuts", default="70,50")
    ap.add_argument("--alphas", default="0.7,0.4")
    ap.add_argument("--jobs", type=int, default=20)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    members = args.members.split(",")
    cuts = [int(x) for x in args.cuts.split(",")]
    alphas = sorted((float(x) for x in args.alphas.split(",")), reverse=True)
    basis = load_basis(args.campaign, args.basis_from)
    k = basis["pca"].n_components
    comps = np.asarray(basis["pca"].pca.components_, dtype=np.float32)  # (k, 4225)
    stack_cv.LINEAR_VARIANT = "ridgecv_block"
    equal = np.full(len(members), 1.0 / len(members))
    tables = {}
    for f in FOLDS:
        t = load_fold(args.campaign, members, f, basis, args.jobs)
        del t["Y"]
        t["members_T"] = member_targets(t)
        t["truth"] = np.column_stack([t["true_coef"], t["S"]])
        tables[f] = t
        print(f"fold {f}: {len(t['truth'])} frames", flush=True)
    report: dict = {"members": members, "cuts": cuts, "alphas": alphas, "folds": {}}
    for f in FOLDS:
        t = tables[f]
        W = stack_cv.fit_linear([tables[j] for j in FOLDS if j != f])
        stage1 = sum(w * m for w, m in zip(equal, t["members_T"], strict=True))
        blend = linear_blend(t, W)
        payload = corrected_payload(t, stage1, blend - stage1, equal, basis)
        lengths = t["lengths"]
        bounds = np.concatenate([[0], np.cumsum(lengths)])
        keys = [f"shot_{i:04d}_psirz" for i in range(len(lengths))]
        entry: dict = {}

        def score(tag: str) -> None:
            (r,) = score_fold(t["files"], payload, args.jobs)["results"].values()
            entry[tag] = r
            c = r["r2_cons_each"]
            print(f"fold {f} {tag}: S {r['S']:.6f} li {c['li']:.6f} kappa {c['kappa']:.6f} "
                  f"dlcfs {r['dlcfs']:.6f} cons {r['consistency']:.6f}", flush=True)
            report["folds"][str(f)] = entry
            (args.out / "report.json").write_text(json.dumps(report, indent=1, default=float))

        score("alpha1")
        for cut in cuts:
            applied = 0.0  # share of the high part removed so far
            for alpha in alphas:
                step = (1.0 - alpha) - applied
                for i, key in enumerate(keys):
                    a, b = bounds[i], bounds[i + 1]
                    hi = (blend[a:b, cut:k].astype(np.float32) @ comps[cut:]).reshape(-1, 65, 65)
                    payload[key] -= np.float32(step) * hi
                applied += step
                score(f"cut{cut}_a{alpha:g}")
            for i, key in enumerate(keys):  # restore alpha = 1 before the next cut
                a, b = bounds[i], bounds[i + 1]
                payload[key] += np.float32(applied) * (
                    blend[a:b, cut:k].astype(np.float32) @ comps[cut:]).reshape(-1, 65, 65)
        del payload
    tags = list(report["folds"]["0"].keys())
    for tag in tags:
        s = [report["folds"][str(f)][tag]["S"] for f in FOLDS]
        li = [report["folds"][str(f)][tag]["r2_cons_each"]["li"] for f in FOLDS]
        print(f"{tag:14s} mean S {np.mean(s):.6f}  li {np.mean(li):.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
