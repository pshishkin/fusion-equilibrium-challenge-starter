#!/usr/bin/env python3
"""Gate (Astra round 2, idea 2): do the members weigh differently in the rare regimes?

    uv run python tools/stack_regime_gate.py runs/cv5_fixcos --members ... --basis-from runs/fc_p70 \
        --out runs/regime_gate [--score]

The Consistency error sits in a tail (5% of shots, ~55% of the li/Z_axis/volume SSE) and the
basis holds those maps (tools/tail_repr_gate.py): the error is the regression's. If some members
are better there, a linear stage with one set of weights for the tail and one for the rest
recovers part of it without training anything.

The regime is read only from what inference has: the seven scalars derived from the stacked map
(the cached 11-member map, runs/consgate_fc11 — a detector, not a target). A frame's distance is
the largest |z| of R_axis, Z_axis, kappa, volume and li against the training folds' median and
MAD; the top q% of the training folds' distances set the threshold. Per regime, ridgecv_block
weights are fitted on the training folds' frames of that regime and compared on the held-out fold
with the single ridgecv_block fit, exactly from sufficient statistics. `--score` scores the gated
stage natively for the best q.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import stack_cv
from cv_oof import score_fold
from stack_cv import corrected_payload, linear_blend, load_basis, load_fold, member_targets
from stack_linear_variants import add, fit_ridge_cv, fold_stats, mse

DETECT = [0, 1, 2, 5, 6]  # R_axis, Z_axis, kappa, volume, li
QS = (0.05, 0.10, 0.20)


def blocks(k: int) -> list[np.ndarray]:
    cols = np.arange(k + 2)
    return [cols[a:min(a + 10, k)] for a in range(0, k, 10)] + [np.array([k]), np.array([k + 1])]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("campaign", type=Path)
    ap.add_argument("--members", required=True)
    ap.add_argument("--basis-from", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--jobs", type=int, default=30)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    folds = [0, 1, 2, 3, 4]
    members = args.members.split(",")
    basis = load_basis(args.campaign, args.basis_from)
    k = basis["pca"].n_components
    tables, det = {}, {}
    for f in folds:
        t = load_fold(args.campaign, members, f, basis, args.jobs)
        del t["Y"]
        t["members_T"] = member_targets(t)
        del t["member_coef"], t["member_qb"]
        t["truth"] = np.column_stack([t["true_coef"], t["S"]])
        det[f] = np.load(f"runs/consgate_fc11/scalars_fold{f}.npz")["pred"][:, DETECT]
        if len(det[f]) != len(t["truth"]):
            raise ValueError(f"fold {f}: detector has {len(det[f])} rows, table {len(t['truth'])}")
        tables[f] = t
        print(f"fold {f}: {len(t['truth'])} frames", flush=True)
    report: dict[str, Any] = {"members": members, "folds": {}}
    best_gain, best_q = -np.inf, None
    for q in QS:
        tot_single = tot_gated = 0.0
        entry = {}
        for f in folds:
            others = [j for j in folds if j != f]
            ref = np.concatenate([det[j] for j in others])
            med = np.nanmedian(ref, 0)
            mad = np.nanmedian(np.abs(ref - med), 0) + 1e-12

            def dist(d: np.ndarray) -> np.ndarray:
                return np.nan_to_num(np.abs(d - med) / mad, nan=1e6).max(1)

            thr = np.quantile(dist(ref), 1 - q)
            masks = {j: dist(det[j]) > thr for j in folds}
            st_all = {j: fold_stats(tables[j]["members_T"], tables[j]["truth"]) for j in folds}
            st_t = {j: fold_stats([m[masks[j]] for m in tables[j]["members_T"]],
                                  tables[j]["truth"][masks[j]]) for j in folds}
            st_n = {j: fold_stats([m[~masks[j]] for m in tables[j]["members_T"]],
                                  tables[j]["truth"][~masks[j]]) for j in folds}
            W = fit_ridge_cv([st_all[j] for j in others], blocks(k))
            Wt = fit_ridge_cv([st_t[j] for j in others], blocks(k))
            Wn = fit_ridge_cv([st_n[j] for j in others], blocks(k))
            e_single = mse(W, st_t[f]) * st_t[f][3] + mse(W, st_n[f]) * st_n[f][3]
            e_gated = mse(Wt, st_t[f]) * st_t[f][3] + mse(Wn, st_n[f]) * st_n[f][3]
            psi = slice(0, k)
            r_tail = mse(Wt, st_t[f])[psi].sum() / mse(W, st_t[f])[psi].sum()
            r_norm = mse(Wn, st_n[f])[psi].sum() / mse(W, st_n[f])[psi].sum()
            r_all = e_gated[psi].sum() / e_single[psi].sum()
            tot_single += e_single[psi].sum(); tot_gated += e_gated[psi].sum()
            entry[f] = {"tail_frames": int(masks[f].sum()), "psi_ratio_tail": r_tail,
                        "psi_ratio_normal": r_norm, "psi_ratio_all": r_all,
                        "qb_ratio": (e_gated[k:] / e_single[k:]).tolist()}
            print(f"q {q:.2f} fold {f}: tail {int(masks[f].sum())} frames | psi MSE gated/single: "
                  f"tail {r_tail:.4f} normal {r_norm:.4f} all {r_all:.4f} | q95 "
                  f"{e_gated[k] / e_single[k]:.4f} betaN {e_gated[k + 1] / e_single[k + 1]:.4f}",
                  flush=True)
            if args.score:
                tables[f].setdefault("gates", {})[q] = (masks[f], W, Wt, Wn)
        report["folds"][str(q)] = entry
        gain = 1 - tot_gated / tot_single
        print(f"q {q:.2f}: pooled psi MSE gain {gain:+.4f}", flush=True)
        if gain > best_gain:
            best_gain, best_q = gain, q
        (args.out / "report.json").write_text(json.dumps(report, indent=1, default=float))
    if not args.score:
        return 0
    equal = np.full(len(members), 1.0 / len(members))
    for f in folds:
        t = tables[f]
        mask, W, Wt, Wn = t["gates"][best_q]
        stage1 = sum(w * m for w, m in zip(equal, t["members_T"], strict=True))
        single = linear_blend(t, W)
        gated = np.where(mask[:, None], linear_blend(t, Wt), linear_blend(t, Wn))
        for name, blend in (("single", single), ("gated", gated)):
            payload = corrected_payload(t, stage1, blend - stage1, equal, basis)
            (r,) = score_fold(t["files"], payload, args.jobs)["results"].values()
            del payload
            report.setdefault("native", {}).setdefault(str(f), {})[name] = r
            print(f"fold {f} {name} (q {best_q}): S {r['S']:.6f} Cons {r['consistency']:.6f}",
                  flush=True)
            (args.out / "report.json").write_text(json.dumps(report, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
