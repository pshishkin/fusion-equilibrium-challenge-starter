#!/usr/bin/env python3
"""Shrink only what the wide basis ADDS over the narrow one (Astra, 2026-10-04).

    uv run python tools/stack_delta_shrink.py runs/cv5_fixcos --members ... \
        --old-basis runs/fc_p70 --new-basis runs/cap_mlp_p100_3x512 --out runs/dshrink_s15

`stack_highcomp_shrink.py` pulled the wide basis's high coefficients towards the basis MEAN and lost
on every fold: that also removes signal the narrow-basis stack already predicted. Here both
linear (ridgecv_block) stacks are built on the same members, one in each basis, and with
Delta = map_new - map_old split into its projection on the new basis's first `cut` components
(Delta_low) and the rest (Delta_high, including whatever lies outside the basis):

    old:        map_old
    a0:         map_old + Delta_low
    a<alpha>:   map_old + Delta_low + alpha * Delta_high
    new:        map_new                        (= a1)

scored natively on every fold. Maps are built fold by fold (weights from the other folds'
sufficient statistics) and cached to disk, so the two bases never share memory.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cv_oof import score_fold
from stack_cv import corrected_payload, linear_blend, load_basis, load_fold, member_targets
from stack_linear_variants import fit_ridge_cv, fold_stats

FOLDS = [0, 1, 2, 3, 4]


def groups(k: int) -> list[np.ndarray]:
    cols = np.arange(k + 2)
    return [cols[a:min(a + 10, k)] for a in range(0, k, 10)] + [np.array([k]), np.array([k + 1])]


def build_maps(campaign: Path, members: list[str], basis_from: Path, cache: Path,
               jobs: int) -> None:
    """Per fold: the linear stack's maps (N, 65, 65) float32 and q95/betaN, cached as .npy."""
    if all((cache / f"maps_fold{f}.npy").exists() for f in FOLDS):
        return
    # Resume: folds already cached (an OOM-killed run leaves them) are not rebuilt.
    cache.mkdir(parents=True, exist_ok=True)
    basis = load_basis(campaign, basis_from)
    k = basis["pca"].n_components
    equal = np.full(len(members), 1.0 / len(members))
    stats = {}
    for f in FOLDS:
        t = load_fold(campaign, members, f, basis, jobs)
        stats[f] = fold_stats(member_targets(t), np.column_stack([t["true_coef"], t["S"]]))
        del t
        print(f"{basis_from.name}: fold {f} statistics", flush=True)
    for f in FOLDS:
        if (cache / f"maps_fold{f}.npy").exists() and (cache / f"lengths_fold{f}.npy").exists():
            continue
        W = fit_ridge_cv([stats[j] for j in FOLDS if j != f], groups(k))
        t = load_fold(campaign, members, f, basis, jobs)
        del t["Y"]
        t["members_T"] = member_targets(t)
        stage1 = sum(w * m for w, m in zip(equal, t["members_T"], strict=True))
        payload = corrected_payload(t, stage1, linear_blend(t, W) - stage1, equal, basis)
        n = len(t["lengths"])
        np.save(cache / f"maps_fold{f}.npy",
                np.concatenate([payload[f"shot_{i:04d}_psirz"] for i in range(n)]))
        np.save(cache / f"lengths_fold{f}.npy", np.asarray(t["lengths"]))
        np.save(cache / f"qb_fold{f}.npy", np.stack(
            [np.concatenate([payload[f"shot_{i:04d}_{s}"] for i in range(n)])
             for s in ("q95", "betaN")], axis=1))
        del payload, t
        print(f"{basis_from.name}: fold {f} maps cached", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("campaign", type=Path)
    ap.add_argument("--members", required=True)
    ap.add_argument("--old-basis", type=Path, required=True)
    ap.add_argument("--new-basis", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cut", type=int, default=70)
    ap.add_argument("--alphas", default="0.5")
    ap.add_argument("--jobs", type=int, default=20)
    args = ap.parse_args()
    members = args.members.split(",")
    alphas = [float(x) for x in args.alphas.split(",")]
    old_c, new_c = args.out / "old", args.out / "new"
    build_maps(args.campaign, members, args.old_basis, old_c, args.jobs)
    build_maps(args.campaign, members, args.new_basis, new_c, args.jobs)
    comps = np.asarray(load_basis(args.campaign, args.new_basis)["pca"].pca.components_[:args.cut],
                       dtype=np.float32)  # (cut, 4225), orthonormal rows
    report: dict = {"members": members, "cut": args.cut, "alphas": alphas, "folds": {}}
    from cv_oof import _fold_files
    import joblib
    for f in FOLDS:
        art = joblib.load(args.campaign / f"fold{f}" / "baseline.joblib")
        files = _fold_files(art)
        m_old = np.load(old_c / f"maps_fold{f}.npy")
        m_new = np.load(new_c / f"maps_fold{f}.npy")
        qb = np.load(new_c / f"qb_fold{f}.npy")
        d = (m_new - m_old).reshape(len(m_new), -1)
        d_low = ((d @ comps.T) @ comps).reshape(m_new.shape)
        d_high = (d.reshape(m_new.shape) - d_low)
        del d
        lengths = np.load(new_c / f"lengths_fold{f}.npy")
        if not np.array_equal(lengths, np.load(old_c / f"lengths_fold{f}.npy")):
            raise ValueError(f"fold {f}: the two stacks scored different frames")
        bounds = np.concatenate([[0], np.cumsum(lengths)])
        entry: dict = {}

        def score(tag: str, maps: np.ndarray, qbv: np.ndarray) -> None:
            payload = {}
            for i in range(len(lengths)):
                a, b = bounds[i], bounds[i + 1]
                payload[f"shot_{i:04d}_psirz"] = maps[a:b]
                payload[f"shot_{i:04d}_q95"] = qbv[a:b, 0]
                payload[f"shot_{i:04d}_betaN"] = qbv[a:b, 1]
            (r,) = score_fold(files, payload, args.jobs)["results"].values()
            entry[tag] = r
            c = r["r2_cons_each"]
            print(f"fold {f} {tag}: S {r['S']:.6f} li {c['li']:.6f} kappa {c['kappa']:.6f} "
                  f"dlcfs {r['dlcfs']:.6f} cons {r['consistency']:.6f}", flush=True)
            report["folds"][str(f)] = entry
            (args.out / "report.json").write_text(json.dumps(report, indent=1, default=float))

        # q95/betaN from the new stack throughout: the question is the map.
        score("old", m_old, qb)
        score("a0", m_old + d_low, qb)
        for a in alphas:
            score(f"a{a:g}", m_old + d_low + np.float32(a) * d_high, qb)
        score("new", m_new, qb)
        del m_old, m_new, d_low, d_high
    tags = list(report["folds"]["0"].keys())
    for tag in tags:
        s = [report["folds"][str(f)][tag]["S"] for f in FOLDS]
        li = [report["folds"][str(f)][tag]["r2_cons_each"]["li"] for f in FOLDS]
        print(f"{tag:8s} mean S {np.mean(s):.6f}  li {np.mean(li):.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
