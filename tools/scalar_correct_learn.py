#!/usr/bin/env python3
"""Cross-fitted scalar corrector: predict the map's functional errors, steer the map, score natively.

    uv run python tools/scalar_correct_learn.py --maps runs/shotcmp_22_34/b --test-fold 0 \
        --out runs/sclearn_f0

Second gate after `scalar_correct_gate.py` (oracle: the map is steerable, D_LCFS -28% with the
truth's functionals as targets). Here the targets are PREDICTED. For frames of the training folds
(every `--every`-th frame of every shot; maps are the linear stack's out-of-fold maps): features =
raw inputs X (incl. the thomson_edge heights), the map's own functionals, its first 30 global PCA
coefficients, relative frame position; labels = truth - map for Z_top, Z_bot, Z_axis, kappa,
volume. One HistGradientBoosting per label, fitted on the training folds, predicts the test fold;
`shrink` scales the predicted corrections. The test sample (same shots/frames as the oracle gate)
is steered with `scalar_correct_gate.frame_task` (targets = map + predicted error, li / R_axis /
triangularities held) and its native terms compared with the uncorrected map.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "fusion_scoring"))

import scalar_correct_gate as G  # noqa: E402
from cv_oof import _fold_files  # noqa: E402
from lcfs import extract_lcfs  # noqa: E402
from derive import derive_frame  # noqa: E402
from my_experiments import baseline_model as B  # noqa: E402
from stack_cv import load_basis  # noqa: E402
from toolkit.parallel import pmap  # noqa: E402

TARGETS = ["Z_top", "Z_bot", "Z_axis", "kappa", "volume"]
# Astra 2026-10-13: the vertical-position controller acts on hidden sensors, so its commands may
# carry the information the map lacks. Up-down antisymmetric coil combinations u_n = I_nA - I_nB
# (A sits at +Z, B at its mirror -Z), and their mean change over windows around the frame —
# past and FUTURE (offline inputs) — relative to the frame's value.
PAIRS = [(f"F{n}A", f"F{n}B") for n in range(1, 10)]
WINDOWS_MS = [(-100.0, -20.0), (-20.0, -5.0), (5.0, 20.0), (20.0, 100.0)]


def window_increments(t: np.ndarray, u: np.ndarray, at: np.ndarray) -> np.ndarray:
    """(len(at), len(WINDOWS_MS)) mean of u over [at+a, at+b] minus u(at), on u's own time base."""
    c = np.concatenate([[0.0], np.cumsum(u)])
    u0 = np.interp(at, t, u)
    out = np.empty((len(at), len(WINDOWS_MS)))
    for j, (a, b) in enumerate(WINDOWS_MS):
        lo = np.searchsorted(t, at + a)
        hi = np.searchsorted(t, at + b)
        n = np.maximum(hi - lo, 1)
        mean = np.where(hi > lo, (c[hi] - c[lo]) / n, u0)   # outside the signal: no change
        out[:, j] = mean - u0
    return out


def pair_features(args: tuple) -> np.ndarray:
    """Raw-signal (0.05 ms) pair increments for the frames `local` of one shot file."""
    path, local, mode, levels, ef = args
    if mode == "frames":
        # Control: the same windows from the EFIT-frame series only (no fast signal).
        cols = [levels[:, B.D3D_MAGNETICS_SIGNALS.index(a)] - levels[:, B.D3D_MAGNETICS_SIGNALS.index(b)]
                for a, b in PAIRS]
        return np.hstack([window_increments(ef, u.astype(np.float64), ef[local]) for u in cols])
    row = pd.read_parquet(path, columns=["magnetics_time", "efit_times"]
                          + [f"magnetics_{x}" for p in PAIRS for x in p]).iloc[0]
    t = np.asarray(row["magnetics_time"], dtype=np.float64)
    at = np.asarray(row["efit_times"], dtype=np.float64)[local]
    return np.hstack([window_increments(t, np.asarray(row[f"magnetics_{a}"], np.float64)
                                        - np.asarray(row[f"magnetics_{b}"], np.float64), at)
                      for a, b in PAIRS])


def label_task(args: tuple) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(ok mask, map functionals, truth functionals) for the frames of one shot."""
    psi_t, psi_p, ctx = args
    G.CTX.update(ctx)
    n = len(psi_t)
    base = np.full((n, len(G.NAMES)), np.nan)
    truth = np.full((n, len(G.NAMES)), np.nan)
    for k in range(n):
        tc = extract_lcfs(psi_t[k], ctx["R"], ctx["Z"], "DIII-D", ctx["mc"], ctx["mf"], n_points=512)
        if tc is None:
            continue
        tc = np.asarray(tc, dtype=np.float64)
        vt = derive_frame(psi_t[k], ctx["R"], ctx["Z"], "DIII-D", ctx["mc"], ctx["mf"], contour=tc)
        truth[k] = [tc[:, 1].max(), tc[:, 1].min()] + [vt[x] for x in G.NAMES[2:]]
        base[k], _ = G.functionals(psi_p[k])
    ok = np.isfinite(base).all(1) & np.isfinite(truth).all(1)
    return ok, base, truth


def fold_table(campaign: Path, maps: Path, fold: int, every: int, comps: np.ndarray,
               ctx: dict, jobs: int, shots: np.ndarray | None = None,
               cache: Path | None = None, extra: str = "none") -> dict:
    files = _fold_files(joblib.load(campaign / f"fold{fold}" / "baseline.joblib"))
    X, Y, _, L = B._read_shots(files, f"fold{fold}", 1.0, 16)
    P = np.load(maps / f"maps_fold{fold}.npy", mmap_mode="r")
    b = np.concatenate([[0], np.cumsum(L)])
    sel = list(range(len(L))) if shots is None else list(shots)
    idx, pos, shot_of = [], [], []
    for i in sel:
        fr = np.arange(b[i], b[i + 1])[::every]
        idx.append(fr)
        pos.append((fr - b[i]) / max(L[i] - 1, 1))
        shot_of.append(np.full(len(fr), i))
    idx, pos, shot_of = np.concatenate(idx), np.concatenate(pos), np.concatenate(shot_of)
    cf = cache / f"labels_fold{fold}_{every}_{len(sel)}.npz" if cache else None
    if cf is not None and cf.exists():
        z = np.load(cf)
        ok, base, truth = z["ok"], z["base"], z["truth"]
    else:
        tasks = [(Y[idx[shot_of == i]].astype(np.float64),
                  np.asarray(P[idx[shot_of == i]], dtype=np.float64), ctx) for i in sel]
        out = pmap(label_task, tasks, jobs)
        ok = np.concatenate([o[0] for o in out])
        base = np.concatenate([o[1] for o in out])
        truth = np.concatenate([o[2] for o in out])
        if cf is not None:
            np.savez(cf, ok=ok, base=base, truth=truth)
    coef = np.asarray(P[idx], dtype=np.float64).reshape(len(idx), -1) @ comps.reshape(len(comps), -1).T
    parts = [X[idx], base, coef, pos]
    if extra != "none":
        ef_all = {}
        tasks = []
        for i in sel:
            loc = idx[shot_of == i] - b[i]
            ef = np.asarray(pd.read_parquet(files[i], columns=["efit_times"]).iloc[0]["efit_times"],
                            np.float64)
            tasks.append((files[i], loc, extra, X[b[i]:b[i + 1], :B.N_SIGNALS], ef))
        parts.append(np.concatenate(pmap(pair_features, tasks, jobs)))
    F = np.column_stack(parts)
    return {"F": F[ok], "base": base[ok], "truth": truth[ok], "idx": idx[ok],
            "Y": Y, "P": P}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--maps", type=Path, required=True)
    ap.add_argument("--campaign", type=Path, default=Path("runs/cv5_fixcos"))
    ap.add_argument("--basis-from", type=Path, default=Path("runs/cap_mlp_p100_3x512"))
    ap.add_argument("--test-fold", type=int, default=0)
    ap.add_argument("--train-every", type=int, default=4)
    ap.add_argument("--test-shots", type=int, default=150)
    ap.add_argument("--test-every", type=int, default=3)
    ap.add_argument("--k", type=int, default=30)
    ap.add_argument("--shrink", default="0.5,1.0")
    ap.add_argument("--cache", type=Path, help="labels (map / truth functionals) cached here")
    ap.add_argument("--extra", default="none", choices=["none", "raw", "frames"],
                    help="add the up-down coil pair increments (raw 0.05 ms or EFIT-frame control)")
    ap.add_argument("--no-steer", action="store_true", help="report the R2 of the errors only")
    ap.add_argument("--jobs", type=int, default=24)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    files = _fold_files(joblib.load(args.campaign / f"fold{args.test_fold}" / "baseline.joblib"))
    r = pd.read_parquet(files[0], columns=["efit_grid_R", "efit_grid_Z"]).iloc[0]
    m = np.load(ROOT / "fusion_scoring" / "masks" / "d3d_envelope.npz")
    ctx = {"R": np.asarray(r["efit_grid_R"], float), "Z": np.asarray(r["efit_grid_Z"], float),
           "mc": m["mask_coarse"].astype(bool), "mf": m["mask_coarse"].astype(float)}
    G.CTX.update(ctx)
    pca = load_basis(args.campaign, args.basis_from)["pca"].pca
    comps = np.asarray(pca.components_[:args.k], dtype=np.float64).reshape(args.k, 65, 65)
    eps = 0.1 * np.sqrt(pca.explained_variance_[:args.k])
    tj = [G.NAMES.index(t) for t in TARGETS]
    trains = []
    for f in range(5):
        if f == args.test_fold:
            continue
        t = fold_table(args.campaign, args.maps, f, args.train_every, comps, ctx, args.jobs,
                       cache=args.cache, extra=args.extra)
        trains.append((t["F"], t["truth"][:, tj] - t["base"][:, tj]))
        del t
        print(f"fold {f}: {len(trains[-1][0])} training frames", flush=True)
    Ftr = np.concatenate([a for a, _ in trains])
    Dtr = np.concatenate([b for _, b in trains])
    del trains
    rng = np.random.default_rng(0)
    nL = len(_fold_files(joblib.load(args.campaign / f"fold{args.test_fold}" / "baseline.joblib")))
    shots = rng.choice(nL, min(args.test_shots, nL), replace=False)
    te = fold_table(args.campaign, args.maps, args.test_fold, args.test_every, comps, ctx,
                    args.jobs, shots, cache=args.cache, extra=args.extra)
    Dte = te["truth"][:, tj] - te["base"][:, tj]
    pred = np.empty_like(Dte)
    rep: dict = {"train_frames": len(Ftr), "test_frames": len(Dte), "r2_delta": {}}
    for j, name in enumerate(TARGETS):
        mdl = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_leaf_nodes=63,
                                            l2_regularization=1.0, random_state=0)
        mdl.fit(Ftr, Dtr[:, j])
        pred[:, j] = mdl.predict(te["F"])
        r2 = 1 - ((Dte[:, j] - pred[:, j]) ** 2).sum() / ((Dte[:, j] - Dte[:, j].mean()) ** 2).sum()
        rep["r2_delta"][name] = float(r2)
        print(f"{name}: R2 of the predicted error {r2:+.3f}", flush=True)
    if args.no_steer:
        (args.out / "summary.json").write_text(json.dumps(rep, indent=1))
        return 0
    truth_all, base_all = te["truth"], te["base"]
    sst = ((truth_all - truth_all.mean(0)) ** 2).sum(0)
    fr = te["idx"]
    sst_psi = float(((te["Y"][fr].astype(np.float64) - te["Y"][fr].mean(0)) ** 2).sum())
    scale = np.std(truth_all, axis=0)
    for s in [float(x) for x in args.shrink.split(",")]:
        # Targets for the steering: the map's own value + the shrunken predicted error.
        tgt = base_all.copy()
        tgt[:, tj] = base_all[:, tj] + s * pred
        tasks = [(te["Y"][k].astype(np.float64), np.asarray(te["P"][k], dtype=np.float64),
                  comps, eps, scale, 0.1, 2, ctx, tgt[i]) for i, k in enumerate(fr)]
        out = pmap(steer_task, tasks, args.jobs)
        keep = [i for i, o in enumerate(out) if o is not None]
        tb, bb = truth_all[keep], base_all[keep]
        nb = np.array([out[i]["vals"] for i in keep])
        dr2 = {n: float((((tb[:, j] - bb[:, j]) ** 2).sum() - ((tb[:, j] - nb[:, j]) ** 2).sum())
                        / sst[j]) for j, n in enumerate(G.NAMES)}
        d_base = float(np.mean([out[i]["d_base"] for i in keep]))
        d_new = float(np.mean([out[i]["d"] for i in keep]))
        dpsi = sum(out[i]["sse_base"] - out[i]["sse"] for i in keep) / sst_psi
        dS = 0.2 / 7 * sum(dr2[n] for n in G.CONS) + 0.10 * (d_base - d_new) + 0.55 * dpsi
        rep[f"shrink_{s:g}"] = {"frames": len(keep), "dR2": dr2, "D_base": d_base,
                                "D_new": d_new, "dR2_psi": dpsi, "dS_est": dS}
        print(f"[shrink {s:g}] {len(keep)} frames dS_est {dS:+.6f} D_LCFS {d_base:.5f} -> "
              f"{d_new:.5f} dR2_psi {dpsi:+.6f}")
        print("   " + "  ".join(f"{n} {dr2[n]:+.5f}" for n in G.NAMES), flush=True)
    (args.out / "summary.json").write_text(json.dumps(rep, indent=1))
    return 0


def steer_task(args: tuple) -> dict | None:
    """Steer one map towards given targets (all nine functionals; held ones = the map's own)."""
    psi_t, psi_p, comps, eps, scale, lam, steps, ctx, target = args
    G.CTX.update(ctx)
    tc = extract_lcfs(psi_t, ctx["R"], ctx["Z"], "DIII-D", ctx["mc"], ctx["mf"], n_points=512)
    if tc is None:
        return None
    tc = np.asarray(tc, dtype=np.float64)
    f0, d0 = G.functionals(psi_p, tc)
    if not np.isfinite(f0).all():
        return None
    psi, f = psi_p.copy(), f0.copy()
    K = len(comps)
    for _ in range(steps):
        J = np.empty((len(G.NAMES), K))
        bad = False
        for i in range(K):
            fi, _ = G.functionals(psi + eps[i] * comps[i])
            if not np.isfinite(fi).all():
                bad = True
                break
            J[:, i] = (fi - f) / eps[i]
        if bad:
            break
        A, bvec = J / scale[:, None], (target - f) / scale
        delta = np.linalg.solve(A.T @ A + lam * np.diag(1.0 / (10.0 * eps) ** 2), A.T @ bvec)
        r0 = float((((target - f) / scale) ** 2).sum())
        moved = False
        for alpha in (1.0, 0.5, 0.25, 0.1):
            cand = psi + alpha * np.tensordot(delta, comps, axes=1)
            fc, _ = G.functionals(cand)
            if np.isfinite(fc).all() and float((((target - fc) / scale) ** 2).sum()) < r0:
                psi, f, moved = cand, fc, True
                break
        if not moved:
            break
    fv, d = G.functionals(psi, tc)
    if not np.isfinite(fv).all():
        return None
    return {"vals": fv, "d": d, "d_base": d0, "sse": float(((psi - psi_t) ** 2).sum()),
            "sse_base": float(((psi_p - psi_t) ** 2).sum())}


if __name__ == "__main__":
    raise SystemExit(main())
