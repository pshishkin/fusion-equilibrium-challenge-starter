#!/usr/bin/env python3
"""Can a predicted map be steered to given geometric scalars, and does the native score follow?

    uv run python tools/scalar_correct_gate.py --maps runs/shotcmp_22_34/b --fold 0 --shots 150 \
        --every 3 --out runs/scgate_oracle_f0

User + Astra, 2026-10-13. D_LCFS is a symmetric Hausdorff distance and its worst point sits at the
very top (43% of frames) or the very bottom / X-point (47%) of the LCFS: the vertical extent, the
same error as Z_axis / kappa / volume. Plan: predict the errors of the scorer's own geometric
functionals and move the map a small, local step to match them while holding li and R_axis.

This is the ORACLE gate: the targets are the truth's functionals. Per frame, the Jacobian of the
scorer functionals (Z_top, Z_bot of the LCFS, Z_axis, kappa, volume, tri_top, tri_bot, R_axis, li)
with respect to the first K components of the stack's global PCA basis is taken by forward
differences AROUND THE CURRENT PREDICTION, a ridge Gauss-Newton step moves the coefficients towards
the targets (rows scaled by the truth spread; held scalars get a zero-change target), and the
native per-frame terms are recomputed: D_LCFS (Hausdorff / Rgeo of the true contour), the seven
consistency errors and the flux error. Reported as an estimated dS on the sample (R2 changes from
SSE changes over the sample's own SST). Variants: `geo` targets Z_top/Z_bot/Z_axis/kappa/volume and
holds the rest; `all` targets all nine.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "fusion_scoring"))

from common import AXIS_SIGN  # noqa: E402
from contour import symmetric_hausdorff  # noqa: E402
from cv_oof import _fold_files  # noqa: E402
from derive import derive_frame  # noqa: E402
from lcfs import extract_lcfs, extract_lcfs_with_sign, major_radius  # noqa: E402
from my_experiments import baseline_model as B  # noqa: E402
from stack_cv import load_basis  # noqa: E402
from toolkit.parallel import pmap  # noqa: E402

NAMES = ["Z_top", "Z_bot", "Z_axis", "kappa", "volume", "tri_top", "tri_bot", "R_axis", "li"]
CONS = ["R_axis", "Z_axis", "kappa", "tri_top", "tri_bot", "volume", "li"]
VARIANTS = {"geo": ["Z_top", "Z_bot", "Z_axis", "kappa", "volume"], "all": NAMES}
CTX: dict = {}


def functionals(psi: np.ndarray, true_c: np.ndarray | None = None) -> tuple[np.ndarray, float]:
    """(values in NAMES order, D_LCFS against true_c or nan) of one map, as the scorer derives them."""
    R, Z, mc, mf = CTX["R"], CTX["Z"], CTX["mc"], CTX["mf"]
    c, _ = extract_lcfs_with_sign(psi, R, Z, "DIII-D", mc, mf, n_iter=22,
                                  axis_sign=AXIS_SIGN["DIII-D"])
    v = derive_frame(psi, R, Z, "DIII-D", mc, mf, contour=c, with_li=True,
                     axis_sign=AXIS_SIGN["DIII-D"])
    if c is None:
        return np.full(len(NAMES), np.nan), np.nan
    c = np.asarray(c, dtype=np.float64)
    out = np.array([c[:, 1].max(), c[:, 1].min()] + [v[n] for n in NAMES[2:]], dtype=np.float64)
    d = (symmetric_hausdorff(c, true_c) / major_radius(true_c)) if true_c is not None else np.nan
    return out, d


def frame_task(args: tuple) -> dict | None:
    psi_t, psi_p, comps, eps, scale, lam, steps, ctx = args
    CTX.update(ctx)   # spawned workers: the grid and masks travel with the task
    R, Z, mc, mf = CTX["R"], CTX["Z"], CTX["mc"], CTX["mf"]
    tc = extract_lcfs(psi_t, R, Z, "DIII-D", mc, mf, n_points=512)
    if tc is None:
        return None
    tc = np.asarray(tc, dtype=np.float64)
    vt = derive_frame(psi_t, R, Z, "DIII-D", mc, mf, contour=tc, with_li=True)
    truth = np.array([tc[:, 1].max(), tc[:, 1].min()] + [vt[n] for n in NAMES[2:]])
    if not np.isfinite(truth).all():
        return None
    f0, d0 = functionals(psi_p, tc)
    if not np.isfinite(f0).all():
        return None
    res = {"truth": truth, "base": f0, "d_base": d0,
           "sse_psi_base": float(((psi_p - psi_t) ** 2).sum())}
    K = len(comps)
    for vname, targeted in VARIANTS.items():
        psi = psi_p.copy()
        f, ok = f0.copy(), True
        for _ in range(steps):
            J = np.empty((len(NAMES), K))
            for i in range(K):
                fi, _ = functionals(psi + eps[i] * comps[i])
                if not np.isfinite(fi).all():
                    ok = False
                    break
                J[:, i] = (fi - f) / eps[i]
            if not ok:
                break
            target = np.where([n in targeted for n in NAMES], truth - f, 0.0)
            A = J / scale[:, None]
            b = target / scale
            # ridge in units of each component's spread (eps is 0.1 spread)
            reg = lam * np.diag(1.0 / (10.0 * eps) ** 2)
            delta = np.linalg.solve(A.T @ A + reg, A.T @ b)
            # Line search: the functionals are piecewise smooth (contour extrema, O-point), so a
            # full Gauss-Newton step can overshoot. Accept the step length that lowers the scaled
            # residual (targets + held rows); none -> stop.
            def resid(fv: np.ndarray) -> float:
                tv = np.where([n in targeted for n in NAMES], truth - fv, f0 - fv)
                return float(((tv / scale) ** 2).sum())
            best, best_r = None, resid(f)
            for alpha in (1.0, 0.5, 0.25, 0.1):
                cand = psi + alpha * np.tensordot(delta, comps, axes=1)
                fc, _ = functionals(cand)
                if np.isfinite(fc).all() and resid(fc) < best_r:
                    best, best_r, fbest = cand, resid(fc), fc
                    break
            if best is None:
                break
            psi, f = best, fbest
        if not ok:
            res[vname] = None
            continue
        f, d = functionals(psi, tc)
        res[vname] = {"vals": f, "d": d, "sse_psi": float(((psi - psi_t) ** 2).sum())}
    return res


def init(ctx: dict) -> None:
    CTX.update(ctx)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--maps", type=Path, required=True, help="dir with maps_fold{f}.npy (linear stack)")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--campaign", type=Path, default=Path("runs/cv5_fixcos"))
    ap.add_argument("--basis-from", type=Path, default=Path("runs/cap_mlp_p100_3x512"))
    ap.add_argument("--shots", type=int, default=150)
    ap.add_argument("--every", type=int, default=3)
    ap.add_argument("--k", type=int, default=30)
    ap.add_argument("--lam", type=float, default=0.1)
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--jobs", type=int, default=24)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    files = _fold_files(joblib.load(args.campaign / f"fold{args.fold}" / "baseline.joblib"))
    _, Y, _, L = B._read_shots(files, f"fold{args.fold}", 1.0, 16)
    P = np.load(args.maps / f"maps_fold{args.fold}.npy", mmap_mode="r")
    if len(P) != len(Y):
        raise ValueError(f"maps {len(P)} frames vs truth {len(Y)}")
    r = pd.read_parquet(files[0], columns=["efit_grid_R", "efit_grid_Z"]).iloc[0]
    m = np.load(ROOT / "fusion_scoring" / "masks" / "d3d_envelope.npz")
    ctx = {"R": np.asarray(r["efit_grid_R"], float), "Z": np.asarray(r["efit_grid_Z"], float),
           "mc": m["mask_coarse"].astype(bool), "mf": m["mask_coarse"].astype(float)}
    init(ctx)
    pca = load_basis(args.campaign, args.basis_from)["pca"].pca
    comps = np.asarray(pca.components_[:args.k], dtype=np.float64).reshape(args.k, 65, 65)
    eps = 0.1 * np.sqrt(pca.explained_variance_[:args.k])
    b = np.concatenate([[0], np.cumsum(L)])
    shots = np.random.default_rng(0).choice(len(L), min(args.shots, len(L)), replace=False)
    frames = np.concatenate([np.arange(b[i], b[i + 1])[::args.every] for i in shots])
    # truth spread of each functional, for row scaling: from a quick pass on the truth maps
    scale_f = []
    for k in frames[::20]:
        f, _ = functionals(Y[k].astype(np.float64))
        if np.isfinite(f).all():
            scale_f.append(f)
    scale = np.std(np.array(scale_f), axis=0)
    print(f"{len(frames)} frames from {len(shots)} shots; K={args.k}; scale {np.round(scale, 4)}",
          flush=True)
    tasks = [(Y[k].astype(np.float64), np.asarray(P[k], dtype=np.float64), comps, eps, scale,
              args.lam, args.steps, ctx) for k in frames]
    out = [o for o in pmap(frame_task, tasks, args.jobs) if o is not None]
    joblib.dump(out, args.out / "frames.joblib")
    truth = np.array([o["truth"] for o in out])
    sst = ((truth - truth.mean(0)) ** 2).sum(0)
    sst_psi = float(((Y[frames].astype(np.float64) - Y[frames].mean(0)) ** 2).sum())
    summary = {"frames": len(out)}
    print(f"{len(out)} usable frames")
    for v in VARIANTS:
        sel = [o for o in out if o[v] is not None]
        if not sel:
            continue
        tb = np.array([o["truth"] for o in sel])
        base = np.array([o["base"] for o in sel])
        new = np.array([o[v]["vals"] for o in sel])
        dr2 = {}
        for j, n in enumerate(NAMES):
            dr2[n] = float((((tb[:, j] - base[:, j]) ** 2).sum()
                            - ((tb[:, j] - new[:, j]) ** 2).sum()) / sst[j])
        d_base = np.mean([o["d_base"] for o in sel])
        d_new = np.mean([o[v]["d"] for o in sel])
        dpsi = (sum(o["sse_psi_base"] for o in sel) - sum(o[v]["sse_psi"] for o in sel)) / sst_psi
        dS = 0.2 / 7 * sum(dr2[n] for n in CONS) + 0.10 * (d_base - d_new) + 0.55 * dpsi
        summary[v] = {"frames": len(sel), "dR2": dr2, "D_base": d_base, "D_new": d_new,
                      "dR2_psi": dpsi, "dS_est": dS}
        print(f"[{v}] {len(sel)} frames  dS_est {dS:+.6f}  D_LCFS {d_base:.5f} -> {d_new:.5f}  "
              f"dR2_psi {dpsi:+.6f}")
        print("   " + "  ".join(f"{n} {dr2[n]:+.5f}" for n in NAMES))
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
