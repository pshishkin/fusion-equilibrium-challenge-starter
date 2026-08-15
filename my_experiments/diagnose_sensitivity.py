#!/usr/bin/env python3
"""
Diagnostic C7 in ideas.md — is an expensive frame under-weighted, or simply harder?

    uv run python my_experiments/diagnose_sensitivity.py --share 0.01 --frames 1500 --jobs 24

C1 measured that the worst 1% of frames carry 22% of the geometry cost. That number alone does not
say what to do about it, because a frame's cost is the product of two things:

    cost_i  ~  || J_i (c_hat_i - c_i) ||^2        J_i = d(scalars)/d(coefficients) at frame i

A large cost can be a large `J_i` — the frame is SENSITIVE, equal coefficient error there buys more
scalar error, and the training loss (which uses one `M` averaged over 300 probe frames) charges it
the same as everywhere else. That is B6, and re-weighting fixes it. Or it can be a large coefficient
error — the frame is simply HARDER — in which case up-weighting it trades the other 90% away for
nothing. The two look identical in C1's output and call for opposite work.

So this measures `J_i` per frame directly, with the same central differences and the same probe step
the training loss is built from, and joins it to the per-frame costs C1 already wrote out. If cost
tracks sensitivity, B6 has its mechanism. If the expensive frames are of ordinary sensitivity, the
tail is difficulty and B6 dies here for the price of an afternoon.

Reads results/frame_costs.csv (run diagnose_frames.py first) and writes results/frame_sens.csv.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "fusion_scoring"))

from common import CONS_SCALARS, N_CONS, W_CONS  # noqa: E402

import local_score  # noqa: E402
from experiments import DEFAULT_LOCAL_DATA_DIR, EFIT_GRID_SIZE, HF_TRAIN_CONFIG  # noqa: E402
from my_experiments.baseline_model import (  # noqa: E402
    ARTIFACT,
    coil_flux,
    features_for_row,
    sorted_shots,
    take_share,
)
from my_experiments.models import load_params  # noqa: E402
from my_experiments.parallel import pimap, resolve_jobs  # noqa: E402
from my_experiments.progress import install_timestamps  # noqa: E402
from my_experiments.target_metric import _jacobian_task, scorer_context  # noqa: E402

COSTS = HERE.parent / "results" / "frame_costs.csv"
OUT = HERE.parent / "results" / "frame_sens.csv"


def main() -> int:
    install_timestamps()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--share", type=float, default=0.01,
                    help="share of shots, from the tail — must match the diagnose_frames.py run")
    ap.add_argument("--frames", type=int, default=1500,
                    help="frames to probe, spread evenly over the fold (default 1500). Each one "
                         "costs 2 x n_pca derivations, so this is the whole cost of the run")
    ap.add_argument("--jobs", type=int, default=0)
    ap.add_argument("--local-data-dir", type=Path, default=DEFAULT_LOCAL_DATA_DIR)
    ap.add_argument("--config", default=HF_TRAIN_CONFIG)
    args = ap.parse_args()

    if not COSTS.exists():
        raise SystemExit(f"{COSTS} not found — run diagnose_frames.py at the same --share first")
    art = joblib.load(ARTIFACT)
    params = load_params()
    plan = art["coil"]
    files = take_share(sorted_shots(args.local_data_dir, args.config, int(art["split_salt"])),
                       args.share, "tail")
    shots = local_score.load_shots(0, 0, "local", args.local_data_dir, args.config, files)

    # One flat index over the fold, so the sample is spread across shots and positions alike, and
    # every probed frame keeps the (shot, frame) identity the costs are keyed by.
    index = [(p.stem, k) for p, s in zip(files, shots, strict=True)
             for k in range(len(s["psi"]))]
    stride = max(1, len(index) // args.frames)
    pick = list(range(0, len(index), stride))[:args.frames]
    print(f"Probing {len(pick)} of {len(index)} frames over {len(files)} shots, "
          f"{params.n_pca} coefficients each")

    psi_by_shot = {p.stem: s["psi"].astype(np.float64) for p, s in zip(files, shots, strict=True)}
    psi = np.stack([psi_by_shot[index[i][0]][index[i][1]] for i in pick])

    # The probe step the training loss is built from. `psi_ss_tot` is the metric's own denominator
    # per frame, over the coil-subtracted residual — recomputed here on the scored shots rather
    # than read from the artifact, which does not store it separately. It sets the scale of the
    # perturbation only; the per-frame SPREAD this run is about does not depend on it.
    resid = []
    for s in shots:
        y = s["psi"].astype(np.float64)
        if plan["subtract"]:
            y = y - coil_flux(plan, features_for_row(s["row"])).astype(np.float64)
        resid.append(y)
    flat = np.concatenate([r.reshape(len(r), -1) for r in resid])
    psi_ss_tot = float(((flat - flat.mean()) ** 2).sum() / len(flat))
    delta = params.jacobian_delta * float(np.sqrt(psi_ss_tot / params.n_pca))
    print(f"  probe step {delta:.4g} Wb/rad")

    images = np.asarray(art["pca"].pca.components_).reshape(params.n_pca, EFIT_GRID_SIZE, -1)
    ctx = scorer_context(plan["grid_R"], plan["grid_Z"], plan["machine"])
    jobs = resolve_jobs(args.jobs, len(psi))
    chunks = np.array_split(np.arange(len(psi)), max(1, min(jobs * 4, len(psi))))
    tasks = [(psi[c], images, delta, ctx, i) for i, c in enumerate(chunks) if len(c)]
    parts = list(pimap(_jacobian_task, tasks, jobs))
    base = np.concatenate([b for b, _, _ in parts])
    jac = np.concatenate([j for _, j, _ in parts])

    good = np.isfinite(jac).all(axis=(1, 2)) & np.isfinite(base).all(axis=1)
    print(f"  {int(good.sum())} of {len(base)} frames gave a finite Jacobian for all {N_CONS} "
          f"scalars")
    var = base[good].var(axis=0)

    # The per-frame form's trace: how much Consistency an isotropic unit of coefficient error costs
    # at this frame. Exactly the per-frame term that `jacobian_form` averages into one M.
    sens = np.zeros(len(base))
    per_scalar = np.zeros((len(base), N_CONS))
    for j in range(N_CONS):
        per_scalar[:, j] = (W_CONS / N_CONS) / var[j] * (jac[:, j, :] ** 2).sum(axis=1)
    sens = per_scalar.sum(axis=1)

    df = pd.DataFrame({"shot": [index[i][0] for i in pick], "frame": [index[i][1] for i in pick],
                       "sens": sens, "good": good,
                       **{f"sens_{CONS_SCALARS[j]}": per_scalar[:, j] for j in range(N_CONS)}})
    df = df[df["good"]].drop(columns="good")
    df.to_csv(OUT, index=False)

    q = np.percentile(df["sens"], [1, 10, 50, 90, 99])
    print(f"\n  per-frame sensitivity: p1 {q[0]:.4g}  p10 {q[1]:.4g}  median {q[2]:.4g}  "
          f"p90 {q[3]:.4g}  p99 {q[4]:.4g}")
    print(f"  spread p99/median = {q[4] / q[2]:.1f}x, p90/p10 = {q[3] / q[1]:.1f}x — B6's own "
          f"refutation clause asked for more than 2x")

    costs = pd.read_csv(COSTS)
    m = df.merge(costs[["shot", "frame", "phase", "cost", "cost_cons"]], on=["shot", "frame"])
    print(f"\n  joined to {len(m)} of {len(df)} probed frames with a measured cost")
    if m.empty:
        raise SystemExit("no frames joined — was diagnose_frames.py run at the same --share?")

    print("\n  by decile of position within the shot (the axis C1 found the cost on):")
    m["decile"] = np.clip((m["phase"] * 10).astype(int), 0, 9)
    med = float(m["sens"].median())
    for key, r in m.groupby("decile").agg(sens=("sens", "median"),
                                          cost=("cost_cons", "mean")).iterrows():
        d = int(key)  # type: ignore[call-overload]
        print(f"      {d / 10:.1f}-{(d + 1) / 10:.1f}   sensitivity {r['sens'] / med:5.2f}x median"
              f"      mean cost {r['cost'] / m['cost_cons'].mean():5.2f}x mean")

    print("\n  THE TEST — mean Consistency cost by quintile of sensitivity:")
    m["sq"] = pd.qcut(m["sens"], 5, labels=False, duplicates="drop")
    overall = float(m["cost_cons"].mean())
    for key, r in m.groupby("sq").agg(sens=("sens", "median"), cost=("cost_cons", "mean"),
                                      n=("cost_cons", "size")).iterrows():
        print(f"      quintile {int(key) + 1}   sensitivity {r['sens'] / med:5.2f}x   "  # type: ignore[call-overload]
              f"mean cost {r['cost'] / overall:5.2f}x   ({int(r['n'])} frames)")
    rho = m[["sens", "cost_cons"]].corr(method="spearman").iloc[0, 1]
    print(f"\n  Spearman(sensitivity, cost) = {rho:+.3f}")
    print("      strongly positive -> the expensive frames are the sensitive ones, and the flat "
          "loss under-charges them: B6 has its mechanism")
    print("      near zero -> the expensive frames are of ordinary sensitivity, so the tail is "
          "coefficient error and re-weighting buys nothing")
    print(f"\n  per frame written to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
