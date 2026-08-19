#!/usr/bin/env python3
"""
Diagnostic C1 in ideas.md — where the score is actually lost, frame by frame.

    uv run python my_experiments/diagnose_frames.py --share 0.01 --jobs 24

`evaluate.py` reports four numbers. This reports the same score decomposed onto every scored
FRAME, on one common scale: how much of the composite S that frame costs. Both terms below are
sums over frames, so the decomposition is exact rather than an approximation — but they are NOT
weighted alike, and assuming they were is a mistake this file made once:

    D_LCFS       0.10 * d_k / (n_shots * frames scored in THIS shot)   <- macro, per shot
    Consistency  0.20/7 * res_kj^2 / SS_tot_j over the seven scalars   <- pooled, per frame

Consistency pools one R2 per scalar over every frame of every shot. D_LCFS does not: metrics.py
averages each SHOT's own mean distance, so a frame of a short shot weighs many times a frame of a
long one. Pooling it instead put the term 0.000034 of S out.

Only the two GEOMETRY terms, which is not a shortcut: C2 measured R2_psi at 0.9998 against a
ceiling of 1.0000, so the flux itself is finished and everything reachable is in these two.

Two questions it answers that the pooled table cannot. Whether the loss is spread evenly over the
frames — in which case only a better regression helps — or concentrated in a few percent of them,
which would make what those frames have in common the whole problem. And, if it is concentrated,
WHAT they have in common: the covariates are written out beside the costs, so the follow-up is a
groupby rather than another run.

Writes results/frame_costs.csv (one row per frame) and prints the summary.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import numpy.typing as npt
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
# The vendored scorer uses flat imports, exactly as it does on the platform — see local_score.py.
sys.path.insert(0, str(HERE.parent / "fusion_scoring"))

from common import (  # noqa: E402
    AXIS_SIGN,
    CONS_SCALARS,
    N_CONS,
    W_CONS,
    W_LCFS,
)
from contour import symmetric_hausdorff  # noqa: E402
from derive import derive_frame  # noqa: E402
from lcfs import extract_lcfs_with_sign  # noqa: E402

import local_score  # noqa: E402
from experiments import (  # noqa: E402
    D3D_MAGNETICS_SIGNALS,
    DEFAULT_LOCAL_DATA_DIR,
    HF_TRAIN_CONFIG,
)
from my_experiments.baseline_model import (  # noqa: E402
    ARTIFACT,
    ENSEMBLE,
    FRAME_STEP_MS,
    features_for_row,
    predict_row,
    project_through_basis,
    sorted_shots,
    take_share,
)
from toolkit.parallel import resolve_jobs  # noqa: E402
from toolkit.progress import install_timestamps  # noqa: E402

# This file is DIII-D's, like everything else in `my_experiments/`, so it names the machine
# itself rather than borrowing a constant from the scorer — `local_score` now reads the
# machine off the shots it is given, because hardcoding it there scored MAST on the wrong grid.
D3D = "DIII-D"

OUT = HERE.parent / "results" / "frame_costs.csv"

# Not a member of the zoo: the ceiling, ground truth pushed through the artifact's own
# 50 components. Same name as evaluate.py's --mode, and the same function behind it.
BASIS = "basis"

# `features_for_row` returns the level block first, in D3D_MAGNETICS_SIGNALS order, so the plasma
# current is addressed by its index there.
IP = D3D_MAGNETICS_SIGNALS.index("plasma_current")
LI = CONS_SCALARS.index("li")


def _frame_task(args: tuple) -> dict[str, Any]:
    """One shot: the per-frame LCFS distance and the per-frame residual of each derived scalar.

    Lifted from `local_score.score_shot`, which computes exactly these and then sums them away.
    The flux sign is carried in `axis_sign`, as it is there — the predicted map itself is NOT
    pre-multiplied, or the extraction would see one convention and the residual another.
    """
    psi_pred, ref, R, Z, mask_coarse, mask_f, axis_sign = args
    contours, cons_gt, cmask, rgeo = ref
    T = psi_pred.shape[0]
    d: npt.NDArray[np.float64] = np.full(T, np.nan)
    res: npt.NDArray[np.float64] = np.full((T, N_CONS), np.nan)
    for k in range(T):
        ref_c, need = contours[k], cmask[k]
        if ref_c is None and not need.any():
            continue
        ex, _ = extract_lcfs_with_sign(psi_pred[k], R, Z, D3D, mask_coarse, mask_f,
                                       n_iter=local_score.N_ITER, axis_sign=axis_sign)
        if ref_c is not None:
            if ex is None or not rgeo > 0:
                d[k] = 1.0
            else:
                dist = symmetric_hausdorff(ex, np.asarray(ref_c, dtype=np.float64))
                d[k] = min(1.0, dist / rgeo)
        if need.any():
            vals = derive_frame(psi_pred[k], R, Z, D3D, mask_coarse, mask_f,
                                contour=ex, with_li=bool(need[LI]), axis_sign=axis_sign)
            for j in np.nonzero(need)[0].tolist():
                v = vals[CONS_SCALARS[j]]
                res[k, j] = cons_gt[k, j] - v if np.isfinite(v) else np.nan
    return {"d": d, "res": res, "cons_gt": cons_gt, "cmask": cmask}


def main() -> int:
    install_timestamps()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--share", type=float, default=0.01,
                    help="share of shots to diagnose, from the tail of the list (default 0.01)")
    ap.add_argument("--model", default=ENSEMBLE,
                    help=f"which member of the zoo to diagnose (default {ENSEMBLE}), or 'basis' "
                         f"for the CEILING — ground truth through the artifact's own components, "
                         f"which decomposes the representation floor onto the same frames")
    ap.add_argument("--jobs", type=int, default=0)
    ap.add_argument("--out", type=Path, default=OUT,
                    help=f"where to write the per-frame table (default {OUT.name}). Give it a "
                         f"name when comparing two models, or the second run overwrites the first")
    ap.add_argument("--local-data-dir", type=Path, default=DEFAULT_LOCAL_DATA_DIR)
    ap.add_argument("--config", default=HF_TRAIN_CONFIG)
    args = ap.parse_args()

    if not ARTIFACT.exists():
        raise SystemExit(f"{ARTIFACT} not found — train first")
    art = joblib.load(ARTIFACT)
    files = take_share(sorted_shots(args.local_data_dir, args.config, int(art["split_salt"])),
                       args.share, "tail")
    overlap = {p.name for p in files} & (set(art["train_files"]) | set(art["val_files"]))
    if overlap:
        raise SystemExit(f"{len(overlap)} of the {len(files)} shots were fitted on; lower --share")
    print(f"Diagnosing {args.model} on {len(files)} held-out shots from {ARTIFACT.name}")

    mask = np.load(HERE.parent / "fusion_scoring" / "masks" / "d3d_envelope.npz")
    R, Z = mask["grid_R"], mask["grid_Z"]
    mask_coarse = mask["mask_coarse"].astype(bool)
    mask_f = mask_coarse.astype(np.float64)

    shots = local_score.load_shots(0, 0, "local", args.local_data_dir, args.config, files)
    jobs = resolve_jobs(args.jobs, len(shots))
    refs = local_score._map(local_score._ref_task,
                            [(s["psi"], R, Z, mask_coarse, mask_f) for s in shots], jobs,
                            "  references from ground truth")
    if args.model == BASIS:
        preds = [{"psirz": project_through_basis(s["row"], s["psi"])} for s in shots]
    else:
        preds = [predict_row(s["row"], "DIII-D", args.model) for s in shots]

    # The flux sign is ONE bit for the whole fold, decided as the scorer decides it.
    psi_all = np.concatenate([s["psi"].astype(np.float64).ravel() for s in shots])
    mean_psi = float(psi_all[np.isfinite(psi_all)].mean())
    psi_sign = local_score.choose_psi_sign(shots, preds, mean_psi)
    print(f"  psi_sign = {psi_sign:+g}")
    axis_sign = AXIS_SIGN[D3D] * psi_sign

    out = local_score._map(
        _frame_task,
        [(np.asarray(p["psirz"], dtype=np.float64), ref, R, Z, mask_coarse, mask_f, axis_sign)
         for p, ref in zip(preds, refs, strict=True)],
        jobs, "  per-frame costs")

    # The denominators are the metric's own, so what is written below sums to the reported terms
    # rather than merely resembling them. THE TWO TERMS ARE NOT WEIGHTED THE SAME WAY, and this
    # was wrong here at first — read `fusion_scoring/metrics.py`, not the shape of the numbers:
    #
    #   Consistency  POOLED. One R2 per scalar over every frame of every shot, against the fold
    #                mean. Every frame counts once, so a frame's share is res^2 / SS_tot exactly.
    #   D_LCFS       MACRO. `metrics.py:114` is np.mean(acc.shot_dlcfs) — the mean over SHOTS of
    #                each shot's own mean over frames. Every SHOT counts once, so a frame of a
    #                9-frame shot carries 41x the weight of a frame of a 372-frame one at the
    #                extremes of this fold. Pooling it instead put the term 0.000034 of S out.
    n_shots_with_lcfs = sum(1 for o in out if np.isfinite(o["d"]).any())
    cons_ss_tot = np.zeros(N_CONS)
    for j in range(N_CONS):
        vals = np.concatenate([o["cons_gt"][o["cmask"][:, j], j] for o in out])
        cons_ss_tot[j] = float(np.sum((vals - vals.mean()) ** 2))

    rows = []
    for path, s, o in zip(files, shots, out, strict=True):
        t = np.asarray(s["row"]["efit_times"], dtype=np.float64).ravel()
        T = len(t)
        ip = features_for_row(s["row"])[:, IP]
        dip = np.gradient(ip, t) if T > 1 else np.zeros(T)
        cost_cons = np.zeros(T)
        for j in range(N_CONS):
            m = o["cmask"][:, j] & np.isfinite(o["res"][:, j])
            if cons_ss_tot[j] > 0:
                cost_cons[m] += (W_CONS / N_CONS) * o["res"][m, j] ** 2 / cons_ss_tot[j]
        # 1/(shots x this shot's own scored frames), which is what the macro mean above expands to.
        scored = int(np.isfinite(o["d"]).sum())
        cost_lcfs = np.where(np.isfinite(o["d"]),
                             W_LCFS * np.nan_to_num(o["d"]) / max(1, scored * n_shots_with_lcfs),
                             0.0)
        rows.append(pd.DataFrame({
            "shot": path.stem, "frame": np.arange(T), "n_frames": T,
            "phase": np.arange(T) / max(1, T - 1), "time_ms": t,
            "ip": ip, "abs_dip": np.abs(dip),
            "gap": np.concatenate([[1.0], np.diff(t) / FRAME_STEP_MS]),
            "d_lcfs": o["d"], "cost_lcfs": cost_lcfs, "cost_cons": cost_cons,
            "cost": cost_lcfs + cost_cons,
            **{f"res_{CONS_SCALARS[j]}": o["res"][:, j] for j in range(N_CONS)},
            # The ground-truth scalars, which `_task` already computes and used to throw away. They
            # cost nothing here and they turn every later question about the READOUT -- would a
            # better estimate of these seven numbers score better than reading them off the map --
            # into an offline sweep over this CSV instead of a scoring pass.
            **{f"gt_{CONS_SCALARS[j]}": o["cons_gt"][:, j] for j in range(N_CONS)},
        }))
    df = pd.concat(rows, ignore_index=True)
    args.out.parent.mkdir(exist_ok=True)
    df.to_csv(args.out, index=False)

    total = float(df["cost"].sum())
    print(f"\n{len(df)} frames. The two geometry terms cost {total:.4f} of S in total "
          f"({df['cost_lcfs'].sum():.4f} LCFS + {df['cost_cons'].sum():.4f} Consistency).")

    print("\n  concentration — share of that cost carried by the worst frames:")
    order = np.sort(df["cost"].to_numpy())[::-1]
    for q in (0.001, 0.01, 0.05, 0.10, 0.25, 0.50):
        k = max(1, round(q * len(order)))
        print(f"      worst {q:6.1%} ({k:5d} frames)   {order[:k].sum() / total:6.1%} of the cost"
              f"   (even would be {q:6.1%})")

    print("\n  by position within the shot:")
    df["decile"] = np.clip((df["phase"] * 10).astype(int), 0, 9)
    for key, r in df.groupby("decile")["cost"].agg(["sum", "count"]).iterrows():
        decile = int(key)  # type: ignore[call-overload]
        print(f"      {decile / 10:.1f}-{(decile + 1) / 10:.1f}   "
              f"{r['sum'] / total:6.1%} of the cost over {r['count'] / len(df):6.1%} of the frames")

    print("\n  by frame gap (1 = the ordinary 20 ms step):")
    for lo, hi, label in [(0.0, 1.5, "1 (no hole)"), (1.5, 2.5, "2"), (2.5, 1e9, "3 or more")]:
        m = (df["gap"] >= lo) & (df["gap"] < hi)
        if m.any():
            print(f"      {label:>12s}   {df.loc[m, 'cost'].sum() / total:6.1%} of the cost over "
                  f"{m.mean():6.1%} of the frames")

    print("\n  by |dIp/dt| quartile (ramps against flat-top):")
    df["ipq"] = pd.qcut(df["abs_dip"], 4, labels=False, duplicates="drop")
    for key, r in df.groupby("ipq")["cost"].agg(["sum", "count"]).iterrows():
        print(f"      quartile {int(key) + 1}   {r['sum'] / total:6.1%} of the cost over "  # type: ignore[call-overload]
              f"{r['count'] / len(df):6.1%} of the frames")

    print("\n  which scalar carries the Consistency loss:")
    for j in range(N_CONS):
        col = df[f"res_{CONS_SCALARS[j]}"].to_numpy()
        fin = np.isfinite(col)
        share = ((W_CONS / N_CONS) * np.sum(col[fin] ** 2) / cons_ss_tot[j]) / total
        print(f"      {CONS_SCALARS[j]:>8s}   {share:6.1%} of the cost")

    print(f"\n  per frame written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
