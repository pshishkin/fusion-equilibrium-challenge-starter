#!/usr/bin/env python3
"""
D7 — is the last tenth of a shot a DIFFERENT function, or merely a harder one?

    uv run python my_experiments/diagnose_tail_expert.py --share 0.15 --jobs 20

C1 measured the last decile carrying 24.1% of the geometry cost over 10.2% of the frames. C7 then
said the frames there are not more SENSITIVE — their Jacobian is an ordinary 0.98x — and that the
cost is coefficient error, 2.22x the fold's. Seeds agree with each other on that error at +0.550, so
a majority of it is shared bias, which averaging cannot remove and a differently-parameterised model
can. That is A22's case for an expert fitted on the tail alone.

**But a tail-only fit beating a global fit on tail frames proves nothing by itself.** It could mean
the tail is a different function, or it could mean any narrow slice beats the global fit on its own
slice — specialisation on fewer, more homogeneous rows. Those two have opposite consequences and the
same signature, so the measurement here is a CONTRAST against a control slice of identical size:

    (tail-only - global, on held-out tail frames) - (mid-only - global, on held-out mid frames)

If the tail's advantage is no larger than the middle's, the tail is not special and A22's
under-representation branch closes. The two slices are a quarter of each shot's frames each, so the
row counts match by construction and "fewer rows" cannot explain a difference between them.

Ridge is the model deliberately: it is deterministic, it has no seed noise to swamp a small
contrast, and it has been a reliable read under feature changes in this fork (0.7651 -> 0.7139 when
Thomson came out, matching a real 0.0023 of S). It is also the one model whose specialisation cannot
be confused with a longer fit.

**And ridge alone is not enough here, which the second half of the script is about.** A19 arm 1
moved `ridge` by +0.0216, the largest move any change has made to the linear baseline in this fork,
and the MLP did not move at all. So the contrast above is a statement about what a LINEAR model
cannot represent. The part that decides is whether the fitted ensemble still carries the difference,
measured on its own residual over shots outside its fit.

The measure is the residual sum of squares on the artifact's own PCA coefficients, which is R2_psi
inside the basis by Parseval — the geometry terms are functionals of the same map, and nothing here
extracts a contour, so this reads the flux error the tail expert would have to fix first.
"""
from __future__ import annotations

import argparse
import sys
from itertools import pairwise
from pathlib import Path

import joblib
import numpy as np
import numpy.typing as npt
from sklearn.linear_model import Ridge

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from experiments import DEFAULT_LOCAL_DATA_DIR, HF_TRAIN_CONFIG  # noqa: E402
from my_experiments.baseline_model import (  # noqa: E402
    ARTIFACT,
    ENSEMBLE,
    _predict_targets,
    _read_shots,
    build_inputs,
    coil_flux,
    sorted_shots,
)
from my_experiments.progress import install_timestamps  # noqa: E402

FloatArray = npt.NDArray[np.floating]

# The two slices, as fractions of the way through a shot. A quarter each, so the fits see the same
# number of rows and the contrast cannot be read as a sample-size effect.
TAIL = (0.75, 1.01)
MID = (0.375, 0.625)


def phases(lengths: npt.NDArray[np.int64]) -> FloatArray:
    """Position in its own shot, 0 at the first frame and 1 at the last, per concatenated frame."""
    return np.concatenate([np.arange(n) / max(1, n - 1) for n in lengths])


def prepare(files: list[Path], art: dict, desc: str,
            jobs: int) -> tuple[FloatArray, FloatArray, FloatArray, npt.NDArray[np.int64]]:
    """(inputs, PCA coefficients, phase, shot lengths) for every frame, in the artifact's space."""
    feats, psi, _, lengths = _read_shots(files, desc, 1.0, jobs)
    plan = art["coil"]
    target = psi.astype(np.float64)
    if plan["subtract"]:
        target = target - coil_flux(plan, feats).astype(np.float64)
    Y = np.asarray(art["pca"].transform(target), dtype=np.float64)
    X = np.asarray(art["scaler"].transform(build_inputs(plan, feats)), dtype=np.float64)
    return X, Y, phases(lengths), lengths


def clock(p: FloatArray) -> FloatArray:
    """Where in its own shot a frame is, which no column of `features_for_row` currently says."""
    return np.column_stack([p, p ** 2, 1.0 - p, (1.0 - p) ** 2])


def main() -> int:
    install_timestamps()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--share", type=float, default=0.15,
                    help="share of shots to FIT on, from the head of the split (default 0.15)")
    ap.add_argument("--eval-share", type=float, default=0.03,
                    help="share of shots to evaluate on, the block right behind the fit")
    ap.add_argument("--alpha", type=float, default=1.0, help="ridge alpha, as params.yaml sets it")
    ap.add_argument("--model", default=ENSEMBLE,
                    help=f"which member of the zoo the second half reads the residual of "
                         f"(default {ENSEMBLE})")
    ap.add_argument("--jobs", type=int, default=0)
    ap.add_argument("--artifact", type=Path, default=ARTIFACT)
    ap.add_argument("--local-data-dir", type=Path, default=DEFAULT_LOCAL_DATA_DIR)
    ap.add_argument("--config", default=HF_TRAIN_CONFIG)
    args = ap.parse_args()

    art = joblib.load(args.artifact)
    ordered = sorted_shots(args.local_data_dir, args.config, int(art["split_salt"]))
    n_fit = round(len(ordered) * args.share)
    n_eval = round(len(ordered) * args.eval_share)
    fit_files, eval_files = ordered[:n_fit], ordered[n_fit:n_fit + n_eval]
    print(f"Fitting on {len(fit_files)} shots, evaluating on {len(eval_files)} behind them, "
          f"salt {art['split_salt']}, basis from {args.artifact.name}")

    Xf, Yf, pf, _ = prepare(fit_files, art, "fit", args.jobs)
    Xe, Ye, pe, len_e = prepare(eval_files, art, "eval", args.jobs)

    def band(p: FloatArray, lo_hi: tuple[float, float]) -> npt.NDArray[np.bool_]:
        lo, hi = lo_hi
        return (p >= lo) & (p < hi)

    slices = {"tail": TAIL, "mid": MID}
    fit_masks = {k: band(pf, v) for k, v in slices.items()}
    eval_masks = {k: band(pe, v) for k, v in slices.items()}
    print(f"\n  fit rows: {len(Xf)} total, "
          + ", ".join(f"{k} {int(m.sum())}" for k, m in fit_masks.items()))
    print(f"  eval rows: {len(Xe)} total, "
          + ", ".join(f"{k} {int(m.sum())}" for k, m in eval_masks.items()))

    fits = {"global": Ridge(alpha=args.alpha).fit(Xf, Yf)}
    for k, m in fit_masks.items():
        fits[k] = Ridge(alpha=args.alpha).fit(Xf[m], Yf[m])

    # R2 inside the basis, per evaluation slice: 1 - SS_res/SS_tot with SS_tot against the slice's
    # own mean coefficient vector, which is what the scorer's fold mean is inside this space.
    def r2(est: Ridge, m: npt.NDArray[np.bool_]) -> float:
        res = est.predict(Xe[m]) - Ye[m]
        tot = Ye[m] - Ye[m].mean(axis=0)
        return 1.0 - float((res ** 2).sum()) / float((tot ** 2).sum())

    # The ABSOLUTE error too, because R2 per slice divides by that slice's own spread and the two
    # slices need not have the same one. The score's R2_psi is pooled against the FOLD mean, so the
    # quantity that reaches the leaderboard is this sum, not the ratio above.
    def sse(est: Ridge, m: npt.NDArray[np.bool_]) -> float:
        return float(((est.predict(Xe[m]) - Ye[m]) ** 2).sum()) / int(m.sum())

    for k, m in eval_masks.items():
        spread = float(((Ye[m] - Ye[m].mean(axis=0)) ** 2).sum()) / int(m.sum())
        print(f"  {k:<5} slice spread about its own mean: {spread:10.4f} (Wb/rad)^2 per frame")

    print("\n  R2 on the artifact's 50 coefficients, held-out shots:")
    print(f"    {'fitted on':<10} {'-> tail frames':>16} {'-> mid frames':>16}"
          f"{'tail SSE/frame':>16} {'mid SSE/frame':>16}")
    table = {}
    for name, est in fits.items():
        row = {k: r2(est, m) for k, m in eval_masks.items()}
        table[name] = row
        print(f"    {name:<10} {row['tail']:>16.5f} {row['mid']:>16.5f}"
              f"{sse(est, eval_masks['tail']):>16.4f} {sse(est, eval_masks['mid']):>16.4f}")

    # The one that decides whether this needs a separate MODEL or one more COLUMN. Nothing in
    # `features_for_row` says where in the shot a frame is: every column is instantaneous, centred
    # or strictly causal. If the global fit recovers the tail once it is told, then A22's expert and
    # D9's frames-to-end column are the same finding and the column is the cheap half of it.
    Xfc, Xec = np.hstack([Xf, clock(pf)]), np.hstack([Xe, clock(pe)])
    told = Ridge(alpha=args.alpha).fit(Xfc, Yf)

    def r2c(m: npt.NDArray[np.bool_]) -> float:
        res = told.predict(Xec[m]) - Ye[m]
        tot = Ye[m] - Ye[m].mean(axis=0)
        return 1.0 - float((res ** 2).sum()) / float((tot ** 2).sum())

    print(f"    {'global+clock':<10} {r2c(eval_masks['tail']):>14.5f} "
          f"{r2c(eval_masks['mid']):>16.5f}")
    recovered = ((r2c(eval_masks["tail"]) - table["global"]["tail"])
                 / max(1e-12, table["tail"]["tail"] - table["global"]["tail"]))
    print(f"\n  four clock columns recover {recovered:.1%} of what a separate tail fit buys")

    # ---- the part that decides it, because ridge has produced this exact false positive before.
    # A19 arm 1 moved `ridge` +0.0216, the largest move any change has made to the linear baseline
    # in this fork, and the MLP was flat. So everything above is a statement about a LINEAR model,
    # and the question is whether the fitted ensemble -- 512x512, four seeds and a bidirectional
    # GRU, on features that partly encode position through the vessel integrals -- still carries
    # position-dependent error the clock can remove.
    #
    # Fitted on half the EVALUATION shots and read on the other half: those shots are outside the
    # ensemble's own fit, so its residual on them is honest, and splitting by shot keeps the two
    # halves independent.
    print("\n  ---- and now against the ENSEMBLE's own residual, which is what actually ships")
    n_pca = int(art["n_pca"])
    # One shot at a time: the sequence member reads a whole discharge and refuses a concatenation
    # rather than silently treating several shots as one, which is the right refusal.
    bounds = np.concatenate([[0], np.cumsum(len_e)])
    ens = np.concatenate([_predict_targets(art, args.model, Xe[a:b])[:, :n_pca]
                          for a, b in pairwise(bounds)])
    resid = ens - Ye
    half = int(np.searchsorted(np.cumsum(len_e), len(Xe) // 2))
    cut = int(np.cumsum(len_e)[half])
    A = np.zeros(len(Xe), dtype=bool)
    A[:cut] = True
    B = ~A
    print(f"  {int(A.sum())} frames to fit the correction, {int(B.sum())} to read it on, "
          f"split at a shot boundary")

    for k, m in eval_masks.items():
        per = float((resid[m] ** 2).sum()) / int(m.sum())
        print(f"  ensemble SSE/frame on {k:<5}: {per:8.4f}")

    # Two corrections, and the CONTRAST between them is the answer. Without the clock, a refit of
    # the residual on the same 84 columns measures only what a second linear pass over features the
    # ensemble already saw can scrape off. With it, the difference is what POSITION adds.
    removed = {}
    for name, Xb in (("features only", Xe), ("features + clock", Xec)):
        corr = Ridge(alpha=args.alpha).fit(Xb[A], resid[A])
        for k in ("tail", "mid"):
            m = B & eval_masks[k]
            before = float((resid[m] ** 2).sum()) / int(m.sum())
            after = float(((resid[m] - corr.predict(Xb[m])) ** 2).sum()) / int(m.sum())
            removed[(name, k)] = (before - after) / before
            print(f"    {name:<17} on {k:<5}: {before:8.4f} -> {after:8.4f} "
                  f"({removed[(name, k)]:+.1%})")

    tail_gain = table["tail"]["tail"] - table["global"]["tail"]
    mid_gain = table["mid"]["mid"] - table["global"]["mid"]
    print(f"\n  ridge's specialisation gain on its OWN slice: tail {tail_gain:+.5f}, "
          f"mid {mid_gain:+.5f}; THE CONTRAST (tail - mid) = {tail_gain - mid_gain:+.5f}")
    clock_adds = removed[("features + clock", "tail")] - removed[("features only", "tail")]
    print(f"  the ensemble's tail error the clock removes, over features alone: {clock_adds:+.2%}")

    # The two halves can disagree, and when they do the ensemble wins: it is what ships. A ridge
    # gate on this feature family has already produced one false positive in this fork (A19 arm 1
    # moved `ridge` +0.0216 and the MLP not at all), so the linear contrast is a statement about
    # what a LINEAR model cannot represent, not about what is missing from the problem.
    if tail_gain - mid_gain <= 0:
        print("\n  Even the linear read says the tail is not a different function: its expert "
              "gains no more on the tail than the middle's does on the middle, so what a "
              "tail-only fit buys is specialisation on a narrower slice. A22's "
              "under-representation branch closes.")
    elif clock_adds < 0.02:
        print("\n  The two halves DISAGREE, and the ensemble is the half that ships. Ridge gains "
              "sharply from a tail-only fit, so the tail is a different LINEAR function — but the "
              "fitted ensemble's own tail error is barely above its mid-shot error and the clock "
              "removes none of it, which says the trained model has already absorbed the position "
              "dependence a linear model cannot represent. That is A19's vessel integrals doing "
              "their job. A22's under-representation branch and D9's clock close together.")
    else:
        print("\n  Both halves agree: the tail is a different function AND the ensemble still "
              "carries the difference. Build it as a REPLACEMENT feature set, never a fourth "
              "member, and gate it on the last decile's cost in diagnose_frames.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
