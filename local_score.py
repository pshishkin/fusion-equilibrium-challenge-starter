#!/usr/bin/env python3
"""
Score your model locally, on held-out DIII-D training shots, with the real metric.

`validate_submission.py` checks that your file has the right *shape*. This scores it. Without it
you can compute R²ψ yourself easily enough, but `D_LCFS` and `Consistency` — 30% of the composite
— need contour extraction and the seven ψ-derived functionals, so most people fly blind on the
part of the score where a plausible-looking flux map and a good one come apart.

    S = 0.55*R2_psi  +  0.15*R2_{q95,betaN}  +  0.10*(1 - D_LCFS)  +  0.20*Consistency

`fusion_scoring/` holds the competition scorer's own modules, copied unmodified, so the
functionals here are byte-identical to the ones Codabench runs. What differs is the *data*: this
scores held-out training shots you provide, the leaderboard scores withheld test shots. Treat the
number as a faithful proxy for model selection, not a leaderboard prediction.

    # sanity-check the harness itself (should print S = 1.0 then S = 0.0)
    uv run python local_score.py --mode perfect --n-shots 3
    uv run python local_score.py --mode zeros   --n-shots 3

    # score your model, on shots it was not trained on
    uv run python local_score.py --n-shots 20 --skip 7000

    # or score a .npz you already built
    uv run python local_score.py --pred my_preds.npz --n-shots 20 --skip 7000

    # score from a downloaded copy of the dataset instead of streaming the Hub
    uv run python local_score.py --source local --n-shots 20 --skip 30

    # score several members of the zoo on the same shots, in one pass over the ground truth
    uv run python local_score.py --source local --n-shots 5 --models ridge catboost ensemble

By default it imports `your_model_predict` from submission_skeleton.py — the same function that
builds your real submission. Note the training rows it receives DO contain `efit_*` ground truth:
if your model reads those, this score is meaningless. Use only the inputs listed in that
function's docstring.

Memory: shots are held in RAM (~20 MB per DIII-D shot). Start around --n-shots 10-20.
"""
from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
# The vendored scorer uses flat imports (`from common import ...`) exactly as it does on the
# platform. Keeping the files byte-identical means syncing them is a plain copy, so they cannot
# quietly drift away from what actually scores your submission.
sys.path.insert(0, str(HERE / "fusion_scoring"))
sys.path.insert(0, str(HERE))

from common import (  # noqa: E402
    AXIS_SIGN,
    CONS_SCALARS,
    N_CONS,
    N_SCALARS,
    PSI_SIGNS,
    SCALARS,
    SCORING_VERSION,
)
from contour import symmetric_hausdorff  # noqa: E402
from derive import derive_frame  # noqa: E402
from lcfs import extract_lcfs, extract_lcfs_with_sign, major_radius  # noqa: E402
from metrics import Accum, finalize_machine  # noqa: E402

from my_experiments.parallel import pimap, resolve_jobs  # noqa: E402

REPO_ID = "Sophelio/fusion-equilibrium-challenge"
TRAIN_CONFIG = "diii_d_train"
# Downloaded copy of the dataset, laid out exactly like the Hub repo:
#   <DEFAULT_LOCAL_DATA_DIR>/data/<config>/*.parquet
# Same default as experiments.py, so `--source local` needs no extra flag on either script.
DEFAULT_LOCAL_DATA_DIR = HERE.parent / "downloaded_huggingface" / "hf_dataset"
MACHINE = "DIII-D"          # the only machine with released ground truth
FloatArray = npt.NDArray[np.floating[Any]]
# What build_reference() returns: per-frame LCFS contours, the seven derived scalars, the mask
# of which of them are defined, and the mean major radius the Hausdorff distance is scaled by.
Reference = tuple[list[Any], FloatArray, npt.NDArray[np.bool_], float]
N_POINTS = 512
N_ITER = 22
LI_IDX = CONS_SCALARS.index("li")


# --------------------------------------------------------------------------- ground truth

def _as_psi_stack(psirz_raw: Any) -> FloatArray:
    """Normalize efit_psirz to (T, 65, 65) whether it arrives as HF nested lists or as the
    object-array-of-object-arrays that pandas/pyarrow hands back for parquet. Same logic as
    experiments.py::_as_psirz_stack — np.asarray raises on the parquet layout rather than
    returning something with ndim != 3, so the fallback has to be reached via except."""
    try:
        psi = np.asarray(psirz_raw, dtype=np.float32)
        if psi.ndim == 3:
            return psi
    except (ValueError, TypeError):
        pass
    return np.stack(
        [np.stack([np.asarray(r, dtype=np.float32) for r in grid]) for grid in psirz_raw]
    )


def _shot_from_row(row: Any) -> dict:
    """Keep ψ and the two value scalars. Works for a streamed dict or a parquet row."""
    psi = _as_psi_stack(row["efit_psirz"])
    return {
        "row": row,
        "psi": psi,
        "q95": np.asarray(row["efit_q95"], dtype=np.float64),
        "betaN": np.asarray(row["efit_beta_n"], dtype=np.float64),
    }


def load_shots_streaming(n_shots: int, skip: int, config: str) -> list[dict]:
    """Stream `n_shots` DIII-D training shots from the Hub, after skipping `skip`."""
    from datasets import load_dataset
    ds = load_dataset(REPO_ID, config, split="train", streaming=True)
    out: list[dict] = []
    for i, row in enumerate(ds):
        if i < skip:
            continue
        if len(out) >= n_shots:
            break
        shot = _shot_from_row(row)
        out.append(shot)
        print(f"  loaded shot {i}  T={shot['psi'].shape[0]}")
    return out


def load_shots_local(n_shots: int, skip: int, local_data_dir: Path, config: str,
                     files: list[Path] | None = None) -> list[dict]:
    """Read shots from a downloaded copy of the dataset: <dir>/data/<config>/*.parquet.

    Files are taken in sorted order, so `--skip` is deterministic and reproducible — unlike
    the Hub stream, whose order is only as stable as the shard listing.

    `files` overrides the whole selection: score exactly these paths, in this order. That is how
    my_experiments/evaluate.py drives the scorer — its split is ordered by a hash of the shot id,
    not lexicographically, so index arithmetic here would pick the wrong shots.
    """
    import pandas as pd

    if files is not None:
        out = []
        bar = tqdm(files, desc="  reading shots", unit="shot")
        for path in bar:
            shot = _shot_from_row(pd.read_parquet(path).iloc[0])
            out.append(shot)
            bar.set_postfix(frames=sum(s["psi"].shape[0] for s in out))
        return out

    data_dir = Path(local_data_dir) / "data" / config
    files = sorted(data_dir.glob("*.parquet"))
    if not files:
        raise SystemExit(
            f"No parquet files found in {data_dir}. Point --local-data-dir at the downloaded "
            f"dataset root (the folder that contains a 'data/' directory), or "
            f"download shots with:\n"
            f"  hf download {REPO_ID} --repo-type dataset --local-dir {local_data_dir} "
            f'--include "data/{config}/*"'
        )
    print(f"  Local: {data_dir} ({len(files)} shots available)")
    if skip >= len(files):
        raise SystemExit(
            f"--skip {skip} is past the end: only {len(files)} shots in {data_dir}. "
            f"Download more, or lower --skip."
        )

    out = []
    bar = tqdm(files[skip:skip + n_shots], desc="  reading shots", unit="shot")
    for path in bar:
        shot = _shot_from_row(pd.read_parquet(path).iloc[0])
        out.append(shot)
        bar.set_postfix(frames=sum(s["psi"].shape[0] for s in out))
    return out


def load_shots(n_shots: int, skip: int, source: str = "hf",
               local_data_dir: Path = DEFAULT_LOCAL_DATA_DIR,
               config: str = TRAIN_CONFIG,
               files: list[Path] | None = None) -> list[dict]:
    if source == "local":
        out = load_shots_local(n_shots, skip, local_data_dir, config, files)
    else:
        out = load_shots_streaming(n_shots, skip, config)
    if not out:
        raise SystemExit(f"No shots loaded (skip={skip} past the end of the split?)")
    if files is None and len(out) < n_shots:
        print(f"  note: only {len(out)} shots available after skip={skip} "
              f"(asked for {n_shots})")
    return out


def build_reference(psi_gt: FloatArray, R: FloatArray, Z: FloatArray,
                    mask_coarse: npt.NDArray[np.bool_], mask_f: FloatArray) -> Reference:
    """The targets the scorer compares against: per-frame LCFS, the seven derived scalars, rgeo.

    This is what the organizers precompute into the reference bundle. Running the same
    `extract_lcfs` / `derive_frame` on the ground truth is what makes a perfect prediction score
    exactly 1 — both sides go through identical code.
    """
    T = psi_gt.shape[0]
    contours, rgeos = [], []
    cons = np.full((T, N_CONS), np.nan)
    for k in range(T):
        c = extract_lcfs(psi_gt[k], R, Z, MACHINE, mask_coarse, mask_f, n_points=N_POINTS)
        contours.append(c)
        if c is not None:
            rgeos.append(major_radius(c))
        vals = derive_frame(psi_gt[k], R, Z, MACHINE, mask_coarse, mask_f, contour=c)
        for j, name in enumerate(CONS_SCALARS):
            cons[k, j] = vals[name]
    # derive_frame already NaNs out physically implausible values, so finiteness is the mask.
    return contours, cons, np.isfinite(cons), (float(np.mean(rgeos)) if rgeos else float("nan"))


# --------------------------------------------------------------------------- predictions

def predict(shots: list[dict], mode: str, pred_npz: Path | None,
            model: str | None = None) -> list[dict]:
    """One prediction dict per shot. `model` names a member of the zoo (see --models)."""
    if pred_npz is not None:
        z = np.load(pred_npz, allow_pickle=False)
        return [{"psirz": z[f"shot_{i:04d}_psirz"].astype(np.float64),
                 "q95": z[f"shot_{i:04d}_q95"].astype(np.float64),
                 "betaN": z[f"shot_{i:04d}_betaN"].astype(np.float64)}
                for i in range(len(shots))]
    if mode == "perfect":       # harness self-check: must score S = 1.0
        return [{"psirz": s["psi"].astype(np.float64), "q95": s["q95"], "betaN": s["betaN"]}
                for s in shots]
    if mode == "zeros":         # harness self-check: must score S = 0.0
        return [{"psirz": np.zeros_like(s["psi"], dtype=np.float64),
                 "q95": np.zeros_like(s["q95"]), "betaN": np.zeros_like(s["betaN"])}
                for s in shots]
    from submission_skeleton import your_model_predict
    return [your_model_predict(s["row"], MACHINE, model)
            for s in tqdm(shots, desc="  predicting", unit="shot")]


# --------------------------------------------------------------------------- parallelism

def _ref_task(args: tuple) -> Reference:
    return build_reference(*args)


def _score_task(args: tuple) -> dict:
    return score_shot(*args)


def _map(fn: Callable[[tuple], Any], tasks: list[tuple], jobs: int, desc: str) -> list:
    """`fn` over `tasks` on `jobs` shared-pool processes, in the original order — see
    my_experiments/parallel.py for why order and the start method are not negotiable."""
    label = desc if jobs == 1 else f"{desc} x{jobs}"
    return list(tqdm(pimap(fn, tasks, jobs), total=len(tasks), desc=label, unit="shot"))


# --------------------------------------------------------------------------- scoring

def psi_residuals(psi_gt: FloatArray, pred: Any,
                  mean_psi: float) -> tuple[dict[int, float], int, bool]:
    """SS_res under both candidate global flux signs (see metrics.py on sign invariance)."""
    g = psi_gt.astype(np.float64)
    p = np.asarray(pred, dtype=np.float64)
    if p.shape != g.shape:
        r = g - mean_psi
        ss = float(np.sum(r * r))
        return {s: ss for s in PSI_SIGNS}, 0, True
    nf = ~np.isfinite(p)
    out = {}
    for s in PSI_SIGNS:
        q = np.where(nf, mean_psi, s * p) if nf.any() else s * p
        r = g - q
        out[s] = float(np.sum(r * r))
    return out, int(nf.sum()), False


def score_shot(gt: dict, ref: Reference, pred: dict, R: FloatArray, Z: FloatArray,
               mask_coarse: npt.NDArray[np.bool_], mask_f: FloatArray, psi_sign: int,
               means: tuple[float, FloatArray, FloatArray]) -> dict:
    """One shot's contribution. Mirrors the platform's per-shot worker."""
    mean_psi, mean_scal, mean_cons = means
    contours, cons_gt, cmask, rgeo = ref
    psi_gt = gt["psi"]
    T = psi_gt.shape[0]
    # Two different signs compose: submission -> stored convention (psi_sign), and stored
    # convention -> "axis is a maximum of phi" (AXIS_SIGN), which the extractors require.
    axis_sign = AXIS_SIGN[MACHINE] * psi_sign

    ss_res_psi, nonfinite, missing = psi_residuals(psi_gt, pred["psirz"], mean_psi)
    ss_tot_shot = float(np.sum((psi_gt.astype(np.float64) - mean_psi) ** 2))
    shot_r2 = {s: (1.0 - ss_res_psi[s] / ss_tot_shot if ss_tot_shot > 0 else 0.0)
               for s in PSI_SIGNS}

    ss_res_scal = np.zeros(N_SCALARS)
    for j, name in enumerate(SCALARS):
        g = np.asarray(gt[name], dtype=np.float64)
        defined = np.isfinite(g)
        if not defined.any():
            continue
        p = np.asarray(pred.get(name), dtype=np.float64) if pred.get(name) is not None else None
        if p is None or p.shape != (T,):
            ss_res_scal[j] = float(np.sum((g[defined] - mean_scal[j]) ** 2))
        else:
            pj = np.where(np.isfinite(p[defined]), p[defined], mean_scal[j])
            ss_res_scal[j] = float(np.sum((g[defined] - pj) ** 2))

    ss_res_cons = np.zeros(N_CONS)
    cons_frames = np.zeros(N_CONS, dtype=np.int64)
    cons_fails = np.zeros(N_CONS, dtype=np.int64)
    ds, n_fail, n_frames = [], 0, 0
    ref_dropped = sum(1 for c in contours if c is None)

    if missing:
        n_frames = T - ref_dropped
        n_fail = n_frames
        ds = [1.0] * n_frames
        for j in range(N_CONS):
            m = cmask[:, j]
            if m.any():
                ss_res_cons[j] = float(np.sum((cons_gt[m, j] - mean_cons[j]) ** 2))
                cons_frames[j] = cons_fails[j] = int(m.sum())
    else:
        psi_pred = np.asarray(pred["psirz"], dtype=np.float64)
        for k in range(T):
            ref_c, need = contours[k], cmask[k]
            if ref_c is None and not need.any():
                continue
            ex, _ = extract_lcfs_with_sign(psi_pred[k], R, Z, MACHINE, mask_coarse, mask_f,
                                           n_iter=N_ITER, axis_sign=axis_sign)
            if ref_c is not None:
                n_frames += 1
                if ex is None:
                    n_fail += 1
                    ds.append(1.0)
                else:
                    d = symmetric_hausdorff(ex, np.asarray(ref_c, dtype=np.float64))
                    ds.append(min(1.0, d / rgeo if rgeo > 0 else 1.0))
            if need.any():
                vals = derive_frame(psi_pred[k], R, Z, MACHINE, mask_coarse, mask_f, contour=ex,
                                    with_li=bool(need[LI_IDX]), axis_sign=axis_sign)
                for j in np.nonzero(need)[0].tolist():
                    v = vals[CONS_SCALARS[j]]
                    cons_frames[j] += 1
                    if not np.isfinite(v):
                        v = mean_cons[j]
                        cons_fails[j] += 1
                    dv = cons_gt[k, j] - v
                    ss_res_cons[j] += dv * dv

    return {
        "ss_res_psi": ss_res_psi, "psi_pixels": int(psi_gt.size), "psi_nonfinite": nonfinite,
        "ss_res_scal": ss_res_scal, "ss_res_cons": ss_res_cons,
        "cons_frames": cons_frames, "cons_fails": cons_fails,
        "shot_r2_psi": shot_r2, "shot_dlcfs": (float(np.mean(ds)) if ds else None),
        "lcfs_frames": n_frames, "lcfs_fails": n_fail, "ref_lcfs_dropped": ref_dropped,
        "missing": missing,
    }


def choose_psi_sign(shots: list[dict], preds: list[dict], mean_psi: float) -> int:
    """The flux sign is ONE bit for the whole set, not per shot — a submission must not be able to
    pick a favourable orientation shot by shot. Decided from the psi residuals, then imposed on
    every orientation-sensitive derivation."""
    totals = {s: 0.0 for s in PSI_SIGNS}
    for shot, p in zip(shots, preds, strict=True):
        rr, _, _ = psi_residuals(shot["psi"], p["psirz"], mean_psi)
        for sg in PSI_SIGNS:
            totals[sg] += rr[sg]
    return min(PSI_SIGNS, key=lambda sg: totals[sg])


def report(res: dict, n_shots: int, label: str | None) -> None:
    """The per-model block: composite, the four weighted terms, and where extraction failed."""
    title = f"  COMPOSITE S = {res['S']:.4f}      ({n_shots} held-out {MACHINE} shots)"
    print("\n" + "=" * 62)
    print(f"{title}{'' if label is None else f'   [{label}]'}")
    print("=" * 62)
    for name, key, w in [("R2_psi", "r2_psi", 0.55), ("R2_{q95,betaN}", "r2_qb", 0.15),
                         ("1 - D_LCFS", "dlcfs", 0.10), ("Consistency", "consistency", 0.20)]:
        v = (1.0 - min(1.0, res["dlcfs"])) if key == "dlcfs" else res[key]
        print(f"  {name:>16s}  {v:8.4f}   x {w:.2f}  =  {w * max(0.0, v):.4f}")
    print("\n  per-derived-scalar R2 (the Consistency term):")
    for k, v in res["r2_cons_each"].items():
        print(f"      {k:>8s}  {'  n/a' if v is None else f'{v:7.4f}'}")
    print(f"\n  LCFS extraction failed on {res['lcfs_fail_frac']:.1%} of frames; "
          f"derivations on {res['cons_fail_frac']:.1%}")
    print(f"  psi_sign = {res['psi_sign']:+d}")


def report_comparison(results: dict[str, dict], n_shots: int) -> None:
    """Every model side by side, best composite first. The point of scoring a zoo at all."""
    print("\n" + "=" * 78)
    print(f"  MODEL COMPARISON   ({n_shots} held-out {MACHINE} shots, same shots for every model)")
    print("=" * 78)
    print(f"  {'model':>16s}  {'S':>8s}  {'R2_psi':>8s}  {'R2_qb':>8s}  {'1-D_LCFS':>9s}  "
          f"{'Cons':>8s}")
    for name, res in sorted(results.items(), key=lambda kv: -kv[1]["S"]):
        print(f"  {name:>16s}  {res['S']:8.4f}  {res['r2_psi']:8.4f}  {res['r2_qb']:8.4f}  "
              f"{1.0 - min(1.0, res['dlcfs']):9.4f}  {res['consistency']:8.4f}")


def main(argv: list[str] | None = None) -> int:
    # argv is threaded through so my_experiments/evaluate.py can drive the scorer as a function
    # instead of reimplementing the metric — one definition of the score, one place to fix.
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-shots", type=int, default=10, help="held-out shots to score (default 10)")
    ap.add_argument("--skip", type=int, default=0,
                    help="skip this many train shots first — use it to score shots you did NOT "
                         "train on")
    ap.add_argument("--mode", choices=["model", "perfect", "zeros"], default="model",
                    help="perfect/zeros verify the harness itself (S must be 1.0 / 0.0)")
    ap.add_argument("--pred", type=Path, help="score this .npz instead of calling your model")
    ap.add_argument("--source", default="hf", choices=["hf", "local"],
                    help="'hf' streams the Hub, 'local' reads a downloaded copy of the dataset")
    ap.add_argument("--local-data-dir", default=str(DEFAULT_LOCAL_DATA_DIR), type=Path,
                    help="Root of the downloaded dataset (contains a 'data/' folder). "
                         "Used with --source local")
    ap.add_argument("--config", default=TRAIN_CONFIG,
                    help=f"Dataset config / split folder to load (default {TRAIN_CONFIG})")
    ap.add_argument("--files", nargs="+", type=Path,
                    help="score exactly these parquet files, in this order (overrides "
                         "--n-shots/--skip; used by my_experiments/evaluate.py)")
    ap.add_argument("--jobs", type=int, default=0,
                    help="worker processes for the per-shot half of scoring — LCFS extraction and "
                         "the seven functionals (default 0 = cores - 2). Results do not depend "
                         "on it")
    ap.add_argument("--models", nargs="+",
                    help="score these members of the zoo (names from params.yaml, plus "
                         "'ensemble') on the same shots, and print them side by side")
    args = ap.parse_args(argv)

    if args.models and args.mode != "model":
        raise SystemExit(f"--models is meaningless with --mode {args.mode}: perfect/zeros do not "
                         f"call your model at all")
    if args.models and args.pred:
        raise SystemExit("--models and --pred contradict each other: one scores the zoo, the "
                         "other scores a prebuilt .npz")

    mask = np.load(HERE / "fusion_scoring" / "masks" / "d3d_envelope.npz")
    R, Z = mask["grid_R"], mask["grid_Z"]
    mask_coarse = mask["mask_coarse"].astype(bool)
    mask_f = mask_coarse.astype(np.float64)

    print(f"Fusion Equilibrium Challenge — local scorer (metric v{SCORING_VERSION}, {MACHINE})")
    src_label = "local download" if args.source == "local" else "Hugging Face stream"
    if args.files:
        print(f"Loading {len(args.files)} training shots (explicit file list)...")
    else:
        print(f"Loading {args.n_shots} training shots (skip={args.skip}, {src_label})...")
    shots = load_shots(args.n_shots, args.skip, args.source, args.local_data_dir, args.config,
                       args.files)

    jobs = resolve_jobs(args.jobs, len(shots))
    print(f"Building reference targets from ground truth (LCFS + 7 derived scalars per frame), "
          f"{jobs} process(es)...")
    refs = _map(_ref_task, [(s["psi"], R, Z, mask_coarse, mask_f) for s in shots], jobs,
                "  references from ground truth")

    psi_sum, psi_sumsq, psi_n = 0.0, 0.0, 0.0
    scal_sum, scal_sumsq, scal_n = np.zeros(N_SCALARS), np.zeros(N_SCALARS), np.zeros(N_SCALARS)
    cons_sum, cons_sumsq, cons_n = np.zeros(N_CONS), np.zeros(N_CONS), np.zeros(N_CONS)
    for s, ref in zip(shots, refs, strict=True):
        g = s["psi"].astype(np.float64)
        fin = np.isfinite(g)
        psi_sum += float(g[fin].sum())
        psi_sumsq += float((g[fin] ** 2).sum())
        psi_n += int(fin.sum())
        for j, name in enumerate(SCALARS):
            v = np.asarray(s[name], dtype=np.float64)
            v = v[np.isfinite(v)]
            scal_sum[j] += v.sum()
            scal_sumsq[j] += (v ** 2).sum()
            scal_n[j] += v.size
        cons_gt, cmask = ref[1], ref[2]
        for j in range(N_CONS):
            v = cons_gt[cmask[:, j], j]
            cons_sum[j] += v.sum()
            cons_sumsq[j] += (v ** 2).sum()
            cons_n[j] += v.size

    ref_stats = {"psi_sum": psi_sum, "psi_sumsq": psi_sumsq, "psi_n": psi_n,
                 "scal_sum": scal_sum, "scal_sumsq": scal_sumsq, "scal_n": scal_n,
                 "cons_sum": cons_sum, "cons_sumsq": cons_sumsq, "cons_n": cons_n}
    mean_psi = psi_sum / psi_n if psi_n else 0.0
    means = (mean_psi,
             np.where(scal_n > 0, scal_sum / np.where(scal_n > 0, scal_n, 1), 0.0),
             np.where(cons_n > 0, cons_sum / np.where(cons_n > 0, cons_n, 1), 0.0))

    # One pass over the ground truth above, then one prediction+scoring pass per model. The
    # expensive half — LCFS extraction and the seven functionals on the ground truth — is shared.
    results: dict[str, dict] = {}
    for model in (args.models or [None]):
        label = model if model is not None else args.mode
        print(f"\nGenerating predictions (mode={args.mode}"
              f"{'' if model is None else f', model={model}'})...")
        preds = predict(shots, args.mode, args.pred, model)

        psi_sign = choose_psi_sign(shots, preds, mean_psi)
        if psi_sign < 0:
            print("  note: your flux is sign-inverted vs the DIII-D convention — normalized for "
                  "you, exactly as the leaderboard does")

        # Only the three arrays score_shot reads are shipped to the workers — `shot` also holds
        # the raw dataset row, tens of megabytes of columns nothing here touches.
        tasks = [({"psi": shot["psi"], "q95": shot["q95"], "betaN": shot["betaN"]},
                  ref, p, R, Z, mask_coarse, mask_f, psi_sign, means)
                 for shot, ref, p in zip(shots, refs, preds, strict=True)]
        acc = Accum()
        acc.psi_sign = psi_sign
        for part in _map(_score_task, tasks, jobs, "  scoring"):
            acc.add(part)

        results[label] = finalize_machine(acc, ref_stats)
        report(results[label], len(shots), model)

    if len(results) > 1:
        report_comparison(results, len(shots))

    if args.mode in ("perfect", "zeros"):
        want = 1.0 if args.mode == "perfect" else 0.0
        got = results[args.mode]["S"]
        ok = abs(got - want) < 1e-4
        print(f"\n  self-check ({args.mode}): S = {got:.6f}, want {want} -> "
              f"{'PASS' if ok else 'FAIL'}")
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
