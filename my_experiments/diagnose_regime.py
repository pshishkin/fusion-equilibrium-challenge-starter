#!/usr/bin/env python3
"""
Diagnostic B11 in ideas.md — does the loss metric point somewhere ELSE on ramp-down?

    uv run python my_experiments/diagnose_regime.py --shots 40 --frames 90 --jobs 6

The `jacobian` loss metric is ONE matrix, `JᵀJ` averaged over 300 probe frames. C7 measured how its
TRACE varies frame to frame — 10x from median to p99 — and found that the variation does not
explain where the score is lost. What nobody has measured is whether `J` points in different
DIRECTIONS: a plasma on ramp-down has a different shape, so a different COMBINATION of coefficients
moves its `kappa`, not merely a larger one. Averaging `JᵀJ` blurs direction as well as scale, and
only scale has been tested.

So: partition frames into three regimes by position within the shot, build `M` on each, and compare
the subspaces. Two numbers say it:

  * **principal angles** between the leading eigenspaces of two regimes' `M`. Near zero means the
    same directions matter and the regimes differ only in how much; large means they genuinely
    disagree about which coefficients the functionals read.
  * **the relative Frobenius distance** between the normalised forms, which is the same question
    without picking a rank.

If the directions agree, B11 dies here and the entry costs nothing further. If they disagree, a
regime-conditional loss is worth the rework — and that rework is real: `TargetScaler` folds `L` into
the target transform once, globally, so a per-row form has to move into the MLP loss instead.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import numpy.typing as npt
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from experiments import (  # noqa: E402
    DEFAULT_LOCAL_DATA_DIR,
    EFIT_GRID_SIZE,
    HF_TRAIN_CONFIG,
    _as_psirz_stack,
)
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

sys.path.insert(0, str(HERE.parent / "fusion_scoring"))
from common import N_CONS, W_CONS  # noqa: E402

FloatArray = npt.NDArray[np.floating]

# Where a shot is in its life, as fractions of its own frame count. The ends are where C1 found the
# cost and C7 found the two opposite mechanisms; the middle is the flat-top that dominates the loss.
REGIMES = (("ramp-up", 0.00, 0.12), ("flat-top", 0.35, 0.65), ("ramp-down", 0.88, 1.00))


def _form(psi: FloatArray, images: FloatArray, delta: float, ctx: dict, jobs: int) -> FloatArray:
    """`M` for one regime: the same quantity `jacobian_form` builds, on these frames alone."""
    chunks = np.array_split(np.arange(len(psi)), max(1, min(jobs * 4, len(psi))))
    tasks = [(psi[c], images, delta, ctx, i) for i, c in enumerate(chunks) if len(c)]
    parts = list(pimap(_jacobian_task, tasks, jobs))
    base = np.concatenate([b for b, _, _ in parts])
    jac = np.concatenate([j for _, j, _ in parts])
    good = np.isfinite(jac).all(axis=(1, 2)) & np.isfinite(base).all(axis=1)
    base, jac = base[good], jac[good]
    var = base.var(axis=0)
    m = np.zeros((images.shape[0], images.shape[0]))
    for j in range(N_CONS):
        if var[j] > 0:
            jj = jac[:, j, :]
            m += (W_CONS / N_CONS) / var[j] * (jj.T @ jj) / len(jj)
    return m


def main() -> int:
    install_timestamps()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shots", type=int, default=40)
    ap.add_argument("--frames", type=int, default=90, help="probe frames PER REGIME")
    ap.add_argument("--rank", type=int, default=10, help="eigenspace rank for the principal angles")
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--local-data-dir", type=Path, default=DEFAULT_LOCAL_DATA_DIR)
    ap.add_argument("--config", default=HF_TRAIN_CONFIG)
    args = ap.parse_args()

    art = joblib.load(ARTIFACT)
    params = load_params()
    plan = art["coil"]
    files = take_share(sorted_shots(args.local_data_dir, args.config, int(art["split_salt"])),
                       0.01, "tail")[:args.shots]
    print(f"Building the loss metric separately on {len(REGIMES)} regimes, "
          f"{args.frames} frames each, from {len(files)} held-out shots")

    banks: dict[str, list[FloatArray]] = {name: [] for name, _, _ in REGIMES}
    resid = []
    for path in files:
        row = pd.read_parquet(path).iloc[0]
        # The parquet layout is an object array of object arrays; experiments.py owns the
        # one function that normalises it, and reimplementing it here is how the two drift.
        psi = _as_psirz_stack(row["efit_psirz"]).astype(np.float64)
        feats = features_for_row(row)
        coil = coil_flux(plan, feats).astype(np.float64) if plan["subtract"] else 0.0
        resid.append((psi - coil).reshape(len(psi), -1))
        n_frames = len(psi)
        for name, lo, hi in REGIMES:
            a = int(lo * n_frames)
            b = max(a + 1, int(hi * n_frames))
            banks[name].extend(psi[a:b])

    flat = np.concatenate(resid)
    psi_ss_tot = float(((flat - flat.mean()) ** 2).sum() / len(flat))
    delta = params.jacobian_delta * float(np.sqrt(psi_ss_tot / params.n_pca))
    images = np.asarray(art["pca"].pca.components_).reshape(params.n_pca, EFIT_GRID_SIZE, -1)
    ctx = scorer_context(plan["grid_R"], plan["grid_Z"], plan["machine"])
    jobs = resolve_jobs(args.jobs, args.frames)

    forms = {}
    for name, _, _ in REGIMES:
        pool = np.asarray(banks[name])
        pick = np.linspace(0, len(pool) - 1, min(args.frames, len(pool))).astype(int)
        print(f"  {name}: {len(pick)} of {len(pool)} frames")
        forms[name] = _form(pool[pick], images, delta, ctx, jobs)

    names = [n for n, _, _ in REGIMES]
    print(f"\n  principal angles between the top-{args.rank} eigenspaces, in degrees "
          f"(0 = the same directions matter):")
    bases = {}
    for nm in names:
        w, v = np.linalg.eigh(forms[nm])
        bases[nm] = v[:, np.argsort(-w)[:args.rank]]
    for i, first in enumerate(names):
        for second in names[i + 1:]:
            s = np.linalg.svd(bases[first].T @ bases[second], compute_uv=False)
            ang = np.degrees(np.arccos(np.clip(s, -1.0, 1.0)))
            print(f"      {first:>9s} vs {second:<9s}  median {np.median(ang):5.1f}   "
                  f"max {ang.max():5.1f}")

    print("\n  relative Frobenius distance between the NORMALISED forms "
          "(0 = identical up to a scale):")
    norm = {nm: forms[nm] / np.linalg.norm(forms[nm]) for nm in names}
    for i, first in enumerate(names):
        for second in names[i + 1:]:
            d = np.linalg.norm(norm[first] - norm[second])
            print(f"      {first:>9s} vs {second:<9s}  {d:6.3f}")

    print("\n  trace, i.e. the SCALE C7 already measured (for reference):")
    for nm in names:
        print(f"      {nm:>9s}  {np.trace(forms[nm]):.4g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
