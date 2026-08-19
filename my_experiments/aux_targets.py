#!/usr/bin/env python3
"""
A9 — the seven scored functionals, precomputed per frame, so the net can be taught them directly.

    uv run python my_experiments/aux_targets.py --share 0.80/1.0 --val-share 0.19/1.0 --jobs 10

The models predict 50 flux coefficients plus q95 and betaN, and the seven CONSISTENCY scalars —
R_axis, Z_axis, kappa, tri_top, tri_bot, volume, li — are never targets. The scorer derives them
from the submitted map, so they cannot be submitted directly; but nothing stops them being
AUXILIARY outputs during training, discarded at inference. C1 measured that kappa and li alone are
42% of the geometry cost, so this is the one untried change aimed straight at where the score is.

Why a separate file and not the shot cache. Each frame costs one `extract_lcfs` and seven
derivations — about 31 ms — so the full training split is roughly 13 core-hours, and paying that on
every run would make the arm unaffordable. Putting it in `shot_cache` instead would rebuild 27 GB
and invalidate every other run in flight. This writes one npz keyed by the exact file list and
share, so a rerun with the same split is free and a different split refuses to use it.

The values come from the SAME `extract_lcfs` and `derive_frame` the scorer runs, on the ground-truth
map, so what the net is taught is exactly what it will be measured on.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import numpy.typing as npt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "fusion_scoring"))

from common import CONS_SCALARS, N_CONS  # noqa: E402
from derive import derive_frame  # noqa: E402
from lcfs import extract_lcfs  # noqa: E402

from experiments import DEFAULT_LOCAL_DATA_DIR, HF_TRAIN_CONFIG  # noqa: E402
from my_experiments.target_metric import scorer_context  # noqa: E402
from toolkit.parallel import pimap, resolve_jobs  # noqa: E402
from toolkit.progress import SHOT_EVERY, bar_kwargs, install_timestamps  # noqa: E402

# This file is DIII-D's, like everything else in `my_experiments/`, so it names the machine
# itself rather than borrowing a constant from the scorer — `local_score` now reads the
# machine off the shots it is given, because hardcoding it there scored MAST on the wrong grid.
D3D = "DIII-D"

FloatArray = npt.NDArray[np.floating]
CACHE = HERE.parent / ".aux_targets"
LI = CONS_SCALARS.index("li")


def _task(args: tuple) -> FloatArray:
    """(T, 7) the seven derived scalars of one shot's GROUND-TRUTH maps, nan where undefined."""
    path, ctx = args
    import pandas as pd

    import local_score
    from experiments import _as_psirz_stack
    psi = _as_psirz_stack(pd.read_parquet(path).iloc[0]["efit_psirz"]).astype(np.float64)
    out = np.full((len(psi), N_CONS), np.nan)
    for k in range(len(psi)):
        c = extract_lcfs(psi[k], ctx["grid_R"], ctx["grid_Z"], ctx["machine"],
                         ctx["mask_coarse"], ctx["mask_f"], n_points=local_score.N_POINTS)
        vals = derive_frame(psi[k], ctx["grid_R"], ctx["grid_Z"], ctx["machine"],
                            ctx["mask_coarse"], ctx["mask_f"], contour=c, with_li=True)
        for j, name in enumerate(CONS_SCALARS):
            out[k, j] = vals[name]
    return out


def cache_path(files: list[Path]) -> Path:
    """Keyed by the exact ordered file list, so a different split cannot silently reuse this."""
    h = hashlib.sha1("|".join(p.name for p in files).encode()).hexdigest()[:16]
    return CACHE / f"cons_{len(files)}_{h}.npz"


def load(files: list[Path]) -> FloatArray | None:
    p = cache_path(files)
    if not p.exists():
        return None
    with np.load(p) as z:
        return np.asarray(z["cons"])


def main() -> int:
    install_timestamps()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--share", default="0.80/1.0")
    ap.add_argument("--val-share", default="0.19/1.0")
    ap.add_argument("--salt", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=10)
    ap.add_argument("--local-data-dir", type=Path, default=DEFAULT_LOCAL_DATA_DIR)
    ap.add_argument("--config", default=HF_TRAIN_CONFIG)
    args = ap.parse_args()

    # Imported HERE, not at module scope: `baseline_model` imports this module for `load`, and a
    # top-level import back into it is a cycle. It broke four queued runs at import and, worse, the
    # POOL WORKERS of a run whose parent had already imported cleanly — a worker re-imports the
    # main module from disk, so editing a module mid-flight breaks a job that looked safe.
    from my_experiments.baseline_model import parse_share, sorted_shots, split_train_val

    shot_share, frame_share = parse_share(args.share)
    val_shot_share, val_frame_share = parse_share(args.val_share)
    if frame_share < 1.0 or val_frame_share < 1.0:
        raise SystemExit("the aux targets are stored per FRAME, so both shares must be /1.0 — "
                         "thinning happens where the training set is built, not here")
    files = sorted_shots(args.local_data_dir, args.config, args.salt)
    train, val = split_train_val(files, shot_share, val_shot_share)

    ctx = None
    CACHE.mkdir(exist_ok=True)
    for name, block in (("train", train), ("val", val)):
        out = cache_path(block)
        if out.exists():
            print(f"  {name}: {out.name} already there, skipping")
            continue
        if ctx is None:
            mask = np.load(HERE.parent / "fusion_scoring" / "masks" / "d3d_envelope.npz")
            ctx = scorer_context(mask["grid_R"], mask["grid_Z"], D3D)
        jobs = resolve_jobs(args.jobs, len(block))
        print(f"  {name}: {len(block)} shots on {jobs} process(es) -> {out.name}")
        parts = []
        from tqdm import tqdm
        for r in tqdm(pimap(_task, [(p, ctx) for p in block], jobs), total=len(block),
                      unit="shot", desc=f"  {name}", **bar_kwargs(SHOT_EVERY)):
            parts.append(r)
        cons = np.concatenate(parts)
        np.savez_compressed(out, cons=cons.astype(np.float32))
        finite = np.isfinite(cons).mean(axis=0)
        print(f"    {len(cons)} frames; defined per scalar: "
              + ", ".join(f"{n} {f:.1%}" for n, f in zip(CONS_SCALARS, finite, strict=True)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
