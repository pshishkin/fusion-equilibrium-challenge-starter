#!/usr/bin/env python3
"""
A9 — the seven scored functionals, precomputed per frame, so the net can be taught them directly.

    uv run python my_experiments/aux_targets.py --share 0.80/1.0 --val-share 0.19/1.0 --jobs 10

The models predict 50 flux coefficients plus q95 and betaN, and the seven CONSISTENCY scalars —
R_axis, Z_axis, kappa, tri_top, tri_bot, volume, li — are never targets. The scorer derives them
from the submitted map, so they cannot be submitted directly; but nothing stops them being
AUXILIARY outputs during training, discarded at inference. The current configuration also uses
these functionals for the static consistency head; see experiments_history.md for measurements.

Why a separate file and not the shot cache. Each frame costs one `extract_lcfs` and seven
derivations — about 31 ms — so the full training split is roughly 13 core-hours, and paying that on
every run would make the arm unaffordable. Putting it in `shot_cache` instead would rebuild 27 GB
and invalidate every other run in flight. This writes one npz keyed by the ordered input identities
and producer code. Full-frame lengths let training apply its own per-shot frame selection.

The values come from the SAME `extract_lcfs` and `derive_frame` the scorer runs, on the ground-truth
map, so what the net is taught is exactly what it will be measured on.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import tempfile
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
from toolkit.records import file_hash  # noqa: E402

# This file is DIII-D's, like everything else in `my_experiments/`, so it names the machine
# itself rather than borrowing a constant from the scorer — `local_score` now reads the
# machine off the shots it is given, because hardcoding it there scored MAST on the wrong grid.
D3D = "DIII-D"

FloatArray = npt.NDArray[np.floating]
CACHE = Path(os.environ.get("FUSION_AUX_CACHE", HERE.parent / ".aux_targets"))
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
    """Versioned producer/data identity; shared by immutable experiment snapshots."""
    h = hashlib.sha256(f"full-frames-with-lengths-v2|{np.__version__}".encode())
    producers = [Path(__file__), HERE / "target_metric.py", HERE.parent / "local_score.py",
                 HERE.parent / "experiments.py", HERE.parent / "uv.lock"]
    producers += sorted((HERE.parent / "fusion_scoring").glob("*.py"))
    producers += sorted((HERE.parent / "fusion_scoring/masks").glob("*.npz"))
    for path in producers:
        h.update(file_hash(path).encode())
    for path in files:
        stat = path.stat()
        h.update(f"{path.name}:{stat.st_size}:{stat.st_mtime_ns}|".encode())
    return CACHE / f"cons_{len(files)}_{h.hexdigest()[:24]}.npz"


def load(files: list[Path], frame_share: float = 1.0) -> FloatArray | None:
    p = cache_path(files)
    if not p.exists():
        return None
    with np.load(p) as z:
        cons = np.asarray(z["cons"])
        lengths = np.asarray(z["lengths"], dtype=np.int64)
    if len(lengths) != len(files) or lengths.sum() != len(cons) or (lengths <= 0).any():
        raise ValueError(f"{p}: lengths {len(lengths)}/{lengths.sum()}, "
                         f"expected {len(files)} shots/{len(cons)} frames")
    if frame_share == 1.0:
        return cons
    from my_experiments.baseline_model import thin_frames
    offsets = np.r_[0, np.cumsum(lengths)[:-1]]
    rows = np.concatenate([offset + thin_frames(int(n), frame_share)
                           for offset, n in zip(offsets, lengths, strict=True)])
    return cons[rows]


def ensure(blocks: list[tuple[str, list[Path]]], jobs: int) -> None:
    """Build missing full-frame targets once, through the shared shot pool."""
    from tqdm import tqdm
    CACHE.mkdir(parents=True, exist_ok=True)
    ctx = None
    for name, files in blocks:
        out = cache_path(files)
        if out.exists():
            continue
        if ctx is None:
            with np.load(HERE.parent / "fusion_scoring/masks/d3d_envelope.npz") as mask:
                ctx = scorer_context(mask["grid_R"], mask["grid_Z"], D3D)
        workers = resolve_jobs(jobs, len(files))
        print(f"  {name}: preparing functionals for {len(files)} full shots -> {out.name}")
        parts = list(tqdm(pimap(_task, [(p, ctx) for p in files], workers), total=len(files),
                          unit="shot", desc=f"  {name} targets", **bar_kwargs(SHOT_EVERY)))
        lengths = np.asarray([len(part) for part in parts], dtype=np.int64)
        cons = np.concatenate(parts).astype(np.float32)
        fd, temporary = tempfile.mkstemp(dir=CACHE, suffix=".npz")
        try:
            with os.fdopen(fd, "wb") as stream:
                np.savez_compressed(stream, cons=cons, lengths=lengths)
            os.replace(temporary, out)
        finally:
            Path(temporary).unlink(missing_ok=True)


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
    from my_experiments.data_check import check_dataset
    check_dataset(args.local_data_dir, args.config)

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

    ensure([("train", train), ("val", val)], args.jobs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
