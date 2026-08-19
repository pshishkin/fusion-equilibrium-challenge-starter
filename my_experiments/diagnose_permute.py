#!/usr/bin/env python3
"""
Diagnostic C3 in ideas.md — which inputs does the fitted model actually use?

    uv run python my_experiments/diagnose_permute.py --share 0.005 --jobs 20

Permute one input block across frames, re-score, restore, repeat. No fitting, one scoring pass per
block. It answers questions the feature entries currently guess at — whether the accepted
derivatives are used or merely harmless, whether Thomson is dead weight, whether the model is
essentially reading the plasma current and the shaping coils.

It also answers a question this fork arrived at from the other direction. The coil gains absorb
each column's turn count, DIII-D's single filament per coil leaves that count out, and the
calibration reports the SOLENOID's gain as +142, +128, +94 or -10 depending on the sample while
every F-coil sits near 0.9. So the solenoid's contribution is the under-determined one — and if the
model barely uses ECOILA, that under-determination cannot be what blocks a transfer to MAST.

**Permutation measures what THIS fit uses, not what is usable.** A block that matters and is
duplicated elsewhere reads as unimportant. So a large value is strong evidence and a zero is weak,
and the report says which is which rather than leaving it to be misread.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import numpy.typing as npt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import local_score  # noqa: E402
from experiments import D3D_MAGNETICS_SIGNALS, DEFAULT_LOCAL_DATA_DIR, HF_TRAIN_CONFIG  # noqa: E402
from my_experiments.baseline_model import (  # noqa: E402
    ARTIFACT,
    DERIV_BLOCKS,
    ENSEMBLE,
    FEATURES_KEY,
    N_GAPS,
    N_SIGNALS,
    N_THOMSON_BLOCK,
    N_VESSEL,
    features_for_row,
    predict_row,
    sorted_shots,
    take_share,
)
from toolkit.progress import install_timestamps  # noqa: E402

FloatArray = npt.NDArray[np.floating]


def blocks() -> dict[str, list[int]]:
    """Column ranges of the cached feature matrix, by name. Mirrors `features_for_row`'s hstack."""
    out: dict[str, list[int]] = {}
    for i, sig in enumerate(D3D_MAGNETICS_SIGNALS):
        out[f"level:{sig}"] = [i]
    base = N_SIGNALS
    for k, name in enumerate(DERIV_BLOCKS):
        out[f"deriv:{name}"] = list(range(base + k * N_SIGNALS, base + (k + 1) * N_SIGNALS))
    base += len(DERIV_BLOCKS) * N_SIGNALS
    out["gaps"] = list(range(base, base + N_GAPS))
    base += N_GAPS
    out["vessel"] = list(range(base, base + N_VESSEL))
    out["thomson"] = list(range(-N_THOMSON_BLOCK, 0))
    return out


def main() -> int:
    install_timestamps()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--share", type=float, default=0.005)
    ap.add_argument("--model", default=ENSEMBLE)
    ap.add_argument("--jobs", type=int, default=20)
    ap.add_argument("--only", nargs="+", help="permute only these blocks (default: all of them)")
    ap.add_argument("--local-data-dir", type=Path, default=DEFAULT_LOCAL_DATA_DIR)
    ap.add_argument("--config", default=HF_TRAIN_CONFIG)
    ap.add_argument("--work", type=Path, default=Path("/tmp/permute"))
    args = ap.parse_args()

    art = joblib.load(ARTIFACT)
    files = take_share(sorted_shots(args.local_data_dir, args.config, int(art["split_salt"])),
                       args.share, "tail")
    if {p.name for p in files} & (set(art["train_files"]) | set(art["val_files"])):
        raise SystemExit("the scoring tail overlaps the fit; lower --share")
    shots = local_score.load_shots(0, 0, "local", args.local_data_dir, args.config, files)
    feats = [features_for_row(s["row"]).copy() for s in shots]
    todo = blocks()
    if args.only:
        unknown = [b for b in args.only if b not in todo]
        if unknown:
            raise SystemExit(f"no block(s) {unknown}; known {sorted(todo)}")
        todo = {b: todo[b] for b in args.only}
    print(f"Permuting {len(todo)} feature blocks over {len(files)} held-out shots, "
          f"{sum(len(f) for f in feats)} frames")

    args.work.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for name in ["(none)", *todo]:
        cols = todo.get(name)
        preds = {}
        for i, (s, f) in enumerate(zip(shots, feats, strict=True)):
            row = dict(s["row"])
            if cols is not None:
                g = f.copy()
                # Permuted ACROSS FRAMES within the shot: it breaks the link to this frame's
                # equilibrium while leaving the column's own distribution untouched, which is what
                # makes the drop attributable to the block rather than to feeding it nonsense.
                g[:, cols] = g[rng.permutation(len(g))][:, cols]
                row[FEATURES_KEY] = g
            out = predict_row(row, "DIII-D", args.model)
            preds[f"shot_{i:04d}_psirz"] = out["psirz"].astype(np.float16)
            preds[f"shot_{i:04d}_q95"] = np.asarray(out["q95"], dtype=np.float32)
            preds[f"shot_{i:04d}_betaN"] = np.asarray(out["betaN"], dtype=np.float32)
        p = args.work / f"{name.replace(':', '_')}.npz"
        np.savez_compressed(p, **preds)  # type: ignore[arg-type]
        print(f"\n=== permuted: {name} ===")
        local_score.main(["--source", "local", "--local-data-dir", str(args.local_data_dir),
                          "--config", args.config, "--jobs", str(args.jobs),
                          "--pred", str(p), "--files", *[str(x) for x in files]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
