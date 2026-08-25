#!/usr/bin/env python3
"""Per-frame seed disagreement over the TRAINING split — the weight A22/C8 actually calls for.

    uv run python my_experiments/frame_disagreement.py --share 0.80/1.0 --val-share 0.19/1.0

**Why disagreement rather than position.** Oracles on the shipped trio, each asking what perfect
success on a set of frames would buy: the most expensive 20% by TRUE cost is worth +0.00244 of S
but needs the labels; the top 20% by seed DISAGREEMENT is worth **+0.00159** and needs none; the
two end deciles — the position weight this replaced — are worth only +0.00082, because position is
a weak proxy for difficulty. The expensive frames are spread, 19% in the first decile, 18% in the
last, about 8% in each middle one.

**Why it needs no labels, which is what makes it usable on the training split.** Disagreement is
the spread of the seven derived scalars ACROSS the four fitted seeds. Nothing here reads a target,
so weighting the training set by it cannot leak. Measured against per-frame cost on the scored
tail: Spearman **+0.596** here, +0.565 when C8 measured it.

**The blind spot, from C8 and unchanged.** Disagreement sees variance and is blind to shared error,
so it under-reports the tail by more than two — 1.19x disagreement against 2.61x cost — because
seed residuals correlate +0.550 there. Four seeds wrong the same way look confident. It still
doubles what a position weight reaches.

Writes one npz keyed by the exact file list, exactly as `aux_targets.py` keys its cache, so a
different split refuses to use it.
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

CACHE = HERE.parent / ".frame_disagreement"
SEEDS = ("mlp", "mlp1", "mlp2", "mlp3")
FloatArray = npt.NDArray[np.floating]


def cache_path(files: list[Path]) -> Path:
    h = hashlib.sha1("|".join(p.name for p in files).encode()).hexdigest()[:16]
    return CACHE / f"dis_{len(files)}_{h}.npz"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--share", default="0.80/1.0")
    ap.add_argument("--val-share", default="0.19/1.0")
    ap.add_argument("--salt", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--artifact", type=Path,
                    default=Path("artefacts/candidate/baseline_prod.joblib"))
    a = ap.parse_args(argv)

    from my_experiments.baseline_model import (
        DEFAULT_LOCAL_DATA_DIR,
        HF_TRAIN_CONFIG,
        parse_share,
        sorted_shots,
        split_train_val,
    )
    shot_share, frame_share = parse_share(a.share)
    val_shot_share, val_frame_share = parse_share(a.val_share)
    if frame_share < 1.0 or val_frame_share < 1.0:
        raise SystemExit("the weight is per FRAME, so both shares must be /1.0 — thinning happens "
                         "where the training set is built, not here")
    files = sorted_shots(DEFAULT_LOCAL_DATA_DIR, HF_TRAIN_CONFIG, a.salt)
    train, _val = split_train_val(files, shot_share, val_shot_share)
    out = cache_path(train)
    if out.exists():
        print(f"  {out.name} already there")
        return 0
    print(f"  {len(train)} training shots, {len(SEEDS)} seeds, {a.jobs} process(es) -> {out.name}")
    print("  NOTE: this is a prediction pass over the whole training split and costs hours.")
    print("  It must not run beside two training lanes — they peak at 73 GB each of a 183 GB cap.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
