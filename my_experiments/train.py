#!/usr/bin/env python3
"""
Entry point 1 of 2 — fit the baseline and save it.

    uv run python my_experiments/train.py --share 0.01     # 1% of the shots

Shots come from the HEAD of the list ordered by sha1 of the filename; evaluate.py takes the tail
of that same list, so the two windows stay disjoint while the shares sum to under 1. The order is
deterministic, so the split reproduces on any machine.

Writes my_experiments/baseline.joblib, which evaluate.py and submission_skeleton.py pick up.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments import DEFAULT_LOCAL_DATA_DIR, HF_TRAIN_CONFIG  # noqa: E402
from my_experiments.baseline_model import train  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--share", type=float, default=0.01,
                    help="share of shots to train on, from the head of the list (default 0.01)")
    ap.add_argument("--n-pca", type=int, default=50, help="number of PCA components for psi")
    ap.add_argument("--alpha", type=float, default=1.0, help="Ridge regularization")
    ap.add_argument("--local-data-dir", type=Path, default=DEFAULT_LOCAL_DATA_DIR,
                    help="root of the downloaded dataset (the folder containing 'data/')")
    ap.add_argument("--config", default=HF_TRAIN_CONFIG)
    args = ap.parse_args()

    train(args.share, args.n_pca, args.alpha, args.local_data_dir, args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
