#!/usr/bin/env python3
"""
Entry point 1 of 2 — train the baseline on the FIRST N local shots and save it.

    uv run python my_experiments/train.py --n-shots 20

Writes my_experiments/baseline.joblib, which `your_model_predict` (and therefore evaluate.py and
submission_skeleton.py) picks up automatically.

FIRST N, in sorted-filename order — evaluate.py scores the LAST M in that same order, so the two
sets cannot overlap as long as N + M <= the number of shots you have downloaded. evaluate.py
enforces that rather than trusting it.
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
    ap.add_argument("--n-shots", type=int, default=20,
                    help="how many of the FIRST local shots to train on (default 20)")
    ap.add_argument("--n-pca", type=int, default=50, help="PCA components for ψ")
    ap.add_argument("--alpha", type=float, default=1.0, help="Ridge regularization")
    ap.add_argument("--local-data-dir", type=Path, default=DEFAULT_LOCAL_DATA_DIR,
                    help="root of the downloaded dataset (contains a 'data/' folder)")
    ap.add_argument("--config", default=HF_TRAIN_CONFIG)
    args = ap.parse_args()

    train(args.n_shots, args.n_pca, args.alpha, args.local_data_dir, args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
