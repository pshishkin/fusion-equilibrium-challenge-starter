#!/usr/bin/env python3
"""
Entry point 2 of 2 — score the saved baseline on the LAST N local shots, with the real metric.

    uv run python my_experiments/evaluate.py --n-shots 20

LAST N, in sorted-filename order, because train.py takes the FIRST N in that same order: the two
windows grow from opposite ends, so a held-out split needs no bookkeeping. The overlap check
below is what makes that a guarantee instead of an assumption — a model scored on shots it was
fitted on reports a number that means nothing.

Scoring itself is `local_score.py`, called as a function so the metric has exactly one
implementation (the vendored `fusion_scoring/` modules the platform runs).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import local_score  # noqa: E402
from experiments import DEFAULT_LOCAL_DATA_DIR, HF_TRAIN_CONFIG  # noqa: E402
from my_experiments.baseline_model import ARTIFACT  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-shots", type=int, default=20,
                    help="how many of the LAST local shots to score on (default 20)")
    ap.add_argument("--mode", choices=["model", "perfect", "zeros"], default="model",
                    help="perfect/zeros verify the harness itself (S must be 1.0 / 0.0)")
    ap.add_argument("--local-data-dir", type=Path, default=DEFAULT_LOCAL_DATA_DIR,
                    help="root of the downloaded dataset (contains a 'data/' folder)")
    ap.add_argument("--config", default=HF_TRAIN_CONFIG)
    args = ap.parse_args()

    data_dir = Path(args.local_data_dir) / "data" / args.config
    files = sorted(data_dir.glob("*.parquet"))
    if not files:
        raise SystemExit(f"No parquet files in {data_dir}")

    n = min(args.n_shots, len(files))
    skip = len(files) - n                      # the LAST n shots

    if args.mode == "model" and ARTIFACT.exists():
        n_train = joblib.load(ARTIFACT).get("n_train_shots", 0)
        if skip < n_train:
            raise SystemExit(
                f"Refusing to score: the last {n} of {len(files)} shots start at index {skip}, "
                f"but the model was trained on the first {n_train}.\n"
                f"They overlap, so the score would be measured partly on training data. "
                f"Download more shots, or train on fewer."
            )
        print(f"Held-out check: trained on shots 0–{n_train - 1}, "
              f"scoring {skip}–{len(files) - 1} — disjoint.")

    return local_score.main([
        "--source", "local",
        "--local-data-dir", str(args.local_data_dir),
        "--config", args.config,
        "--n-shots", str(n),
        "--skip", str(skip),
        "--mode", args.mode,
    ])


if __name__ == "__main__":
    raise SystemExit(main())
