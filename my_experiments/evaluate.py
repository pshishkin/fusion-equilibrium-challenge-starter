#!/usr/bin/env python3
"""
Entry point 2 of 2 — score the saved baseline with the real competition metric.

    uv run python my_experiments/evaluate.py --share 0.02     # 2% of the shots

Shots come from the TAIL of the list ordered by sha1 of the filename, while train.py takes the
head of the same list. The overlap check is explicit — not index arithmetic, but the filenames
recorded in the artifact at training time: a model scored on shots it was fitted on reports a
number that means nothing.

Scoring itself is local_score.py, called as a function so the metric has exactly one
implementation (the vendored fusion_scoring/ modules, the same ones the platform runs).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import local_score  # noqa: E402
from experiments import DEFAULT_LOCAL_DATA_DIR, HF_TRAIN_CONFIG  # noqa: E402
from my_experiments.baseline_model import ARTIFACT, sorted_shots, take_share  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--share", type=float, default=0.02,
                    help="share of shots to score on, from the tail of the list (default 0.02)")
    ap.add_argument("--mode", choices=["model", "perfect", "zeros"], default="model",
                    help="perfect/zeros verify the harness itself (S must be 1.0 / 0.0)")
    ap.add_argument("--local-data-dir", type=Path, default=DEFAULT_LOCAL_DATA_DIR,
                    help="root of the downloaded dataset (the folder containing 'data/')")
    ap.add_argument("--config", default=HF_TRAIN_CONFIG)
    args = ap.parse_args()

    all_files = sorted_shots(args.local_data_dir, args.config)
    files = take_share(all_files, args.share, "tail")
    print(f"Scoring {len(files)} shots ({args.share:.1%} of {len(all_files)}), tail of the list")

    if args.mode == "model":
        if not ARTIFACT.exists():
            raise SystemExit(
                f"{ARTIFACT} not found — without it the splits cannot be checked for overlap.\n"
                f"Train first:  uv run python my_experiments/train.py --share 0.01"
            )
        art = joblib.load(ARTIFACT)
        if not art.get("train_files"):
            raise SystemExit(f"{ARTIFACT} carries no list of training files — artifact from an "
                             f"older version, retrain")
        overlap = {p.name for p in files} & set(art["train_files"])
        if overlap:
            raise SystemExit(
                f"Splits overlap: {len(overlap)} shots appear in both training and scoring "
                f"(e.g. {sorted(overlap)[:3]}).\n"
                f"The model was fitted on {art['n_train_shots']} shots "
                f"({art.get('train_share', '?'):.1%}), and you are asking for {args.share:.1%} "
                f"from the tail — over 100% together. Lower the share."
            )
        print(f"Held-out check: {art['n_train_shots']} shots trained on, {len(files)} scored — "
              f"no overlap.")

    return local_score.main([
        "--source", "local",
        "--local-data-dir", str(args.local_data_dir),
        "--config", args.config,
        "--mode", args.mode,
        "--files", *[str(p) for p in files],
    ])


if __name__ == "__main__":
    raise SystemExit(main())
