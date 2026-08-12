#!/usr/bin/env python3
"""
Entry point 2 of 2 — score the saved zoo with the real competition metric.

    uv run python my_experiments/evaluate.py --share 0.001              # every model + ensemble
    uv run python my_experiments/evaluate.py --share 0.001 --models ridge ensemble

Every model in the artifact is scored on the same held-out shots and printed side by side, with
the ensemble — their weighted average, weights from params.yaml — as one more row of the table.
The ground truth is prepared once and reused across models, so N models cost far less than N runs,
and the per-shot half of the work runs on `--jobs` processes.

Shots come from the TAIL of the list ordered by sha1 of the filename, while train.py takes the
head of the same list and the validation window right behind it. The overlap check is explicit —
not index arithmetic, but the filenames recorded in the artifact at training time, training AND
validation: a model scored on shots it was fitted on, or on shots it stopped on, reports a number
that means nothing.

Scoring itself is local_score.py, called as a function so the metric has exactly one
implementation (the vendored fusion_scoring/ modules, the same ones the platform runs).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import local_score
from experiments import DEFAULT_LOCAL_DATA_DIR, HF_TRAIN_CONFIG
from my_experiments.baseline_model import (
    ARTIFACT,
    model_names,
    sorted_shots,
    take_share,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--share", type=float, default=0.02,
                    help="share of shots to score on, from the tail of the list (default 0.02)")
    ap.add_argument("--mode", choices=["model", "perfect", "zeros"], default="model",
                    help="perfect/zeros verify the harness itself (S must be 1.0 / 0.0)")
    ap.add_argument("--models", nargs="+",
                    help="score only these members of the zoo (default: every model in the "
                         "artifact, plus the ensemble)")
    ap.add_argument("--jobs", type=int, default=0,
                    help="worker processes for the per-shot half of scoring (default 0 = "
                         "cores - 2, 1 = serial). Results do not depend on it")
    ap.add_argument("--local-data-dir", type=Path, default=DEFAULT_LOCAL_DATA_DIR,
                    help="root of the downloaded dataset (the folder containing 'data/')")
    ap.add_argument("--config", default=HF_TRAIN_CONFIG)
    args = ap.parse_args()

    all_files = sorted_shots(args.local_data_dir, args.config)
    files = take_share(all_files, args.share, "tail")
    print(f"Scoring {len(files)} shots ({args.share:.1%} of {len(all_files)}), tail of the list")

    extra: list[str] = []
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
        if "val_files" not in art:
            raise SystemExit(f"{ARTIFACT} carries no list of validation files — artifact from an "
                             f"older version, retrain")
        # Both windows the fit touched, not just the training one: early stopping read the
        # validation shots every iteration, so scoring on them would be scoring on seen data.
        seen = set(art["train_files"]) | set(art["val_files"])
        overlap = {p.name for p in files} & seen
        if overlap:
            raise SystemExit(
                f"Splits overlap: {len(overlap)} shots appear both in the fit and in the scoring "
                f"set (e.g. {sorted(overlap)[:3]}).\n"
                f"The models were fitted on {art['n_train_shots']} shots "
                f"({art.get('train_share', '?'):.1%}) and stopped on {art['n_val_shots']} "
                f"({art.get('val_share', '?'):.1%}), and you are asking for {args.share:.1%} "
                f"from the tail — over 100% together. Lower the share."
            )
        print(f"Held-out check: {art['n_train_shots']} shots trained on, {art['n_val_shots']} "
              f"validated on, {len(files)} scored — no overlap.")

        available = model_names(art)
        models = args.models if args.models else available
        unknown = set(models) - set(available)
        if unknown:
            raise SystemExit(f"{ARTIFACT} holds {available}, but --models asks for "
                             f"{sorted(unknown)}. Enable them in params.yaml and retrain.")
        print(f"Models: {', '.join(models)}")
        extra = ["--models", *models]
    elif args.models:
        raise SystemExit(f"--models is meaningless with --mode {args.mode}: perfect/zeros do not "
                         f"call the model at all")

    return local_score.main([
        "--source", "local",
        "--local-data-dir", str(args.local_data_dir),
        "--config", args.config,
        "--mode", args.mode,
        "--jobs", str(args.jobs),
        *extra,
        "--files", *[str(p) for p in files],
    ])


if __name__ == "__main__":
    raise SystemExit(main())
