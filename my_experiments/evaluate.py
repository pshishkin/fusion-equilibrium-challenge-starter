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
from my_experiments.progress import install_timestamps


def main() -> int:
    install_timestamps()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--share", type=float, default=0.02,
                    help="share of shots to score on, from the tail of the list (default 0.02)")
    ap.add_argument("--mode", choices=["model", "perfect", "zeros"], default="model",
                    help="perfect/zeros verify the harness itself (S must be 1.0 / 0.0)")
    ap.add_argument("--models", nargs="+",
                    help="score only these members of the zoo (default: every model in the "
                         "artifact, plus the ensemble). 'a+b' scores their equally weighted "
                         "average, so every combination of what is already fitted can be "
                         "compared without refitting anything")
    ap.add_argument("--jobs", type=int, default=0,
                    help="worker processes for the per-shot half of scoring (default 0 = "
                         "cores - 2, 1 = serial). Results do not depend on it")
    ap.add_argument("--local-data-dir", type=Path, default=DEFAULT_LOCAL_DATA_DIR,
                    help="root of the downloaded dataset (the folder containing 'data/')")
    ap.add_argument("--config", default=HF_TRAIN_CONFIG)
    ap.add_argument("--on-holdout", action="store_true",
                    help="score the shots the artifact reserved as its holdout instead of the "
                         "tail of this salt's order. That is the only set on which models fitted "
                         "under DIFFERENT salts can be compared, since it is untouched by all of "
                         "them; --share is ignored")
    args = ap.parse_args()

    # The shot ORDER has to be the one training used, so the salt comes from the artifact rather
    # than from params.yaml: the file on disk may have moved on since the model was fitted, and a
    # different order would silently score a different tail while the overlap check still passed.
    art = None
    if args.mode == "model":
        if not ARTIFACT.exists():
            raise SystemExit(
                f"{ARTIFACT} not found — without it the splits cannot be checked for overlap.\n"
                f"Train first:  uv run python my_experiments/train.py --share 0.01"
            )
        art = joblib.load(ARTIFACT)
        if "split_salt" not in art:
            raise SystemExit(f"{ARTIFACT} does not say which split salt it was trained on — "
                             f"artifact from an older version, retrain")
    salt = int(art["split_salt"]) if art is not None else 0

    all_files = sorted_shots(args.local_data_dir, args.config, salt)
    if args.on_holdout:
        if not art or not art.get("holdout_files"):
            raise SystemExit(
                f"--on-holdout, but {ARTIFACT} reserved none. Set split.holdout_share in "
                f"params.yaml and retrain; a holdout cannot be carved after the fact, because the "
                f"point of it is that the fit never saw those shots."
            )
        keep = set(art["holdout_files"])
        files = [p for p in all_files if p.name in keep]
        if len(files) != len(keep):
            raise SystemExit(f"{ARTIFACT} names {len(keep)} holdout shots but {len(files)} are "
                             f"present in {args.local_data_dir} — the data directory has changed")
        print(f"Scoring the {len(files)}-shot HOLDOUT ({art['holdout_share']:.1%}), which no salt "
              f"trains on — so members fitted under different salts are comparable here")
    else:
        files = take_share(all_files, args.share, "tail")
        print(f"Scoring {len(files)} shots ({args.share:.1%} of {len(all_files)}), tail of the "
              f"list{'' if salt == 0 else f', split salt {salt}'}")

    extra: list[str] = []
    if art is not None:
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
        # `a+b` is an equally weighted average of two fitted members, assembled at scoring time —
        # so the check is per PART. Which members to combine is a question about a fitted
        # artifact, and answering it should cost one training run rather than one per combination.
        unknown = {p for m in models for p in m.split("+")} - set(available)
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
