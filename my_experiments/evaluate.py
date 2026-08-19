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
    SALTS_PREFIX,
    WEIGHT_SEP,
    artifact_path,
    model_names,
    sorted_shots,
    take_share,
)
from toolkit.progress import install_timestamps


def main() -> int:
    install_timestamps()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--share", type=float, default=0.02,
                    help="share of shots to score on, from the tail of the list (default 0.02)")
    ap.add_argument("--mode", choices=["model", "perfect", "zeros", "basis"], default="model",
                    help="perfect/zeros verify the harness itself (S must be 1.0 / 0.0); basis "
                         "scores ground truth pushed through the artifact's PCA — the ceiling a "
                         "perfect regression would hit, per term")
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
    ap.add_argument("--select-on-val", type=int, metavar="N",
                    help="score on N of the artifact's VALIDATION shots instead of the unseen "
                         "tail. Those shots are not clean — early stopping read them — so this is "
                         "for CHOOSING among things that are already fitted (which members to "
                         "average, with what weights), never for reporting a score. What early "
                         "stopping took from them is one scalar per model, the same for every "
                         "candidate, so it cannot prefer one combination over another; the tail "
                         "then confirms the choice on data nothing has touched. Any number "
                         "printed under this flag is a SELECTION number, not a result")
    ap.add_argument("--on-holdout", action="store_true",
                    help="score the shots the artifact reserved as its holdout instead of the "
                         "tail of this salt's order. That is the only set on which models fitted "
                         "under DIFFERENT salts can be compared, since it is untouched by all of "
                         "them; --share is ignored")
    args = ap.parse_args()

    # The shot ORDER has to be the one training used, so the salt comes from the artifact rather
    # than from params.yaml: the file on disk may have moved on since the model was fitted, and a
    # different order would silently score a different tail while the overlap check still passed.
    # `basis` needs the artifact as much as `model` does — it is that artifact's PCA being
    # measured, and it has to be measured on the same shots the models were scored on or the
    # ceiling is not comparable to what it is a ceiling for.
    art = None
    if args.mode in ("model", "basis"):
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
    elif args.select_on_val:
        if not art or not art.get("val_files"):
            raise SystemExit("--select-on-val needs an artifact that records its validation shots")
        keep = set(art["val_files"])
        pool = [p for p in all_files if p.name in keep]
        # Evenly spaced through the window rather than its head, so the sample is not one corner
        # of the sha1 order.
        step = max(1, len(pool) // args.select_on_val)
        files = pool[::step][:args.select_on_val]
        print(f"SELECTION SET: {len(files)} of {len(pool)} VALIDATION shots. Early stopping read "
              f"these, so what comes out is a ranking and not a score — confirm the winner on the "
              f"tail before believing any of it.")
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
        # `--select-on-val` asks for the validation window ON PURPOSE, so the overlap check would
        # be refusing what was requested. Everything else is still checked, and the flag prints its
        # own warning above.
        seen = (set() if args.select_on_val
                else set(art["train_files"]) | set(art["val_files"]))
        available = model_names(art)
        # `basis` calls no model, so there is nothing to name — but the overlap check below still
        # runs, because a ceiling measured on shots the PCA was fitted on is not the ceiling.
        models = [] if args.mode == "basis" else (args.models if args.models else available)
        # A `salts:` model reaches into other artifacts, and every one of them has its own split.
        # The check has to see all of them or it certifies the wrong thing: a shot held out of THIS
        # fit may well be in another member's training set, which is exactly the mistake the
        # holdout exists to prevent — so the members are loaded before the overlap is tested.
        # `--on-holdout` used to be REQUIRED here. It is not the real condition and it ruled out a
        # legitimate case: two artifacts fitted on the SAME split — same salt, same shares —
        # differing only in features or in architecture. Averaging those is exactly what an
        # ensemble is, and the tail is untouched by both. The actual requirement is that no member
        # has seen a scored shot, and the overlap check below is that requirement, applied to every
        # member's own file lists rather than to an assumption about which set is safe.
        for spec in [m for m in models if m.startswith(SALTS_PREFIX)]:
            for name in spec[len(SALTS_PREFIX):].split("+"):
                member = joblib.load(artifact_path(name))
                seen |= set(member["train_files"]) | set(member["val_files"])
                print(f"  {artifact_path(name).name}: salt {member['split_salt']}, "
                      f"{member['n_train_shots']} train + {member['n_val_shots']} val shots "
                      f"folded into the overlap check")
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

        # `a+b` is an equally weighted average of two fitted members, assembled at scoring time —
        # so the check is per PART. Which members to combine is a question about a fitted
        # artifact, and answering it should cost one training run rather than one per combination.
        # `salts:` names are artifacts, not members of this one, so they are checked by loading.
        # A name may be `a`, `a+b`, or `psi=a+b,qb=c`; every leaf still has to be a fitted model.
        leaves = set()
        for m in models:
            if m.startswith(SALTS_PREFIX):
                continue
            for block in m.split(","):
                # `name*weight` — the weight is not part of the name, so strip it before checking
                # the leaf against the zoo.
                leaves |= {p.partition(WEIGHT_SEP)[0] for p in block.split("=")[-1].split("+")}
        unknown = leaves - set(available)
        if unknown:
            raise SystemExit(f"{ARTIFACT} holds {available}, but --models asks for "
                             f"{sorted(unknown)}. Enable them in params.yaml and retrain.")
        if models:
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
