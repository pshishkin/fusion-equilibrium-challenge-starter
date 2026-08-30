#!/usr/bin/env python3
"""
Train and immediately score, in one command.

    uv run python my_experiments/train_eval.py 0.01 0.002 0.001

Three shares of one sha1-ordered shot list, in the order they are used:

    train   the head — what the models are fitted on
    val     right behind it — what CatBoost and the MLP stop on and pick their best iteration by
    test    the tail — what the metric is computed on, untouched by any fit

Anything after the three shares goes to train.py unchanged:

    train_eval.py 0.05/0.2 0.05/0.2 0.01 --only mlp --salt 1 --jobs 24

Validation is part of fitting, so it cannot double as the score: a model that stopped on those
shots has already read them. Only the tail is clean. All three windows are disjoint by
construction while the shares sum to under 1, and evaluate.py checks that against the filenames
the artifact recorded rather than trusting the arithmetic.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from my_experiments import evaluate, train
from toolkit.progress import install_timestamps


def main(argv: list[str]) -> int:
    # Before either half runs, so training and scoring share one clock and one memory trace.
    install_timestamps()
    # The three shares FIRST and positionally, then anything else straight through to train.py —
    # `--salt`, `--only`, `--jobs`. Order is not negotiable rather than inferred: `--only mlp` and
    # `--salt 1` both carry a value that does not start with a dash, so "the arguments that look
    # positional" would quietly pick up `mlp` as a share.
    if len(argv) < 3:
        raise SystemExit(__doc__)
    train_share, val_share, test_share = argv[:3]
    for i, share in enumerate(argv[:3]):
        if share.startswith("-"):
            raise SystemExit(f"argument {i + 1} is {share!r}: the three shares come first, then "
                             f"any flags.\n\n{__doc__}")
    passthrough = list(argv[3:])
    # Scoring reads the salt and the model names out of the ARTIFACT, so those need no forwarding —
    # the artifact is the one copy that cannot disagree with what was fitted. `--jobs` is different:
    # it is about this machine, not about the configuration, and both halves need it.
    jobs = passthrough[passthrough.index("--jobs") + 1:][:1] if "--jobs" in passthrough else []
    # Which members to SCORE, which is a different question from which to FIT and belongs to
    # neither half's own flags: `--only` picks what train.py builds, `--models` picks what
    # evaluate.py measures. Scoring is 48% of a run and costs ~2.5 min per member, so a lane that
    # reads only the ensemble row is paying for five tables nobody opens. Removed from the
    # passthrough rather than forwarded, because train.py would reject an argument it does not know.
    score_models: list[str] = []
    if "--score-models" in passthrough:
        i = passthrough.index("--score-models")
        j = i + 1
        while j < len(passthrough) and not passthrough[j].startswith("-"):
            j += 1
        score_models = passthrough[i + 1:j]
        if not score_models:
            raise SystemExit("--score-models needs at least one name, or omit it to score all")
        passthrough = passthrough[:i] + passthrough[j:]
    if "/" in test_share:
        raise SystemExit(
            f"test share {test_share!r} carries a frame fraction. Scoring reads every frame of "
            f"the shots it scores — that is what the competition metric is — so only the train "
            f"and validation shares may be thinned."
        )

    print("=" * 70)
    print(f"TRAINING on {train_share}, stopping on {val_share}")
    print("=" * 70)
    sys.argv = ["train.py", "--share", train_share, "--val-share", val_share, *passthrough]
    train.main()

    print("\n" + "=" * 70)
    print(f"SCORING on {float(test_share):.1%}")
    print("=" * 70)
    sys.argv = ["evaluate.py", "--share", test_share, *(["--jobs", *jobs] if jobs else []),
                *(["--models", *score_models] if score_models else [])]
    return evaluate.main()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
