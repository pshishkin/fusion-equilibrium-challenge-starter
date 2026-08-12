#!/usr/bin/env python3
"""
Train and immediately score, in one command.

    uv run python my_experiments/train_eval.py 0.01 0.02

First argument is the share of shots to train on (head of the list), second is the share to
score on (tail). Both steps walk the same sha1-ordered list, so the split reproduces, and the
non-overlap is checked inside evaluate.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from my_experiments import evaluate, train  # noqa: E402


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        raise SystemExit(__doc__)
    train_share, eval_share = argv

    print("=" * 70)
    print(f"TRAINING on {float(train_share):.1%}")
    print("=" * 70)
    sys.argv = ["train.py", "--share", train_share]
    train.main()

    print("\n" + "=" * 70)
    print(f"SCORING on {float(eval_share):.1%}")
    print("=" * 70)
    sys.argv = ["evaluate.py", "--share", eval_share]
    return evaluate.main()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
