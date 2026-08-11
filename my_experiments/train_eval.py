#!/usr/bin/env python3
"""
Обучить и сразу оценить, одной командой.

    uv run python my_experiments/train_eval.py 0.01 0.02

Первый аргумент — доля шотов на обучение (с начала списка), второй — на оценку (с конца).
Оба шага используют один и тот же порядок, отсортированный по sha1 от имени файла, поэтому
сплит воспроизводим, а непересечение проверяется внутри evaluate.py.
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
    print(f"ОБУЧЕНИЕ, доля {float(train_share):.1%}")
    print("=" * 70)
    sys.argv = ["train.py", "--share", train_share]
    train.main()

    print("\n" + "=" * 70)
    print(f"ОЦЕНКА, доля {float(eval_share):.1%}")
    print("=" * 70)
    sys.argv = ["evaluate.py", "--share", eval_share]
    return evaluate.main()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
