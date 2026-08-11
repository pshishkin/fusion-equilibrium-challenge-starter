#!/usr/bin/env python3
"""
Точка входа 1 из 2 — обучить бейзлайн и сохранить его.

    uv run python my_experiments/train.py --share 0.01     # 1% шотов

Шоты берутся с НАЧАЛА списка, отсортированного по sha1 от имени файла. evaluate.py берёт
с конца того же списка, поэтому окна не пересекаются, пока сумма долей не превысит 1.
Порядок детерминирован, так что сплит воспроизводится на любой машине.

Пишет my_experiments/baseline.joblib, который дальше подхватывают evaluate.py
и submission_skeleton.py.
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
                    help="доля шотов для обучения, с начала списка (по умолчанию 0.01)")
    ap.add_argument("--n-pca", type=int, default=50, help="число компонент PCA для ψ")
    ap.add_argument("--alpha", type=float, default=1.0, help="регуляризация Ridge")
    ap.add_argument("--local-data-dir", type=Path, default=DEFAULT_LOCAL_DATA_DIR,
                    help="корень скачанного датасета (папка, содержащая 'data/')")
    ap.add_argument("--config", default=HF_TRAIN_CONFIG)
    args = ap.parse_args()

    train(args.share, args.n_pca, args.alpha, args.local_data_dir, args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
