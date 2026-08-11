#!/usr/bin/env python3
"""
Точка входа 2 из 2 — оценить сохранённый бейзлайн реальной метрикой соревнования.

    uv run python my_experiments/evaluate.py --share 0.02     # 2% шотов

Шоты берутся с КОНЦА списка, отсортированного по sha1 от имени файла, а train.py берёт с
начала того же списка. Пересечение проверяется явно — не по арифметике индексов, а по именам
файлов, записанным в артефакт при обучении: модель, оценённая на шотах, на которых училась,
даёт бессмысленное число.

Сам счёт делает local_score.py, вызываемый как функция, чтобы метрика имела ровно одну
реализацию (вендоренные модули fusion_scoring/, те же, что крутит платформа).
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
                    help="доля шотов для оценки, с конца списка (по умолчанию 0.02)")
    ap.add_argument("--mode", choices=["model", "perfect", "zeros"], default="model",
                    help="perfect/zeros проверяют сам харнесс (S должен быть 1.0 / 0.0)")
    ap.add_argument("--local-data-dir", type=Path, default=DEFAULT_LOCAL_DATA_DIR,
                    help="корень скачанного датасета (папка, содержащая 'data/')")
    ap.add_argument("--config", default=HF_TRAIN_CONFIG)
    args = ap.parse_args()

    all_files = sorted_shots(args.local_data_dir, args.config)
    files = take_share(all_files, args.share, "tail")
    print(f"Оценка на {len(files)} шотах ({args.share:.1%} от {len(all_files)}), конец списка")

    if args.mode == "model" and ARTIFACT.exists():
        art = joblib.load(ARTIFACT)
        overlap = {p.name for p in files} & set(art.get("train_files", []))
        if overlap:
            raise SystemExit(
                f"Пересечение сплитов: {len(overlap)} шотов есть и в обучении, и в оценке "
                f"(например {sorted(overlap)[:3]}).\n"
                f"Модель училась на {art['n_train_shots']} шотах "
                f"({art.get('train_share', '?'):.1%}), сейчас просите {args.share:.1%} с конца — "
                f"в сумме больше 100%. Уменьшите долю."
            )
        print(f"Проверка отложенности: обучение {art['n_train_shots']} шотов, "
              f"оценка {len(files)} — пересечений нет.")

    return local_score.main([
        "--source", "local",
        "--local-data-dir", str(args.local_data_dir),
        "--config", args.config,
        "--mode", args.mode,
        "--files", *[str(p) for p in files],
    ])


if __name__ == "__main__":
    raise SystemExit(main())
