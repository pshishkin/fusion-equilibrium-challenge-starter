#!/usr/bin/env python3
"""
The ψ baseline itself: training, the saved artifact, and inference. Not an entry point —
`train.py` trains it and `evaluate.py` scores it.

`experiments.py` compares models but never persists one, so nothing it trains survives the
process and there is no artifact for `local_score.py` to score. This module closes that gap:

    train.py  ->  my_experiments/baseline.joblib  ->  your_model_predict  ->  evaluate.py

The model is deliberately the plain baseline from MODELING_GUIDE.md: 21 interpolated magnetics
features -> StandardScaler -> Ridge -> PCA coefficients -> ψ(R,Z), with a second Ridge for
q95/betaN. Swap the estimators here once the plumbing is proven.

Shots are taken in sorted-filename order, NOT sampled randomly like `experiments.py --source
local` does, precisely so that training on the first N and scoring on the last M is a disjoint
split by construction.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments import (  # noqa: E402
    D3D_MAGNETICS_SIGNALS,
    DEFAULT_LOCAL_DATA_DIR,
    EFIT_GRID_SIZE,
    HF_TRAIN_CONFIG,
    TargetPCA,
    _as_psirz_stack,
    interpolate_magnetics_to_efit,
)

HERE = Path(__file__).resolve().parent
ARTIFACT = HERE / "baseline.joblib"
SUBMITTED_SCALARS = ["efit_q95", "efit_beta_n"]   # -> q95, betaN


# --------------------------------------------------------------------------- порядок шотов

def shot_key(path: Path) -> str:
    """Ключ сортировки: sha1 от идентификатора шота.

    Именно hashlib, а НЕ встроенный hash(): тот солится на каждый запуск процесса, и порядок
    менялся бы от вызова к вызову — train и evaluate разъехались бы, а сплит перестал быть
    воспроизводимым. sha1 от имени файла даёт один и тот же порядок всегда и на любой машине.
    """
    return hashlib.sha1(path.stem.encode()).hexdigest()


def sorted_shots(local_data_dir: Path = DEFAULT_LOCAL_DATA_DIR,
                 config: str = HF_TRAIN_CONFIG) -> list[Path]:
    """Все шоты конфига, перемешанные детерминированно. Обучение берёт начало списка,
    оценка — конец, поэтому окна не пересекаются, пока их доли в сумме не превысят 1."""
    data_dir = Path(local_data_dir) / "data" / config
    files = sorted(data_dir.glob("*.parquet"), key=shot_key)
    if not files:
        raise SystemExit(f"Нет parquet-файлов в {data_dir}")
    return files


def take_share(files: list[Path], share: float, side: str) -> list[Path]:
    """Доля списка с начала ('head') или с конца ('tail'), минимум один шот."""
    if not 0 < share <= 1:
        raise SystemExit(f"--share должен быть в (0, 1], получено {share}")
    n = max(1, round(len(files) * share))
    return files[:n] if side == "head" else files[-n:]


# --------------------------------------------------------------------------- features

def inputs_only_shot(row) -> dict:
    """The minimal shot dict the features need: `efit_times` + the magnetics inputs.

    Deliberately NOT experiments.load_shot_from_hf_row — that one reads `efit_psirz`, which the
    public/private test configs do not have (targets are withheld), so it raises KeyError on
    exactly the rows a submission is built from. Building inputs only also makes it structurally
    impossible for inference to peek at `efit_*` targets, which would make any local score
    meaningless. Works for a streamed dict and for a parquet row alike — `in` hits dict keys and
    Series index the same way.
    """
    for col in ("magnetics_time", "magnetics_plasma_current_times", "efit_times"):
        if col not in row:
            raise ValueError(f"в строке нет колонки {col} — это не строка DIII-D?")
    shared = np.asarray(row["magnetics_time"], dtype=np.float64)
    ip_times = np.asarray(row["magnetics_plasma_current_times"], dtype=np.float64)

    magnetics = {}
    for sig in D3D_MAGNETICS_SIGNALS:
        col = f"magnetics_{sig}"
        if col not in row:
            raise ValueError(f"в строке нет сигнала {col}; ожидались все "
                             f"{len(D3D_MAGNETICS_SIGNALS)}: {D3D_MAGNETICS_SIGNALS}")
        times = ip_times if sig == "plasma_current" else shared
        magnetics[sig] = {"values": np.asarray(row[col], dtype=np.float32), "times": times}

    return {"efit_times": np.asarray(row["efit_times"], dtype=np.float64), "magnetics": magnetics}


def features_for_row(row) -> np.ndarray:
    """(T, 21) magnetics features on the EFIT time base, ровно len(efit_times) строк."""
    shot = inputs_only_shot(row)
    feats = interpolate_magnetics_to_efit(shot)
    if feats.shape != (len(shot["efit_times"]), len(D3D_MAGNETICS_SIGNALS)):
        raise ValueError(f"признаки {feats.shape}, ожидалось "
                         f"({len(shot['efit_times'])}, {len(D3D_MAGNETICS_SIGNALS)})")
    if not np.isfinite(feats).all():
        raise ValueError(f"в признаках {int((~np.isfinite(feats)).sum())} нефинитных значений")
    return feats


# --------------------------------------------------------------------------- training

def train(share: float, n_pca: int, alpha: float,
          local_data_dir: Path, config: str) -> dict:
    all_files = sorted_shots(local_data_dir, config)
    files = take_share(all_files, share, "head")
    print(f"Обучение на {len(files)} шотах ({share:.1%} от {len(all_files)}), "
          f"начало списка, порядок по sha1")

    X_parts, Y_parts, S_parts = [], [], []
    bar = tqdm(files, desc="чтение шотов", unit="шот")
    for path in bar:
        row = pd.read_parquet(path).iloc[0]
        feats = features_for_row(row)                 # тот же путь, что и в инференсе
        psi = _as_psirz_stack(row["efit_psirz"])      # только train: на тесте цели withheld
        T = len(feats)
        if len(psi) != T:
            raise ValueError(f"{path.name}: признаков {T} строк, кадров ψ {len(psi)}")
        if not np.isfinite(psi).all():
            raise ValueError(f"{path.name}: в ψ {int((~np.isfinite(psi)).sum())} нефинитных значений")

        scal = np.empty((T, len(SUBMITTED_SCALARS)), dtype=np.float64)
        for j, name in enumerate(SUBMITTED_SCALARS):
            if name not in row:
                raise ValueError(f"{path.name}: нет целевой колонки {name}")
            arr = np.asarray(row[name], dtype=np.float64).ravel()
            if len(arr) != T:
                raise ValueError(f"{path.name}: {name} длины {len(arr)}, ожидалось {T}")
            if not np.isfinite(arr).all():
                raise ValueError(f"{path.name}: в {name} "
                                 f"{int((~np.isfinite(arr)).sum())} нефинитных значений")
            scal[:, j] = arr

        X_parts.append(feats)
        Y_parts.append(psi)
        S_parts.append(scal)
        # Кадры копятся в постфиксе: по нему видно и объём выборки, и что прогресс живой,
        # без строки на каждый шот.
        bar.set_postfix(кадров=sum(len(x) for x in X_parts))

    X = np.concatenate(X_parts)
    Y = np.concatenate(Y_parts)
    S = np.concatenate(S_parts)
    del X_parts, Y_parts, S_parts

    print(f"  X {X.shape}  Y {Y.shape}  S {S.shape}")

    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)

    if n_pca > min(len(Xs), EFIT_GRID_SIZE ** 2):
        raise ValueError(f"--n-pca {n_pca} больше доступного: кадров {len(Xs)}, "
                         f"пикселей {EFIT_GRID_SIZE ** 2}. Увеличьте --share или уменьшите --n-pca.")
    pca = TargetPCA(n_components=n_pca).fit(Y)
    print(f"  PCA: {n_pca} компонент объясняют "
          f"{np.cumsum(pca.explained_variance_ratio)[-1] * 100:.1f}% дисперсии ψ")

    psi_model = Ridge(alpha=alpha).fit(Xs, pca.transform(Y))

    scalar_models = {}
    for j, name in enumerate(SUBMITTED_SCALARS):
        scalar_models[name] = Ridge(alpha=alpha).fit(Xs, S[:, j])

    artifact = {
        "scaler": scaler, "pca": pca, "psi_model": psi_model,
        "scalar_models": scalar_models,
        # Имена файлов, а не индексы: evaluate.py проверяет пересечение по ним напрямую,
        # и проверка остаётся верной, даже если каталог с данными пополнился.
        "train_files": [p.name for p in files],
        "n_train_shots": len(files), "train_share": share, "config": config,
        "n_pca": n_pca, "alpha": alpha,
    }
    joblib.dump(artifact, ARTIFACT)
    print(f"\nСохранено: {ARTIFACT}")
    print("Оценить:  uv run python my_experiments/evaluate.py --share 0.02")
    return artifact


# --------------------------------------------------------------------------- inference

_CACHE: dict | None = None


def _load() -> dict:
    global _CACHE
    if _CACHE is None:
        if not ARTIFACT.exists():
            raise FileNotFoundError(
                f"{ARTIFACT} не найден — сначала обучите:\n"
                f"  uv run python my_experiments/train.py --share 0.01"
            )
        _CACHE = joblib.load(ARTIFACT)
    return _CACHE


def predict_row(row, source: str = "DIII-D") -> dict:
    """Предсказание {psirz (T,65,65), q95 (T,), betaN (T,)} для одной строки датасета.

    Использует только `magnetics_*` и `efit_times` — никогда `efit_*` цели, которые есть в
    обучающих строках и обесценили бы любой локальный скор.
    """
    art = _load()
    if source != "DIII-D":
        raise NotImplementedError(
            f"модель обучена только на DIII-D, а строка помечена source={source!r}. "
            f"Для MAST нужен свой набор катушек и своё обучение — молча выдавать нули нельзя, "
            f"это выглядело бы как рабочее предсказание."
        )

    T = len(np.asarray(row["efit_times"]))
    Xs = art["scaler"].transform(features_for_row(row))   # features_for_row уже требует конечности

    out = {"psirz": art["pca"].inverse_transform(art["psi_model"].predict(Xs))}
    for name, key in zip(SUBMITTED_SCALARS, ["q95", "betaN"]):
        out[key] = art["scalar_models"][name].predict(Xs).astype(np.float32)

    for key, want in [("psirz", (T, EFIT_GRID_SIZE, EFIT_GRID_SIZE)), ("q95", (T,)), ("betaN", (T,))]:
        if out[key].shape != want:
            raise ValueError(f"{key}: форма {out[key].shape}, контракт требует {want}")
        if not np.isfinite(out[key]).all():
            raise ValueError(f"{key}: в предсказании есть нефинитные значения")
    return out


# No CLI here on purpose: `train.py` is the one way to train and `evaluate.py` the one way to
# score, so there is never a second set of defaults to keep in sync. This module is the model.
