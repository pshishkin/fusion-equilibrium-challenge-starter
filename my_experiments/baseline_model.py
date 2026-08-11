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
    times_col = "magnetics_time"
    shared = np.asarray(row[times_col], dtype=np.float64) if times_col in row else None
    ip_col = "magnetics_plasma_current_times"
    ip_times = np.asarray(row[ip_col], dtype=np.float64) if ip_col in row else shared

    magnetics = {}
    for sig in D3D_MAGNETICS_SIGNALS:
        col = f"magnetics_{sig}"
        if col not in row:
            continue
        times = ip_times if sig == "plasma_current" else shared
        if times is None:
            continue
        magnetics[sig] = {"values": np.asarray(row[col], dtype=np.float32), "times": times}

    return {"efit_times": np.asarray(row["efit_times"], dtype=np.float64), "magnetics": magnetics}


def features_for_row(row) -> np.ndarray:
    """(T, 21) magnetics features on the EFIT time base. Always exactly T rows, so a
    prediction can never come out misaligned with `efit_times`. Signals missing from the row
    become a zero column, exactly as interpolate_magnetics_to_efit does for the training rows."""
    return interpolate_magnetics_to_efit(inputs_only_shot(row))


# --------------------------------------------------------------------------- training

def train(share: float, n_pca: int, alpha: float,
          local_data_dir: Path, config: str) -> dict:
    all_files = sorted_shots(local_data_dir, config)
    files = take_share(all_files, share, "head")
    print(f"Обучение на {len(files)} шотах ({share:.1%} от {len(all_files)}), "
          f"начало списка, порядок по sha1")

    X_parts, Y_parts, S_parts = [], [], []
    for i, path in enumerate(files):
        row = pd.read_parquet(path).iloc[0]
        feats = features_for_row(row)                 # same code path as inference
        psi = _as_psirz_stack(row["efit_psirz"])      # train rows only — targets are withheld on test
        T = min(len(feats), len(psi))

        scal = np.full((T, len(SUBMITTED_SCALARS)), np.nan, dtype=np.float64)
        for j, name in enumerate(SUBMITTED_SCALARS):
            if name not in row:
                continue
            arr = np.asarray(row[name], dtype=np.float64).ravel()
            m = min(T, len(arr))
            scal[:m, j] = arr[:m]

        X_parts.append(feats[:T])
        Y_parts.append(psi[:T])
        S_parts.append(scal)
        print(f"  [{i + 1}/{len(files)}] {path.name}  T={T}")

    X = np.concatenate(X_parts)
    Y = np.concatenate(Y_parts)
    S = np.concatenate(S_parts)
    del X_parts, Y_parts, S_parts

    valid = np.isfinite(X).all(axis=1) & np.isfinite(Y.reshape(len(Y), -1)).all(axis=1)
    if not valid.all():
        print(f"  dropped {int((~valid).sum())} frames with non-finite inputs or ψ")
        X, Y, S = X[valid], Y[valid], S[valid]
    print(f"  X {X.shape}  Y {Y.shape}  S {S.shape}")

    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)

    n_components = min(n_pca, len(Xs), EFIT_GRID_SIZE ** 2)
    pca = TargetPCA(n_components=n_components).fit(Y)
    print(f"  PCA: {n_components} components capture "
          f"{np.cumsum(pca.explained_variance_ratio)[-1] * 100:.1f}% of ψ variance")

    psi_model = Ridge(alpha=alpha).fit(Xs, pca.transform(Y))

    # One Ridge per submitted scalar: q95 and betaN are undefined on different frames, so a
    # shared row mask would throw away data that the other target still has.
    scalar_models, scalar_fallback = {}, {}
    for j, name in enumerate(SUBMITTED_SCALARS):
        m = np.isfinite(S[:, j])
        scalar_fallback[name] = float(np.nanmean(S[m, j])) if m.any() else 0.0
        if m.sum() >= 10:
            scalar_models[name] = Ridge(alpha=alpha).fit(Xs[m], S[m, j])
            print(f"  {name}: fitted on {int(m.sum())}/{len(S)} frames")
        else:
            print(f"  {name}: only {int(m.sum())} valid frames — falling back to the mean")

    artifact = {
        "scaler": scaler, "pca": pca, "psi_model": psi_model,
        "scalar_models": scalar_models, "scalar_fallback": scalar_fallback,
        # Имена файлов, а не индексы: evaluate.py проверяет пересечение по ним напрямую,
        # и проверка остаётся верной, даже если каталог с данными пополнился.
        "train_files": [p.name for p in files],
        "n_train_shots": len(files), "train_share": share, "config": config,
        "n_pca": n_components, "alpha": alpha,
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
                f"{ARTIFACT} not found — train first:\n"
                f"  uv run python my_experiments/baseline_model.py --n-shots 30"
            )
        _CACHE = joblib.load(ARTIFACT)
    return _CACHE


def predict_row(row, source: str = "DIII-D") -> dict:
    """Predict {psirz (T,65,65), q95 (T,), betaN (T,)} for one dataset row.

    Uses only `magnetics_*` and `efit_times` — never the `efit_*` targets, which are present in
    training rows and would make any local score meaningless.
    """
    art = _load()
    T = len(np.asarray(row["efit_times"]))

    if source != "DIII-D":      # trained on DIII-D only; MAST would need its own fit
        return {"psirz": np.zeros((T, EFIT_GRID_SIZE, EFIT_GRID_SIZE), dtype=np.float32),
                "q95": np.zeros(T, dtype=np.float32), "betaN": np.zeros(T, dtype=np.float32)}

    # Every frame must get a prediction — the submission contract is exactly T rows — so
    # non-finite features are imputed to the training mean (0 after scaling) rather than dropped.
    Xs = np.nan_to_num(art["scaler"].transform(features_for_row(row)),
                       nan=0.0, posinf=0.0, neginf=0.0)

    out = {"psirz": art["pca"].inverse_transform(art["psi_model"].predict(Xs))}
    for name, key in zip(SUBMITTED_SCALARS, ["q95", "betaN"]):
        model = art["scalar_models"].get(name)
        vals = (model.predict(Xs) if model is not None
                else np.full(len(Xs), art["scalar_fallback"][name]))
        out[key] = vals.astype(np.float32)
    return out


# No CLI here on purpose: `train.py` is the one way to train and `evaluate.py` the one way to
# score, so there is never a second set of defaults to keep in sync. This module is the model.
