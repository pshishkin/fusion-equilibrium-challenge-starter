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

Shots are ordered by a hash of the shot id, NOT sampled randomly like `experiments.py --source
local` does: training takes the head of that list and scoring the tail, so the split is disjoint
by construction as long as the two shares sum to under 1.
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


# --------------------------------------------------------------------------- shot ordering

def shot_key(path: Path) -> str:
    """Sort key: sha1 of the shot id.

    hashlib, NOT the builtin hash(): that one is salted per process, so the order would change
    between the train run and the evaluate run and the split would stop being reproducible —
    silently. sha1 of the filename gives the same order every time, on any machine.
    """
    return hashlib.sha1(path.stem.encode()).hexdigest()


def sorted_shots(local_data_dir: Path = DEFAULT_LOCAL_DATA_DIR,
                 config: str = HF_TRAIN_CONFIG) -> list[Path]:
    """Every shot of the config, deterministically shuffled. Training takes the head of the list
    and evaluation the tail, so the windows stay disjoint while the shares sum to under 1."""
    data_dir = Path(local_data_dir) / "data" / config
    files = sorted(data_dir.glob("*.parquet"), key=shot_key)
    if not files:
        raise SystemExit(f"No parquet files in {data_dir}")
    return files


def take_share(files: list[Path], share: float, side: str) -> list[Path]:
    """A share of the list from the head or the tail, at least one shot."""
    if not 0 < share <= 1:
        raise SystemExit(f"--share must be in (0, 1], got {share}")
    n = max(1, round(len(files) * share))
    return files[:n] if side == "head" else files[-n:]


# --------------------------------------------------------- plasma-current time base

# `magnetics_plasma_current_times` is not a per-shot axis: the same template array, always
# starting at -858.1871 ms, is stamped into every DIII-D shot, while `magnetics_time` genuinely
# varies. For the ~70% of shots recorded at 0.05 ms it is the wrong axis, so interpolating Ip onto
# `efit_times` returns pre-shot noise — 4 kA where the trace actually sits at 1000 kA. The correct
# origin for those is the shot's own `magnetics_time[0]`; the sampling step is fine either way.
# See my_experiments/eda_ip_offset.py for the evidence, and README for the summary.
IP_ON = 0.05              # |Ip| above this fraction of its peak counts as "current flowing"
IP_COVERAGE_OK = 0.9      # fraction of EFIT frames that must see current flowing


def align_ip_times(efit_times: np.ndarray, ip_times: np.ndarray, ip_values: np.ndarray,
                   mag_times: np.ndarray) -> np.ndarray:
    """`ip_times` re-origined onto the shot's own acquisition clock, or unchanged if it already is.

    Shots whose frames already see the current are left strictly alone: for the 0.5 ms acquisition
    the template happens to be the right axis, and re-origining those would break them (coverage
    drops to 0.00, measured).

    For the rest the fix is exact rather than fitted — keep the 0.5 ms sampling, move the origin to
    `magnetics_time[0]`. Over 150 shots that puts current under 100% of the EFIT frames of every
    affected shot, where matching waveform windows only reached 93% in the worst case.
    """
    peak = np.abs(ip_values).max()
    if peak <= 0:
        raise ValueError("plasma current is identically zero")

    def coverage(axis: np.ndarray) -> float:
        return float((np.abs(np.interp(efit_times, axis, ip_values)) > IP_ON * peak).mean())

    if coverage(ip_times) >= IP_COVERAGE_OK:
        return ip_times

    corrected = mag_times[0] + (ip_times - ip_times[0])
    got = coverage(corrected)
    if got < IP_COVERAGE_OK:
        raise ValueError(
            f"cannot align the plasma current: re-origining it at magnetics_time[0] = "
            f"{mag_times[0]:.0f} ms leaves only {got:.0%} of EFIT frames with current flowing "
            f"(need {IP_COVERAGE_OK:.0%}). Ip axis starts at {ip_times[0]:.0f} ms and spans "
            f"{ip_times[-1] - ip_times[0]:.0f} ms; EFIT window is "
            f"[{efit_times.min():.0f}, {efit_times.max():.0f}] ms."
        )
    return corrected


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
            raise ValueError(f"row has no column {col} — is this a DIII-D row?")
    shared = np.asarray(row["magnetics_time"], dtype=np.float64)
    efit_times = np.asarray(row["efit_times"], dtype=np.float64)
    ip_times = align_ip_times(
        efit_times,
        np.asarray(row["magnetics_plasma_current_times"], dtype=np.float64),
        np.asarray(row["magnetics_plasma_current"], dtype=np.float64),
        shared,
    )

    magnetics = {}
    for sig in D3D_MAGNETICS_SIGNALS:
        col = f"magnetics_{sig}"
        if col not in row:
            raise ValueError(f"row has no signal {col}; all "
                             f"{len(D3D_MAGNETICS_SIGNALS)} are expected: {D3D_MAGNETICS_SIGNALS}")
        times = ip_times if sig == "plasma_current" else shared
        magnetics[sig] = {"values": np.asarray(row[col], dtype=np.float32), "times": times}

    return {"efit_times": efit_times, "magnetics": magnetics}


def features_for_row(row) -> np.ndarray:
    """(T, 21) magnetics features on the EFIT time base, exactly len(efit_times) rows."""
    shot = inputs_only_shot(row)
    feats = interpolate_magnetics_to_efit(shot)
    if feats.shape != (len(shot["efit_times"]), len(D3D_MAGNETICS_SIGNALS)):
        raise ValueError(f"features {feats.shape}, expected "
                         f"({len(shot['efit_times'])}, {len(D3D_MAGNETICS_SIGNALS)})")
    if not np.isfinite(feats).all():
        raise ValueError(f"features contain {int((~np.isfinite(feats)).sum())} non-finite values")
    return feats


# --------------------------------------------------------------------------- training

def train(share: float, n_pca: int, alpha: float,
          local_data_dir: Path, config: str) -> dict:
    all_files = sorted_shots(local_data_dir, config)
    files = take_share(all_files, share, "head")
    print(f"Training on {len(files)} shots ({share:.1%} of {len(all_files)}), "
          f"head of the list, ordered by sha1")

    X_parts, Y_parts, S_parts = [], [], []
    bar = tqdm(files, desc="reading shots", unit="shot")
    for path in bar:
        row = pd.read_parquet(path).iloc[0]
        feats = features_for_row(row)                 # same code path as inference
        psi = _as_psirz_stack(row["efit_psirz"])      # train rows only: targets are withheld on test
        T = len(feats)
        if len(psi) != T:
            raise ValueError(f"{path.name}: {T} feature rows, {len(psi)} psi frames")
        if not np.isfinite(psi).all():
            raise ValueError(f"{path.name}: psi has {int((~np.isfinite(psi)).sum())} non-finite values")

        scal = np.empty((T, len(SUBMITTED_SCALARS)), dtype=np.float64)
        for j, name in enumerate(SUBMITTED_SCALARS):
            if name not in row:
                raise ValueError(f"{path.name}: target column {name} is missing")
            arr = np.asarray(row[name], dtype=np.float64).ravel()
            if len(arr) != T:
                raise ValueError(f"{path.name}: {name} has length {len(arr)}, expected {T}")
            if not np.isfinite(arr).all():
                raise ValueError(f"{path.name}: {name} has "
                                 f"{int((~np.isfinite(arr)).sum())} non-finite values")
            scal[:, j] = arr

        X_parts.append(feats)
        Y_parts.append(psi)
        S_parts.append(scal)
        # Frames accumulate in the postfix: it shows both the sample size and that progress is
        # alive, without a line per shot.
        bar.set_postfix(frames=sum(len(x) for x in X_parts))

    X = np.concatenate(X_parts)
    Y = np.concatenate(Y_parts)
    S = np.concatenate(S_parts)
    del X_parts, Y_parts, S_parts

    print(f"  X {X.shape}  Y {Y.shape}  S {S.shape}")

    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)

    if n_pca > min(len(Xs), EFIT_GRID_SIZE ** 2):
        raise ValueError(f"--n-pca {n_pca} exceeds what the data supplies: {len(Xs)} frames, "
                         f"{EFIT_GRID_SIZE ** 2} pixels. Raise --share or lower --n-pca.")
    pca = TargetPCA(n_components=n_pca).fit(Y)
    print(f"  PCA: {n_pca} components explain "
          f"{np.cumsum(pca.explained_variance_ratio)[-1] * 100:.1f}% of the psi variance")

    psi_model = Ridge(alpha=alpha).fit(Xs, pca.transform(Y))

    scalar_models = {}
    for j, name in enumerate(SUBMITTED_SCALARS):
        scalar_models[name] = Ridge(alpha=alpha).fit(Xs, S[:, j])

    artifact = {
        "scaler": scaler, "pca": pca, "psi_model": psi_model,
        "scalar_models": scalar_models,
        # Filenames, not indices: evaluate.py intersects them directly, and the check stays
        # correct even if the data directory grows.
        "train_files": [p.name for p in files],
        "n_train_shots": len(files), "train_share": share, "config": config,
        "n_pca": n_pca, "alpha": alpha,
    }
    joblib.dump(artifact, ARTIFACT)
    print(f"\nSaved {ARTIFACT}")
    print("Now score it:  uv run python my_experiments/evaluate.py --share 0.02")
    return artifact


# --------------------------------------------------------------------------- inference

_CACHE: dict | None = None


def _load() -> dict:
    global _CACHE
    if _CACHE is None:
        if not ARTIFACT.exists():
            raise FileNotFoundError(
                f"{ARTIFACT} not found — train first:\n"
                f"  uv run python my_experiments/train.py --share 0.01"
            )
        _CACHE = joblib.load(ARTIFACT)
    return _CACHE


def predict_row(row, source: str = "DIII-D") -> dict:
    """Predict {psirz (T,65,65), q95 (T,), betaN (T,)} for one dataset row.

    Uses only `magnetics_*` and `efit_times` — never the `efit_*` targets, which are present in
    training rows and would make any local score meaningless.
    """
    art = _load()
    if source != "DIII-D":
        raise NotImplementedError(
            f"the model is trained on DIII-D only, but the row is marked source={source!r}. "
            f"MAST has its own coil set and needs its own fit — returning zeros silently is not "
            f"an option, it would look like a working prediction."
        )

    T = len(np.asarray(row["efit_times"]))
    Xs = art["scaler"].transform(features_for_row(row))   # features_for_row already demands finiteness

    out = {"psirz": art["pca"].inverse_transform(art["psi_model"].predict(Xs))}
    for name, key in zip(SUBMITTED_SCALARS, ["q95", "betaN"]):
        out[key] = art["scalar_models"][name].predict(Xs).astype(np.float32)

    for key, want in [("psirz", (T, EFIT_GRID_SIZE, EFIT_GRID_SIZE)), ("q95", (T,)), ("betaN", (T,))]:
        if out[key].shape != want:
            raise ValueError(f"{key}: shape {out[key].shape}, the contract requires {want}")
        if not np.isfinite(out[key]).all():
            raise ValueError(f"{key}: the prediction contains non-finite values")
    return out


# No CLI here on purpose: `train.py` is the one way to train and `evaluate.py` the one way to
# score, so there is never a second set of defaults to keep in sync. This module is the model.
