#!/usr/bin/env python3
"""
The psi pipeline itself: features, targets, training of the zoo, the saved artifact, inference.
Not an entry point — `train.py` trains it and `evaluate.py` scores it.

`experiments.py` compares models but never persists one, so nothing it trains survives the
process and there is no artifact for `local_score.py` to score. This module closes that gap:

    train.py  ->  my_experiments/baseline.joblib  ->  your_model_predict  ->  evaluate.py

What is fixed here and not configurable: 21 interpolated magnetics features -> StandardScaler ->
a model -> [PCA coefficients of psi, q95, betaN] -> psi(R,Z). Which models, and with which
hyper-parameters, is params.yaml (see `models.py`); every enabled model is fitted on the same
X and Y, and their weighted average is scored alongside them as `ensemble`.

Shots are ordered by a hash of the shot id, NOT sampled randomly like `experiments.py --source
local` does: training takes the head of that list and scoring the tail, so the split is disjoint
by construction as long as the two shares sum to under 1.
"""
from __future__ import annotations

import hashlib
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments import (
    D3D_MAGNETICS_SIGNALS,
    DEFAULT_LOCAL_DATA_DIR,
    EFIT_GRID_SIZE,
    HF_TRAIN_CONFIG,
    TargetPCA,
    _as_psirz_stack,
    interpolate_magnetics_to_efit,
)
from my_experiments.coil_field import (
    CoilBasis,
    build_basis,
    coil_pca_transform,
    fit_flux_gains,
)
from my_experiments.models import (
    DEFAULT_PARAMS_PATH,
    INPUT_MODES,
    FloatArray,
    Params,
    TargetScaler,
    load_params,
)
from my_experiments.parallel import pimap, resolve_jobs

HERE = Path(__file__).resolve().parent
ARTIFACT = HERE / "baseline.joblib"
SUBMITTED_SCALARS = ["efit_q95", "efit_beta_n"]   # -> q95, betaN
ENSEMBLE = "ensemble"                             # the reserved name of the weighted average

Artifact = dict[str, Any]
Row = Any                                         # a pandas Series or a streamed dict, alike


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


def parse_share(token: str) -> tuple[float, float]:
    """`"0.05"` -> (0.05, 1.0); `"0.05/0.1"` -> 5% of the shots, 10% of each shot's frames.

    Frames inside one shot are near-duplicates — the equilibrium moves on the current-diffusion
    timescale, hundreds of milliseconds, while EFIT frames come far faster. Measured on this data,
    53% of the psi variance is BETWEEN shots and 47% within, so at a fixed row budget the shots are
    what is worth spending it on. Thinning the frames is how you buy more of them.
    """
    parts = token.split("/")
    if len(parts) > 2:
        raise SystemExit(f"share {token!r}: expected 'shots' or 'shots/frames'")
    try:
        shots = float(parts[0])
        frames = float(parts[1]) if len(parts) == 2 else 1.0
    except ValueError as exc:
        raise SystemExit(f"share {token!r}: {exc}") from exc
    for name, v in (("shot share", shots), ("frame share", frames)):
        if not 0 < v <= 1:
            raise SystemExit(f"{name} in {token!r} must be in (0, 1], got {v}")
    return shots, frames


def thin_frames(n_frames: int, frame_share: float) -> npt.NDArray[np.intp]:
    """Evenly spaced frame indices, `frame_share` of them, at least one.

    Evenly spaced rather than random: a shot runs through ramp-up, flat-top and ramp-down, which
    are physically different regimes, and a uniform draw over 234 frames leaves clumps and gaps in
    them. A stride covers all three by construction and needs no seed to reproduce.
    """
    if frame_share >= 1:
        return np.arange(n_frames)
    n_keep = max(1, round(n_frames * frame_share))
    return np.unique(np.linspace(0, n_frames - 1, n_keep).round().astype(np.intp))


def take_share(files: list[Path], share: float, side: str) -> list[Path]:
    """A share of the list from the head or the tail, at least one shot."""
    if not 0 < share <= 1:
        raise SystemExit(f"--share must be in (0, 1], got {share}")
    n = max(1, round(len(files) * share))
    return files[:n] if side == "head" else files[-n:]


def split_train_val(files: list[Path], train_share: float,
                    val_share: float) -> tuple[list[Path], list[Path]]:
    """The training window and, immediately after it, the validation window.

    Three windows now share one sha1-ordered list: training from the head, validation right behind
    it, scoring from the tail. Validation has to come out of the head end — it is part of fitting
    (early stopping reads it every iteration), so a model that stopped on it has seen it, and only
    the untouched tail can still measure generalization.
    """
    if not 0 < train_share + val_share < 1:
        raise SystemExit(f"train share {train_share} + validation share {val_share} = "
                         f"{train_share + val_share}, which leaves no tail to score on")
    n_train = max(1, round(len(files) * train_share))
    n_val = max(1, round(len(files) * val_share))
    return files[:n_train], files[n_train:n_train + n_val]


# --------------------------------------------------------- plasma-current time base

# `magnetics_plasma_current_times` is not a per-shot axis: the same template array, always
# starting at -858.1871 ms, is stamped into every DIII-D shot, while `magnetics_time` genuinely
# varies. For the ~70% of shots recorded at 0.05 ms it is the wrong axis, so interpolating Ip onto
# `efit_times` returns pre-shot noise — 4 kA where the trace actually sits at 1000 kA. The correct
# origin for those is the shot's own `magnetics_time[0]`; the sampling step is fine either way.
# See my_experiments/eda_ip_offset.py for the evidence, and README for the summary.
IP_ON = 0.05              # |Ip| above this fraction of its peak counts as "current flowing"
IP_COVERAGE_MIN = 0.5     # the better of the two axes must reach at least this, or something is
                          # wrong with the shot rather than with the choice between them


def align_ip_times(efit_times: FloatArray, ip_times: FloatArray, ip_values: FloatArray,
                   mag_times: FloatArray) -> FloatArray:
    """Whichever Ip axis puts the current under more EFIT frames: the shipped one, or that same
    sampling re-origined at `magnetics_time[0]`.

    The correction is exact rather than fitted — keep the 0.5 ms sampling, move the origin to the
    shot's own acquisition clock. Over 150 shots it puts current under 100% of the EFIT frames of
    every affected shot, where matching waveform windows only reached 93% in the worst case. Shots
    recorded at 0.5 ms need no correction at all, and re-origining those breaks them: coverage
    drops to 0.00, measured.

    A COMPARISON, not a threshold, and the difference is not cosmetic. This used to accept the
    shipped axis at 90% coverage and force the correction below it, which quietly assumed that low
    coverage means a wrong axis. It can also mean a short EFIT window: `d3d_shot_21f23b2392` has
    15 frames spanning 160-440 ms, sitting on the current ramp-up where |Ip| is genuinely under 5%
    of its peak for half of them. Its shipped axis is right and scores 0.53, the correction scores
    0.00, and the threshold rule "corrected" it and then raised. Picking the better of the two
    reproduces every other decision exactly — 761 corrected and 342 left alone over 1104 shots,
    zero changed — so the floor below is what still catches a genuinely broken shot. No shot in
    that sample comes near it.
    """
    peak = np.abs(ip_values).max()
    if peak <= 0:
        raise ValueError("plasma current is identically zero")

    def coverage(axis: FloatArray) -> float:
        return float((np.abs(np.interp(efit_times, axis, ip_values)) > IP_ON * peak).mean())

    corrected = mag_times[0] + (ip_times - ip_times[0])
    as_shipped, as_corrected = coverage(ip_times), coverage(corrected)
    if max(as_shipped, as_corrected) < IP_COVERAGE_MIN:
        raise ValueError(
            f"cannot align the plasma current: the shipped axis puts current under "
            f"{as_shipped:.0%} of the EFIT frames and re-origining it at magnetics_time[0] = "
            f"{mag_times[0]:.0f} ms reaches {as_corrected:.0%}, both below {IP_COVERAGE_MIN:.0%}. "
            f"Ip axis starts at {ip_times[0]:.0f} ms and spans "
            f"{ip_times[-1] - ip_times[0]:.0f} ms; EFIT window is "
            f"[{efit_times.min():.0f}, {efit_times.max():.0f}] ms over {efit_times.size} frames."
        )
    return ip_times if as_shipped >= as_corrected else corrected


# --------------------------------------------------------------------------- features

def inputs_only_shot(row: Row) -> dict[str, Any]:
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

    magnetics: dict[str, dict[str, npt.NDArray[Any]]] = {}
    for sig in D3D_MAGNETICS_SIGNALS:
        col = f"magnetics_{sig}"
        if col not in row:
            raise ValueError(f"row has no signal {col}; all "
                             f"{len(D3D_MAGNETICS_SIGNALS)} are expected: {D3D_MAGNETICS_SIGNALS}")
        times = ip_times if sig == "plasma_current" else shared
        magnetics[sig] = {"values": np.asarray(row[col], dtype=np.float32), "times": times}

    return {"efit_times": efit_times, "magnetics": magnetics}


# The geometry and time columns inference reads verbatim — small, and needed as they are.
GEOMETRY_COLUMNS: tuple[str, ...] = (
    "source", "efit_times", "efit_grid_R", "efit_grid_Z",
    "coil_input_column", "coil_R", "coil_Z", "coil_width", "coil_height",
)

# Where `slim_row` parks the already-interpolated magnetics, and where `features_for_row` looks
# for them before doing the work again.
FEATURES_KEY = "magnetics_features"


def slim_row(row: Row) -> dict[str, Any]:
    """The inference inputs of one row, detached from the rest of it and reduced to size.

    A dataset row retains 101 MB, of which inference reads 79 — and 77 of those are the raw
    magnetics traces, 480256 samples per signal at the 0.05 ms acquisition. The first thing
    inference does with them is interpolate them onto `efit_times`, ~300 points, so what it
    actually needs is the (T, 21) result: 0.03 MB. Everything else — the flux map, which is a
    TARGET, and the Thomson profiles this model does not use — is not read at all.

    That distinction only matters because `local_score` holds one row per scored shot for the
    whole run: the metric pools R² across the fold, so nothing can be finalized shot by shot. At
    101 MB a 704-shot fold needed 71 GB before a frame had been scored, on a machine with 62, and
    went to swap. Reduced, the same fold holds about 7 GB.

    Detached, not a view: the parquet row and everything hanging off it must become collectable.
    The interpolation is the same call inference would have made, so the features are identical
    to the ones a full row produces — this is a memory change, not a numerical one.
    """
    out: dict[str, Any] = {}
    for col in GEOMETRY_COLUMNS:
        if col not in row:
            raise ValueError(f"row has no column {col}, which inference reads — is this a "
                             f"DIII-D row?")
        v = row[col]
        out[col] = v if isinstance(v, str) else np.array(v, copy=True)
    out[FEATURES_KEY] = features_for_row(row)
    return out


def features_for_row(row: Row) -> FloatArray:
    """(T, 21) magnetics features on the EFIT time base, exactly len(efit_times) rows."""
    if FEATURES_KEY in row:
        ready: FloatArray = np.asarray(row[FEATURES_KEY])
        want = (len(np.asarray(row["efit_times"])), len(D3D_MAGNETICS_SIGNALS))
        if ready.shape != want:
            raise ValueError(f"precomputed {FEATURES_KEY} has shape {ready.shape}, expected "
                             f"{want}")
        return ready
    shot = inputs_only_shot(row)
    feats: FloatArray = interpolate_magnetics_to_efit(shot)
    if feats.shape != (len(shot["efit_times"]), len(D3D_MAGNETICS_SIGNALS)):
        raise ValueError(f"features {feats.shape}, expected "
                         f"({len(shot['efit_times'])}, {len(D3D_MAGNETICS_SIGNALS)})")
    if not np.isfinite(feats).all():
        raise ValueError(f"features contain {int((~np.isfinite(feats)).sum())} non-finite values")
    return feats


# --------------------------------------------------------------------- the coil field

# Everything the pipeline needs to reproduce the decomposition at inference. The scaled maps are
# stored rather than rebuilt: 19 x 65 x 65 is 642 kB, against re-evaluating elliptic integrals in
# every worker process, and it removes any chance of training and inference disagreeing about the
# geometry. The grid travels with them so a row on a different one is an error, not a silent
# mismatch.
CoilPlan = dict[str, Any]


def coil_plan(basis: CoilBasis, params: Params, X: FloatArray, Y: FloatArray) -> CoilPlan:
    """Fit the flux gains and, if the inputs need it, the coil-flux principal directions."""
    index = [D3D_MAGNETICS_SIGNALS.index(c.removeprefix("magnetics_")) for c in basis.columns]
    currents = X[:, index].astype(np.float64)
    gains = fit_flux_gains(basis.maps, currents, Y)
    maps = basis.maps * gains[:, None, None]

    mean = w = None
    if params.inputs in ("coil_pca", "both"):
        mean, w = coil_pca_transform(maps, currents, params.n_coil_pca)
    return {
        "subtract": params.subtract_coil_field, "inputs": params.inputs,
        "columns": list(basis.columns), "index": index, "gains": gains, "maps": maps,
        "grid_R": basis.grid_R, "grid_Z": basis.grid_Z, "machine": basis.machine,
        # Signals with no poloidal-plane rectangle, so no map and no principal direction: they
        # survive `coil_pca` verbatim or the plasma current would be thrown away with the wiring.
        "passthrough": [i for i, sig in enumerate(D3D_MAGNETICS_SIGNALS)
                        if f"magnetics_{sig}" not in basis.columns],
        "cur_mean": mean, "coil_pca_W": w,
    }


def coil_flux(plan: CoilPlan, feats: FloatArray) -> FloatArray:
    """(T, 65, 65) the coil field of these frames, in the stored flux convention."""
    maps: FloatArray = plan["maps"]
    return np.tensordot(feats[:, plan["index"]].astype(np.float64), maps, axes=(1, 0))


def build_inputs(plan: CoilPlan, feats: FloatArray) -> FloatArray:
    """(T, n_features) — what the models actually see, identical in training and in inference."""
    mode = plan["inputs"]
    if mode == "currents":
        return feats
    coil = (feats[:, plan["index"]].astype(np.float64) - plan["cur_mean"]) @ plan["coil_pca_W"]
    if mode == "coil_pca":
        return np.hstack([coil, feats[:, plan["passthrough"]]])
    if mode == "both":
        return np.hstack([feats, coil])
    raise ValueError(f"unknown input mode {mode!r}; known: {sorted(INPUT_MODES)}")


def basis_for_row(row: Row) -> CoilBasis:
    """The coil basis of this row's machine, built once per process.

    Keyed on the geometry itself, not on the machine name: two rows that disagree about where the
    coils are must not share a basis, and on this dataset they never do.
    """
    key = "|".join(str(np.asarray(row[c]).tobytes() if c != "coil_input_column"
                       else list(row[c]))
                   for c in ("coil_input_column", "coil_R", "coil_Z", "coil_width", "coil_height",
                             "efit_grid_R", "efit_grid_Z"))
    if key not in _BASIS_CACHE:
        _BASIS_CACHE[key] = build_basis(row)
    return _BASIS_CACHE[key]


_BASIS_CACHE: dict[str, CoilBasis] = {}


def check_grid(plan: CoilPlan, row: Row) -> None:
    """The artifact's maps only mean anything on the grid they were built on."""
    for name, want in (("efit_grid_R", plan["grid_R"]), ("efit_grid_Z", plan["grid_Z"])):
        got = np.asarray(row[name], dtype=np.float64)
        if got.shape != want.shape or not np.allclose(got, want):
            raise ValueError(
                f"{name} of this row spans {got.min():.3f}..{got.max():.3f} in {got.size} points, "
                f"but the coil maps in {ARTIFACT} were built on "
                f"{want.min():.3f}..{want.max():.3f} in {want.size}. The decomposition cannot be "
                f"transferred between grids."
            )


# --------------------------------------------------------------------------- training

def _read_training_shot(path: Path,
                        frame_share: float = 1.0) -> tuple[FloatArray, FloatArray, FloatArray]:
    """(features, psi, scalars) for one training shot, with every shape checked.

    `frame_share` thins the frames as they are read, before anything is concatenated: the point of
    thinning is to afford more shots, and holding every frame of every shot in memory first would
    defeat it.
    """
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

    keep = thin_frames(T, frame_share)
    return feats[keep], psi[keep], scal[keep]


def _read_task(args: tuple) -> tuple[FloatArray, FloatArray, FloatArray]:
    return _read_training_shot(*args)


def _read_shots(files: list[Path], desc: str, frame_share: float = 1.0,
                jobs: int = 0) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Every kept frame of every shot, concatenated: features, psi, and the two scalar targets.

    Shots are independent, so this is the easiest parallelism in the repo: parquet decode plus
    interpolation, one process per shot, results collected in order so the frame order does not
    depend on the core count.
    """
    n = resolve_jobs(jobs, len(files))
    tasks = [(path, frame_share) for path in files]
    X_parts, Y_parts, S_parts = [], [], []
    bar = tqdm(pimap(_read_task, tasks, n), total=len(files),
               desc=desc if n == 1 else f"{desc} x{n}", unit="shot")
    for feats, psi, scal in bar:
        X_parts.append(feats)
        Y_parts.append(psi)
        S_parts.append(scal)
        # Frames accumulate in the postfix: it shows both the sample size and that progress is
        # alive, without a line per shot.
        bar.set_postfix(frames=sum(len(x) for x in X_parts))
    return np.concatenate(X_parts), np.concatenate(Y_parts), np.concatenate(S_parts)


def train(share: str, val_share: str, local_data_dir: Path, config: str,
          params_path: Path = DEFAULT_PARAMS_PATH, jobs: int = 0) -> Artifact:
    """Fit every enabled model of params.yaml on the head of the split, and save them together.

    Both shares are `"shots"` or `"shots/frames"` (see parse_share). `val_share` is the window
    right behind the training one; it never enters a fit as data, it is what CatBoost and the MLP
    stop on and pick their best iteration by.
    """
    params: Params = load_params(params_path)
    shot_share, frame_share = parse_share(share)
    val_shot_share, val_frame_share = parse_share(val_share)
    all_files = sorted_shots(local_data_dir, config)
    files, val_files = split_train_val(all_files, shot_share, val_shot_share)
    print(f"Training on {len(files)} shots ({shot_share:.1%} of {len(all_files)}, "
          f"{frame_share:.0%} of their frames), validating on {len(val_files)} "
          f"({val_shot_share:.1%}, {val_frame_share:.0%} of frames), head of the list, "
          f"ordered by sha1")
    print(f"Models from {params.path}: {', '.join(params.models)}  "
          f"(ensemble: {', '.join(f'{k} x {w:.2f}' for k, w in params.ensemble.items())})")

    X, Y, S = _read_shots(files, "reading train shots", frame_share, jobs)
    Xv, Yv, Sv = _read_shots(val_files, "reading val shots", val_frame_share, jobs)

    print(f"  X {X.shape}  Y {Y.shape}  S {S.shape}   val X {Xv.shape}")

    # The metric's own denominator, per frame: sum over pixels of (psi - m)^2 with m the single
    # FLAT mean of the training flux — not the mean image. Computed here, on the flux as stored,
    # BEFORE any decomposition: the metric scores total psi, so the balance the target scaling
    # strikes between psi and the scalars must not move when the target becomes a residual.
    # By sums rather than by materializing (Y - m), which would be a 550 MB float64 temporary.
    psi_total = float(Y.sum(dtype=np.float64))
    psi_sumsq = float(np.einsum("ijk,ijk->", Y, Y, dtype=np.float64))
    psi_ss_tot = (psi_sumsq - psi_total ** 2 / Y.size) / len(Y)

    basis = basis_for_row(pd.read_parquet(files[0]).iloc[0])
    plan = coil_plan(basis, params, X, Y)
    print(f"  coil field: {len(plan['columns'])} maps on {plan['machine']}, gains "
          f"{plan['gains'].min():.2f}..{plan['gains'].max():.2f}; "
          f"subtract={plan['subtract']}, inputs={plan['inputs']}")
    if plan["subtract"]:
        # In place and in chunks: at production scale a second copy of Y is 2 GB, and the whole
        # point of thinning frames was to afford more shots with the memory we have.
        for Yi, Xi in ((Y, X), (Yv, Xv)):
            for lo in range(0, len(Yi), 4096):
                Yi[lo:lo + 4096] -= coil_flux(plan, Xi[lo:lo + 4096]).astype(Yi.dtype)
        print(f"  targets are now the plasma part alone: psi std "
              f"{np.sqrt(psi_ss_tot / Y[0].size):.4f} -> {Y.std():.4f} Wb/rad")

    # Every preprocessing step is fitted on the training frames alone and merely applied to the
    # validation ones: a scaler or a PCA that had seen the validation set would make early stopping
    # stop on a number that is partly its own reflection.
    F, Fv = build_inputs(plan, X), build_inputs(plan, Xv)
    print(f"  inputs: {F.shape[1]} features per frame ({plan['inputs']})")
    scaler = StandardScaler().fit(F)
    Xs = scaler.transform(F)
    Xvs = scaler.transform(Fv)

    if params.n_pca > min(len(Xs), EFIT_GRID_SIZE ** 2):
        raise ValueError(f"n_pca {params.n_pca} exceeds what the data supplies: {len(Xs)} frames, "
                         f"{EFIT_GRID_SIZE ** 2} pixels. Raise --share or lower n_pca in "
                         f"{params.path}.")
    pca = TargetPCA(n_components=params.n_pca)
    # Pin the randomized SVD. sklearn chooses that solver for 50 components out of 4225 pixels and
    # TargetPCA leaves random_state at None, so unpinned it fits different components on every run
    # — different targets for every model, and a score that moves without the code changing.
    # Set on the inner estimator rather than by editing the organizers' experiments.py.
    pca.pca.random_state = params.pca_seed
    pca.fit(Y)
    print(f"  PCA: {params.n_pca} components explain "
          f"{np.cumsum(pca.explained_variance_ratio)[-1] * 100:.1f}% of the psi variance")

    # One target matrix for every model: [PCA coefficients | q95 | betaN]. Fitting the flux map
    # and the two scalars jointly is what makes the models interchangeable and averageable.
    Tgt = np.hstack([np.asarray(pca.transform(Y), dtype=np.float64), S])
    Tgt_val = np.hstack([np.asarray(pca.transform(Yv), dtype=np.float64), Sv])
    if Tgt.shape != (len(Xs), params.n_targets):
        raise ValueError(f"targets {Tgt.shape}, expected {(len(Xs), params.n_targets)}")
    if Tgt_val.shape != (len(Xvs), params.n_targets):
        raise ValueError(f"validation targets {Tgt_val.shape}, expected "
                         f"{(len(Xvs), params.n_targets)}")

    # Scaled once, for every model alike — the scaling decides which loss they minimize, so it is
    # a property of the problem, not of any one estimator. Ridge is provably indifferent to it
    # (it is separable per output and the scale cancels in the solution), which makes its score a
    # check that this step changed nothing it should not have.
    target_scaler = TargetScaler(params.n_pca).fit(Tgt, psi_ss_tot)
    Tgt = target_scaler.transform(Tgt)
    Tgt_val = target_scaler.transform(Tgt_val)
    var = Tgt.var(axis=0)
    print(f"  targets: {params.n_targets} outputs, scaled as the metric weights them; "
          f"loss share psi {var[:params.n_pca].sum() / var.sum():.0%} / "
          f"scalars {var[params.n_pca:].sum() / var.sum():.0%}")

    for name, model in params.models.items():
        print(f"\n  fitting {name} ({model.kind}) on {Xs.shape[0]} frames "
              f"-> {params.n_targets} targets, stopping on {Xvs.shape[0]} validation frames")
        t0 = time.perf_counter()
        model.fit(Xs, Tgt, Xvs, Tgt_val)
        print(f"  {name}: fitted in {time.perf_counter() - t0:.1f} s{model.fit_report()}")

    artifact: Artifact = {
        "scaler": scaler, "pca": pca, "target_scaler": target_scaler,
        "models": params.models, "ensemble": params.ensemble,
        "n_pca": params.n_pca, "coil": plan,
        # The whole params file, verbatim: the artifact says what produced it even after the file
        # on disk has moved on.
        "params_yaml": params.path.read_text(encoding="utf-8"),
        # Filenames, not indices: evaluate.py intersects them directly, and the check stays
        # correct even if the data directory grows.
        "train_files": [p.name for p in files],
        "val_files": [p.name for p in val_files],
        "n_train_shots": len(files), "train_share": shot_share, "train_frame_share": frame_share,
        "n_val_shots": len(val_files), "val_share": val_shot_share,
        "val_frame_share": val_frame_share, "config": config,
    }
    joblib.dump(artifact, ARTIFACT)
    print(f"\nSaved {ARTIFACT}: {', '.join(model_names(artifact))}")
    print("Now score it:  uv run python my_experiments/evaluate.py --share 0.001")
    return artifact


# --------------------------------------------------------------------------- inference

_CACHE: Artifact | None = None


def _load() -> Artifact:
    global _CACHE
    if _CACHE is None:
        if not ARTIFACT.exists():
            raise FileNotFoundError(
                f"{ARTIFACT} not found — train first:\n"
                f"  uv run python my_experiments/train.py --share 0.01"
            )
        _CACHE = joblib.load(ARTIFACT)
    return _CACHE


def model_names(artifact: Artifact | None = None) -> list[str]:
    """Every scoreable name in the artifact: each fitted model, then the ensemble."""
    art = artifact if artifact is not None else _load()
    return [*art["models"], ENSEMBLE]


def _predict_targets(art: Artifact, model: str, Xs: FloatArray) -> FloatArray:
    """(T, n_pca + 2) — one model's targets, or the weighted average of the ensemble members.

    Averaging targets and averaging the flux maps they decode to are the same thing: PCA's
    inverse transform is affine and the weights sum to 1, so the mean image is added back exactly
    once either way.
    """
    inverse = art["target_scaler"].inverse_transform
    if model == ENSEMBLE:
        out = np.zeros((len(Xs), art["n_pca"] + len(SUBMITTED_SCALARS)), dtype=np.float64)
        for name, weight in art["ensemble"].items():
            out += weight * inverse(art["models"][name].predict(Xs))
        return out
    if model not in art["models"]:
        raise KeyError(f"no model {model!r} in {ARTIFACT}; it holds {model_names(art)}")
    return inverse(art["models"][model].predict(Xs))


def predict_row(row: Row, source: str = "DIII-D", model: str = ENSEMBLE) -> dict[str, FloatArray]:
    """Predict {psirz (T,65,65), q95 (T,), betaN (T,)} for one dataset row.

    `model` selects one member of the zoo by name, or the default `ensemble` — their weighted
    average. Uses only `magnetics_*` and `efit_times`, never the `efit_*` targets, which are
    present in training rows and would make any local score meaningless.
    """
    art = _load()
    if source != "DIII-D":
        raise NotImplementedError(
            f"the model is trained on DIII-D only, but the row is marked source={source!r}. "
            f"MAST has its own coil set and needs its own fit — returning zeros silently is not "
            f"an option, it would look like a working prediction."
        )

    T = len(np.asarray(row["efit_times"]))
    if "coil" not in art:
        raise KeyError(
            f"{ARTIFACT} predates the coil-field decomposition and does not say which features it "
            f"was fitted on. Retrain rather than guess: uv run python my_experiments/train.py "
            f"--share 0.05/0.2 --val-share 0.05/0.2"
        )
    plan: CoilPlan = art["coil"]
    feats = features_for_row(row)                          # features_for_row demands finiteness
    if plan["subtract"] or plan["inputs"] != "currents":
        check_grid(plan, row)
    Xs = art["scaler"].transform(build_inputs(plan, feats))
    P = _predict_targets(art, model, Xs)

    n_pca = art["n_pca"]
    psirz = np.asarray(art["pca"].inverse_transform(P[:, :n_pca]))
    if plan["subtract"]:
        # What was taken off the targets, put back on the prediction — the same maps, the same
        # gains, the same currents. Exact, so the decomposition can only change how well the
        # regression does its half, never introduce an error of its own.
        psirz = psirz + coil_flux(plan, feats).astype(psirz.dtype)
    out: dict[str, FloatArray] = {"psirz": psirz}
    for j, key in enumerate(["q95", "betaN"]):
        out[key] = P[:, n_pca + j].astype(np.float32)

    for key, want in [("psirz", (T, EFIT_GRID_SIZE, EFIT_GRID_SIZE)), ("q95", (T,)),
                      ("betaN", (T,))]:
        if out[key].shape != want:
            raise ValueError(f"{key}: shape {out[key].shape}, the contract requires {want}")
        if not np.isfinite(out[key]).all():
            raise ValueError(f"{key}: the prediction contains non-finite values")
    return out


# No CLI here on purpose: `train.py` is the one way to train and `evaluate.py` the one way to
# score, so there is never a second set of defaults to keep in sync. This module is the pipeline.
