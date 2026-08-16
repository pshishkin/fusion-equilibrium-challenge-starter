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

import functools
import hashlib
import os
import sys
import time
import warnings
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
from my_experiments import shot_cache
from my_experiments.coil_field import (
    CoilBasis,
    build_basis,
    coil_pca_transform,
    fit_flux_gains,
)
from my_experiments.models import (
    DEFAULT_PARAMS_PATH,
    DERIV_MODES,
    INPUT_MODES,
    FloatArray,
    Params,
    TargetScaler,
    load_params,
)
from my_experiments.monitor import build_monitor
from my_experiments.parallel import pimap, release, resolve_jobs
from my_experiments.progress import SHOT_EVERY, bar_kwargs
from my_experiments.target_metric import (
    CONS_SCALARS,
    boundary_form,
    factor,
    jacobian_form,
    metric_form,
    scorer_context,
)

HERE = Path(__file__).resolve().parent
# Where a fit is written and read. Selectable by environment variable, the same way shot_cache
# already does it, so several fits can coexist under their own names — which is what averaging
# across salts needs, since each salt's fit is a whole pipeline of its own and not a member of
# anyone else's artifact.
ARTIFACT = Path(os.environ.get("FUSION_ARTIFACT", HERE / "baseline.joblib"))

# `salts:a+b` as a model name averages the DECODED predictions of `baseline_a.joblib` and
# `baseline_b.joblib`. It has to be decoded rather than coefficient space: every fit has its own
# PCA basis, its own target scaler and its own coil gains, so their coefficients are not
# commensurable and averaging them would be adding up numbers that mean different things.
SALTS_PREFIX = "salts:"

# `psi=mlp+seq,qb=mlp` — one combination for the flux coefficients, another for q95 and betaN.
BLOCK_SEP, PSI_KEY, QB_KEY = ",", "psi=", "qb="
# `mlp*0.3+seq*0.7` — how a member carries an unequal weight in a combination assembled at scoring
# time. `*` and not `:` because a model name may not contain it and a shell will not eat it inside
# the quotes evaluate.py's --models needs anyway.
WEIGHT_SEP = "*"
SUBMITTED_SCALARS = ["efit_q95", "efit_beta_n"]   # -> q95, betaN
ENSEMBLE = "ensemble"                             # the reserved name of the weighted average

Artifact = dict[str, Any]
Row = Any                                         # a pandas Series or a streamed dict, alike


# --------------------------------------------------------------------------- shot ordering

def shot_key(path: Path, salt: int = 0) -> str:
    """Sort key: sha1 of the shot id, optionally salted to draw a different split.

    hashlib, NOT the builtin hash(): that one is salted per process, so the order would change
    between the train run and the evaluate run and the split would stop being reproducible —
    silently. sha1 of the filename gives the same order every time, on any machine.

    `salt` reshuffles which shots land in the training, validation and scoring windows, at the same
    shares. That is the replicate that matters: the MLP seed only varies the optimisation, while
    the salt varies the DATA, which is what a claim like "this change helps" is really about. Salt
    0 hashes the bare filename, so it is the split every number recorded so far was measured on and
    stays the canonical one; only compare absolute scores within one salt.
    """
    stem = path.stem if salt == 0 else f"{salt}:{path.stem}"
    return hashlib.sha1(stem.encode()).hexdigest()


def sorted_shots(local_data_dir: Path = DEFAULT_LOCAL_DATA_DIR,
                 config: str = HF_TRAIN_CONFIG, salt: int = 0) -> list[Path]:
    """Every shot of the config, deterministically shuffled. Training takes the head of the list
    and evaluation the tail, so the windows stay disjoint while the shares sum to under 1."""
    data_dir = Path(local_data_dir) / "data" / config
    files = sorted(data_dir.glob("*.parquet"), key=lambda p: shot_key(p, salt))
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


def reserve_holdout(files: list[Path], share: float) -> tuple[list[Path], list[Path]]:
    """Split off a set of shots no salt may train on, and the pool that remains.

    An ensemble whose members were fitted under DIFFERENT salts cannot be scored the ordinary way:
    each member trains on 99% of the shots, so almost every shot in one salt's scoring tail is in
    another's training set, and `evaluate.py` refuses the run — correctly. Averaging such members
    is still legitimate for a SUBMISSION, whose test set no salt has seen; what is missing is a
    local number, and choosing between ensembles without one is guessing.

    So the holdout is carved from the FIXED salt-0 order, before the per-salt shuffle, and every
    salt splits only what is left. The same shots are then untouched by every member, whatever
    salt fitted it, and the ensemble is measurable.
    """
    if not 0 <= share < 1:
        raise SystemExit(f"holdout share must be in [0, 1), got {share}")
    if not share:
        return [], files
    n = max(1, round(len(files) * share))
    keep = {p.name for p in sorted(files, key=lambda p: shot_key(p, 0))[-n:]}
    return ([p for p in files if p.name in keep], [p for p in files if p.name not in keep])


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


# Half-width of the raw-grid derivative, in milliseconds. The magnetics are sampled at 0.05 ms, so
# an adjacent-sample difference there is mostly digitizer noise, while the quantity that acts
# physically — the loop voltage that drives the plasma current — lives on the plasma's own response
# time. One millisecond is twenty samples: short against the equilibrium, long against the noise.
RAW_DERIV_HALF_MS = 1.0

# Every feature block, always computed and always cached. Which of them the models SEE is
# `features.derivatives` in params.yaml, applied in `build_inputs` — so switching arms costs a
# training run and not a 25 GB cache rebuild.
DERIV_BLOCKS = ("raw", "interp")

# Which signals get a derivative. The physics names two of the twenty-one, and taking all of them
# tests a weaker claim than the one that was argued for:
#   ECOILA          the ohmic solenoid. dI/dt is the LOOP VOLTAGE, which is what drives the plasma
#                   current — the level drives nothing.
#   plasma_current  dIp/dt separates ramp-up, flat-top and ramp-down, regimes with different
#                   current profiles, and `li` is a current-profile quantity.
# The eighteen F-coils are the eddy-current argument, which is real but weaker — and weaker still
# since the per-shot constant turned out not to be explained by current history at all.
# `bcoil` is toroidal and drives NO poloidal flux, which is why it has no rectangle in the coil
# basis either; its derivative cannot inform psi and is excluded from both sets.
DERIV_SIGNAL_SETS = {
    "driving": ("ECOILA", "plasma_current"),
    "poloidal": tuple(s for s in D3D_MAGNETICS_SIGNALS if s != "bcoil"),
}
N_SIGNALS = len(D3D_MAGNETICS_SIGNALS)
# Twelve Thomson summaries follow the derivative blocks; see N_THOMSON below.
# Two columns: the gap back to the previous frame and forward to the next. They sit between the
# derivative blocks and the Thomson block, so both of the slices that address those keep working —
# the derivatives are counted from the front and Thomson from the back.
N_GAPS = 2

# A19 / A22. Time constants, in milliseconds, of the leaky integrals of dI/dt that follow the gap
# columns. The physics is one line: a conducting loop around a changing coil current obeys
# L dIv/dt + R Iv = -M dIc/dt, so the induced current IS an exponential moving average of dIc/dt
# with time constant tau = L/R. A bank of them is therefore a hand-built observer of the vessel
# state — the one quantity a memoryless model provably cannot form, and the leading candidate for
# why the last decile of a shot costs 24% of the geometry loss with only ordinary sensitivity.
#
# The constants were first set at 10 / 50 / 200 ms from the vessel L/R argument alone, and C6 then
# measured that the sequence model — which learns its own time constants and is the one member
# whose help concentrates at the end of a shot — chose a MEDIAN of 90.7 frames forward and 122.0
# backward, i.e. 1.8 and 2.4 SECONDS, with the channels its output actually reads longer still.
# That is an order of magnitude above the vessel bracket and sits where current-profile diffusion
# does, which is `li`, one of the two most expensive scalars in the whole composite.
#
# So the bank spans both rather than choosing: 20 ms is one EFIT frame, 2500 ms is a third of the
# longest shot, and the spacing is geometric because a time constant is a ratio. Four numbers, not
# a fit — the model weights them.
VESSEL_TAUS_MS = (20.0, 100.0, 500.0, 2500.0)
N_VESSEL = N_SIGNALS * len(VESSEL_TAUS_MS)
FEATURE_WIDTH = (N_SIGNALS * (1 + len(DERIV_BLOCKS)) + N_GAPS + N_VESSEL
                 + 27 + 2 * 16)   # = N_THOMSON_BLOCK


# The EFIT clock, measured over 202 shots and 44729 steps: 97.86% of steps are 20 ms and every
# step is a multiple of it, so this is one grid with frames missing rather than a rate that varies
# by shot. 77% of shots drop at least one frame and the largest hole is 2580 ms.
FRAME_STEP_MS = 20.0
# Where a hole stops being a number and becomes a category. The gap is clipped here because 2.14%
# of steps are longer than one and the longest is 129 — standardized without a clip, that tail
# would carry ten times the variance of the 1-versus-2 distinction that actually occurs, and
# squash the common case into a tenth of a standard deviation.
MAX_GAP_STEPS = 5.0


def _frame_gaps(efit_times: FloatArray) -> FloatArray:
    """(T, 2) — the step back to the previous frame and forward to the next, in base steps.

    In units of the 20 ms clock rather than in milliseconds, so a dropped frame reads as "this step
    was three frames long" instead of as an unfamiliar 60. The first frame has no predecessor and
    the last no successor; both are given 1.0, an ordinary step, because a sequence model already
    knows where its sequence ends and an out-of-range marker there would only be noise.
    """
    t = np.asarray(efit_times, dtype=np.float64).ravel()
    d = np.clip(np.diff(t) / FRAME_STEP_MS, 1.0, MAX_GAP_STEPS)
    return np.column_stack([np.concatenate([[1.0], d]), np.concatenate([d, [1.0]])])


def _raw_derivatives(shot: dict[str, Any]) -> FloatArray:
    """d/dt of each signal on ITS OWN time base, then interpolated onto the EFIT times.

    This order is the point. Differentiating after interpolation measures the interpolator: the
    EFIT frames are 20 ms apart while the signal is sampled at 0.05 ms, so everything between two
    frames has already been thrown away by the time the difference is taken.

    A centred difference over a fixed PHYSICAL half-width, not over one sample: the sampling
    interval varies between shots (~70% are at 0.05 ms), so a fixed index offset would mean a
    different derivative on different shots.
    """
    efit_times = shot["efit_times"]
    out = []
    for sig in D3D_MAGNETICS_SIGNALS:
        if sig not in shot["magnetics"]:
            out.append(np.zeros(len(efit_times), dtype=np.float32))
            continue
        mag = shot["magnetics"][sig]
        t = np.asarray(mag["times"], dtype=np.float64)
        v = np.asarray(mag["values"], dtype=np.float64)
        if len(t) < 3:
            raise ValueError(f"signal {sig} has {len(t)} samples, too few to differentiate")
        step = float(np.median(np.diff(t)))
        if not step > 0:
            raise ValueError(f"signal {sig} has a median sample step of {step}, expected positive")
        k = max(1, round(RAW_DERIV_HALF_MS / step))
        lo = np.clip(np.arange(len(t)) - k, 0, len(t) - 1)
        hi = np.clip(np.arange(len(t)) + k, 0, len(t) - 1)
        dt = t[hi] - t[lo]
        d = np.where(dt > 0, (v[hi] - v[lo]) / np.maximum(dt, 1e-30), 0.0)
        out.append(np.interp(efit_times, t, d).astype(np.float32))
    return np.column_stack(out)


def _vessel_currents(shot: dict[str, Any]) -> FloatArray:
    """(T, N_SIGNALS * len(VESSEL_TAUS_MS)) — leaky integrals of each signal's dI/dt.

    One exponential moving average per (signal, tau), computed on the signal's OWN 0.05 ms base and
    then interpolated to the EFIT frames, for the same reason `_raw_derivatives` is: tau = 10 ms is
    half of one EFIT step, so an average taken on the frame grid would be aliasing rather than
    filtering.

    The recursion is `y_i = a*y_{i-1} + (1-a)*x_i` with `a = exp(-step/tau)`, the exact solution of
    the first-order ODE over one sample interval. `step` is the shot's own MEDIAN sampling interval,
    the same quantity `_raw_derivatives` builds its half-width from — a per-sample `a` would be more
    faithful on the handful of irregular samples and cannot be run through a C-speed recursive
    filter, and 30 million Python iterations per shot is not a trade worth making for that.

    Initialised at the steady state for `x[0]` rather than at zero: a shot starts with the vessel
    in equilibrium with whatever the coils were already doing, and starting at zero would inject a
    transient of the model's own making into the first tau of every shot.
    """
    from scipy.signal import lfilter, lfilter_zi

    efit_times = shot["efit_times"]
    out = []
    for sig in D3D_MAGNETICS_SIGNALS:
        if sig not in shot["magnetics"]:
            out.append(np.zeros((len(efit_times), len(VESSEL_TAUS_MS)), dtype=np.float32))
            continue
        mag = shot["magnetics"][sig]
        t = np.asarray(mag["times"], dtype=np.float64)
        v = np.asarray(mag["values"], dtype=np.float64)
        if len(t) < 3:
            raise ValueError(f"signal {sig} has {len(t)} samples, too few to integrate")
        step = float(np.median(np.diff(t)))
        if not step > 0:
            raise ValueError(f"signal {sig} has a median sample step of {step}, expected positive")
        k = max(1, round(RAW_DERIV_HALF_MS / step))
        lo = np.clip(np.arange(len(t)) - k, 0, len(t) - 1)
        hi = np.clip(np.arange(len(t)) + k, 0, len(t) - 1)
        dt = t[hi] - t[lo]
        x = np.where(dt > 0, (v[hi] - v[lo]) / np.maximum(dt, 1e-30), 0.0)
        cols = []
        for tau in VESSEL_TAUS_MS:
            a = float(np.exp(-step / tau))
            b, denom = np.array([1.0 - a]), np.array([1.0, -a])
            y, _ = lfilter(b, denom, x, zi=lfilter_zi(b, denom) * x[0])
            cols.append(np.interp(efit_times, t, y).astype(np.float32))
        out.append(np.column_stack(cols))
    return np.hstack(out)


def _interp_derivatives(feats: FloatArray, efit_times: FloatArray) -> FloatArray:
    """d/dt of the already-interpolated features, on the EFIT time base.

    The cheap arm, and not obviously the wrong one: it measures the rate on the timescale the
    equilibrium actually moves on rather than on the digitizer's. `np.gradient` is given the times
    themselves because the EFIT step is 20 ms for 98% of intervals and up to 900 ms for the rest.
    """
    if len(feats) < 2:
        return np.zeros_like(feats)
    return np.gradient(feats.astype(np.float64), efit_times, axis=0).astype(feats.dtype)


# Thomson scattering, reduced to a few numbers per frame. Pressure is one of the two free
# functions of Grad-Shafranov and the one thing the magnetics genuinely cannot see: coil currents
# constrain the flux at the boundary, not how the pressure sits inside.
#
# The geometry is not what the name suggests, and it decides what is computable. Measured on the
# shipped rows: the CORE system's 44 channels all sit at R = 1.940 and differ in Z — a VERTICAL
# chord through the upper half of the plasma. The EDGE system's 10 channels sit near Z = -0.06 and
# span R = 1.68..2.06 — a horizontal cut across the outboard edge. So the radial derivative that
# Grad-Shafranov actually asks for, dp/dR, exists on the EDGE system and nowhere else, and it lands
# exactly on the pedestal.
#
# Missing data is encoded as an exact ZERO, not as nan — the trap AGENTS warns about. Inside the
# EFIT window 14.1% of core values are zero; outside it, where there is no plasma, most are. Every
# reduction below therefore runs on a validity mask, never on the raw array.
N_THOMSON = 27          # 13 per system + a validity flag; matches FEATURE_WIDTH above
# Points of the resampled RAW pressure profile, per system. The summaries above compress a
# 44-channel chord into three numbers, and the group sweep showed temperature summaries are
# worth nothing while pressure carries everything — so the question is whether the SHAPE that
# a summary throws away is what matters. Grad-Shafranov reads p(psi), not its peak.
N_PROFILE = 16
# The whole Thomson block as it sits in a cached row: summaries, the validity flag, then the two
# raw profiles. Kept separate from N_THOMSON so that adding to one end cannot silently truncate
# the other — which it did once, on the day the profiles were added.
N_THOMSON_BLOCK = 27 + 2 * N_PROFILE
THOMSON_MIN_LIVE = 3


# The Thomson summaries, in groups that `features.thomson_groups` can select. Order matters: it is
# the column order of every cached row, and the group slices below index into it.
THOMSON_STATS = (
    "peak_te", "at_te", "int_te", "slope_te",
    "peak_ne", "int_ne", "slope_ne",
    "peak_p", "at_p", "int_p", "slope_p",
    "peaking_te", "width_te",
)
THOMSON_GROUPS = {
    "te": ("peak_te", "int_te", "slope_te"),
    "ne": ("peak_ne", "int_ne", "slope_ne"),
    # `slope_p` is the one the equation names: Grad-Shafranov's source term is p'(psi), so the
    # pressure GRADIENT is the physical quantity and the profile is its integral.
    "p": ("peak_p", "int_p", "slope_p"),
    # WHERE along the chord the peak sits. Measured by permutation importance at 0.1% to 2.3% —
    # the four least useful columns of the block, and unsurprising: the chord is fixed in the
    # machine and the plasma barely moves along it, so this is nearly a constant.
    "pos": ("at_te", "at_p"),
    # Dimensionless shape, independent of the absolute calibration of either channel. Measured at
    # exactly zero: `te,ne,p` and `te,ne,p,shape` scored 0.9874 alike.
    "shape": ("peaking_te", "width_te"),
}


def _masked_slope(values: FloatArray, coord: FloatArray, live: FloatArray,
                  n: FloatArray) -> FloatArray:
    """Least-squares slope of the live points, per time sample: cov(x, y) / var(x), all masked.

    A slope rather than the largest step between neighbours: the local version needs two ADJACENT
    live channels, and outside the discharge — where most of a Thomson record sits — the live ones
    are few and scattered, so it is undefined for whole shots.
    """
    x = np.broadcast_to(coord, values.shape)
    k = np.maximum(n, 1)[:, None]
    dx = np.where(live, x - (np.where(live, x, 0.0).sum(axis=1, keepdims=True) / k), 0.0)
    dy = np.where(live, values - (np.where(live, values, 0.0).sum(axis=1, keepdims=True) / k), 0.0)
    var = (dx * dx).sum(axis=1)
    return np.where(var > 0, (dx * dy).sum(axis=1) / np.where(var > 0, var, 1.0), np.nan)


def _profile_stats(values: FloatArray, dens: FloatArray, coord: FloatArray) -> FloatArray:
    """(n_times, 13) summaries of one Thomson system, per time sample, over live channels only.

    `coord` is the spatial axis of that system — Z for the vertical core chord, R for the
    horizontal edge one — and must be sorted, because the channels are NOT stored in positional
    order and a difference taken in channel order would be meaningless.

    A channel is live where its TEMPERATURE is non-zero: missing samples are encoded as exact
    zeros in this dataset, not as nan, and density and pressure are read on the same mask so all
    thirteen numbers describe the same set of channels.
    """
    live = values != 0
    n = live.sum(axis=1)
    enough = n >= THOMSON_MIN_LIVE
    span = float(coord[-1] - coord[0]) or 1.0
    press = values * dens

    def masked(a: FloatArray) -> FloatArray:
        return np.where(live, a, np.nan)

    # `errstate` covers numpy's floating-point flags; "Mean of empty slice" is a RuntimeWarning
    # raised by the nan-functions themselves and needs its own filter. It fires on frames where
    # EVERY channel is dead, which are a subset of `~enough` and are overwritten with nan below —
    # so the warning describes a case this function already handles on purpose, and the model is
    # told about it through the validity flag rather than being handed a quiet zero. Measured at
    # production: 0.7% of training frames. Filtered by message so a DIFFERENT nan warning here
    # would still be heard.
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.filterwarnings("ignore", "(Mean|All-NaN slice) of empty slice",
                                RuntimeWarning)
        warnings.filterwarnings("ignore", "All-NaN slice encountered", RuntimeWarning)
        te, ne_, pr = masked(values), masked(dens), masked(press)
        peak_te, peak_ne, peak_p = (np.nanmax(v, axis=1) for v in (te, ne_, pr))
        at_te = coord[np.nanargmax(np.where(np.isnan(te), -np.inf, te), axis=1)]
        at_p = coord[np.nanargmax(np.where(np.isnan(pr), -np.inf, pr), axis=1)]
        mean_te, mean_ne, mean_p = (np.nanmean(v, axis=1) for v in (te, ne_, pr))
        slope_te = _masked_slope(values, coord, live, n)
        slope_ne = _masked_slope(dens, coord, live, n)
        slope_p = _masked_slope(press, coord, live, n)
        peaking = peak_te / np.where(mean_te > 0, mean_te, np.nan)
        # Profile-weighted spread of the coordinate: how wide the temperature sits on the chord.
        w = np.where(live & (values > 0), values, 0.0)
        wsum = np.maximum(w.sum(axis=1), 1e-30)
        cbar = (w * coord).sum(axis=1) / wsum
        width = np.sqrt(np.maximum((w * (coord - cbar[:, None]) ** 2).sum(axis=1) / wsum, 0.0))

    out = np.column_stack([peak_te, at_te, mean_te * span, slope_te,
                           peak_ne, mean_ne * span, slope_ne,
                           peak_p, at_p, mean_p * span, slope_p,
                           peaking, width])
    if out.shape[1] != len(THOMSON_STATS):
        raise ValueError(f"produced {out.shape[1]} Thomson stats, THOMSON_STATS lists "
                         f"{len(THOMSON_STATS)}")
    out[~enough] = np.nan
    return out


def _profile_raw(values: FloatArray, dens: FloatArray, coord: FloatArray) -> FloatArray:
    """(n_times, N_PROFILE) the PRESSURE profile itself, on a fixed grid along the chord.

    A fixed grid rather than the channels as they come: channels die and revive from frame to
    frame, so a column tied to a channel would mean a different place at different times. Resampled
    from the live points only, which is also what makes a missing channel harmless rather than a
    zero the model has to learn around.
    """
    live = values != 0
    press = values * dens
    grid = np.linspace(float(coord[0]), float(coord[-1]), N_PROFILE)
    out = np.zeros((len(values), N_PROFILE), dtype=np.float32)
    for t in range(len(values)):
        m = live[t]
        if m.sum() < 2:
            out[t] = np.nan
            continue
        out[t] = np.interp(grid, coord[m], press[t, m]).astype(np.float32)
    return out


def _fill_in_time(feats: FloatArray, src_times: FloatArray,
                  efit_times: FloatArray) -> tuple[FloatArray, bool]:
    """Put a per-Thomson-sample feature series onto the EFIT frames, and say whether it is real.

    The reduction happens on the Thomson time base and the INTERPOLATION afterwards, never the
    other way round: a profile averaged across time would mix a live channel with a dead one.

    A shot whose Thomson never fired returns zeros and False rather than raising. That is NOT the
    banned "substitute the mean and carry on": the caller turns the flag into a feature column, so
    the model is told which shots have no measurement instead of being handed a fabricated one,
    and `train` prints the rate. Measured: `d3d_shot_5a79f2123a` has 1884 core samples and 423
    edge samples with zero live channels in every one of them — the diagnostic simply did not run
    for that discharge, which is a property of the data and not a mismatch to fail on.
    """
    out = np.zeros((len(efit_times), feats.shape[1]), dtype=np.float32)
    for j in range(feats.shape[1]):
        ok = np.isfinite(feats[:, j])
        if ok.sum() < 2:
            return out, False
        out[:, j] = np.interp(efit_times, src_times[ok], feats[ok, j]).astype(np.float32)
    return out, True


def _thomson_features(row: Row, efit_times: FloatArray) -> FloatArray:
    """(T, 27 + 2*N_PROFILE) — the summaries and the flag, then the raw pressure profiles."""
    names = [str(x) for x in np.asarray(row["thomson_chord_name"])]
    chord_R = np.asarray(row["thomson_chord_R"], dtype=np.float64)
    chord_Z = np.asarray(row["thomson_chord_Z"], dtype=np.float64)

    blocks: list[FloatArray] = []
    raws: list[FloatArray] = []
    valid: list[bool] = []
    for prefix, key, coords in (("TS_core", "thomson_core", chord_Z),
                                ("TS_tangential", "thomson_edge", chord_R)):
        te = np.stack([np.asarray(x, dtype=np.float64) for x in row[f"{key}_Te"]])
        ne = np.stack([np.asarray(x, dtype=np.float64) for x in row[f"{key}_ne"]])
        times = np.asarray(row[f"{key}_times"], dtype=np.float64)
        n_ch = te.shape[1]
        start = 0 if prefix == "TS_core" else len(np.asarray(row["thomson_core_Te"][0]))
        labels = names[start:start + n_ch]
        if not labels or not all(n.startswith(prefix) for n in labels):
            raise ValueError(f"chords {start}..{start + n_ch} are {labels[:3]}..., expected "
                             f"{prefix}* — the Thomson channel blocks are not laid out as assumed")
        order = np.argsort(coords[start:start + n_ch])
        c = coords[start:start + n_ch][order]
        stats, ok = _fill_in_time(_profile_stats(te[:, order], ne[:, order], c), times, efit_times)
        raw, ok_raw = _fill_in_time(_profile_raw(te[:, order], ne[:, order], c), times, efit_times)
        blocks.append(stats)
        raws.append(raw)
        valid.append(ok and ok_raw)
    flag = np.full((len(efit_times), 1), float(all(valid)), dtype=np.float32)
    return np.hstack([*blocks, flag, *raws])


def features_for_row(row: Row) -> FloatArray:
    """(T, 90) — 21 magnetics levels, two derivative blocks of 21, and 27 Thomson columns.

    Every block is always produced. `build_inputs` decides which reach the model, so an experiment
    over `features.derivatives` or `features.thomson` never has to rebuild the shot cache.
    """
    if FEATURES_KEY in row:
        ready: FloatArray = np.asarray(row[FEATURES_KEY])
        want = (len(np.asarray(row["efit_times"])), FEATURE_WIDTH)
        if ready.shape != want:
            raise ValueError(f"precomputed {FEATURES_KEY} has shape {ready.shape}, expected "
                             f"{want}")
        return ready
    shot = inputs_only_shot(row)
    levels: FloatArray = interpolate_magnetics_to_efit(shot)
    if levels.shape != (len(shot["efit_times"]), N_SIGNALS):
        raise ValueError(f"features {levels.shape}, expected "
                         f"({len(shot['efit_times'])}, {N_SIGNALS})")
    feats = np.hstack([levels, _raw_derivatives(shot),
                       _interp_derivatives(levels, shot["efit_times"]),
                       _frame_gaps(shot["efit_times"]),
                       _vessel_currents(shot),
                       _thomson_features(row, shot["efit_times"])])
    if feats.shape != (len(shot["efit_times"]), FEATURE_WIDTH):
        raise ValueError(f"features {feats.shape}, expected "
                         f"({len(shot['efit_times'])}, {FEATURE_WIDTH})")
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
    # `index` numbers the magnetics signals, so it addresses the LEVEL block; the derivative
    # blocks that follow carry the same signal order and must not be picked up here.
    currents = X[:, index].astype(np.float64)
    gains = fit_flux_gains(basis.maps, currents, Y)
    maps = basis.maps * gains[:, None, None]

    mean = w = None
    if params.inputs in ("coil_pca", "both"):
        mean, w = coil_pca_transform(maps, currents, params.n_coil_pca)
    return {
        "subtract": params.subtract_coil_field, "inputs": params.inputs,
        "derivatives": params.derivatives,
        "derivative_signals": params.derivative_signals,
        "vessel": params.vessel,
        "thomson": params.thomson, "frame_gaps": params.frame_gaps,
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
    # The level block only: `index` numbers the magnetics signals, and the derivative blocks that
    # follow them carry the same signal order.
    return np.tensordot(feats[:, plan["index"]].astype(np.float64), maps, axes=(1, 0))


@functools.cache
def thomson_columns(groups: str) -> tuple[int, ...]:
    """Indices into the 27-column Thomson block for a comma-separated list of groups.

    The validity flag is always last and always kept: a model told that a measurement is missing
    can learn to ignore the zeros beside it, and a model not told cannot.
    """
    names: list[str] = []
    raw = False
    for g in groups.split(","):
        g = g.strip()
        if g == "raw":
            raw = True
            continue
        if g not in THOMSON_GROUPS:
            raise ValueError(f"unknown thomson group {g!r}; known: raw, {sorted(THOMSON_GROUPS)}")
        names += [n for n in THOMSON_GROUPS[g] if n not in names]
    per = len(THOMSON_STATS)
    idx = [THOMSON_STATS.index(n) + block * per for block in (0, 1) for n in names]
    tail = list(range(2 * per + 1, 2 * per + 1 + 2 * N_PROFILE)) if raw else []
    return (*sorted(idx), 2 * per, *tail)   # + the validity flag, then the raw profiles


def build_inputs(plan: CoilPlan, feats: FloatArray) -> FloatArray:
    """(T, n_features) — what the models actually see, identical in training and in inference.

    `feats` always carries every block `features_for_row` produces; the two selections here are
    orthogonal. `inputs` decides how the LEVELS are represented (raw currents, or the coil flux in
    its own principal directions); `derivatives` decides which rate blocks are appended to that.
    """
    levels = feats[:, :N_SIGNALS]
    mode = plan["inputs"]
    if mode == "currents":
        base = levels
    else:
        coil = (levels[:, plan["index"]].astype(np.float64) - plan["cur_mean"]) @ plan["coil_pca_W"]
        if mode == "coil_pca":
            base = np.hstack([coil, levels[:, plan["passthrough"]]])
        elif mode == "both":
            base = np.hstack([levels, coil])
        else:
            raise ValueError(f"unknown input mode {mode!r}; known: {sorted(INPUT_MODES)}")

    want = plan["derivatives"]
    # The gap block sits immediately after the derivative blocks, which is where the slice below
    # finds it; Thomson is addressed from the back and so is unaffected by it being there.
    gaps = N_SIGNALS * (1 + len(DERIV_BLOCKS))
    tail = ([feats[:, gaps:gaps + N_GAPS]] if plan["frame_gaps"] else [])
    # The vessel block sits immediately after the gap columns, so it is addressed from the front
    # like the derivatives and leaves Thomson's back-addressed slice alone.
    # `.get`, and this is the one place a default is right: an artifact fitted before the block
    # existed saw none of these columns, which is exactly what "none" means. Reading it as absent
    # is not a guess about that fit, it is a statement of fact about it.
    if plan.get("vessel", "none") != "none":
        vessel = feats[:, gaps + N_GAPS:gaps + N_GAPS + N_VESSEL].reshape(
            len(feats), N_SIGNALS, len(VESSEL_TAUS_MS))
        keep = [D3D_MAGNETICS_SIGNALS.index(s) for s in DERIV_SIGNAL_SETS[plan["vessel"]]]
        tail += [vessel[:, keep, :].reshape(len(feats), -1)]
    tail += ([feats[:, -N_THOMSON_BLOCK:][:, thomson_columns(plan["thomson"])]]
             if plan["thomson"] else [])
    if want == "none":
        return np.hstack([base, *tail]) if tail else base
    if want not in DERIV_MODES:
        raise ValueError(f"unknown derivative mode {want!r}; known: {sorted(DERIV_MODES)}")
    blocks = DERIV_BLOCKS if want == "both" else (want,)
    pick = [D3D_MAGNETICS_SIGNALS.index(s) for s in DERIV_SIGNAL_SETS[plan["derivative_signals"]]]
    cols = [base]
    for name in blocks:
        k = DERIV_BLOCKS.index(name) + 1
        cols.append(feats[:, k * N_SIGNALS:(k + 1) * N_SIGNALS][:, pick])
    return np.hstack([*cols, *tail])


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

@functools.cache
def _shot_code_key() -> str:
    """The fingerprint of every function whose output the shot cache stores. Computed once."""
    # EVERY function whose output reaches the cache. Missing one is the failure this key exists to
    # prevent: on 2026-08-13 the Thomson reduction was rewritten while `_profile_stats` was absent
    # from this list, and the stale entries were caught only because the column count happened to
    # change too. `extra` carries the widths for the same reason — a formula change that keeps the
    # shape would otherwise be invisible.
    return shot_cache.fingerprint(
        _read_training_shot, features_for_row, inputs_only_shot, align_ip_times,
        interpolate_magnetics_to_efit, _as_psirz_stack,
        _raw_derivatives, _interp_derivatives, _frame_gaps, _vessel_currents,
        _thomson_features, _profile_stats, _fill_in_time,
        extra=f"{D3D_MAGNETICS_SIGNALS}|{SUBMITTED_SCALARS}|{FEATURE_WIDTH}|{RAW_DERIV_HALF_MS}|{VESSEL_TAUS_MS}",
    )


def _read_training_shot(path: Path,
                        frame_share: float = 1.0) -> tuple[FloatArray, FloatArray, FloatArray]:
    """(features, psi, scalars) for one training shot, with every shape checked.

    `frame_share` thins the frames as they are read, before anything is concatenated: the point of
    thinning is to afford more shots, and holding every frame of every shot in memory first would
    defeat it.

    Decoding is cached — see shot_cache. The cache holds EVERY frame whatever this call asks for,
    so one build serves any frame share, and the thinning below is applied to the cached arrays
    exactly as it is to freshly decoded ones.
    """
    def keep(n_frames: int) -> npt.NDArray[np.intp]:
        return thin_frames(n_frames, frame_share)

    code = _shot_code_key()
    hit = shot_cache.load(path, code, keep)
    if hit is not None:
        return hit

    row = pd.read_parquet(path).iloc[0]
    try:
        feats = features_for_row(row)             # same code path as inference
    except ValueError as exc:
        # Every message from in there describes a property of ONE shot, and on a corpus of
        # thousands an assertion without coordinates is not actionable.
        raise ValueError(f"{path.name}: {exc}") from exc
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

    shot_cache.store(path, code, feats, psi, scal)
    rows = keep(T)
    return feats[rows], psi[rows], scal[rows]


def _read_task(args: tuple) -> tuple[FloatArray, FloatArray, FloatArray]:
    return _read_training_shot(*args)


def _read_shots(files: list[Path], desc: str, frame_share: float = 1.0,
                jobs: int = 0) -> tuple[FloatArray, FloatArray, FloatArray, npt.NDArray[np.int64]]:
    """Every kept frame of every shot, concatenated: features, psi, the two scalar targets — and
    how many frames each shot contributed.

    That last array is the only thing concatenation destroys, and a model whose unit is the shot
    cannot be fitted without it. It is returned always rather than on request: a boundary that is
    computed only when someone asks is a boundary that can be wrong for everyone else.

    Shots are independent, so this is the easiest parallelism in the repo: parquet decode plus
    interpolation, one process per shot, results collected in order so the frame order does not
    depend on the core count.
    """
    n = resolve_jobs(jobs, len(files))
    tasks = [(path, frame_share) for path in files]
    X_parts, Y_parts, S_parts = [], [], []
    bar = tqdm(pimap(_read_task, tasks, n), total=len(files), **bar_kwargs(SHOT_EVERY),
               desc=desc if n == 1 else f"{desc} x{n}", unit="shot")
    for feats, psi, scal in bar:
        X_parts.append(feats)
        Y_parts.append(psi)
        S_parts.append(scal)
        # Frames accumulate in the postfix: it shows both the sample size and that progress is
        # alive, without a line per shot. refresh=False or it redraws the bar on EVERY shot and
        # defeats the thinning that keeps a teed log readable — measured, it was 2835 of the
        # 2849 segments in a production log.
        bar.set_postfix(frames=sum(len(x) for x in X_parts), refresh=False)
    lengths = np.array([len(x) for x in X_parts], dtype=np.int64)
    return (np.concatenate(X_parts), np.concatenate(Y_parts), np.concatenate(S_parts), lengths)


def train(share: str, val_share: str, local_data_dir: Path, config: str,
          params_path: Path = DEFAULT_PARAMS_PATH, jobs: int = 0, salt: int | None = None,
          only: list[str] | None = None, sets: list[str] | None = None) -> Artifact:
    """Fit every enabled model of params.yaml on the head of the split, and save them together.

    Both shares are `"shots"` or `"shots/frames"` (see parse_share). `val_share` is the window
    right behind the training one; it never enters a fit as data, it is what CatBoost and the MLP
    stop on and pick their best iteration by.
    """
    params: Params = load_params(params_path, salt, only, sets)
    shot_share, frame_share = parse_share(share)
    val_shot_share, val_frame_share = parse_share(val_share)
    ordered = sorted_shots(local_data_dir, config, params.split_salt)
    holdout, all_files = reserve_holdout(ordered, params.holdout_share)
    if holdout:
        print(f"Holdout: {len(holdout)} shots ({params.holdout_share:.1%}) reserved from the "
              f"SALT-0 order and excluded from this fit, so models fitted under different salts "
              f"can be scored on the same untouched shots")
    files, val_files = split_train_val(all_files, shot_share, val_shot_share)
    print(f"Training on {len(files)} shots ({shot_share:.1%} of {len(all_files)}, "
          f"{frame_share:.0%} of their frames), validating on {len(val_files)} "
          f"({val_shot_share:.1%}, {val_frame_share:.0%} of frames), head of the list, "
          f"ordered by sha1")
    # A model whose unit is the shot is fitted on a TIME SERIES, and thinning the frames changes
    # the spacing of that series without changing anything it can see: at 0.2 the steps are ~100 ms
    # apart while inference feeds it every frame at 20 ms, so it would learn the dynamics at one
    # spacing and be asked for them at another. It cannot detect this for itself — all it receives
    # is frame counts — so the check belongs here, where the shares are known.
    sequential = [n for n, m in params.models.items() if m.kind == "torch_seq"]
    if sequential and (frame_share < 1.0 or val_frame_share < 1.0):
        raise SystemExit(
            f"model(s) {sequential} read the shot as a sequence and need every frame, but the "
            f"shares thin them to {frame_share:.0%} of training and {val_frame_share:.0%} of "
            f"validation frames. Use '<shots>/1.0' for both."
        )
    print(f"Models from {params.path}: {', '.join(params.models)}  "
          f"(ensemble: {', '.join(f'{k} x {w:.2f}' for k, w in params.ensemble.items())})")

    X, Y, S, shots = _read_shots(files, "reading train shots", frame_share, jobs)
    print(f"  train: X {X.shape}  Y {Y.shape}  S {S.shape}, flux {Y.nbytes / 2 ** 30:.2f} GiB")
    Xv, Yv, Sv, shots_val = _read_shots(val_files, "reading val shots", val_frame_share, jobs)
    print(f"  val:   X {Xv.shape}  Y {Yv.shape}, flux {Yv.nbytes / 2 ** 30:.2f} GiB")

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
    print(f"  inputs: {F.shape[1]} features per frame ({plan['inputs']}, "
          f"derivatives={plan['derivatives']}/{plan['derivative_signals']}, "
          f"thomson={plan['thomson'] or 'off'}, "
          f"frame_gaps={'on' if plan['frame_gaps'] else 'off'}, "
          f"vessel={plan.get('vessel', 'none')})")
    if plan["thomson"]:
        # The validity flag sits at the end of the SUMMARY part of the Thomson block, before the raw
        # profiles; a shot whose diagnostic never fired
        # contributes zeros and a 0 here, and the rate belongs in the log rather than in a feature.
        dead = float((X[:, -N_THOMSON_BLOCK + N_THOMSON - 1] == 0).mean())
        print(f"  thomson: {1 - dead:.1%} of training frames carry a real measurement"
              f"{'' if dead == 0 else f', {dead:.1%} are flagged missing'}")
    scaler = StandardScaler().fit(F)
    Xs = scaler.transform(F)
    Xvs = scaler.transform(Fv)

    if params.n_pca > min(len(Xs), EFIT_GRID_SIZE ** 2):
        raise ValueError(f"n_pca {params.n_pca} exceeds what the data supplies: {len(Xs)} frames, "
                         f"{EFIT_GRID_SIZE ** 2} pixels. Raise --share or lower n_pca in "
                         f"{params.path}.")
    # The PCA does not need every frame: 50 components out of 4225 pixels are estimated from a
    # sample, and a stride over frames covers ramp-up, flat-top and ramp-down by construction (see
    # thin_frames). Fitting on a share of them is the difference between a randomized SVD over the
    # whole flux array and one over a fraction of it, in both time and peak memory — the whole
    # centred copy sklearn makes is the peak of this stage.
    pca_rows = thin_frames(len(Y), params.pca_frame_share)
    print(f"  features standardized; fitting PCA on {len(pca_rows)} of {len(Y)} frames "
          f"({params.pca_frame_share:.0%}) of {EFIT_GRID_SIZE ** 2} pixels")
    pca = TargetPCA(n_components=params.n_pca)
    # Pin the randomized SVD. sklearn chooses that solver for 50 components out of 4225 pixels and
    # TargetPCA leaves random_state at None, so unpinned it fits different components on every run
    # — different targets for every model, and a score that moves without the code changing.
    # Set on the inner estimator rather than by editing the organizers' experiments.py.
    pca.pca.random_state = params.pca_seed
    pca.fit(Y if params.pca_frame_share >= 1.0 else Y[pca_rows])
    print(f"  PCA: {params.n_pca} components explain "
          f"{np.cumsum(pca.explained_variance_ratio)[-1] * 100:.1f}% of the psi variance")

    # One target matrix for every model: [PCA coefficients | q95 | betaN]. Fitting the flux map
    # and the two scalars jointly is what makes the models interchangeable and averageable.
    Tgt = np.hstack([np.asarray(pca.transform(Y), dtype=np.float64), S])
    Tgt_val = np.hstack([np.asarray(pca.transform(Yv), dtype=np.float64), Sv])
    # A9: the seven scored functionals as EXTRA columns. `predict_row` reads only the first
    # n_pca + 2, so these never reach a submission — they exist to put a gradient on the quantities
    # C1 found the cost in. Precomputed by aux_targets.py; a missing cache is an error rather than
    # a silent fallback, because a run that quietly drops them looks exactly like one that kept
    # them and did not help.
    aux_names = params.aux_names
    if aux_names:
        # Imported here, not at module scope: aux_targets is a CLI that reaches for
        # local_score, which imports this module. A pipeline module must not depend on
        # the tools built on top of it.
        from my_experiments import aux_targets
        cols = [CONS_SCALARS.index(n) for n in aux_names]
        blocks = []
        for block_files, n_rows, label in ((files, len(Tgt), "training"),
                                           (val_files, len(Tgt_val), "validation")):
            cons = aux_targets.load(block_files)
            if cons is None:
                raise SystemExit(
                    f"loss.aux_scalars is {params.aux_scalars!r} but the {label} functionals are "
                    f"not cached for this split. Build them once:\n"
                    f"  uv run python my_experiments/aux_targets.py --share {share} "
                    f"--val-share {val_share} --salt {salt}"
                )
            if len(cons) != n_rows:
                raise ValueError(f"{label} functionals hold {len(cons)} frames against "
                                 f"{n_rows} targets — the cache is for a different frame share")
            a = cons[:, cols].astype(np.float64)
            # A functional the scorer could not define is imputed at the column mean, so the frame
            # contributes no signal there rather than a nan that would poison the whole fit.
            for j in range(a.shape[1]):
                bad = ~np.isfinite(a[:, j])
                if bad.any():
                    a[bad, j] = np.nanmean(a[:, j])
            blocks.append(a)
        print(f"  A9: {len(aux_names)} auxiliary outputs ({', '.join(aux_names)}) at weight "
              f"{params.aux_weight}, discarded at inference")
        Tgt = np.hstack([Tgt, blocks[0]])
        Tgt_val = np.hstack([Tgt_val, blocks[1]])
    if Tgt.shape != (len(Xs), params.n_targets):
        raise ValueError(f"targets {Tgt.shape}, expected {(len(Xs), params.n_targets)}")
    if Tgt_val.shape != (len(Xvs), params.n_targets):
        raise ValueError(f"validation targets {Tgt_val.shape}, expected "
                         f"{(len(Xvs), params.n_targets)}")

    # Scaled once, for every model alike — the scaling decides which loss they minimize, so it is
    # a property of the problem, not of any one estimator. Ridge is provably indifferent to it
    # (it is separable per output and the scale cancels in the solution), which makes its score a
    # check that this step changed nothing it should not have.
    # The psi block's loss metric: identity (Parseval — the pixel error of the map, which is
    # exactly R2_psi) unless loss.metric asks for the measured one. Built from the PCA basis, so it
    # exists only after the fit above, and folded into the target transform — no model sees it.
    psi_L = None
    images = np.asarray(pca.pca.components_).reshape(params.n_pca, EFIT_GRID_SIZE, -1)
    if params.loss_metric == "jacobian":
        # The functionals read the TOTAL flux, so the probe has to be applied to it — with the
        # coil field subtracted, Y holds the plasma part alone and the coil part is a constant
        # offset that the perturbation does not touch but the LCFS extraction certainly does.
        probe = thin_frames(len(Y), min(1.0, params.jacobian_frames / len(Y)))
        total = Y[probe].astype(np.float64)
        if plan["subtract"]:
            total = total + coil_flux(plan, X[probe])
        delta = params.jacobian_delta * float(np.sqrt(psi_ss_tot / params.n_pca))
        ctx = scorer_context(plan["grid_R"], plan["grid_Z"], plan["machine"])
        m_cons, var, ratio, n_used, probe_ok = jacobian_form(images, total, delta, ctx, jobs,
                                                             params.calibrate_scalars)
        m_bnd = d_probe = None
        if params.boundary:
            m_bnd, d_probe, n_bnd, clipped = boundary_form(images, total, delta, ctx, jobs)
            print(f"  boundary term: {n_bnd} contours, probe moves the boundary "
                  f"{d_probe:.4g} of rgeo, {clipped:.1%} of contour points on the gradient floor")
        psi_L = factor(metric_form(m_cons, psi_ss_tot, m_bnd, d_probe or 0.0))
        print(f"  loss metric: jacobian over {n_used} of {probe.size} probe frames, step "
              f"{delta:.4g} Wb/rad, condition number {np.linalg.cond(psi_L @ psi_L.T):.1f}")
        for name, r, v, ok in zip(CONS_SCALARS, ratio, var, probe_ok, strict=True):
            print(f"    {name:8s} linear/actual {r:6.2f}   spread {np.sqrt(v):10.4g}"
                  f"{'' if ok > 0.999 else f'   ({1 - ok:.1%} of probes undefined)'}"
                  f"{'   (down-weighted)' if params.calibrate_scalars and r > 1.1 else ''}")
    target_scaler = (TargetScaler(params.n_pca, len(params.aux_names), params.aux_weight)
                     .with_psi_metric(psi_L)
                     .fit(Tgt, psi_ss_tot))
    Tgt = target_scaler.transform(Tgt)
    Tgt_val = target_scaler.transform(Tgt_val)
    var = Tgt.var(axis=0)
    print(f"  targets: {params.n_targets} outputs, scaled as the metric weights them; "
          f"loss share psi {var[:params.n_pca].sum() / var.sum():.0%} / "
          f"scalars {var[params.n_pca:].sum() / var.sum():.0%}")

    # A13. Built only if some model asks to stop on the composite, because it costs a few hundred
    # LCFS extractions to compile and it has to happen HERE — the validation flux maps are released
    # ten lines below and the monitor needs them to build its reference.
    monitor = monitor_idx = None
    if any(getattr(m, "stop_on", "loss") == "composite" for m in params.models.values()):
        monitor_idx = thin_frames(len(Yv), min(1.0, params.monitor_frames / len(Yv)))
        total_v = Yv[monitor_idx].astype(np.float64)
        if plan["subtract"]:
            total_v = total_v + coil_flux(plan, Xv[monitor_idx])
        mctx = scorer_context(plan["grid_R"], plan["grid_Z"], plan["machine"])
        mctx["coil"] = (coil_flux(plan, Xv[monitor_idx]).astype(np.float64)
                        if plan["subtract"] else np.zeros_like(total_v))
        t0 = time.perf_counter()
        monitor = build_monitor(pca, target_scaler, mctx, plan["machine"], total_v,
                                Sv[monitor_idx].astype(np.float64), params.n_pca)
        print(f"  stopping on the COMPOSITE: reference built from {len(monitor_idx)} validation "
              f"frames in {time.perf_counter() - t0:.1f} s")

    # The flux maps are dead from here on: the models see standardized inputs and scaled targets,
    # and the PCA keeps its own components. At production Y and Yv are over 3 GiB together, which
    # is the difference between fitting in RAM and fitting against swap — and the fit is the
    # longest stage, so it is the worst place to be paging.
    print(f"  releasing {(Y.nbytes + Yv.nbytes) / 2 ** 30:.2f} GiB of flux arrays before fitting")
    del X, Y, S, Xv, Yv, Sv, F, Fv
    # And the reader pool with them. Its workers hold far more than the data does — parquet decode
    # buffers they never return to the OS — and the fit below is the longest stage of the run, so
    # it is the worst possible time to be holding them. Scoring spawns its own pool later.
    release()
    print("  released, reader pool shut down; the fit sees inputs and targets only")

    for name, model in params.models.items():
        print(f"\n  fitting {name} ({model.kind}) on {Xs.shape[0]} frames "
              f"-> {params.n_targets} targets, stopping on {Xvs.shape[0]} validation frames")
        t0 = time.perf_counter()
        model.fit(Xs, Tgt, Xvs, Tgt_val, shots, shots_val, monitor, monitor_idx)
        print(f"  {name}: fitted in {time.perf_counter() - t0:.1f} s{model.fit_report()}")

    artifact: Artifact = {
        "scaler": scaler, "pca": pca, "target_scaler": target_scaler,
        "models": params.models, "ensemble": params.ensemble,
        "n_pca": params.n_pca, "coil": plan,
        # The whole params file, verbatim: the artifact says what produced it even after the file
        # on disk has moved on.
        "params_yaml": params.effective_yaml,
        # Filenames, not indices: evaluate.py intersects them directly, and the check stays
        # correct even if the data directory grows.
        "train_files": [p.name for p in files],
        "val_files": [p.name for p in val_files],
        "n_train_shots": len(files), "train_share": shot_share, "train_frame_share": frame_share,
        "n_val_shots": len(val_files), "val_share": val_shot_share,
        "val_frame_share": val_frame_share, "config": config,
        # evaluate.py must order the shots exactly as this run did, or its "tail" is a different
        # set of shots and the overlap check passes while measuring the wrong thing.
        "split_salt": params.split_salt,
        # Named, not counted: evaluate.py has to score exactly these and no others, and a share
        # recomputed there would drift the moment the data directory changes.
        "holdout_files": [p.name for p in holdout],
        "holdout_share": params.holdout_share,
    }
    joblib.dump(artifact, ARTIFACT)
    print(f"\nSaved {ARTIFACT}: {', '.join(model_names(artifact))}")
    print("Now score it:  uv run python my_experiments/evaluate.py --share 0.001")
    return artifact


# --------------------------------------------------------------------------- inference

_CACHE: dict[Path, Artifact] = {}


def artifact_path(name: str) -> Path:
    """`baseline_<name>.joblib` beside the default artifact — where a per-salt fit is written."""
    return ARTIFACT.parent / f"baseline_{name}.joblib"


def _load(path: Path | None = None) -> Artifact:
    """One artifact, cached by path — several are held at once when averaging across salts."""
    path = path or ARTIFACT
    if path not in _CACHE:
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found — train first:\n"
                f"  uv run python my_experiments/train.py --share 0.01"
            )
        _CACHE[path] = joblib.load(path)
    return _CACHE[path]


def model_names(artifact: Artifact | None = None) -> list[str]:
    """Every scoreable name in the artifact: each fitted model, then the ensemble."""
    art = artifact if artifact is not None else _load()
    return [*art["models"], ENSEMBLE]


def _predict_targets(art: Artifact, model: str, Xs: FloatArray) -> FloatArray:
    """(T, n_pca + 2) — one model's targets, or an average of several.

    Three forms of `model`, and the third is why this is not just a lookup:

    * a fitted model's name;
    * `ensemble`, the weighted average params.yaml declared;
    * **`a+b+c`, an equally weighted average of any subset**, assembled at SCORING time. Which
      members to combine is a question about a fitted artifact, not about a fit, so answering it
      should cost one training run and not one per combination — seven combinations of three
      models is seven fits the slow way and one the fast way.

    Averaging targets and averaging the flux maps they decode to are the same thing: PCA's
    inverse transform is affine and the weights sum to 1, so the mean image is added back exactly
    once either way.
    """
    n_submitted = int(art["n_pca"]) + len(SUBMITTED_SCALARS)

    def inverse(scaled: FloatArray) -> FloatArray:
        """Unscale, and DROP A9's auxiliary columns here rather than anywhere downstream.

        The auxiliary outputs exist to put a gradient on the scored functionals during training and
        are meaningless afterwards — the scorer derives those quantities from the submitted map. So
        the one place that turns a model's raw output into targets is the one place that should
        forget them, and everything past this point keeps working on the 52 columns it always had.
        Without this the ensemble's accumulator, sized from n_pca + 2, fails to broadcast against a
        54-column member — measured, and it cost three runs their scoring pass.
        """
        return np.asarray(art["target_scaler"].inverse_transform(scaled))[:, :n_submitted]
    # `psi=a+b,qb=c` takes the flux coefficients from one combination and the two scalars from
    # another. The composite reads them separately — R2_psi, D_LCFS and Consistency are all
    # functions of the MAP, while R2_qb is only the two directly regressed scalars — so nothing
    # says one model has to supply both. Costs nothing to try: the members are already fitted.
    if BLOCK_SEP in model or model.startswith(PSI_KEY):
        blocks = dict(p.split("=", 1) for p in model.split(BLOCK_SEP))
        if set(blocks) != {PSI_KEY[:-1], QB_KEY[:-1]}:
            raise KeyError(f"{model!r}: a split is written 'psi=<spec>,qb=<spec>' and needs "
                           f"both blocks; got {sorted(blocks)}")
        n_pca = int(art["n_pca"])
        psi_block = _predict_targets(art, blocks[PSI_KEY[:-1]], Xs)
        qb_block = _predict_targets(art, blocks[QB_KEY[:-1]], Xs)
        return np.hstack([psi_block[:, :n_pca], qb_block[:, n_pca:]])
    if model == ENSEMBLE:
        out = np.zeros((len(Xs), art["n_pca"] + len(SUBMITTED_SCALARS)), dtype=np.float64)
        for name, weight in art["ensemble"].items():
            out += weight * inverse(art["models"][name].predict(Xs))
        return out
    # `a*0.3+b*0.7` — an UNEQUALLY weighted average. Without a weight a member gets 1.0, so `a+b`
    # keeps meaning the equal average it always did. Weights are renormalised to sum to 1, which is
    # what makes averaging coefficients the same as averaging the decoded maps; a negative weight
    # is refused rather than renormalised, since it is almost always a typo and the one time it is
    # not, extrapolating outside the members' hull is a different idea that should be argued for.
    members, weights = [], []
    for part in model.split("+"):
        name, _, w = part.partition(WEIGHT_SEP)
        members.append(name)
        try:
            weights.append(1.0 if not w else float(w))
        except ValueError:
            raise KeyError(f"{part!r}: a weight is written 'name{WEIGHT_SEP}0.4' and must be a "
                           f"number, got {w!r}") from None
    missing = [m for m in members if m not in art["models"]]
    if missing:
        raise KeyError(f"no model {missing[0]!r} in {ARTIFACT}; it holds {model_names(art)}. "
                       f"A combination is written 'a+b' or 'a{WEIGHT_SEP}0.3+b{WEIGHT_SEP}0.7' "
                       f"and every part must be a fitted model.")
    if any(w < 0 for w in weights):
        raise KeyError(f"{model!r} carries a negative weight; the average is over the members, "
                       f"not outside them")
    total = sum(weights)
    if not total > 0:
        raise KeyError(f"{model!r}: the weights sum to {total}, so there is nothing to average")
    if len(members) == 1:
        return inverse(art["models"][members[0]].predict(Xs))
    out = np.zeros((len(Xs), art["n_pca"] + len(SUBMITTED_SCALARS)), dtype=np.float64)
    for name, weight in zip(members, weights, strict=True):
        out += inverse(art["models"][name].predict(Xs)) * (weight / total)
    return out


def project_through_basis(row: Row, psi_gt: FloatArray,
                          artifact: Path | None = None) -> FloatArray:
    """Ground-truth flux encoded into the artifact's basis and decoded straight back.

    Diagnostic C1 in ideas.md, and it answers a question no amount of training can: the models
    predict COEFFICIENTS, so whatever the 50 directions cannot represent is lost before any
    regression begins. Scoring this reconstruction is therefore the ceiling of the whole pipeline —
    what a perfect model would get — and the gap between it and 1.0 is the part of the remaining
    budget that no better fit can ever reach.

    The coil field is subtracted and added back exactly as `predict_row` does it, because the basis
    was fitted on the residual and projecting the raw map onto it would measure a different basis.
    """
    art = _load(artifact)
    plan: CoilPlan = art["coil"]
    psi = np.asarray(psi_gt, dtype=np.float64)
    coil = None
    if plan["subtract"]:
        check_grid(plan, row)
        coil = coil_flux(plan, features_for_row(row)).astype(np.float64)
        if coil.shape != psi.shape:
            raise ValueError(f"coil field {coil.shape} against flux {psi.shape}")
        psi = psi - coil
    out = np.asarray(art["pca"].inverse_transform(art["pca"].transform(psi)), dtype=np.float64)
    return out if coil is None else out + coil


def predict_row(row: Row, source: str = "DIII-D", model: str = ENSEMBLE,
                artifact: Path | None = None) -> dict[str, FloatArray]:
    """Predict {psirz (T,65,65), q95 (T,), betaN (T,)} for one dataset row.

    `model` selects one member of the zoo by name, `a+b` their equally weighted average, or the
    default `ensemble` — the weighted average params.yaml declared. Uses only `magnetics_*` and
    `efit_times`, never the `efit_*` targets, which are present in training rows and would make
    any local score meaningless.

    **`salts:a+b` is a different kind of average** and is handled here rather than in
    `_predict_targets`: those members live in separate artifacts, fitted on separate splits, and
    each has its own PCA basis, target scaler and coil gains. Their coefficients are therefore not
    commensurable — averaging them would be summing numbers that mean different things — so the
    average is taken here, on the decoded flux maps and scalars, which are in physical units and
    mean the same thing in every fit.
    """
    if model.startswith(SALTS_PREFIX):
        names = model[len(SALTS_PREFIX):].split("+")
        if len(names) < 2:
            raise ValueError(f"{model!r} names one fit; use its own model name instead")
        parts = [predict_row(row, source, ENSEMBLE, artifact_path(n)) for n in names]
        averaged: dict[str, FloatArray] = {}
        for key in parts[0]:
            stacked = np.stack([p[key] for p in parts])
            averaged[key] = stacked.mean(axis=0).astype(parts[0][key].dtype)
        return averaged
    art = _load(artifact)
    if source != "DIII-D":
        raise NotImplementedError(
            f"the model is trained on DIII-D only, but the row is marked source={source!r}. "
            f"MAST has its own coil set and needs its own fit — returning zeros silently is not "
            f"an option, it would look like a working prediction."
        )

    T = len(np.asarray(row["efit_times"]))
    if "coil" not in art:
        raise KeyError(
            f"{artifact or ARTIFACT} predates the coil-field decomposition and does not say which "
            f"features it "
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
