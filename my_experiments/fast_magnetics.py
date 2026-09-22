"""Artifact-owned centered raw-current features and a separate full-shot sidecar cache.

The 208-column base shot cache is independent of this module. Each sidecar stores only the extra
columns for one explicit YAML specification, before per-shot frame thinning. A scored row can
instead carry the same small columns under ``FAST_KEY`` after its raw traces are discarded.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from tqdm import tqdm

from toolkit.parallel import pimap, release, resolve_jobs
from toolkit.progress import SHOT_EVERY, bar_kwargs
from toolkit.records import file_hash

ROOT = Path(__file__).resolve().parent.parent
CACHE = Path(os.environ.get(
    "FUSION_FAST_CACHE",
    Path(os.environ.get("FUSION_SHOT_CACHE", ROOT / ".shot_cache")).parent
    / ".fast_magnetics_cache",
))
LOCK = ROOT / "uv.lock"
MANIFEST = Path(__file__).with_name("diii_d_dataset.json")
FAST_KEY = "fast_magnetics_features"
ALLOWED_SIGNALS = ("ECOILA", "plasma_current")
INPUT_COLUMNS = ("efit_times", "magnetics_time", "magnetics_plasma_current_times",
                 "magnetics_ECOILA", "magnetics_plasma_current")
FastSpec = tuple[tuple[str, float], ...]
FloatArray = npt.NDArray[np.float32]


def normalize_spec(raw: Any, context: str) -> FastSpec:
    """Validate an ordered YAML list; no widths or signal choices are supplied by code defaults."""
    if not isinstance(raw, list):
        raise ValueError(f"{context}: expected a list of signal/halfwidth_ms mappings, got "
                         f"{type(raw).__name__}")
    out: list[tuple[str, float]] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict) or set(item) != {"signal", "halfwidth_ms"}:
            raise ValueError(f"{context}[{i}]: expected exactly signal and halfwidth_ms, "
                             f"got {item!r}")
        signal = item["signal"]
        if signal not in ALLOWED_SIGNALS:
            raise ValueError(f"{context}[{i}]: signal {signal!r}, expected one of "
                             f"{ALLOWED_SIGNALS}")
        value = item["halfwidth_ms"]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{context}[{i}]: halfwidth_ms={value!r}, expected a number")
        width = float(value)
        if not np.isfinite(width) or width <= 0:
            raise ValueError(f"{context}[{i}]: halfwidth_ms={width}, expected finite positive ms")
        if (signal, width) in out:
            raise ValueError(f"{context}[{i}]: duplicate {signal} at halfwidth_ms={width:g}")
        out.append((signal, width))
    return tuple(out)


def spec_to_yaml_value(spec: FastSpec) -> list[dict[str, Any]]:
    """Canonical literal saved in a fitted artifact's coil plan."""
    return [{"signal": signal, "halfwidth_ms": width} for signal, width in spec]


def feature_name(signal: str, width: float) -> str:
    """A stable key for one column in a slim scored row."""
    return f"{signal}:centered_d1:{width:.17g}ms"


def _array(row: Any, column: str, dtype: Any, filename: str) -> np.ndarray:
    """Read one required input column without silently substituting an empty trace."""
    if column not in row:
        raise ValueError(f"{filename}: missing input column {column}")
    return np.asarray(row[column], dtype=dtype).ravel()


def _strict_times(t: np.ndarray, v: np.ndarray, column: str, filename: str) -> None:
    """Reject ambiguous interpolation coordinates and nonfinite current values."""
    if len(t) != len(v):
        raise ValueError(f"{filename}: {column} has {len(v)} values against {len(t)} times")
    if len(t) < 3:
        raise ValueError(f"{filename}: {column} has {len(t)} samples, expected at least 3")
    if not np.isfinite(t).all() or not np.isfinite(v).all():
        raise ValueError(f"{filename}: {column} has "
                         f"{int((~np.isfinite(t)).sum())} nonfinite times and "
                         f"{int((~np.isfinite(v)).sum())} nonfinite values")
    diff = np.diff(t)
    if np.any(diff <= 0):
        raise ValueError(f"{filename}: {column} has {int(np.count_nonzero(diff <= 0))} "
                         "duplicate or decreasing time steps")


def compute_from_row(row: Any, spec: FastSpec, filename: str) -> FloatArray:
    """Compute centered dI/dt at EFIT times, using each signal's corrected raw clock.

    Every endpoint must lie within acquisition support. ``np.interp`` is called only after that
    check, so neither its clamping behavior nor the base feature pipeline's extrapolator can
    silently manufacture a derivative at the boundary.
    """
    efit = _array(row, "efit_times", np.float64, filename)
    if not np.isfinite(efit).all() or (len(efit) > 1 and np.any(np.diff(efit) <= 0)):
        raise ValueError(f"{filename}: efit_times has {len(efit)} values or nonfinite/"
                         "nonmonotonic steps")
    if not spec:
        return np.empty((len(efit), 0), dtype=np.float32)
    from my_experiments.baseline_model import align_ip_times

    shared = _array(row, "magnetics_time", np.float64, filename)
    times_and_values: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    if any(signal == "ECOILA" for signal, _ in spec):
        coil = _array(row, "magnetics_ECOILA", np.float32, filename)
        _strict_times(shared, coil, "magnetics_ECOILA", filename)
        times_and_values["ECOILA"] = shared, coil
    if any(signal == "plasma_current" for signal, _ in spec):
        ip_shipped = _array(row, "magnetics_plasma_current_times", np.float64, filename)
        ip = _array(row, "magnetics_plasma_current", np.float32, filename)
        _strict_times(ip_shipped, ip, "magnetics_plasma_current", filename)
        axis = align_ip_times(efit, ip_shipped, ip, shared)
        _strict_times(axis, ip, "corrected magnetics_plasma_current", filename)
        times_and_values["plasma_current"] = axis, ip
    columns: list[np.ndarray] = []
    for signal, width in spec:
        t, v = times_and_values[signal]
        left, right = efit - width, efit + width
        missing = (left < t[0]) | (right > t[-1])
        if np.any(missing):
            raise ValueError(
                f"{filename}: {signal} halfwidth_ms={width:g} has "
                f"{int(missing.sum())}/{len(efit)} windows outside raw support "
                f"[{t[0]:.6g}, {t[-1]:.6g}] ms; requested endpoint range "
                f"[{left.min():.6g}, {right.max():.6g}] ms"
            )
        derivative = ((np.interp(right, t, v) - np.interp(left, t, v)) / (2 * width))
        if not np.isfinite(derivative).all():
            raise ValueError(f"{filename}: {signal} halfwidth_ms={width:g} produced "
                             f"{int((~np.isfinite(derivative)).sum())} nonfinite derivatives")
        columns.append(derivative.astype(np.float32))
    out = np.column_stack(columns)
    if out.shape != (len(efit), len(spec)):
        raise ValueError(f"{filename}: fast feature shape {out.shape}, expected "
                         f"({len(efit)}, {len(spec)})")
    return out


def features_for_row(row: Any, spec: FastSpec, filename: str = "row") -> FloatArray:
    """Select an artifact's columns from a full row or a slim union of artifact requirements."""
    if FAST_KEY not in row:
        return compute_from_row(row, spec, filename)
    stored = row[FAST_KEY]
    if not isinstance(stored, dict):
        raise ValueError(f"{filename}: {FAST_KEY} is {type(stored).__name__}, expected mapping")
    n = len(np.asarray(row["efit_times"]))
    columns = []
    for signal, width in spec:
        name = feature_name(signal, width)
        if name not in stored:
            raise ValueError(f"{filename}: slim row lacks fast feature {name}; "
                             f"has {sorted(stored)}")
        col = np.asarray(stored[name])
        if col.shape != (n,) or col.dtype != np.float32 or not np.isfinite(col).all():
            raise ValueError(f"{filename}: slim {name} has shape {col.shape}, dtype "
                             f"{col.dtype}, nonfinite={int((~np.isfinite(col)).sum())}; "
                             f"expected ({n},) float32 finite")
        columns.append(col)
    return np.column_stack(columns) if columns else np.empty((n, 0), dtype=np.float32)


def slim_features(row: Any, union: FastSpec, filename: str = "row") -> dict[str, FloatArray]:
    """Precompute each needed descriptor once, then discard all large raw input streams."""
    matrix = compute_from_row(row, union, filename)
    return {feature_name(signal, width): matrix[:, i].copy()
            for i, (signal, width) in enumerate(union)}


def required_for_models(model_specs: tuple[str | None, ...]) -> FastSpec:
    """Union all artifact-owned columns named by local scoring's model expressions.

    ``salts:`` expressions may mix old and new artifacts. Their decoded predictions are averaged
    elsewhere; this routine only prepares the small input columns each member will request. A
    missing default artifact is allowed solely for the submission skeleton's no-model fallback.
    """
    from my_experiments.baseline_model import ARTIFACT, SALTS_PREFIX, _load, artifact_path

    paths: list[Path] = []
    for model in model_specs:
        if model is not None and model.startswith(SALTS_PREFIX):
            for entry in model[len(SALTS_PREFIX):].split("+"):
                name = entry.partition("*")[0]
                if not name:
                    raise ValueError(f"{model!r}: empty artifact name in salts expression")
                paths.append(artifact_path(name))
        else:
            if model is not None and not ARTIFACT.exists():
                raise FileNotFoundError(f"{ARTIFACT}: named model {model!r} needs an artifact")
            if ARTIFACT.exists():
                paths.append(ARTIFACT)
    union: list[tuple[str, float]] = []
    for path in dict.fromkeys(paths):
        art = _load(path)
        plan = art.get("coil", {})
        spec = normalize_spec(plan.get("fast_magnetics", []),
                              f"{path}: coil.fast_magnetics")
        for descriptor in spec:
            if descriptor not in union:
                union.append(descriptor)
    return tuple(union)


@cache
def _manifest_doc() -> dict[str, Any]:
    """Read the pinned manifest once per process, not once per shot."""
    return json.loads(MANIFEST.read_text())


def _manifest_entry(path: Path) -> tuple[int, str]:
    """Pinned identity of one source parquet, independent of cache directory contents."""
    doc = _manifest_doc()
    config = path.parent.name
    if config not in doc["configs"] or path.name not in doc["configs"][config]:
        raise ValueError(f"{path}: no entry in pinned manifest {MANIFEST}")
    size, digest = doc["configs"][config][path.name]
    return int(size), str(digest)


@cache
def _producer_identity() -> dict[str, str]:
    """Hash source and dependency files once per worker for the whole cache build."""
    from my_experiments.baseline_model import align_ip_times, thin_frames

    producer = hashlib.sha256(
        Path(__file__).read_bytes() + inspect.getsource(align_ip_times).encode()
        + inspect.getsource(thin_frames).encode()
    ).hexdigest()
    return {
        "producer_sha256": producer,
        "lock_sha256": file_hash(LOCK),
        "manifest_sha256": file_hash(MANIFEST),
    }


def _cache_identity(path: Path, spec: FastSpec) -> str:
    """Version key over producer/dependency code, explicit spec, and pinned input identity."""
    stat = path.stat()
    size, expected_digest = _manifest_entry(path)
    if stat.st_size != size:
        raise ValueError(f"{path}: {stat.st_size} bytes, pinned manifest expects {size}")
    identity = {
        "schema": 1,
        **_producer_identity(),
        "config": path.parent.name,
        "file": path.name,
        "source_expected_sha256": expected_digest,
        "source_size": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns,
        "spec": spec_to_yaml_value(spec),
    }
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def _cache_path(path: Path, spec: FastSpec) -> Path:
    """Separate sidecars for separate ordered specifications on the same parquet."""
    code = hashlib.sha256(json.dumps(spec_to_yaml_value(spec), sort_keys=True).encode()).hexdigest()
    return CACHE / path.parent.name / f"{path.stem}-{code[:16]}.npz"


def _read_one(task: tuple[Path, FastSpec, float]) -> tuple[FloatArray, int, int]:
    """Load or atomically build full-shot extras, then thin exactly as base training does."""
    from my_experiments.baseline_model import thin_frames

    path, spec, frame_share = task
    cache_path = _cache_path(path, spec)
    key = _cache_identity(path, spec)
    matrix: FloatArray | None = None
    full_count = -1
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as saved:
            if str(saved["key"]) == key:
                matrix = np.asarray(saved["features"])
                full_count = int(saved["full_frames"])
                source_hash = str(saved["source_sha256"])
                _, expected = _manifest_entry(path)
                if source_hash != expected:
                    raise ValueError(f"{cache_path}: cached source SHA-256 {source_hash} "
                                     f"differs from pinned {expected}")
    if matrix is None:
        actual = file_hash(path)
        _, expected = _manifest_entry(path)
        if actual != expected:
            raise ValueError(f"{path}: SHA-256 {actual}, pinned manifest expects {expected}")
        columns = ["efit_times", "magnetics_time"]
        for signal, _ in spec:
            if signal == "ECOILA" and "magnetics_ECOILA" not in columns:
                columns.append("magnetics_ECOILA")
            if signal == "plasma_current":
                columns += ["magnetics_plasma_current_times", "magnetics_plasma_current"]
        row = pd.read_parquet(path, columns=list(dict.fromkeys(columns))).iloc[0]
        matrix = compute_from_row(row, spec, path.name)
        full_count = len(matrix)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_name(f"{cache_path.name}.{os.getpid()}.tmp")
        with temporary.open("wb") as stream:
            np.savez(stream, key=key, features=matrix, full_frames=full_count,
                     source_sha256=actual)
        os.replace(temporary, cache_path)
    if matrix.dtype != np.float32 or matrix.shape != (full_count, len(spec)):
        raise ValueError(f"{cache_path}: cached features shape {matrix.shape}, dtype "
                         f"{matrix.dtype}; expected ({full_count}, {len(spec)}) float32")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{cache_path}: features contain "
                         f"{int((~np.isfinite(matrix)).sum())} nonfinite values")
    keep = thin_frames(full_count, frame_share)
    return matrix[keep], full_count, len(keep)


def load_many(files: list[Path], frame_share: float, spec: FastSpec, jobs: int,
              label: str) -> tuple[FloatArray, npt.NDArray[np.int64]]:
    """Read sidecars in shot order and return the same per-shot thinning counts as base cache."""
    if not spec:
        raise ValueError("load_many requires a nonempty fast feature spec")
    if not files:
        raise ValueError(f"{label}: no files for fast feature sidecars")
    tasks = [(path, spec, frame_share) for path in files]
    workers = resolve_jobs(jobs, len(tasks))
    rows = list(tqdm(pimap(_read_one, tasks, workers), total=len(tasks), desc=label,
                     unit="shot", **bar_kwargs(SHOT_EVERY)))
    counts = np.asarray([n for _, _, n in rows], dtype=np.int64)
    return np.concatenate([x for x, _, _ in rows]), counts


def main() -> None:
    """Prewarm both production windows from one frozen arm's effective YAML before fitting."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", type=Path, required=True,
                        help="frozen arm's complete effective params.yaml")
    parser.add_argument("--train-share", required=True)
    parser.add_argument("--val-share", required=True)
    parser.add_argument("--jobs", type=int, required=True)
    parser.add_argument("--local-data-dir", type=Path, required=True)
    parser.add_argument("--config", default="diii_d_train")
    args = parser.parse_args()

    from my_experiments.baseline_model import parse_share, sorted_shots, split_train_val
    from my_experiments.data_check import check_dataset
    from my_experiments.models import load_params

    params = load_params(args.params)
    if not params.fast_magnetics:
        raise ValueError(f"{args.params}: features.fast_magnetics is empty; nothing to prewarm")
    train_share, train_frames = parse_share(args.train_share)
    val_share, val_frames = parse_share(args.val_share)
    if train_frames != 1.0 or val_frames != 1.0:
        raise ValueError(f"sidecars store full frames: --train-share {args.train_share!r} and "
                         f"--val-share {args.val_share!r} must both use /1.0")
    check_dataset(args.local_data_dir, args.config)
    ordered = sorted_shots(args.local_data_dir, args.config, params.split_salt)
    train, validation = split_train_val(ordered, train_share, val_share)
    for label, files in (("train", train), ("validation", validation)):
        matrix, lengths = load_many(files, 1.0, params.fast_magnetics, args.jobs,
                                    f"prewarming {label} fast magnetics")
        if int(lengths.sum()) != len(matrix):
            raise ValueError(f"{label}: per-shot lengths sum to {lengths.sum()}, but matrix "
                             f"has {len(matrix)} rows")
        print(f"{label}: {len(files)} shots, {len(matrix)} full frames, "
              f"{matrix.shape[1]} extra columns in {CACHE}")
    release()


if __name__ == "__main__":
    main()
