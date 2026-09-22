"""Preregistered D8/S6 screen of old-ensemble shot bias from whole-shot inputs.

This intentionally uses the 2026-08-15 predictor source and its original validation shots.
Nothing calls the historical training-shot reader or writes its decoded-shot cache. The 338-shot
diagnostic block was seen by the base ensemble's early stopping, so this is a rule-out screen,
never a competition score or independent confirmation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, NoReturn

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import explained_variance_score, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from toolkit.parallel import pimap, release
from toolkit.progress import SHOT_EVERY, bar_kwargs
from toolkit.records import file_hash, write_json

ROOT = Path(__file__).resolve().parent.parent
SPEC = ROOT / "experiments/d8_whole_shot_prereg.json"
RESULT = ROOT / "results/d8_whole_shot_20260922.json"
SUMMARIES = ROOT / "logs/d8_whole_shot_summaries.npz"
MANIFEST = ROOT / "my_experiments/diii_d_dataset.json"


def _old_module(source_root: Path) -> Any:
    """Import the archived predictor, rejecting accidental current-source package reuse."""
    source_root = source_root.resolve()
    sys.path.insert(0, str(source_root))
    from my_experiments import baseline_model as old

    loaded = Path(old.__file__).resolve()
    if not loaded.is_relative_to(source_root):
        raise ValueError(f"old predictor loaded from {loaded}, expected {source_root}")
    return old


def _forbid_cache_write(*args: Any, **kwargs: Any) -> NoReturn:
    """Turn any accidental call into the historical shot cache into an immediate failure."""
    raise RuntimeError("D8 diagnostic must not call the historical shot cache")


def _sha256_bytes(value: np.ndarray) -> str:
    """Hash one numerical control array with its dtype and shape made explicit by the caller."""
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _row_summaries(path: Path, artifact_path: Path, source_root: Path,
                   expected_sha256: str, full_width: int, no_thomson_width: int,
                   n_pca: int, one_frame_duration_ms: float) -> dict[str, Any]:
    """Read one parquet directly and return only small input/target summaries."""
    old = _old_module(source_root)
    old.shot_cache.store = _forbid_cache_write
    old._read_training_shot = _forbid_cache_write
    actual_sha256 = file_hash(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(f"{path}: SHA-256 {actual_sha256}, pinned manifest expects "
                         f"{expected_sha256}")
    artifact = _old_artifact(artifact_path, source_root)
    row = pd.read_parquet(path).iloc[0]
    if str(row["source"]) != "DIII-D":
        raise ValueError(f"{path}: source {row['source']!r}, expected DIII-D")
    try:
        features = old.features_for_row(row)
        inputs = old.build_inputs(artifact["coil"], features)
    except ValueError as exc:
        raise ValueError(f"{path}: {exc}") from exc
    n_frames = len(np.asarray(row["efit_times"]))
    if inputs.shape != (n_frames, full_width):
        raise ValueError(f"{path}: model inputs {inputs.shape}, expected "
                         f"({n_frames}, {full_width})")
    thomson_width = len(old.thomson_columns(artifact["coil"]["thomson"]))
    if full_width - thomson_width != no_thomson_width:
        raise ValueError(f"{path}: all input width {full_width} minus Thomson {thomson_width} "
                         f"is {full_width - thomson_width}, expected {no_thomson_width}")
    if not np.isfinite(inputs).all():
        raise ValueError(f"{path}: model inputs have "
                         f"{int((~np.isfinite(inputs)).sum())} nonfinite values")
    times = np.asarray(row["efit_times"], dtype=np.float64)
    if n_frames < 1 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError(f"{path}: efit_times have {n_frames} frames or nonmonotonic/"
                         "nonfinite values")
    duration = (one_frame_duration_ms if n_frames == 1 else float(times[-1] - times[0]))
    if duration < 0 or (n_frames > 1 and duration == 0):
        raise ValueError(f"{path}: EFIT duration {duration} ms over {n_frames} frames")
    mean = np.mean(inputs, axis=0, dtype=np.float64)
    std = np.std(inputs, axis=0, dtype=np.float64, ddof=0)
    summary = np.concatenate([mean, std, [duration]])

    scaled = artifact["scaler"].transform(inputs)
    predicted = old._predict_targets(artifact, "ensemble", scaled)
    if predicted.shape != (n_frames, n_pca + 2):
        raise ValueError(f"{path}: predicted target shape {predicted.shape}, expected "
                         f"({n_frames}, {n_pca + 2})")
    from experiments import _as_psirz_stack

    psi = _as_psirz_stack(row["efit_psirz"]).astype(np.float64)
    if psi.shape != (n_frames, 65, 65) or not np.isfinite(psi).all():
        raise ValueError(f"{path}: true flux shape {psi.shape}, expected "
                         f"({n_frames}, 65, 65) finite")
    # PCA, the coil map, and TargetScaler are affine/linear. Their transform of a shot mean is
    # the mean of their per-frame transforms, avoiding hundreds of unneeded 65x65 coil maps.
    mean_coil = old.coil_flux(artifact["coil"], features.mean(axis=0, keepdims=True))[0]
    true_coeff = np.asarray(artifact["pca"].transform((psi.mean(axis=0) - mean_coil)[None]))[0]
    if true_coeff.shape != (n_pca,):
        raise ValueError(f"{path}: true PCA mean has shape {true_coeff.shape}, expected "
                         f"({n_pca},)")
    q95 = np.asarray(row["efit_q95"], dtype=np.float64)
    beta_n = np.asarray(row["efit_beta_n"], dtype=np.float64)
    if q95.shape != (n_frames,) or beta_n.shape != (n_frames,):
        raise ValueError(f"{path}: scalar target lengths {len(q95)}/{len(beta_n)} "
                         f"against {n_frames} EFIT frames")
    true_mean = np.r_[true_coeff, q95.mean(), beta_n.mean()][None]
    pred_mean = predicted.mean(axis=0, keepdims=True)
    target = (artifact["target_scaler"].transform(true_mean)
              - artifact["target_scaler"].transform(pred_mean))[0, :n_pca]
    if not np.isfinite(summary).all() or not np.isfinite(target).all():
        raise ValueError(f"{path}: summary/target contains nonfinite values")
    return {"file": path.name, "source_sha256": actual_sha256, "frames": n_frames,
            "summary": summary, "target": target}


_ARTIFACT: dict[tuple[Path, Path], Any] = {}


def _old_artifact(path: Path, source_root: Path) -> Any:
    """Load the 14 MB archive once per worker, after the old classes are imported."""
    key = path.resolve(), source_root.resolve()
    if key not in _ARTIFACT:
        _old_module(source_root)
        _ARTIFACT[key] = joblib.load(path)
    return _ARTIFACT[key]


def _task(task: tuple[Path, Path, Path, str, int, int, int, float]) -> dict[str, Any]:
    """Top-level spawn-safe worker for one validation shot."""
    return _row_summaries(*task)


def _parity_control(spec: dict[str, Any], artifact: Any) -> dict[str, Any]:
    """Decode one old target prediction and compare it to a retained old-source NPZ."""
    source_root = Path(spec["source_root"])
    old = _old_module(source_root)
    old.shot_cache.store = _forbid_cache_write
    old._read_training_shot = _forbid_cache_write
    c = spec["parity_control"]
    prediction_path = ROOT / c["prediction_npz"]
    sidecar = json.loads((ROOT / c["prediction_sidecar"]).read_text())
    if sidecar["files"][0] != c["file"]:
        raise ValueError(f"{prediction_path}: first file {sidecar['files'][0]}, expected "
                         f"{c['file']}")
    if file_hash(prediction_path) != sidecar["prediction_sha256"]:
        raise ValueError(f"{prediction_path}: SHA-256 differs from its sidecar")
    path = Path(spec["data_root"]) / "data/diii_d_train" / c["file"]
    row = pd.read_parquet(path).iloc[0]
    features = old.features_for_row(row)
    scaled = artifact["scaler"].transform(old.build_inputs(artifact["coil"], features))
    predicted = old._predict_targets(artifact, "ensemble", scaled)
    decoded = np.asarray(artifact["pca"].inverse_transform(
        predicted[:, :artifact["n_pca"]]
    ))
    manual = (decoded + old.coil_flux(artifact["coil"], features).astype(
        decoded.dtype
    )).astype(np.float32)
    prefix = c["npz_key_prefix"]
    with np.load(prediction_path, allow_pickle=False) as packed:
        expected = np.asarray(packed[f"{prefix}_psirz"])
        expected_q = np.asarray(packed[f"{prefix}_q95"])
        expected_b = np.asarray(packed[f"{prefix}_betaN"])
    if not np.array_equal(manual, expected):
        raise ValueError(f"{path}: decoded old ensemble differs from retained old-source map")
    if not np.array_equal(predicted[:, artifact["n_pca"]].astype(np.float32), expected_q):
        raise ValueError(f"{path}: old q95 differs from retained prediction")
    if not np.array_equal(predicted[:, artifact["n_pca"] + 1].astype(np.float32), expected_b):
        raise ValueError(f"{path}: old betaN differs from retained prediction")
    return {"file": path.name, "frames": len(manual), "prediction_npz": str(prediction_path),
            "prediction_sha256": file_hash(prediction_path),
            "psirz_sha256": _sha256_bytes(manual), "bit_identical": True}


def _score(y_true: np.ndarray, y_pred: np.ndarray,
           sample_weight: np.ndarray | None = None) -> dict[str, float]:
    """Use sklearn diagnostics, never the competition metric."""
    explained = float(explained_variance_score(
        y_true, y_pred, sample_weight=sample_weight,
        multioutput="variance_weighted", force_finite=False
    ))
    r2 = float(r2_score(y_true, y_pred, sample_weight=sample_weight,
                        multioutput="variance_weighted",
                        force_finite=False))
    if not np.isfinite([explained, r2]).all():
        raise ValueError(f"diagnostic explained variance/R2 is {explained}/{r2}")
    return {"explained_variance": explained, "r2": r2}


def _fit_diagnostic(x: np.ndarray, y: np.ndarray, calibration: int,
                    alphas: list[float], cv: dict[str, Any],
                    frame_counts: np.ndarray) -> dict[str, Any]:
    """Choose alpha inside the first block; touch the last 338 only after fitting."""
    train_x, test_x = x[:calibration], x[calibration:]
    train_y, test_y = y[:calibration], y[calibration:]
    test_weights = frame_counts[calibration:]
    folds = KFold(n_splits=int(cv["folds"]), shuffle=bool(cv["shuffle"]),
                  random_state=int(cv["random_state"]))
    means: dict[float, float] = {}
    fold_scores: dict[str, list[float]] = {}
    for alpha in alphas:
        scores = []
        for fit_idx, val_idx in folds.split(train_x):
            pipeline = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
            pipeline.fit(train_x[fit_idx], train_y[fit_idx])
            scores.append(_score(train_y[val_idx], pipeline.predict(train_x[val_idx]))[
                "explained_variance"
            ])
        means[alpha] = float(np.mean(scores))
        fold_scores[f"{alpha:g}"] = scores
    selected = max(alphas, key=lambda alpha: (means[alpha], -alpha))
    chosen = make_pipeline(StandardScaler(), Ridge(alpha=selected))
    chosen.fit(train_x, train_y)
    control = DummyRegressor(strategy="mean")
    control.fit(train_x, train_y)
    return {
        "selected_alpha": selected,
        "inner_fold_explained_variance": fold_scores,
        "inner_mean_explained_variance": {f"{k:g}": v for k, v in means.items()},
        "diagnostic": _score(test_y, chosen.predict(test_x)),
        "diagnostic_frame_weighted": _score(test_y, chosen.predict(test_x), test_weights),
        "mean_control": _score(test_y, control.predict(test_x)),
        "mean_control_frame_weighted": _score(test_y, control.predict(test_x),
                                              test_weights),
    }


def main() -> None:
    """Run the parity gate or the entire predeclared validation-only rule-out."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("parity", "run"), required=True)
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.jobs <= 2:
        raise ValueError(f"--jobs={args.jobs}, expected 1 or 2 for this diagnostic")
    spec = json.loads(SPEC.read_text())
    artifact_path = ROOT / spec["artifact"]
    if file_hash(artifact_path) != spec["artifact_sha256"]:
        raise ValueError(f"{artifact_path}: SHA-256 differs from preregistration")
    source_root = Path(spec["source_root"])
    _old_module(source_root)
    artifact = _old_artifact(artifact_path, source_root)
    if int(artifact["n_pca"]) != int(spec["target"]["pca_components"]):
        raise ValueError(f"{artifact_path}: {artifact['n_pca']} PCA components, expected "
                         f"{spec['target']['pca_components']}")
    if int(artifact["scaler"].n_features_in_) != int(spec["feature_sets"]["all"][
        "input_prefix_width"
    ]):
        raise ValueError(f"{artifact_path}: scaler width differs from preregistered features")
    one_frame_duration_ms = float(spec["feature_sets"]["all"]["one_frame_duration_ms"])
    if (one_frame_duration_ms != 0.0 or float(spec["feature_sets"]["without_thomson"][
        "one_frame_duration_ms"
    ]) != one_frame_duration_ms):
        raise ValueError(f"{SPEC}: one-frame duration must be exactly 0 ms in both arms")
    control = _parity_control(spec, artifact)
    print(f"Parity: {control['file']} {control['frames']} frames, bit-identical old prediction")
    if args.mode == "parity":
        return

    names = list(artifact["val_files"])
    calibration = int(spec["selection"]["calibration_count"])
    diagnostic = int(spec["selection"]["diagnostic_count"])
    if len(names) != calibration + diagnostic or len(set(names)) != len(names):
        raise ValueError(f"{artifact_path}: validation has {len(names)} distinct filenames, "
                         f"expected {calibration}+{diagnostic}")
    manifest = json.loads(MANIFEST.read_text())
    entries = manifest["configs"][artifact["config"]]
    folder = Path(spec["data_root"]) / "data" / artifact["config"]
    paths = [folder / name for name in names]
    tasks = []
    for path in paths:
        if path.name not in entries:
            raise ValueError(f"{path}: missing from pinned corpus manifest")
        size, digest = entries[path.name]
        if path.stat().st_size != size:
            raise ValueError(f"{path}: size {path.stat().st_size}, manifest expects {size}")
        tasks.append((path, artifact_path, source_root, digest,
                      int(spec["feature_sets"]["all"]["input_prefix_width"]),
                      int(spec["feature_sets"]["without_thomson"]["input_prefix_width"]),
                      int(spec["target"]["pca_components"]), one_frame_duration_ms))
    rows: list[dict[str, Any]] = []
    try:
        for row in tqdm(pimap(_task, tasks, args.jobs), total=len(tasks),
                        desc="D8 validation summaries", unit="shot", **bar_kwargs(SHOT_EVERY)):
            rows.append(row)
    finally:
        release()
    if [row["file"] for row in rows] != names:
        raise ValueError("D8 extraction returned filenames in a different order")
    x = np.stack([row["summary"] for row in rows])
    y = np.stack([row["target"] for row in rows])
    frame_counts = np.asarray([row["frames"] for row in rows], dtype=np.int64)
    width = int(spec["feature_sets"]["all"]["input_prefix_width"])
    short = int(spec["feature_sets"]["without_thomson"]["input_prefix_width"])
    if x.shape != (len(names), 2 * width + 1) or y.shape != (len(names), int(
        spec["target"]["pca_components"]
    )):
        raise ValueError(f"D8 summaries {x.shape} and targets {y.shape}, unexpected")
    SUMMARIES.parent.mkdir(parents=True, exist_ok=True)
    with SUMMARIES.open("wb") as stream:
        np.savez_compressed(stream, files=np.asarray(names), frames=frame_counts,
                            features=x, targets=y)
    feature_sets = {
        "all": x,
        "without_thomson": np.column_stack([x[:, :short], x[:, width:width + short], x[:, -1]]),
    }
    cv = spec["inner_cv"]
    alphas = [float(a) for a in cv["alphas"]]
    comparisons = {
        name: _fit_diagnostic(value, y, calibration, alphas, cv, frame_counts)
        for name, value in feature_sets.items()
    }
    outcome = comparisons[spec["gate"]["primary_feature_set"]]["diagnostic"][
        "explained_variance"
    ]
    weighted_outcome = comparisons[spec["gate"]["primary_feature_set"]][
        "diagnostic_frame_weighted"
    ]["explained_variance"]
    weighted_floor = float(spec["supplemental_reporting"]["frame_weighted_promotion_floor"])
    record = {
        "kind": "validation-only diagnostic, not competition S",
        "preregistration": str(SPEC.relative_to(ROOT)),
        "preregistration_sha256": file_hash(SPEC),
        "script_sha256": file_hash(Path(__file__)),
        "artifact_sha256": file_hash(artifact_path),
        "source_sha256": {str(p.relative_to(source_root)): file_hash(p) for p in (
            source_root / "my_experiments/baseline_model.py",
            source_root / "my_experiments/models.py",
            source_root / "my_experiments/seq_model.py", source_root / "experiments.py"
        )},
        "manifest_sha256": file_hash(MANIFEST),
        "validation_files": names,
        "validation_file_sha256": {row["file"]: row["source_sha256"] for row in rows},
        "calibration_files": names[:calibration],
        "diagnostic_files": names[calibration:],
        "frames": frame_counts.tolist(),
        "summary_npz": str(SUMMARIES.relative_to(ROOT)),
        "summary_npz_sha256": file_hash(SUMMARIES),
        "parity_control": control,
        "worker_processes": args.jobs,
        "ensemble_model_threads": {
            name: getattr(artifact["models"][name], "threads", None)
            for name in artifact["ensemble"]
        },
        "thread_environment": {name: os.environ.get(name)
                               for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                                            "OPENBLAS_NUM_THREADS")},
        "feature_widths": {name: value.shape[1] for name, value in feature_sets.items()},
        "comparisons": comparisons,
        "gate_value": outcome,
        "gate_minimum": float(spec["gate"]["outer_explained_variance_min"]),
        "gate_passed": outcome >= float(spec["gate"]["outer_explained_variance_min"]),
        "frame_weighted_explained_variance": weighted_outcome,
        "frame_weighted_promotion_floor": weighted_floor,
        "qualified_for_native_followup": (
            outcome >= float(spec["gate"]["outer_explained_variance_min"])
            and weighted_outcome >= weighted_floor
        ),
        "limitation": spec["limitation"],
    }
    write_json(RESULT, record)
    print(f"D8 primary outer explained variance {outcome:.6f}; gate "
          f"{record['gate_minimum']:.2f}: {'PASS' if record['gate_passed'] else 'FAIL'}")
    print(f"D8 frame-weighted outer explained variance {weighted_outcome:.6f}; "
          f"native-followup floor {weighted_floor:.2f}: "
          f"{'PASS' if record['qualified_for_native_followup'] else 'FAIL'}")
    print(f"Wrote {RESULT}")


if __name__ == "__main__":
    main()
