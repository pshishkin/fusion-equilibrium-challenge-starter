"""One fixed nonlinear whole-shot residual gate on genuinely OOF DIII-D shots."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import explained_variance_score
from sklearn.preprocessing import StandardScaler

from my_experiments.oof_split import names_hash
from my_experiments.run import source_files
from toolkit.parallel import pimap, release
from toolkit.records import file_hash, write_json
from tools.b10_head_model import decode_residual

ROOT = Path(__file__).resolve().parent.parent
SPEC = ROOT / "experiments/d8-nonlinear-shot-context-v2.json"
PARAMS = ROOT / "baselines/d8-nonlinear-shot-context.params.yaml"


def shot_summary(selected84: np.ndarray, duration_ms: float) -> np.ndarray:
    """Population mean/std of the frozen 84 inputs plus input-only duration."""
    values = np.asarray(selected84, dtype=np.float64)
    if (values.ndim != 2 or values.shape[1] != 84 or len(values) < 1
            or not np.isfinite(values).all() or not np.isfinite(duration_ms)
            or duration_ms < 0):
        raise ValueError(f"shot summary selected features/duration "
                         f"{values.shape}/{duration_ms}, expected finite (T,84), T>=1, "
                         "nonnegative duration")
    result = np.r_[values.mean(axis=0), values.std(axis=0, ddof=0), duration_ms]
    if result.shape != (169,) or not np.isfinite(result).all():
        raise ValueError(f"whole-shot summary {result.shape}, expected finite (169,)")
    return result


def _read_input_summary(task: tuple[str, int, str, str, str]) -> dict[str, Any]:
    name, expected_frames, data_s, cached_s, expected_cache_sha = task
    path = Path(data_s) / "data/diii_d_train" / name
    times = np.asarray(pd.read_parquet(path, columns=["efit_times"]).iloc[0][
        "efit_times"], dtype=np.float64)
    if (times.shape != (expected_frames,) or not np.isfinite(times).all()
            or (len(times) > 1 and np.any(np.diff(times) <= 0))):
        raise ValueError(f"{path}: EFIT input times {times.shape}/order/finiteness, "
                         f"expected ({expected_frames},) monotone finite")
    cache = Path(cached_s)
    if file_hash(cache) != expected_cache_sha:
        raise ValueError(f"{cache}: frozen OOF input cache SHA differs")
    with np.load(cache, allow_pickle=False) as saved:
        selected = np.asarray(saved["x84"])
    if selected.shape != (expected_frames, 84) or selected.dtype != np.float32:
        raise ValueError(f"{cache}: selected84 shape/dtype "
                         f"{selected.shape}/{selected.dtype}, "
                         f"expected ({expected_frames},84)/float32")
    duration = float(times[-1] - times[0])
    return {"shot": name, "frames": expected_frames,
            "duration_ms": duration, "summary": shot_summary(selected, duration)}


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    if path.exists():
        raise FileExistsError(f"{path}: sealed D8 output already exists")
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".npz")
    try:
        with os.fdopen(fd, "wb") as stream:
            np.savez_compressed(stream, **arrays)  # type: ignore[arg-type]
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _roles(names: list[str]) -> tuple[list[str], list[str]]:
    if len(names) != 1355 or len(set(names)) != 1355:
        raise ValueError(f"D8 OOF filenames {len(names)}/{len(set(names))}, expected 1355")
    ordered = sorted(names, key=lambda name: hashlib.sha256(
        ("d8-nonlinear-v1:" + name).encode()).hexdigest())
    return ordered[:1100], ordered[1100:]


def prepare(run: Path) -> None:
    """Freeze code/config, fit1100/held255 and gate200 before reading targets."""
    started = time.perf_counter()
    run = run.resolve()
    if run.exists():
        raise FileExistsError(f"{run}: D8 nonlinear campaign already exists")
    spec = json.loads(SPEC.read_text())
    if file_hash(Path(__file__)) != spec["helper_sha256"]:
        raise ValueError(f"{SPEC}: D8 helper source SHA differs")
    parent = ROOT / spec["amends_spec"]
    if file_hash(parent) != spec["amends_spec_sha256"]:
        raise ValueError(f"{parent}: reviewed D8 protocol SHA differs")
    if file_hash(PARAMS) != spec["effective_params_sha256"]:
        raise ValueError(f"{PARAMS}: fixed D8 ExtraTrees/Ridge YAML SHA differs")
    inputs = spec["frozen_inputs_and_roles"]
    for key, hash_key in (("OOF_filename_manifest", "OOF_filename_manifest_sha256"),
                          ("B10_sealed_cache_manifest", "B10_sealed_cache_manifest_sha256"),
                          ("B10_target_and_gate_source_cache",
                           "B10_target_and_gate_source_cache_sha256"),
                          ("B10_source_gate200_prediction",
                           "B10_source_gate200_prediction_sha256"),
                          ("B10_source_gate200_native_json",
                           "B10_source_gate200_native_json_sha256"),
                          ("base_artifact", "base_artifact_sha256")):
        path = ROOT / inputs[key]
        if file_hash(path) != inputs[hash_key]:
            raise ValueError(f"{path}: frozen D8 {key} SHA differs")
    files = json.loads((ROOT / inputs["OOF_filename_manifest"]).read_text())
    old = json.loads((ROOT / inputs["B10_sealed_cache_manifest"]).read_text())
    cached = json.loads((ROOT / inputs["B10_target_and_gate_source_cache"]).read_text())
    oof = files["folds"][0]["oof_files"]
    fit, held = _roles(oof)
    if (names_hash(fit) != spec["role_hashes"]["fit1100"]
            or names_hash(held) != spec["role_hashes"]["held255"]
            or old["head_train_files"] != oof or old["gate_files"] != files["gate_files"]
            or old["reserved_tail_files"] != files["reserved_tail_files"]
            or [row["shot"] for row in cached["entries"][:1355]] != oof
            or [row["shot"] for row in cached["entries"][1355:]] != old["gate_files"]
            or len(set(oof) & (set(old["gate_files"]) | set(old["reserved_tail_files"])))):
        raise ValueError(f"{SPEC}: OOF fit/held/gate/tail roles or hashes differ")
    artifact: dict[str, Any] = joblib.load(ROOT / inputs["base_artifact"])
    if (set(artifact["train_files"]) & set(oof + old["gate_files"] +
                                            old["reserved_tail_files"])
            or artifact["val_files"] != []):
        raise ValueError(f"{inputs['base_artifact']}: OOF/gate/tail overlap in base fit")
    run.mkdir(parents=True)
    source = run / "source"
    source.mkdir()
    for path in source_files():
        target = source / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    for path in (SPEC, parent, PARAMS):
        target = source / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    hashes = {str(path.relative_to(ROOT)): file_hash(source / path.relative_to(ROOT))
              for path in source_files()}
    hashes[str(SPEC.relative_to(ROOT))] = file_hash(source / SPEC.relative_to(ROOT))
    hashes[str(parent.relative_to(ROOT))] = file_hash(source / parent.relative_to(ROOT))
    hashes[str(PARAMS.relative_to(ROOT))] = file_hash(source / PARAMS.relative_to(ROOT))
    write_json(run / "manifest.json", {
        "classification": "D8_nonlinear_shot_context_frozen_before_target_summary_or_fit",
        "origin_root": str(ROOT),
        "data_dir": str(ROOT.parent / "downloaded_huggingface/hf_dataset"),
        "source_sha256": hashes,
        "spec_sha256": file_hash(SPEC),
        "OOF_filenames": oof,
        "fit_files": fit,
        "held_files": held,
        "gate_files": old["gate_files"],
        "reserved_tail_files": old["reserved_tail_files"],
        "prepare_wall_seconds": time.perf_counter() - started,
    })
    subprocess.run([
        sys.executable, "-c",
        "import pathlib,tools.d8_nonlinear_gate as g; "
        "assert pathlib.Path(g.__file__).resolve()=="
        "pathlib.Path('tools/d8_nonlinear_gate.py').resolve()",
    ], cwd=source, env={**os.environ, "PYTHONPATH": str(source)}, check=True)
    print(f"Frozen D8 nonlinear gate {run}: 1100 fit, 255 held, 200 native gate")


def _context(run: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    if ROOT.resolve() != (run / "source").resolve():
        raise ValueError(f"D8 code imported from {ROOT}, expected {run / 'source'}")
    manifest = json.loads((run / "manifest.json").read_text())
    spec_path = run / "source/experiments/d8-nonlinear-shot-context-v2.json"
    spec = json.loads(spec_path.read_text())
    if file_hash(spec_path) != manifest["spec_sha256"]:
        raise ValueError(f"{spec_path}: frozen D8 spec SHA differs")
    for relative, expected in manifest["source_sha256"].items():
        path = run / "source" / relative
        if file_hash(path) != expected:
            raise ValueError(f"{path}: frozen D8 source SHA differs")
    return manifest, spec


def run(run_dir: Path, jobs: int) -> None:
    """One ExtraTrees/Ridge held-shot gate, then at most one native candidate."""
    started = time.perf_counter()
    run_dir = run_dir.resolve()
    manifest, spec = _context(run_dir)
    result_path = run_dir / "result.json"
    if result_path.exists():
        raise FileExistsError(f"{result_path}: D8 gate already measured")
    origin = Path(manifest["origin_root"])
    inputs = spec["frozen_inputs_and_roles"]
    cache = json.loads((origin / inputs["B10_target_and_gate_source_cache"]).read_text())
    by_name = {entry["shot"]: entry for entry in cache["entries"]}
    names = manifest["fit_files"] + manifest["held_files"]
    input_tasks = [(name, by_name[name]["frames"], manifest["data_dir"],
                    by_name[name]["output_path"], by_name[name]["output_sha256"])
                   for name in names]
    feature_started = time.perf_counter()
    summaries: list[np.ndarray] = []
    frames: list[int] = []
    try:
        for i, item in enumerate(pimap(_read_input_summary, input_tasks, jobs)):
            if item["shot"] != names[i] or item["frames"] != input_tasks[i][1]:
                raise ValueError(f"{names[i]}: input summary filename/frame order differs")
            summaries.append(item["summary"])
            frames.append(item["frames"])
            if (i + 1) % 100 == 0:
                print(f"D8 input-only summaries {i + 1}/1355 shots", flush=True)
    finally:
        release()
    X = np.stack(summaries)
    frame_counts = np.asarray(frames, dtype=np.int64)
    if (X.shape != (1355, 169) or int(frame_counts[:1100].sum()) != 244043
            or int(frame_counts[1100:].sum()) != 56750):
        raise ValueError(f"{run_dir}: summary shape/fit-held frame totals "
                         f"{X.shape}/{int(frame_counts[:1100].sum())}/"
                         f"{int(frame_counts[1100:].sum())}, "
                         "expected (1355,169)/244043/56750")
    feature_seconds = time.perf_counter() - feature_started
    scaler = StandardScaler().fit(X[:1100])
    Xfit = scaler.transform(X[:1100])
    Xheld = scaler.transform(X[1100:])
    y_fit: list[np.ndarray] = []
    for name in manifest["fit_files"]:
        entry = by_name[name]
        path = Path(entry["output_path"])
        if file_hash(path) != entry["output_sha256"]:
            raise ValueError(f"{path}: fit residual target SHA differs")
        with np.load(path, allow_pickle=False) as saved:
            target = np.asarray(saved["target_scaled52"])
        if target.shape != (entry["frames"], 52) or not np.isfinite(target).all():
            raise ValueError(f"{name}: fit residual frames/52/finiteness differs")
        y_fit.append(np.mean(target, axis=0, dtype=np.float64))
    Yfit = np.stack(y_fit)
    params_path = run_dir / "source/baselines/d8-nonlinear-shot-context.params.yaml"
    if file_hash(params_path) != spec["effective_params_sha256"]:
        raise ValueError(f"{params_path}: frozen D8 model YAML SHA differs")
    cfg = yaml.safe_load(params_path.read_text())
    if (set(cfg) != {"extra_trees", "ridge"}
            or set(cfg["extra_trees"]) != {"n_estimators", "max_depth",
                                               "min_samples_leaf", "max_features",
                                               "bootstrap", "random_state", "n_jobs"}
            or set(cfg["ridge"]) != {"alpha", "fit_intercept"}):
        raise ValueError(f"{params_path}: D8 model YAML keys differ from frozen recipe")
    fit_started = time.perf_counter()
    tree = ExtraTreesRegressor(**cfg["extra_trees"])
    ridge = Ridge(**cfg["ridge"])
    tree.fit(Xfit, Yfit)
    ridge.fit(Xfit, Yfit)
    mean = np.mean(Yfit, axis=0)
    fit_seconds = time.perf_counter() - fit_started
    tree_pred = np.asarray(tree.predict(Xheld), dtype=np.float64)
    ridge_pred = np.asarray(ridge.predict(Xheld), dtype=np.float64)
    mean_pred = np.broadcast_to(mean, tree_pred.shape).copy()
    if (tree_pred.shape != (255, 52) or ridge_pred.shape != tree_pred.shape
            or not np.isfinite(tree_pred).all() or not np.isfinite(ridge_pred).all()):
        raise ValueError(f"{run_dir}: held255 predictor shapes/finiteness differ")
    model_path = run_dir / "nonlinear_and_ridge.joblib"
    if model_path.exists():
        raise FileExistsError(f"{model_path}: D8 models already fitted")
    joblib.dump({"input_scaler": scaler, "extra_trees": tree, "ridge": ridge,
                 "fit_mean": mean, "fit_files": manifest["fit_files"],
                 "spec_sha256": manifest["spec_sha256"]}, model_path)
    pred_path = run_dir / "held255_predictions.npz"
    _atomic_npz(pred_path, {"extra_trees": tree_pred.astype(np.float32),
                            "ridge": ridge_pred.astype(np.float32),
                            "fit_mean": mean_pred.astype(np.float32)})
    held_y: list[np.ndarray] = []
    for name in manifest["held_files"]:
        entry = by_name[name]
        path = Path(entry["output_path"])
        if file_hash(path) != entry["output_sha256"]:
            raise ValueError(f"{path}: held residual target SHA differs")
        with np.load(path, allow_pickle=False) as saved:
            target = np.asarray(saved["target_scaled52"])
        if target.shape != (entry["frames"], 52) or not np.isfinite(target).all():
            raise ValueError(f"{name}: held residual frames/52/finiteness differs")
        held_y.append(np.mean(target, axis=0, dtype=np.float64))
    Yheld = np.stack(held_y)
    internal: dict[str, dict[str, Any]] = {}
    for label, predicted in (("extra_trees", tree_pred), ("ridge", ridge_pred),
                             ("fit_mean", mean_pred)):
        internal[label] = {
            "frame_weighted_explained_variance": float(explained_variance_score(
                Yheld, predicted, sample_weight=frame_counts[1100:],
                multioutput="variance_weighted")),
            "unweighted_explained_variance": float(explained_variance_score(
                Yheld, predicted, multioutput="variance_weighted")),
            "per_output_explained_variance": np.asarray(explained_variance_score(
                Yheld, predicted, sample_weight=frame_counts[1100:],
                multioutput="raw_values")).tolist(),
        }
    value = internal["extra_trees"]["frame_weighted_explained_variance"]
    pass_internal = bool(np.isfinite(value) and value >= 0.25)
    record: dict[str, Any] = {
        "classification": "D8_nonlinear_OOF1355_fixed_shot_mean_residual_gate",
        "campaign_manifest_sha256": file_hash(run_dir / "manifest.json"),
        "spec_sha256": manifest["spec_sha256"],
        "fit_shots": 1100, "fit_frames": 244043,
        "held_shots": 255, "held_frames": 56750,
        "fit_names_sha256": names_hash(manifest["fit_files"]),
        "held_names_sha256": names_hash(manifest["held_files"]),
        "model_sha256": file_hash(model_path),
        "held_predictions_sha256": file_hash(pred_path),
        "internal_metrics": internal,
        "internal_threshold": 0.25,
        "internal_gate_passed": pass_internal,
        "input_feature_build_wall_seconds": feature_seconds,
        "nonlinear_and_ridge_fit_wall_seconds": fit_seconds,
        "gate200_target_columns_read_before_prediction": [],
        "reserved_tail70_accessed": False,
    }
    if not pass_internal:
        record["decision"] = "STOP_before_gate200_native_or_candidate_maps"
        record["total_wall_seconds"] = time.perf_counter() - started
        write_json(result_path, record)
        print(f"D8 nonlinear held255 weighted EV {value:.6f} < 0.25; stop before native gate")
        return
    native_started = time.perf_counter()
    gate_names = manifest["gate_files"]
    gate_tasks = [(name, by_name[name]["frames"], manifest["data_dir"],
                   by_name[name]["output_path"], by_name[name]["output_sha256"])
                  for name in gate_names]
    gate_summaries: list[np.ndarray] = []
    try:
        for i, item in enumerate(pimap(_read_input_summary, gate_tasks, jobs)):
            if item["shot"] != gate_names[i] or item["frames"] != gate_tasks[i][1]:
                raise ValueError(f"{gate_names[i]}: input-only gate summary order differs")
            gate_summaries.append(item["summary"])
    finally:
        release()
    Xgate = scaler.transform(np.stack(gate_summaries))
    gate_predictions = np.asarray(tree.predict(Xgate), dtype=np.float64)
    if gate_predictions.shape != (200, 52) or not np.isfinite(gate_predictions).all():
        raise ValueError(f"{run_dir}: gate model prediction {gate_predictions.shape}/"
                         f"{int((~np.isfinite(gate_predictions)).sum())}, "
                         "expected (200,52)/0")
    artifact: dict[str, Any] = joblib.load(origin / inputs["base_artifact"])
    arrays: dict[str, np.ndarray] = {}
    for i, name in enumerate(gate_names):
        entry = by_name[name]
        path = Path(entry["output_path"])
        if file_hash(path) != entry["output_sha256"]:
            raise ValueError(f"{path}: input-only gate source SHA differs")
        with np.load(path, allow_pickle=False) as saved:
            psi = np.asarray(saved["source_psi"])
            q95 = np.asarray(saved["source_q95"])
            beta = np.asarray(saved["source_betaN"])
            base = np.asarray(saved["base_scaled52"])
        zero = decode_residual(psi, q95, beta, base,
                               np.zeros((entry["frames"], 52)), artifact)
        if any(zero[field].tobytes() != source.tobytes()
               for field, source in (("psirz", psi), ("q95", q95), ("betaN", beta))):
            raise ValueError(f"{name}: zero-residual source bytes changed")
        output = decode_residual(psi, q95, beta, base,
                                 np.broadcast_to(gate_predictions[i],
                                                 (entry["frames"], 52)), artifact)
        for field in ("psirz", "q95", "betaN"):
            arrays[f"shot_{i:04d}_{field}"] = output[field]
    candidate_path = run_dir / "gate200_candidate.npz"
    _atomic_npz(candidate_path, arrays)
    write_json(candidate_path.with_suffix(".json"), {
        "files": gate_names,
        "prediction_sha256": file_hash(candidate_path),
        "model_sha256": file_hash(model_path),
        "gate_target_columns_requested": [],
        "zero_residual_exact_source_bytes_checked_shots": 200,
    })
    record["gate_candidate_build_wall_seconds"] = time.perf_counter() - native_started
    source_npz = origin / inputs["B10_source_gate200_prediction"]
    source_metrics = origin / inputs["B10_source_gate200_native_json"]
    if (file_hash(source_npz) != inputs["B10_source_gate200_prediction_sha256"]
            or file_hash(source_metrics) != inputs["B10_source_gate200_native_json_sha256"]):
        raise ValueError(f"{run_dir}: pinned B10 gate source prediction/native SHA differs")
    scorer = origin / inputs["native_scorer_root"]
    if file_hash(scorer / "local_score.py") != inputs["native_local_score_py_sha256"]:
        raise ValueError(f"{scorer}: native scorer SHA differs")
    files_path = run_dir / "gate200_files.json"
    write_json(files_path, {"files": gate_names})
    scored_path = run_dir / "gate200_candidate.metrics.json"
    score_started = time.perf_counter()
    subprocess.run([
        sys.executable, "-m", "tools.historical_compare", "score",
        "--prediction", str(candidate_path), "--files-json", str(files_path),
        "--data-dir", manifest["data_dir"], "--metrics-json", str(scored_path),
        "--jobs", str(jobs), "--scorer-root", str(scorer),
    ], cwd=run_dir / "source", env={**os.environ, "PYTHONPATH": str(run_dir / "source")},
        check=True)
    source_native = json.loads(source_metrics.read_text())
    candidate_native = json.loads(scored_path.read_text())
    if source_native["files"] != gate_names or candidate_native["files"] != gate_names:
        raise ValueError(f"{run_dir}: source/candidate native scored gate filenames differ")
    original = source_native["results"]["model"]
    corrected = candidate_native["results"]["model"]
    delta = corrected["S"] - original["S"]
    failure_ok = bool(corrected["lcfs_fail_frac"] <= original["lcfs_fail_frac"]
                      and corrected["cons_fail_frac"] <= original["cons_fail_frac"])
    record.update({
        "gate_shots": 200,
        "gate_frames": 43142,
        "gate_candidate_prediction_sha256": file_hash(candidate_path),
        "gate_candidate_native_json_sha256": file_hash(scored_path),
        "source_native": original,
        "candidate_native": corrected,
        "candidate_minus_source_S": delta,
        "required_native_gain": 0.0026,
        "native_no_increased_failures": failure_ok,
        "native_gate_passed": bool(delta >= 0.0026 and failure_ok),
        "native_scorer_wall_seconds": time.perf_counter() - score_started,
        "decision": ("root_review_one_whole_shot_model_only" if delta >= 0.0026 and failure_ok
                     else "STOP_after_fixed_native_gate"),
        "total_wall_seconds": time.perf_counter() - started,
    })
    write_json(result_path, record)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "run"):
        part = sub.add_parser(command)
        part.add_argument("--campaign", type=Path, required=True)
        part.add_argument("--jobs", type=int, default=24)
    args = parser.parse_args()
    if not 1 <= args.jobs <= 24:
        raise ValueError(f"jobs={args.jobs}, expected 1..24")
    if args.command == "prepare":
        prepare(args.campaign)
    else:
        run(args.campaign, args.jobs)


if __name__ == "__main__":
    main()
