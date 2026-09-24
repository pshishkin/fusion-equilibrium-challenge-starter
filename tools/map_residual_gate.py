"""Predict and natively gate one fixed physical-map U-Net across source families."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
import yaml

from my_experiments import baseline_model, run
from my_experiments.map_residual_unet import MapResidualUNet, apply_delta, grid_channels
from toolkit.parallel import pimap, release
from toolkit.records import file_hash, write_json
from tools.direct_contour_compact_val_predict import INPUT_COLUMNS
from tools.map_residual_campaign import ROOT, verify_campaign
from tools.map_residual_data import _atomic_npz

# The vendored scorer has flat imports inside fusion_scoring; this points to this
# campaign's frozen copy, whose functional hashes are checked against the scorer.
sys.path.insert(0, str(ROOT / "fusion_scoring"))
from fusion_scoring import common, derive, lcfs

KINDS = ("fold0", "fold1", "mean2")
FIELDS = ("psirz", "q95", "betaN")


def _source_one(task: tuple[str, str, int, str, str, str, str]) -> dict[str, Any]:
    name, data_s, size, expected_sha, base_s, feature_artifact_s, output_s = task
    path = Path(data_s) / "data/diii_d_train" / name
    if path.stat().st_size != size or file_hash(path) != expected_sha:
        raise ValueError(f"{path}: input-only gate parquet bytes/SHA differ")
    row = pd.read_parquet(path, columns=INPUT_COLUMNS).iloc[0]
    forbidden = [key for key in row.index if key.startswith("efit_")
                 and key not in ("efit_times", "efit_grid_R", "efit_grid_Z")]
    if forbidden:
        raise ValueError(f"{path}: gate inference row contains EFIT targets {forbidden}")
    n = len(np.asarray(row["efit_times"]))
    selected_artifact: dict[str, Any] = joblib.load(feature_artifact_s)
    features = baseline_model.features_for_row(row)
    selected = baseline_model.build_inputs(selected_artifact["coil"], features)
    predicted = baseline_model.predict_row(
        baseline_model.slim_row(row), "DIII-D", "mlp", Path(base_s))
    source = np.asarray(predicted["psirz"], dtype=np.float32)
    q95 = np.asarray(predicted["q95"], dtype=np.float32)
    beta = np.asarray(predicted["betaN"], dtype=np.float32)
    if (source.shape != (n, 65, 65) or selected.shape != (n, 84)
            or q95.shape != (n,) or beta.shape != (n,)
            or not np.isfinite(source).all() or not np.isfinite(selected).all()
            or not np.isfinite(q95).all() or not np.isfinite(beta).all()):
        raise ValueError(f"{path}: source map/features/scalars {source.shape}/"
                         f"{selected.shape}/{q95.shape}/{beta.shape}, "
                         f"expected ({n},65,65)/({n},84)/({n},)/({n},) finite")
    output = Path(output_s) / f"{Path(name).stem}.npz"
    _atomic_npz(output, {"source_psi": source,
                         "selected84": np.asarray(selected, dtype=np.float32),
                         "q95": q95, "betaN": beta})
    return {"shot": name, "frames": n, "source_parquet_sha256": expected_sha,
            "gate_efit_target_columns_requested": [],
            "output_path": str(output), "output_sha256": file_hash(output)}


def _mean_one(task: tuple[str, str, str, str]) -> dict[str, Any]:
    name, fold0_s, fold1_s, output_s = task
    with (np.load(fold0_s, allow_pickle=False) as first,
          np.load(fold1_s, allow_pickle=False) as second):
        a = {key: np.asarray(first[key]) for key in ("source_psi", "selected84", "q95", "betaN")}
        b = {key: np.asarray(second[key]) for key in ("source_psi", "selected84", "q95", "betaN")}
    if a["selected84"].tobytes() != b["selected84"].tobytes():
        raise ValueError(f"{name}: fold0/fold1 allowed feature bytes differ")
    frames = len(a["source_psi"])
    if (a["source_psi"].shape != (frames, 65, 65)
            or b["source_psi"].shape != a["source_psi"].shape
            or any(a[key].shape != (frames,) or b[key].shape != (frames,)
                   for key in ("q95", "betaN"))):
        raise ValueError(f"{name}: source family map/scalar frame counts differ")
    arrays = {
        "source_psi": ((a["source_psi"].astype(np.float64)
                        + b["source_psi"].astype(np.float64)) / 2).astype(np.float32),
        "selected84": a["selected84"],
        "q95": ((a["q95"].astype(np.float64)
                 + b["q95"].astype(np.float64)) / 2).astype(np.float32),
        "betaN": ((a["betaN"].astype(np.float64)
                   + b["betaN"].astype(np.float64)) / 2).astype(np.float32),
    }
    if any(not np.isfinite(value).all() for value in arrays.values()):
        raise ValueError(f"{name}: decoded physical two-base mean contains nonfinite values")
    output = Path(output_s) / f"{Path(name).stem}.npz"
    _atomic_npz(output, arrays)
    return {"shot": name, "frames": frames,
            "gate_efit_target_columns_requested": [],
            "output_path": str(output), "output_sha256": file_hash(output)}


def cache_source(campaign: Path, kind: str, jobs: int) -> None:
    """Seal input-only physical source predictions in gate filename order."""
    started = time.perf_counter()
    campaign = campaign.resolve()
    meta = verify_campaign(campaign)
    if kind not in KINDS or not 1 <= jobs <= 24:
        raise ValueError(f"source kind/jobs {kind}/{jobs}, expected {KINDS}/1..24")
    ready_path = campaign / f"gate_source_{kind}_ready.json"
    if ready_path.exists():
        raise FileExistsError(f"{ready_path}: input-only gate source already sealed")
    names = meta["gate_files"]
    output_dir = campaign / "gate_sources" / kind
    output_dir.mkdir(parents=True, exist_ok=True)
    if kind == "mean2":
        sources = [json.loads((campaign / f"gate_source_{previous}_ready.json").read_text())
                   for previous in ("fold0", "fold1")]
        if any([entry["shot"] for entry in source["entries"]] != names for source in sources):
            raise ValueError(f"{campaign}: fold0/fold1 gate source order differs")
        mean_tasks = [(name, sources[0]["entries"][i]["output_path"],
                       sources[1]["entries"][i]["output_path"], str(output_dir))
                      for i, name in enumerate(names)]
        source_artifact_sha = "decoded_equal_fold0_fold1"
    else:
        origin = Path(meta["origin_root"])
        feature_artifact = origin / "runs/oof-normal-k8-5fold-20260923/fold0/base.joblib"
        base = (feature_artifact if kind == "fold0"
                else campaign / "fold1/base.joblib")
        if not base.is_file():
            raise FileNotFoundError(f"{base}: {kind} base artifact is missing")
        fitted: dict[str, Any] = joblib.load(base)
        old_campaign = json.loads((origin / "runs/oof-normal-k8-5fold-20260923/"
                                   "campaign.json").read_text())
        old_files = json.loads((origin / "runs/oof-normal-k8-5fold-20260923/"
                                "files.json").read_text())
        fold_index = 0 if kind == "fold0" else 1
        own_oof = old_files["folds"][fold_index]["oof_files"]
        own_set = set(own_oof)
        expected_train = [name for name in old_files["allowed_files"]
                          if name not in own_set]
        if (set(names) & (set(fitted["train_files"]) | set(fitted["val_files"]))
                or fitted["train_files"] != expected_train or fitted["val_files"] != []
                or fitted["oof_files"] != own_oof or fitted["oof_fold"] != fold_index
                or set(fitted["models"]) != {"mlp"}
                or fitted["fit_mode"] != "fixed_steps"
                or fitted["source_sha256"] != old_campaign["source_bundle_sha256"]
                or fitted["models"]["mlp"]._best_step != 33000):
            raise ValueError(f"{base}: fold{fold_index} fit/OOF/gate/step/source "
                             "identity differs from frozen manifest")
        if kind == "fold0" and file_hash(base) != meta["fold0_artifact_sha256"]:
            raise ValueError(f"{base}: fold0 base SHA differs from frozen campaign")
        dataset = json.loads((campaign / "source/my_experiments/diii_d_dataset.json").read_text())
        source_tasks = [(name, meta["data_dir"], *dataset["configs"]["diii_d_train"][name],
                         str(base), str(feature_artifact), str(output_dir)) for name in names]
        source_artifact_sha = file_hash(base)
    try:
        entries = (list(pimap(_mean_one, mean_tasks, jobs)) if kind == "mean2"
                   else list(pimap(_source_one, source_tasks, jobs)))
    finally:
        release()
    if ([entry["shot"] for entry in entries] != names
            or sum(entry["frames"] for entry in entries) != 43142
            or any(entry["gate_efit_target_columns_requested"] for entry in entries)):
        raise ValueError(f"{campaign}: {kind} gate source order/frames/target stripping differs")
    parity_arrays = 0
    if kind == "fold0":
        prior = Path(meta["origin_root"]) / (
            "runs/b10-matched-heads-20260923/gate_predictions/source.npz")
        spec_path = campaign / "source/experiments/map-residual-unet-review.json"
        spec = json.loads(spec_path.read_text())
        if file_hash(prior) != spec["frozen_references"]["prior_fold0_gate_source_sha256"]:
            raise ValueError(f"{prior}: prior fold0 input-only source SHA differs")
        with np.load(prior, allow_pickle=False) as packed:
            for i, entry in enumerate(entries):
                with np.load(entry["output_path"], allow_pickle=False) as local:
                    for field, key in (("psirz", "source_psi"),
                                       ("q95", "q95"), ("betaN", "betaN")):
                        if local[key].tobytes() != packed[f"shot_{i:04d}_{field}"].tobytes():
                            raise ValueError(f"{entry['shot']}: new input-only {field} differs "
                                             "from pinned prior fold0 source")
                        parity_arrays += 1
    write_json(ready_path, {
        "classification": f"input_only_gate_{kind}_physical_source",
        "source_bundle_sha256": meta["source_bundle_sha256"],
        "source_artifact_sha256": source_artifact_sha,
        "shots": len(entries),
        "frames": sum(entry["frames"] for entry in entries),
        "parity_arrays_with_prior_fold0": parity_arrays,
        "gate_target_columns_requested": [],
        "entries": entries,
        "elapsed_seconds": time.perf_counter() - started,
    })


def _li(psi: np.ndarray, grid_r: np.ndarray, grid_z: np.ndarray,
        mask: np.ndarray) -> tuple[float | None, str | None]:
    sign = int(common.AXIS_SIGN["DIII-D"])
    physical = psi.astype(np.float64)
    contour, _ = lcfs.extract_lcfs_with_sign(
        physical, grid_r, grid_z, "DIII-D", mask, mask.astype(float), axis_sign=sign)
    if contour is None:
        return None, "lcfs_invalid"
    value = float(derive.internal_inductance(physical, grid_r, grid_z, "DIII-D", contour))
    if not np.isfinite(value):
        return None, "li_invalid"
    return value, None


def guard_maps(source: np.ndarray, candidate: np.ndarray,
               grid_r: np.ndarray, grid_z: np.ndarray, mask: np.ndarray,
               threshold: float) -> tuple[np.ndarray, dict[str, int], np.ndarray]:
    """Exact source fallback for explicit input-only li/contour reasons."""
    original = np.asarray(source)
    changed = np.asarray(candidate)
    if (original.shape != changed.shape or original.ndim != 3
            or original.shape[1:] != (65, 65)
            or original.dtype != np.float32 or changed.dtype != np.float32
            or not np.isfinite(original).all() or not np.isfinite(changed).all()
            or threshold != 0.05):
        raise ValueError(f"guard source/candidate {original.shape}/{changed.shape} "
                         f"{original.dtype}/{changed.dtype}, expected finite matching "
                         "float32 T/65/65 and threshold 0.05")
    out = changed.copy()
    reasons = {"source_lcfs_invalid": 0, "source_li_invalid": 0,
               "candidate_lcfs_invalid": 0, "candidate_li_invalid": 0,
               "abs_li_change_over_threshold": 0}
    differences = np.full(len(original), np.nan, dtype=np.float64)
    for i, (before, after) in enumerate(zip(original, changed, strict=True)):
        source_li, source_reason = _li(before, grid_r, grid_z, mask)
        if source_reason is not None:
            reasons[f"source_{source_reason}"] += 1
            out[i] = before
            continue
        candidate_li, candidate_reason = _li(after, grid_r, grid_z, mask)
        if candidate_reason is not None:
            reasons[f"candidate_{candidate_reason}"] += 1
            out[i] = before
            continue
        assert source_li is not None and candidate_li is not None
        differences[i] = candidate_li - source_li
        if abs(differences[i]) > threshold:
            reasons["abs_li_change_over_threshold"] += 1
            out[i] = before
    return out, reasons, differences


def _guard_one(task: tuple[str, str, str, str, str, str]) -> dict[str, Any]:
    name, source_s, checkpoint_s, scaler_s, params_s, output_s = task
    with np.load(source_s, allow_pickle=False) as packed:
        psi = np.asarray(packed["source_psi"])
        q95 = np.asarray(packed["q95"])
        beta = np.asarray(packed["betaN"])
        selected = np.asarray(packed["selected84"])
    frames = len(psi)
    if (psi.shape != (frames, 65, 65) or psi.dtype != np.float32
            or selected.shape != (frames, 84) or selected.dtype != np.float32
            or q95.shape != (frames,) or beta.shape != (frames,)
            or q95.dtype != np.float32 or beta.dtype != np.float32):
        raise ValueError(f"{name}: gate source map/features/scalars shape/dtype differs")
    with np.load(ROOT / "fusion_scoring/masks/d3d_envelope.npz") as packed:
        r = np.asarray(packed["grid_R"], dtype=np.float64)
        z = np.asarray(packed["grid_Z"], dtype=np.float64)
        mask = np.asarray(packed["mask_coarse"], dtype=np.bool_)
    params: dict[str, Any] = yaml.safe_load(Path(params_s).read_text())
    channels, extra, scale = grid_channels(
        psi, r, z, mask, int(params["source"]["source_range_min_float32_ulps"]), name)
    features86 = np.column_stack([selected, extra]).astype(np.float32)
    scaler = joblib.load(scaler_s)
    if scaler.n_features_in_ != 86:
        raise ValueError(f"{scaler_s}: U-Net head feature width differs from 86")
    standardized = scaler.transform(features86).astype(np.float32)
    checkpoint = torch.load(checkpoint_s, map_location="cpu", weights_only=False)
    model = MapResidualUNet(params["model"], 86)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    torch.set_num_threads(1)
    parts: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, frames, 64):
            stop = min(frames, start + 64)
            out = model(torch.from_numpy(channels[start:stop].copy()),
                        torch.from_numpy(standardized[start:stop].copy()))
            parts.append(out[:, 0].numpy())
    predicted_delta = np.concatenate(parts).astype(np.float32)
    candidate = apply_delta(psi, predicted_delta, scale, name)
    guarded, reasons, differences = guard_maps(
        psi, candidate, r, z, mask,
        float(params["inference"]["delta_li_source_fallback_threshold"]))
    output = Path(output_s) / f"{Path(name).stem}.npz"
    _atomic_npz(output, {"psirz": guarded, "q95": q95, "betaN": beta})
    with np.load(output, allow_pickle=False) as saved:
        if saved["q95"].tobytes() != q95.tobytes() or saved["betaN"].tobytes() != beta.tobytes():
            raise ValueError(f"{output}: submitted scalars changed from source bits")
    finite = np.abs(differences[np.isfinite(differences)])
    return {
        "shot": name, "frames": frames,
        "fallback_reasons": reasons,
        "finite_li_difference_frames": len(finite),
        "mean_abs_li_difference": float(np.mean(finite)) if len(finite) else None,
        "max_abs_li_difference": float(np.max(finite)) if len(finite) else None,
        "source_scalar_bits_unchanged": True,
        "output_path": str(output), "output_sha256": file_hash(output),
    }


def predict_guarded(campaign: Path, kind: str, jobs: int) -> None:
    """Seal source-conditioned corrected maps and explicit fallback counts."""
    started = time.perf_counter()
    campaign = campaign.resolve()
    meta = verify_campaign(campaign)
    if kind not in KINDS or not 1 <= jobs <= 24:
        raise ValueError(f"guard kind/jobs {kind}/{jobs}, expected {KINDS}/1..24")
    source_ready = json.loads((campaign / f"gate_source_{kind}_ready.json").read_text())
    if ([entry["shot"] for entry in source_ready["entries"]] != meta["gate_files"]
            or source_ready["frames"] != 43142):
        raise ValueError(f"{campaign}: gate {kind} source roles or frames differ")
    checkpoint = campaign / "models/fold0_map_residual.pt"
    checkpoint_record = json.loads(checkpoint.with_suffix(".json").read_text())
    if file_hash(checkpoint) != checkpoint_record["checkpoint_sha256"]:
        raise ValueError(f"{checkpoint}: fixed head checkpoint SHA differs")
    assembly = json.loads((campaign / "assembly.json").read_text())
    scaler = campaign / "fit_data/scaler.joblib"
    if file_hash(scaler) != assembly["files_sha256"]["scaler.joblib"]:
        raise ValueError(f"{scaler}: train-only scaler SHA differs")
    output_dir = campaign / "gate_predictions" / kind
    output_dir.mkdir(parents=True, exist_ok=True)
    ready_path = campaign / f"gate_guarded_{kind}_ready.json"
    if ready_path.exists():
        raise FileExistsError(f"{ready_path}: guarded gate predictions already sealed")
    tasks = [(entry["shot"], entry["output_path"], str(checkpoint), str(scaler),
              str(campaign / "source/baselines/map-residual-unet.params.yaml"),
              str(output_dir)) for entry in source_ready["entries"]]
    try:
        entries = list(pimap(_guard_one, tasks, jobs))
    finally:
        release()
    if ([entry["shot"] for entry in entries] != meta["gate_files"]
            or sum(entry["frames"] for entry in entries) != 43142
            or not all(entry["source_scalar_bits_unchanged"] for entry in entries)):
        raise ValueError(f"{campaign}: guarded {kind} order/frames/scalar identity differs")
    reasons = {key: sum(entry["fallback_reasons"][key] for entry in entries)
               for key in entries[0]["fallback_reasons"]}
    write_json(ready_path, {
        "classification": f"sealed_input_only_li_guarded_unet_{kind}",
        "source_bundle_sha256": meta["source_bundle_sha256"],
        "checkpoint_sha256": file_hash(checkpoint),
        "gate_source_ready_sha256": file_hash(campaign / f"gate_source_{kind}_ready.json"),
        "shots": len(entries), "frames": 43142,
        "fallback_reasons": reasons,
        "fallback_frames": sum(reasons.values()),
        "fallback_fraction": sum(reasons.values()) / 43142,
        "source_scalar_bits_unchanged": True,
        "entries": entries,
        "elapsed_seconds": time.perf_counter() - started,
    })


def _aggregate(paths: list[Path], names: list[str], output: Path,
               source: bool) -> None:
    arrays: dict[str, np.ndarray] = {}
    for i, (name, path) in enumerate(zip(names, paths, strict=True)):
        if not path.is_file():
            raise FileNotFoundError(f"{path}: {name} gate prediction is missing")
        with np.load(path, allow_pickle=False) as packed:
            for field in FIELDS:
                source_key = "source_psi" if source and field == "psirz" else field
                array = np.asarray(packed[source_key])
                if array.dtype != np.float32 or not np.isfinite(array).all():
                    raise ValueError(f"{path}: {source_key} not finite float32")
                arrays[f"shot_{i:04d}_{field}"] = array
    _atomic_npz(output, arrays)
    write_json(output.with_suffix(".json"), {
        "schema": "submission_npz_v1", "files": names,
        "prediction_sha256": file_hash(output),
        "method": "sealed physical source" if source else "fixed U-Net plus source-li guard",
    })


def _native_score(prediction: Path, files_json: Path, data: Path, output: Path,
                  scorer_root: Path, source_root: Path, jobs: int) -> dict[str, Any]:
    started = time.perf_counter()
    command = [sys.executable, "-u", "-m", "tools.historical_compare", "score",
               "--prediction", str(prediction), "--files-json", str(files_json),
               "--data-dir", str(data), "--metrics-json", str(output),
               "--jobs", str(jobs), "--scorer-root", str(scorer_root)]
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.update({"OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4",
                "MKL_NUM_THREADS": "4", "MPLCONFIGDIR": "/tmp/matplotlib-mapunet-gate"})
    subprocess.run(command, cwd=source_root, env=env, check=True)
    metrics = json.loads(output.read_text())
    return {"native": metrics["results"]["model"],
            "metrics_sha256": file_hash(output),
            "wall_seconds": time.perf_counter() - started}


def score_gate(campaign: Path, kind: str, jobs: int) -> None:
    """Score fixed source and guarded candidate once on sealed gate200."""
    started = time.perf_counter()
    campaign = campaign.resolve()
    meta = verify_campaign(campaign)
    if kind not in KINDS or not 1 <= jobs <= 24:
        raise ValueError(f"native gate kind/jobs {kind}/{jobs}, expected {KINDS}/1..24")
    if kind != "fold0":
        first = json.loads((campaign / "gate_result_fold0.json").read_text())
        if not first["passed"]:
            raise ValueError(f"{campaign}: fold0 gate failed; foreign source score forbidden")
    source_ready = json.loads((campaign / f"gate_source_{kind}_ready.json").read_text())
    guarded_ready = json.loads((campaign / f"gate_guarded_{kind}_ready.json").read_text())
    names = meta["gate_files"]
    if (source_ready["frames"] != 43142 or guarded_ready["frames"] != 43142
            or [entry["shot"] for entry in source_ready["entries"]] != names
            or [entry["shot"] for entry in guarded_ready["entries"]] != names):
        raise ValueError(f"{campaign}: gate {kind} source/guarded identities differ")
    folder = campaign / "native" / kind
    folder.mkdir(parents=True, exist_ok=True)
    source_npz = folder / "source.npz"
    guarded_npz = folder / "guarded.npz"
    if (source_npz.exists() or guarded_npz.exists()
            or (campaign / f"gate_result_{kind}.json").exists()):
        raise FileExistsError(f"{folder}: native gate outputs already exist")
    _aggregate([Path(entry["output_path"]) for entry in source_ready["entries"]],
               names, source_npz, True)
    _aggregate([Path(entry["output_path"]) for entry in guarded_ready["entries"]],
               names, guarded_npz, False)
    files_json = folder / "files.json"
    write_json(files_json, {"files": names, "config": "diii_d_train"})
    origin = Path(meta["origin_root"])
    scorer_root = origin / "runs/geometry-loss-screen-4a5516b9b1684353/source"
    scorer_meta = run.verify_source(scorer_root.parent)
    spec = json.loads((campaign / "source/experiments/map-residual-unet-review.json").read_text())
    if scorer_meta["source_hash"] != spec["frozen_references"]["native_scorer_bundle_sha256"]:
        raise ValueError(f"{scorer_root}: pinned native scorer bundle SHA differs")
    source_score = _native_score(source_npz, files_json, Path(meta["data_dir"]),
                                 folder / "source.metrics.json", scorer_root,
                                 campaign / "source", jobs)
    guarded_score = _native_score(guarded_npz, files_json, Path(meta["data_dir"]),
                                  folder / "guarded.metrics.json", scorer_root,
                                  campaign / "source", jobs)
    if kind == "fold0":
        prior = json.loads((origin / (
            "runs/b10-matched-heads-20260923/gate_predictions/source.metrics.json")).read_text())
        if source_score["native"] != prior["results"]["model"]:
            raise ValueError(f"{folder}: fold0 native source differs from pinned B10 source")
    source_native = source_score["native"]
    corrected_native = guarded_score["native"]
    failures = ("psi_nonfinite_frac", "lcfs_fail_frac", "cons_fail_frac")
    no_extra = all(corrected_native[key] <= source_native[key] for key in failures)
    no_extra &= all(corrected_native["cons_fail_each"][key] <= value
                    for key, value in source_native["cons_fail_each"].items())
    for i in range(len(names)):
        with (np.load(source_npz, allow_pickle=False) as source_pack,
              np.load(guarded_npz, allow_pickle=False) as guarded_pack):
            for field in ("q95", "betaN"):
                key = f"shot_{i:04d}_{field}"
                if source_pack[key].tobytes() != guarded_pack[key].tobytes():
                    raise ValueError(f"{names[i]}: {field} submitted scalar bits changed")
    delta = corrected_native["S"] - source_native["S"]
    threshold = 0.0015 if kind == "mean2" else 0.003
    passed = (delta >= threshold and no_extra
              and guarded_ready["fallback_fraction"] <= 0.02)
    write_json(campaign / f"gate_result_{kind}.json", {
        "classification": f"fixed_map_residual_unet_{kind}_native_selection_gate",
        "source_bundle_sha256": meta["source_bundle_sha256"],
        "scorer_source_bundle_sha256": scorer_meta["source_hash"],
        "source_prediction_sha256": file_hash(source_npz),
        "guarded_prediction_sha256": file_hash(guarded_npz),
        "files_sha256": file_hash(files_json),
        "shots": len(names), "frames": 43142,
        "source_native": source_score,
        "guarded_native": guarded_score,
        "delta_S": delta,
        "required_delta_S": threshold,
        "no_extra_native_failures": no_extra,
        "fallback_reasons": guarded_ready["fallback_reasons"],
        "fallback_frames": guarded_ready["fallback_frames"],
        "fallback_fraction": guarded_ready["fallback_fraction"],
        "submitted_scalar_arrays_bit_identical": 2 * len(names),
        "passed": passed,
        "elapsed_seconds": time.perf_counter() - started,
        "scope": "Reused gate200 selection evidence only; original tail70 unopened",
    })
    print(f"{kind} native source S {source_native['S']:.6f}, guarded "
          f"{corrected_native['S']:.6f}, delta {delta:+.6f}, gate {passed}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for action in ("source", "guard", "score"):
        p = sub.add_parser(action)
        p.add_argument("--campaign", type=Path, required=True)
        p.add_argument("--kind", choices=KINDS, required=True)
        p.add_argument("--jobs", type=int, default=24 if action != "guard" else 4)
    args = parser.parse_args()
    if args.command == "source":
        cache_source(args.campaign, args.kind, args.jobs)
    elif args.command == "guard":
        predict_guarded(args.campaign, args.kind, args.jobs)
    else:
        score_gate(args.campaign, args.kind, args.jobs)


if __name__ == "__main__":
    main()
