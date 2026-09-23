"""Freeze and run the training-only fold-0 K8 normal-displacement pilot.

Preparation reads input filenames/times only. The later pilot uses true LCFS only on
the 100 predeclared original-TRAIN shots; it never opens gate200 or tail70.
"""

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

from experiments import _as_psirz_stack
from fusion_scoring import common, contour, derive, lcfs
from my_experiments import baseline_model
from my_experiments.data_check import check_dataset
from my_experiments.oof_split import build_manifest
from my_experiments.run import source_files
from toolkit.parallel import pimap, release
from toolkit.records import file_hash, write_json
from tools.contour_learnability import (
    canonical_contour,
    correct_map_from_displacement,
    decode_normal_coefficients,
    encode_normal_target,
    outward_normals,
)
from tools.geometry_source_li_compensation import _compensate_frame

ROOT = Path(__file__).resolve().parent.parent
SPEC = ROOT / "experiments/oof-normal-contour-mlp-review.json"
COMPENSATION = ROOT / "experiments/geometry-additive-source-li-compensation-v3.json"
REFERENCE = ROOT / "runs/compact-production-9301af36275620f7/base_s0/baseline.joblib"
DATA = ROOT.parent / "downloaded_huggingface/hf_dataset"
SCORER = ROOT / "runs/geometry-loss-screen-4a5516b9b1684353/source"


def _count_input_frames(task: tuple[str, str]) -> tuple[str, int]:
    name, data_s = task
    path = Path(data_s) / "data/diii_d_train" / name
    row = pd.read_parquet(path, columns=["efit_times"]).iloc[0]
    times = np.asarray(row["efit_times"], dtype=np.float64)
    if (times.ndim != 1 or len(times) < 1 or not np.isfinite(times).all()
            or (len(times) > 1 and np.any(np.diff(times) <= 0))):
        raise ValueError(f"{path}: input efit_times length/order/finiteness "
                         f"{len(times)}, expected positive monotone finite")
    return name, len(times)


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    if path.exists():
        raise FileExistsError(f"{path}: sealed prediction already exists")
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".npz")
    try:
        with os.fdopen(fd, "wb") as stream:
            np.savez_compressed(stream, **arrays)  # type: ignore[arg-type]
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def prepare(run: Path, jobs: int) -> None:
    """Freeze code/config/files before any pilot base fit or true-target read."""
    started = time.perf_counter()
    run = run.resolve()
    if run.exists():
        raise FileExistsError(f"{run}: campaign already exists")
    if run / "source" == ROOT:
        raise ValueError(f"{run}: cannot prepare inside its own source snapshot")
    reference: dict[str, Any] = joblib.load(REFERENCE)
    params = yaml.safe_load(reference["params_yaml"])
    for name, model in params["models"].items():
        model["enabled"] = name == "mlp"
    params["ensemble"]["members"] = {"mlp": 1.0}
    params["split"]["salt"] = 0
    params["split"]["holdout_share"] = 0.01
    params["models"]["mlp"]["max_steps"] = 33000
    params["models"]["mlp"]["patience_steps"] = 0
    if params["models"]["mlp"]["device"] != "cuda":
        raise ValueError(f"{REFERENCE}: MLP device is "
                         f"{params['models']['mlp']['device']}, expected cuda")
    dataset_identity = check_dataset(DATA, "diii_d_train")
    ordered = baseline_model.sorted_shots(DATA, "diii_d_train", 0)
    source = run / "source"
    source.mkdir(parents=True)
    for path in source_files():
        target = source / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    for path in (SPEC, COMPENSATION):
        target = source / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    effective = source / "params.yaml"
    effective.write_text(yaml.safe_dump(params, sort_keys=False))
    manifest = build_manifest(ordered, REFERENCE, source / SPEC.relative_to(ROOT), effective)
    write_json(run / "files.json", manifest)
    pilot = manifest["pilot_train_oof_files"]
    try:
        counted = list(pimap(_count_input_frames, [(name, str(DATA)) for name in pilot], jobs))
    finally:
        release()
    if [name for name, _ in counted] != pilot:
        raise ValueError(f"{run}: pilot input frame order differs from manifest")
    hashes = {
        str(path.relative_to(ROOT)): file_hash(source / path.relative_to(ROOT))
        for path in source_files()
    }
    hashes[str(SPEC.relative_to(ROOT))] = file_hash(source / SPEC.relative_to(ROOT))
    hashes[str(COMPENSATION.relative_to(ROOT))] = file_hash(
        source / COMPENSATION.relative_to(ROOT))
    hashes["params.yaml"] = file_hash(effective)
    source_bundle_sha = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    (run / "fold0").mkdir()
    write_json(run / "campaign.json", {
        "classification": "frozen_before_fivefold_oof_pilot_fit",
        "origin_root": str(ROOT),
        "data_dir": str(DATA),
        "scorer_root": str(SCORER),
        "scorer_source_sha256": file_hash(SCORER / "local_score.py"),
        "review_spec_sha256": file_hash(SPEC),
        "reference_artifact_sha256": file_hash(REFERENCE),
        "files_manifest_sha256": file_hash(run / "files.json"),
        "source_sha256": hashes,
        "source_bundle_sha256": source_bundle_sha,
        "dataset_identity": dataset_identity,
        "pilot_files": pilot,
        "pilot_input_frame_counts": [n for _, n in counted],
        "pilot_full_frames": sum(n for _, n in counted),
        "pilot_target_columns_read_during_prepare": [],
        "reserved_gate_shots": len(manifest["gate_files"]),
        "reserved_tail_shots": len(manifest["reserved_tail_files"]),
        "fold0_fit_command_not_executed": [
            sys.executable, "my_experiments/train.py", "--share", "0.80/1.0",
            "--val-share", "0/1.0", "--fit-mode", "fixed_steps",
            "--params", str(effective), "--jobs", "24",
            "--local-data-dir", str(DATA), "--explicit-files-manifest",
            str(run / "files.json"), "--explicit-fold", "0",
        ],
        "fold0_fit_environment_hashes_not_executed": {
            "FUSION_ARTIFACT": str(run / "fold0/base.joblib"),
            "FUSION_OOF_MANIFEST_SHA256": file_hash(run / "files.json"),
            "FUSION_SOURCE_SHA256": source_bundle_sha,
            "FUSION_RUN_ID": run.name,
            "FUSION_AUX_CACHE": str(ROOT / ".aux_targets"),
        },
        "prepare_wall_seconds": time.perf_counter() - started,
    })
    check = subprocess.run(
        [sys.executable, "-c",
         "import pathlib,my_experiments.baseline_model,my_experiments.oof_split; "
         "p=pathlib.Path(my_experiments.baseline_model.__file__).resolve(); "
         "assert p==pathlib.Path('my_experiments/baseline_model.py').resolve(); print(p)"],
        cwd=source, env={**os.environ, "PYTHONPATH": str(source)},
        capture_output=True, text=True, check=True,
    )
    print(f"Frozen OOF campaign {run}: {len(pilot)} pilot TRAIN shots, "
          f"{sum(n for _, n in counted)} input frames; import {check.stdout.strip()}")


def _checked_context(run: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    source = run / "source"
    if ROOT.resolve() != source.resolve():
        raise ValueError(f"pilot imported from {ROOT}, expected frozen {source}")
    campaign = json.loads((run / "campaign.json").read_text())
    for relative, expected in campaign["source_sha256"].items():
        path = source / relative
        if file_hash(path) != expected:
            raise ValueError(f"{path}: SHA {file_hash(path)}, expected {expected}")
    files_path = run / "files.json"
    if file_hash(files_path) != campaign["files_manifest_sha256"]:
        raise ValueError(f"{files_path}: frozen filename manifest SHA differs")
    manifest = json.loads(files_path.read_text())
    if (manifest["pilot_train_oof_files"] != campaign["pilot_files"]
            or len(campaign["pilot_input_frame_counts"]) != 100
            or sum(campaign["pilot_input_frame_counts"]) != campaign["pilot_full_frames"]):
        raise ValueError(f"{run}: pilot filenames/frame counts differ from frozen campaign")
    return campaign, manifest


def _pilot_shot(task: tuple[str, int, str, str, str]) -> dict[str, Any]:
    name, expected_frames, data_s, artifact_s, source_s = task
    path = Path(data_s) / "data/diii_d_train" / name
    pinned = json.loads((Path(source_s) / "my_experiments/diii_d_dataset.json").read_text())
    expected_size, expected_sha = pinned["configs"]["diii_d_train"][name]
    if path.stat().st_size != expected_size or file_hash(path) != expected_sha:
        raise ValueError(f"{path}: size/SHA differ from pinned dataset manifest")
    row = pd.read_parquet(path).iloc[0]
    slim = baseline_model.slim_row(row)
    forbidden = [key for key in slim if key.startswith("efit_")
                 and key not in ("efit_times", "efit_grid_R", "efit_grid_Z")]
    if forbidden:
        raise ValueError(f"{path}: pilot inference row contains target fields {forbidden}")
    predicted = baseline_model.predict_row(slim, "DIII-D", "mlp", Path(artifact_s))
    source_psi = np.asarray(predicted["psirz"], dtype=np.float32)
    truth_psi = _as_psirz_stack(row["efit_psirz"]).astype(np.float64)
    if (source_psi.shape != (expected_frames, 65, 65)
            or truth_psi.shape != source_psi.shape
            or not np.isfinite(source_psi).all() or not np.isfinite(truth_psi).all()):
        raise ValueError(f"{path}: source/truth frames {source_psi.shape}/{truth_psi.shape}, "
                         f"expected ({expected_frames},65,65) finite")
    R = np.asarray(row["efit_grid_R"], dtype=np.float64)
    Z = np.asarray(row["efit_grid_Z"], dtype=np.float64)
    with np.load(Path(source_s) / "fusion_scoring/masks/d3d_envelope.npz") as saved:
        mask = saved["mask_coarse"].astype(bool)
        if not np.array_equal(R, saved["grid_R"]) or not np.array_equal(Z, saved["grid_Z"]):
            raise ValueError(f"{path}: grid {R.shape}/{Z.shape} differs from pinned scorer")
    sign = int(common.AXIS_SIGN["DIII-D"])
    a_grid = json.loads((Path(source_s) /
                         "experiments/geometry-additive-source-li-compensation-v3.json")
                        .read_text())["family"]["candidate_a_grid"]
    from my_experiments.baseline_model import thin_frames
    selected = set(int(i) for i in thin_frames(expected_frames, 0.2))
    energy_num = 0.0
    energy_den = 0.0
    rms: list[float] = []
    oracle: list[np.ndarray] = []
    for frame, (psi, truth) in enumerate(zip(source_psi, truth_psi, strict=True)):
        c_source, _ = lcfs.extract_lcfs_with_sign(
            psi, R, Z, "DIII-D", mask, mask.astype(float), axis_sign=sign)
        c_true = lcfs.extract_lcfs(truth, R, Z, "DIII-D", mask, mask.astype(float))
        if c_source is None or c_true is None:
            raise ValueError(f"{path} frame {frame}: source/true LCFS missing "
                             f"{c_source is None}/{c_true is None}, expected both present")
        if (np.any(c_source[:, 0] < R[0]) or np.any(c_source[:, 0] > R[-1])
                or np.any(c_source[:, 1] < Z[0]) or np.any(c_source[:, 1] > Z[-1])
                or np.any(c_true[:, 0] < R[0]) or np.any(c_true[:, 0] > R[-1])
                or np.any(c_true[:, 1] < Z[0]) or np.any(c_true[:, 1] > Z[-1])):
            raise ValueError(f"{path} frame {frame}: source/true LCFS leaves scorer grid")
        try:
            coefficients, displacement = encode_normal_target(c_source, c_true)
            movement = decode_normal_coefficients(coefficients, c_source)
            if frame in selected:
                canonical, start = canonical_contour(c_source)
                normal = outward_normals(canonical)
                lowpass = np.sum(np.roll(movement, -start, axis=0) * normal, axis=1)
                energy_num += float(np.sum(lowpass**2))
                energy_den += float(np.sum(displacement**2))
                rms.append(float(np.sqrt(np.mean((displacement - lowpass) ** 2))))
            axis_r, axis_z, _, _ = derive.magnetic_axis(
                psi, R, Z, "DIII-D", mask, sign)
            additive, _ = correct_map_from_displacement(
                psi, c_source, movement, (axis_r, axis_z), R, Z)
            profile, _ = _compensate_frame(
                psi, additive, R, Z, mask, sign, a_grid, derive, lcfs, contour)
        except ValueError as exc:
            raise ValueError(f"{path} frame {frame}: {exc}") from exc
        oracle.append(profile)
    q95 = np.asarray(predicted["q95"], dtype=np.float32)
    beta = np.asarray(predicted["betaN"], dtype=np.float32)
    if (q95.shape != (expected_frames,) or beta.shape != (expected_frames,)
            or not np.isfinite(q95).all() or not np.isfinite(beta).all()):
        raise ValueError(f"{path}: q95/betaN shape/finiteness {q95.shape}/{beta.shape}, "
                         f"expected ({expected_frames},)/({expected_frames},) finite")
    return {
        "shot": name,
        "frames": expected_frames,
        "source_psi": source_psi,
        "oracle_psi": np.stack(oracle).astype(np.float32),
        "q95": q95,
        "betaN": beta,
        "energy_numerator": energy_num,
        "energy_denominator": energy_den,
        "rms_m": rms,
        "minimum_grid_spacing_m": float(min(np.diff(R).min(), np.diff(Z).min())),
    }


def verify_pilot_artifact(artifact: dict[str, Any], campaign: dict[str, Any],
                          manifest: dict[str, Any], path: Path) -> None:
    """Reject a base artifact whose own pilot, gate or tail affected its fit."""
    pilot_names = manifest["pilot_train_oof_files"]
    oof_names = manifest["folds"][0]["oof_files"]
    oof_set = set(oof_names)
    expected_fit = [name for name in manifest["allowed_files"] if name not in oof_set]
    fit_set = set(artifact["train_files"])
    if (artifact.get("oof_manifest_sha256") != campaign["files_manifest_sha256"]
            or artifact.get("source_sha256") != campaign["source_bundle_sha256"]
            or artifact.get("oof_fold") != 0
            or artifact["train_files"] != expected_fit
            or artifact.get("oof_files") != oof_names
            or artifact["val_files"] != []
            or fit_set & set(pilot_names)
            or fit_set & set(manifest["gate_files"])
            or fit_set & set(manifest["reserved_tail_files"])):
        raise ValueError(f"{path}: fold0 artifact overlap/provenance differs")


def pilot(run: Path, jobs: int) -> None:
    """Measure only the fixed train-shot K8 representation and truth-assisted gate."""
    started = time.perf_counter()
    run = run.resolve()
    campaign, manifest = _checked_context(run)
    if (run / "pilot_result.json").exists():
        raise FileExistsError(f"{run / 'pilot_result.json'}: pilot already completed")
    artifact_path = run / "fold0/base.joblib"
    if not artifact_path.exists():
        raise FileNotFoundError(f"{artifact_path}: fold0 base fit is required")
    artifact: dict[str, Any] = joblib.load(artifact_path)
    data = Path(campaign["data_dir"])
    scorer = Path(campaign["scorer_root"])
    if file_hash(scorer / "local_score.py") != campaign["scorer_source_sha256"]:
        raise ValueError(f"{scorer}: native scorer source SHA differs from campaign")
    verify_pilot_artifact(artifact, campaign, manifest, artifact_path)
    pilot_names = manifest["pilot_train_oof_files"]
    tasks = [
        (name, campaign["pilot_input_frame_counts"][i], str(data), str(artifact_path),
         str(run / "source"))
        for i, name in enumerate(pilot_names)
    ]
    source_arrays: dict[str, np.ndarray] = {}
    oracle_arrays: dict[str, np.ndarray] = {}
    energy_num = 0.0
    energy_den = 0.0
    rms: list[float] = []
    spacings: list[float] = []
    try:
        for i, item in enumerate(pimap(_pilot_shot, tasks, jobs)):
            if item["shot"] != pilot_names[i] or item["frames"] != tasks[i][1]:
                raise ValueError(f"{pilot_names[i]}: pilot result name/frames differ")
            for field in ("psirz", "q95", "betaN"):
                key = f"shot_{i:04d}_{field}"
                value = item["source_psi"] if field == "psirz" else item[field]
                source_arrays[key] = value
                oracle_arrays[key] = item["oracle_psi"] if field == "psirz" else value
            energy_num += item["energy_numerator"]
            energy_den += item["energy_denominator"]
            rms.extend(item["rms_m"])
            spacings.append(item["minimum_grid_spacing_m"])
            if (i + 1) % 10 == 0:
                print(f"OOF pilot built {i + 1}/100 TRAIN shots", flush=True)
    finally:
        release()
    if energy_den <= 0 or len(rms) == 0:
        raise ValueError(f"{run}: K8 energy {energy_den}, RMS frames {len(rms)}, "
                         "expected positive")
    retention = energy_num / energy_den
    median_rms = float(np.median(rms))
    spacing = min(spacings)
    representation_pass = bool(retention >= 0.8 and median_rms <= spacing)
    pilot_dir = run / "pilot"
    pilot_dir.mkdir(exist_ok=False)
    write_json(pilot_dir / "files.json", {"files": pilot_names})
    _atomic_npz(pilot_dir / "source.npz", source_arrays)
    _atomic_npz(pilot_dir / "oracle.npz", oracle_arrays)
    for label in ("source", "oracle"):
        write_json(pilot_dir / f"{label}.json", {
            "files": pilot_names,
            "prediction_sha256": file_hash(pilot_dir / f"{label}.npz"),
            "source_artifact_sha256": file_hash(artifact_path),
            "submitted_scalars_float32_exact": True,
        })
    build_seconds = time.perf_counter() - started
    scores: dict[str, dict[str, Any]] = {}
    score_seconds: dict[str, float] = {}
    for label in ("source", "oracle"):
        score_started = time.perf_counter()
        command = [
            sys.executable, "-m", "tools.historical_compare", "score",
            "--prediction", str(pilot_dir / f"{label}.npz"),
            "--files-json", str(pilot_dir / "files.json"),
            "--data-dir", str(data),
            "--metrics-json", str(pilot_dir / f"{label}.metrics.json"),
            "--jobs", str(jobs), "--scorer-root", str(scorer),
        ]
        subprocess.run(command, cwd=run / "source",
                       env={**os.environ, "PYTHONPATH": str(run / "source")}, check=True)
        scored = json.loads((pilot_dir / f"{label}.metrics.json").read_text())
        if scored["files"] != pilot_names:
            raise ValueError(f"{label}: native scorer files differ from frozen pilot")
        scores[label] = scored["results"]
        score_seconds[label] = time.perf_counter() - score_started
    control = scores["source"]["model"]
    treated = scores["oracle"]["model"]
    if (control["lcfs_fail_frac"] != 0 or treated["lcfs_fail_frac"] != 0
            or control["cons_fail_frac"] != 0 or treated["cons_fail_frac"] != 0):
        native_valid = False
    else:
        native_valid = True
    delta = treated["S"] - control["S"]
    write_json(run / "pilot_result.json", {
        "classification": "fold0_original_train_K8_truth_assisted_pilot",
        "campaign_sha256": file_hash(run / "campaign.json"),
        "files_manifest_sha256": campaign["files_manifest_sha256"],
        "base_artifact_sha256": file_hash(artifact_path),
        "pilot_files": pilot_names,
        "pilot_full_frames": campaign["pilot_full_frames"],
        "retained_normal_energy": retention,
        "median_rms_m": median_rms,
        "minimum_grid_spacing_m": spacing,
        "representation_passed": representation_pass,
        "native_source": control,
        "native_truth_assisted_K8_additive_source_li": treated,
        "native_delta_S": delta,
        "native_margin_threshold": 0.0026,
        "native_valid": native_valid,
        "pilot_passed": bool(representation_pass and native_valid and delta >= 0.0026),
        "build_wall_seconds": build_seconds,
        "native_scorer_wall_seconds": score_seconds,
        "total_wall_seconds": time.perf_counter() - started,
        "reserved_gate_and_tail_labels_read": False,
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "pilot"):
        part = sub.add_parser(name)
        part.add_argument("--run", type=Path, required=True)
        part.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.jobs <= 24:
        raise ValueError(f"jobs={args.jobs}, expected 1..24")
    if args.command == "prepare":
        prepare(args.run, args.jobs)
    else:
        pilot(args.run, args.jobs)


if __name__ == "__main__":
    main()
