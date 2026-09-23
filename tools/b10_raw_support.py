"""Input-only full-window clock/support preflight for the bounded B10 pilot."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from experiments import D3D_MAGNETICS_SIGNALS, interpolate_magnetics_to_efit
from my_experiments import baseline_model
from my_experiments.data_check import check_dataset
from my_experiments.oof_split import names_hash
from my_experiments.run import source_files
from toolkit.parallel import pimap, release
from toolkit.records import file_hash, write_json

ROOT = Path(__file__).resolve().parent.parent
SPEC = ROOT / "experiments/b10-two-rate-information-gate-v3.json"
DATA = ROOT.parent / "downloaded_huggingface/hf_dataset"
OOF = ROOT / "runs/oof-normal-k8-5fold-20260923"
INPUT_COLUMNS = [
    "efit_times", "magnetics_time", "magnetics_plasma_current_times",
    *(f"magnetics_{signal}" for signal in D3D_MAGNETICS_SIGNALS),
]


def classify_clock(axis: np.ndarray, label: str, tolerance: float,
                   max_step_factor: float) -> tuple[float, float, float]:
    """Require one of the two measured DIII-D raw clock rates, in milliseconds."""
    times = np.asarray(axis, dtype=np.float64)
    if times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all():
        raise ValueError(f"{label}: time axis shape/finiteness {times.shape}/"
                         f"{int((~np.isfinite(times)).sum())}, expected >=2 finite samples")
    steps = np.diff(times)
    median = float(np.median(steps))
    if np.any(steps <= 0) or not np.isfinite(median):
        raise ValueError(f"{label}: nonpositive/nonfinite raw step; median {median} ms")
    options = (0.05, 0.5)
    nearest = min(options, key=lambda value: abs(median - value))
    if abs(median - nearest) > tolerance * nearest or float(np.max(steps)) > (
            max_step_factor * median):
        raise ValueError(f"{label}: median/max step {median}/{float(np.max(steps))} ms, "
                         f"expected 0.05 or 0.5 ms within {tolerance:.1%}, maximum "
                         f"<= {max_step_factor} x median")
    return nearest, median, float(np.max(steps))


def raw_bin_support(efit_times: np.ndarray, raw_times: np.ndarray,
                    centres_ms: np.ndarray, half_bin_ms: float) -> tuple[int, float, float]:
    """Count exact requested raw bins outside acquisition support; no padding."""
    centres = np.asarray(efit_times, dtype=np.float64)[:, None] + centres_ms[None, :]
    left = centres - half_bin_ms
    right = centres + half_bin_ms
    unsupported = (left < raw_times[0]) | (right > raw_times[-1])
    return int(np.sum(unsupported)), float(np.min(left)), float(np.max(right))


def _one(task: tuple[str, str, int, str, dict[str, float]]) -> dict[str, Any]:
    name, data_s, expected_size, expected_sha, numeric = task
    path = Path(data_s) / "data/diii_d_train" / name
    if path.stat().st_size != expected_size or file_hash(path) != expected_sha:
        raise ValueError(f"{path}: size/SHA differ from frozen dataset manifest")
    row = pd.read_parquet(path, columns=INPUT_COLUMNS).iloc[0]
    shot = baseline_model.inputs_only_shot(row)
    efit = np.asarray(shot["efit_times"], dtype=np.float64)
    if (efit.ndim != 1 or len(efit) < 1 or not np.isfinite(efit).all()
            or (len(efit) > 1 and np.any(np.diff(efit) <= 0))):
        raise ValueError(f"{path}: EFIT times length/order/finiteness "
                         f"{len(efit)}, expected positive monotone milliseconds")
    if len(efit) > 1:
        steps = np.diff(efit)
        multiples = steps / 20.0
        if not np.allclose(multiples, np.rint(multiples), rtol=0, atol=0.001):
            raise ValueError(f"{path}: EFIT step {float(np.max(steps))} ms, expected "
                             "positive multiples of 20 ms")
    shared = np.asarray(shot["magnetics"]["ECOILA"]["times"], dtype=np.float64)
    ip = np.asarray(shot["magnetics"]["plasma_current"]["times"], dtype=np.float64)
    rate, shared_step, shared_max = classify_clock(
        shared, f"{path}: magnetics_time", numeric["sample_rate_tolerance"],
        numeric["maximum_step_factor"])
    ip_rate, ip_step, ip_max = classify_clock(
        ip, f"{path}: magnetics_plasma_current_times aligned",
        numeric["sample_rate_tolerance"], numeric["maximum_step_factor"])
    if ip_rate != 0.5:
        raise ValueError(f"{path}: corrected Ip clock rate {ip_rate} ms, expected 0.5 ms")
    centres = np.arange(-10.0, 10.0 + 0.25, 0.5, dtype=np.float64)
    if len(centres) != 41 or centres[0] != -10.0 or centres[-1] != 10.0:
        raise ValueError(f"{path}: subframe centres {len(centres)} "
                         f"{centres[0]}/{centres[-1]}, expected 41 -10..10 ms")
    expected = interpolate_magnetics_to_efit(shot)
    if expected.shape != (len(efit), 21):
        raise ValueError(f"{path}: existing EFIT interpolation {expected.shape}, "
                         f"expected ({len(efit)},21)")
    failures: list[dict[str, Any]] = []
    for j, signal in enumerate(D3D_MAGNETICS_SIGNALS):
        values = np.asarray(shot["magnetics"][signal]["values"], dtype=np.float64)
        axis = np.asarray(shot["magnetics"][signal]["times"], dtype=np.float64)
        if (len(values) != len(axis) or not np.isfinite(values).all()
                or (signal != "plasma_current" and not np.array_equal(axis, shared))):
            raise ValueError(f"{path}: {signal} values/time length/finiteness "
                             f"{len(values)}/{len(axis)}/"
                             f"{int((~np.isfinite(values)).sum())}, expected equal finite "
                             "and shared axis for non-Ip")
        count, requested_min, requested_max = raw_bin_support(
            efit, axis, centres, numeric["half_bin_ms"])
        if count:
            failures.append({
                "signal": signal,
                "unsupported_bins": count,
                "requested_min_ms": requested_min,
                "requested_max_ms": requested_max,
                "raw_min_ms": float(axis[0]),
                "raw_max_ms": float(axis[-1]),
            })
            continue
        observed = np.interp(efit, axis, values).astype(np.float32)
        if not np.allclose(observed, expected[:, j], rtol=numeric["value_rtol"],
                           atol=numeric["value_atol"]):
            delta = float(np.max(np.abs(observed - expected[:, j])))
            raise ValueError(f"{path}: {signal} EFIT-centre interpolation maximum "
                             f"difference {delta}, expected rtol={numeric['value_rtol']} "
                             f"atol={numeric['value_atol']}")
    cadence = efit[:, None] + centres[None, :]
    padded = (cadence < efit[0]) | (cadence > efit[-1])
    return {
        "shot": name,
        "frames": len(efit),
        "shared_rate_class_ms": rate,
        "shared_median_step_ms": shared_step,
        "shared_max_step_ms": shared_max,
        "ip_median_step_ms": ip_step,
        "ip_max_step_ms": ip_max,
        "efit_edge_padded_window_bins_per_signal": int(np.sum(padded)),
        "raw_support_failures": failures,
    }


def prepare(run: Path) -> None:
    """Freeze input lists and source before reading any selected raw parquet."""
    run = run.resolve()
    if run.exists():
        raise FileExistsError(f"{run}: B10 support campaign already exists")
    spec = json.loads(SPEC.read_text())
    if file_hash(Path(__file__)) != spec["preaudit_helper_sha256"]:
        raise ValueError(f"{SPEC}: B10 support helper SHA differs")
    if file_hash(ROOT / spec["amends_spec"]) != spec["amends_spec_sha256"]:
        raise ValueError(f"{SPEC}: reviewed B10 protocol SHA differs")
    if file_hash(ROOT / spec["amends_spec_v2"]) != spec["amends_spec_v2_sha256"]:
        raise ValueError(f"{SPEC}: prior v2 B10 protocol SHA differs")
    files_path = OOF / "files.json"
    artifact_path = OOF / "fold0/base.joblib"
    if (file_hash(files_path) != spec["frozen_data_plan"]["OOF_files_manifest_sha256"]
            or file_hash(artifact_path) != spec["frozen_data_plan"]["fold0_artifact_sha256"]):
        raise ValueError(f"{OOF}: frozen OOF filenames/artifact SHA differs")
    files = json.loads(files_path.read_text())
    artifact = joblib.load(artifact_path)
    oof = files["folds"][0]["oof_files"]
    gate = files["gate_files"]
    if (len(oof) != 1355 or len(gate) != 200
            or names_hash(oof) != spec["frozen_data_plan"]["head_gradient_train_ordered_sha256"]
            or names_hash(gate) != spec["frozen_data_plan"]["sealed_native_gate_ordered_sha256"]
            or len(set(oof + gate + files["reserved_tail_files"])) != 1625
            or set(artifact["train_files"]) & set(oof + gate + files["reserved_tail_files"])
            or artifact["val_files"] != []
            or artifact["scaler"].n_features_in_ != 84):
        raise ValueError(f"{OOF}: OOF/gate/tail shot/width provenance differs")
    data_identity = check_dataset(DATA, "diii_d_train")
    run.mkdir(parents=True)
    source = run / "source"
    source.mkdir()
    for path in source_files():
        target = source / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    for path in (SPEC, ROOT / spec["amends_spec"], ROOT / spec["amends_spec_v2"]):
        target = source / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    hashes = {str(path.relative_to(ROOT)): file_hash(source / path.relative_to(ROOT))
              for path in source_files()}
    hashes[str(SPEC.relative_to(ROOT))] = file_hash(source / SPEC.relative_to(ROOT))
    for path in (ROOT / spec["amends_spec"], ROOT / spec["amends_spec_v2"]):
        hashes[str(path.relative_to(ROOT))] = file_hash(source / path.relative_to(ROOT))
    write_json(run / "manifest.json", {
        "classification": "B10_input_only_support_audit_frozen_before_raw_read",
        "origin_root": str(ROOT),
        "data_dir": str(DATA),
        "source_sha256": hashes,
        "spec_sha256": file_hash(SPEC),
        "OOF_artifact_sha256": file_hash(artifact_path),
        "OOF_files_manifest_sha256": file_hash(files_path),
        "dataset_identity": data_identity,
        "head_train_files": oof,
        "gate_files": gate,
        "reserved_tail_files": files["reserved_tail_files"],
    })
    subprocess.run([
        sys.executable, "-c",
        "import pathlib,tools.b10_raw_support as p; "
        "assert pathlib.Path(p.__file__).resolve()=="
        "pathlib.Path('tools/b10_raw_support.py').resolve()",
    ], cwd=source, env={**os.environ, "PYTHONPATH": str(source)}, check=True)
    print(f"Frozen B10 raw support {run}: {len(oof)} head-train + {len(gate)} gate shots")


def audit(run: Path, jobs: int) -> int:
    """Read input-only clocks/values and record complete support or a frozen stop."""
    started = time.perf_counter()
    run = run.resolve()
    if ROOT.resolve() != (run / "source").resolve():
        raise ValueError(f"B10 preaudit imported from {ROOT}, expected {run / 'source'}")
    manifest = json.loads((run / "manifest.json").read_text())
    spec = json.loads((run / "source/experiments/b10-two-rate-information-gate-v3.json")
                      .read_text())
    for relative, expected in manifest["source_sha256"].items():
        path = run / "source" / relative
        if file_hash(path) != expected:
            raise ValueError(f"{path}: frozen source SHA differs")
    if check_dataset(Path(manifest["data_dir"]), "diii_d_train") != manifest["dataset_identity"]:
        raise ValueError(f"{run}: DIII-D dataset identity differs from frozen preflight")
    output = run / "support_audit.json"
    if output.exists():
        raise FileExistsError(f"{output}: support audit already recorded")
    names = manifest["head_train_files"] + manifest["gate_files"]
    dataset = json.loads((run / "source/my_experiments/diii_d_dataset.json").read_text())
    numeric = spec["numeric_preflight"]
    tasks = [(name, manifest["data_dir"], *dataset["configs"]["diii_d_train"][name],
              numeric) for name in names]
    rows: list[dict[str, Any]] = []
    try:
        for i, item in enumerate(pimap(_one, tasks, jobs)):
            if item["shot"] != names[i]:
                raise ValueError(f"{names[i]}: raw support task order differs")
            rows.append(item)
            if (i + 1) % 100 == 0:
                print(f"B10 input-only support {i + 1}/{len(names)} shots", flush=True)
    finally:
        release()
    train_rows = rows[:1355]
    gate_rows = rows[1355:]
    if (len(train_rows) != 1355 or len(gate_rows) != 200
            or sum(row["frames"] for row in train_rows) != 300793
            or sum(row["frames"] for row in gate_rows) != 43142):
        raise ValueError(f"{run}: train/gate input frames "
                         f"{sum(row['frames'] for row in train_rows)}/"
                         f"{sum(row['frames'] for row in gate_rows)}, "
                         "expected 300793/43142")
    failures = [{"shot": row["shot"], **issue} for row in rows
                for issue in row["raw_support_failures"]]
    classes: dict[str, dict[str, Any]] = {}
    for label, selected in (("0.05", 0.05), ("0.5", 0.5)):
        class_rows = [row for row in gate_rows if row["shared_rate_class_ms"] == selected]
        class_names = [row["shot"] for row in class_rows]
        classes[label] = {"gate_shots": len(class_names),
                          "gate_frames": sum(row["frames"] for row in class_rows),
                          "gate_files": class_names,
                          "gate_ordered_sha256": names_hash(class_names)}
    if sum(v["gate_shots"] for v in classes.values()) != 200:
        raise ValueError(f"{run}: sampling-rate classes do not cover all gate200 shots")
    write_json(output, {
        "classification": "B10_full_1555_input_only_raw_support_preflight",
        "manifest_sha256": file_hash(run / "manifest.json"),
        "spec_sha256": manifest["spec_sha256"],
        "train_shots": len(train_rows),
        "train_frames": sum(row["frames"] for row in train_rows),
        "gate_shots": len(gate_rows),
        "gate_frames": sum(row["frames"] for row in gate_rows),
        "reserved_tail70_accessed": False,
        "target_columns_requested": [],
        "source_parquet_sha256_verified_shots": len(rows),
        "sampling_rate_classes": classes,
        "raw_unsupported_bin_failures": failures,
        "raw_support_complete": not failures,
        "efit_edge_padded_window_bins_per_signal": sum(
            row["efit_edge_padded_window_bins_per_signal"] for row in rows),
        "per_shot": rows,
        "wall_seconds": time.perf_counter() - started,
        "decision": ("ready_for_raw_sidecar_and_head_fits" if not failures else
                     "STOP_before_targets_or_fits_due_to_raw_support"),
    })
    print(f"B10 raw support audit: {len(rows)} shots, {len(failures)} unsupported "
          f"shot/signal bins, classes "
          f"{[(key, value['gate_shots']) for key, value in classes.items()]}")
    return 0 if not failures else 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "audit"):
        part = sub.add_parser(command)
        part.add_argument("--run", type=Path, required=True)
        part.add_argument("--jobs", type=int, default=24)
    args = parser.parse_args()
    if not 1 <= args.jobs <= 24:
        raise ValueError(f"jobs={args.jobs}, expected 1..24")
    if args.command == "prepare":
        prepare(args.run)
        return 0
    return audit(args.run, args.jobs)


if __name__ == "__main__":
    raise SystemExit(main())
