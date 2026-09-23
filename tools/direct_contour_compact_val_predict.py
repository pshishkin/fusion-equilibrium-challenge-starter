"""Isolated compact-source prediction on its fixed, seen validation200 selection window.

This script must be launched as a separate process with cwd and PYTHONPATH set to
the compact campaign's frozen source. Its output is for a selection-only oracle,
never an independent validation claim.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from my_experiments import baseline_model
from toolkit.parallel import pimap, release
from toolkit.records import file_hash, write_json

INPUT_COLUMNS = list(dict.fromkeys((
    *baseline_model.GEOMETRY_COLUMNS,
    "magnetics_time", "magnetics_plasma_current_times",
    *(f"magnetics_{signal}" for signal in baseline_model.D3D_MAGNETICS_SIGNALS),
    "thomson_chord_name", "thomson_chord_R", "thomson_chord_Z",
    "thomson_core_Te", "thomson_core_ne", "thomson_core_times",
    "thomson_edge_Te", "thomson_edge_ne", "thomson_edge_times",
)))


def _save(path: Path, arrays: dict[str, np.ndarray]) -> None:
    if path.exists():
        raise FileExistsError(f"{path}: compact predictions are already sealed")
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".npz")
    try:
        with os.fdopen(fd, "wb") as stream:
            np.savez_compressed(stream, **arrays)  # type: ignore[arg-type]
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _one(task: tuple[str, str, str, int]) -> dict[str, np.ndarray]:
    name, data_s, artifact_s, frames = task
    path = Path(data_s) / "data/diii_d_train" / name
    row = pd.read_parquet(path, columns=INPUT_COLUMNS).iloc[0]
    if any(key.startswith("efit_") and key not in (
            "efit_times", "efit_grid_R", "efit_grid_Z") for key in row.index):
        raise ValueError(f"{path}: compact validation input row contains an EFIT target")
    slim = baseline_model.slim_row(row)
    out = baseline_model.predict_row(slim, "DIII-D", "ensemble", Path(artifact_s))
    expected = {"psirz": (frames, 65, 65), "q95": (frames,), "betaN": (frames,)}
    result: dict[str, np.ndarray] = {}
    for field, shape in expected.items():
        value = np.asarray(out[field])
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f"{path}: {field} predicted shape/nonfinite count "
                             f"{value.shape}/{int((~np.isfinite(value)).sum())}, "
                             f"expected {shape}/0")
        result[field] = value.astype(np.float32)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.jobs <= 8:
        raise ValueError(f"jobs={args.jobs}, expected 1..8 for compact CPU inference")
    started = time.perf_counter()
    run = args.run.resolve()
    manifest = json.loads((run / "manifest.json").read_text())
    source = Path(manifest["compact_source"])
    loaded = Path(baseline_model.__file__).resolve()
    if loaded != source / "my_experiments/baseline_model.py":
        raise ValueError(f"baseline_model imported {loaded}, expected {source}")
    for relative, digest in manifest["compact_predictor_source_sha256"].items():
        path = source / relative
        if file_hash(path) != digest:
            raise ValueError(f"{path}: compact predictor source SHA differs")
    artifact = Path(manifest["compact_artifact"])
    if file_hash(artifact) != manifest["compact_artifact_sha256"]:
        raise ValueError(f"{artifact}: compact artifact SHA differs")
    packed = joblib.load(artifact)
    names = manifest["validation200_files"]
    if (list(packed["val_files"][:200]) != names
            or set(names) & set(packed["train_files"])):
        raise ValueError(f"{artifact}: validation200 provenance changed or overlaps training")
    if len(names) != 200 or len(manifest["validation200_frames"]) != 200:
        raise ValueError(f"{run}: expected 200 ordered validation shot/frame identities")
    tasks = [(name, manifest["data_dir"], str(artifact), manifest["validation200_frames"][i])
             for i, name in enumerate(names)]
    arrays: dict[str, np.ndarray] = {}
    try:
        for i, predicted in enumerate(pimap(_one, tasks, args.jobs)):
            for field in ("psirz", "q95", "betaN"):
                arrays[f"shot_{i:04d}_{field}"] = predicted[field]
            if (i + 1) % 25 == 0:
                print(f"compact validation predictions: {i + 1}/200 shots", flush=True)
    finally:
        release()
    if len(arrays) != 600:
        raise ValueError(f"{run}: {len(arrays)} prediction arrays, expected 600")
    output = run / "compact_validation200.npz"
    _save(output, arrays)
    write_json(output.with_suffix(".json"), {
        "classification": "selection_only_compact_validation_predictions",
        "prediction_sha256": file_hash(output),
        "manifest_sha256": file_hash(run / "manifest.json"),
        "compact_artifact_sha256": manifest["compact_artifact_sha256"],
        "compact_source_sha256": manifest["compact_predictor_source_sha256"],
        "files": names,
        "shots": 200,
        "frames": sum(manifest["validation200_frames"]),
        "all_arrays_float32": True,
        "input_target_columns_requested": [],
        "source_artifact_saw_these_files_for_early_stopping": True,
        "wall_seconds": time.perf_counter() - started,
    })
    print(f"Sealed compact validation200 source {output}, "
          f"{sum(manifest['validation200_frames'])} frames")


if __name__ == "__main__":
    main()
