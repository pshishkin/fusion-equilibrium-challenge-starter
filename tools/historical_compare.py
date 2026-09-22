"""Score archived DIII-D artifacts through their own inference code and today's scorer.

Predict runs in a separate process for each immutable source tree. It writes the existing
submission NPZ schema, ordered by a scorer-native JSON file's exact shot list. Blend averages
decoded maps and submitted scalars with explicitly supplied member counts. Score invokes
``local_score.main``; this tool contains no competition metric implementation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

from toolkit.parallel import pimap, release
from toolkit.progress import SHOT_EVERY, bar_kwargs, install_timestamps
from toolkit.records import file_hash, write_json

ROOT = Path(__file__).resolve().parent.parent
FIELDS = ("psirz", "q95", "betaN")


def _files(path: Path) -> list[str]:
    record = json.loads(path.read_text())
    names = record["files"]
    if not isinstance(names, list) or not names or len(set(names)) != len(names):
        raise ValueError(f"{path}: expected a nonempty ordered list of distinct files")
    if any(not isinstance(name, str) or Path(name).name != name for name in names):
        raise ValueError(f"{path}: file names must be parquet basenames")
    return names


def _keys(n: int) -> list[str]:
    return [f"shot_{i:04d}_{field}" for i in range(n) for field in FIELDS]


def _predict_task(task: tuple[Path, Path, Path, str]) -> dict[str, np.ndarray]:
    path, artifact, source_root, model = task
    sys.path.insert(0, str(source_root))
    from my_experiments import baseline_model

    loaded = Path(baseline_model.__file__).resolve()
    if not loaded.is_relative_to(source_root):
        raise ValueError(f"{path}: predictor loaded from {loaded}, expected {source_root}")
    row = pd.read_parquet(path).iloc[0]
    out = baseline_model.predict_row(row, "DIII-D", model, artifact)
    n = len(np.asarray(row["efit_times"]))
    expected = {"psirz": (n, 65, 65), "q95": (n,), "betaN": (n,)}
    result: dict[str, np.ndarray] = {}
    for field, shape in expected.items():
        value = np.asarray(out[field])
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f"{path}: predicted {field} shape {value.shape}, expected {shape}; "
                             f"nonfinite count {int((~np.isfinite(value)).sum())}")
        result[field] = value.astype(np.float32)
    return result


def _save(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"{path}: prediction file already exists")
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".npz")
    try:
        with os.fdopen(fd, "wb") as stream:
            np.savez_compressed(stream, **arrays)  # type: ignore[arg-type]
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _meta(path: Path) -> dict[str, Any]:
    return json.loads(path.with_suffix(".json").read_text())


def predict(source_root: Path, artifact: Path, files_json: Path, data_dir: Path,
            model: str, output: Path, jobs: int, artifact_sha256: str,
            input_config: str | None = None) -> None:
    started = time.perf_counter()
    source_root, artifact, data_dir = (p.resolve() for p in (source_root, artifact, data_dir))
    if jobs < 1:
        raise ValueError(f"jobs is {jobs}, expected at least 1")
    actual_hash = file_hash(artifact)
    if actual_hash != artifact_sha256:
        raise ValueError(f"{artifact}: SHA-256 {actual_hash}, expected {artifact_sha256}")
    names = _files(files_json)
    file_manifest = json.loads(files_json.read_text())
    source_files = [source_root / name for name in (
        "my_experiments/baseline_model.py", "my_experiments/models.py",
        "my_experiments/seq_model.py", "experiments.py",
    )]
    for path in source_files:
        if not path.is_file():
            raise FileNotFoundError(f"{path}: predictor source is missing")
    art = joblib.load(artifact)
    config = input_config if input_config is not None else art["config"]
    if file_manifest.get("config") is not None and file_manifest["config"] != config:
        raise ValueError(f"{files_json}: config {file_manifest['config']!r}, expected {config!r}")
    if config != art["config"] and not file_manifest.get("dataset_revision"):
        raise ValueError(f"{files_json}: cross-config inference needs a pinned dataset_revision")
    available = set(art["models"]) | {"ensemble"}
    parts = model.split("+")
    if model not in available and (len(parts) < 2 or set(parts) - set(art["models"])):
        raise ValueError(f"{artifact}: model {model!r}, available {sorted(available)}")
    seen = set(art["train_files"]) | set(art["val_files"])
    overlap = seen & set(names)
    if overlap:
        raise ValueError(f"{artifact}: {len(overlap)} scored files overlap fit/validation: "
                         f"{sorted(overlap)[:3]}")
    paths = [data_dir / "data" / config / name for name in names]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"{path}: scored parquet file is missing")
    tasks = [(path, artifact, source_root, model) for path in paths]
    arrays: dict[str, np.ndarray] = {}
    try:
        for i, out in enumerate(tqdm(pimap(_predict_task, tasks, jobs), total=len(tasks),
                                     desc=f"predict {model}", unit="shot",
                                     **bar_kwargs(SHOT_EVERY))):
            for field in FIELDS:
                arrays[f"shot_{i:04d}_{field}"] = out[field]
    finally:
        release()
    if list(arrays) != _keys(len(names)):
        raise AssertionError(f"{output}: prediction keys differ from {len(names)} scored files")
    _save(output, arrays)
    record = {
        "schema": "submission_npz_v1",
        "model": model,
        "artifact": str(artifact),
        "artifact_sha256": actual_hash,
        "input_config": config,
        "dataset_revision": file_manifest.get("dataset_revision"),
        "source_root": str(source_root),
        "source_sha256": {str(path.relative_to(source_root)): file_hash(path)
                          for path in source_files},
        "files": names,
        "files_json": str(files_json.resolve()),
        "files_json_sha256": file_hash(files_json),
        "fit_overlap_count": 0,
        "dtype": {field: "float32" for field in FIELDS},
        "prediction_npz": str(output.resolve()),
        "prediction_sha256": file_hash(output),
        "prediction_seconds": time.perf_counter() - started,
    }
    write_json(output.with_suffix(".json"), record)
    print(f"Wrote {output}: {len(names)} finite predictions, "
          f"{record['prediction_seconds']:.1f} s")


def _blend_arrays(base: np.ndarray, candidate: np.ndarray,
                  base_members: int, candidate_members: int) -> np.ndarray:
    if (base.shape != candidate.shape or not np.isfinite(base).all()
            or not np.isfinite(candidate).all()):
        raise ValueError(f"decoded prediction shapes/finiteness disagree: "
                         f"{base.shape} against {candidate.shape}")
    total = base_members + candidate_members
    return ((base.astype(np.float64) * base_members
             + candidate.astype(np.float64) * candidate_members) / total).astype(np.float32)


def blend(base: Path, candidate: Path, output: Path,
          base_members: int, candidate_members: int) -> None:
    started = time.perf_counter()
    if base_members < 1 or candidate_members < 1:
        raise ValueError(f"member weights must be positive, got "
                         f"{base_members} and {candidate_members}")
    base_meta, candidate_meta = _meta(base), _meta(candidate)
    names = base_meta["files"]
    if names != candidate_meta["files"]:
        raise ValueError(f"{base} and {candidate}: scored filenames differ in order or contents")
    expected = _keys(len(names))
    arrays: dict[str, np.ndarray] = {}
    with np.load(base, allow_pickle=False) as left, np.load(candidate, allow_pickle=False) as right:
        if list(left.files) != expected or list(right.files) != expected:
            raise ValueError(f"{base} or {candidate}: NPZ keys differ from {len(names)} shots")
        for key in expected:
            arrays[key] = _blend_arrays(left[key], right[key], base_members, candidate_members)
    _save(output, arrays)
    record = {
        "schema": "submission_npz_v1",
        "method": "decoded maps and both submitted scalars, fixed member-count weighting",
        "weights": {"baseline_members": base_members, "candidate_members": candidate_members},
        "base": str(base.resolve()),
        "base_sha256": file_hash(base),
        "candidate": str(candidate.resolve()),
        "candidate_sha256": file_hash(candidate),
        "files": names,
        "prediction_npz": str(output.resolve()),
        "prediction_sha256": file_hash(output),
        "blend_seconds": time.perf_counter() - started,
    }
    write_json(output.with_suffix(".json"), record)
    print(f"Wrote {output}: {len(names)} shots, "
          f"{base_members}:{candidate_members} fixed weights")


def _blend_many_arrays(arrays: list[np.ndarray], weights: list[int]) -> np.ndarray:
    if len(arrays) != len(weights) or len(arrays) < 2 or any(w < 1 for w in weights):
        raise ValueError(f"blend has {len(arrays)} arrays and weights {weights}; expected "
                         "matching lists with positive integer weights")
    shape = arrays[0].shape
    if any(a.shape != shape or not np.isfinite(a).all() for a in arrays):
        raise ValueError(f"decoded blend shapes/finiteness differ: "
                         f"{[a.shape for a in arrays]}")
    total = np.zeros(shape, dtype=np.float64)
    for arr, weight in zip(arrays, weights, strict=True):
        total += arr.astype(np.float64) * weight
    return (total / sum(weights)).astype(np.float32)


def blend_many(inputs: list[Path], weights: list[int], output: Path) -> None:
    started = time.perf_counter()
    if len(inputs) != len(weights) or len(inputs) < 2 or any(w < 1 for w in weights):
        raise ValueError(f"blend has {len(inputs)} inputs and weights {weights}; expected "
                         "matching lists with positive integer weights")
    records = [_meta(path) for path in inputs]
    names = records[0]["files"]
    if any(record["files"] != names for record in records[1:]):
        raise ValueError("blend inputs have different scored filenames or order")
    expected = _keys(len(names))
    result: dict[str, np.ndarray] = {}
    with ExitStack() as stack:
        packed = [stack.enter_context(np.load(path, allow_pickle=False)) for path in inputs]
        for path, record, z in zip(inputs, records, packed, strict=True):
            digest = file_hash(path)
            if record["prediction_sha256"] != digest or list(z.files) != expected:
                raise ValueError(f"{path}: prediction hash or NPZ keys differ from provenance")
        for key in expected:
            result[key] = _blend_many_arrays([z[key] for z in packed], weights)
    _save(output, result)
    record = {
        "schema": "submission_npz_v1",
        "method": "single float64 weighted sum of decoded maps and scalars, one float32 cast",
        "inputs": [{"path": str(path.resolve()), "prediction_sha256": file_hash(path),
                    "members": weight}
                   for path, weight in zip(inputs, weights, strict=True)],
        "total_members": sum(weights),
        "files": names,
        "prediction_npz": str(output.resolve()),
        "prediction_sha256": file_hash(output),
        "blend_seconds": time.perf_counter() - started,
    }
    write_json(output.with_suffix(".json"), record)
    print(f"Wrote {output}: {len(names)} shots, member weights {weights}")


def splice_scalars(map_source: Path, scalar_source: Path, output: Path) -> None:
    """Keep all map bits from one blend and both scalar blocks from another."""
    map_record, scalar_record = _meta(map_source), _meta(scalar_source)
    names = map_record["files"]
    if names != scalar_record["files"]:
        raise ValueError("map and scalar sources have different scored filenames or order")
    expected = _keys(len(names))
    arrays: dict[str, np.ndarray] = {}
    with (np.load(map_source, allow_pickle=False) as maps,
          np.load(scalar_source, allow_pickle=False) as scalars):
        for path, record, packed in ((map_source, map_record, maps),
                                     (scalar_source, scalar_record, scalars)):
            if file_hash(path) != record["prediction_sha256"] or list(packed.files) != expected:
                raise ValueError(f"{path}: prediction hash or keys differ from provenance")
        for i in range(len(names)):
            map_key = f"shot_{i:04d}_psirz"
            psi = maps[map_key]
            if psi.dtype != np.float32 or not np.isfinite(psi).all():
                raise ValueError(f"{map_source}: {map_key} must be finite float32, got {psi.dtype}")
            arrays[map_key] = psi
            for field in ("q95", "betaN"):
                key = f"shot_{i:04d}_{field}"
                value = scalars[key]
                if (value.dtype != np.float32 or value.shape != (len(psi),)
                        or not np.isfinite(value).all()):
                    raise ValueError(f"{scalar_source}: {key} has shape/dtype "
                                     f"{value.shape}/{value.dtype}, expected "
                                     f"({len(psi)},)/float32 and finite")
                arrays[key] = value
    _save(output, arrays)
    with (np.load(output, allow_pickle=False) as selected,
          np.load(map_source, allow_pickle=False) as maps,
          np.load(scalar_source, allow_pickle=False) as scalars):
        for key in expected:
            reference = maps[key] if key.endswith("_psirz") else scalars[key]
            if selected[key].tobytes() != reference.tobytes():
                raise AssertionError(f"{output}: selected {key} differs from its source bits")
    record = {
        "schema": "submission_npz_v1",
        "method": "bit-exact maps from map source; bit-exact q95/betaN from scalar source",
        "map_source": str(map_source.resolve()),
        "map_source_sha256": file_hash(map_source),
        "scalar_source": str(scalar_source.resolve()),
        "scalar_source_sha256": file_hash(scalar_source),
        "files": names,
        "prediction_npz": str(output.resolve()),
        "prediction_sha256": file_hash(output),
        "exact_map_arrays": len(names),
        "exact_scalar_arrays": 2 * len(names),
    }
    write_json(output.with_suffix(".json"), record)
    print(f"Wrote {output}: {len(names)} exact maps and {2 * len(names)} exact scalar arrays")


def score(prediction: Path, files_json: Path, data_dir: Path,
          metrics_json: Path, jobs: int, scorer_root: Path | None = None) -> None:
    if scorer_root is not None:
        scorer_root = scorer_root.resolve()
        sys.path.insert(0, str(scorer_root))
    import local_score

    source = Path(local_score.__file__).resolve()
    if scorer_root is not None and not source.is_relative_to(scorer_root):
        raise ValueError(f"scorer loaded from {source}, expected {scorer_root}")
    names = _files(files_json)
    meta = _meta(prediction)
    if names != meta["files"] or file_hash(prediction) != meta["prediction_sha256"]:
        raise ValueError(f"{prediction}: scored files or prediction hash differ from its record")
    paths = [data_dir.resolve() / "data/diii_d_train" / name for name in names]
    result = local_score.main([
        "--source", "local", "--local-data-dir", str(data_dir.resolve()),
        "--config", "diii_d_train", "--jobs", str(jobs),
        "--pred", str(prediction.resolve()),
        "--metrics-json", str(metrics_json.resolve()),
        "--files", *[str(path) for path in paths],
    ])
    if result != 0:
        raise RuntimeError(f"{prediction}: scorer returned {result}")
    scored = json.loads(metrics_json.read_text())
    if scored["files"] != names:
        raise ValueError(f"{metrics_json}: scorer files differ from prediction provenance")
    record = {
        "scorer_source": str(source),
        "scorer_source_sha256": file_hash(source),
        "scorer_root": str(scorer_root) if scorer_root is not None else None,
        "prediction": str(prediction.resolve()),
        "prediction_sha256": file_hash(prediction),
        "files": names,
        "metrics_json": str(metrics_json.resolve()),
        "metrics_json_sha256": file_hash(metrics_json),
    }
    write_json(metrics_json.with_suffix(".provenance.json"), record)
    print(f"Native scorer JSON: {metrics_json} on {len(names)} identical shots")


def main() -> int:
    install_timestamps()
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("predict")
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--artifact", type=Path, required=True)
    p.add_argument("--artifact-sha256", required=True)
    p.add_argument("--input-config")
    p.add_argument("--files-json", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--jobs", type=int, required=True)
    p = sub.add_parser("blend")
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--base-members", type=int, required=True)
    p.add_argument("--candidate-members", type=int, required=True)
    p = sub.add_parser("blend-many")
    p.add_argument("--inputs", type=Path, nargs="+", required=True)
    p.add_argument("--weights", type=int, nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("splice-scalars")
    p.add_argument("--map-source", type=Path, required=True)
    p.add_argument("--scalar-source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("score")
    p.add_argument("--prediction", type=Path, required=True)
    p.add_argument("--files-json", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--metrics-json", type=Path, required=True)
    p.add_argument("--jobs", type=int, required=True)
    p.add_argument("--scorer-root", type=Path)
    args = parser.parse_args()
    if args.command == "predict":
        predict(args.source_root, args.artifact, args.files_json, args.data_dir,
                args.model, args.output, args.jobs, args.artifact_sha256, args.input_config)
    elif args.command == "blend":
        blend(args.base, args.candidate, args.output,
              args.base_members, args.candidate_members)
    elif args.command == "blend-many":
        blend_many(args.inputs, args.weights, args.output)
    elif args.command == "splice-scalars":
        splice_scalars(args.map_source, args.scalar_source, args.output)
    else:
        score(args.prediction, args.files_json, args.data_dir, args.metrics_json, args.jobs,
              args.scorer_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
