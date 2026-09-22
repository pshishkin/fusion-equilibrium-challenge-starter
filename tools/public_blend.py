"""Build one pinned, fully local DIII-D public-test prediction asset.

The public shot order is lexicographic by parquet basename, matching submission_skeleton.py.
Each fitted artifact is predicted in a separate process against its immutable source tree. The
final NPZ streams one shot at a time: eleven-model maps and the registered eight-model scalar
block are averaged in float64 and cast once to float32. No public target is read or required.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
import zipfile
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from toolkit.records import file_hash, write_json
from tools.historical_compare import _blend_many_arrays, _keys

ROOT = Path(__file__).resolve().parent.parent
TARGETS = {"efit_psirz", "efit_q95", "efit_beta_n"}


def _load_spec(path: Path) -> dict[str, Any]:
    spec = json.loads(path.read_text())
    if len(spec["sources"]) != 6 or spec["psi_member_weights"] != [5, 1, 1, 1, 2, 1]:
        raise ValueError(f"{path}: expected six registered sources with 5:1:1:1:2:1 map weights")
    if spec["q95_betaN_source_indices"] != [0, 1, 2, 3]:
        raise ValueError(f"{path}: scalar source indices differ from registered prior-eight block")
    if spec["q95_betaN_member_weights"] != [5, 1, 1, 1]:
        raise ValueError(f"{path}: scalar member weights differ from 5:1:1:1")
    if spec["dataset"]["config"] != "diii_d_public_test":
        raise ValueError(f"{path}: this builder only supports DIII-D public test")
    return spec


def _hash_names(names: list[str]) -> str:
    return hashlib.sha256("\n".join(names).encode()).hexdigest()


def prepare_manifest(spec_path: Path, data_dir: Path, output: Path) -> dict[str, Any]:
    spec = _load_spec(spec_path)
    dataset = spec["dataset"]
    source_manifest = ROOT / dataset["local_dataset_manifest"]
    download_record = ROOT / dataset["local_download_record"]
    for path, expected in ((source_manifest, dataset["local_dataset_manifest_sha256"]),
                           (download_record, dataset["local_download_record_sha256"])):
        got = file_hash(path)
        if got != expected:
            raise ValueError(f"{path}: SHA-256 {got}, expected {expected}")
    pinned = json.loads(source_manifest.read_text())
    downloaded = json.loads(download_record.read_text())
    revision = dataset["revision"]
    if (pinned["revision"] != revision or downloaded["revision"] != revision
            or downloaded["sha256_verified"] is not True):
        raise ValueError(f"{source_manifest}: dataset revisions or SHA verification disagree "
                         f"with pinned revision {revision}")
    config = dataset["config"]
    prefix = f"data/{config}/"
    expected = {Path(item["path"]).name: item for item in pinned["files"]
                if item["path"].startswith(prefix) and item["path"].endswith(".parquet")}
    folder = data_dir / "data" / config
    files = sorted(folder.glob("*.parquet"))
    names = [path.name for path in files]
    if len(files) != dataset["shots"] or set(names) != set(expected):
        raise ValueError(f"{folder}: {len(files)} parquet files, expected {dataset['shots']}; "
                         f"missing={sorted(set(expected) - set(names))[:3]}, "
                         f"unexpected={sorted(set(names) - set(expected))[:3]}")
    ordered_hash = _hash_names(names)
    if ordered_hash != dataset["ordered_filenames_sha256"]:
        raise ValueError(f"{folder}: ordered filename SHA-256 {ordered_hash}, expected "
                         f"{dataset['ordered_filenames_sha256']}")
    frames: list[int] = []
    identities: list[dict[str, Any]] = []
    for path in files:
        stat = path.stat()
        expected_size = int(expected[path.name]["size"])
        if stat.st_size != expected_size:
            raise ValueError(f"{path}: {stat.st_size} bytes, expected {expected_size}")
        with path.open("rb") as stream:
            start = stream.read(4)
            stream.seek(-4, 2)
            end = stream.read(4)
        if start != b"PAR1" or end != b"PAR1":
            raise ValueError(f"{path}: parquet markers {start!r}/{end!r}, expected PAR1/PAR1")
        parquet = pq.ParquetFile(path)
        meta = parquet.metadata
        columns = set(parquet.schema_arrow.names)
        if meta.num_rows != 1 or meta.num_row_groups != 1 or TARGETS & columns:
            raise ValueError(f"{path}: rows/groups {meta.num_rows}/{meta.num_row_groups}, "
                             f"target columns {sorted(TARGETS & columns)}; expected one "
                             "input-only shot")
        values = [meta.row_group(0).column(i).num_values
                  for i in range(meta.row_group(0).num_columns)
                  if meta.row_group(0).column(i).path_in_schema == "efit_times.list.element"]
        if len(values) != 1 or values[0] < 1:
            raise ValueError(f"{path}: efit_times metadata {values}, expected one positive "
                             "frame count")
        frames.append(int(values[0]))
        identities.append({"name": path.name, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                           "dataset_oid": expected[path.name]["oid"]})
    if sum(frames) != dataset["frames"]:
        raise ValueError(f"{folder}: {sum(frames)} public frames, expected {dataset['frames']}")
    record = {
        "config": config,
        "dataset_revision": revision,
        "dataset_repo": dataset["repo_id"],
        "dataset_manifest_sha256": file_hash(source_manifest),
        "download_record_sha256": file_hash(download_record),
        "sha256_verified_at_download": True,
        "files": names,
        "ordered_filenames_sha256": ordered_hash,
        "frames": frames,
        "total_frames": sum(frames),
        "input_identities": identities,
        "target_columns_absent_in_all_files": True,
        "source_spec_sha256": file_hash(spec_path),
    }
    write_json(output, record)
    return record


def _verify_old_source(root: Path, commit: str, tree: str) -> None:
    got_tree = subprocess.check_output(["git", "rev-parse", f"{commit}^{{tree}}"],
                                       cwd=ROOT, text=True).strip()
    if got_tree != tree:
        raise ValueError(f"git tree for {commit} is {got_tree}, expected {tree}")
    tracked = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", commit],
                                      cwd=ROOT, text=True).splitlines()
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*")
              if path.is_file() and "__pycache__" not in path.parts}
    if actual != set(tracked):
        raise ValueError(f"{root}: {len(actual)} files, git {commit} has {len(tracked)}; "
                         f"missing={sorted(set(tracked) - actual)[:3]}, "
                         f"unexpected={sorted(actual - set(tracked))[:3]}")
    for rel in tracked:
        # git archive smudges the three Git LFS demo parquets into their large data bytes,
        # while git show reads the tracked pointers. They are not imported by inference.
        if rel.startswith("parquet_data/") and rel.endswith(".parquet"):
            continue
        expected = subprocess.check_output(["git", "show", f"{commit}:{rel}"], cwd=ROOT)
        path = root / rel
        if hashlib.sha256(path.read_bytes()).digest() != hashlib.sha256(expected).digest():
            raise ValueError(f"{path}: bytes differ from git {commit}:{rel}")


def _verify_sources(spec: dict[str, Any]) -> None:
    from my_experiments.run import verify_source

    old = spec["sources"][0]
    _verify_old_source(ROOT / old["source_root"], spec["historical_git_commit"],
                       spec["historical_git_tree"])
    for item in spec["sources"]:
        artifact = ROOT / item["artifact"]
        got = file_hash(artifact)
        if got != item["artifact_sha256"]:
            raise ValueError(f"{artifact}: SHA-256 {got}, expected {item['artifact_sha256']}")
        if "source_hash" in item:
            source = ROOT / item["source_root"]
            campaign = source.parent
            meta = verify_source(campaign)
            if meta["source_hash"] != item["source_hash"]:
                raise ValueError(f"{campaign}: source hash {meta['source_hash']}, expected "
                                 f"{item['source_hash']}")


def _predict_sources(spec: dict[str, Any], manifest_path: Path, data_dir: Path,
                     prediction_dir: Path) -> list[Path]:
    names = json.loads(manifest_path.read_text())["files"]
    prediction_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for item in spec["sources"]:
        out = prediction_dir / f"{item['name']}.npz"
        sidecar = out.with_suffix(".json")
        if out.exists() or sidecar.exists():
            if not out.is_file() or not sidecar.is_file():
                raise ValueError(f"{out}: incomplete prior prediction or provenance")
            record = json.loads(sidecar.read_text())
            if (record["prediction_sha256"] != file_hash(out) or record["files"] != names
                    or record["artifact_sha256"] != item["artifact_sha256"]
                    or record["files_json_sha256"] != file_hash(manifest_path)
                    or record["source_root"] != str((ROOT / item["source_root"]).resolve())
                    or record["input_config"] != spec["dataset"]["config"]
                    or record["dataset_revision"] != spec["dataset"]["revision"]):
                raise ValueError(f"{out}: existing prediction differs from frozen public spec")
            print(f"Reusing verified {out}", flush=True)
        else:
            cmd = [sys.executable, "-u", "-m", "tools.historical_compare", "predict",
                   "--source-root", str(ROOT / item["source_root"]),
                   "--artifact", str(ROOT / item["artifact"]),
                   "--artifact-sha256", item["artifact_sha256"],
                   "--files-json", str(manifest_path), "--data-dir", str(data_dir),
                   "--input-config", spec["dataset"]["config"],
                   "--model", item["model"], "--output", str(out),
                   "--jobs", str(spec["jobs"])]
            env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
                       MKL_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
            print(f"Predicting {item['name']} on {len(names)} public shots", flush=True)
            subprocess.run(cmd, cwd=ROOT, env=env, check=True)
        outputs.append(out)
    return outputs


def _write_npy(zipped: zipfile.ZipFile, name: str, array: np.ndarray) -> None:
    with zipped.open(f"{name}.npy", "w", force_zip64=True) as stream:
        np.lib.format.write_array(stream, array, allow_pickle=False)


def assemble(spec: dict[str, Any], manifest: dict[str, Any], inputs: list[Path],
             output: Path) -> dict[str, Any]:
    """Stream final NPZ per shot; no full-map intermediate or second cast."""
    if len(inputs) != len(spec["sources"]):
        raise ValueError(f"{output}: {len(inputs)} inputs, expected {len(spec['sources'])}")
    if output.exists():
        raise FileExistsError(f"{output}: final public asset already exists")
    weights = spec["psi_member_weights"]
    scalar_indices = spec["q95_betaN_source_indices"]
    scalar_weights = spec["q95_betaN_member_weights"]
    names = manifest["files"]
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=output.parent, suffix=".npz")
    try:
        with os.fdopen(fd, "w+b") as stream, ExitStack() as stack:
            packed = [stack.enter_context(np.load(path, allow_pickle=False)) for path in inputs]
            expected = _keys(len(names))
            for item, path, z in zip(spec["sources"], inputs, packed, strict=True):
                sidecar = json.loads(path.with_suffix(".json").read_text())
                if (sidecar["prediction_sha256"] != file_hash(path)
                        or sidecar["artifact_sha256"] != item["artifact_sha256"]
                        or sidecar["files"] != names or list(z.files) != expected):
                    raise ValueError(f"{path}: source prediction differs from its provenance")
            with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED,
                                 compresslevel=6, allowZip64=True) as zipped:
                for i, (name, n) in enumerate(zip(names, manifest["frames"], strict=True)):
                    prefix = f"shot_{i:04d}"
                    maps = [z[f"{prefix}_psirz"] for z in packed]
                    if any(a.shape != (n, 65, 65) or a.dtype != np.float32 for a in maps):
                        raise ValueError(f"{name}: map shapes/dtypes "
                                         f"{[(a.shape, str(a.dtype)) for a in maps]}, "
                                         f"expected {(n, 65, 65)}/float32")
                    _write_npy(zipped, f"{prefix}_psirz", _blend_many_arrays(maps, weights))
                    for field in ("q95", "betaN"):
                        scalars = [packed[j][f"{prefix}_{field}"] for j in scalar_indices]
                        if any(a.shape != (n,) or a.dtype != np.float32 for a in scalars):
                            raise ValueError(f"{name}: {field} shapes/dtypes "
                                             f"{[(a.shape, str(a.dtype)) for a in scalars]}, "
                                             f"expected {(n,)}/float32")
                        _write_npy(zipped, f"{prefix}_{field}",
                                   _blend_many_arrays(scalars, scalar_weights))
                    if (i + 1) % 50 == 0:
                        print(f"Assembled {i + 1}/{len(names)} public shots", flush=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, output)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return {"path": str(output), "sha256": file_hash(output), "bytes": output.stat().st_size,
            "shots": len(names), "frames": sum(manifest["frames"]),
            "dtype": {"psirz": "float32", "q95": "float32", "betaN": "float32"}}


def build(spec_path: Path, data_dir: Path, manifest_path: Path) -> dict[str, Any]:
    from validate_submission import validate

    started = time.perf_counter()
    spec = _load_spec(spec_path)
    expected_manifest = prepare_manifest(spec_path, data_dir, manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if manifest != expected_manifest:
        raise ValueError(f"{manifest_path}: public input manifest changed during preflight")
    _verify_sources(spec)
    run = (ROOT / spec["output"]).parent.parent
    prediction_dir = run / "predictions"
    before = time.perf_counter()
    inputs = _predict_sources(spec, manifest_path, data_dir, prediction_dir)
    predicted_seconds = time.perf_counter() - before
    output = ROOT / spec["output"]
    before = time.perf_counter()
    final = assemble(spec, manifest, inputs, output)
    assembled_seconds = time.perf_counter() - before
    before = time.perf_counter()
    if validate(output, spec["dataset"]["config"], 0, "local", data_dir) != 0:
        raise RuntimeError(f"{output}: full 874-shot local submission validation failed")
    validation_seconds = time.perf_counter() - before
    record = {
        "spec": str(spec_path.resolve()), "spec_sha256": file_hash(spec_path),
        "manifest": str(manifest_path.resolve()), "manifest_sha256": file_hash(manifest_path),
        "dataset_revision": manifest["dataset_revision"],
        "historical_git_commit": spec["historical_git_commit"],
        "historical_git_tree": spec["historical_git_tree"],
        "ordered_filenames_sha256": manifest["ordered_filenames_sha256"],
        "input_config": manifest["config"],
        "input_files": len(manifest["files"]), "input_frames": manifest["total_frames"],
        "sources": [
            {"name": item["name"], "members": item["members"],
             "artifact_sha256": item["artifact_sha256"],
             "prediction_npz": str(path), "prediction_sha256": file_hash(path),
             "source_root": item["source_root"], "source_hash": item.get("source_hash")}
            for item, path in zip(spec["sources"], inputs, strict=True)
        ],
        "psi_member_weights": spec["psi_member_weights"],
        "q95_betaN_source_indices": spec["q95_betaN_source_indices"],
        "q95_betaN_member_weights": spec["q95_betaN_member_weights"],
        "final_npz": final,
        "full_local_validation_passed": True,
        "writer_source_sha256": file_hash(Path(__file__)),
        "historical_adapter_sha256": file_hash(ROOT / "tools/historical_compare.py"),
        "timing_seconds": {"predict_sources": predicted_seconds,
                           "assemble_final": assembled_seconds,
                           "validate_final": validation_seconds,
                           "total": time.perf_counter() - started},
    }
    write_json(run / "public_build.json", record)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("manifest", "build"):
        p = sub.add_parser(command)
        p.add_argument("--spec", type=Path, required=True)
        p.add_argument("--data-dir", type=Path, required=True)
        p.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "manifest":
        result = prepare_manifest(args.spec, args.data_dir, args.manifest)
        print(f"Pinned {len(result['files'])} input-only public DIII-D shots, "
              f"{result['total_frames']} frames -> {args.manifest}")
    else:
        result = build(args.spec, args.data_dir, args.manifest)
        print(json.dumps(result["final_npz"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
