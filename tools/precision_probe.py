"""One-off flux-only float16 probe from a pinned float32 prediction NPZ.

The output is deliberately separate from the normal float32 submission path. Both
submitted scalar arrays retain their exact float32 bits. This tool has no scorer or
model implementation.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from toolkit.records import file_hash, write_json

FIELDS = ("psirz", "q95", "betaN")


def _keys(shots: int) -> list[str]:
    return [f"shot_{i:04d}_{field}" for i in range(shots) for field in FIELDS]


def _cast_flux(key: str, value: np.ndarray, frames: int) -> np.ndarray:
    expected = (frames, 65, 65)
    if value.shape != expected or value.dtype != np.float32 or not np.isfinite(value).all():
        raise ValueError(
            f"{key}: shape/dtype/nonfinite {value.shape}/{value.dtype}/"
            f"{int((~np.isfinite(value)).sum())}, expected {expected}/float32/0"
        )
    with np.errstate(over="ignore", invalid="ignore"):
        half = value.astype(np.float16)
    if not np.isfinite(half).all():
        raise ValueError(
            f"{key}: float16 cast produced {int((~np.isfinite(half)).sum())} nonfinite values"
        )
    return half


def _scalar(key: str, value: np.ndarray, frames: int) -> np.ndarray:
    expected = (frames,)
    if value.shape != expected or value.dtype != np.float32 or not np.isfinite(value).all():
        raise ValueError(
            f"{key}: shape/dtype/nonfinite {value.shape}/{value.dtype}/"
            f"{int((~np.isfinite(value)).sum())}, expected {expected}/float32/0"
        )
    return value


def _npy(zipped: zipfile.ZipFile, key: str, value: np.ndarray) -> None:
    with zipped.open(f"{key}.npy", "w", force_zip64=True) as stream:
        np.lib.format.write_array(stream, value, allow_pickle=False)


def build(spec_path: Path, arm: str) -> dict[str, Any]:
    """Create and re-open a complete NPZ before atomically publishing it."""
    started = time.perf_counter()
    spec = json.loads(spec_path.read_text())
    if spec.get("transform") != "psirz_float32_to_float16_only" or arm not in ("held70", "public"):
        raise ValueError(f"{spec_path}: unsupported transform or arm {arm}")
    producer_hash = file_hash(Path(__file__))
    if producer_hash != spec["producer_sha256"]:
        raise ValueError(
            f"{spec_path}: producer SHA-256 {producer_hash}, expected {spec['producer_sha256']}"
        )
    recipe = Path(spec["recipe"])
    if file_hash(recipe) != spec["recipe_sha256"]:
        raise ValueError(f"{recipe}: recipe SHA-256 differs from {spec_path}")
    item = spec[arm]
    source = Path(item["source"])
    target = Path(item["target"])
    manifest_path = Path(item["manifest"])
    source_record = Path(item["source_record"])
    if target.exists():
        raise FileExistsError(f"{target}: refusing to replace a completed probe")
    for path, expected in (
        (source, item["source_sha256"]),
        (manifest_path, item["manifest_sha256"]),
        (source_record, item["source_record_sha256"]),
    ):
        actual = file_hash(path)
        if actual != expected:
            raise ValueError(f"{path}: SHA-256 {actual}, expected {expected}")
    manifest = json.loads(manifest_path.read_text())
    names = manifest["files"]
    shots = item["shots"]
    if len(names) != shots or len(set(names)) != shots:
        raise ValueError(f"{manifest_path}: {len(names)} files, expected {shots} distinct files")
    record = json.loads(source_record.read_text())
    recorded_hash = record.get("prediction_sha256")
    if recorded_hash is None:
        recorded_hash = record["final_npz"]["sha256"]
    if recorded_hash != item["source_sha256"]:
        raise ValueError(f"{source_record}: prediction SHA differs from pinned source")
    if "files" in record and record["files"] != names:
        raise ValueError(f"{source_record}: filename order differs from {manifest_path}")
    expected_keys = _keys(shots)
    if "frames" in manifest:
        frames = manifest["frames"]
        if len(frames) != shots or sum(frames) != item["frames"]:
            raise ValueError(f"{manifest_path}: frame counts differ from {item['frames']}")
    else:
        frames = None
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=target.parent, prefix=f".{target.stem}.", suffix=".npz")
    changed = 0
    total = 0
    max_error = 0.0
    try:
        with os.fdopen(fd, "w+b") as stream, np.load(source, allow_pickle=False) as original:
            if list(original.files) != expected_keys:
                raise ValueError(f"{source}: keys differ from {shots} ordered shots")
            with zipfile.ZipFile(
                stream,
                "w",
                compression=zipfile.ZIP_DEFLATED,
                compresslevel=6,
                allowZip64=True,
            ) as zipped:
                frame_sum = 0
                for i in range(shots):
                    prefix = f"shot_{i:04d}"
                    n = int(frames[i]) if frames is not None else len(original[f"{prefix}_q95"])
                    frame_sum += n
                    key = f"{prefix}_psirz"
                    psi = original[key]
                    half = _cast_flux(key, psi, n)
                    restored = half.astype(np.float32)
                    changed += int(np.count_nonzero(restored != psi))
                    total += psi.size
                    max_error = max(max_error, float(np.max(np.abs(restored - psi))))
                    _npy(zipped, key, half)
                    for field in FIELDS[1:]:
                        key = f"{prefix}_{field}"
                        _npy(zipped, key, _scalar(key, original[key], n))
                if frame_sum != item["frames"]:
                    raise ValueError(f"{source}: {frame_sum} frames, expected {item['frames']}")
            stream.flush()
            os.fsync(stream.fileno())
        with (
            np.load(source, allow_pickle=False) as original,
            np.load(temporary, allow_pickle=False) as produced,
        ):
            if list(produced.files) != expected_keys:
                raise ValueError(f"{temporary}: output keys changed")
            for key in expected_keys:
                before, after = original[key], produced[key]
                if key.endswith("_psirz"):
                    if after.dtype != np.float16 or not np.array_equal(
                        after, before.astype(np.float16)
                    ):
                        raise ValueError(f"{temporary}: quantized {key} differs")
                elif after.dtype != np.float32 or after.tobytes() != before.tobytes():
                    raise ValueError(f"{temporary}: scalar {key} changed bits")
        os.link(temporary, target)
    finally:
        Path(temporary).unlink(missing_ok=True)
    result: dict[str, Any] = {
        "variant": "float16-flux-only-probe",
        "arm": arm,
        "source": str(source),
        "source_sha256": item["source_sha256"],
        "source_record": str(source_record),
        "source_record_sha256": item["source_record_sha256"],
        "manifest": str(manifest_path),
        "manifest_sha256": item["manifest_sha256"],
        "spec": str(spec_path),
        "spec_sha256": file_hash(spec_path),
        "producer_sha256": producer_hash,
        "target": str(target),
        "target_sha256": file_hash(target),
        "shots": shots,
        "frames": item["frames"],
        "arrays": len(expected_keys),
        "dtypes": {"psirz": "float16", "q95": "float32", "betaN": "float32"},
        "flux_values_changed": changed,
        "flux_values_total": total,
        "flux_cast_max_abs_error": max_error,
        "scalar_bits_unchanged": True,
        "all_array_shapes_unchanged": True,
        "nonfinite_count": 0,
        "output_bytes": target.stat().st_size,
        "build_verify_seconds": time.perf_counter() - started,
        "files": names,
        "prediction_sha256": file_hash(target),
    }
    write_json(target.with_suffix(".json"), result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--arm", choices=("held70", "public"), required=True)
    args = parser.parse_args()
    result = build(args.spec, args.arm)
    print(
        f"{args.arm}: {result['shots']} shots, {result['flux_values_changed']} / "
        f"{result['flux_values_total']} flux values changed; scalar bits unchanged"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
