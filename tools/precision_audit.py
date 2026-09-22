"""Create a scorer input that exactly mirrors the submission's float16 flux cast.

This is an audit artifact. It never changes the source float32 predictions or the submission
writer. The two submitted scalars retain their original float32 bits.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from toolkit.records import file_hash, write_json
from tools.historical_compare import _keys, _meta, _save


def quantize_flux(source: Path, target: Path) -> dict[str, object]:
    record = _meta(source)
    names = record["files"]
    if file_hash(source) != record["prediction_sha256"]:
        raise ValueError(f"{source}: prediction hash differs from its provenance record")
    expected = _keys(len(names))
    arrays: dict[str, np.ndarray] = {}
    changed = 0
    total = 0
    max_error = 0.0
    with np.load(source, allow_pickle=False) as original:
        if list(original.files) != expected:
            raise ValueError(f"{source}: keys differ from {len(names)} ordered scored shots")
        for key in expected:
            value = np.asarray(original[key])
            if value.dtype != np.float32 or not np.isfinite(value).all():
                raise ValueError(f"{source}: {key} dtype {value.dtype}, expected finite float32; "
                                 f"nonfinite {int((~np.isfinite(value)).sum())}")
            if key.endswith("_psirz"):
                with np.errstate(over="ignore", invalid="ignore"):
                    half = value.astype(np.float16)
                if not np.isfinite(half).all():
                    raise ValueError(f"{source}: {key} float16 cast produced "
                                     f"{int((~np.isfinite(half)).sum())} nonfinite values")
                restored = half.astype(np.float32)
                changed += int(np.count_nonzero(restored != value))
                total += value.size
                max_error = max(max_error, float(np.max(np.abs(restored - value))))
                arrays[key] = half
            else:
                arrays[key] = value
    _save(target, arrays)
    with (np.load(source, allow_pickle=False) as before,
          np.load(target, allow_pickle=False) as after):
        for key in expected:
            if key.endswith("_psirz"):
                if after[key].dtype != np.float16 or after[key].tobytes() != arrays[key].tobytes():
                    raise AssertionError(f"{target}: quantized flux {key} changed on publication")
            elif after[key].dtype != np.float32 or after[key].tobytes() != before[key].tobytes():
                raise AssertionError(f"{target}: scalar {key} changed during flux-only cast")
    output: dict[str, object] = {
        "schema": "submission_npz_v1",
        "method": "submission_skeleton.py psirz float32-to-float16 cast; q95/betaN unchanged",
        "source_prediction": str(source.resolve()),
        "source_prediction_sha256": file_hash(source),
        "files": names,
        "prediction_npz": str(target.resolve()),
        "prediction_sha256": file_hash(target),
        "dtypes": {"psirz": "float16", "q95": "float32", "betaN": "float32"},
        "flux_values_changed": changed,
        "flux_values_total": total,
        "flux_cast_max_abs_error": max_error,
        "scalar_bits_unchanged": True,
        "float16_nonfinite_count": 0,
        "source_bytes": source.stat().st_size,
        "output_bytes": target.stat().st_size,
    }
    write_json(target.with_suffix(".json"), output)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    args = parser.parse_args()
    result = quantize_flux(args.source, args.target)
    print(f"Wrote {args.target}: {result['flux_values_changed']} / "
          f"{result['flux_values_total']} flux values changed; scalar bits unchanged")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
