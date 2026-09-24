"""Build the three fixed decoded float32 blocks of the four-salt screen."""

from __future__ import annotations

import argparse
import json
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import numpy as np

from toolkit.records import file_hash, write_json
from tools.historical_compare import _save

ROOT = Path(__file__).resolve().parent.parent
FIELDS = ("psirz", "q95", "betaN")
OLD_WEIGHTS = (5, 1, 1, 1, 2, 1)
SALTS = (1, 2, 3, 4)
MODELS = ("mlp", "seq")


def _record(path: Path, names: list[str]) -> dict[str, Any]:
    record: dict[str, Any] = json.loads(path.with_suffix(".json").read_text())
    if record["files"] != names or record["prediction_sha256"] != file_hash(path):
        raise ValueError(f"{path}: prediction hash or ordered 70 files differ")
    return record


def _checked_array(packed: Any, key: str, path: Path, shape: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(packed[key])
    if array.dtype != np.float32 or array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{path}: {key} shape/dtype/finite "
                         f"{array.shape}/{array.dtype}/{bool(np.isfinite(array).all())}, "
                         f"expected {shape}/float32/true")
    return array


def _decoded_block(old: list[np.ndarray], new: list[np.ndarray],
                   old_scalar: np.ndarray | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One float64 accumulation and one float32 cast for each predeclared block."""
    if len(old) != len(OLD_WEIGHTS) or len(new) != 8:
        raise ValueError(f"old/new member arrays {len(old)}/{len(new)}, expected 6/8")
    shape = old[0].shape
    if any(array.shape != shape for array in [*old, *new]):
        raise ValueError(f"old/new array shapes differ from {shape}")
    old_total = np.zeros(shape, dtype=np.float64)
    new_total = np.zeros(shape, dtype=np.float64)
    for array, weight in zip(old, OLD_WEIGHTS, strict=True):
        old_total += array.astype(np.float64) * weight
    for array in new:
        new_total += array.astype(np.float64)
    if old_scalar is not None:
        if old_scalar.shape != shape or old_scalar.dtype != np.float32:
            raise ValueError(f"old scalar {old_scalar.shape}/{old_scalar.dtype}, "
                             f"expected {shape}/float32")
        old_block = old_scalar.astype(np.float64)
    else:
        old_block = old_total / sum(OLD_WEIGHTS)
    new_block = new_total / len(new)
    mixed = 0.5 * old_block + 0.5 * new_block
    return old_block.astype(np.float32), new_block.astype(np.float32), mixed.astype(np.float32)


def build(campaign: Path, grid_path: Path, files_json: Path,
          old_scalar_path: Path, old_anchor_path: Path, output_dir: Path) -> None:
    """Verify all sources and build matched old, new-eight and fixed half/half rows."""
    started = time.perf_counter()
    campaign, output_dir = campaign.resolve(), output_dir.resolve()
    manifest_path = campaign / "source/experiments/cross-salt-four-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    names = manifest["score_files"]
    if len(names) != 70 or json.loads(files_json.read_text())["files"] != names:
        raise ValueError(f"{files_json}: expected exact 70 frozen salt-0 score files")
    grid = json.loads(grid_path.read_text())
    old_paths = [ROOT / item["prediction"] for item in grid["sources"]]
    if tuple(item["members"] for item in grid["sources"]) != OLD_WEIGHTS:
        raise ValueError(f"{grid_path}: old map weights differ from {OLD_WEIGHTS}")
    for path, item in zip(old_paths, grid["sources"], strict=True):
        if file_hash(path) != item["sha256"]:
            raise ValueError(f"{path}: old decoded source hash differs from pinned grid")
    scalar_expected = grid["candidate_psi_eleven_qb_prior_eight"]["q95_betaN_source_sha256"]
    if file_hash(old_scalar_path) != scalar_expected:
        raise ValueError(f"{old_scalar_path}: prior-eight scalar source differs from grid")
    review = json.loads((campaign / "source/experiments/cross-salt-four-review.json").read_text())
    if file_hash(old_anchor_path) != review["evaluation"]["old_control_sha256"]:
        raise ValueError(f"{old_anchor_path}: saved old native anchor differs from preregistration")
    new_paths = [campaign / "predictions" / f"s{salt}_{model}.npz"
                 for salt in SALTS for model in MODELS]
    for path in [*old_paths, old_scalar_path, old_anchor_path, *new_paths]:
        _record(path, names)
    for salt in SALTS:
        complete = json.loads((campaign / f"base_s{salt}/complete.json").read_text())
        expected_artifact = complete["sha256"]["baseline.joblib"]
        for model in MODELS:
            path = campaign / "predictions" / f"s{salt}_{model}.npz"
            record = _record(path, names)
            if record["model"] != model or record["artifact_sha256"] != expected_artifact:
                raise ValueError(f"{path}: member or artifact hash differs from frozen fit")
    arrays: dict[str, dict[str, np.ndarray]] = {name: {} for name in
                                                   ("old_decoded", "new_eight", "mixed_half")}
    max_anchor_difference = {field: 0.0 for field in FIELDS}
    differing_anchor_arrays = {field: 0 for field in FIELDS}
    with ExitStack() as stack:
        old = [stack.enter_context(np.load(path, allow_pickle=False)) for path in old_paths]
        new = [stack.enter_context(np.load(path, allow_pickle=False)) for path in new_paths]
        scalar = stack.enter_context(np.load(old_scalar_path, allow_pickle=False))
        anchor = stack.enter_context(np.load(old_anchor_path, allow_pickle=False))
        expected_keys = [f"shot_{index:04d}_{field}" for index in range(70)
                         for field in FIELDS]
        for path, packed in zip([*old_paths, *new_paths, old_scalar_path, old_anchor_path],
                                [*old, *new, scalar, anchor], strict=True):
            if list(packed.files) != expected_keys:
                raise ValueError(f"{path}: NPZ keys differ from exact 70-shot schema")
        for index in range(70):
            map_key = f"shot_{index:04d}_psirz"
            shape = anchor[map_key].shape
            if len(shape) != 3 or shape[1:] != (65, 65):
                raise ValueError(f"{old_anchor_path}: {map_key} shape {shape}, expected T/65/65")
            for field in FIELDS:
                key = f"shot_{index:04d}_{field}"
                expected_shape = shape if field == "psirz" else (shape[0],)
                old_arrays = [_checked_array(packed, key, path, expected_shape)
                              for packed, path in zip(old, old_paths, strict=True)]
                new_arrays = [_checked_array(packed, key, path, expected_shape)
                              for packed, path in zip(new, new_paths, strict=True)]
                old_scalar = (_checked_array(scalar, key, old_scalar_path, expected_shape)
                              if field != "psirz" else None)
                decoded, added, mixed = _decoded_block(old_arrays, new_arrays, old_scalar)
                arrays["old_decoded"][key] = decoded
                arrays["new_eight"][key] = added
                arrays["mixed_half"][key] = mixed
                saved = _checked_array(anchor, key, old_anchor_path, expected_shape)
                if decoded.tobytes() != saved.tobytes():
                    differing_anchor_arrays[field] += 1
                    max_anchor_difference[field] = max(
                        max_anchor_difference[field],
                        float(np.max(np.abs(decoded.astype(np.float64) - saved))))
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Any] = {}
    for name, contents in arrays.items():
        output = output_dir / f"{name}.npz"
        _save(output, contents)
        record = {
            "schema": "submission_npz_v1",
            "method": "pre-registered decoded float64 block averaging, one float32 cast",
            "block": name,
            "files": names,
            "old_inputs": [{"path": str(path), "sha256": file_hash(path), "weight": weight}
                           for path, weight in zip(old_paths, OLD_WEIGHTS, strict=True)],
            "new_inputs": [{"path": str(path), "sha256": file_hash(path), "weight": 1}
                           for path in new_paths],
            "old_scalar_source_sha256": file_hash(old_scalar_path),
            "prediction_sha256": file_hash(output),
        }
        write_json(output.with_suffix(".json"), record)
        outputs[name] = {"path": str(output), "sha256": record["prediction_sha256"]}
    write_json(output_dir / "blocks.json", {
        "classification": "fixed_four_salt_float32_decoded_blocks_before_native_score",
        "campaign_source_sha256": json.loads(
            (campaign / "campaign.json").read_text())["source_hash"],
        "manifest_sha256": file_hash(manifest_path),
        "review_sha256": file_hash(campaign / "source/experiments/cross-salt-four-review.json"),
        "files_json_sha256": file_hash(files_json),
        "old_native_anchor_sha256": file_hash(old_anchor_path),
        "old_decoded_vs_native_differing_arrays": differing_anchor_arrays,
        "old_decoded_vs_native_max_abs": max_anchor_difference,
        "outputs": outputs,
        "elapsed_seconds": time.perf_counter() - started,
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--grid", type=Path, required=True)
    parser.add_argument("--files-json", type=Path, required=True)
    parser.add_argument("--old-scalars", type=Path, required=True)
    parser.add_argument("--old-anchor", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    build(args.campaign, args.grid, args.files_json, args.old_scalars,
          args.old_anchor, args.output_dir)


if __name__ == "__main__":
    main()
