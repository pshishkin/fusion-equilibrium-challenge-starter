"""Build the gate-excluding OOF auxiliary cache once and compose five fit subsets."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

from my_experiments import aux_targets, baseline_model
from my_experiments.data_check import check_dataset
from my_experiments.oof_split import select_fit_files
from toolkit.parallel import pimap, release
from toolkit.records import file_hash, write_json
from tools.oof_normal_pilot import _checked_context


def _input_identity_and_frames(task: tuple[str, str, int, str]) -> tuple[str, int]:
    name, data_s, expected_size, expected_sha = task
    path = Path(data_s) / "data/diii_d_train" / name
    if path.stat().st_size != expected_size or file_hash(path) != expected_sha:
        raise ValueError(f"{path}: size/SHA differ from pinned dataset manifest")
    import pandas as pd

    times = np.asarray(pd.read_parquet(path, columns=["efit_times"]).iloc[0]["efit_times"],
                       dtype=np.float64)
    if (times.ndim != 1 or len(times) < 1 or not np.isfinite(times).all()
            or (len(times) > 1 and np.any(np.diff(times) <= 0))):
        raise ValueError(f"{path}: efit_times length/order/finiteness {len(times)}, "
                         "expected positive monotone finite")
    return name, len(times)


def select_blocks(cons: np.ndarray, lengths: np.ndarray, allowed: list[str],
                  fitting: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Copy whole source shot blocks, preserving float32 payload bits including NaNs."""
    if (cons.ndim != 2 or cons.shape[1] != 7 or cons.dtype != np.float32
            or lengths.shape != (len(allowed),) or np.any(lengths <= 0)
            or int(lengths.sum()) != len(cons)):
        raise ValueError(f"allowed cache arrays {cons.shape}/{cons.dtype}, lengths "
                         f"{lengths.shape}/{int(lengths.sum())}, expected "
                         f"(frames,7)/float32 and {len(allowed)} positive lengths")
    if len(set(allowed)) != len(allowed) or len(set(fitting)) != len(fitting):
        raise ValueError(f"allowed/fit unique filenames {len(set(allowed))}/"
                         f"{len(set(fitting))}, expected {len(allowed)}/{len(fitting)}")
    lookup = {name: index for index, name in enumerate(allowed)}
    missing = [name for name in fitting if name not in lookup]
    if missing:
        raise ValueError(f"fit filenames absent from allowed cache: {missing[:3]}")
    offsets = np.r_[0, np.cumsum(lengths)]
    parts = [cons[offsets[lookup[name]]:offsets[lookup[name] + 1]] for name in fitting]
    selected = np.concatenate(parts)
    selected_lengths = np.asarray([lengths[lookup[name]] for name in fitting], dtype=np.int64)
    if selected.dtype != np.float32 or int(selected_lengths.sum()) != len(selected):
        raise ValueError(f"selected cache rows/lengths {len(selected)}/"
                         f"{int(selected_lengths.sum())}, expected equal float32")
    return selected, selected_lengths


def _publish(path: Path, cons: np.ndarray, lengths: np.ndarray) -> None:
    if path.exists():
        with np.load(path) as saved:
            existing = saved["cons"]
            existing_lengths = saved["lengths"]
        if (existing.dtype != np.float32 or existing.shape != cons.shape
                or not np.array_equal(existing.view(np.uint32), cons.view(np.uint32))
                or not np.array_equal(existing_lengths, lengths)):
            raise ValueError(f"{path}: existing subset cache differs in target bits or lengths")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".npz")
    try:
        with os.fdopen(fd, "wb") as stream:
            np.savez_compressed(stream, cons=cons, lengths=lengths)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    with np.load(path) as saved:
        published = saved["cons"]
        published_lengths = saved["lengths"]
    if (not np.array_equal(published.view(np.uint32), cons.view(np.uint32))
            or not np.array_equal(published_lengths, lengths)):
        raise ValueError(f"{path}: published target bits/lengths differ from selected source")


def prewarm(run: Path, jobs: int) -> None:
    """Extract only allowed labels, then prove every fold cache exact by shot blocks."""
    started = time.perf_counter()
    run = run.resolve()
    campaign, manifest = _checked_context(run)
    output = run / "aux_cache_ready.json"
    if output.exists():
        raise FileExistsError(f"{output}: auxiliary cache already recorded")
    origin = Path(campaign["origin_root"])
    data_dir = Path(campaign["data_dir"])
    if check_dataset(data_dir, "diii_d_train") != campaign["dataset_identity"]:
        raise ValueError(f"{data_dir}: dataset identity differs from frozen campaign")
    cache_root = origin / ".aux_targets"
    if aux_targets.CACHE.resolve() != cache_root.resolve():
        raise ValueError(f"FUSION_AUX_CACHE={aux_targets.CACHE}, expected {cache_root}")
    ordered = baseline_model.sorted_shots(data_dir, "diii_d_train", 0)
    by_name = {path.name: path for path in ordered}
    allowed_names = manifest["allowed_files"]
    if (len(allowed_names) != 6771
            or set(allowed_names) & (set(manifest["gate_files"])
                                      | set(manifest["reserved_tail_files"]))):
        raise ValueError(f"{run}: allowed/gate/tail file sets overlap or count differs")
    allowed = [by_name[name] for name in allowed_names]
    dataset = json.loads((run / "source/my_experiments/diii_d_dataset.json").read_text())
    try:
        counted = list(pimap(_input_identity_and_frames, [
            (name, str(data_dir), *dataset["configs"]["diii_d_train"][name])
            for name in allowed_names
        ], jobs))
    finally:
        release()
    expected_lengths = np.asarray([count for _, count in counted], dtype=np.int64)
    if [name for name, _ in counted] != allowed_names:
        raise ValueError(f"{run}: input identity task order differs from allowed filenames")
    aux_targets.ensure([("OOF-allowed", allowed)], jobs)
    union_path = aux_targets.cache_path(allowed)
    with np.load(union_path) as saved:
        cons = np.asarray(saved["cons"])
        lengths = np.asarray(saved["lengths"], dtype=np.int64)
    if not np.array_equal(lengths, expected_lengths):
        raise ValueError(f"{union_path}: target lengths {lengths.shape}/{int(lengths.sum())}, "
                         f"expected input lengths {expected_lengths.shape}/"
                         f"{int(expected_lengths.sum())} in identical filename order")
    folds: list[dict[str, Any]] = []
    files_path = run / "files.json"
    spec = Path(manifest["spec_path"])
    params = Path(manifest["params_path"])
    reference = Path(manifest["reference_artifact"])
    for fold_index in range(5):
        fit_files, oof_files, reserved = select_fit_files(
            files_path, ordered, reference, spec, params, fold_index)
        if set(path.name for path in reserved) != set(manifest["reserved_tail_files"]):
            raise ValueError(f"fold {fold_index}: reserved tail file identity differs")
        fit_names = [path.name for path in fit_files]
        subset, subset_lengths = select_blocks(cons, lengths, allowed_names, fit_names)
        subset_path = aux_targets.cache_path(fit_files)
        _publish(subset_path, subset, subset_lengths)
        folds.append({
            "index": fold_index,
            "fit_shots": len(fit_files),
            "oof_shots": len(oof_files),
            "fit_frames": len(subset),
            "cache_path": str(subset_path),
            "cache_sha256": file_hash(subset_path),
            "target_bits_sha256": hashlib.sha256(subset.tobytes()).hexdigest(),
        })
    write_json(output, {
        "classification": "verified_allowed_only_auxiliary_cache_and_five_exact_subsets",
        "campaign_sha256": file_hash(run / "campaign.json"),
        "files_manifest_sha256": campaign["files_manifest_sha256"],
        "allowed_shots": len(allowed),
        "allowed_frames": len(cons),
        "allowed_cache_path": str(union_path),
        "allowed_cache_sha256": file_hash(union_path),
        "producer_sha256": file_hash(Path(aux_targets.__file__)),
        "local_scorer_sha256": file_hash(run / "source/local_score.py"),
        "dependency_lock_sha256": file_hash(run / "source/uv.lock"),
        "excluded_gate_shots": len(manifest["gate_files"]),
        "excluded_tail_shots": len(manifest["reserved_tail_files"]),
        "allowed_full_parquet_sha256_verified_shots": len(allowed),
        "folds": folds,
        "elapsed_seconds": time.perf_counter() - started,
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=24)
    args = parser.parse_args()
    if not 1 <= args.jobs <= 24:
        raise ValueError(f"jobs={args.jobs}, expected 1..24")
    prewarm(args.run, args.jobs)


if __name__ == "__main__":
    main()
