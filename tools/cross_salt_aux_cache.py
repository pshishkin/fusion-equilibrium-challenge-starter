"""Prewarm one verified allowed-shot target cache and compose four salt splits."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from my_experiments import aux_targets, baseline_model, run
from my_experiments.cross_salt_split import select_fit_files
from my_experiments.data_check import check_dataset
from toolkit.parallel import pimap, release
from toolkit.records import file_hash, write_json
from tools.oof_aux_cache import _input_identity_and_frames, _publish, select_blocks


def prewarm(campaign: Path, data_dir: Path, reference: Path, jobs: int) -> None:
    """Prove exact input identities and whole-shot target bits before GPU fitting."""
    started = time.perf_counter()
    campaign, data_dir, reference = (path.resolve() for path in
                                     (campaign, data_dir, reference))
    meta = run.verify_source(campaign)
    identity = meta["identity"]
    if identity["protocol"].get("selection") != "cross_salt_exact":
        raise ValueError(f"{campaign}: expected cross_salt_exact frozen protocol")
    manifest_rel = identity["spec"]["manifest"]
    manifest_path = campaign / "source" / manifest_rel
    manifest: dict[str, Any] = json.loads(manifest_path.read_text())
    if file_hash(reference) != manifest["reference_artifact_sha256"]:
        raise ValueError(f"{reference}: reference artifact SHA differs from frozen manifest")
    os.environ["FUSION_CROSS_SALT_MANIFEST_SHA256"] = identity["source"][manifest_rel]
    dataset_identity = check_dataset(data_dir, "diii_d_train")
    zero = baseline_model.sorted_shots(data_dir, "diii_d_train", 0)
    allowed = zero[:-70]
    allowed_names = [path.name for path in allowed]
    if ([path.name for path in zero[-70:]] != manifest["score_files"]
            or len(allowed) != 6971):
        raise ValueError(f"{manifest_path}: allowed/score roles differ from pinned 6971/70")
    dataset = json.loads((campaign / "source/my_experiments/diii_d_dataset.json").read_text())
    try:
        counted = list(pimap(_input_identity_and_frames, [
            (name, str(data_dir), *dataset["configs"]["diii_d_train"][name])
            for name in allowed_names
        ], jobs))
    finally:
        release()
    if [name for name, _ in counted] != allowed_names:
        raise ValueError(f"{manifest_path}: verified input task order differs")
    expected_lengths = np.asarray([count for _, count in counted], dtype=np.int64)
    aux_targets.ensure([("cross-salt-allowed", allowed)], jobs)
    union_path = aux_targets.cache_path(allowed)
    with np.load(union_path) as saved:
        cons = np.asarray(saved["cons"])
        lengths = np.asarray(saved["lengths"], dtype=np.int64)
    if (cons.dtype != np.float32 or len(cons) != int(lengths.sum())
            or not np.array_equal(lengths, expected_lengths)):
        raise ValueError(f"{union_path}: target dtype/shot lengths disagree with "
                         f"{len(allowed)} SHA-verified input files")
    blocks: list[dict[str, Any]] = []
    for salt in (1, 2, 3, 4):
        ordered = baseline_model.sorted_shots(data_dir, "diii_d_train", salt)
        train, validation, held = select_fit_files(
            manifest_path, ordered, zero, reference,
            campaign / "source/my_experiments/diii_d_dataset.json", salt)
        if set(path.name for path in held) != set(manifest["score_files"]):
            raise ValueError(f"{manifest_path}: salt {salt} heldout filenames differ")
        for role, files in (("train", train), ("validation", validation)):
            selected, selected_lengths = select_blocks(
                cons, lengths, allowed_names, [path.name for path in files])
            cache = aux_targets.cache_path(files)
            _publish(cache, selected, selected_lengths)
            blocks.append({
                "salt": salt,
                "role": role,
                "shots": len(files),
                "frames": len(selected),
                "cache_path": str(cache),
                "cache_sha256": file_hash(cache),
            })
    write_json(campaign / "aux_cache_ready.json", {
        "classification": "verified_cross_salt_allowed_auxiliary_cache",
        "source_hash": meta["source_hash"],
        "manifest_sha256": file_hash(manifest_path),
        "reference_artifact_sha256": file_hash(reference),
        "dataset_identity": dataset_identity,
        "allowed_shots": len(allowed),
        "allowed_frames": len(cons),
        "full_parquet_sha256_verified_shots": len(allowed),
        "union_cache_path": str(union_path),
        "union_cache_sha256": file_hash(union_path),
        "producer_sha256": file_hash(Path(aux_targets.__file__)),
        "scorer_sha256": file_hash(campaign / "source/local_score.py"),
        "dependency_lock_sha256": file_hash(campaign / "source/uv.lock"),
        "blocks": blocks,
        "elapsed_seconds": time.perf_counter() - started,
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--reference-artifact", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=24)
    args = parser.parse_args()
    if not 1 <= args.jobs <= 24:
        raise ValueError(f"jobs={args.jobs}, expected 1..24")
    prewarm(args.campaign, args.data_dir, args.reference_artifact, args.jobs)


if __name__ == "__main__":
    main()
