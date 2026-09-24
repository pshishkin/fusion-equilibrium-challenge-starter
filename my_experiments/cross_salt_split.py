"""Select exact validation-mode splits with a common, sealed salt-0 holdout."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import joblib

from my_experiments.oof_split import names_hash
from toolkit.records import file_hash


def select_fit_files(manifest_path: Path, ordered_salt: list[Path],
                     ordered_zero: list[Path], reference_artifact: Path,
                     dataset_manifest: Path,
                     salt: int) -> tuple[list[Path], list[Path], list[Path]]:
    """Rebuild every role from filenames and reject a changed corpus or reference."""
    expected_sha = os.environ.get("FUSION_CROSS_SALT_MANIFEST_SHA256")
    actual_sha = file_hash(manifest_path)
    if expected_sha != actual_sha:
        raise ValueError(f"{manifest_path}: SHA256 {actual_sha}, expected frozen {expected_sha!r}")
    record: dict[str, Any] = json.loads(manifest_path.read_text())
    if salt not in (1, 2, 3, 4) or set(record["salts"]) != {"1", "2", "3", "4"}:
        raise ValueError(f"{manifest_path}: salt {salt}, expected one of 1,2,3,4")
    zero = [path.name for path in ordered_zero]
    names = [path.name for path in ordered_salt]
    if (len(zero), len(names), len(set(zero)), len(set(names))) != (7041, 7041, 7041, 7041):
        raise ValueError(f"{manifest_path}: corpus counts/distinct "
                         f"{len(zero)}/{len(names)}/{len(set(zero))}/{len(set(names))}, "
                         "expected 7041 each")
    if set(zero) != set(names):
        raise ValueError(f"{manifest_path}: salt-{salt} corpus and salt-0 corpus differ")
    if record["corpus_salt0_ordered_sha256"] != names_hash(zero):
        raise ValueError(f"{manifest_path}: salt-0 corpus filename hash differs")
    dataset_sha = file_hash(dataset_manifest)
    if dataset_sha != record["pinned_dataset_manifest_sha256"]:
        raise ValueError(f"{dataset_manifest}: SHA256 {dataset_sha}, expected "
                         f"{record['pinned_dataset_manifest_sha256']}")
    artifact_sha = file_hash(reference_artifact)
    if artifact_sha != record["reference_artifact_sha256"]:
        raise ValueError(f"{reference_artifact}: SHA256 {artifact_sha}, expected "
                         f"{record['reference_artifact_sha256']}")
    reference: dict[str, Any] = joblib.load(reference_artifact)
    original_score = zero[-70:]
    if (reference["train_files"] + reference["val_files"] != zero[:-70]
            or record["score_files"] != original_score
            or record["score_ordered_sha256"] != names_hash(original_score)):
        raise ValueError(f"{manifest_path}: original compact roles or 70 scored files differ")
    held = set(original_score)
    holdout = [path for path in ordered_salt if path.name in held]
    remaining = [path for path in ordered_salt if path.name not in held]
    train, validation = remaining[:5633], remaining[5633:]
    salt_record = record["salts"][str(salt)]
    train_names = [path.name for path in train]
    validation_names = [path.name for path in validation]
    if ((len(train), len(validation), len(holdout)) != (5633, 1338, 70)
            or salt_record["train_files"] != train_names
            or salt_record["validation_files"] != validation_names
            or salt_record["train_ordered_sha256"] != names_hash(train_names)
            or salt_record["validation_ordered_sha256"] != names_hash(validation_names)
            or salt_record["holdout_ordered_sha256"]
            != names_hash([path.name for path in holdout])
            or set(train_names) & set(validation_names)
            or (set(train_names) | set(validation_names)) & held):
        raise ValueError(f"{manifest_path}: salt {salt} train/validation/holdout role "
                         "lists, counts, hashes or disjointness differ")
    return train, validation, holdout
