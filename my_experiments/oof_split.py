"""Prove explicit cross-fit shot lists against the compact salt-0 reference.

This module selects filenames only. No competition target is read here.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import joblib

from toolkit.records import file_hash


def names_hash(names: list[str]) -> str:
    """Hash an ordered, length-delimited filename list."""
    return hashlib.sha256(json.dumps(names, separators=(",", ":")).encode()).hexdigest()


def build_manifest(ordered_files: list[Path], reference_artifact: Path,
                   spec_path: Path, params_path: Path) -> dict[str, Any]:
    """Construct five disjoint OOF folds while sealing gate200 and tail70."""
    names = [path.name for path in ordered_files]
    if len(names) != 7041 or len(set(names)) != len(names):
        raise ValueError(f"{ordered_files[0].parent if ordered_files else 'corpus'}: "
                         f"files {len(names)}, distinct {len(set(names))}, expected 7041/7041")
    artifact: dict[str, Any] = joblib.load(reference_artifact)
    train = list(artifact["train_files"])
    validation = list(artifact["val_files"])
    if (len(train), len(validation)) != (5633, 1338):
        raise ValueError(f"{reference_artifact}: train/validation shot counts "
                         f"{len(train)}/{len(validation)}, expected 5633/1338")
    tail = names[-70:]
    if train + validation != names[:-70]:
        raise ValueError(f"{reference_artifact}: ordered train+validation filenames "
                         f"{len(train) + len(validation)}, expected exact corpus head "
                         f"{len(names) - 70}")
    gate = validation[-200:]
    allowed = train + validation[:-200]
    if len(allowed) != 6771 or len(set(allowed + gate + tail)) != len(names):
        raise ValueError(f"{reference_artifact}: allowed/gate/tail counts "
                         f"{len(allowed)}/{len(gate)}/{len(tail)}, expected 6771/200/70 "
                         "with distinct filenames")
    folds: list[dict[str, Any]] = []
    for index in range(5):
        oof = allowed[index::5]
        fitting = [name for position, name in enumerate(allowed) if position % 5 != index]
        if len(oof) + len(fitting) != len(allowed):
            raise ValueError(f"fold {index}: OOF/fit shot counts {len(oof)}/{len(fitting)}, "
                             f"expected sum {len(allowed)}")
        folds.append({
            "index": index,
            "oof_files": oof,
            "oof_ordered_sha256": names_hash(oof),
            "fit_count": len(fitting),
            "fit_ordered_sha256": names_hash(fitting),
        })
    train_set = set(train)
    pilot = [name for name in folds[0]["oof_files"] if name in train_set][:100]
    if len(pilot) != 100:
        raise ValueError(f"fold 0: original-TRAIN pilot shots {len(pilot)}, expected 100")
    return {
        "classification": "fivefold_oof_explicit_file_manifest_before_fit",
        "reference_artifact": str(reference_artifact.resolve()),
        "reference_artifact_sha256": file_hash(reference_artifact),
        "spec_path": str(spec_path.resolve()),
        "spec_sha256": file_hash(spec_path),
        "params_path": str(params_path.resolve()),
        "params_sha256": file_hash(params_path),
        "corpus_ordered_sha256": names_hash(names),
        "original_train_ordered_sha256": names_hash(train),
        "original_validation_ordered_sha256": names_hash(validation),
        "allowed_files": allowed,
        "allowed_ordered_sha256": names_hash(allowed),
        "gate_files": gate,
        "gate_ordered_sha256": names_hash(gate),
        "reserved_tail_files": tail,
        "reserved_tail_ordered_sha256": names_hash(tail),
        "folds": folds,
        "pilot_train_oof_files": pilot,
        "pilot_train_oof_ordered_sha256": names_hash(pilot),
    }


def select_fit_files(manifest_path: Path, ordered_files: list[Path],
                     reference_artifact: Path, spec_path: Path, params_path: Path,
                     fold_index: int) -> tuple[list[Path], list[Path], list[Path]]:
    """Rebuild the manifest from local identities before selecting any fit file."""
    if fold_index not in range(5):
        raise ValueError(f"{manifest_path}: fold index {fold_index}, expected 0..4")
    recorded: dict[str, Any] = json.loads(manifest_path.read_text())
    expected = build_manifest(ordered_files, reference_artifact, spec_path, params_path)
    if recorded != expected:
        raise ValueError(f"{manifest_path}: manifest content/identity differs from the "
                         "current pinned corpus, reference artifact, spec or params")
    allowed = expected["allowed_files"]
    oof = expected["folds"][fold_index]["oof_files"]
    oof_set = set(oof)
    gate_set = set(expected["gate_files"])
    tail_set = set(expected["reserved_tail_files"])
    selected_names = [name for name in allowed if name not in oof_set]
    if (len(selected_names) != expected["folds"][fold_index]["fit_count"]
            or names_hash(selected_names) != expected["folds"][fold_index]["fit_ordered_sha256"]
            or set(selected_names) & (oof_set | gate_set | tail_set)):
        raise ValueError(f"{manifest_path}: fold {fold_index} fit files overlap OOF/gate/tail "
                         "or their count/hash differs")
    by_name = {path.name: path for path in ordered_files}
    return ([by_name[name] for name in selected_names],
            [by_name[name] for name in oof],
            [by_name[name] for name in expected["reserved_tail_files"]])
