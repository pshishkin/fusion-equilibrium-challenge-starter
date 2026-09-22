"""Compose an exact full-frame refit target cache from verified train and validation caches.

This only rearranges cached rows. It never extracts a contour or changes the auxiliary target
producer. The completed control artifact supplies the ordered filenames; the frozen refit campaign
supplies the intended union split and the producer identity. A cache is published only after all
of those identities and every concatenated target row have been checked.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import numpy.typing as npt
import yaml

from my_experiments import aux_targets
from my_experiments.baseline_model import parse_share, sorted_shots, split_train_val
from my_experiments.data_check import check_dataset
from my_experiments.run import completed, read_json, verify_source
from toolkit.records import file_hash, write_json

ConsArray = npt.NDArray[np.float32]
LengthsArray = npt.NDArray[np.int64]


def _producer_paths(root: Path) -> list[Path]:
    paths = [root / rel for rel in (
        "my_experiments/aux_targets.py", "my_experiments/target_metric.py",
        "local_score.py", "experiments.py", "uv.lock",
    )]
    paths += sorted((root / "fusion_scoring").glob("*.py"))
    paths += sorted((root / "fusion_scoring/masks").glob("*.npz"))
    return paths


def _verify_producers(roots: tuple[Path, Path, Path]) -> dict[str, str]:
    """The exact files in aux_targets.cache_path must agree across all three code trees."""
    relative = [{p.relative_to(root).as_posix() for p in _producer_paths(root)}
                for root in roots]
    if relative[0] != relative[1] or relative[0] != relative[2]:
        raise ValueError(f"auxiliary producer file sets differ: "
                         f"{[len(paths) for paths in relative]}")
    hashes: dict[str, str] = {}
    for rel in sorted(relative[0]):
        values = [file_hash(root / rel) for root in roots]
        if len(set(values)) != 1:
            raise ValueError(f"auxiliary producer {rel} differs across control, refit, and "
                             f"working trees: {values}")
        hashes[rel] = values[0]
    return hashes


def _load_block(path: Path, shots: int) -> tuple[ConsArray, LengthsArray]:
    if not path.is_file():
        raise FileNotFoundError(f"{path}: source target cache for {shots} shots is missing")
    with np.load(path, allow_pickle=False) as payload:
        if set(payload.files) != {"cons", "lengths"}:
            raise ValueError(f"{path}: arrays {payload.files}, expected cons and lengths")
        cons = np.asarray(payload["cons"])
        lengths = np.asarray(payload["lengths"])
    if cons.dtype != np.float32 or cons.ndim != 2 or cons.shape[1] != 7:
        raise ValueError(f"{path}: cons shape/dtype {cons.shape}/{cons.dtype}, expected "
                         f"(frames, 7)/float32")
    if lengths.dtype != np.int64 or lengths.ndim != 1 or len(lengths) != shots:
        raise ValueError(f"{path}: lengths shape/dtype {lengths.shape}/{lengths.dtype}, "
                         f"expected ({shots},)/int64")
    if np.any(lengths <= 0) or int(lengths.sum()) != len(cons):
        raise ValueError(f"{path}: lengths sum {int(lengths.sum())}, cons rows {len(cons)}; "
                         f"all {shots} shot lengths must be positive")
    return cons, lengths


def _compose(first: tuple[ConsArray, LengthsArray],
             second: tuple[ConsArray, LengthsArray]) -> tuple[ConsArray, LengthsArray]:
    cons_a, lengths_a = first
    cons_b, lengths_b = second
    cons = np.concatenate((cons_a, cons_b), axis=0)
    lengths = np.concatenate((lengths_a, lengths_b))
    if not np.array_equal(cons[:len(cons_a)], cons_a, equal_nan=True):
        raise AssertionError("concatenated training target rows changed")
    if not np.array_equal(cons[len(cons_a):], cons_b, equal_nan=True):
        raise AssertionError("concatenated validation target rows changed")
    if cons[:len(cons_a)].tobytes() != cons_a.tobytes():
        raise AssertionError("concatenated training target bits changed")
    if cons[len(cons_a):].tobytes() != cons_b.tobytes():
        raise AssertionError("concatenated validation target bits changed")
    if not np.array_equal(lengths, np.concatenate((lengths_a, lengths_b))):
        raise AssertionError("concatenated shot lengths changed")
    return cons, lengths


def _verify_ready(path: Path, expected: tuple[ConsArray, LengthsArray], shots: int) -> None:
    actual = _load_block(path, shots)
    for label, got, want in (("cons", actual[0], expected[0]),
                             ("lengths", actual[1], expected[1])):
        if got.shape != want.shape or got.tobytes() != want.tobytes():
            raise ValueError(f"{path}: existing {label} differs from the verified source blocks")


def _publish(path: Path, payload: tuple[ConsArray, LengthsArray], shots: int) -> str:
    """Publish complete NPZ bytes atomically without overwriting another ready cache."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        _verify_ready(path, payload, shots)
        return "already_present_verified"
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".npz")
    try:
        with os.fdopen(fd, "wb") as stream:
            np.savez_compressed(stream, cons=payload[0], lengths=payload[1])
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            _verify_ready(path, payload, shots)
            return "already_present_verified"
        _verify_ready(path, payload, shots)
        return "published"
    finally:
        Path(temporary).unlink(missing_ok=True)


def _names_hash(names: list[str]) -> str:
    return hashlib.sha256("\n".join(names).encode()).hexdigest()


def _check_name_identity(train: list[str], validation: list[str],
                         refit: list[str], scored: list[str]) -> None:
    union = train + validation
    if len(set(union)) != len(union):
        raise ValueError(f"control train+validation has {len(union)} names but only "
                         f"{len(set(union))} distinct names")
    overlap = set(union) & set(scored)
    if overlap:
        raise ValueError(f"control fit and scored files overlap on {len(overlap)} names: "
                         f"{sorted(overlap)[:3]}")
    if union != refit:
        raise ValueError(f"refit has {len(refit)} ordered files, expected the control's "
                         f"{len(union)} ordered train+validation files")


def compose_cache(baseline_artifact: Path, refit_campaign: Path, data_dir: Path,
                  record_path: Path) -> dict[str, Any]:
    started = time.perf_counter()
    baseline_artifact = baseline_artifact.resolve()
    refit_campaign = refit_campaign.resolve()
    data_dir = data_dir.resolve()
    if baseline_artifact.name != "baseline.joblib":
        raise ValueError(f"{baseline_artifact}: expected completed baseline.joblib")
    control_campaign = baseline_artifact.parent.parent
    control_meta = verify_source(control_campaign)
    refit_meta = verify_source(refit_campaign)
    if not completed(baseline_artifact.parent, control_meta["source_hash"]):
        raise ValueError(f"{baseline_artifact.parent}: no verified completion record")
    producer_hashes = _verify_producers((control_campaign / "source",
                                         refit_campaign / "source",
                                         Path(__file__).resolve().parent.parent))
    refit_source = refit_campaign / "source/my_experiments/baseline_model.py"
    current_source = Path(__file__).with_name("baseline_model.py")
    if file_hash(refit_source) != file_hash(current_source):
        raise ValueError(f"{current_source}: split implementation differs from frozen refit "
                         f"{refit_source}")
    artifact = joblib.load(baseline_artifact)
    dataset = check_dataset(data_dir, artifact["config"])
    if artifact["dataset_identity"] != dataset:
        raise ValueError(f"{baseline_artifact}: dataset identity differs from {data_dir}")
    train_names = list(artifact["train_files"])
    val_names = list(artifact["val_files"])
    union_names = train_names + val_names
    metrics = read_json(baseline_artifact.parent / "metrics.json")
    scored_names = list(metrics["files"])
    protocol = refit_meta["identity"]["protocol"]
    if protocol.get("fit_mode") != "fixed_steps" or protocol["validation"] != "0/1.0":
        raise ValueError(f"{refit_campaign}: protocol is {protocol}, expected fixed_steps "
                         f"with validation 0/1.0")
    runs = refit_meta["identity"]["runs"]
    if len(runs) != 1:
        raise ValueError(f"{refit_campaign}: {len(runs)} refit runs, expected exactly 1")
    refit_config = yaml.safe_load(runs[0]["params_yaml"])
    refit_salt = int(refit_config["split"]["salt"])
    if refit_salt != artifact["split_salt"]:
        raise ValueError(f"{refit_campaign}: split salt {refit_salt}, control salt "
                         f"{artifact['split_salt']}")
    ordered = sorted_shots(data_dir, artifact["config"], refit_salt)
    train_share, frame_share = parse_share(protocol["train"])
    val_share, val_frame_share = parse_share(protocol["validation"], allow_zero_shots=True)
    if frame_share != 1.0 or val_frame_share != 1.0:
        raise ValueError(f"{refit_campaign}: cache composition needs all frames, got "
                         f"{frame_share} and {val_frame_share}")
    refit_files, empty_val = split_train_val(ordered, train_share, val_share,
                                             allow_empty_validation=True)
    if empty_val:
        raise ValueError(f"{refit_campaign}: validation split has {len(empty_val)} files, "
                         f"expected 0")
    _check_name_identity(train_names, val_names, [p.name for p in refit_files], scored_names)
    expected_score = max(1, round(len(ordered) * float(protocol["score"])))
    if (len(scored_names) != expected_score
            or [p.name for p in ordered[-expected_score:]] != scored_names):
        raise ValueError(f"{refit_campaign}: control scored filenames differ from its "
                         f"{expected_score}-shot tail")
    data_folder = data_dir / "data" / artifact["config"]
    train_files = [data_folder / name for name in train_names]
    val_files = [data_folder / name for name in val_names]
    all_files = train_files + val_files
    source_paths = [aux_targets.cache_path(train_files), aux_targets.cache_path(val_files)]
    target = aux_targets.cache_path(all_files)
    source_hashes = [file_hash(p) for p in source_paths]
    first = _load_block(source_paths[0], len(train_files))
    second = _load_block(source_paths[1], len(val_files))
    if [file_hash(p) for p in source_paths] != source_hashes:
        raise ValueError(f"source cache bytes changed while loading {source_paths}")
    if ([aux_targets.cache_path(train_files), aux_targets.cache_path(val_files)] != source_paths
            or aux_targets.cache_path(all_files) != target):
        raise ValueError("dataset file identities changed while loading source caches")
    payload = _compose(first, second)
    if len(payload[1]) != len(all_files) or int(payload[1].sum()) != len(payload[0]):
        raise ValueError(f"{target}: composed {len(payload[1])} shots/{len(payload[0])} frames, "
                         f"expected {len(all_files)} shots and matching row count")
    status = _publish(target, payload, len(all_files))
    record: dict[str, Any] = {
        "method": "Exact ordered concatenation of full-shot float32 functional arrays and "
                  "int64 frame counts; no new functional extraction",
        "composition_helper_sha256": file_hash(Path(__file__)),
        "numpy_version": np.__version__,
        "control_artifact": str(baseline_artifact),
        "control_artifact_sha256": file_hash(baseline_artifact),
        "control_source_hash": control_meta["source_hash"],
        "refit_campaign": str(refit_campaign),
        "refit_source_hash": refit_meta["source_hash"],
        "dataset_identity": dataset,
        "producer_sha256": producer_hashes,
        "ordered_names_sha256": {
            "train": _names_hash(train_names), "validation": _names_hash(val_names),
            "union": _names_hash(union_names), "scored": _names_hash(scored_names),
        },
        "source_caches": [
            {"path": str(path), "sha256": digest, "shots": len(names),
             "frames": len(block[0])}
            for path, digest, names, block in zip(source_paths, source_hashes,
                                                  (train_names, val_names),
                                                  (first, second), strict=True)
        ],
        "target_cache": {"path": str(target), "sha256": file_hash(target),
                         "shots": len(all_files), "frames": len(payload[0]),
                         "status": status},
        "checks": {"all_target_rows_equal_source_blocks_including_nan": True,
                   "all_target_bits_equal_source_blocks": True,
                   "all_full_shot_lengths_equal_source_blocks": True,
                   "scored_overlap": 0},
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json(record_path, record)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline_artifact", type=Path)
    parser.add_argument("refit_campaign", type=Path)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--record", type=Path, required=True)
    args = parser.parse_args()
    result = compose_cache(args.baseline_artifact, args.refit_campaign, args.data_dir,
                           args.record)
    print(json.dumps(result["target_cache"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
