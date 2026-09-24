"""Freeze the single OOF physical-map residual U-Net pilot before real data."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import joblib

from my_experiments.data_check import check_dataset
from my_experiments.run import digest, source_files, verify_source
from toolkit.records import file_hash, write_json

ROOT = Path(__file__).resolve().parent.parent
SPEC = ROOT / "experiments/map-residual-unet-review.json"
PARAMS = ROOT / "baselines/map-residual-unet.params.yaml"
DATA = ROOT.parent / "downloaded_huggingface/hf_dataset"
OOF = ROOT / "runs/oof-normal-k8-5fold-20260923"
SMOKE = ROOT / "runs/map-residual-unet-preflight/synthetic.json"


def verify_campaign(campaign: Path) -> dict[str, Any]:
    """Reject an edited snapshot, source artifact, role list or synthetic gate."""
    campaign = campaign.resolve()
    meta: dict[str, Any] = json.loads((campaign / "manifest.json").read_text())
    if ROOT.resolve() != (campaign / "source").resolve():
        raise ValueError(f"{campaign}: code imported from {ROOT}, expected frozen source")
    for relative, expected in meta["source_sha256"].items():
        path = campaign / "source" / relative
        if file_hash(path) != expected:
            raise ValueError(f"{path}: frozen source SHA differs from campaign")
    origin = Path(meta["origin_root"])
    for relative, key in (("runs/oof-normal-k8-5fold-20260923/files.json",
                            "oof_files_manifest_sha256"),
                          ("runs/oof-normal-k8-5fold-20260923/fold0/base.joblib",
                            "fold0_artifact_sha256"),
                          ("runs/map-residual-unet-preflight/synthetic.json",
                            "synthetic_smoke_sha256")):
        path = origin / relative
        if file_hash(path) != meta[key]:
            raise ValueError(f"{path}: frozen external dependency SHA differs")
    scorer = origin / "runs/geometry-loss-screen-4a5516b9b1684353/source"
    for relative, expected in meta["scorer_functional_sha256"].items():
        if (file_hash(scorer / relative) != expected
                or file_hash(campaign / "source" / relative) != expected):
            raise ValueError(f"{relative}: scorer-native functional source differs from "
                             "campaign snapshot or pinned scorer")
    old_campaign = json.loads((origin / "runs/oof-normal-k8-5fold-20260923/"
                               "campaign.json").read_text())
    for relative, expected in old_campaign["source_sha256"].items():
        if file_hash(origin / "runs/oof-normal-k8-5fold-20260923/source" / relative) != expected:
            raise ValueError(f"{relative}: old OOF base source snapshot differs")
    return meta


def prepare(campaign: Path) -> None:
    """Snapshot all source bytes and exact OOF/gate/tail filename roles."""
    started = time.perf_counter()
    campaign = campaign.resolve()
    if campaign.exists():
        raise FileExistsError(f"{campaign}: map residual campaign already exists")
    spec: dict[str, Any] = json.loads(SPEC.read_text())
    if file_hash(PARAMS) != spec["frozen_references"]["model_yaml_sha256"]:
        raise ValueError(f"{PARAMS}: frozen model YAML SHA differs from review")
    previous = spec["frozen_references"]
    files_path = OOF / "files.json"
    artifact_path = OOF / "fold0/base.joblib"
    if (file_hash(OOF / "campaign.json") != previous["oof_campaign_sha256"]
            or file_hash(files_path) != previous["oof_files_manifest_sha256"]
            or file_hash(artifact_path) != previous["fold0_artifact_sha256"]
            or file_hash(ROOT / "my_experiments/diii_d_dataset.json")
            != previous["dataset_manifest_sha256"]
            or file_hash(ROOT / "uv.lock") != previous["dependency_lock_sha256"]):
        raise ValueError(f"{OOF}: pinned OOF campaign/artifact/dataset/dependency changed")
    old_campaign = json.loads((OOF / "campaign.json").read_text())
    if old_campaign["source_bundle_sha256"] != previous["oof_source_bundle_sha256"]:
        raise ValueError(f"{OOF}: old base source bundle SHA differs from review")
    for relative, expected in old_campaign["source_sha256"].items():
        if file_hash(OOF / "source" / relative) != expected:
            raise ValueError(f"{OOF / 'source' / relative}: old base source was edited")
    prior_inputs = ((previous["prior_fold0_gate_source"],
                     previous["prior_fold0_gate_source_sha256"]),
                    ("runs/b10-matched-heads-20260923/gate_predictions/source.metrics.json",
                     previous["prior_fold0_gate_native_json_sha256"]))
    for relative, expected in prior_inputs:
        if file_hash(ROOT / relative) != expected:
            raise ValueError(f"{ROOT / relative}: prior sealed gate source/score SHA differs")
    if not SMOKE.is_file():
        raise FileNotFoundError(f"{SMOKE}: synthetic CUDA resource gate is missing")
    synthetic: dict[str, Any] = json.loads(SMOKE.read_text())
    if (not synthetic["resource_gate_passed"] or synthetic["real_data_read"]
            or synthetic["params_sha256"] != file_hash(PARAMS)):
        raise ValueError(f"{SMOKE}: synthetic resource gate did not pass on fixed YAML")
    files = json.loads(files_path.read_text())
    artifact: dict[str, Any] = joblib.load(artifact_path)
    train = files["folds"][0]["oof_files"]
    gate = files["gate_files"]
    tail = files["reserved_tail_files"]
    if (len(train) != 1355 or len(gate) != 200 or len(tail) != 70
            or artifact["oof_files"] != train or artifact["val_files"] != []
            or set(artifact["train_files"]) & set(train + gate + tail)
            or artifact["scaler"].n_features_in_ != 84):
        raise ValueError(f"{OOF}: source fit/OOF/gate/tail roles or input width differ")
    dataset_identity = check_dataset(DATA, "diii_d_train")
    scorer = ROOT / "runs/geometry-loss-screen-4a5516b9b1684353/source"
    if verify_source(scorer.parent)["source_hash"] != previous["native_scorer_bundle_sha256"]:
        raise ValueError(f"{scorer}: pinned native scorer bundle SHA differs from review")
    functional_paths = ("fusion_scoring/common.py", "fusion_scoring/lcfs.py",
                        "fusion_scoring/derive.py",
                        "fusion_scoring/masks/d3d_envelope.npz")
    functional_sha = {relative: file_hash(ROOT / relative) for relative in functional_paths}
    if any(file_hash(scorer / relative) != functional_sha[relative]
           for relative in functional_paths):
        raise ValueError(f"{scorer}: contour/li functional implementation differs from "
                         "the model's fixed input-only li guard")
    paths = set(source_files()) | {SPEC, PARAMS}
    campaign.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=campaign.parent, prefix=".map-unet-freeze-") as temp:
        stage = Path(temp) / "campaign"
        source = stage / "source"
        source.mkdir(parents=True)
        for path in sorted(paths):
            relative = path.relative_to(ROOT)
            target = source / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            if file_hash(target) != file_hash(path):
                raise ValueError(f"{path}: source changed while freezing {campaign}")
        hashes = {str(path.relative_to(ROOT)): file_hash(source / path.relative_to(ROOT))
                  for path in sorted(paths)}
        write_json(stage / "manifest.json", {
            "classification": "single_fixed_physical_map_unet_frozen_before_real_target_cache",
            "origin_root": str(ROOT),
            "data_dir": str(DATA),
            "spec_sha256": file_hash(SPEC),
            "model_yaml_sha256": file_hash(PARAMS),
            "source_sha256": hashes,
            "source_bundle_sha256": digest(hashes),
            "dataset_identity": dataset_identity,
            "scorer_functional_sha256": functional_sha,
            "oof_files_manifest_sha256": file_hash(files_path),
            "fold0_artifact_sha256": file_hash(artifact_path),
            "synthetic_smoke_sha256": file_hash(SMOKE),
            "train_files": train,
            "gate_files": gate,
            "reserved_tail_files": tail,
            "train_full_frames": 300793,
            "train_kept_frames": 60171,
            "gate_full_frames": 43142,
            "prepare_wall_seconds": time.perf_counter() - started,
        })
        subprocess.run([
            sys.executable, "-I", "-c",
            "import pathlib,sys; root=pathlib.Path(sys.argv[1]).resolve(); "
            "sys.path.insert(0,str(root)); "
            "import my_experiments.map_residual_unet,tools.map_residual_campaign,"
            "tools.map_residual_data,tools.map_residual_train,tools.map_residual_gate; "
            "assert all(pathlib.Path(module.__file__).resolve().is_relative_to(root) "
            "for module in (my_experiments.map_residual_unet,tools.map_residual_campaign,"
            "tools.map_residual_data,tools.map_residual_train,tools.map_residual_gate))",
            str(source),
        ], cwd="/tmp", env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"},
            check=True)
        stage.rename(campaign)
    print(f"Frozen map residual U-Net campaign {campaign}: "
          f"{len(train)} OOF train, {len(gate)} sealed gate, {len(tail)} reserved tail")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.campaign)


if __name__ == "__main__":
    main()
