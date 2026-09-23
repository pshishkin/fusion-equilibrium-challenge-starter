"""Freeze the bounded B10 matched-head campaign after raw support passed."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import joblib

from my_experiments.data_check import check_dataset
from my_experiments.run import source_files
from toolkit.records import file_hash, write_json

ROOT = Path(__file__).resolve().parent.parent
SPEC = ROOT / "experiments/b10-two-rate-information-gate-v4.json"
PARAMS = ROOT / "baselines/b10-two-rate-info-gate.params.yaml"
DATA = ROOT.parent / "downloaded_huggingface/hf_dataset"


def prepare(run: Path) -> None:
    """Pin source/HP, all shot roles and the completed input-only support audit."""
    started = time.perf_counter()
    run = run.resolve()
    if run.exists():
        raise FileExistsError(f"{run}: B10 matched-head campaign already exists")
    spec = json.loads(SPEC.read_text())
    if file_hash(Path(__file__)) != spec["campaign_helper_sha256"]:
        raise ValueError(f"{SPEC}: campaign source SHA differs")
    parents = (spec["amends_spec"], spec["amends_spec_v2"],
               spec["amends_spec_v3"])
    for relative in parents:
        if not (ROOT / relative).exists():
            raise ValueError(f"{ROOT / relative}: parent B10 spec missing")
    if file_hash(ROOT / spec["amends_spec_v2"]) != spec["amends_spec_v2_sha256"]:
        raise ValueError(f"{SPEC}: prior v2 B10 spec SHA differs")
    if file_hash(ROOT / spec["amends_spec_v3"]) != spec["amends_spec_v3_sha256"]:
        raise ValueError(f"{SPEC}: prior v3 B10 spec SHA differs")
    if file_hash(PARAMS) != spec["effective_params_sha256"]:
        raise ValueError(f"{PARAMS}: fixed B10 model YAML SHA differs")
    for relative, key in (("tools/b10_waveforms.py", "waveform_helper_sha256"),
                          ("tools/b10_targets.py", "target_helper_sha256"),
                          ("tools/b10_assemble.py", "assembly_helper_sha256"),
                          ("tools/b10_head_model.py", "head_helper_sha256"),
                          ("tools/b10_gate.py", "gate_helper_sha256")):
        if file_hash(ROOT / relative) != spec[key]:
            raise ValueError(f"{ROOT / relative}: frozen B10 helper {key} differs")
    support_path = ROOT / spec["support_audit"]["path"]
    if file_hash(support_path) != spec["support_audit"]["sha256"]:
        raise ValueError(f"{support_path}: complete raw support SHA differs")
    support = json.loads(support_path.read_text())
    if (not support["raw_support_complete"] or support["train_shots"] != 1355
            or support["gate_shots"] != 200 or support["train_frames"] != 300793
            or support["gate_frames"] != 43142):
        raise ValueError(f"{support_path}: B10 raw support status/counts differ")
    old = ROOT / spec["frozen_data_plan"]["OOF_campaign"]
    files_path = old / "files.json"
    artifact_path = ROOT / spec["base_artifact"]["path"]
    if (file_hash(files_path) != spec["frozen_data_plan"]["OOF_files_manifest_sha256"]
            or file_hash(artifact_path) != spec["base_artifact"]["sha256"]):
        raise ValueError(f"{old}: base OOF file/artifact SHA differs")
    files = json.loads(files_path.read_text())
    artifact: dict[str, Any] = joblib.load(artifact_path)
    head = files["folds"][0]["oof_files"]
    gate = files["gate_files"]
    tail = files["reserved_tail_files"]
    if (len(head) != 1355 or len(gate) != 200 or len(tail) != 70
            or artifact["oof_files"] != head or artifact["val_files"] != []
            or set(artifact["train_files"]) & set(head + gate + tail)
            or artifact["scaler"].n_features_in_ != 84
            or head + gate != [row["shot"] for row in support["per_shot"]]):
        raise ValueError(f"{old}: B10 head/gate/tail/base artifact roles or width differ")
    dataset_identity = check_dataset(DATA, "diii_d_train")
    run.mkdir(parents=True)
    source = run / "source"
    source.mkdir()
    for path in source_files():
        target = source / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    extras = [SPEC, PARAMS, *(ROOT / relative for relative in parents)]
    for path in set(extras):
        target = source / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    hashes = {str(path.relative_to(ROOT)): file_hash(source / path.relative_to(ROOT))
              for path in source_files()}
    for path in set(extras):
        hashes[str(path.relative_to(ROOT))] = file_hash(source / path.relative_to(ROOT))
    write_json(run / "manifest.json", {
        "classification": "B10_matched_raw_lowpass_heads_frozen_before_supervised_targets",
        "origin_root": str(ROOT),
        "data_dir": str(DATA),
        "spec_sha256": file_hash(SPEC),
        "effective_params_sha256": file_hash(PARAMS),
        "source_sha256": hashes,
        "dataset_identity": dataset_identity,
        "support_audit_sha256": spec["support_audit"]["sha256"],
        "base_artifact_sha256": spec["base_artifact"]["sha256"],
        "head_train_files": head,
        "gate_files": gate,
        "reserved_tail_files": tail,
        "head_train_frames": 300793,
        "gate_frames": 43142,
        "original_rate_class_gate": support["sampling_rate_classes"],
        "prepare_wall_seconds": time.perf_counter() - started,
    })
    subprocess.run([
        sys.executable, "-c",
        "import pathlib,tools.b10_campaign,tools.b10_waveforms,tools.b10_targets,"
        "tools.b10_assemble,tools.b10_head_model,tools.b10_gate; "
        "assert pathlib.Path(tools.b10_gate.__file__).resolve()=="
        "pathlib.Path('tools/b10_gate.py').resolve()",
    ], cwd=source, env={**os.environ, "PYTHONPATH": str(source)}, check=True)
    print(f"Frozen B10 matched-head campaign {run}: {len(head)} OOF train + "
          f"{len(gate)} sealed gate shots")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.run)


if __name__ == "__main__":
    main()
