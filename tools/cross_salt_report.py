"""Predict and natively score the twelve fixed rows of the four-salt campaign."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import joblib

from my_experiments import run
from toolkit import records
from toolkit.records import file_hash, write_json
from tools import cross_salt_blend, historical_compare
from tools.cross_salt_blend import build

ROOT = Path(__file__).resolve().parent.parent
SALTS = (1, 2, 3, 4)
MODELS = ("mlp", "seq")


def _native(path: Path, expected_files: list[str]) -> dict[str, Any]:
    metrics = json.loads(path.read_text())
    if metrics["files"] != expected_files or set(metrics["results"]) != {"model"}:
        raise ValueError(f"{path}: native results or ordered score files differ")
    return metrics["results"]["model"]


def _call(command: list[str], cwd: Path, env: dict[str, str]) -> float:
    started = time.perf_counter()
    subprocess.run(command, cwd=cwd, env=env, check=True)
    return time.perf_counter() - started


def _fit_stages(folder: Path) -> dict[str, float]:
    metrics = json.loads((folder / "metrics.json").read_text())
    status = json.loads((folder / "status.json").read_text())
    text = Path(status["log"]).read_text(errors="replace")
    individual: dict[str, float] = {}
    for model in MODELS:
        values = re.findall(rf"\b{model}: fitted in ([0-9.]+) s", text)
        if len(values) != 1:
            raise ValueError(f"{status['log']}: fitted {model} timing rows {len(values)}, "
                             "expected one")
        individual[f"{model}_fit"] = float(values[0])
    timing = metrics["timing_seconds"]
    if status["state"] != "scored" or timing["train"] < sum(individual.values()):
        raise ValueError(f"{folder}: worker status or train/individual fit timing differs")
    return {
        "worker_wall": status["wall_s"],
        "train_including_data_and_preparation": timing["train"],
        "score_own_holdout_ensemble": timing["score"],
        "other_train_read_preparation": timing["train"] - sum(individual.values()),
        **individual,
    }


def _freeze_controller(campaign: Path) -> dict[str, str]:
    """Retain exact live orchestration bytes separately from the evaluator snapshot."""
    sources = {
        "cross_salt_report.py": Path(__file__),
        "cross_salt_blend.py": Path(cross_salt_blend.__file__),
        "historical_compare.py": Path(historical_compare.__file__),
        "run.py": Path(run.__file__),
        "records.py": Path(records.__file__),
    }
    folder = campaign / "comparison_controller"
    folder.mkdir(exist_ok=True)
    hashes: dict[str, str] = {}
    for name, source in sources.items():
        target = folder / name
        digest = file_hash(source)
        if target.exists():
            if file_hash(target) != digest:
                raise ValueError(f"{target}: frozen comparison controller differs from {source}")
        else:
            shutil.copyfile(source, target)
            if file_hash(target) != digest:
                raise ValueError(f"{target}: copied comparison controller hash differs")
        hashes[name] = digest
    write_json(folder / "controller.json", {
        "classification": "live_comparison_controller_copied_before_predictions_or_score",
        "files_sha256": hashes,
    })
    return hashes


def _isolated_source_import(source: Path, modules: tuple[str, ...]) -> None:
    command = [sys.executable, "-I", "-c",
               "import importlib, pathlib, sys; "
               "root=pathlib.Path(sys.argv[1]).resolve(); sys.path.insert(0,str(root)); "
               "mods=[importlib.import_module(x) for x in sys.argv[2:]]; "
               "assert all(pathlib.Path(x.__file__).resolve().is_relative_to(root) for x in mods)",
               str(source), *modules]
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    result = subprocess.run(command, cwd="/tmp", env=environment,
                            capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError(f"{source}: isolated import of {modules} failed "
                           f"with code {result.returncode}: {result.stderr[-500:]}")


def report(campaign: Path, data_dir: Path, predict_jobs: int, score_jobs: int) -> None:
    """Keep every new member fixed and score only the preregistered blend rows."""
    started = time.perf_counter()
    campaign, data_dir = campaign.resolve(), data_dir.resolve()
    meta = run.verify_source(campaign)
    if meta["identity"]["protocol"].get("selection") != "cross_salt_exact":
        raise ValueError(f"{campaign}: expected frozen cross_salt_exact campaign")
    source = campaign / "source"
    manifest_path = source / meta["identity"]["spec"]["manifest"]
    manifest = json.loads(manifest_path.read_text())
    files_json = ROOT / "runs/compact-production-9301af36275620f7/base_s0/metrics.json"
    names = manifest["score_files"]
    if json.loads(files_json.read_text())["files"] != names:
        raise ValueError(f"{files_json}: scorer files differ from cross-salt manifest")
    review = json.loads((source / meta["identity"]["spec"]["review"]).read_text())
    grid_path = ROOT / "grids/historical-eleven-blocks.json"
    old_scalar = ROOT / "runs/historical-five-geometry-20260922/historical_positive_stack.npz"
    old_anchor = ROOT / review["evaluation"]["old_control"]
    scorer_root = ROOT / "runs/geometry-loss-screen-4a5516b9b1684353/source"
    scorer_meta = run.verify_source(scorer_root.parent)
    grid = json.loads(grid_path.read_text())
    if (file_hash(old_anchor) != review["evaluation"]["old_control_sha256"]
            or run.digest(meta["identity"]["source"]) != meta["source_hash"]
            or scorer_meta["source_hash"] != grid["scorer_source_hash"]):
        raise ValueError(f"{campaign}: old anchor, candidate or scorer source identity differs")
    controller_hashes = _freeze_controller(campaign)
    _isolated_source_import(source, ("my_experiments.baseline_model",))
    _isolated_source_import(scorer_root, ("local_score", "machines"))
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.update({
        "FUSION_SHOT_CACHE": str(ROOT / ".shot_cache"),
        "FUSION_AUX_CACHE": str(ROOT / ".aux_targets"),
        "MPLCONFIGDIR": "/tmp/matplotlib-cross-salt-report",
        "OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
    })
    predictions = campaign / "predictions"
    predictions.mkdir(exist_ok=True)
    prediction_seconds: dict[str, float] = {}
    for salt in SALTS:
        folder = campaign / f"base_s{salt}"
        if not run.completed(folder, meta["source_hash"]):
            raise ValueError(f"{folder}: fit and own holdout score are incomplete")
        artifact = folder / "baseline.joblib"
        complete = json.loads((folder / "complete.json").read_text())
        fitted: dict[str, Any] = joblib.load(artifact)
        salt_roles = manifest["salts"][str(salt)]
        if (fitted["train_files"] != salt_roles["train_files"]
                or fitted["val_files"] != salt_roles["validation_files"]
                or set(fitted["holdout_files"]) != set(names)
                or fitted.get("split_mode") != "cross_salt_exact_validation"
                or fitted.get("cross_salt_manifest_sha256") != file_hash(manifest_path)
                or fitted.get("source_sha256") != meta["source_hash"]
                or set(fitted["models"]) != set(MODELS)):
            raise ValueError(f"{artifact}: train/validation/holdout/model/source identity "
                             "differs from the frozen four-salt manifest")
        for model in MODELS:
            name = f"s{salt}_{model}"
            output = predictions / f"{name}.npz"
            if output.exists():
                record = json.loads(output.with_suffix(".json").read_text())
                if (record["prediction_sha256"] != file_hash(output)
                        or record["artifact_sha256"]
                        != complete["sha256"]["baseline.joblib"]
                        or record["model"] != model or record["files"] != names):
                    raise ValueError(f"{output}: existing prediction provenance differs")
                prediction_seconds[name] = record["prediction_seconds"]
                continue
            print(f"Predicting frozen new member {name}", flush=True)
            prediction_seconds[name] = _call([
                sys.executable, "-u", "-m", "tools.historical_compare", "predict",
                "--source-root", str(source), "--artifact", str(artifact),
                "--artifact-sha256", complete["sha256"]["baseline.joblib"],
                "--files-json", str(files_json), "--data-dir", str(data_dir),
                "--model", model, "--output", str(output), "--jobs", str(predict_jobs),
            ], source, env)
    blocks_dir = campaign / "fixed_blocks"
    if (blocks_dir / "blocks.json").exists():
        blocks = json.loads((blocks_dir / "blocks.json").read_text())
        for output in blocks["outputs"].values():
            if file_hash(Path(output["path"])) != output["sha256"]:
                raise ValueError(f"{blocks_dir}: existing block output hash differs")
    else:
        build(campaign, grid_path, files_json, old_scalar, old_anchor, blocks_dir)
        blocks = json.loads((blocks_dir / "blocks.json").read_text())
    rows = {
        "old_native": old_anchor,
        "old_decoded": blocks_dir / "old_decoded.npz",
        **{f"s{salt}_{model}": predictions / f"s{salt}_{model}.npz"
           for salt in SALTS for model in MODELS},
        "new_eight": blocks_dir / "new_eight.npz",
        "mixed_half": blocks_dir / "mixed_half.npz",
    }
    if len(rows) != 12:
        raise AssertionError(f"scored rows {len(rows)}, expected 12")
    native_dir = campaign / "native"
    native_dir.mkdir(exist_ok=True)
    score_seconds: dict[str, float] = {}
    for name, prediction in rows.items():
        output = native_dir / f"{name}.json"
        if output.exists():
            _native(output, names)
            provenance = json.loads(output.with_suffix(".provenance.json").read_text())
            if (provenance["prediction_sha256"] != file_hash(prediction)
                    or provenance["metrics_json_sha256"] != file_hash(output)):
                raise ValueError(f"{output}: existing native score provenance differs")
            score_seconds[name] = 0.0
            continue
        print(f"Native scoring fixed row {name}", flush=True)
        score_seconds[name] = _call([
            sys.executable, "-u", "-m", "tools.historical_compare", "score",
            "--prediction", str(prediction), "--files-json", str(files_json),
            "--data-dir", str(data_dir), "--metrics-json", str(output),
            "--jobs", str(score_jobs), "--scorer-root", str(scorer_root),
        ], source, env)
    native = {name: _native(native_dir / f"{name}.json", names) for name in rows}
    archive = json.loads((ROOT / "runs/historical-five-geometry-20260922/"
                          "historical_psi_eleven_qb_eight.metrics.json").read_text())
    if native["old_native"] != archive["results"]["model"]:
        raise ValueError(f"{native_dir}: original native anchor does not reproduce archived terms")
    old = native["old_decoded"]
    mixed = native["mixed_half"]
    failures = ("psi_nonfinite_frac", "lcfs_fail_frac", "cons_fail_frac")
    no_more_failures = all(mixed[field] <= old[field] for field in failures)
    no_more_failures &= all(mixed["cons_fail_each"][name] <= value for name, value
                            in old["cons_fail_each"].items())
    delta = mixed["S"] - old["S"]
    summary = {
        "classification": "new_four_salt_reused_tail_diagnostic_not_historical_six_salt",
        "campaign": str(campaign),
        "campaign_source_sha256": meta["source_hash"],
        "scorer_source_sha256": scorer_meta["source_hash"],
        "live_controller_sha256": controller_hashes,
        "manifest_sha256": file_hash(manifest_path),
        "prereg_sha256": file_hash(source / meta["identity"]["spec"]["review"]),
        "files_json_sha256": file_hash(files_json),
        "score_files": names,
        "score_shots": len(names),
        "native_rows": native,
        "native_metrics_sha256": {name: file_hash(native_dir / f"{name}.json")
                                  for name in rows},
        "prediction_sha256": {name: file_hash(path) for name, path in rows.items()},
        "old_decoded_minus_native_S": old["S"] - native["old_native"]["S"],
        "mixed_minus_matched_decoded_old_S": delta,
        "native_failures_no_increase": no_more_failures,
        "investment_gate_passed": delta >= 0.0015 and no_more_failures,
        "old_decoded_vs_native_arrays": {
            "differing": blocks["old_decoded_vs_native_differing_arrays"],
            "max_abs": blocks["old_decoded_vs_native_max_abs"],
        },
        "timing_seconds": {
            "auxiliary_prewarm": json.loads(
                (campaign / "aux_cache_ready.json").read_text())["elapsed_seconds"],
            "fit_stages": {str(salt): _fit_stages(campaign / f"base_s{salt}")
                           for salt in SALTS},
            "predictions": prediction_seconds,
            "fixed_blocks": blocks["elapsed_seconds"],
            "native_scoring": score_seconds,
            "report_wall": time.perf_counter() - started,
        },
        "scope": "Same 70 repeatedly used for selection; no public score, "
                 "no independent confirmation",
    }
    write_json(campaign / "cross_salt_result.json", summary)
    print(f"Fixed mixed S {mixed['S']:.6f}; matched old S {old['S']:.6f}; "
          f"delta {delta:+.6f}; investment gate {summary['investment_gate_passed']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--predict-jobs", type=int, default=8)
    parser.add_argument("--score-jobs", type=int, default=24)
    args = parser.parse_args()
    if not 1 <= args.predict_jobs <= 24 or not 1 <= args.score_jobs <= 24:
        raise ValueError("prediction/scorer jobs must both be in 1..24")
    report(args.campaign, args.data_dir, args.predict_jobs, args.score_jobs)


if __name__ == "__main__":
    main()
