"""Seal and natively score matched B10 waveform residual heads on gate200."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
import yaml

from toolkit.parallel import pimap, release
from toolkit.records import file_hash, write_json
from tools.b10_head_model import WaveformResidualNet, decode_residual

ROOT = Path(__file__).resolve().parent.parent
SPEC = ROOT / "experiments/b10-two-rate-information-gate-v4.json"
ARMS = ("source", "lowpass_seed0", "lowpass_seed1", "raw_seed0")
_MODEL_CACHE: dict[str, WaveformResidualNet] = {}
_BASE_CACHE: dict[str, dict[str, Any]] = {}


def _loaded_model(path: Path, config: dict[str, Any], params_sha: str) -> WaveformResidualNet:
    key = str(path)
    if key not in _MODEL_CACHE:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint["params_sha256"] != params_sha:
            raise ValueError(f"{path}: checkpoint YAML SHA differs from frozen predictor")
        model = WaveformResidualNet(config)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        model.eval()
        _MODEL_CACHE[key] = model
    return _MODEL_CACHE[key]


def _loaded_base(path: Path) -> dict[str, Any]:
    key = str(path)
    if key not in _BASE_CACHE:
        _BASE_CACHE[key] = joblib.load(path)
    return _BASE_CACHE[key]


def _predict_one(task: tuple[str, int, str, str, str, str,
                             list[float], list[float], dict[str, Any], str]
                 ) -> dict[str, Any]:
    (name, frames, sidecar_s, base_s, artifact_s, model_root_s, mean_list,
     std_list, cfg, params_sha) = task
    torch.set_num_threads(2)
    with (np.load(sidecar_s, allow_pickle=False) as sidecar,
          np.load(base_s, allow_pickle=False) as saved):
        raw = np.asarray(sidecar["raw"])
        control = np.asarray(sidecar["lowpass"])
        mask = np.asarray(sidecar["mask"])
        rates = np.asarray(sidecar["rates"])
        frame84 = np.asarray(saved["x84"])
        base52 = np.asarray(saved["base_scaled52"])
        source_psi = np.asarray(saved["source_psi"])
        source_q95 = np.asarray(saved["source_q95"])
        source_beta = np.asarray(saved["source_betaN"])
    if (raw.shape != control.shape or raw.shape != (frames, 41, 21)
            or mask.shape != (frames, 41) or rates.shape != (21,)
            or frame84.shape != (frames, 84) or base52.shape != (frames, 52)
            or source_psi.shape != (frames, 65, 65)
            or source_q95.shape != (frames,) or source_beta.shape != (frames,)):
        raise ValueError(f"{name}: B10 gate sidecar/base shapes differ from {frames} frames")
    for label, values in (("raw", raw), ("lowpass", control), ("mask", mask),
                          ("rates", rates), ("x84", frame84), ("base52", base52),
                          ("source_psi", source_psi), ("source_q95", source_q95),
                          ("source_betaN", source_beta)):
        if values.dtype != np.float32 or not np.isfinite(values).all():
            raise ValueError(f"{name}: {label} dtype/nonfinite count {values.dtype}/"
                             f"{int((~np.isfinite(values)).sum())}, expected float32/0")
    mean = np.asarray(mean_list, dtype=np.float64)
    std = np.asarray(std_list, dtype=np.float64)
    if mean.shape != (21,) or std.shape != (21,) or np.any(std <= 0):
        raise ValueError(f"{name}: shared train-only raw scaler shape/positivity differs")
    windows = {
        "lowpass": ((control.astype(np.float64) - mean) / std).astype(np.float32),
        "raw": ((raw.astype(np.float64) - mean) / std).astype(np.float32),
    }
    if any(not np.isfinite(value).all() for value in windows.values()):
        raise ValueError(f"{name}: normalized B10 gate waveform is nonfinite")
    base_artifact = _loaded_base(Path(artifact_s))
    outputs: dict[str, dict[str, np.ndarray]] = {
        "source": {"psirz": source_psi, "q95": source_q95, "betaN": source_beta}
    }
    for label in ARMS[1:]:
        arm, _, seed_text = label.partition("_seed")
        model = _loaded_model(Path(model_root_s) / f"{label}.pt", cfg, params_sha)
        residual_parts: list[np.ndarray] = []
        with torch.no_grad():
            for lo in range(0, frames, 512):
                hi = min(lo + 512, frames)
                rates_full = np.broadcast_to(rates, (frames, 21))
                residual = model(
                    torch.from_numpy(np.array(frame84[lo:hi], copy=True)),
                    torch.from_numpy(np.array(base52[lo:hi], copy=True)),
                    torch.from_numpy(np.array(windows[arm][lo:hi], copy=True)),
                    torch.from_numpy(np.array(mask[lo:hi], copy=True)),
                    torch.from_numpy(np.array(rates_full[lo:hi], copy=True)),
                )
                residual_parts.append(residual.numpy())
        pred = np.concatenate(residual_parts).astype(np.float32)
        if pred.shape != (frames, 52):
            raise ValueError(f"{name}: {label} residual {pred.shape}, expected ({frames},52)")
        outputs[label] = decode_residual(source_psi, source_q95, source_beta,
                                         base52, pred, base_artifact)
        if int(seed_text) not in (0, 1):
            raise ValueError(f"{name}: {label} seed {seed_text}, expected 0 or 1")
    return {"shot": name, "frames": frames, "outputs": outputs}


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    if path.exists():
        raise FileExistsError(f"{path}: B10 gate prediction already sealed")
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".npz")
    try:
        with os.fdopen(fd, "wb") as stream:
            np.savez_compressed(stream, **arrays)  # type: ignore[arg-type]
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _score(prediction: Path, files: list[str], data: Path, scorer: Path,
           metrics: Path, jobs: int, cwd: Path) -> dict[str, Any]:
    files_path = metrics.with_suffix(".files.json")
    write_json(files_path, {"files": files})
    subprocess.run([
        sys.executable, "-m", "tools.historical_compare", "score",
        "--prediction", str(prediction), "--files-json", str(files_path),
        "--data-dir", str(data), "--metrics-json", str(metrics),
        "--jobs", str(jobs), "--scorer-root", str(scorer),
    ], cwd=cwd, env={**os.environ, "PYTHONPATH": str(cwd)}, check=True)
    scored = json.loads(metrics.read_text())
    if scored["files"] != files:
        raise ValueError(f"{metrics}: native scorer files differ from sealed order")
    return scored["results"]["model"]


def run(campaign: Path, predict_jobs: int, score_jobs: int) -> None:
    """Score four fixed aggregate arms and their two frozen rate-class subsets."""
    started = time.perf_counter()
    campaign = campaign.resolve()
    if ROOT.resolve() != (campaign / "source").resolve():
        raise ValueError(f"gate code imported from {ROOT}, expected {campaign / 'source'}")
    spec = json.loads(SPEC.read_text())
    if file_hash(Path(__file__)) != spec["gate_helper_sha256"]:
        raise ValueError(f"{campaign}: frozen B10 gate helper SHA differs")
    params_path = campaign / "source/baselines/b10-two-rate-info-gate.params.yaml"
    if file_hash(params_path) != spec["effective_params_sha256"]:
        raise ValueError(f"{params_path}: frozen B10 model YAML SHA differs")
    params = yaml.safe_load(params_path.read_text())
    manifest = json.loads((campaign / "manifest.json").read_text())
    for relative, expected in manifest["source_sha256"].items():
        path = campaign / "source" / relative
        if file_hash(path) != expected:
            raise ValueError(f"{path}: frozen gate source SHA differs")
    output = campaign / "gate_result.json"
    if output.exists():
        raise FileExistsError(f"{output}: B10 gate already scored")
    assembled = json.loads((campaign / "assembly.json").read_text())
    sidecars = json.loads((campaign / "waveform_sidecars_ready.json").read_text())
    targets = json.loads((campaign / "targets_and_base_ready.json").read_text())
    support = json.loads((Path(manifest["origin_root"])
                          / spec["support_audit"]["path"]).read_text())
    names = manifest["gate_files"]
    if (len(names) != 200 or targets["gate_frames"] != 43142
            or sidecars["gate_frames"] != 43142
            or names != [row["shot"] for row in targets["entries"][1355:]]):
        raise ValueError(f"{campaign}: gate200 source/sidecar/target identities differ")
    origin = Path(manifest["origin_root"])
    base_artifact = origin / spec["base_artifact"]["path"]
    if file_hash(base_artifact) != spec["base_artifact"]["sha256"]:
        raise ValueError(f"{base_artifact}: fold0 artifact SHA differs")
    model_paths = [campaign / "models" / f"{label}.pt" for label in ARMS[1:]]
    for path in model_paths:
        record = json.loads(path.with_suffix(".json").read_text())
        if (file_hash(path) != record["artifact_sha256"]
                or record["assembly_sha256"] != file_hash(campaign / "assembly.json")
                or record["steps"] != int(params["train"]["max_steps"])
                or record["params_sha256"] != spec["effective_params_sha256"]):
            raise ValueError(f"{path}: frozen model artifact/steps/assembly differ")
    tasks = [
        (name, support["per_shot"][1355 + i]["frames"],
         sidecars["entries"][1355 + i]["sidecar_path"],
         targets["entries"][1355 + i]["output_path"], str(base_artifact),
         str(campaign / "models"), assembled["raw_channel_mean_train_only"],
         assembled["raw_channel_std_train_only"], params["model"],
         spec["effective_params_sha256"])
        for i, name in enumerate(names)
    ]
    arrays: dict[str, dict[str, np.ndarray]] = {arm: {} for arm in ARMS}
    try:
        for i, item in enumerate(pimap(_predict_one, tasks, predict_jobs)):
            if item["shot"] != names[i] or item["frames"] != tasks[i][1]:
                raise ValueError(f"{names[i]}: B10 gate prediction order/frames differ")
            for arm in ARMS:
                for field in ("psirz", "q95", "betaN"):
                    key = f"shot_{i:04d}_{field}"
                    value = np.asarray(item["outputs"][arm][field])
                    if value.dtype != np.float32 or not np.isfinite(value).all():
                        raise ValueError(f"{names[i]}: {arm}/{field} dtype/finiteness "
                                         f"{value.dtype}, expected float32/finite")
                    arrays[arm][key] = value
            if (i + 1) % 25 == 0:
                print(f"B10 gate predictions sealed in memory {i + 1}/200 shots", flush=True)
    finally:
        release()
    pred_dir = campaign / "gate_predictions"
    pred_dir.mkdir(exist_ok=False)
    for arm in ARMS:
        path = pred_dir / f"{arm}.npz"
        _atomic_npz(path, arrays[arm])
        write_json(path.with_suffix(".json"), {
            "files": names, "prediction_sha256": file_hash(path),
            "arm": arm, "source_artifact_sha256": spec["base_artifact"]["sha256"],
            "gate_target_columns_requested": []})
    prediction_seconds = time.perf_counter() - started
    scorer = origin / spec["native_scorer"]["root"]
    if file_hash(scorer / "local_score.py") != spec["native_scorer"][
            "local_score_py_sha256"]:
        raise ValueError(f"{scorer}: native scorer source SHA differs")
    data = Path(manifest["data_dir"])
    scores: dict[str, dict[str, Any]] = {}
    scored_seconds: dict[str, float] = {}
    for arm in ARMS:
        t0 = time.perf_counter()
        scores[arm] = _score(pred_dir / f"{arm}.npz", names, data, scorer,
                             pred_dir / f"{arm}.metrics.json", score_jobs, campaign / "source")
        scored_seconds[arm] = time.perf_counter() - t0
    class_scores: dict[str, dict[str, dict[str, Any]]] = {}
    for label in ("0.05", "0.5"):
        selected = support["sampling_rate_classes"][label]["gate_files"]
        selected_set = set(selected)
        selected_indices = [i for i, name in enumerate(names) if name in selected_set]
        if [names[i] for i in selected_indices] != selected:
            raise ValueError(f"{label}: gate sampling-class filename order differs")
        class_scores[label] = {}
        for arm in ARMS:
            subset: dict[str, np.ndarray] = {}
            for j, original_index in enumerate(selected_indices):
                for field in ("psirz", "q95", "betaN"):
                    subset[f"shot_{j:04d}_{field}"] = arrays[arm][
                        f"shot_{original_index:04d}_{field}"]
            path = pred_dir / f"{arm}_rate_{label}.npz"
            _atomic_npz(path, subset)
            write_json(path.with_suffix(".json"), {
                "files": selected, "prediction_sha256": file_hash(path),
                "arm": arm, "sampling_rate_ms": label})
            t0 = time.perf_counter()
            class_scores[label][arm] = _score(
                path, selected, data, scorer,
                pred_dir / f"{arm}_rate_{label}.metrics.json", score_jobs,
                campaign / "source")
            scored_seconds[f"{arm}_rate_{label}"] = time.perf_counter() - t0
    noise = abs(scores["lowpass_seed1"]["S"] - scores["lowpass_seed0"]["S"])
    delta = scores["raw_seed0"]["S"] - scores["lowpass_seed0"]["S"]
    class_delta = {label: class_scores[label]["raw_seed0"]["S"] -
                   class_scores[label]["lowpass_seed0"]["S"]
                   for label in ("0.05", "0.5")}
    failure_ok = (scores["raw_seed0"]["lcfs_fail_frac"] <=
                  scores["lowpass_seed0"]["lcfs_fail_frac"] and
                  scores["raw_seed0"]["cons_fail_frac"] <=
                  scores["lowpass_seed0"]["cons_fail_frac"])
    gate = bool(delta >= 0.0026 and delta >= 2 * noise and
                all(value > 0 for value in class_delta.values()) and failure_ok)
    write_json(output, {
        "classification": "B10_matched_raw_vs_EFIT_lowpass_residual_head_native_gate200",
        "campaign_manifest_sha256": file_hash(campaign / "manifest.json"),
        "spec_sha256": file_hash(SPEC),
        "shots": 200, "frames": 43142,
        "aggregate_native": scores,
        "sampling_rate_class_native": class_scores,
        "sampling_rate_class_counts": support["sampling_rate_classes"],
        "raw_minus_lowpass_seed0_S": delta,
        "lowpass_A_A_abs_delta_S": noise,
        "two_times_noise_threshold": 2 * noise,
        "class_raw_minus_lowpass_seed0_S": class_delta,
        "raw_no_increased_native_failures": failure_ok,
        "required_primary_delta_S": 0.0026,
        "investment_gate_passed": gate,
        "prediction_sha256": {arm: file_hash(pred_dir / f"{arm}.npz") for arm in ARMS},
        "gate_target_columns_requested": [],
        "reserved_tail70_shots_accessed": 0,
        "prediction_build_wall_seconds": prediction_seconds,
        "native_scorer_wall_seconds": scored_seconds,
        "total_wall_seconds": time.perf_counter() - started,
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--predict-jobs", type=int, default=4)
    parser.add_argument("--score-jobs", type=int, default=24)
    args = parser.parse_args()
    if not 1 <= args.predict_jobs <= 8 or not 1 <= args.score_jobs <= 24:
        raise ValueError(f"jobs prediction/scoring {args.predict_jobs}/{args.score_jobs}, "
                         "expected 1..8/1..24")
    run(args.campaign, args.predict_jobs, args.score_jobs)


if __name__ == "__main__":
    main()
