"""Build leakage-checked OOF residual targets and sealed input-only gate source."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from experiments import _as_psirz_stack
from my_experiments import baseline_model
from toolkit.parallel import pimap, release
from toolkit.records import file_hash, write_json
from tools.direct_contour_compact_val_predict import INPUT_COLUMNS

ROOT = Path(__file__).resolve().parent.parent
SPEC = ROOT / "experiments/b10-two-rate-information-gate-v4.json"
TARGET_COLUMNS = ["efit_psirz", "efit_q95", "efit_beta_n"]


def _atomic(path: Path, arrays: dict[str, np.ndarray]) -> None:
    if path.exists():
        raise FileExistsError(f"{path}: B10 target/prediction file already sealed")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".npz")
    try:
        with os.fdopen(fd, "wb") as stream:
            np.savez_compressed(stream, **arrays)  # type: ignore[arg-type]
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _one(task: tuple[str, str, str, str, str, int, str, str]) -> dict[str, Any]:
    name, window, data_s, artifact_s, sidecar_s, frames, expected_sha, out_s = task
    path = Path(data_s) / "data/diii_d_train" / name
    if file_hash(path) != expected_sha:
        raise ValueError(f"{path}: parquet SHA differs from frozen support audit")
    cols = INPUT_COLUMNS if window == "gate" else [*INPUT_COLUMNS, *TARGET_COLUMNS]
    row = pd.read_parquet(path, columns=cols).iloc[0]
    slim = baseline_model.slim_row(row)
    forbidden = [key for key in slim if key.startswith("efit_")
                 and key not in ("efit_times", "efit_grid_R", "efit_grid_Z")]
    if forbidden:
        raise ValueError(f"{path}: inference row contains EFIT target fields {forbidden}")
    artifact: dict[str, Any] = joblib.load(artifact_s)
    features = baseline_model.features_for_row(row)
    plan = artifact["coil"]
    selected = baseline_model.build_inputs(plan, features)
    if selected.shape != (frames, 84):
        raise ValueError(f"{path}: selected base features {selected.shape}, "
                         f"expected ({frames},84)")
    x84 = artifact["scaler"].transform(selected)
    prediction = baseline_model.predict_row(slim, "DIII-D", "mlp", Path(artifact_s))
    psi = np.asarray(prediction["psirz"], dtype=np.float64)
    q95 = np.asarray(prediction["q95"], dtype=np.float64)
    beta = np.asarray(prediction["betaN"], dtype=np.float64)
    if (psi.shape != (frames, 65, 65) or q95.shape != (frames,)
            or beta.shape != (frames,) or not np.isfinite(psi).all()
            or not np.isfinite(q95).all() or not np.isfinite(beta).all()):
        raise ValueError(f"{path}: source psi/q95/betaN shapes/finiteness "
                         f"{psi.shape}/{q95.shape}/{beta.shape}, expected "
                         f"({frames},65,65)/({frames},)/({frames},) finite")
    coil = baseline_model.coil_flux(plan, features) if plan["subtract"] else 0.0
    source_coeff = artifact["pca"].transform(psi - coil)
    if source_coeff.shape != (frames, 50):
        raise ValueError(f"{path}: source PCA coefficients {source_coeff.shape}, "
                         f"expected ({frames},50)")
    physical_base = np.column_stack([source_coeff, q95, beta])
    base_scaled = artifact["target_scaler"].transform(physical_base)
    with np.load(sidecar_s, allow_pickle=False) as saved:
        raw_shape = saved["raw"].shape
        low_shape = saved["lowpass"].shape
        mask_shape = saved["mask"].shape
        rates_shape = saved["rates"].shape
    if (raw_shape != (frames, 41, 21) or low_shape != raw_shape
            or mask_shape != (frames, 41) or rates_shape != (21,)):
        raise ValueError(f"{path}: waveform sidecar raw/control/mask/rates "
                         f"{raw_shape}/{low_shape}/{mask_shape}/{rates_shape}, "
                         f"expected ({frames},41,21)/({frames},41)/(21,)")
    arrays: dict[str, np.ndarray] = {
        "x84": np.asarray(x84, dtype=np.float32),
        "base_scaled52": np.asarray(base_scaled, dtype=np.float32),
    }
    if window == "head_train":
        truth = _as_psirz_stack(row["efit_psirz"]).astype(np.float64)
        scalars = [np.asarray(row[key], dtype=np.float64).ravel() for key in TARGET_COLUMNS[1:]]
        if (truth.shape != (frames, 65, 65) or any(value.shape != (frames,) for value in scalars)
                or not np.isfinite(truth).all()
                or any(not np.isfinite(value).all() for value in scalars)):
            raise ValueError(f"{path}: EFIT target psi/scalar frame counts or finiteness differ "
                             f"from {frames} full frames")
        truth_coeff = artifact["pca"].transform(truth - coil)
        true_vector = np.column_stack([truth_coeff, *scalars])
        target = artifact["target_scaler"].transform(true_vector) - base_scaled
        if target.shape != (frames, 52) or not np.isfinite(target).all():
            raise ValueError(f"{path}: scaled OOF residual {target.shape}/"
                             f"{int((~np.isfinite(target)).sum())}, "
                             f"expected ({frames},52)/0")
        arrays["target_scaled52"] = np.asarray(target, dtype=np.float32)
    else:
        arrays["source_psi"] = np.asarray(psi, dtype=np.float32)
        arrays["source_q95"] = np.asarray(q95, dtype=np.float32)
        arrays["source_betaN"] = np.asarray(beta, dtype=np.float32)
    output = Path(out_s) / window / f"{Path(name).stem}.npz"
    _atomic(output, arrays)
    return {"shot": name, "window": window, "frames": frames,
            "target_columns_requested": TARGET_COLUMNS if window == "head_train" else [],
            "source_parquet_sha256": expected_sha,
            "sidecar_sha256": file_hash(Path(sidecar_s)),
            "output_path": str(output), "output_sha256": file_hash(output)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=24)
    args = parser.parse_args()
    if not 1 <= args.jobs <= 24:
        raise ValueError(f"jobs={args.jobs}, expected 1..24")
    campaign = args.campaign.resolve()
    meta = json.loads((campaign / "manifest.json").read_text())
    spec = json.loads((campaign / "source/experiments/b10-two-rate-information-gate-v4.json")
                      .read_text())
    if ROOT.resolve() != (campaign / "source").resolve():
        raise ValueError(f"target producer imported from {ROOT}, "
                         f"expected {campaign / 'source'}")
    for relative, expected in meta["source_sha256"].items():
        path = campaign / "source" / relative
        if file_hash(path) != expected:
            raise ValueError(f"{path}: frozen target source SHA differs")
    if file_hash(Path(__file__)) != spec["target_helper_sha256"]:
        raise ValueError(f"{campaign}: target helper source SHA differs")
    support_path = Path(meta["origin_root"]) / spec["support_audit"]["path"]
    if file_hash(support_path) != spec["support_audit"]["sha256"]:
        raise ValueError(f"{campaign}: full input-only support SHA differs")
    support = json.loads(support_path.read_text())
    ready_path = campaign / "waveform_sidecars_ready.json"
    if not ready_path.exists():
        raise FileNotFoundError(f"{ready_path}: input-only sidecars required before targets")
    ready = json.loads(ready_path.read_text())
    if ready["shots"] != 1555 or len(ready["entries"]) != 1555:
        raise ValueError(f"{ready_path}: sidecar count {len(ready['entries'])}, expected 1555")
    output = campaign / "targets_and_base_ready.json"
    if output.exists():
        raise FileExistsError(f"{output}: targets/base predictions already recorded")
    artifact = Path(meta["origin_root"]) / spec["base_artifact"]["path"]
    if file_hash(artifact) != spec["base_artifact"]["sha256"]:
        raise ValueError(f"{artifact}: fold0 base artifact SHA differs")
    packed = joblib.load(artifact)
    names = meta["head_train_files"] + meta["gate_files"]
    if (set(names) & set(packed["train_files"])
            or packed["val_files"] != []
            or len(names) != 1555):
        raise ValueError(f"{artifact}: OOF/head/gate source overlap or counts differ")
    dataset = json.loads((campaign / "source/my_experiments/diii_d_dataset.json")
                         .read_text())
    for i, name in enumerate(names):
        sidecar = ready["entries"][i]
        if (sidecar["shot"] != name or sidecar["frames"] != support["per_shot"][i]["frames"]
                or sidecar["source_sha256"]
                != dataset["configs"]["diii_d_train"][name][1]
                or file_hash(Path(sidecar["sidecar_path"])) != sidecar["sidecar_sha256"]):
            raise ValueError(f"{name}: frozen waveform sidecar filename/frame/source/hash differs")
    tasks = [
        (name, "head_train" if i < 1355 else "gate", meta["data_dir"],
         str(artifact), ready["entries"][i]["sidecar_path"],
         support["per_shot"][i]["frames"],
         dataset["configs"]["diii_d_train"][name][1],
         str(campaign / "target_cache"))
        for i, name in enumerate(names)
    ]
    started = time.perf_counter()
    entries: list[dict[str, Any]] = []
    try:
        for i, item in enumerate(pimap(_one, tasks, args.jobs)):
            if item["shot"] != names[i] or item["frames"] != tasks[i][5]:
                raise ValueError(f"{names[i]}: target/base worker shot/frame order differs")
            if (i >= 1355 and item["target_columns_requested"]):
                raise ValueError(f"{names[i]}: sealed gate row requested EFIT targets")
            entries.append(item)
            if (i + 1) % 100 == 0:
                print(f"B10 base/target cache {i + 1}/1555 shots", flush=True)
    finally:
        release()
    write_json(output, {
        "classification": "B10_OOF_training_residuals_and_sealed_gate_base_predictions",
        "support_audit_sha256": spec["support_audit"]["sha256"],
        "waveform_sidecars_ready_sha256": file_hash(ready_path),
        "base_artifact_sha256": spec["base_artifact"]["sha256"],
        "train_shots": len(entries[:1355]),
        "train_frames": sum(item["frames"] for item in entries[:1355]),
        "gate_shots": len(entries[1355:]),
        "gate_frames": sum(item["frames"] for item in entries[1355:]),
        "gate_target_columns_requested": [],
        "entries": entries,
        "wall_seconds": time.perf_counter() - started,
    })


if __name__ == "__main__":
    main()
