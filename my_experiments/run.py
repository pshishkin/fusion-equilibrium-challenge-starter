"""Prepare immutable DIII-D campaigns, execute/resume them, and compare ensembles."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import time
from contextlib import suppress
from pathlib import Path
from typing import Any

import yaml

from my_experiments.data_check import check_dataset
from toolkit.records import file_hash, write_json

ROOT = Path(__file__).resolve().parent.parent
PROFILES = ROOT / "experiments" / "protocols.json"
DATA = ROOT.parent / "downloaded_huggingface" / "hf_dataset"


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected an object, got {type(value).__name__}")
    return value


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def environment() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "executable": sys.executable,
        "platform": platform.platform(),
        "numerical_environment": {
            k: os.environ.get(k)
            for k in (
                "FUSION_AMP",
                "FUSION_COEF_LAMBDA",
                "FUSION_COEF_LAMBDA_FROM",
                "CUDA_VISIBLE_DEVICES",
                "CUBLAS_WORKSPACE_CONFIG",
                "PYTHONHASHSEED",
            )
        },
        "packages": dict(
            sorted((d.metadata["Name"], d.version) for d in importlib.metadata.distributions())
        ),
    }


def check_device(params: Path) -> dict[str, Any]:
    doc = yaml.safe_load(params.read_text())
    devices = {
        name: cfg["device"]
        for name, cfg in doc["models"].items()
        if cfg["enabled"] and "device" in cfg
    }
    import torch

    if "cuda" in devices.values() and not torch.cuda.is_available():
        raise RuntimeError(
            f"{params}: device=cuda requested by {devices}, available CUDA devices=0"
        )
    return {
        "devices": devices,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cpu_count": os.cpu_count(),
    }


def source_files() -> list[Path]:
    files = list(ROOT.glob("*.py"))
    files += [
        ROOT / name
        for name in (
            ".python-version", "pyproject.toml", "uv.lock", "params.yaml", "Makefile",
            "experiments/protocols.json",
        )
    ]
    for directory in ("my_experiments", "toolkit", "fusion_scoring", "mast", "solver"):
        files += list((ROOT / directory).rglob("*.py"))
    files += [ROOT / "my_experiments/diii_d_dataset.json"]
    files += list((ROOT / "fusion_scoring/masks").glob("*.npz"))
    return sorted(files)


def prepare(spec_path: Path, output: Path, jobs: int | None = None) -> Path:
    """Freeze all arms before any fit; later workspace edits cannot change the queue."""
    from my_experiments.models import load_params

    spec = read_json(spec_path)
    if jobs is not None:
        spec["jobs"] = jobs
    allowed = {"name", "params", "profile", "salts", "arms", "jobs"}
    if set(spec) != allowed:
        raise ValueError(f"{spec_path}: keys {sorted(spec)}, expected {sorted(allowed)}")
    if not re.fullmatch(r"[a-z0-9_-]+", spec["name"]):
        raise ValueError(f"{spec_path}: invalid campaign name {spec['name']!r}")
    if not isinstance(spec["jobs"], int) or spec["jobs"] < 1:
        raise ValueError(f"{spec_path}: jobs={spec['jobs']}, expected positive integer")
    profiles = read_json(PROFILES)
    if spec["profile"] not in profiles:
        raise ValueError(f"{spec_path}: profile {spec['profile']!r}, expected {sorted(profiles)}")
    profile = profiles[spec["profile"]]
    required_profile = {"train", "validation", "score", "purpose"}
    extra_profile = set(profile) - required_profile
    if not required_profile <= set(profile) or extra_profile - {"fit_mode"}:
        raise ValueError(
            f"{PROFILES}: profile {spec['profile']!r} keys {sorted(profile)}, "
            f"expected {sorted(required_profile)} with optional fit_mode"
        )
    if "fit_mode" in profile and profile["fit_mode"] != "fixed_steps":
        raise ValueError(
            f"{PROFILES}: profile {spec['profile']!r} fit_mode={profile['fit_mode']!r}, "
            "expected fixed_steps"
        )
    if not spec["salts"] or len(set(spec["salts"])) != len(spec["salts"]):
        raise ValueError(f"{spec_path}: salts must be nonempty and unique")
    if any(type(s) is not int or s < 0 for s in spec["salts"]):
        raise ValueError(f"{spec_path}: salts must be nonnegative integers")
    names = [a["name"] for a in spec["arms"]]
    if not names or len(set(names)) != len(names):
        raise ValueError(f"{spec_path}: arm names must be nonempty and unique")
    params_path = (spec_path.parent / spec["params"]).resolve()
    runs = []
    for salt in spec["salts"]:
        for arm in spec["arms"]:
            if set(arm) != {"name", "only", "set", "score_models"}:
                raise ValueError(f"{spec_path}: invalid keys for arm {arm}")
            if not re.fullmatch(r"[a-z0-9_-]+", arm["name"]):
                raise ValueError(f"{spec_path}: invalid arm name {arm['name']!r}")
            if not arm["score_models"]:
                raise ValueError(f"{spec_path}: {arm['name']} has no score_models")
            cfg = load_params(params_path, salt, arm["only"], arm["set"])
            if not isinstance(arm["score_models"], list) or any(
                not isinstance(model, str) for model in arm["score_models"]
            ):
                raise ValueError(f"{spec_path}: score_models must be a list of model expressions")
            for model in arm["score_models"]:
                if model.startswith("salts:"):
                    raise ValueError(f"{spec_path}: use run compare for external artifacts")
                leaves = {
                    part.partition("*")[0]
                    for block in model.split(",")
                    for part in block.split("=")[-1].split("+")
                }
                unknown = leaves - {*cfg.models, "ensemble"}
                if unknown:
                    raise ValueError(f"{spec_path}: score_models names disabled/unknown {unknown}")
            runs.append(
                {
                    "name": f"{arm['name']}_s{salt}",
                    "salt": salt,
                    "arm": arm["name"],
                    "params_yaml": cfg.effective_yaml,
                    "score_models": arm["score_models"],
                }
            )
    hashes = {str(p.relative_to(ROOT)): file_hash(p) for p in source_files()}
    identity = {
        "spec": spec,
        "protocol": profile,
        "source": hashes,
        "environment": environment(),
        "runs": runs,
    }
    campaign = output.resolve() / f"{spec['name']}-{digest(identity)[:16]}"
    if campaign.exists():
        if read_json(campaign / "campaign.json")["identity"] != identity:
            raise ValueError(f"{campaign}: existing identity differs from requested identity")
        verify_source(campaign)
        return campaign
    campaign.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=campaign.parent, prefix=".preparing-") as staging:
        stage = Path(staging) / "campaign"
        stage.mkdir()
        for rel in hashes:
            target = stage / "source" / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / rel, target)
            if file_hash(target) != hashes[rel]:
                raise ValueError(f"{rel}: source changed while preparing {campaign}; prepare again")
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
        write_json(
            stage / "campaign.json",
            {
                "schema_version": 1,
                "identity": identity,
                "git_head": revision,
                "source_hash": digest(hashes),
            },
        )
        for run in runs:
            folder = stage / run["name"]
            folder.mkdir()
            (folder / "params.yaml").write_text(run["params_yaml"])
        # Preparation never publishes a half-written campaign under its stable identity.
        with (campaign.parent / ".prepare.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if campaign.exists():
                if verify_source(campaign)["identity"] != identity:
                    raise ValueError(f"{campaign}: concurrently prepared identity differs")
            else:
                stage.rename(campaign)
    return campaign


def verify_source(campaign: Path) -> dict[str, Any]:
    meta = read_json(campaign / "campaign.json")
    for rel, expected in meta["identity"]["source"].items():
        got = file_hash(campaign / "source" / rel)
        if got != expected:
            raise ValueError(f"{campaign}/source/{rel}: SHA-256 {got}, expected {expected}")
    for run in meta["identity"]["runs"]:
        path = campaign / run["name"] / "params.yaml"
        if path.read_text() != run["params_yaml"]:
            raise ValueError(f"{path}: effective configuration differs from frozen campaign")
    return meta


def completed(folder: Path, source_hash: str | None = None) -> bool:
    path = folder / "complete.json"
    if not path.exists():
        return False
    record = read_json(path)
    expected_names = {"baseline.joblib", "metrics.json", "params.yaml"}
    if set(record["sha256"]) != expected_names:
        raise ValueError(f"{path}: completion hashes must cover {sorted(expected_names)}")
    if source_hash is not None and record.get("source_hash") != source_hash:
        raise ValueError(f"{path}: completed source differs from frozen campaign {source_hash}")
    for name, expected in record["sha256"].items():
        got = file_hash(folder / name)
        if got != expected:
            raise ValueError(f"{folder / name}: SHA-256 {got}, completed record expects {expected}")
    return True


def process_scope() -> dict[str, str]:
    """A PID is meaningful only within the host boot and PID namespace that issued it."""
    return {
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "pid_namespace": os.readlink("/proc/self/ns/pid"),
    }


def active_group(folder: Path) -> bool:
    path = folder / "status.json"
    if not path.exists():
        return False
    status = read_json(path)
    if status.get("state") != "running":
        return False
    recorded = status.get("process_scope")
    current = process_scope()
    if recorded is None:
        raise RuntimeError(f"{folder}: running record has no process scope; inspect its workers")
    if recorded["boot_id"] != current["boot_id"]:
        return False
    if recorded["pid_namespace"] != current["pid_namespace"]:
        raise RuntimeError(
            f"{folder}: worker PID namespace {recorded['pid_namespace']}, "
            f"current {current['pid_namespace']}; inspect/execute in the worker's namespace"
        )
    try:
        os.killpg(status["pgid"], 0)
    except ProcessLookupError:
        return False
    return True


def stream_process(
    command: list[str], cwd: Path, env: dict[str, str], log: Path, status: Path
) -> None:
    """Tee live output; terminate the entire worker group on interruption."""
    latest = ROOT / "logs/latest.log"
    latest.parent.mkdir(exist_ok=True)
    latest.unlink(missing_ok=True)
    latest.symlink_to(log.resolve())
    started = time.time()
    with log.open("a") as out:
        child = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        write_json(
            status,
            {
                "state": "running",
                "pid": child.pid,
                "pgid": child.pid,
                "process_scope": process_scope(),
                "started_unix": started,
                "command": command,
                "log": str(log),
            },
        )
        try:
            assert child.stdout is not None
            for line in child.stdout:
                out.write(line)
                out.flush()
                print(line, end="", flush=True)
            code = child.wait()
            if code:
                raise subprocess.CalledProcessError(code, command)
        except BaseException:
            with suppress(ProcessLookupError):
                os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            write_json(
                status,
                {
                    "state": "failed",
                    "wall_s": time.time() - started,
                    "returncode": child.returncode,
                    "log": str(log),
                },
            )
            raise
        finally:
            if child.stdout is not None:
                child.stdout.close()
    write_json(status, {"state": "scored", "wall_s": time.time() - started, "log": str(log)})


def execute(campaign: Path, data: Path) -> None:
    campaign, data = campaign.resolve(), data.resolve()
    with (campaign / ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        meta = verify_source(campaign)
        identity = meta["identity"]
        if environment() != identity["environment"]:
            raise ValueError(f"{campaign}: Python environment changed; prepare a new campaign")
        dataset = check_dataset(data)
        for run in identity["runs"]:
            folder = campaign / run["name"]
            if completed(folder, meta["source_hash"]):
                print(f"Verified completed run: {folder.name}", flush=True)
                continue
            if active_group(folder):
                raise RuntimeError(
                    f"{folder}: recorded worker group is still alive; refusing a second fit"
                )
            device = check_device(folder / "params.yaml")
            write_json(folder / "preflight.json", {"dataset": dataset, "environment": device})
            protocol = identity["protocol"]
            command = [
                sys.executable,
                "-u",
                "my_experiments/train_eval.py",
                protocol["train"],
                protocol["validation"],
                str(protocol["score"]),
                "--params",
                str(folder / "params.yaml"),
                *(["--fit-mode", protocol["fit_mode"]] if "fit_mode" in protocol else []),
                "--jobs",
                str(identity["spec"]["jobs"]),
                "--local-data-dir",
                str(data),
                "--metrics-json",
                str(folder / "metrics.json"),
                "--score-models",
                *run["score_models"],
            ]
            env = dict(
                os.environ,
                FUSION_ARTIFACT=str(folder / "baseline.joblib"),
                FUSION_RUN_ID=f"{campaign.name}/{folder.name}",
                FUSION_SOURCE_SHA256=meta["source_hash"],
                FUSION_SHOT_CACHE=str(ROOT / ".shot_cache"),
                FUSION_AUX_CACHE=str(ROOT / ".aux_targets"),
                OMP_NUM_THREADS="4",
                OPENBLAS_NUM_THREADS="4",
                MKL_NUM_THREADS="4",
            )
            # A failed attempt is retained; never mistake its previous metrics for success.
            (folder / "metrics.json").unlink(missing_ok=True)
            log = folder / f"{time.time_ns()}-run.log"
            stream_process(command, campaign / "source", env, log, folder / "status.json")
            metrics = read_json(folder / "metrics.json")
            if set(metrics["results"]) != set(run["score_models"]):
                raise ValueError(
                    f"{folder}: scored {list(metrics['results'])}, expected {run['score_models']}"
                )
            write_json(
                folder / "complete.json",
                {
                    "sha256": {
                        n: file_hash(folder / n)
                        for n in ("baseline.joblib", "metrics.json", "params.yaml")
                    },
                    "source_hash": meta["source_hash"],
                    "dataset": dataset,
                },
            )
            print(f"Completed: {folder}", flush=True)


def doctor(data: Path, params: Path) -> int:
    report: dict[str, Any] = {"environment": environment(), "errors": []}
    for label, fn in (
        ("dataset", lambda: check_dataset(data)),
        ("device", lambda: check_device(params)),
    ):
        try:
            report[label] = fn()
        except (ValueError, RuntimeError, OSError, ImportError) as exc:
            report["errors"].append(f"{label}: {exc}")
    write_json(ROOT / "logs/doctor.json", report)
    print(json.dumps(report, indent=2))
    return int(bool(report["errors"]))


def summary(campaign: Path, control: str) -> dict[str, Any]:
    """Paired deltas only; do not assign an acceptance threshold after seeing them."""
    meta = verify_source(campaign)
    values: dict[str, dict[int, dict[str, Any]]] = {}
    for run in meta["identity"]["runs"]:
        folder = campaign / run["name"]
        if completed(folder, meta["source_hash"]):
            values.setdefault(run["arm"], {})[run["salt"]] = read_json(folder / "metrics.json")
    if control not in values:
        raise ValueError(f"{campaign}: no completed runs for control {control!r}")
    report: dict[str, Any] = {"control": control, "comparisons": {}}
    for arm, salts in values.items():
        if arm == control:
            continue
        for model in sorted({m for v in salts.values() for m in v["results"]}):
            pairs = []
            for salt in sorted(set(salts) & set(values[control])):
                base, treat = values[control][salt], salts[salt]
                if base["files"] != treat["files"]:
                    raise ValueError(f"{campaign}: {arm} salt {salt} scored different files")
                if model not in base["results"] or model not in treat["results"]:
                    continue
                pairs.append(
                    {
                        "salt": salt,
                        "delta": treat["results"][model]["S"] - base["results"][model]["S"],
                    }
                )
            if pairs:
                deltas = [p["delta"] for p in pairs]
                report["comparisons"][f"{arm}/{model}"] = {
                    "pairs": pairs,
                    "n": len(pairs),
                    "mean": statistics.mean(deltas),
                    "se": statistics.stdev(deltas) / len(deltas) ** 0.5
                    if len(deltas) > 1
                    else None,
                    "positive": sum(d > 0 for d in deltas),
                }
    write_json(campaign / "summary.json", report)
    return report


def compare(args: argparse.Namespace) -> None:
    """Use the real scorer for candidate, baseline blend, and augmented blend together."""
    data = args.data_dir.resolve()
    check_dataset(data)
    if len(args.baseline) != len(args.weights):
        raise ValueError(f"baseline paths={len(args.baseline)}, weights={len(args.weights)}")
    if any(not 0 < w < float("inf") for w in [*args.weights, args.candidate_weight]):
        raise ValueError("All artifact weights must be finite and positive")
    folder = args.output.resolve() / f"compare-{time.time_ns()}"
    folder.mkdir(parents=True)
    inputs = []
    for i, path in enumerate([*args.baseline, args.candidate]):
        dest = folder / f"baseline_m{i}.joblib"
        shutil.copyfile(path, dest)
        inputs.append({"path": str(path.resolve()), "sha256": file_hash(dest)})
    base = "+".join(f"m{i}*{w}" for i, w in enumerate(args.weights))
    candidate = f"m{len(args.baseline)}*{args.candidate_weight}"
    specs = {
        "baseline": f"salts:{base}",
        "candidate": f"salts:{candidate}",
        "augmented": f"salts:{base}+{candidate}",
    }
    hashes = {str(p.relative_to(ROOT)): file_hash(p) for p in source_files()}
    for rel in hashes:
        dest = folder / "source" / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / rel, dest)
        if file_hash(dest) != hashes[rel]:
            raise ValueError(f"{rel}: source changed while preparing comparison")
    write_json(
        folder / "request.json",
        {
            "inputs": inputs,
            "specs": specs,
            "source": hashes,
            "share": args.share,
            "holdout": args.on_holdout,
            "environment": environment(),
        },
    )
    command = [
        sys.executable,
        "-u",
        "my_experiments/evaluate.py",
        "--share",
        str(args.share),
        "--jobs",
        str(args.jobs),
        "--local-data-dir",
        str(data),
        "--metrics-json",
        str(folder / "metrics.json"),
        "--models",
        *specs.values(),
        *(["--on-holdout"] if args.on_holdout else []),
    ]
    env = dict(
        os.environ,
        FUSION_ARTIFACT=str(folder / "baseline_m0.joblib"),
        FUSION_SHOT_CACHE=str(ROOT / ".shot_cache"),
        OMP_NUM_THREADS="4",
        OPENBLAS_NUM_THREADS="4",
        MKL_NUM_THREADS="4",
    )
    stream_process(command, folder / "source", env, folder / "compare.log", folder / "status.json")
    scores = read_json(folder / "metrics.json")
    results = {role: scores["results"][spec] for role, spec in specs.items()}
    write_json(
        folder / "comparison.json",
        {
            "results": results,
            "n_shots": scores["n_shots"],
            "files": scores["files"],
            "candidate_minus_baseline": results["candidate"]["S"] - results["baseline"]["S"],
            "augmented_minus_baseline": results["augmented"]["S"] - results["baseline"]["S"],
        },
    )
    print(f"Comparison: {folder / 'comparison.json'}")


def compare_replacement(args: argparse.Namespace) -> None:
    """Score a single-member replacement against its unchanged two-member ensemble."""
    import joblib

    baseline_path = args.baseline.resolve()
    candidate_path = args.candidate.resolve()
    for path in (baseline_path, candidate_path):
        if not completed(path.parent):
            raise ValueError(f"{path}: no verified campaign completion record")
        if path.name != "baseline.joblib":
            raise ValueError(f"{path}: expected campaign artifact baseline.joblib")
    baseline_hash = file_hash(baseline_path)
    if baseline_hash != args.baseline_sha256:
        raise ValueError(
            f"{baseline_path}: SHA-256 {baseline_hash}, expected {args.baseline_sha256}"
        )
    baseline = joblib.load(baseline_path)
    candidate = joblib.load(candidate_path)
    reference = args.reference_member
    companion = args.companion_member
    selected = args.candidate_member
    if reference == companion:
        raise ValueError(f"{baseline_path}: reference and companion both name {reference!r}")
    if set(baseline["models"]) != {reference, companion}:
        raise ValueError(
            f"{baseline_path}: fitted members {sorted(baseline['models'])}, "
            f"expected {sorted({reference, companion})}"
        )
    if selected not in candidate["models"]:
        raise ValueError(
            f"{candidate_path}: no fitted member {selected!r}; "
            f"available {sorted(candidate['models'])}"
        )
    if set(candidate["models"]) != {selected} and candidate_path != baseline_path:
        raise ValueError(
            f"{candidate_path}: fitted members {sorted(candidate['models'])}, "
            f"expected only {selected!r}"
        )
    if any(not 0 < w < float("inf") for w in (args.reference_weight, args.companion_weight)):
        raise ValueError("Reference and companion weights must be finite and positive")
    requested_weights = {reference: args.reference_weight, companion: args.companion_weight}
    actual_weights = baseline["ensemble"]
    if set(actual_weights) != set(requested_weights) or any(
        abs(actual_weights[k] - requested_weights[k]) > 1e-12 for k in requested_weights
    ):
        raise ValueError(
            f"{baseline_path}: ensemble weights {actual_weights}, expected {requested_weights}"
        )
    identity_keys = (
        "dataset_identity",
        "split_salt",
        "train_files",
        "val_files",
        "holdout_files",
        "train_share",
        "train_frame_share",
        "val_share",
        "val_frame_share",
        "holdout_share",
    )
    for key in identity_keys:
        if baseline[key] != candidate[key]:
            raise ValueError(
                f"{candidate_path}: {key} differs from {baseline_path}: "
                f"{str(candidate[key])[:120]} against {str(baseline[key])[:120]}"
            )
    data = args.data_dir.resolve()
    check_dataset(data)
    folder = args.output.resolve() / f"replace-{time.time_ns()}"
    folder.mkdir(parents=True)
    members = (
        ("m0", baseline_path, baseline, None),
        ("m1", candidate_path, candidate, selected),
        ("m2", baseline_path, baseline, companion),
        ("m3", baseline_path, baseline, reference),
    )
    projected = []
    for name, source_path, source_artifact, member in members:
        dest = folder / f"baseline_{name}.joblib"
        if member is None:
            shutil.copyfile(source_path, dest)
        else:
            view = dict(source_artifact)
            view["models"] = {member: source_artifact["models"][member]}
            view["ensemble"] = {member: 1.0}
            view["projection"] = {
                "kind": "member_view_not_refit",
                "source_path": str(source_path),
                "source_sha256": file_hash(source_path),
                "selected_member": member,
            }
            joblib.dump(view, dest)
            if view["source_sha256"] != source_artifact["source_sha256"]:
                raise ValueError(f"{dest}: projected model source hash changed")
            if (
                view["train_files"] != source_artifact["train_files"]
                or view["val_files"] != source_artifact["val_files"]
            ):
                raise ValueError(f"{dest}: projected model changed fit or validation filenames")
        projected.append(
            {
                "alias": name,
                "source": str(source_path),
                "source_sha256": file_hash(source_path),
                "selected_member": member or "original_ensemble",
                "projection_sha256": file_hash(dest),
                "source_fit_sha256": source_artifact["source_sha256"],
            }
        )
    ref_weight = args.reference_weight
    companion_weight = args.companion_weight
    specs = {
        "baseline_native": "salts:m0",
        "candidate": "salts:m1",
        "control_decoded": f"salts:m3*{ref_weight}+m2*{companion_weight}",
        "replacement": f"salts:m1*{ref_weight}+m2*{companion_weight}",
    }
    source_campaign = args.source_campaign.resolve()
    source_meta = verify_source(source_campaign)
    if candidate_path != baseline_path and candidate["source_sha256"] != source_meta["source_hash"]:
        raise ValueError(
            f"{candidate_path}: fitted source hash {candidate['source_sha256']}, "
            f"evaluator snapshot {source_campaign} hash {source_meta['source_hash']}"
        )
    source_root = source_campaign / "source"
    hashes = source_meta["identity"]["source"]
    controller_hash = file_hash(Path(__file__))
    for rel in hashes:
        dest = folder / "source" / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_root / rel, dest)
        if file_hash(dest) != hashes[rel]:
            raise ValueError(f"{rel}: source changed while preparing comparison")
    write_json(
        folder / "request.json",
        {
            "baseline_sha256_expected": args.baseline_sha256,
            "inputs": projected,
            "effective_member_weights": {
                "baseline_native": actual_weights,
                "control_decoded": {reference: ref_weight, companion: companion_weight},
                "candidate": {selected: 1.0},
                "replacement": {
                    f"{reference}<-{selected}": ref_weight,
                    companion: companion_weight,
                },
            },
            "specs": specs,
            "source": hashes,
            "source_campaign": str(source_campaign),
            "source_hash": source_meta["source_hash"],
            "controller_run_sha256": controller_hash,
            "share": args.share,
            "environment": environment(),
        },
    )
    shutil.copyfile(Path(__file__), folder / "controller_run.py")
    if file_hash(folder / "controller_run.py") != controller_hash:
        raise ValueError(f"{folder}: comparison controller changed while being copied")
    command = [
        sys.executable,
        "-u",
        "my_experiments/evaluate.py",
        "--share",
        str(args.share),
        "--jobs",
        str(args.jobs),
        "--local-data-dir",
        str(data),
        "--metrics-json",
        str(folder / "metrics.json"),
        "--models",
        *specs.values(),
    ]
    env = dict(
        os.environ,
        FUSION_ARTIFACT=str(folder / "baseline_m0.joblib"),
        FUSION_SHOT_CACHE=str(ROOT / ".shot_cache"),
        OMP_NUM_THREADS="4",
        OPENBLAS_NUM_THREADS="4",
        MKL_NUM_THREADS="4",
    )
    stream_process(command, folder / "source", env, folder / "compare.log", folder / "status.json")
    scores = read_json(folder / "metrics.json")
    baseline_metrics = read_json(baseline_path.parent / "metrics.json")
    candidate_metrics = read_json(candidate_path.parent / "metrics.json")
    if scores["files"] != baseline_metrics["files"]:
        raise ValueError(f"{folder}: scored filenames differ from frozen baseline metrics")
    if scores["files"] != candidate_metrics["files"]:
        raise ValueError(f"{folder}: scored filenames differ from candidate's frozen metrics")
    results = {role: scores["results"][spec] for role, spec in specs.items()}
    if results["baseline_native"] != baseline_metrics["results"]["ensemble"]:
        raise ValueError(f"{folder}: native baseline terms differ from frozen result")
    if results["candidate"] != candidate_metrics["results"][selected]:
        raise ValueError(f"{folder}: candidate native terms differ from frozen result")
    if candidate_path == baseline_path and results["replacement"] != results["control_decoded"]:
        raise ValueError(f"{folder}: original MLP replacement differs from decoded control")
    if args.decoded_control is not None:
        control = read_json(args.decoded_control.resolve())
        if results["control_decoded"] != control["results"]["control_decoded"]:
            raise ValueError(f"{folder}: decoded control differs from {args.decoded_control}")
    write_json(
        folder / "comparison.json",
        {
            "results": results,
            "files": scores["files"],
            "n_shots": scores["n_shots"],
            "replacement_minus_decoded_control": results["replacement"]["S"]
            - results["control_decoded"]["S"],
            "candidate_minus_native_baseline": results["candidate"]["S"]
            - results["baseline_native"]["S"],
            "decoded_minus_native_baseline": results["control_decoded"]["S"]
            - results["baseline_native"]["S"],
            "request_sha256": file_hash(folder / "request.json"),
            "metrics_sha256": file_hash(folder / "metrics.json"),
        },
    )
    print(f"Replacement comparison: {folder / 'comparison.json'}")


def interrupted(signum: int, frame: Any) -> None:
    raise KeyboardInterrupt(f"Received signal {signum}")


def main() -> int:
    signal.signal(signal.SIGTERM, interrupted)
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare", help="freeze code and all effective arm configurations")
    p.add_argument("spec", type=Path)
    p.add_argument("--output", type=Path, default=ROOT / "runs")
    p.add_argument("--jobs", type=int, help="explicit override, frozen in campaign identity")
    p = sub.add_parser("execute", help="run/resume a frozen campaign")
    p.add_argument("campaign", type=Path)
    p.add_argument("--data-dir", type=Path, default=DATA)
    p = sub.add_parser(
        "doctor", help="check corpus, dependencies and the explicitly configured device"
    )
    p.add_argument("--data-dir", type=Path, default=DATA)
    p.add_argument("--params", type=Path, default=ROOT / "params.yaml")
    p = sub.add_parser("status")
    p.add_argument("campaign", type=Path)
    p = sub.add_parser("summary")
    p.add_argument("campaign", type=Path)
    p.add_argument("--control", required=True)
    p = sub.add_parser(
        "compare", help="score candidate, baseline and their blend on the same shots"
    )
    p.add_argument("--baseline", type=Path, nargs="+", required=True)
    p.add_argument("--weights", type=float, nargs="+", required=True)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--candidate-weight", type=float, required=True)
    p.add_argument("--share", type=float, required=True)
    p.add_argument("--on-holdout", action="store_true")
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--data-dir", type=Path, default=DATA)
    p.add_argument("--output", type=Path, default=ROOT / "runs")
    p = sub.add_parser("compare-replacement", help="score one fitted member in place of another")
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--baseline-sha256", required=True)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--reference-member", required=True)
    p.add_argument("--companion-member", required=True)
    p.add_argument("--candidate-member", required=True)
    p.add_argument("--source-campaign", type=Path, required=True)
    p.add_argument("--decoded-control", type=Path)
    p.add_argument("--reference-weight", type=float, required=True)
    p.add_argument("--companion-weight", type=float, required=True)
    p.add_argument("--share", type=float, required=True)
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--data-dir", type=Path, default=DATA)
    p.add_argument("--output", type=Path, default=ROOT / "runs")
    args = ap.parse_args()
    if args.command == "prepare":
        print(prepare(args.spec.resolve(), args.output, args.jobs))
    elif args.command == "execute":
        execute(args.campaign, args.data_dir)
    elif args.command == "doctor":
        return doctor(args.data_dir, args.params)
    elif args.command == "compare":
        compare(args)
    elif args.command == "compare-replacement":
        compare_replacement(args)
    elif args.command == "summary":
        print(json.dumps(summary(args.campaign, args.control), indent=2))
    else:
        meta = verify_source(args.campaign)
        for run in meta["identity"]["runs"]:
            folder = args.campaign / run["name"]
            status = read_json(folder / "status.json") if (folder / "status.json").exists() else {}
            state = (
                "complete (hashes verified)"
                if completed(folder, meta["source_hash"])
                else status.get("state", "pending")
            )
            if state == "running":
                try:
                    if not active_group(folder):
                        state = "interrupted (no worker group; execute resumes this run)"
                except RuntimeError as exc:
                    state = f"unverified running state: {exc}"
            print(f"{folder.name}: {state}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
