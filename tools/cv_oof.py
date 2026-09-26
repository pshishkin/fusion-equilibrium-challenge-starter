#!/usr/bin/env python3
"""K-fold out-of-fold predictions: dump them once, score any blend of them natively.

    uv run python tools/cv_oof.py dump runs/cv5/fold0/baseline.joblib runs/cv5/fold0/oof.npz
    uv run python tools/cv_oof.py score runs/cv5 --blend mlp*0.5+seq*0.5 --json out.json

`dump` predicts every member of a `--kfold` artifact on that artifact's own scored fold and stores
the TARGETS (PCA coefficients plus q95 and betaN, unscaled) per frame, not the decoded maps: 54
numbers per frame instead of 4225, so all five folds of every member fit in memory. The maps are
rebuilt at scoring time with the fold's own PCA and coil field.

`score` decodes a blend per fold, writes the scorer's `.npz` layout and runs the vendored native
scorer on that fold's files. Folds are scored separately (pooled R2 needs the whole corpus in
memory at once), and the per-fold S, their mean and its standard error are reported.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import local_score
from experiments import DEFAULT_LOCAL_DATA_DIR, HF_TRAIN_CONFIG
from my_experiments import baseline_model as B
from toolkit.parallel import pmap, resolve_jobs


def _fold_files(art: dict[str, Any]) -> list[Path]:
    if art.get("split_mode") != "kfold_validation":
        raise ValueError(f"artifact split_mode {art.get('split_mode')!r}, expected kfold_validation")
    data_dir = Path(DEFAULT_LOCAL_DATA_DIR) / "data" / art["config"]
    files = [data_dir / name for name in art["holdout_files"]]
    missing = [f for f in files if not f.exists()]
    if missing:
        raise ValueError(f"{len(missing)} scored files missing, e.g. {missing[0]}")
    seen = set(art["train_files"]) | set(art["val_files"])
    overlap = seen & set(art["holdout_files"])
    if overlap:
        raise ValueError(f"{len(overlap)} scored shots were fitted on, e.g. {sorted(overlap)[0]}")
    return files


def dump(artifact: Path, out: Path, jobs: int) -> None:
    art = joblib.load(artifact)
    if not art["holdout_files"]:
        # An all-data fit scores nothing; the marker lets a queue treat the job as complete.
        np.savez(out, all_data=np.array(True))
        print(f"{artifact}: all-data fit, no scored fold; wrote marker {out}")
        return
    files = _fold_files(art)
    X, _, S, lengths = B._read_shots(files, "reading scored fold", 1.0, jobs)
    Xs = art["scaler"].transform(B.build_inputs(art["coil"], X))
    bounds = np.concatenate([[0], np.cumsum(lengths)])
    res: dict[str, Any] = {"lengths": lengths, "files": np.array([f.name for f in files]),
                           "true_scalars": S}
    for name in art["models"]:
        parts = [B._predict_targets(art, name, Xs[bounds[i]:bounds[i + 1]])
                 for i in range(len(files))]
        res[f"targets_{name}"] = np.concatenate(parts).astype(np.float64)
        print(f"  {name}: {res[f'targets_{name}'].shape}")
    np.savez(out, **res)
    print(f"wrote {out}")


def parse_blend(spec: str) -> dict[str, float]:
    weights: dict[str, float] = {}
    for part in spec.split("+"):
        name, _, w = part.partition("*")
        weights[name] = float(w) if w else 1.0
    total = sum(weights.values())
    if not total > 0 or any(v < 0 for v in weights.values()):
        raise ValueError(f"blend {spec!r}: weights must be non-negative with a positive sum")
    return {k: v / total for k, v in weights.items()}


def _decode_one(fold_dir: Path, member: str, jobs: int, oof_name: str
                ) -> tuple[list[Path], np.ndarray, np.ndarray, np.ndarray]:
    """(files, maps (N, 65, 65), q95/betaN (N, 2), lengths) of one member on one fold."""
    art = joblib.load(fold_dir / "baseline.joblib")
    files = _fold_files(art)
    z = np.load(fold_dir / oof_name)
    if list(z["files"]) != [f.name for f in files]:
        raise ValueError(f"{fold_dir}: {oof_name} file order differs from the artifact's holdout")
    n = int(art["n_pca"])
    T = z[f"targets_{member}"]
    X, _, _, lengths = B._read_shots(files, "reading inputs", 1.0, jobs)
    if not np.array_equal(lengths, z["lengths"]):
        raise ValueError(f"{fold_dir}: frame counts differ from {oof_name}")
    maps = np.empty((len(X), 65, 65), dtype=np.float32)
    for a in range(0, len(X), 20000):
        b = min(a + 20000, len(X))
        maps[a:b] = (art["pca"].inverse_transform(T[a:b, :n]).reshape(-1, 65, 65)
                     + B.coil_flux(art["coil"], X[a:b]))
    return files, maps, T[:, n:n + 2], lengths


def decode(fold_dir: Path, weights: dict[str, float], jobs: int,
           oof_name: str = "oof.npz") -> tuple[list[Path], dict[str, np.ndarray]]:
    """The scorer's `.npz` payload for one fold and one blend of decoded maps.

    A member is `name` (from `fold_dir`) or `<campaign dir>:name` (the same fold number of
    another campaign, which must score exactly the same files).
    """
    files, maps, qb, lengths = None, 0.0, 0.0, None
    for member, w in weights.items():
        src, _, name = member.rpartition(":")
        d = Path(src) / fold_dir.name if src else fold_dir
        f, m, q, lens = _decode_one(d, name, jobs, oof_name)
        if files is None:
            files, lengths = f, lens
        elif [x.name for x in f] != [x.name for x in files]:
            raise ValueError(f"{d}: scored files differ from {fold_dir}")
        maps = maps + np.float32(w) * m
        qb = qb + w * q
    assert files is not None and lengths is not None
    bounds = np.concatenate([[0], np.cumsum(lengths)])
    out: dict[str, np.ndarray] = {}
    for i in range(len(files)):
        a, b = bounds[i], bounds[i + 1]
        out[f"shot_{i:04d}_psirz"] = np.asarray(maps[a:b], dtype=np.float32)
        out[f"shot_{i:04d}_q95"] = np.asarray(qb[a:b, 0], dtype=np.float32)
        out[f"shot_{i:04d}_betaN"] = np.asarray(qb[a:b, 1], dtype=np.float32)
    return files, out


REF_CACHE = Path(__file__).resolve().parent.parent / "runs" / "ref_cache"


def _cached_refs(files: list[Path]) -> Any:
    """Wrap local_score._map so ground-truth references are computed once per file list.

    References (LCFS contours and the seven functionals of the TRUE maps) do not depend on the
    prediction, and they are half the cost of a native pass. They are keyed by the ordered file
    names and the scorer version, and every other call of `_map` is passed straight through.
    """
    import hashlib

    key = hashlib.sha256(json.dumps([local_score.SCORING_VERSION,
                                     [f.name for f in files]]).encode()).hexdigest()[:24]
    path = REF_CACHE / f"refs_{key}.joblib"
    original = local_score._map

    def wrapped(fn: Any, tasks: list, jobs: int, desc: str) -> list:
        if fn is not local_score._ref_task:
            return original(fn, tasks, jobs, desc)
        if path.exists():
            refs = joblib.load(path)
            if len(refs) != len(tasks):
                raise ValueError(f"{path}: {len(refs)} cached references, {len(tasks)} shots")
            print(f"  references loaded from {path.name}")
            return refs
        refs = original(fn, tasks, jobs, desc)
        REF_CACHE.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        joblib.dump(refs, tmp)
        tmp.rename(path)
        return refs

    return original, wrapped


def score_fold(files: list[Path], payload: dict[str, np.ndarray], jobs: int) -> dict[str, Any]:
    original, wrapped = _cached_refs(files)
    local_score._map = wrapped
    try:
        return _score_fold(files, payload, jobs)
    finally:
        local_score._map = original


SCORED_COLUMNS = ["efit_psirz", "efit_q95", "efit_beta_n", "source", "efit_grid_R", "efit_grid_Z"]


def _read_scored_shot(path: Path) -> dict:
    """The scorer's shot dict from only the columns scoring reads. `row` (the inference inputs,
    most of a 100 MB row) is left empty: `predict` is served from memory and never reads it."""
    import pandas as pd
    row = pd.read_parquet(path, columns=SCORED_COLUMNS).iloc[0]
    return {"row": {}, "psi": local_score._as_psi_stack(row["efit_psirz"]),
            "q95": np.asarray(row["efit_q95"], dtype=np.float64),
            "betaN": np.asarray(row["efit_beta_n"], dtype=np.float64),
            "machine": str(row["source"]),
            "grid_R": np.asarray(row["efit_grid_R"], dtype=np.float64),
            "grid_Z": np.asarray(row["efit_grid_Z"], dtype=np.float64)}


def _score_fold(files: list[Path], payload: dict[str, np.ndarray], jobs: int) -> dict[str, Any]:
    """Native score of an in-memory payload.

    The scorer reads predictions through `local_score.predict`, which loads an .npz; writing a
    5 GB file single-threaded and reading it back cost minutes per pass, so `predict` is swapped
    for one that serves the same arrays from memory. The .npz path it is given is never opened.
    """
    original = local_score.predict

    def from_memory(shots: list[dict], mode: str, pred_npz: Path | None,
                    model: str | None = None) -> list[dict]:
        if len(shots) != len(files):
            raise ValueError(f"scorer loaded {len(shots)} shots, payload holds {len(files)}")
        # float32 as stored: score_shot promotes each shot to float64 inside its worker, so
        # the parent never holds a float64 copy of the whole fold (10 GB).
        return [{"psirz": payload[f"shot_{i:04d}_psirz"],
                 "q95": payload[f"shot_{i:04d}_q95"].astype(np.float64),
                 "betaN": payload[f"shot_{i:04d}_betaN"].astype(np.float64)}
                for i in range(len(shots))]

    def read_parallel(n_shots: int, skip: int, local_data_dir: Path, config: str,
                      paths: list[Path] | None = None,
                      model_specs: tuple[str | None, ...] = ()) -> list[dict]:
        # The scorer reads its shots one parquet at a time: 8 minutes a pass on a busy box.
        if paths is None or model_specs:
            raise ValueError("parallel shot reading serves explicit file lists in pred mode only")
        return pmap(_read_scored_shot, paths, resolve_jobs(jobs, len(paths)))

    original_load = local_score.load_shots_local
    local_score.predict = from_memory
    local_score.load_shots_local = read_parallel
    try:
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent.parent / "runs") as tmp:
            metrics = Path(tmp) / "metrics.json"
            local_score.main(["--source", "local", "--config", HF_TRAIN_CONFIG,
                              "--files", *map(str, files), "--pred", str(Path(tmp) / "unused.npz"),
                              "--jobs", str(jobs), "--metrics-json", str(metrics)])
            return json.loads(metrics.read_text())
    finally:
        local_score.predict = original
        local_score.load_shots_local = original_load


def score(campaign: Path, spec: str, jobs: int, out: Path | None, oof_name: str,
          only_folds: list[int] | None = None) -> None:
    weights = parse_blend(spec)
    folds = sorted(p for p in campaign.glob("fold*") if (p / oof_name).exists())
    if only_folds:
        folds = [p for p in folds if p.name in {f"fold{i}" for i in only_folds}]
    if not folds:
        raise ValueError(f"{campaign}: no fold*/{oof_name}")
    results = []
    for fold in folds:
        files, payload = decode(fold, weights, jobs, oof_name)
        m = score_fold(files, payload, jobs)
        results.append({"fold": fold.name, "n_shots": len(files), "metrics": m})
    keys = ["S", "r2_psi", "r2_qb", "dlcfs", "consistency"]
    rows = []
    for r in results:
        (res,) = r["metrics"]["results"].values()
        rows.append({k: res[k] for k in keys} | res["r2_cons_each"])
    summary = {k: float(np.mean([row[k] for row in rows])) for k in rows[0]}
    se = float(np.std([row["S"] for row in rows], ddof=1) / np.sqrt(len(rows))) if len(rows) > 1 else None
    record = {"blend": weights, "summary_mean_over_folds": summary, "S_se_over_folds": se,
              "per_fold": rows, "folds": results}
    print(f"blend {weights}: folds {len(rows)}, mean S {summary['S']:.6f} (SE over folds {se})")
    print("  " + "  ".join(f"{k} {v:.6f}" for k, v in summary.items()))
    if out is not None:
        out.write_text(json.dumps(record, indent=1))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dump")
    d.add_argument("artifact", type=Path)
    d.add_argument("out", type=Path)
    d.add_argument("--jobs", type=int, default=24)
    s = sub.add_parser("score")
    s.add_argument("campaign", type=Path)
    s.add_argument("--blend", required=True)
    s.add_argument("--json", type=Path)
    s.add_argument("--oof-name", default="oof.npz")
    s.add_argument("--jobs", type=int, default=24)
    s.add_argument("--folds", type=int, nargs="+")
    args = ap.parse_args()
    if args.cmd == "dump":
        dump(args.artifact, args.out, args.jobs)
    else:
        score(args.campaign, args.blend, args.jobs, args.json, args.oof_name, args.folds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
