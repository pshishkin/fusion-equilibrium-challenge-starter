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


def score_fold(files: list[Path], payload: dict[str, np.ndarray], jobs: int) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
        pred, metrics = Path(tmp) / "pred.npz", Path(tmp) / "metrics.json"
        np.savez(pred, **payload)
        local_score.main(["--source", "local", "--config", HF_TRAIN_CONFIG,
                          "--files", *map(str, files), "--pred", str(pred),
                          "--jobs", str(jobs), "--metrics-json", str(metrics)])
        return json.loads(metrics.read_text())


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
