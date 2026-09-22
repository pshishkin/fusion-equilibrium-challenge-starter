#!/usr/bin/env python3
"""
Submission skeleton — a runnable, format-correct example of a challenge submission.

This is the executable version of README → "3. What you predict". Run it and read the
printed shapes to see exactly what a submission looks like — no Git LFS / sample data needed
(it streams the public test inputs from Hugging Face).

What you submit (per shot, at each `efit_times` timestamp), grouped per shot in one .npz per config:
    shot_0000_psirz    (T, H, W)   flux map   DIII-D 65x65 / MAST 65x65 (both dense, no NaN region)
    shot_0000_q95      (T,)        edge safety factor
    shot_0000_betaN    (T,)        normalized beta

That is the WHOLE contract (metric v2): the flux map plus the only two scalars a flux map cannot
contain (q95 needs the toroidal field function F(psi), betaN needs the pressure profile p(psi)).
Everything else — the LCFS boundary, magnetic axis (R_axis/Z_axis), elongation, triangularity,
volume and internal inductance (li) — is DERIVED from your submitted
flux map by the scorer, with the same published functionals it applies to the ground-truth flux.
A scalar can only be earned by a psi that implies it: there is no separate scalar head to tune.

The leaderboard score is the composite
    S = 0.55*R2_psi + 0.15*R2_{q95,betaN} + 0.10*(1 - D_LCFS) + 0.20*Consistency
where Consistency is the mean agreement of the seven psi-derived scalars, f(psi_pred) vs
f(psi_gt). A perfect flux map scores D_LCFS = 0 and Consistency = 1 by construction.

To make a real submission, replace `your_model_predict()` with your trained model. The placeholder
here emits zeros of the correct shape (a valid-but-useless submission) so you can confirm the
plumbing before plugging in a model.

This one script does the whole submission: build -> validate -> push to Hugging Face -> write the
zip you upload to Codabench.

    # quick format check, 5 shots, build only
    uv run python submission_skeleton.py --max-shots 5

    # the real thing: every shot, pushed, with the timestamped pointer zip ready to upload
    uv run python submission_skeleton.py --max-shots 0 \
        --repo your-username/fusion-eq-predictions --read-token hf_...

    # build from a downloaded copy instead of streaming the test fold twice (build + validate)
    uv run python submission_skeleton.py --max-shots 0 --source local \
        --configs diii_d_public_test

Run it once without --read-token to create the repo; Hugging Face cannot scope a token to a repo
that does not exist yet. See README -> "5. Build and submit".
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
from datasets import load_dataset

from tools.submission_format import EXPORT_FORMAT_VERSION, PSI_DTYPE, SCALAR_DTYPE, check_file

REPO_ID = "Sophelio/fusion-equilibrium-challenge"
TEST_CONFIGS = [("diii_d_public_test", "public_test"), ("mast_public_test", "public_test")]
# Downloaded copy of the dataset, same layout and same default as experiments.py / local_score.py:
#   <DEFAULT_LOCAL_DATA_DIR>/data/<config>/*.parquet
DEFAULT_LOCAL_DATA_DIR = (Path(__file__).resolve().parent.parent
                          / "downloaded_huggingface" / "hf_dataset")
# Native flux grid per machine (rows=Z, cols=R). Both machines are a dense, fully finite 65x65
# grid — MAST's upstream EFIT 65x129 grid (65 real R columns interleaved with 64 empty ones) is
# collapsed to a dense 65x65 in the corrected dataset, so there is no central NaN region.
GRID = {"DIII-D": (65, 65), "MAST": (65, 65)}
# The only two submitted scalars (metric v2), each under its own per-shot key (named, not
# positional, to make column mix-ups impossible). Everything else is derived from your flux map.
SCALARS = ["q95", "betaN"]
# Printed once, not once per shot: the zeros placeholder is a state of the repo, not an
# event of this row.
_WARNED_NO_MODEL = False


def your_model_predict(row: dict, source: str, model: str | None = None) -> dict:
    """REPLACE ME. Return predictions for this shot, aligned to row['efit_times'], as a dict:
        {"psirz":  (T, H, W) flux map,
         "q95":    (T,),
         "betaN":  (T,)}

    Inputs in `row` (NO magnetic-diagnostic array — that's the point of the challenge):
      - `magnetics_*` COIL CURRENTS = commanded actuators (F-coils, ECOILA/bcoil, MAST P-coils,
        Solenoid, TF, …), not measurements of the plasma's field.
      - `thomson_*` = kinetic profiles (electron temperature & density).
      - `efit_times` (+ MAST: `efit_grid_R/Z`).
    The targets are withheld in test configs. `magnetics_dsep` is EFIT-DERIVED (computed from
    the target equilibrium) and is neither an input nor a scored target. Also available as
    inputs: `coil_*` (PF-coil positions/turns, joined to the current columns by
    `coil_input_column`) and `thomson_chord_R/Z` (chord positions). Focus on making psi(R,Z)
    right; the geometry terms (boundary, axis, shape, li) all follow from it.

    `model` picks one member of the trained zoo by name (params.yaml); None means the default,
    which is the ensemble. Submissions never pass it — only local_score.py --models does."""
    # Delegate to `machines.py`, the one file that knows both challenges exist: DIII-D goes to the
    # trained model in my_experiments/, MAST to the Grad-Shafranov solve in mast/. `source` is not
    # passed on — the dispatcher reads the row's own `source` column, so a row can only ever be
    # predicted by the module for the machine it actually came from. Only "there is no model yet"
    # is swallowed here; a bug inside either predictor must surface, not silently score as zeros.
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from machines import predict_row
        return predict_row(row, model)
    except (ImportError, FileNotFoundError) as exc:
        global _WARNED_NO_MODEL
        if not _WARNED_NO_MODEL:
            print(f"  note: no trained model ({type(exc).__name__}: {exc})\n"
                  f"        emitting zeros — train one with "
                  f"'uv run python my_experiments/train.py --share 0.01'")
            _WARNED_NO_MODEL = True

    T = len(np.asarray(row["efit_times"]))
    H, W = GRID[source]
    out: dict[str, np.ndarray] = {"psirz": np.zeros((T, H, W), dtype=np.float32)}
    for name in SCALARS:
        out[name] = np.zeros(T, dtype=np.float32)               # placeholder scalars
    return out


def iter_dataset_rows(config: str, split: str, source: str,
                      local_data_dir: Path) -> Iterator[Any]:
    """Yield the config's rows in the SAME order the Hub stream would.

    Order is not cosmetic: the scorer matches `shot_XXXX_*` keys positionally against its own
    reference, so a different order silently scores every shot against the wrong ground truth.
    The local path is safe because the Hub order is itself sorted by filename — datasets resolves
    data files through `fs.glob` (data_files.py), and fsspec's glob returns `sorted(...)`.
    """
    if source != "local":
        yield from load_dataset(REPO_ID, config, split=split, streaming=True)
        return

    import pandas as pd

    data_dir = Path(local_data_dir) / "data" / config
    files = sorted(data_dir.glob("*.parquet"))
    if not files:
        raise SystemExit(
            f"No parquet files in {data_dir}. Download them with:\n"
            f'  hf download {REPO_ID} --repo-type dataset --local-dir {local_data_dir} '
            f'--include "data/{config}/*"'
        )
    print(f"  {config}: reading {len(files)} local shots from {data_dir}")
    for path in files:
        yield pd.read_parquet(path).iloc[0]


def predict_one(args: tuple) -> dict:
    """One shot, from its parquet path — the unit `--jobs` spreads over worker processes.

    Takes a PATH rather than a row on purpose: a MAST row is a few megabytes and a DIII-D one is
    over a hundred, so shipping rows through the pool would cost more in pickling than the solve
    costs in arithmetic. Each worker reads its own file instead.
    """
    import pandas as pd

    path, model = args
    row: Any = pd.read_parquet(path).iloc[0]
    out = your_model_predict(row, str(row.get("source", "DIII-D")), model)
    return {"psirz": np.asarray(out["psirz"], dtype=PSI_DTYPE),
            **{name: np.asarray(out[name], dtype=SCALAR_DTYPE) for name in SCALARS},
            "T": len(np.asarray(row["efit_times"])),
            "source": str(row.get("source", "DIII-D"))}


MAST_CACHE = Path(".mast_cache")


def _mast_cache_key(config: str, local_data_dir: Path, max_shots: int) -> str:
    """Everything that can change a MAST .npz, hashed.

    **Why a MAST cache is sound at all, and it is not a shortcut.** `machines.predict_row` sends a
    MAST row to `mast.predict.predict_row(row)`, which takes the ROW AND NOTHING ELSE — no model, no
    artifact. So every DIII-D submission recomputes 1206 MAST shots that cannot have changed, and on
    2026-09-04 that was more than half the wall time of a submission whose DIII-D half was the only
    thing being varied.

    The key is the same shape as the shot cache's: the CODE that produces the output plus the
    identity of the inputs. `mast/*.py` is the solver, `machines.py` is the dispatch that chooses
    it, and this file assembles the arrays — a change in any of them changes the file. The parquet
    files are keyed by name, size and mtime rather than content, because hashing 1206 shots to avoid
    predicting them would defeat the point.

    What it cannot see is the same thing the shot cache cannot: an environment change that alters
    the arithmetic without touching a line of source, a numpy upgrade being the standard example.
    Delete `.mast_cache` when that happens.
    """
    h = hashlib.sha256()
    for f in [*sorted(Path("mast").glob("*.py")), Path("machines.py"), Path(__file__),
              Path(__file__).resolve().parent / "tools/submission_format.py"]:
        if f.exists():
            h.update(f.name.encode())
            h.update(f.read_bytes())
    h.update(EXPORT_FORMAT_VERSION.encode())
    for f in sorted((Path(local_data_dir) / "data" / config).glob("*.parquet")):
        st = f.stat()
        h.update(f"{f.name}:{st.st_size}:{int(st.st_mtime)}".encode())
    h.update(f"max_shots={max_shots}".encode())
    return h.hexdigest()[:32]


def build_submission(config: str, split: str, out_dir: Path, max_shots: int,
                     source: str = "hf",
                     local_data_dir: Path = DEFAULT_LOCAL_DATA_DIR,
                     model: str | None = None, jobs: int = 0) -> Path:
    # MAST does not depend on `model`, so a DIII-D submission reuses it rather than spending an
    # hour reproducing it. Only when the key matches — see `_mast_cache_key`.
    cached = None
    if config.startswith("mast_") and source == "local":
        cached = MAST_CACHE / _mast_cache_key(config, local_data_dir, max_shots) / f"{config}.npz"
        if cached.exists():
            check_file(cached)
            out_path = out_dir / f"{config}.npz"
            out_dir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(cached, out_path)
            print(f"  {config}: reused {cached} ({out_path.stat().st_size / 1e6:.0f} MB) — MAST is "
                  f"a function of the row alone, and neither its code nor the data has changed")
            return out_path

    preds: dict[str, np.ndarray] = {}
    n = 0
    for i, out in enumerate(_predictions(config, split, source, local_data_dir, max_shots,
                                         model, jobs)):
        T, machine = out["T"], out["source"]
        H, W = GRID[machine]
        assert out["psirz"].shape == (T, H, W), \
            f"{config} shot {i}: psirz {out['psirz'].shape} != {(T, H, W)}"
        # Preserve the decoded map at float32 precision. The metric derives contours and seven
        # functionals from it; the paired held-out cast measurement is in
        # results/float16_export_audit_20260922.json.
        preds[f"shot_{i:04d}_psirz"] = out["psirz"]
        for name in SCALARS:
            arr = out[name]
            assert arr.shape == (T,), f"{config} shot {i}: {name} {arr.shape} != ({T},)"
            preds[f"shot_{i:04d}_{name}"] = arr

        n = i + 1
        if n % 25 == 0:
            print(f"  {config}: {n} shots")

    out_path = out_dir / f"{config}.npz"
    # The numpy stubs type savez_compressed's second parameter as a positional flag, so
    # keyword-splatting the per-shot arrays — the documented way to write named keys —
    # does not type-check.
    np.savez_compressed(out_path, **preds)  # type: ignore[arg-type]
    check_file(out_path)
    if cached is not None:
        cached.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(out_path, cached)
        print(f"  {config}: cached at {cached}")
    psh = preds.get("shot_0000_psirz", np.empty(0)).shape
    print(f"  {config}: {n} shots -> {out_path.name}  (e.g. shot_0000_psirz {psh}, "
          f"shot_0000_q95 {preds.get('shot_0000_q95', np.empty(0)).shape})")
    return out_path


def _predictions(config: str, split: str, source: str, local_data_dir: Path, max_shots: int,
                 model: str | None, jobs: int) -> Iterator[dict]:
    """Predictions in shot order, serial or over a process pool.

    Order is not cosmetic — the scorer matches `shot_XXXX_*` keys positionally against its own
    reference — so the pool has to be order-preserving. `pimap` is (see toolkit/parallel.py).
    """
    if jobs and source != "local":
        raise SystemExit("--jobs needs --source local: the pool reads each shot's parquet in its "
                         "own worker, and the streaming path has no per-shot file to hand it")
    if not jobs:
        for i, row in enumerate(iter_dataset_rows(config, split, source, local_data_dir)):
            if max_shots and i >= max_shots:
                break
            out = your_model_predict(row, str(row.get("source", "DIII-D")), model)
            yield {"psirz": np.asarray(out["psirz"], dtype=PSI_DTYPE),
                   **{name: np.asarray(out[name], dtype=SCALAR_DTYPE) for name in SCALARS},
                   "T": len(np.asarray(row["efit_times"])),
                   "source": str(row.get("source", "DIII-D"))}
        return

    from toolkit.parallel import pimap, resolve_jobs

    files = sorted((Path(local_data_dir) / "data" / config).glob("*.parquet"))
    if max_shots:
        files = files[:max_shots]
    if not files:
        raise SystemExit(f"No parquet files in {Path(local_data_dir) / 'data' / config}")
    n_jobs = resolve_jobs(jobs, len(files))
    print(f"  {config}: {len(files)} local shots over {n_jobs} process(es)")
    yield from pimap(predict_one, [(f, model) for f in files], n_jobs)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Build a submission, validate it, push it to Hugging Face, and write the "
                    "pointer zip you upload to Codabench.")
    ap.add_argument("--max-shots", type=int, default=5, help="cap shots per config (0 = all)")
    ap.add_argument("--model",
                    help="which member of the zoo to submit, default the artifact's own ensemble. "
                         "Takes every form evaluate.py --models does, including 'a+b', "
                         "'psi=a+b,qb=c' and 'salts:a+b' — the last averages the DECODED maps of "
                         "several artifacts, which is how an ensemble across feature sets or "
                         "splits is submitted. Without this a measured combination cannot be sent")
    ap.add_argument("--out", type=Path, default=Path("submission"))
    ap.add_argument("--repo",
                    help="Hugging Face dataset repo to push to, e.g. you/fusion-eq-preds. "
                         "Given this, the script also pushes and writes the pointer zip.")
    ap.add_argument("--read-token", default=os.environ.get("HF_READ_TOKEN"),
                    help="fine-grained READ token scoped to --repo (or set HF_READ_TOKEN)")
    from push_predictions import POINTER_DIR, default_pointer_path
    ap.add_argument("--zip", dest="zip_out", type=Path, default=default_pointer_path(),
                    help=f"pointer zip to upload to Codabench "
                         f"(default: {POINTER_DIR}/submission_pointer_<UTC timestamp>.zip)")
    ap.add_argument("--skip-validate", action="store_true",
                    help="skip the structure check (not recommended -- it is what catches a "
                         "malformed .npz before you spend a submission slot)")
    ap.add_argument("--source", default="hf", choices=["hf", "local"],
                    help="'hf' streams the test inputs, 'local' reads a downloaded copy")
    ap.add_argument("--local-data-dir", type=Path, default=DEFAULT_LOCAL_DATA_DIR,
                    help="Root of the downloaded dataset (contains a 'data/' folder)")
    ap.add_argument("--jobs", type=int, default=0,
                    help="worker processes for prediction (needs --source local; 0 = serial). "
                         "The MAST solve is ~0.7 s a frame and the split is 1206 shots, so this "
                         "is the difference between 13 hours and half of one")
    ap.add_argument("--configs", nargs="+", choices=[c for c, _ in TEST_CONFIGS],
                    help="build only these configs (default: both). A DIII-D-only entry is valid "
                         "for Challenge 1 and scores G_ratio = 0 on Challenge 2.")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    configs = ([(c, s) for c, s in TEST_CONFIGS if c in args.configs]
               if args.configs else TEST_CONFIGS)

    print("Building submission:")
    written = []
    for config, split in configs:
        written.append(build_submission(config, split, args.out, args.max_shots,
                                        args.source, args.local_data_dir, args.model,
                                        args.jobs).name)

    # No manifest is written here: the scorer locates predictions by FILENAME, and on the
    # pointer route push_and_write_pointer() writes the real {repo_id, revision, token} one.
    total_mb = sum((args.out / w).stat().st_size for w in written) / 1e6
    print(f"\nWrote {', '.join(written)} to {args.out.resolve()}  ({total_mb:.0f} MB)")

    if not args.skip_validate:
        print("\nValidating structure...")
        from validate_submission import validate
        for config, _ in configs:
            if validate(args.out / f"{config}.npz", config, args.max_shots,
                        args.source, args.local_data_dir):
                print(f"\nValidation FAILED for {config}. Fix your_model_predict and rebuild; "
                      "nothing was pushed.", file=sys.stderr)
                return 1

    if args.repo:
        print()
        from push_predictions import push_and_write_pointer
        return push_and_write_pointer(args.repo, args.read_token, args.out, args.zip_out)

    print("\nNext: re-run with --repo <you>/fusion-eq-predictions to push and build the zip you\n"
          "      upload to Codabench. See README -> '5. Build and submit'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
