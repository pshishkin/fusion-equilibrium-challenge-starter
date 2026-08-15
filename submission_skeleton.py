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
import os
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
from datasets import load_dataset

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
    # Delegate to the trained baseline if one has been saved. Only "there is no model yet" is
    # swallowed — a bug inside the model must surface, not silently score as zeros.
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from my_experiments.baseline_model import ENSEMBLE, predict_row
        return predict_row(row, source, model if model is not None else ENSEMBLE)
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


def build_submission(config: str, split: str, out_dir: Path, max_shots: int,
                     source: str = "hf",
                     local_data_dir: Path = DEFAULT_LOCAL_DATA_DIR,
                     model: str | None = None) -> Path:
    preds: dict[str, np.ndarray] = {}
    n = 0
    for i, row in enumerate(iter_dataset_rows(config, split, source, local_data_dir)):
        if max_shots and i >= max_shots:
            break
        source = row.get("source", "DIII-D")
        T = len(np.asarray(row["efit_times"]))
        H, W = GRID[source]
        out = your_model_predict(row, source, model)

        assert out["psirz"].shape == (T, H, W), \
            f"{config} shot {i}: psirz {out['psirz'].shape} != {(T, H, W)}"
        # float16 keeps relative precision everywhere and costs ~0.1% of score; the scorer
        # upcasts on read. Do NOT instead round to a fixed number of decimals -- np.round(psi, 3)
        # leaves R2_psi at 0.99997 while destroying ~35% of the MAST Consistency term.
        preds[f"shot_{i:04d}_psirz"] = out["psirz"].astype(np.float16)
        for name in SCALARS:
            arr = np.asarray(out[name])
            assert arr.shape == (T,), f"{config} shot {i}: {name} {arr.shape} != ({T},)"
            preds[f"shot_{i:04d}_{name}"] = arr.astype(np.float32)

        n = i + 1
        if n % 25 == 0:
            print(f"  {config}: {n} shots")

    out_path = out_dir / f"{config}.npz"
    # The numpy stubs type savez_compressed's second parameter as a positional flag, so
    # keyword-splatting the per-shot arrays — the documented way to write named keys —
    # does not type-check.
    np.savez_compressed(out_path, **preds)  # type: ignore[arg-type]
    psh = preds.get("shot_0000_psirz", np.empty(0)).shape
    print(f"  {config}: {n} shots -> {out_path.name}  (e.g. shot_0000_psirz {psh}, "
          f"shot_0000_q95 {preds.get('shot_0000_q95', np.empty(0)).shape})")
    return out_path


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
                                        args.source, args.local_data_dir, args.model).name)

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
