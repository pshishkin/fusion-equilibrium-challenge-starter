# Fusion Equilibrium Challenge — pshishkin's fork

Working notes for this fork. The organizers' full guide — physics, data dictionary, scoring
detail, submission rules — is [`README_ORIGINAL.md`](README_ORIGINAL.md), kept unmodified.

## What this fork adds

| | |
|---|---|
| `my_experiments/train.py`, `evaluate.py` | The two entry points: train on the first N shots, score on the last N. The model itself lives in `baseline_model.py`, which **saves** what it trains (`baseline.joblib`) — the starter kit's `experiments.py` never persists a model, so nothing it trains can be scored or submitted. |
| `--source local` | Added to `local_score.py`, `submission_skeleton.py`, `validate_submission.py` — read a downloaded copy of the dataset instead of streaming the Hub. `experiments.py` already had it. |
| `--configs` | On `submission_skeleton.py`, to build a DIII-D-only submission without downloading MAST. |

## Data

Downloaded copy lives at `../downloaded_huggingface/hf_dataset`, laid out exactly like the Hub
repo (`data/<config>/*.parquet`). That path is the default for every `--local-data-dir` flag, so
`--source local` needs no extra argument.

```bash
# add more training shots (each parquet is one shot, ~13 MB)
hf download Sophelio/fusion-equilibrium-challenge --repo-type dataset \
  --local-dir ../downloaded_huggingface/hf_dataset \
  --include "data/diii_d_train/d3d_shot_01*" --max-workers 16
```

**Prefer `--source local` over streaming.** `experiments.py --source hf` fills a shuffle buffer of
up to 500 rows, and a row costs ~420 MB as a Python dict — measured, not estimated. `--n-shots 50`
therefore asks for ~200 GB and gets OOM-killed before the first shot loads.

## Workflow

Two entry points. Training takes the **first** N shots, evaluation the **last** N, both in sorted
filename order — the windows grow from opposite ends, so the split is held out by construction.
`evaluate.py` checks that rather than trusting it and refuses to score if they overlap.

```bash
# 1. train on the first 20 shots -> my_experiments/baseline.joblib
uv run python my_experiments/train.py --n-shots 20

# 2. score on the last 20 shots, with the real competition metric
uv run python my_experiments/evaluate.py --n-shots 20

# 3. build the submission (874 local DIII-D test shots, ~1.8 GB .npz)
uv run python submission_skeleton.py --max-shots 0 --source local \
    --configs diii_d_public_test \
    --repo pshishkin/fusion-eq-predictions --read-token hf_...

# 4. upload the newest submissions/submission_pointer_*.zip to codabench.org/competitions/17456
```

Step 3 sends nothing anywhere without `--repo`; with `--repo` but no `--read-token` it only creates
the private Hugging Face repo. Only both together upload.

`evaluate.py --mode perfect` / `--mode zeros` check the harness itself — they must print S = 1.0
and S = 0.0 exactly.

### Where the baseline stands

Trained on the first 20 shots, scored on the last 20 (of 7034 downloaded):

```
COMPOSITE S = 0.0876
          R2_psi   -0.2166      Consistency    0.0050
  R2_{q95,betaN}   -1.0901       1 - D_LCFS    0.8657
```

Negative R² means the model is worse than predicting the training mean — 21 coil currents through
one linear map do not transfer across shots. Only `1 - D_LCFS` scores, and that term is forgiving.
The plumbing is proven end to end; the model is the open problem. `MODELING_GUIDE.md` is where to
start, and the per-scalar breakdown that `evaluate.py` prints says which derived quantity is worst
(currently `Z_axis`, R² = −24).

## Things that already cost time

**Train and score must not overlap.** `baseline_model.py` takes shots in sorted filename order
(`files[:n_shots]`), so `--skip n_shots` in the scorer is disjoint by construction.
`experiments.py --source local` instead samples randomly, so its splits give no such guarantee.

**Test rows have no `efit_*` targets.** `experiments.load_shot_from_hf_row` reads `efit_psirz`
unconditionally and dies on every test row; `baseline_model.inputs_only_shot` reads only
`efit_times` + `magnetics_*`. Anything on the inference path must go through the latter.

**Shot order is load-bearing.** The scorer matches `shot_XXXX_*` keys positionally. Local reads use
`sorted(dir.glob(...))`, which reproduces the Hub's order because `datasets` resolves files through
`fs.glob` and fsspec sorts. Cheap check against the real thing, one shard rather than the whole
fold:

```bash
uv run python validate_submission.py submission/diii_d_public_test.npz \
    --config diii_d_public_test --max-shots 5
```

**MAST is unimplemented.** `predict_row` returns zeros for it, so submissions are DIII-D only:
Challenge 1 scores, Challenge 2 shows `G_ratio = 0`.

## Pointer zips

Each push writes `submissions/submission_pointer_<UTC timestamp>.zip` rather than one fixed
filename, so rebuilding never overwrites the zip of a submission already on the leaderboard — its
manifest pins a commit SHA, which makes the old zip a record of exactly what was scored. Upload
the newest; keep the rest.

## Not in git

`baseline.joblib` (retrain with the command above), `submission/`, `submissions/` and
`manifest.json` — the last two hold the read token in plaintext.
