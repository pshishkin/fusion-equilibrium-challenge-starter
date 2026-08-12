# Fusion Equilibrium Challenge — pshishkin's fork

Working notes for this fork. The organizers' full guide — physics, data dictionary, scoring
detail, submission rules — is [`README_ORIGINAL.md`](README_ORIGINAL.md), kept unmodified.
Coding conventions, starting with the fail-fast rule, are in [`AGENTS.md`](AGENTS.md) — read that
before adding code under `my_experiments/`.

## What this fork adds

| | |
|---|---|
| `my_experiments/train.py`, `evaluate.py` | The two entry points: `--share` of the shot list, ordered by a hash of the shot id — training takes the head, evaluation the tail. The model itself lives in `baseline_model.py`, which **saves** what it trains (`baseline.joblib`) — the starter kit's `experiments.py` never persists a model, so nothing it trains can be scored or submitted. |
| `my_experiments/eda*.py` | One shot printed transposed with a shape column, for the training split and both public test splits. |
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

Two entry points, both taking a **share** of one shot list ordered by `sha1` of the shot id.
Training takes the head of that list, evaluation the tail — the windows grow from opposite ends, so
the split is held out as long as the shares sum to under 1. `evaluate.py` checks that against the
filenames recorded in the artifact rather than trusting index arithmetic, and refuses to score on
overlap. `hashlib`, never the builtin `hash()`, which is salted per process and would silently
reorder between the two runs.

```bash
# train and score in one go — this is the standard command for checking the metric
uv run python my_experiments/train_eval.py 0.01 0.001

# 1. train on the head of the list -> my_experiments/baseline.joblib
uv run python my_experiments/train.py --share 0.01

# 2. score on the tail, with the real competition metric
uv run python my_experiments/evaluate.py --share 0.02

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

`train_eval.py 0.01 0.001` — 70 shots to fit, 7 held out, of 7041. This is the command we measure
with, so quote its numbers:

```
COMPOSITE S = 0.1157
          R2_psi   -0.1074      Consistency    0.1196
  R2_{q95,betaN}   -0.0991       1 - D_LCFS    0.9181
```

Only compare figures produced by the same command. Seven held-out shots is a small sample, and a
wider split lands somewhere else entirely — `train_eval.py 0.1 0.02` gave `S = 0.7193`,
`R2_psi = 0.8469` on 141 held-out shots, measured just before the plasma-current axis fix was
finalised. Both are the same model; the evaluation sets differ.

Reference points measured on the same metric, worth keeping in mind: predicting a single flat
constant scores R²ψ = 0 by construction, and predicting the mean ψ *image* already scores 0.36. The
useful range of R²ψ starts around 0.36, not at zero.

## Things that already cost time

**Test rows have no `efit_*` targets.** `experiments.load_shot_from_hf_row` reads `efit_psirz`
unconditionally and dies on every test row; `baseline_model.inputs_only_shot` reads only
`efit_times` + `magnetics_*`. Anything on the inference path must go through the latter.

**DIII-D: `magnetics_plasma_current_times` is a shared template, not a per-shot axis.** The same
30719 samples starting at −858.1871 ms are stamped into every shot, while `magnetics_time` really
does vary. For the ~70% of shots recorded at 0.05 ms that template is the wrong axis, so
interpolating Ip onto `efit_times` returns pre-shot noise — 4 kA where the trace sits near 1000 kA.
The correct origin is the shot's own `magnetics_time[0]`, which puts current under 100% of the
frames of every affected shot. `baseline_model.align_ip_times` applies that, and only to shots that
need it; `my_experiments/eda_ip_offset.py` prints the evidence.

**Roughly 10% of DIII-D shots run reversed plasma current**, and the linear baseline scores worse
than a constant on every one of them. The whole input vector flips sign with Ip (the shaping coils
mirror it), so the two polarities are two regimes for one global linear map.

**Shot order is load-bearing.** The scorer matches `shot_XXXX_*` keys positionally. Local reads use
`sorted(dir.glob(...))`, which reproduces the Hub's order because `datasets` resolves files through
`fs.glob` and fsspec sorts. Cheap check against the real thing, one shard rather than the whole
fold:

```bash
uv run python validate_submission.py submission/diii_d_public_test.npz \
    --config diii_d_public_test --max-shots 5
```

**MAST is unimplemented.** `predict_row` raises `NotImplementedError` on a MAST row rather than
returning zeros that would look like a working prediction, so submissions are DIII-D only:
Challenge 1 scores, Challenge 2 shows `G_ratio = 0`.

## Pointer zips

Each push writes `submissions/submission_pointer_<UTC timestamp>.zip` rather than one fixed
filename, so rebuilding never overwrites the zip of a submission already on the leaderboard — its
manifest pins a commit SHA, which makes the old zip a record of exactly what was scored. Upload
the newest; keep the rest.

## Not in git

`baseline.joblib` (retrain with the command above), `submission/`, `submissions/` and
`manifest.json` — the last two hold the read token in plaintext.
