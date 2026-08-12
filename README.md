# Fusion Equilibrium Challenge — pshishkin's fork

Working notes for this fork. The organizers' full guide — physics, data dictionary, scoring
detail, submission rules — is [`README_ORIGINAL.md`](README_ORIGINAL.md), kept unmodified.
Coding conventions, starting with the fail-fast rule, are in [`AGENTS.md`](AGENTS.md) — read that
before adding code under `my_experiments/`.

## What this fork adds

| | |
|---|---|
| `my_experiments/train.py`, `evaluate.py` | The two entry points: shares of the shot list, ordered by a hash of the shot id — training takes the head, the validation window sits right behind it, scoring takes the tail. The pipeline lives in `baseline_model.py`, which **saves** what it trains (`baseline.joblib`) — the starter kit's `experiments.py` never persists a model, so nothing it trains can be scored or submitted. |
| `my_experiments/models.py` + `params.yaml` | The model zoo: ridge, CatBoost, a torch MLP, and a weighted average of the last two. All are fitted on the same features and targets and scored side by side. Every hyper-parameter and the ensemble weights live in `params.yaml`. |
| `my_experiments/eda*.py` | One shot printed transposed with a shape column, for the training split and both public test splits. |
| `Makefile` | `make ci` = ruff + mypy + the standard metric run. |
| `--source local` | Added to `local_score.py`, `submission_skeleton.py`, `validate_submission.py` — read a downloaded copy of the dataset instead of streaming the Hub. `experiments.py` already had it. |
| `--models` | On `local_score.py` and `evaluate.py`, to score several members of the zoo on one pass over the ground truth. |
| `--jobs` | Everything per-shot on one shared process pool: reading shots (22.6 s to 8.5 s for 352) and the per-shot half of scoring (85.7 s to 28.5 s for 14 x 4 models). Results are bit-identical — see `my_experiments/parallel.py`. |
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

**Three** windows on one shot list ordered by `sha1` of the shot id: training from the head,
validation immediately behind it, scoring from the tail. Validation belongs to the head end because
it is part of *fitting* — CatBoost and the MLP read it every iteration to decide when to stop and
which iteration to keep — so only the untouched tail can still measure generalization.
`evaluate.py` checks the tail against the training **and** validation filenames recorded in the
artifact, rather than trusting index arithmetic, and refuses to score on overlap. `hashlib`, never
the builtin `hash()`, which is salted per process and would silently reorder between the two runs.

```bash
# the number that means something: 352 shots at a fifth of their frames, 70 scored
uv run python my_experiments/train_eval.py 0.05/0.2 0.05/0.2 0.01        # make quality

# does it work at all, and how fast: 70 shots, 14 scored, ~2.5 min
uv run python my_experiments/train_eval.py 0.01/0.2 0.01/0.2 0.002       # make test

# the smoke run plus the linter and the type checker
make ci

# 1. train on the head of the list -> my_experiments/baseline.joblib
uv run python my_experiments/train.py --share 0.05/0.2 --val-share 0.05/0.2

# 2. score on the tail, with the real competition metric — every model, then the comparison table
uv run python my_experiments/evaluate.py --share 0.01
uv run python my_experiments/evaluate.py --share 0.01 --models catboost ensemble

# 3. build the submission (874 local DIII-D test shots, 1.38 GB .npz, ~3 min)
uv run python submission_skeleton.py --max-shots 0 --source local \
    --configs diii_d_public_test

# 3b. push the file already built, without spending 3 minutes rebuilding it
uv run python push_predictions.py --repo pshishkin/fusion-eq-predictions --read-token hf_...

# or all of it in one command: predict, validate, push, write the pointer zip
make predict_and_submit_to_hf

# 4. upload the newest submissions/submission_pointer_*.zip to codabench.org/competitions/17456
```

Step 3 sends nothing anywhere without `--repo`; with `--repo` but no `--read-token` it only creates
the private Hugging Face repo. Only both together upload. It predicts with the **ensemble** —
`predict_row`'s default — so what goes to the leaderboard is whatever `ensemble.members` in
`params.yaml` weights.

`evaluate.py --mode perfect` / `--mode zeros` check the harness itself — they must print S = 1.0
and S = 0.0 exactly.

## The models

`params.yaml` is the whole configuration: which models are fitted, with which hyper-parameters,
and how the ensemble weights them. The code carries no defaults to fall back on and rejects an
unknown key rather than ignoring it, so a mistyped hyper-parameter fails the run instead of
quietly not being applied. What is deliberately *not* configurable is how the targets are scaled:
the metric decides that one, so it lives in code.

All of them regress the same vector — `[50 PCA coefficients of ψ, q95, betaN]` — from the same 21
scaled magnetics features, which is what makes them comparable and averageable:

| | |
|---|---|
| `ridge` | The linear baseline this fork started from. Trained and scored, but **not** an ensemble member — averaging a model that scores 0.12 into two that score ~0.7 only drags them down — 0.5936 with it against 0.7466 without, both measured on the same run. It stays as the figure everything else is read against. |
| `catboost` | Gradient-boosted trees, one `MultiRMSE` model over all 52 outputs. |
| `mlp` | 21 → 256 → 256 → 52 on CPU torch, our own training loop. |
| `ensemble` | The weighted average of `ensemble.members`. Averaging coefficients and averaging flux maps are the same thing here — the PCA decoder is affine and the weights sum to 1. |

### Target scaling, and why it is not a setting

Some scaling is unavoidable — the leading PCA component carries variance ~1e10 against ~1 for
betaN, so a raw squared-error loss is a fit of that one component and nothing else. But the
*right* scaling is not a matter of taste, so `TargetScaler` in `my_experiments/models.py` fixes it
rather than exposing a knob:

- **One shared divisor for the whole PCA block.** The basis is orthonormal, so by Parseval the
  pixel error of a flux map is the **unweighted** sum of the coefficient errors, ‖Δψ‖² = Σₖ Δcₖ².
  One divisor preserves that geometry, so minimizing the loss is minimizing R²ψ.
- **Each of the two scalars on its own std** — their term of the metric is an R² per scalar,
  against its own variance.
- **The blocks weighted as the composite weights them:** ψ coefficients divided by
  `sqrt(SS_tot_psi per frame / W_PSI)`, each scalar by `std / sqrt(W_QB / 2)`, with the weights
  imported from `fusion_scoring/common.py` rather than copied. The summed squared error then *is*
  the differentiable half of the composite, and the split lands at 71 / 29 — not 79 / 21, because
  `SS_tot_psi` is measured against a single flat mean rather than the mean image.

The alternatives were measured before being discarded, on `train_eval.py 0.01 0.001` with fixed
budgets of 500 iterations / 60 epochs:

| scaling | ensemble | catboost | mlp | ridge |
|---|---|---|---|---|
| as shipped | **0.7466** | **0.7370** | 0.7010 | 0.1157 |
| shared divisor, blocks left at 96 / 4 | 0.7408 | 0.7317 | 0.7023 | 0.1157 |
| every component standardized separately | 0.7155 | 0.7001 | 0.6796 | 0.1157 |

Ridge is identical in all three because it is separable per output and the scale cancels in its
solution — which made it a free check that the scaling step changed nothing it should not have.

### Frames inside one shot are near-duplicates

The equilibrium moves on the current-diffusion timescale, hundreds of milliseconds, while EFIT
frames come far faster — 234 per shot on average. So 16363 training rows are nowhere near 16363
independent observations, and of the psi variance in this data 53% sits BETWEEN shots against 47%
within. At a fixed row budget, shots are what the budget should be spent on.

Both fitting shares therefore take a `shots/frames` form: `0.05/0.2` reads 5% of the shots and
keeps every fifth frame of each. The frames are thinned as each shot is read, before anything is
concatenated, and they are **evenly spaced rather than randomly drawn** — a shot runs through
ramp-up, flat-top and ramp-down, and a uniform draw over 234 frames leaves clumps and gaps across
regimes that are physically different. A stride covers all three and needs no seed to reproduce.

The test share refuses a frame fraction: the metric is defined over every frame of the shots it
scores, and thinning them would produce a number that is not the competition metric.

What it is worth, at equal rows — 70 shots x every frame (16363 rows) against 352 shots x every
fifth (15422 rows), same 14 scored shots:

| | ensemble | catboost | mlp | ridge |
|---|---|---|---|---|
| `0.01 0.01 0.002` | 0.8111 | 0.7753 | 0.7668 | 0.3693 |
| `0.05/0.2 0.05/0.2 0.002` | **0.9319** | **0.9304** | **0.9030** | **0.4423** |

R²ψ goes 0.9240 to 0.9898 and Consistency 0.4472 to 0.7475 for the ensemble. Ridge gains too, and
ridge cannot use extra capacity — the gain is purely in WHICH frames it saw. This is the largest
single improvement measured in this fork, and it cost nothing but reading more files.

Note where CatBoost ended: iteration 2985 of the 3000 ceiling, i.e. it is budget-limited again now
that the data is more varied. Raising `iterations` is the obvious next thing to try.

### Early stopping

CatBoost and the MLP take the validation window as an upper bound on their own budget: CatBoost
with `eval_set` + `early_stopping_rounds` + `use_best_model`, the MLP by evaluating the validation
MSE each epoch and keeping the best epoch's weights rather than the last. `iterations` and `epochs`
in `params.yaml` are therefore ceilings, not targets. Ridge has no iterations, so it accepts the
validation set and ignores it.

`iterations` and `epochs` are ceilings set far above anything the fit should need; both models stop
themselves — CatBoost after ~5061 iterations keeping its best at 4861, the MLP after 686 epochs
keeping epoch 585.

**Does it pay?** Measured on the fixed 7-shot tail, so every row is comparable, with "off" meaning
train the full budget and keep the **last** iteration:

| | catboost | mlp | ensemble |
|---|---|---|---|
| 70-shot validation, patience 200 / 100 | 0.7438 | **0.7273** | — |
| 14-shot validation, patience 200 / 20 | 0.7439 | 0.7130 | 0.7427 |
| 70-shot validation, patience 200 / **20** | 0.7438 | 0.5521 | 0.6585 |
| off, 2000 iters / 300 epochs, keep last | 0.7435 | 0.7070 | 0.7462 |
| off, 500 iters / 60 epochs (the old fixed budget) | 0.7370 | 0.7010 | 0.7466 |

Four things that table says:

**CatBoost does not care.** 0.7435 to 0.7439 across every configuration, while its chosen stopping
point moves from 1942 to 4861 iterations. Three thousand extra trees bought nothing measurable.

**The MLP cares a lot, and about patience, not about early stopping as such.** The same validation
window gives 0.5521 at patience 20 and 0.7273 at patience 100, because its validation curve has a
false floor: patience 20 stopped at epoch 27 keeping epoch 6 — 192 Adam steps for a 75k-parameter
net, undertrained by construction. An epoch here is 32 minibatches, so patience counted in epochs
is a far shorter leash than it sounds.

**A wider validation window is not automatically better.** It moved the MLP from 0.7130 to 0.5521
until the patience was fixed to match. The window and the patience are one decision, not two.

**Validation MSE is not the composite.** The MLP's validation loss keeps improving past the epoch
where its score peaks. The loss sees 52 scaled targets; the score also sees `D_LCFS` and the seven
derived scalars. Do not read a falling validation curve as a rising S.

### Where the models stand

Two commands, and which one produced a number has to be said every time — see AGENTS.md, "How we
test the metric".

**`train_eval.py 0.01/0.2 0.01/0.2 0.002`** — the smoke run: 70 shots at a fifth of their frames
(3272 rows), 14 scored. **2 m 24 s** end to end, which is what makes it usable after every change:

```
             model         S    R2_psi     R2_qb   1-D_LCFS      Cons
          ensemble    0.7980    0.9131    0.8047     0.9414    0.4047
          catboost    0.7874    0.9181    0.7259     0.9275    0.4040
               mlp    0.7399    0.8882    0.7101     0.9371    0.2561
             ridge    0.3566    0.3416    0.2669     0.9401    0.1735
```

| | |
|---|---|
| ruff + mypy | ~10 s |
| reading 140 shots (3272 + 3111 rows kept) | 8 s |
| PCA of psi + target scaling | ~5 s |
| fitting ridge / CatBoost / MLP | 0.1 s / 87 s / 23 s |
| scoring 14 shots x 4 models, `--jobs` auto | 28 s |

**`train_eval.py 0.2/0.2 0.2/0.2 0.01`** — the production run, what a submission is built from:
1408 shots to fit, 1408 to stop on, 70 scored, **6 m 28 s**. CatBoost disabled for this one, so the
ensemble is the MLP alone:

```
             model         S    R2_psi     R2_qb   1-D_LCFS      Cons
               mlp    0.9677    0.9990    0.9822     0.9794    0.8648
          ensemble    0.9677    0.9990    0.9822     0.9794    0.8648
             ridge    0.7126    0.8079    0.5607     0.9525    0.4444
```

The MLP fits 62968 frames in 266 s, stopping at epoch 767 of the 868 it ran. R²ψ = 0.9990 and every
derived scalar above 0.68. Against 0.9319 for the same recipe on a fifth of the shots, the lesson
holds: at this scale shots are still buying score, so a submission is retrained at 0.2 and never at
0.05.

**`train_eval.py 0.05/0.2 0.05/0.2 0.01`** — the quality run: 352 shots, 70 scored, roughly a tenth
of the cost. On the earlier 14-shot fold it scored `ensemble 0.9319 / catboost 0.9304 / mlp 0.9030
/ ridge 0.4423`.

**These numbers are not comparable to the ones in the section above.** The scored fold changed with
the shares: ridge went from 0.1157 to 0.3693 without a line of code changing, purely because the
tail of 14 shots is a different, easier set than the tail of 7. Only ever compare rows produced by
the same command — which is why the early-stopping table above was re-measured on one fixed fold
rather than quoted from whatever run produced it.

Only compare figures produced by the same command. Seven held-out shots is a small sample, and a
wider split lands somewhere else entirely — `train_eval.py 0.1 0.02` gave `S = 0.7193`,
`R2_psi = 0.8469` on 141 held-out shots for ridge alone, measured just before the plasma-current
axis fix was finalised. The evaluation sets differ, so the two numbers are not each other's
baseline.

Reference points measured on the same metric, worth keeping in mind: predicting a single flat
constant scores R²ψ = 0 by construction, and predicting the mean ψ *image* already scores 0.36. The
useful range of R²ψ starts around 0.36, not at zero.

### Checks that must keep passing

```bash
make ci                                                     # ruff, mypy, then the metric run
uv run python my_experiments/evaluate.py --share 0.0005 --mode perfect   # S = 1.0000
uv run python my_experiments/evaluate.py --share 0.0005 --mode zeros     # S = 0.0000
```

`make ci` lints and type-checks this fork's own code only — `my_experiments/` plus the root
scripts we edited. The organizers' files and the vendored `fusion_scoring/` stay byte-identical to
upstream and are excluded in `pyproject.toml`.

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

**The PCA of ψ was not reproducible.** `experiments.TargetPCA` builds `PCA(n_components=50)` and
leaves `random_state` at `None`, and sklearn picks the *randomized* SVD solver for 50 components
out of 4225 pixels. Two runs on the same shots therefore fitted different components — different
targets for every model, and a score that moved on its own: the MLP scored 0.6769 and then 0.6846
on the same 7 held-out shots. `train()` now pins the inner estimator's `random_state` to
`features.pca_seed` from `params.yaml`, rather than editing the organizers' file.

**Roughly 10% of DIII-D shots run reversed plasma current**, and `ridge` scores worse than a
constant on every one of them. The whole input vector flips sign with Ip (the shaping coils mirror
it), so the two polarities are two regimes for one global linear map. Measured on ridge only —
whether the trees and the net, which are free to split on the sign, still suffer from it has not
been re-measured.

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

`baseline.joblib` (retrain with the command above), `submission/`, `submissions/`,
`manifest.json` and `.env` — the last three hold the read token in plaintext.

`.env` is where the token lives, as `HF_READ_TOKEN=hf_...`; the Makefile includes and exports it,
so `make predict_and_submit_to_hf` needs no argument and the token never reaches the command line.
`.env.example` is the tracked stub. It is deliberately not the Makefile itself — that file IS
tracked, and this fork is public.
