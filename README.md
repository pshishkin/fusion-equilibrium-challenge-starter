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
| `my_experiments/eda*.py` | One shot printed transposed with a shape column, for the training split and both public test splits. `eda_coil_field.py` is the odd one out — it plots the flux decomposition below. |
| `my_experiments/coil_field.py` | ψ = ψ_coil + ψ_plasma, with the first term computed from the shipped coil rectangles and currents instead of learned. Not wired into the model yet. |
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

All of them regress the same vector — `[50 PCA coefficients of ψ, q95, betaN]` — from the same
scaled feature vector, which is what makes them comparable and averageable. Two keys under
`features:` decide what those two things are, and both default to the behaviour that predates them
(see "The flux decomposition"):

- `subtract_coil_field` — fit on ψ − ψ_coil and add the coil field back at inference;
- `inputs` — `currents` (the 21 shipped signals), `coil_pca` (the coil flux in its own principal
  directions, plus the signals that produce no poloidal flux), or `both`.


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

## The flux decomposition (measured, not yet wired in)

```bash
uv run python my_experiments/eda_coil_field.py        # -> results/eda_coil_field.png
```

ψ = ψ_coil + ψ_plasma, and the first term is a Green's function times a current, not a modelling
problem. Every ingredient ships on every row of every split — `coil_R/Z/width/height`,
`coil_input_column` to join them to `magnetics_*`, `efit_grid_R/Z` — so `coil_field.py` computes it
for DIII-D and MAST alike and nothing it reads is withheld at test time.

**The calculation checks out, to about 10%.** Fitted outside the plasma boundary with a per-frame
plasma filament and a per-frame constant projected out, the F-coils come out as a group at
**0.87–0.89** depending on the sample, and individually mostly between 0.7 and 1.2 — F6A and F6B,
the cleanest geometry of the set, at 1.01 and 1.02. Independently, the Green's function annihilates
the Grad-Shafranov operator to 5·10⁻³ of itself, which is the finite-difference truncation level.

Two things had to be right to get there, and both were wrong first:

- the flux per radian is `μ0 I sqrt(aR)/(π k) [(1-k²/2)K(k) - E(k)]` — dropping the `1/π` puts every
  gain at 1/π, which is exactly what the first fit reported;
- the stored ψ has the machine's own sign, `AXIS_SIGN` in `fusion_scoring/common.py` (DIII-D −1,
  MAST +1). `coil_field` folds it in, so its output is directly comparable to `efit_psirz`.

**Do not quote a calibration from ten shots.** The first fit here gave 0.9969 for the F-coils and
looked like a confirmation to three decimals. It was not: at 40 shots the same fit gives 0.911, and
giving each frame its own plasma amplitude — a single filament is a crude 1 MA plasma, and pooling
its amplitude pushes the misfit into the coils — brings it to 0.87–0.89 at any sample size tried.
The residual spread across individual coils (F4A at 0.63, F1A at 1.19) is the parallelogram
approximation on F5/F7, the coils sitting on the grid edge, and the vessel eddy currents nobody
ships.

**The plasma filament belongs to the calibration and nowhere else.** It reads `efit_r_axis` /
`efit_z_axis`, which are labels. It is in the design only because without it the coil gains launder
the plasma's field into themselves — a 1 MA plasma against ~140 kA·turn per shaping coil reaches
well past the boundary — and the first fit that omitted it returned gains scattered from −0.15 to
−0.57 and both signs.

**`ECOILA` is not identifiable, so nothing hardcodes it.** It ships in kA with the turn count not
folded in, `ECOILB` is a second co-located group that is not shipped at all, and a solenoid's field
over this grid is nearly degenerate with a constant offset: its gain reads +142, +128, +94 or −10
depending on the sample and on whether a per-frame constant is free. The pipeline therefore fits
its own gains at training time (`coil_field.fit_flux_gains`) and stores them in the artifact. That
is a different question with no true value to recover — *what linear-in-current field leaves the
model the least to learn* — and the split stays exact for any gains at all, since whatever is
subtracted is added back unchanged at inference.

### What it is worth

Four configurations × three MLP seeds × two scoring folds — 24 runs, all trained the same way
(`0.05/0.2 0.05/0.2`, 352 shots fitted, CatBoost disabled so the ensemble is the MLP alone), each
scored on 70 shots (`0.01`) and on 704 (`0.1`). S, mean of the three seeds:

| `subtract_coil_field` | `inputs` | 70 shots | 704 shots |
|---|---|---|---|
| false | currents | 0.9271 | 0.9327 |
| **true** | **currents** | **0.9405** | **0.9416** |
| false | coil_pca | 0.9336 | 0.9365 |
| true | coil_pca | 0.9350 | 0.9370 |

**Subtracting the coil field wins on every seed of both folds** — +0.0083, +0.0157, +0.0025 on the
large fold, +0.0114, +0.0279, +0.0011 on the small — and essentially all of it is Consistency,
0.7229 → 0.7639 averaged over seeds. That is the term the score is actually losing in, and it
moved while R²ψ did not. Per derived scalar the gain is broad rather than one lucky functional:
`Z_axis`, `li`, `volume` and `R_axis` all rise. Three seeds is three replicates, not six — the
same seed trains the same model on both folds — so a paired t gives p ≈ 0.15. The direction is
consistent; the magnitude is not pinned down.

**`coil_pca` is indistinguishable from the raw currents**, and the earlier claim that it *helped*
was noise: its per-seed differences against the baseline are +0.0102, −0.0042, +0.0052. It costs
nothing, which is what it had to show before MAST can be built on it, and that is all it shows.
The interaction that a single run suggested — that subtracting and `coil_pca` are substitutes and
hurt together — did not survive either: on the large fold that cell is 0.9370 against 0.9416 and
0.9365, inside the noise.

Ridge scores 0.7083 in every one of the twelve large-fold runs (0.7082 with `coil_pca`) and 0.7117
in every small-fold one. It should: a linear model cannot tell the difference between fitting ψ and
fitting ψ minus a linear-in-currents field that is added back afterwards, and `coil_pca` spans the
same subspace as the currents it is built from. That row is the check that the decomposition is
exact and the add-back is not quietly lossy.

### How precise the metric is, and what makes it imprecise

The spread of S over three MLP seeds, everything else fixed, pooled over the four configurations:

| scoring fold | pooled σ over seeds | wall per run |
|---|---|---|
| 70 shots (`0.01`) | **0.0060** | ~120 s |
| 704 shots (`0.1`) | **0.0060** | ~600 s |

**Ten times the shots bought no precision at all.** The noise is the MLP's optimisation, not the
choice of scored shots: early stopping lands anywhere between epoch 363 and 1037 across seeds, and
that is what moves the score. Scoring more shots cannot average it away — it is a property of the
model, not of the measurement.

The comparison is exact rather than approximate, because training is deterministic: one seed and
one configuration produce *the same model* in both grids, and only the scored shots differ. The
same model scored on 704 shots instead of 70 moves by +0.0029 ± 0.0039 — real, but smaller than
the seed spread and mostly a level shift, since the ranking of the four configurations is
**identical on both folds**.

So: `make quality` stays at `0.01`. Spend a bigger budget on seeds, never on shots — and treat any
difference under ~0.006 as unmeasured until it has been repeated.

What the decomposition is **not** worth: a more compact basis. The residual is genuinely simpler —
a single smooth blob following the plasma, against ψ's blob plus the coils' hot spots at the grid
edge — and at 10 components it shows: R²ψ ceiling 0.999978 against 0.999619. But `params.yaml` uses
50, where both targets are exact to six decimals. **The model's R²ψ = 0.9990 is therefore entirely
regression error and none of it is representation error.** Whatever this buys has to come from
making the regression easier, not from the basis.

Residual std is 81% of ψ's, so the subtraction removes about a third of the variance and does not
shrink the target much; the per-frame mean level, which is coil-driven, drops from 0.1453 to
0.0667.

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

**A low-coverage Ip axis is not the same thing as a wrong one.** The check above used to accept
the shipped axis when current flowed under 90% of the EFIT frames and force the correction below
that. On `d3d_shot_21f23b2392` — 1 shot in the 704 scored, 0 in 1104 sampled outside it — that
inverted the truth: it is a 0.5 ms shot whose shipped axis is correct, but its EFIT window is 15
frames over 160–440 ms sitting on the current ramp-up, where |Ip| is genuinely below 5% of its
peak for half of them. Coverage 0.53, "corrected" to 0.00, and the run died at load time.
`align_ip_times` now takes whichever of the two axes covers more frames and raises only if the
better one falls under 50%. Over 1104 shots that reproduces every previous decision exactly — 761
corrected, 342 left alone, none changed — and no shot comes near the floor.
`results/eda_bad_ip_shot.png` is the picture.

**Scoring a large fold used to need 71 GB.** `local_score` holds every scored shot for the whole
run, because the metric pools R² across the fold and nothing can be finalized shot by shot. It
kept the whole parquet row per shot: 101 MB, of which 77 are the raw magnetics traces — 480256
samples per signal at the 0.05 ms acquisition — whose only use is to be interpolated onto
`efit_times`, ~300 points, before anything reads them. `baseline_model.slim_row` now does that
interpolation once at load and keeps the (T, 21) result plus the coil geometry, and `local_score`
hands the pool 64 shots at a time instead of pickling all of them into the queue at once. The
marginal cost per shot went from 101 MB to 11.6 MB, so a 704-shot fold holds about 9 GB. The
metric is untouched — the smoke run scores the same to four decimals.

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
