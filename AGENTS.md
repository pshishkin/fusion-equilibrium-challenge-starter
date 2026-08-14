# Conventions for this fork

Rules the code under `my_experiments/` holds to. Read this before adding anything.

## English only

All code, comments, docstrings, error messages, progress labels, commit messages and repository
documents are in English. The conversation around the work can be in any language; the artifacts
cannot. This fork is public and descends from an English project, so a mixed-language repository is
unreadable to anyone but its author — including to the organizers, if a data question ends up in an
issue quoting our code.

## Git: never commit or push unasked

`git commit` and `git push` run only when the current message explicitly asks for them. Consent
does not carry over from a previous one: "let's do X" or "fix Y" authorizes editing the files, not
committing them. When the work is done, leave the tree dirty, say what is ready, and ask.

Pushing is outward-facing — it lands on a public fork that the organizers may read — so it is the
author's call, not the assistant's.

## Keep README.md and AGENTS.md current, in the same change

A change is not finished while these two files still describe the world as it was. Before calling
work done, reread both and update whatever the change invalidated — in the same commit, not later:

- **README.md** — the workflow commands, the numbers quoted for where the baseline stands, the list
  of known traps, and what is and is not implemented.
- **AGENTS.md** — a new convention agreed during the work, or an old one the change contradicts.

Numbers get stale silently, which is the worst kind: a README claiming `S = 0.4550` after the split
changed is not out of date in an obvious way, it is simply wrong, and the next person compares
against it. When quoting a score, say which command produced it.

This rule exists because it was broken: the plasma-current time base was corrected in code while
README kept saying "Not yet corrected in this fork" — right underneath the section describing the
defect.

## Fail fast, never quietly repair

The competition data is full of traps: shifted time bases, gaps encoded as zeros, a channel count
that varies per shot, quantities that look like inputs but are targets. Defensive code of the
"take the minimum" or "substitute the mean" kind **hides** those traps: training completes, the
metric prints, and nothing signals that a third of the features are garbage.

So, in `my_experiments/`: **a mismatch against expectations is an exception, not a reason to
continue.**

Specifically banned:

| instead of this | do this |
|---|---|
| `T = min(len(a), len(b))` | `if len(a) != len(b): raise ValueError(...)` |
| `if col not in row: continue` | `raise ValueError(f"no column {col}")` |
| `np.nan_to_num(x)` | `if not np.isfinite(x).all(): raise ValueError(...)` |
| `if mask.sum() >= 10: fit() else: mean` | fit on everything; too little data is an exception |
| `n = min(n_pca, available)` | `if n_pca > available: raise ValueError(...)` |
| returning zeros for an unsupported case | `raise NotImplementedError(...)` |
| `print("warning: ...")` and carry on | `raise` |

The message must carry **the filename or column and both numbers** that disagreed — on a corpus of
a few thousand shots an assertion without coordinates is not actionable.

The one deliberate exception is `submission_skeleton.your_model_predict`, which falls back to zeros
when no model has been trained yet, so the build pipeline can be exercised before a model exists.
Even it catches only `ImportError` / `FileNotFoundError` — "no model", never "broken model".

Why: over the course of this fork, silent branches hid the missing `efit_psirz` on test rows, the
three-second offset of the plasma-current axis, and a shape helper truncating psi to `(T, 65)`.
Each was found by accident, hours after it had already influenced the numbers.

## Measure, do not reason

Any claim about the data is backed by a measurement over a sample of shots, and the message quotes
the number. "The current seems to always be positive" is not a conclusion; "reversed current in 21
of 200 shots" is.

**Where the record lives.** Three files, and putting a thing in the wrong one is how it gets lost:

| file | what belongs in it |
|---|---|
| [README.md](README.md) | what the fork **is** — the workflow, the current numbers, the data traps, what is and is not implemented |
| [AGENTS.md](AGENTS.md) | what the rules **are** — conventions, and only conventions |
| [experiments_history.md](experiments_history.md) | what we **learned** — one line per experiment, dated, with its number and a verdict |
| [ideas.md](ideas.md) | what nobody has measured yet — dated, and explicitly not claims |

An idea starts in `ideas.md`. When it is measured it moves to the history file with a number
attached, whichever way the number went. **Hypotheses that did not survive stay recorded**, so they
do not get re-litigated — this fork has already believed and un-believed several things, and the
record is what stopped them coming back a third time.

Do not restate an experiment's numbers in README or AGENTS. Reference the entry.

Three rules of measurement, which came out of experiments and are now conventions:

- **Replicate over SPLIT SALTS, not over model seeds.** `split.salt` reshuffles which shots train,
  validate and score; the MLP seed only varies the optimisation. Ridge is deterministic given a
  split and moves 0.7117 → 0.6219 → 0.5067 across salts 0, 1, 2, against 0.0 for any seed change.
  Compare configurations PAIRED within a salt; never compare absolute scores between salts.
- **Read every difference against the measured noise, not against a habit.** At production the
  seed-to-seed sigma is 0.0013 of S for a single net and about half that for the four-seed
  ensemble that production now uses — the full table, per term and per scalar, is in experiments_history. It is very
  uneven: nearly all of it lives in Consistency, and inside that in `R_axis` and `Z_axis`.
  More scored shots do not help — the noise is the fit, not the fold.
- **Quality screens, production decides.** See "How we test the metric".

## The split

Shots are ordered by `sha1` of the filename (`baseline_model.sorted_shots`). Training takes the
head of the list, **validation the window right behind it**, scoring the tail; the sizes are given
as shares. Never the builtin `hash()`: it is salted per process, and the split would stop being
reproducible — silently.

Validation is on the head side because it is part of the fit: early stopping reads it every
iteration, so those shots are seen. Anything that measures generalization comes from the tail and
nowhere else.

Non-overlap is checked against the **filenames** stored in the artifact — both `train_files` and
`val_files` — not by index arithmetic, so the check stays correct when the data directory grows.

## How we test the metric

Three commands, and which one is being quoted must always be said.

```bash
# 1. production — what a submission is built from, and what DECIDES anything. FOUR nets.
make prod       # train_eval.py 0.60/0.1 0.15/0.1 0.01 --jobs 24

# 2. quality — a SCREEN at a quarter of the cost, on production's OWN data. ONE net.
make quality    # train_eval.py 0.60/0.1 0.15/0.1 0.01 --only ridge mlp --jobs 24

# 3. smoke — does it work at all, and how fast. ONE net.
make test       # train_eval.py 0.01/0.2 0.01/0.2 0.002 --only mlp --jobs 24
```

**The screen and production now differ in ONE thing: one net against four. The data is
identical.** That is deliberate, and it repairs the failure this section was written about. The
reversal below — `n_pca = 30` beating 50 at quality on three salts, then losing at production — was
caused by the screen's 352 shots, not by its ensemble: at that scale the tail PCA components are
noise the model cannot predict, and at 4225 they are signal. A screen run on production's own data
cannot make that class of mistake.

What the screen still costs is noise. The seed sigma is 0.0013 of S for one net against about half
that for four, so a difference read at quality must clear twice the floor a production run's does —
and a change that helps the ensemble by averaging away seed variance will not show here at all.
**Quality numbers recorded before 2026-08-14 came from 352 shots and four nets, and compare to
neither of these.**

`--only` and `--salt` override params.yaml for one run. That is not a second home for
hyper-parameters — they have no defaults of their own, they are printed at the top of the run, and
the artifact stores the EFFECTIVE configuration rather than the file's text, so a screening run
still says exactly what produced it.

**Quality screens; production decides. A change that measures well at quality is a candidate, not
a result, and nothing reaches a submission without a production-scale confirmation.**

This is not caution, it is measured. `n_pca = 30` beat 50 at quality by +0.0043 on all three salts
with t = 4.84 — as clean a result as anything here — and at production 50 beat 30. The reversal
has an ordinary cause: with 352 shots the tail PCA components are noise the model cannot predict
and the loss over-weights, and with 3168 shots they become signal, so truncating throws it away.
The optimal capacity grows with the data, and every conclusion drawn at a tenth of the data
silently assumed it does not.

The corollary is uncomfortable and worth stating: several results in README were measured at
quality scale and have NOT been confirmed at production. Where that is so, say so.

`make prod`, `make quality` and `make test` (the last is what `make ci` runs). The production run
is what the artifact behind a submission must come from: 4225 shots to fit, 1056 to stop on, 70
scored, about 5 minutes with the shot cache warm. Measured there, with CatBoost disabled so the
ensemble is the MLP alone, with the coil field subtracted and the Jacobian loss metric:

```
             model         S    R2_psi     R2_qb   1-D_LCFS      Cons
               mlp    0.9875    0.9994    0.9905     0.9860    0.9530
          ensemble    0.9914    0.9997    0.9937     0.9879    0.9685
             ridge    0.7022    0.8076    0.5461     0.9513    0.4050
```

Salt 0; the same recipe gives 0.9889 on salt 1 and 0.9898 on salt 2. The fork sat at
0.9677 before the coil-field decomposition and the Jacobian loss, 0.9769 on 1408 shots, and 0.9809 with a
256x256 MLP and no Thomson features until 2026-08-13. Consistency carries essentially all of the
movement.

`loss.jacobian_delta` is SCALE-DEPENDENT and the one number here that has to be revisited: the
probe has to sit at the error the model actually makes, so it is sqrt(1 - R2_psi) of the run it is
used in — 0.05 at quality scale (R2_psi 0.997) and 0.033 at production (0.9989). Refitting it on
the production model moved S by +0.0029. Against the paired sd of 0.0013 measured later that is
about two sigma — a real gain, not the "unmeasured" it was first recorded as, because the floor it
was read against at the time was five times too big.

The MLP fits 62968 frames in 266 s and stops at epoch 767 of the 868 it ran. Note how far this is
from the quality run (0.9319) on a fifth of the shots: at this scale shots are still buying score,
so a submission is retrained at 0.2, never at 0.05. The smoke run is 70 shots
thinned to a fifth of their frames against 14 scored shots, about 2.5 minutes; the quality run is
352 shots and 70 scored. Any claim about a model scoring better than another comes from the
quality run — the smoke one fits 3272 rows and exists to prove the chain works: shot ordering, the
three-way split and its overlap check, feature building, frame thinning, every model in
params.yaml with its early stopping, the artifact, inference, and the real metric.

Do not silently substitute other shares. If a run at a different scale is wanted, ask first.
Numbers from different shares are NOT comparable — the scored fold changes with the test share,
and ridge alone moved 0.1157 -> 0.3693 across one such change.

History, because the numbers in README carry it: `0.01 0.001` (two shares, no validation) ->
`0.01 0.002 0.001` (validation added for early stopping) -> `0.01 0.01 0.002` (wider validation)
-> the pair above, once frame thinning showed that shots buy more than frames do.

## Every function is annotated

`mypy` runs with `disallow_untyped_defs`, so a new function without annotations fails CI. Arrays
are `FloatArray` (`my_experiments/models.py`); a dataset row, which is a pandas Series or a
streamed dict depending on the source, is `Row` (`baseline_model.py`).

## Hyper-parameters live in params.yaml

Not in argparse defaults, not in the code. `models.py` builds each model straight from the file's
keys, so an unknown one is a `ValueError` naming the file and the key rather than a setting that
silently does nothing. A new model type is a `TargetModel` subclass plus one line in
`MODEL_TYPES` — everything downstream (training, the artifact, `--models`, the ensemble) picks it
up from there.

Models are compared, never quietly replaced: `evaluate.py` scores every member on the same
held-out shots and prints them side by side, because "the new model is better" is a claim about
one number against another, on one split.

## Per-shot work is parallel, and that must not show in the numbers

Shots are independent, so both halves that iterate over them run on one shared process pool
(`my_experiments/parallel.py`, `--jobs`, 0 = cores - 2, 1 = serial): reading them in
`baseline_model` and, in `local_score`, extracting the LCFS and the seven functionals.

Three rules keep it honest. Results come back in SUBMISSION order, so float sums accumulate
exactly as they did serially and a parallel run is bit-identical to a serial one — the score must
not depend on the core count. The pool is spawned, not forked, because CatBoost's and torch's
thread pools are alive by scoring time and forking on top of a live OpenMP pool is a documented
way to hang. And there is exactly one pool for the whole run: a pool per phase paid spawn startup
five times over and gave back most of the speed-up.

New per-shot loops go through `pimap`, not through a private pool.

## The core count must not change the score. The device does, and that is a different rule

The paragraph above is about *parallelism*, and it stays absolute: `--jobs` must never move a
number. `device` is not parallelism, and cannot be held to it — a GPU reduces in a different order
than a CPU does, Adam amplifies that round-off to the size of `lr` within a few dozen steps, and no
two such runs agree on the weights afterwards. So:

- **Do not check a device change for bit-identity.** It will fail, and failing means nothing. Check
  the arithmetic instead, where round-off has not yet been amplified: the forward pass and the
  gradient at initialisation, against the same net on the CPU.
- **Then check the SCORE against the measured noise** — paired within a salt, read against the
  0.0009 seed sigma at production, exactly as any other change is.
- **Keep a deterministic control in the run.** `ridge` is fitted on the same features and is
  unaffected by any of this, so it reproducing to four decimals is free evidence that the change
  touched only what it claimed to.
- **`device: auto` does not exist, on purpose.** A run whose arithmetic depends on which hardware
  happened to be free is not reproducible, and pinning `split.salt` and `features.pca_seed` was
  only worth doing because everything else about a configuration is fixed too.

Inference is exempt and stays on the CPU whatever `device` says: the artifact has to unpickle and
predict where a submission is scored, which is a machine this fork does not control.

## One implementation of the metric

Scoring always goes through `local_score.py`, called as a function. Do not write your own R² or your
own sums of squares: the competition metric lives in `fusion_scoring/` and is not obvious — R² is
pooled across the whole fold against a single scalar mean, while `D_LCFS` is averaged per shot.

## Progress through tqdm

One redrawing line per loop, not a line per shot. Over thousands of shots, per-line printing makes
the output unreadable and buries the messages that matter.

Periodic lines ALONGSIDE the bar are fine, and both long fits use them: CatBoost prints its own
loss every `iterations / 10` steps, and the MLP writes a train/val line every `MLP_LOG_EVERY`
epochs through `bar.write`. A postfix only ever shows "now" — it cannot show that the validation
loss turned around four hundred epochs ago, which is exactly what one wants to see in a fit that
runs for minutes. What stays banned is a line per shot in a loop over thousands of shots.
