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

Hypotheses that did not survive are recorded, so they do not get re-litigated. Already tested and
**refuted**:

- Thomson does not explain the model's failures (R² correlates −0.07 with Te, 0.00 with ne, 200 shots).
- The error on failing shots is not merely an offset in the psi level (removing the per-shot
  constant recovers only 21% of it).
- Normalizing polarity by the sign of Ip makes the score worse (0.63 against 0.68).
- Standardizing each PCA component separately is not a neutral preprocessing step: it whitens the
  loss and costs the ensemble 0.7466 → 0.7155 against scaling the block as the metric weights it.
- Averaging ridge into the ensemble does not diversify it, it dilutes it (0.7466 → 0.5936).
- Early stopping buys CatBoost nothing measurable: 0.7435 to 0.7439 across every budget and
  patience tried, while its chosen stopping point moved from 1942 to 4861 iterations.
- For the MLP the patience IS the hyper-parameter, not the stopping: 0.5521 at patience 20 against
  0.7273 at patience 100 on the same shots, because its validation curve has a false floor and 20
  epochs is only ~640 Adam steps. A validation window and a patience are one decision, not two.
- A falling validation MSE does not imply a rising composite: the loss sees 52 scaled targets, the
  score also sees D_LCFS and the seven derived scalars.
- **Confirmed, and the largest effect measured so far:** at a fixed row budget, more shots with
  thinned frames beat fewer shots with every frame — 0.8111 -> 0.9319 for the ensemble, and ridge
  gains as well, so it is coverage and not capacity. Frames inside a shot are near-duplicates;
  53% of the psi variance is between shots. Spend the budget on shots.

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

Two commands, and which one is being quoted must always be said.

```bash
# 1. quality — the number that means something
uv run python my_experiments/train_eval.py 0.05/0.2 0.05/0.2 0.01

# 2. smoke — does it work at all, and how fast
uv run python my_experiments/train_eval.py 0.01/0.2 0.01/0.2 0.002
```

`make quality` and `make test` (the latter is what `make ci` runs). The smoke run is 70 shots
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
