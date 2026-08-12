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

## The split

Shots are ordered by `sha1` of the filename (`baseline_model.sorted_shots`). Training takes the
head of the list, evaluation the tail, and the sizes are given as `--share`. Never the builtin
`hash()`: it is salted per process, and the split would stop being reproducible — silently.

Non-overlap is checked against the **filenames** stored in the artifact, not by index arithmetic,
so the check stays correct when the data directory grows.

## How we test the metric

```bash
uv run python my_experiments/train_eval.py 0.01 0.001
```

This is the command. 70 shots to fit, 7 to score, about a minute end to end, and it exercises the
whole chain: shot ordering, the split's overlap check, feature building, the artifact, inference,
and the real metric.

Run it after touching anything on the train or score path, and report its numbers. Do not silently
substitute a larger share — if a run at a different scale is wanted, ask first.

## One implementation of the metric

Scoring always goes through `local_score.py`, called as a function. Do not write your own R² or your
own sums of squares: the competition metric lives in `fusion_scoring/` and is not obvious — R² is
pooled across the whole fold against a single scalar mean, while `D_LCFS` is averaged per shot.

## Progress through tqdm

One redrawing line per loop, not a line per shot. Over thousands of shots, per-line printing makes
the output unreadable and buries the messages that matter.
