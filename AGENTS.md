# Conventions for this fork

Rules the code under `my_experiments/` holds to. Read this before adding anything.

## English only

All code, comments, docstrings, error messages, progress labels, commit messages and repository
documents are in English. The conversation around the work can be in any language; the artifacts
cannot. This fork is public and descends from an English project, so a mixed-language repository is
unreadable to anyone but its author — including to the organizers, if a data question ends up in an
issue quoting our code.

## Git: never commit or push unasked

**`git commit` runs autonomously — standing authorisation given 2026-08-25.** Commit in coherent
units, with a message that says what was measured or fixed and why; the log reads as a record of
findings, not of file edits.

**`git push` is still a separate act and needs the current message to ask for it.** The fork is
PUBLIC. Before every commit, run `git grep -lE "hf_[A-Za-z]{30,}"` over tracked files: `.env` holds
the HF token and `submissions/` holds pointer zips whose `manifest.json` carries the read token, and
both must stay gitignored.

**Fitted artifacts and derived caches never go in.** `artefacts/*` and `.aux_targets/` are 14 MB and
up per file and every one is reproducible from `params.yaml` plus a salt. Note that some were
tracked before this rule existed, and `.gitignore` does not apply to a file already tracked.

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

## One package per challenge, and they do not import each other

`my_experiments/` is Challenge 1's, DIII-D. `mast/` is Challenge 2's, MAST. **Neither may import
the other, in either direction** — the check is `grep -r "my_experiments" mast/` and the reverse,
and both must be empty. They meet in exactly one place: `machines.py` at the repository root, which
reads a row's own `source` column and dispatches. `local_score.py`, `submission_skeleton.py` and the
Makefile go through that file and nowhere else.

Two packages belong to neither and exist so that nothing has to reach across the line for them:

- **`solver/`** — the free-boundary Grad-Shafranov solve. Every machine-dependent number it needs
  arrives through `machine.Machine`: the envelope mask, the inboard wall radius, the flux sign, the
  column names, the vertical pin. It carries a DIII-D entry, and that is not Challenge 1 work —
  three MAST demo shots have refuted three decisions fitted on them, so a choice for MAST has to be
  testable where there are 7041 labelled shots.
- **`toolkit/`** — the process pool and the progress bar. No knowledge of a machine, a model or the
  metric. It used to live under `my_experiments/`, which meant the two top-level files serving BOTH
  machines reached into Challenge 1's package to get it.

A machine constant lives in the package that owns that machine, as a literal with the measurement
that produced it beside it — `mast/calibration.py` is the pattern, and `solver/machine.py` is the
pattern for the handful the solver itself needs.

This is not tidiness. The two machines share one of twenty-one magnetics signals, a flux grid that
starts at R = 0.06 m against 0.84, and an aspect ratio of 1.3 against 2.7; MAST has no training
data at all, by design. A shared feature pipeline would be a shared set of assumptions, and every
one of them is false on one side or the other. A shared helper that "just needs a machine flag" is
the first step to a function whose MAST branch nobody reads while tuning DIII-D.

What they may share is the vendored scorer in `fusion_scoring/`, because both are graded by it.

Two things follow for anything new:

- **A physical constant that differs by machine belongs to that machine's package**, as a literal
  with the measurement that produced it beside it — `mast/calibration.py` is the pattern.
- **Documentation splits the same way.** `mast/README.md` carries what MAST is and what is left in
  it; the root README says where both challenges stand and points there. `ideas.md` ranks by ΔS,
  which is Challenge 1's currency, so MAST's open items are ranked in its own README instead.

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

**Where the record lives.** Keep current state, conventions and historical evidence separate:

| file | what belongs in it |
|---|---|
| [README.md](README.md) | what the fork **is** — the workflow, the current numbers, the data traps, what is and is not implemented |
| [AGENTS.md](AGENTS.md) | what the rules **are** — conventions, and only conventions |
| [experiments_history.md](experiments_history.md) | DIII-D: what we **learned** — one line per experiment, dated, with its number and a verdict |
| [ideas.md](ideas.md) | DIII-D: what nobody has measured yet — dated, and explicitly not claims |
| [experiment_status.md](experiment_status.md) | current DIII-D decisions and pointers to their evidence |
| [experiments_mast.md](experiments_mast.md) | MAST: the same record, kept separate because the two machines share almost no result |
| [ideas_mast.md](ideas_mast.md) | MAST: the same idea list, ranked against Challenge 2's own arithmetic |

**Sweeps can run two at a time, and the docstring that says otherwise is only half right.**
`sweep.py` refuses concurrency because "every run writes the same `my_experiments/baseline.joblib`"
— true, and the way round it is one line: it passes `{**os.environ, **ENV_CAPS}` to the subprocess,
so `FUSION_ARTIFACT=<some other path> make sweep ...` gives the second lane its own artifact. What
actually bounds it is memory. Resolve the current cgroup from `/proc/self/cgroup` and its mount
before reading limits: v2 uses `memory.max`/`memory.current`, v1 uses
`memory.limit_in_bytes`/`memory.usage_in_bytes`. Usage includes page cache. Do not infer a
container limit from `free`, or reuse an old machine's limit. The former training container had
183 GB against a 251 GB host; those are historical measurements, not this checkout's resources.

Two lanes fit on that former container; measure available resources again on a new host. What does NOT fit is the wreckage of killed ones: `kill <lane script>`
leaves every worker pool its children spawned reparented to init, holding memory and pids for good.
On 2026-08-20 eight queue restructures leaked **734 workers holding 145 GB and 7000 of the 7680
pids**; nine runs died rc=137 and two live sweeps then wedged for 41 minutes on
`SemLock._rebuild → FileNotFoundError`, which is what pid exhaustion looks like. Clearing them took
the cgroup from 150 GB to 2 GB.

**Kill process groups, not scripts** — `setsid` at launch, `kill -- -<pgid>` to stop — and check
`ps -eL --no-headers | wc -l` against the current cgroup's `pids.max` before blaming memory. Anything that edits the training
path — `models.py`, `baseline_model.py` — must NOT be touched while a lane is queued: every sweep
entry re-imports it, so a bad edit lands in every remaining arm.

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

## How we run and compare experiments

`experiments/protocols.json` is the machine-readable source of data shares:

- Smoke: `0.01/0.2 0.01/0.2 0.002`, one MLP, correctness only.
- Production: `0.80/1.0 0.19/1.0 0.01`; `make quality` uses the same data with ridge and one MLP,
  while `make prod` uses the enabled ensemble, including the sequence model.
- Paired lanes: `0.80/1.0 0.15/1.0 0.05`, the existing lane protocol, paired within split salt.

Do not silently substitute other shares. A change of shares requires the user's agreement.
Identify the protocol, command, effective config and scored filenames when quoting a score.
Quality is a screen; confirmation uses the intended production ensemble. New salts do not make
reused shots an independent dataset. Selection and confirmation must be separated in advance.

Use `python -m my_experiments.run prepare` to freeze a campaign before executing it. Each campaign
records source hashes, package versions, complete effective YAML and its data protocol. Do not edit
its snapshot or configs; prepare a new campaign for a change. Editing the working tree while a
FROZEN campaign runs is allowed because its workers import the snapshot. This exception does not
apply to the old shell lanes, which re-import the working tree.

Each arm/salt owns its artifact, log and scorer-native JSON. Resume requires a successful completion
record with matching artifact/config/metrics hashes, never just a filename or CSV row. Preserve
failed logs and terminate process groups together. New orchestration must not parse printed scores.
Record the host boot and PID namespace with process IDs; a hidden worker is not a dead worker.
The scorer's JSON `dlcfs` is the distance; legacy CSV `D_LCFS` contains `1-D_LCFS`.

Auxiliary target caches record full-shot frame counts and use the same per-shot thinning as training.
They may be shared by snapshots only when producer code, dependency lock and input identities match.
Missing caches are built explicitly through `pimap`; incomplete files are never published as ready.

The pinned DIII-D manifest must match the complete local corpus before a fit. Missing, unexpected,
wrong-sized or invalid parquet files are errors. Full SHA-256 verification is available separately.
An explicit CUDA configuration requires CUDA; never quietly switch to CPU. A sandbox may restrict
GPU access, so distinguish process visibility from a host driver failure.

`experiment_status.md` indexes current decisions and references historical evidence. Keep measured
nulls, uncertainty and user acceptance decisions distinct. The submitted reference manifest and the
iteration recipe are separate: without original artifacts and their hashes, do not claim that a
current config reproduces a historical leaderboard submission.

A candidate's individual score, replacement effect and contribution when ADDED to the baseline
ensemble are different measurements. Report the baseline, candidate and augmented ensemble on the
same unseen shots with explicit artifact weights. A cross-salt ensemble requires a common holdout
excluded before every fit; the ordinary per-salt tail is not enough.

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
(`toolkit/parallel.py`, `--jobs`, 0 = cores - 2, 1 = serial): reading them in
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

## Every run is watchable while it runs

Every long recipe tees into `logs/<UTC timestamp>-<target>.log` and repoints `logs/latest.log` at
the newest, so `tail -F logs/latest.log` follows whatever is training right now. **Capital -F**:
`-f` follows an inode, so when the next run repoints the symlink it goes on watching a
file that will never grow again, and the terminal looks frozen while the fit runs. `logs/` is
gitignored; the grids under `grids/` are not, because a grid file IS the pre-registration and a
description in experiments_history can be rewritten afterwards while a committed JSON cannot.

Three details, each of which had to be got right for this to be worth having:

- **`SHELL := /bin/bash` with `-o pipefail`.** `cmd | tee` returns TEE's exit code, so under
  `/bin/sh` a failed training run reports success. That is not hypothetical — it is how the first
  crash of the GPU work looked like a clean run.
- **Write to the file descriptor, not to a buffer.** `subprocess.run(capture_output=True)` produces
  the same file at the end and nothing to watch in the meantime, which defeats the point.
- **Point `latest.log` at a run BEFORE starting it**, not after it finishes.

## Progress bars are for terminals; logs get a slower one

`toolkit/progress.bar_kwargs` thins a tqdm bar when stderr is NOT a terminal — every 1000
shots for the reading and scoring loops, every 50 epochs for the fit. On a terminal nothing
changes, because a redrawing bar is what makes one useful to a human.

Two traps, both measured on a smoke run that started at 788 KB of log:

- **`mininterval` must go to 0 alongside `miniters`.** tqdm redraws when EITHER threshold is met,
  so the default 0.1 s keeps the per-second spam whatever `miniters` says.
- **`bar.set_postfix()` defaults to `refresh=True`** and forces a redraw every single call,
  quietly defeating both. It is the larger of the two effects, and it has to be fixed at EVERY
  call site: patching the MLP's alone left the shot reader spamming, which was 2835 of the 2849
  segments in a production log. `grep -rn set_postfix` before believing a bar is quiet.

## Progress through tqdm

One redrawing line per loop, not a line per shot. Over thousands of shots, per-line printing makes
the output unreadable and buries the messages that matter.

Periodic lines ALONGSIDE the bar are fine, and both long fits use them: CatBoost prints its own
loss every `iterations / 10` steps, and the MLP writes a train/val line every
`MLP_LOG_EVERY_STEPS` through `bar.write`. A postfix only ever shows "now" — it cannot show that
the validation loss turned around twenty thousand steps ago, which is exactly what one wants to
see in a fit that runs for minutes. What stays banned is a line per shot in a loop over thousands of shots.
