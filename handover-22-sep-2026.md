# Handover — 22 September 2026

## Scope and user intent

The user is Pavel and speaks Russian. Repository artifacts must be in English.
Modeling work is **DIII-D only**. The user moved this project from Claude and requested the
competition data, a review of the history, and improvements to iteration workflow. They then
explicitly authorized all five proposed workflow improvements; those are implemented.
This handover is documentation, not authorization to start a new modeling sweep.

The user confirmed their last leaderboard score was **0.9941**, submitted **4 September 2026**.
Do not confuse that score with a local smoke result or claim to have reproduced the submission.

Read `AGENTS.md` before editing. Commits have standing authorization; pushes require an explicit
request in the current message. No push or new submission was made during this work.
Scan tracked files for tokens before every commit. Never commit fitted artifacts, caches, `.env`,
pointer submission ZIPs or download logs. Preserve the DIII-D/MAST package boundary.

## Starting state

Implementation commit: **`c61e03c` — Freeze DIII-D runs and verify resume, scoring and dataset identity**.
The working tree was clean after that commit. This handover is a subsequent documentation change.
No training campaign or dataset download was left running at the end of the implementation.
Recheck current state rather than treating this dated statement as a live process monitor.

Repository: `/home/pavel/fusion-equilibrium-challenge-starter`.

Read in this order:

1. `AGENTS.md` — conventions and accepted protocols.
2. `experiment_status.md` — current decisions and pointers to evidence.
3. `README.md`, especially “Reproducible DIII-D iterations”.
4. `experiments_history.md`, `backlog.md`, `ideas.md`, `single_model_results.md`,
   `single_model_hypotheses.md` — historical findings; old “running” labels are historical.
5. `results/workflow_20260922.json` — machine-readable evidence for the completed workflow checks.

## Two different baselines

- `baselines/diii_d_20260904.json` records the user-reported submission and the inferred recipe.
  Scripts name salts 1/2/3, ten artifacts per salt, and 26 model weights per salt (78 total).
  The original artifact hashes, submission identifier and exact submitted code revision are
  unavailable. The script composition is evidence of a recipe, not proof of the uploaded bytes.
- `baselines/iteration.params.yaml` is a frozen copy of the checkout's configuration on this date.
  It retains the accepted static consistency head and `cons_split`; sequence LR schedule is `none`.
  **No fresh production score has been measured for this control.**

Standard experiment specs point to the frozen iteration YAML. Editing root `params.yaml` alone
therefore does not change `make test`, `make quality` or `make prod`. Use an explicit experiment
spec/override or deliberately introduce a new control; do not mutate an already prepared campaign.

## Data and environment are ready

Dataset directory: `/home/pavel/downloaded_huggingface/hf_dataset` (outside the repository).
Repository: `Sophelio/fusion-equilibrium-challenge`.
Pinned revision: `1e280905b85f2a6fdde7e06fca8cf3a1edf447cb`.

All **9,121 parquet files, 103,859,905,394 bytes**, were downloaded and SHA-256 verified against
the Hugging Face manifest. DIII-D train has 7,041 shots; public test has 874. The other 1,206
files are MAST public test, downloaded as part of the complete competition dataset.
The DIII-D inventory is tracked in `my_experiments/diii_d_dataset.json`.
Local completion evidence is `logs/dataset-download-complete.json`; its content is also retained
in the tracked workflow result. Demo parquet LFS objects were restored and `git lfs fsck` passed.

- Python **3.12.14**, `.venv`, `.python-version` pinned to 3.12.
- `uv` and `git-lfs`: `/home/pavel/.local/bin` (add this directory to PATH if needed).
- `uv sync --locked --python 3.12` passed; **`uv.lock` was not changed**.
- GPU: **RTX 3090, 24 GB**, PyTorch **2.5.1+cu124**; driver access was verified.
- Host reports 32 logical CPUs. Standard runs use `JOBS=24` and four BLAS/OpenMP threads.
  Do not reuse the old container's 183 GB memory limit as a current resource measurement.

Network downloads of large Python wheels were unusually slow. Verified wheels remain under
ignored `logs/wheels/` and `logs/wheels-small/`; the installed environment is complete. No need to
repeat the workaround or change dependency versions.

**Sandbox caveat:** the agent sandbox can hide GPU devices and host processes. GPU fits and their
live process checks were run outside that sandbox with the execution tool's escalation mechanism.
A failed sandbox CUDA check does not mean the host driver is broken. Worker records now include
boot ID and PID namespace; inspect/resume live workers in their own namespace. Never silently
change `device=cuda` to CPU. An explicit CPU smoke spec exists if the user wants it.

## Implemented workflow

`my_experiments/run.py` provides:

- `prepare`: validates the spec, freezes all effective configurations and source files, records
  package versions and numerical environment, and atomically publishes a content-addressed campaign.
- `execute`: checks corpus/device, locks the campaign, executes snapshot code, owns a separate
  artifact/log/JSON result per arm and salt, and terminates worker groups on interruption.
- Verified resume: skips only successful runs with matching artifact/config/metrics hashes.
  Interrupted fits restart training; optimizer checkpoints are not implemented.
- `status`: reports recorded state and verifies completed artifacts.
- `summary`: pairs differences within salt and checks identical scored filenames; reports mean,
  SE and signs without inventing an acceptance threshold.
- `compare`: copies input artifacts and source, evaluates baseline, candidate, and their weighted
  decoded-map blend on the same unseen shots. Reports replacement and addition deltas separately.
- `doctor`: checks the pinned corpus and explicitly configured device; writes `logs/doctor.json`.

Scoring still uses `local_score.py` and the vendored scorer. JSON retains native results, including
per-functional detail and scored filenames; `train_eval.py` adds phase timings. No second metric
implementation was introduced. JSON `dlcfs` is distance, while legacy CSV `D_LCFS` means `1-D_LCFS`.

Head/auxiliary targets build automatically on a cold start through the shared process pool.
Their cache records full-shot lengths and applies thinning separately within each shot. The cache
key covers producer code, dependency lock and source identities. This repaired a cold-start gap
and a frame-alignment issue. Decoded-shot and auxiliary caches live in `.shot_cache/` and
`.aux_targets/`; campaign snapshots share them. Runtime artifacts remain under ignored `runs/`.

A single external artifact is now valid as `salts:<name>*<weight>` so a candidate can be evaluated
alone beside a blend. Its training AND validation filenames still enter the overlap check.
Legacy shell scripts now locate the repo relative to themselves instead of `/workspace/...`.
Small lint/type fixes were also made in historical diagnostics; no MAST modeling change was made.

## Commands and protocols

```bash
export PATH="$HOME/.local/bin:$PATH"
cd /home/pavel/fusion-equilibrium-challenge-starter
make doctor
make test
make check-workflow
make ci

# Prepare a proposal without fitting it:
uv run python -m my_experiments.run prepare experiments/paired-example.json
# Use the exact printed path in subsequent commands:
uv run python -m my_experiments.run execute runs/<campaign>
uv run python -m my_experiments.run status runs/<campaign>
uv run python -m my_experiments.run summary runs/<campaign> --control base
# Convenience wrapper, which DOES execute:
make experiment SPEC=experiments/paired-example.json

tail -F logs/latest.log
```

`experiments/paired-example.json` is an **unexecuted proposal** to compare sequence cosine with
the frozen control over salts 0/1/2. Preparing it is not evidence of a modeling gain.

| Profile | Train shots/frames | Validation shots/frames | Scored shots |
|---|---|---|---|
| Smoke | 0.01/0.2 | 0.01/0.2 | 0.002 |
| Production / quality | 0.80/1.0 | 0.19/1.0 | 0.01 |
| Paired lanes | 0.80/1.0 | 0.15/1.0 | 0.05 |

Smoke fits one MLP. Quality fits ridge and one MLP. Production uses the enabled ensemble, including
the sequence model. Do not silently change these shares. Quality screens; the intended production
ensemble decides. Compare configurations within the same salt and identical scored filenames.

`run compare` takes `--baseline <artifact>... --weights <weight>... --candidate <artifact>
--candidate-weight <weight> --share <share> --jobs 24`, optionally `--on-holdout`.
Choose weights to reflect the actual artifact composition, not just the number of artifact files.
Cross-salt blends need a common holdout excluded before ALL fits; changing salt does not create an
independent test dataset. A trained member's ordinary tail may overlap another member's training.

## Completed validation and local artifacts

Final `make ci` passed: ruff, mypy over 79 source files, 11 workflow tests and real CUDA smoke.
Perfect/zero scorer checks on `parquet_data/d3d_shot_203702.parquet` returned exactly 1 and 0.

Three smoke fits used the standard protocol, salt 0, 70 train / 70 validation / 14 scored shots.
All returned **S = 0.721159**, with identical native score terms. This small-data correctness run
is NOT a production baseline, a regression estimate against 0.9941, or a new leaderboard result.

- Cold validation: `runs/smoke-08102de97736f89f/base_s0/`, 144.46 s child-process wall time.
- Warm validation: `runs/smoke-8be41a95897a47d1/base_s0/`, 91.03 s.
- Final CI artifact: **`runs/smoke-8003e11f09473ef8/base_s0/baseline.joblib`**.
  Its sibling `metrics.json`, `complete.json`, `params.yaml` and campaign `campaign.json` preserve
  results, hashes, effective parameters and provenance. Verified repeat `make test` took 1.24 s.
- Identity comparison: `runs/compare-1790061618595715235/comparison.json`. The same warm artifact
  was supplied as baseline and candidate; all three scores matched and both deltas were zero.
  This validates the comparison path, not a candidate improvement.

The earlier comparison directory `runs/compare-1790061552418782462/` is a preserved failed attempt
from before single-artifact selection was supported. Do not interpret it as a successful result.
The source hash changed between smoke fits as orchestration guards were fixed; use each run's
recorded source/config rather than treating names or Git HEAD alone as complete provenance.

Useful ignored logs: `logs/ci-final.log`, `logs/checks-final.log`, `logs/compare-validation.log`,
`logs/resume-final.log`, `logs/dataset-download-progress.log`. The portable checked-in record is
`results/workflow_20260922.json`. Download/install helper scripts in `logs/` are local recovery
artifacts, not the supported workflow interface.

## Next modeling work, when requested

1. Recover the original submitted artifacts if available, otherwise measure the frozen control at
   production scale and keep that provenance separate from the historical leaderboard reference.
2. Read the current evidence table in `experiment_status.md` before selecting an experiment.
   Sequence cosine has a positive historical paired result but is not enabled in the frozen control.
   Consistency reweight and the static head were accepted despite uncertainty; do not undo those
   decisions merely because their historical uncertainty intervals include zero.
3. Measure contribution to the actual ensemble, not only candidate-alone performance. PCA 70's
   unresolved replacement gain does not refute its usefulness as a blend member.
4. Other unmeasured directions already recorded: final refitting with fixed duration (requires
   agreement on changed shares), implemented geometry-loss options, and richer sub-frame features.
5. Do not reopen EMA/Huber, ordinary scalar auxiliary targets or blind width increases without
   reading the negative results. Predeclare screening and confirmation; do not repeatedly tune
   against the same tail and call a new salt independent confirmation.

No production retraining, model-improvement sweep, public submission, or push was performed.
