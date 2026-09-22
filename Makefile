# make ci — everything that must pass before a change is done.
#
# Scope is this fork's own code: my_experiments/ plus the root scripts we edited. The organizers'
# files and the vendored scorer stay byte-identical to upstream, so they are excluded in
# pyproject.toml rather than fixed. See AGENTS.md.

# Local secrets, untracked. The leading `-` keeps every other target working without the file.
# Exported rather than passed on the command line, so the token stays out of `make`'s echo and out
# of the process list: submission_skeleton.py defaults --read-token to $HF_READ_TOKEN.
# bash and pipefail, both load-bearing. Every long recipe below pipes through `tee` so a run is
# watchable while it happens, and `cmd | tee` returns TEE's exit code — under /bin/sh a failed
# training run would report success. This fork has already been bitten by exactly that.
SHELL := /bin/bash
.SHELLFLAGS := -o pipefail -c

# Every run writes logs/<UTC timestamp>-<target>.log and the terminal at the same time, and points
# logs/latest.log at the newest, so `tail -F logs/latest.log` always follows whatever is running.
# CAPITAL -F: `tail -f` follows an INODE, so when the next run replaces the symlink it keeps
# watching the finished file forever and the terminal looks frozen.
# The directory is gitignored: these hold tqdm's carriage returns and run to megabytes.
LOG_DIR ?= logs
LOG_START = mkdir -p $(LOG_DIR); LOG=$(LOG_DIR)/$$(date -u +%Y%m%d-%H%M%S)-$@.log; \
            ln -sfn $$(basename $$LOG) $(LOG_DIR)/latest.log; \
            echo "--> $$LOG  (tail -F $(LOG_DIR)/latest.log)"
TEE = 2>&1 | tee -a $$LOG

-include .env
export HF_READ_TOKEN
# HF_TOKEN is NOT exported here, and that is the whole point. It exists to raise the anonymous rate
# limit on `download_dataset`, so it is exported by that target ALONE, below.
#
# Exported globally it silently breaks uploading. huggingface_hub resolves a token in this order:
# the HF_TOKEN environment variable first, the token stored by `hf auth login` second. So a
# read-only HF_TOKEN in the environment SHADOWS a perfectly good write login, and the push fails
# with "403 Forbidden: you must use a write token" — which reads as a problem with the account
# rather than with this line. Measured on 2026-08-15: it cost a failed push of a finished
# submission.
HF_REPO ?= pshishkin/fusion-eq-predictions

.PHONY: ci lint format typecheck test quality prod train eval predict_and_submit_to_hf clean \
        clean-cache warm-cache download_dataset sweep doctor experiment check-workflow

# Where the dataset lives. It sits BESIDE the repo, not inside it, so a `git clean` cannot delete
# 104 GB — and every `--local-data-dir` flag already defaults to this path.
DATA_DIR ?= ../downloaded_huggingface/hf_dataset
DATASET_REPO ?= Sophelio/fusion-equilibrium-challenge
DATASET_REVISION ?= 1e280905b85f2a6fdde7e06fca8cf3a1edf447cb
DATASET_CONFIGS ?= diii_d_train diii_d_public_test mast_public_test

# Which machines the submission carries. BOTH by default: one entry feeds both leaderboards, and a
# DIII-D-only file scores G_ratio = 0 on Challenge 2 however good it is on Challenge 1. Override
# with SUBMIT_CONFIGS=diii_d_public_test to send Challenge 1 alone.
SUBMIT_CONFIGS ?= diii_d_public_test mast_public_test
# Worker processes for prediction. The MAST solve is ~63 ms a frame on ONE core and the split is
# 1206 shots, so this is 12 minutes against 1.1 hours; `mast/predict.py` pins each worker to a
# single BLAS thread, which is what makes a pool of 28 an actual 28x.
SUBMIT_JOBS ?= 28

ci: lint typecheck check-workflow test

lint:
	uv run ruff check .

# Not part of `ci`: reformatting is a separate decision from a clean lint.
format:
	uv run ruff format .

typecheck:
	uv run mypy

# Everything the pipeline reads: 7041 DIII-D training shots (93.43 GB), the DIII-D public test set
# (7.19 GB) and MAST (3.24 GB) — 103.86 GB, so budget the disk before starting. No token needed; the
# dataset is public.
#
# Resumable: `hf download` skips files already present, so a killed run only has to be repeated.
# One config at a time rather than one call for all three, so a failure says which one died and a
# rerun does not re-scan the other two.
#
# Fetch less by naming what you want:
#   make download_dataset DATASET_CONFIGS=diii_d_train
#   make download_dataset DATASET_CONFIGS="diii_d_train diii_d_public_test"
#
# A DIII-D submission needs diii_d_public_test; training needs diii_d_train.
# mast_public_test belongs to Challenge 2 and is not used by the DIII-D runner.
# The one target that wants HF_TOKEN, and the only one that gets it — see the note beside
# `-include .env` on why a global export breaks the submission push.
download_dataset: export HF_TOKEN := $(HF_TOKEN)
download_dataset:
	@for cfg in $(DATASET_CONFIGS); do \
		echo "==> $$cfg"; \
		uv run hf download $(DATASET_REPO) --repo-type dataset --revision $(DATASET_REVISION) \
			--local-dir $(DATA_DIR) --include "data/$$cfg/*" --max-workers 16 || exit 1; \
	done
	@du -sh $(DATA_DIR)/data/* 2>/dev/null || true

# Standard shares are recorded in experiments/protocols.json. Smoke is one MLP; quality uses
# ridge + one MLP on production data; production uses the full enabled zoo. The separate paired
# protocol preserves the historical lane's 5% scored tail. Every new run freezes its code/config.
# JOBS controls per-shot work and never the fitted model's arithmetic.
JOBS ?= 24
TORCH_ENV = OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4

# Every standard run snapshots code and effective config before fitting. The returned campaign
# path is content-addressed; repeating the command resumes only identical, verified work.
test quality prod:
	@spec=$@; [ "$$spec" != test ] || spec=smoke; [ "$$spec" != prod ] || spec=production; \
	campaign=$$(uv run python -m my_experiments.run prepare experiments/$$spec.json --jobs $(JOBS)) && \
	uv run python -m my_experiments.run execute "$$campaign" --data-dir "$(DATA_DIR)"

check-workflow:
	uv run python -m unittest discover -s tests -v

doctor:
	uv run python -m my_experiments.run doctor --data-dir "$(DATA_DIR)"

experiment:
	@test -n "$(SPEC)" || { echo "SPEC=experiments/example.json is required"; exit 2; }
	@campaign=$$(uv run python -m my_experiments.run prepare "$(SPEC)" --jobs $(JOBS)) && \
	uv run python -m my_experiments.run execute "$$campaign" --data-dir "$(DATA_DIR)"

# A pre-declared grid of configurations, one at a time, collected into a CSV. Each run tees into
# its own logs/<timestamp>-<name>.log and repoints logs/latest.log, so the sweep is watchable the
# whole way through. Resumes: a name already in OUT is skipped, so a killed sweep costs one run.
#
#   make sweep GRID=grids/lr.json OUT=results/lr.csv
sweep:
	@$(LOG_START); \
	test -n "$(GRID)" || { echo "GRID= is required"; exit 2; }; \
	$(TORCH_ENV) uv run python my_experiments/sweep.py $(GRID) $${OUT:-results/sweep.csv} $(TEE)

# Full frames, not a fifth of them: `seq` is enabled in params.yaml and train.py refuses to fit a
# sequence model on a thinned clock. Fewer shots instead, so the target stays a quick one.
train:
	uv run python my_experiments/train.py --share 0.01/1.0 --val-share 0.01/1.0

eval:
	uv run python my_experiments/evaluate.py --share 0.01

# Predict every public-test shot with the trained artifact, push the result to Hugging Face and
# write the pointer zip to upload to Codabench. Structure validation runs in between; scoring does
# not — run `make eval` (or a full `make prod`) yourself if you want to know what you are sending.
#
# The token comes from .env, which is NOT in git — this file is, and the fork is public.
# SUBMIT_DIR= gives this build a directory of its own for the .npz. Two builds running at once
# share `submission/` by default and overwrite each other's files: on 2026-09-04 a single-salt and a
# three-salt build ran together and only the accident that their phases did not line up kept one
# from pushing the other's half-written array. The pointer zip names a pinned HF commit rather than
# a local path, so the finished submission was safe — but that is luck, not a guarantee.
#
# MODEL= picks which member of the zoo is submitted; without it the artifact's own ensemble goes.
# `MODEL="salts:a+b"` averages the decoded maps of several artifacts, which is how an ensemble
# across feature sets is sent — the members are baseline_<name>.joblib beside FUSION_ARTIFACT.
# FUSION_ARTIFACT defaults to my_experiments/baseline.joblib, which EVERY training run and every
# sweep entry overwrites — so a submission built without setting it ships whatever experiment
# finished last. That happened on 2026-08-19: a sweep left a single-MLP fit there and two
# submissions carried it, and Challenge 1 went 0.9938 -> 0.8316 on the board. The guard below
# refuses to build unless FUSION_ARTIFACT is set explicitly. To send the production entry:
#
#   FUSION_ARTIFACT=artefacts/candidate/baseline_prod.joblib \
#     make predict_and_submit_to_hf MODEL="salts:prod+vesseld+vesselp"
predict_and_submit_to_hf:
	@test -n "$$FUSION_ARTIFACT" || { \
		echo "FUSION_ARTIFACT is unset, so this would ship my_experiments/baseline.joblib —"; \
		echo "whatever the last training run or sweep entry left there, NOT the production fit."; \
		echo "This exact mistake cost Challenge 1 0.9938 -> 0.8316 on 2026-08-19."; \
		echo "Set it explicitly, e.g.:"; \
		echo "  FUSION_ARTIFACT=artefacts/candidate/baseline_prod.joblib \\"; \
		echo "    make predict_and_submit_to_hf MODEL=\"salts:prod+vesseld+vesselp\""; \
		exit 2; }
	@test -z "$$HF_TOKEN" || { \
		echo "HF_TOKEN is in the environment. huggingface_hub prefers it over the token from"; \
		echo "'hf auth login', so a read-only one shadows your write login and the upload fails"; \
		echo "with 403. Unset it for this command:  env -u HF_TOKEN make predict_and_submit_to_hf"; \
		exit 1; }
	@test -n "$$HF_READ_TOKEN" || { \
		echo "HF_READ_TOKEN is empty. Copy .env.example to .env and fill it in."; exit 1; }
	uv run python submission_skeleton.py --max-shots 0 --source local \
		--configs $(SUBMIT_CONFIGS) --jobs $(SUBMIT_JOBS) --repo $(HF_REPO) \
		$(if $(SUBMIT_DIR),--out $(SUBMIT_DIR)) \
		$(if $(MODEL),--model "$(MODEL)")

clean:
	rm -rf .ruff_cache .mypy_cache my_experiments/__pycache__ mast/__pycache__ toolkit/__pycache__ __pycache__

# The decoded-shot cache. It invalidates itself when the code that fills it changes — the key is a
# hash of that code plus the source parquet's size and mtime — so this is for the case the hash
# cannot see, such as a numpy upgrade that decodes a value differently. Rebuilding costs the read
# time of whatever the next run touches, and nothing else.
clean-cache:
	rm -rf .shot_cache

# Decode every training shot once, so no later run pays for parquet. ~26 GB and a few minutes.
# Worth it before a sweep over split.salt: the cache is keyed by shot, but the salt decides which
# shots the training window pulls in, so a new salt otherwise re-reads whatever salt 0 never saw.
warm-cache:
	uv run python my_experiments/warm_cache.py
