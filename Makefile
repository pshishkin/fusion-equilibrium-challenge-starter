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
        clean-cache warm-cache download_dataset sweep

# Where the dataset lives. It sits BESIDE the repo, not inside it, so a `git clean` cannot delete
# 97 GB — and every `--local-data-dir` flag already defaults to this path.
DATA_DIR ?= ../downloaded_huggingface/hf_dataset
DATASET_REPO ?= Sophelio/fusion-equilibrium-challenge
DATASET_CONFIGS ?= diii_d_train diii_d_public_test mast_public_test

ci: lint typecheck test

lint:
	uv run ruff check .

# Not part of `ci`: reformatting is a separate decision from a clean lint.
format:
	uv run ruff format .

typecheck:
	uv run mypy

# Everything the pipeline reads: 7041 DIII-D training shots (88 GB), the DIII-D public test set
# (6.7 GB) and MAST (3.1 GB) — 98 GB, so budget the disk before starting. No token needed; the
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
# A submission needs diii_d_public_test; training needs diii_d_train; mast_public_test is only for
# Challenge 2, which this fork does not implement yet.
# The one target that wants HF_TOKEN, and the only one that gets it — see the note beside
# `-include .env` on why a global export breaks the submission push.
download_dataset: export HF_TOKEN := $(HF_TOKEN)
download_dataset:
	@for cfg in $(DATASET_CONFIGS); do \
		echo "==> $$cfg"; \
		uv run hf download $(DATASET_REPO) --repo-type dataset \
			--local-dir $(DATA_DIR) --include "data/$$cfg/*" --max-workers 16 || exit 1; \
	done
	@du -sh $(DATA_DIR)/data/* 2>/dev/null || true

# The three standard runs. AGENTS.md, "How we test the metric".
#
#   test     does it work, and how fast — 70 shots thinned to a fifth of their frames, 14 scored
#   quality  what does it score — production's own data, 4225 shots, 70 scored, ONE net
#   prod     what a submission is built from — the same data, FOUR MLP seeds averaged
#
# `quality` and `prod` now differ in ONE thing: one net against four. The data is identical.
#
# That is the point. The old screen ran on 352 shots, and the reversal AGENTS.md records — n_pca 30
# beating 50 at quality by +0.0043 on three salts, then losing at production — was caused by that
# scale and not by the ensemble: with 352 shots the tail PCA components are noise, with 4225 they
# are signal. A screen at production's data cannot make that class of mistake, because the only
# thing left varying is the seed count.
#
# What the screen still costs is noise: the seed sigma is 0.0013 of S for one net against about
# half that for four, so a difference read here must clear twice the floor a production run's does.
# Quality numbers recorded before 2026-08-14 came from 352 shots and four nets, and compare to
# neither of these.
#
# JOBS is the reader/scorer pool. It is capped here because os.cpu_count() reports the HOST's
# cores, not this container's share, and the pids ceiling is shared with everything else running.
JOBS ?= 24
TORCH_ENV = OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4

test:
	@$(LOG_START); \
	$(TORCH_ENV) uv run python my_experiments/train_eval.py 0.01/0.2 0.01/0.2 0.002 \
		--only mlp --jobs $(JOBS) $(TEE)

quality:
	@$(LOG_START); \
	$(TORCH_ENV) uv run python my_experiments/train_eval.py 0.80/1.0 0.19/1.0 0.01 \
		--only ridge mlp --jobs $(JOBS) $(TEE)

prod:
	@$(LOG_START); \
	$(TORCH_ENV) uv run python my_experiments/train_eval.py 0.80/1.0 0.19/1.0 0.01 \
		--jobs $(JOBS) $(TEE)

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
# MODEL= picks which member of the zoo is submitted; without it the artifact's own ensemble goes.
# `MODEL="salts:a+b"` averages the decoded maps of several artifacts, which is how an ensemble
# across feature sets is sent — the members are baseline_<name>.joblib beside FUSION_ARTIFACT.
predict_and_submit_to_hf:
	@test -z "$$HF_TOKEN" || { \
		echo "HF_TOKEN is in the environment. huggingface_hub prefers it over the token from"; \
		echo "'hf auth login', so a read-only one shadows your write login and the upload fails"; \
		echo "with 403. Unset it for this command:  env -u HF_TOKEN make predict_and_submit_to_hf"; \
		exit 1; }
	@test -n "$$HF_READ_TOKEN" || { \
		echo "HF_READ_TOKEN is empty. Copy .env.example to .env and fill it in."; exit 1; }
	uv run python submission_skeleton.py --max-shots 0 --source local \
		--configs diii_d_public_test --repo $(HF_REPO) $(if $(MODEL),--model "$(MODEL)")

clean:
	rm -rf .ruff_cache .mypy_cache my_experiments/__pycache__ __pycache__

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

