# make ci — everything that must pass before a change is done.
#
# Scope is this fork's own code: my_experiments/ plus the root scripts we edited. The organizers'
# files and the vendored scorer stay byte-identical to upstream, so they are excluded in
# pyproject.toml rather than fixed. See AGENTS.md.

# Local secrets, untracked. The leading `-` keeps every other target working without the file.
# Exported rather than passed on the command line, so the token stays out of `make`'s echo and out
# of the process list: submission_skeleton.py defaults --read-token to $HF_READ_TOKEN.
-include .env
export HF_READ_TOKEN
HF_REPO ?= pshishkin/fusion-eq-predictions

.PHONY: ci lint format typecheck test quality prod train eval predict_and_submit_to_hf clean \
        clean-cache warm-cache download_dataset

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
#   quality  what does it score — 352 shots thinned the same way, 70 scored
#   prod     what a submission is built from — 4225 shots to fit, 1056 to stop on, 70 scored.
#            Four MLP seeds averaged, so ~20 min with the shot cache warm.
test:
	uv run python my_experiments/train_eval.py 0.01/0.2 0.01/0.2 0.002

quality:
	uv run python my_experiments/train_eval.py 0.05/0.2 0.05/0.2 0.01

prod:
	uv run python my_experiments/train_eval.py 0.60/0.1 0.15/0.1 0.01

train:
	uv run python my_experiments/train.py --share 0.05/0.2 --val-share 0.05/0.2

eval:
	uv run python my_experiments/evaluate.py --share 0.01

# Predict every public-test shot with the trained artifact, push the result to Hugging Face and
# write the pointer zip to upload to Codabench. Structure validation runs in between; scoring does
# not — run `make eval` (or a full `make prod`) yourself if you want to know what you are sending.
#
# The token comes from .env, which is NOT in git — this file is, and the fork is public.
predict_and_submit_to_hf:
	@test -n "$$HF_READ_TOKEN" || { \
		echo "HF_READ_TOKEN is empty. Copy .env.example to .env and fill it in."; exit 1; }
	uv run python submission_skeleton.py --max-shots 0 --source local \
		--configs diii_d_public_test --repo $(HF_REPO)

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
