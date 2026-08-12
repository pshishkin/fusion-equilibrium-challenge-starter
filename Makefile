# make ci — everything that must pass before a change is done.
#
# Scope is this fork's own code: my_experiments/ plus the root scripts we edited. The organizers'
# files and the vendored scorer stay byte-identical to upstream, so they are excluded in
# pyproject.toml rather than fixed. See AGENTS.md.

.PHONY: ci lint format typecheck test quality train eval clean

ci: lint typecheck test

lint:
	uv run ruff check .

# Not part of `ci`: reformatting is a separate decision from a clean lint.
format:
	uv run ruff format .

typecheck:
	uv run mypy

# The two standard runs. AGENTS.md, "How we test the metric".
#
#   test     does it work, and how fast — 70 shots thinned to a fifth of their frames, 14 scored
#   quality  what does it score — 352 shots thinned the same way, 70 scored
test:
	uv run python my_experiments/train_eval.py 0.01/0.2 0.01/0.2 0.002

quality:
	uv run python my_experiments/train_eval.py 0.05/0.2 0.05/0.2 0.01

train:
	uv run python my_experiments/train.py --share 0.05/0.2 --val-share 0.05/0.2

eval:
	uv run python my_experiments/evaluate.py --share 0.01

clean:
	rm -rf .ruff_cache .mypy_cache my_experiments/__pycache__ __pycache__
