#!/bin/bash
# Refit catboost at the doubled ceiling, on all three salts.
#
# Why refit rather than leave it: at 8000 the fit reported 'best: 7999 of 8000' on every salt and on
# production before that, with early stopping (patience 50) never firing. A best iteration equal to
# the last one allowed is a TRUNCATED fit, not a converged one, and the score it reports is a lower
# bound. Keeping the old members would put a knowingly-truncated model in the collection.
#
# Why all three and not just the future: a collection whose composition differs by salt cannot be
# compared across salts, and comparing across salts is the entire validation design.
#
# The old rows and artifacts are removed first, because every runner here resumes by CSV — leaving
# them would make this a no-op that looks like it ran.
cd /workspace/fusion-equilibrium-challenge-starter
for p in "$@"; do while kill -0 "$p" 2>/dev/null; do sleep 120; done; done
for s in 1 2 3; do
  f=results/c_bake_s${s}.csv
  [ -f "$f" ] && { grep -v "^c_cat_s${s}," "$f" > "$f.tmp" && mv "$f.tmp" "$f"; }
  rm -f "artefacts/lanes/baseline_c_cat_s${s}.joblib"
done
echo "=== $(date -u +%F\ %H:%M) old catboost members removed; refitting at 16000 iterations ==="
for s in 1 2 3; do SALT=$s tools/bake.sh; done
echo "REFIT COMPLETE"
