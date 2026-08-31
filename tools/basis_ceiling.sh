#!/bin/bash
# THE CEILING OF A SHAPED BASIS, measured before any model is fitted to it.
#
# `evaluate.py --mode basis` pushes GROUND TRUTH through the artifact's 50 components and straight
# back, handing q95 and betaN over exactly. So R2_qb is 1.0 by construction and the only three
# numbers that can move are the three the shaped basis trades between: R2_psi down, D_LCFS and
# Consistency up, or the hypothesis is wrong. No regression is involved, which is what makes this
# a RANKING instrument: it costs one shot read and one scoring pass per candidate instead of a
# six-salt lane, and it answers whether there is anything there to train towards at all.
#
# `ridge` is fitted only because `train.py` fits something; it is never called here.
#
# Usage: tools/basis_ceiling.sh tag:envelope:sobolev[:n_pca] [...]   (n_pca defaults to 50)
#
# n_pca is here because it is the OTHER lever on the same ceiling, and unlike the shaped basis
# it can only ever raise it — more components reconstruct strictly better. So the ceiling
# cannot refute it and is not being asked to: it prices the prize, and
# my_experiments/diagnose_components.py asks the question that can say no, which is whether the
# regression reaches the coordinates at all.
cd /workspace/fusion-equilibrium-challenge-starter
mkdir -p artefacts/basis logs results
OUT="results/${OUTFILE:-x_basis_ceiling.csv}"
[ -f "$OUT" ] || echo "name,envelope,sobolev,n_pca,S,R2_psi,R2_qb,D_LCFS,Cons,wall_s" > "$OUT"
for cand in "$@"; do
  IFS=: read -r tag env sob npca <<< "$cand"
  npca="${npca:-50}"
  grep -q "^${tag}," "$OUT" && { echo "have $tag"; continue; }
  ART="artefacts/basis/baseline_${tag}.joblib"
  LOG="logs/$(date -u +%Y%m%d-%H%M%S)-basis-${tag}.log"
  echo "=== $(date -u +%H:%M) $tag  envelope x${env}  sobolev ${sob}  n_pca ${npca} ==="
  T0=$SECONDS
  FUSION_ARTIFACT="$ART" OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 \
    uv run python my_experiments/train.py --share "${SHARE:-0.80/1.0}" \
    --val-share "${VALSHARE:-0.15/1.0}" --jobs "${JOBS:-20}" --only ridge \
    --set "features.basis_envelope=${env}" "features.basis_sobolev=${sob}" \
          "features.n_pca=${npca}" > "$LOG" 2>&1 \
    || { echo "  FIT FAILED, see $LOG"; continue; }
  FUSION_ARTIFACT="$ART" uv run python my_experiments/evaluate.py --mode basis \
    --share "${SCORE:-0.05}" --jobs "${JOBS:-20}" >> "$LOG" 2>&1 \
    || { echo "  SCORING FAILED, see $LOG"; continue; }
  # The single-model block, since --mode basis prints no comparison table. Same parser as
  # tools/lane.sh's fallback, and COMPOSITE S is field 4 — a transcription bug once made it 5.
  R=$(awk '
    /COMPOSITE S = /        {s=$4}
    /^R2_psi /              {p=$2}
    /^R2_\{q95,betaN\} /    {q=$2}
    /^1 - D_LCFS /          {d=$4}
    /^Consistency /         {c=$2}
    END {if (s!="" && p!="" && q!="" && d!="" && c!="") print s","p","q","d","c}
  ' <(sed -E 's/^\[[^]]*\] *//' "$LOG"))
  [ -z "$R" ] && { echo "  NO ceiling row in $LOG"; continue; }
  echo "${tag},${env},${sob},${npca},${R},$((SECONDS-T0))" >> "$OUT"
  echo "  -> ${tag},${R}"
done
echo "BASIS CEILING SWEEP COMPLETE"
