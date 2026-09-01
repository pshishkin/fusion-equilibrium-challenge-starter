#!/bin/bash
# DOES A BIG BLEND HELP? Answered with no training at all.
#
# The design the user described on 2026-09-01: many diverse models blended WITHIN one salt, and the
# result validated by repeating the whole blend at another salt — ensemble at salt 0 on salt 0's
# tail against ensemble at salt 1 on salt 1's. Every member of a within-salt blend shares the split,
# so its 352-shot tail is untouched by all of them and no holdout is involved.
#
# Nineteen artifacts already exist at salt 0, each carrying four MLPs and most of them a sequence
# model, so nineteen of them IS the hundred-model ensemble — assembled at scoring time, for free.
#
# The order is ALPHABETICAL and fixed. Ordering by score would make the curve a selection: the first
# members would be the best ones and the shape would say more about the ranking than about blending.
# Alphabetical is arbitrary, which is the point — it includes the deliberately degraded ablation
# arms (n_pca=30, head_weight=0, wider nets) at whatever position they fall, which is exactly the
# question "does throwing everything in help".
cd /workspace/fusion-equilibrium-challenge-starter
ALL=$(ls artefacts/lanes/*_s0.joblib | sed 's|.*/baseline_||; s|\.joblib$||' | grep -v "^y_seq_" | sort)
OUT=results/x_blend_curve.csv
[ -f "$OUT" ] || echo "n_members,S,R2_psi,R2_qb,D_LCFS,Cons,members" > "$OUT"
for n in ${SIZES:-1 2 4 8 13 19}; do
  grep -q "^${n}," "$OUT" && { echo "have n=$n"; continue; }
  MEMBERS=$(echo "$ALL" | head -"$n" | tr '\n' '+' | sed 's/+$//')
  LOG="logs/$(date -u +%Y%m%d-%H%M%S)-blend${n}.log"
  echo "=== $(date -u +%H:%M) blend of $n ==="
  FUSION_ARTIFACT=artefacts/lanes/baseline_x_base_s0.joblib \
  OMP_NUM_THREADS=4 uv run python my_experiments/evaluate.py \
    --share 0.05 --jobs "${JOBS:-6}" --models "salts:${MEMBERS}" > "$LOG" 2>&1 \
    || { echo "  FAILED, see $LOG"; continue; }
  R=$(awk '
    /COMPOSITE S = /        {s=$4}
    /^R2_psi /              {p=$2}
    /^R2_\{q95,betaN\} /    {q=$2}
    /^1 - D_LCFS /          {d=$4}
    /^Consistency /         {c=$2}
    END {if (s!="") print s","p","q","d","c}
  ' <(sed -E 's/^\[[^]]*\] *//' "$LOG"))
  [ -z "$R" ] && { echo "  NO row in $LOG"; continue; }
  echo "${n},${R},${MEMBERS}" >> "$OUT"
  echo "  -> ${n},${R}"
done
echo "BLEND CURVE COMPLETE"
