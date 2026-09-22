#!/bin/bash
# Width x components, the user's hypothesis of 2026-09-02.
#
# Neither factor paid alone. n_pca 70 measured +0.000003 over 7 pairs (t +0.05) and 100 measured
# -0.000095 over 5; the wide MLP measured +0.000249 but on ONE pair before the lane was diverted,
# and `h512`, the same capacity question asked of `seq`, closed at +0.00004. So this is an
# INTERACTION: a [512,512] trunk producing 102 coordinates instead of 52 may simply lack the
# capacity to use them, which would explain a ceiling that rose while the score did not.
#
# ONLY THE MLP IS FITTED. At 70 and 100 components the sequence model already exists on disk
# (z_npca_p70_s*, z_npca_p100_s*), and at 50 a standard-width seq would be the base run's seq
# exactly. Nothing is retrained that already exists.
#
# Interaction hypotheses are where noise gets over-read, and this one is being asked of an
# instrument that resolves 0.000088 at twelve pairs. These are baked as MEMBERS first, where the
# bar is "errs elsewhere" rather than "beats the control" — a wide net at 100 components differs
# from a narrow one at 50 along two axes at once, whatever its own score turns out to be.
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
for p in "$@"; do while kill -0 "$p" 2>/dev/null; do sleep 120; done; done
WIDE="models.mlp.hidden_sizes=[1024,1024] models.mlp.dropout=0.02"
for SALT in ${SALTS:-1 2 3}; do
  OUT=results/c_bake_s${SALT}.csv
  [ -f "$OUT" ] || echo "name,S,R2_psi,R2_qb,D_LCFS,Cons,wall_s" > "$OUT"
  for spec in "wide:50" "widep70:70" "widep100:100"; do
    tag=${spec%%:*}; npca=${spec##*:}
    row="c_${tag}_s${SALT}"
    grep -q "^${row}," "$OUT" && { echo "have $row"; continue; }
    LOG="logs/$(date -u +%Y%m%d-%H%M%S)-${row}.log"
    echo "=== $(date -u +%H:%M) $row  [mlp 1024x1024 dropout 0.02, n_pca ${npca}]"
    T0=$SECONDS
    FUSION_ARTIFACT="artefacts/lanes/baseline_${row}.joblib" \
    OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
      uv run python my_experiments/train_eval.py 0.80/1.0 0.15/1.0 0.05 \
      --jobs 20 --salt "$SALT" --only mlp --score-models mlp \
      --set $WIDE "features.n_pca=${npca}" > "$LOG" 2>&1 \
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
    echo "${row},${R},$((SECONDS-T0))" >> "$OUT"
    echo "  -> ${row},${R}"
  done
done
echo "WIDE BAKE COMPLETE"
