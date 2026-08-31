#!/bin/bash
# One lane, one ARTIFACT PATH OF ITS OWN.
#
# THE BUG THIS FIXES, found 2026-08-22. Every lane wrote its fitted artifact to the default
# `my_experiments/baseline.joblib`. Two lanes running at once overwrite each other BETWEEN the fit
# and the scoring, so a run scores whatever artifact happened to be on disk. Three runs died of it
# with `KeyError: no model 'seq' ... it holds ['ridge','mlp',...]` and vice versa.
#
# The crashes were the LUCKY case. Two lanes whose model sets share a name would not crash — they
# would silently score the wrong model and report the number as their own. It was checked, run by
# run, that no recorded result suffered that: every log's trained set matches its scored set, which
# holds because the concurrent lanes happened to fit different model names. That is luck, not
# design, and this removes the dependence on it.
#
# Usage: NAME=<tag> ARM_SET="<--set args>" MODELS="<--only args>" SALTS="..." lane.sh [wait-pids...]
cd /workspace/fusion-equilibrium-challenge-starter
for p in "$@"; do while kill -0 "$p" 2>/dev/null; do sleep 300; done; done
OUT="results/${OUTFILE}"
[ -f "$OUT" ] || echo "name,S,R2_psi,R2_qb,D_LCFS,Cons,wall_s" > "$OUT"
for salt in $SALTS; do
  for arm in $ARMS; do
    row="${NAME}_${arm}_s${salt}"
    grep -q "^${row}," "$OUT" && { echo "have $row"; continue; }
    eval "SET=(\$ARM_SET_${arm})"
    LOG="logs/$(date -u +%Y%m%d-%H%M%S)-${row}.log"
    echo "=== $(date -u +%H:%M) $row ==="
    T0=$SECONDS
    # ONE ARTIFACT PER SALT AND ARM, not one per lane. Until 2026-08-25 this was
    # artefacts/lanes/${NAME}.joblib, so every salt overwrote the previous and five sixths of every
    # lane's fitted models were thrown away. They are free — the lane trains them anyway — and they
    # are the one diversity axis never measured: seeds correlate at 0.56 and feature sets at 0.75,
    # but WHICH 80% of shots a member trains on has never been varied. Naming them
    # baseline_<row>.joblib is what `salts:a+b+c` expects, so a blend can be assembled later with no
    # extra training at all.
    FUSION_ARTIFACT="artefacts/lanes/baseline_${row}.joblib" \
    OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
      uv run python my_experiments/train_eval.py 0.80/1.0 0.15/1.0 0.05 \
      --jobs ${JOBS:-20} --salt "$salt" --only $MODELS \
      --score-models ${SCORE:-ensemble} "${SET[@]}" > "$LOG" 2>&1
    # TWO formats, because scoring ONE model prints no comparison table. The table row is the
    # cheaper parse and is tried first; the single-model block is the fallback. Both are read
    # rather than one, because `--score-models ensemble` is now the default for speed and it is
    # exactly what stops the table from printing — the regression that silently discarded a
    # completed 45-minute run on 2026-08-31, and the same shape as the lambda sweep's in August.
    R=$(grep -E "^\[[^]]*\]\s+ensemble\s" "$LOG" | tail -1 \
        | sed -E 's/^\[[^]]*\]\s+ensemble\s+//' | awk '{print $1","$2","$3","$4","$5}')
    if [ -z "$R" ]; then
      R=$(awk '
        /COMPOSITE S = /        {s=$4}
        /^R2_psi /              {p=$2}
        /^R2_\{q95,betaN\} /    {q=$2}
        /^1 - D_LCFS /          {d=$4}
        /^Consistency /         {c=$2}
        END {if (s!="" && p!="" && q!="" && d!="" && c!="") print s","p","q","d","c}
      ' <(sed -E 's/^\[[^]]*\] *//' "$LOG"))
    fi
    [ -z "$R" ] && { echo "  NO ensemble row in $LOG"; continue; }
    echo "${row},${R},$((SECONDS-T0))" >> "$OUT"
    echo "  -> ${row},${R}"
  done
done
echo "LANE ${NAME} COMPLETE"
